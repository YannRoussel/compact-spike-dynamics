"""Staged passive, fast-channel, slow-K, and trust-region optimization."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cell_optimization import (
    CELL_PARAMETER_NAMES,
    OBJECTIVE_NAMES,
    CandidateEvaluation,
    CellOptimizationConfig,
    load_cell_parameter_bounds,
    optimize_cell,
    representative_feature_comparison,
)
from .cell_targets import (
    CellOptimizationTarget,
    target_metadata,
)
from .features import FeatureConfig, detect_spikes, voltage_feature_trace
from .hh_model import Stimulus
from .model_ladder import (
    VARIANT_PARAMETER_NAMES,
    LadderFitResult,
    LadderOptimizationConfig,
    _bounded_parameters,
    _objective_config,
    _unbounded_parameters,
    _validation_table,
    baseline_ladder_result,
    evaluate_ladder_candidate,
    ladder_scalar_score,
    optimize_ladder_variant,
)
from .protocols import BiologicalScreenConfig
from .raw_patchseq import read_current_clamp_sweeps


@dataclass(frozen=True)
class StagedLadderConfig:
    fast_population_size: int = 12
    fast_generations: int = 3
    fast_elites: int = 2
    slow_population_size: int = 6
    slow_generations: int = 2
    final_population_size: int = 8
    final_generations: int = 2
    workers: int = 1
    seed: int = 42
    fast_diversity_distance: float = 0.08
    trust_fraction: float = 0.20
    voltage_shift_margin_mv: float = 4.0
    trust_regularization: float = 0.10
    sigma: float = 0.45
    gna_gk_factor: float = 1.5


@dataclass(frozen=True)
class StagedLadderResult:
    target: CellOptimizationTarget
    fast_history: pd.DataFrame
    fast_elites: pd.DataFrame
    stage1: LadderFitResult
    stage2: LadderFitResult
    stage3: LadderFitResult
    joint_history: pd.DataFrame
    metadata: Mapping[str, object]


def _fast_scalar(frame: pd.DataFrame) -> pd.Series:
    return (
        2.0 * frame["passive"]
        + 0.5 * frame["excitability"]
        + frame["spike_shape"]
        + frame["spike_dynamics"]
        + frame["phase_geometry"]
        + 0.1 * frame["conductance_prior"]
    )


def _diverse_fast_elites(
    history: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    minimum_distance: float,
) -> pd.DataFrame:
    valid = history.loc[history["valid"]].copy()
    if valid.empty:
        raise RuntimeError("Fast-channel optimization produced no valid model")
    valid["fast_scalar_score"] = _fast_scalar(valid)
    valid = valid.sort_values("fast_scalar_score")
    selected = []
    span = np.maximum(upper - lower, 1e-12)
    for index, row in valid.iterrows():
        vector = row.loc[list(CELL_PARAMETER_NAMES)].to_numpy(dtype=float)
        normalized = (vector - lower) / span
        if all(
            np.linalg.norm(normalized - previous) / math.sqrt(len(vector))
            >= minimum_distance
            for previous in selected
        ):
            selected.append(normalized)
            if len(selected) >= count:
                break
    if len(selected) < count:
        used = {
            tuple(np.round(vector, 10))
            for vector in selected
        }
        for _, row in valid.iterrows():
            vector = (
                row.loc[list(CELL_PARAMETER_NAMES)].to_numpy(dtype=float)
                - lower
            ) / span
            key = tuple(np.round(vector, 10))
            if key not in used:
                selected.append(vector)
                used.add(key)
            if len(selected) >= count:
                break
    indices = []
    for normalized in selected:
        distances = np.linalg.norm(
            (
                valid.loc[:, list(CELL_PARAMETER_NAMES)].to_numpy(dtype=float)
                - lower
            )
            / span
            - normalized,
            axis=1,
        )
        indices.append(valid.index[int(np.argmin(distances))])
    return valid.loc[indices].reset_index(drop=True)


def _trust_bounds(
    names: Sequence[str],
    center: np.ndarray,
    absolute_lower: np.ndarray,
    absolute_upper: np.ndarray,
    config: StagedLadderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    multiplicative_width = math.log1p(config.trust_fraction)
    widths = np.asarray(
        [
            (
                config.voltage_shift_margin_mv
                if "voltage_shift_mv" in name
                else multiplicative_width
            )
            for name in names
        ],
        dtype=float,
    )
    lower = np.maximum(absolute_lower, center - widths)
    upper = np.minimum(absolute_upper, center + widths)
    if np.any(lower >= upper):
        raise ValueError("Trust-region bounds collapsed")
    return lower, upper


def _joint_payload(
    unbounded: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    ladder_config: LadderOptimizationConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    regularization: float,
) -> tuple[float, tuple[float, ...], bool, str, np.ndarray]:
    parameters = _bounded_parameters(unbounded, lower, upper)
    base_count = len(CELL_PARAMETER_NAMES)
    evaluation = evaluate_ladder_candidate(
        parameters[base_count:],
        "k-slow",
        parameters[:base_count],
        target,
        screen_config,
        ladder_config,
    )
    if not evaluation.valid:
        return (
            ladder_config.invalid_score,
            evaluation.objectives,
            False,
            evaluation.reason,
            parameters,
        )
    half_width = np.maximum((upper - lower) / 2.0, 1e-12)
    prior = float(np.mean(((parameters - center) / half_width) ** 2))
    score = ladder_scalar_score(evaluation.objectives) + regularization * prior
    return score, evaluation.objectives, True, evaluation.reason, parameters


def _joint_refinement(
    base_center: np.ndarray,
    extension_center: np.ndarray,
    target: CellOptimizationTarget,
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    config: StagedLadderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, CandidateEvaluation]:
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Joint refinement requires DEAP") from error

    base_lower, base_upper = load_cell_parameter_bounds(
        bounds_path,
        config.gna_gk_factor,
    )
    extension_names = VARIANT_PARAMETER_NAMES["k-slow"]
    from .model_ladder import extension_parameter_bounds

    extension_lower, extension_upper, _ = extension_parameter_bounds("k-slow")
    center = np.concatenate((base_center, extension_center))
    absolute_lower = np.concatenate((base_lower, extension_lower))
    absolute_upper = np.concatenate((base_upper, extension_upper))
    names = CELL_PARAMETER_NAMES + extension_names
    lower, upper = _trust_bounds(
        names,
        center,
        absolute_lower,
        absolute_upper,
        config,
    )
    centroid = _unbounded_parameters(center, lower, upper)
    ladder_config = LadderOptimizationConfig(
        population_size=config.final_population_size,
        generations=config.final_generations,
        workers=config.workers,
        seed=config.seed + 1000,
        sigma=config.sigma,
        gna_gk_factor=config.gna_gk_factor,
    )

    fitness_name = "FitnessStagedJointCMA"
    individual_name = "IndividualStagedJointCMA"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)
    np.random.seed(config.seed + 1000)
    strategy = cma.Strategy(
        centroid=centroid,
        sigma=config.sigma,
        lambda_=config.final_population_size,
    )

    rows = []
    initial_evaluation = evaluate_ladder_candidate(
        extension_center,
        "k-slow",
        base_center,
        target,
        screen_config,
        ladder_config,
    )
    rows.append(
        {
            "generation": -1,
            "individual": -1,
            **dict(zip(names, center)),
            **dict(zip(OBJECTIVE_NAMES, initial_evaluation.objectives)),
            "scalar_score": (
                ladder_scalar_score(initial_evaluation.objectives)
                if initial_evaluation.valid
                else ladder_config.invalid_score
            ),
            "valid": initial_evaluation.valid,
            "reason": initial_evaluation.reason,
        }
    )

    executor = (
        ProcessPoolExecutor(max_workers=config.workers)
        if config.workers > 1
        else None
    )
    mapper = executor.map if executor is not None else map
    try:
        for generation in range(config.final_generations):
            population = strategy.generate(individual_type)
            payloads = list(
                mapper(
                    _joint_payload,
                    population,
                    [target] * len(population),
                    [screen_config] * len(population),
                    [ladder_config] * len(population),
                    [lower] * len(population),
                    [upper] * len(population),
                    [center] * len(population),
                    [config.trust_regularization] * len(population),
                )
            )
            for index, (individual, payload) in enumerate(
                zip(population, payloads)
            ):
                score, objectives, valid, reason, parameters = payload
                individual.fitness.values = (score,)
                rows.append(
                    {
                        "generation": generation,
                        "individual": index,
                        **dict(zip(names, parameters)),
                        **dict(zip(OBJECTIVE_NAMES, objectives)),
                        "scalar_score": score,
                        "valid": valid,
                        "reason": reason,
                    }
                )
            strategy.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    history = pd.DataFrame(rows)
    valid = history.loc[history["valid"]].copy()
    if valid.empty:
        raise RuntimeError("Joint trust-region refinement produced no valid model")
    best = valid.nsmallest(1, "scalar_score").copy()
    parameters = best.loc[best.index[0], list(names)].to_numpy(dtype=float)
    evaluation = evaluate_ladder_candidate(
        parameters[len(CELL_PARAMETER_NAMES) :],
        "k-slow",
        parameters[: len(CELL_PARAMETER_NAMES)],
        target,
        screen_config,
        ladder_config,
        include_validation=True,
        include_traces=True,
    )
    if not evaluation.valid:
        raise RuntimeError(
            f"Best joint model failed reevaluation: {evaluation.reason}"
        )
    return history, best, evaluation


def _fit_result_from_joint(
    best: pd.DataFrame,
    history: pd.DataFrame,
    evaluation: CandidateEvaluation,
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: StagedLadderConfig,
) -> LadderFitResult:
    ladder_config = LadderOptimizationConfig(
        population_size=config.final_population_size,
        generations=config.final_generations,
        workers=config.workers,
        seed=config.seed + 1000,
        gna_gk_factor=config.gna_gk_factor,
    )
    return LadderFitResult(
        variant="stage3-joint",
        target=target,
        parameter_names=(
            CELL_PARAMETER_NAMES + VARIANT_PARAMETER_NAMES["k-slow"]
        ),
        history=history,
        best_parameters=best,
        evaluation=evaluation,
        feature_comparison=representative_feature_comparison(
            target,
            evaluation,
            _objective_config(ladder_config),
        ),
        validation=_validation_table(evaluation, target, ladder_config),
        metadata={
            "version": 1,
            "algorithm": "bounded trust-region CMA-ES",
            "variant": "stage3-joint",
            "total_model_parameter_count": (
                len(CELL_PARAMETER_NAMES)
                + len(VARIANT_PARAMETER_NAMES["k-slow"])
            ),
            "optimization_config": asdict(config),
            "screen_config": asdict(screen_config),
            "target": target_metadata(target),
            "best_biophysics": evaluation.biophysics,
            "best_rheobase_lower_pa": evaluation.rheobase_lower_pa,
            "best_rheobase_upper_pa": evaluation.rheobase_upper_pa,
        },
    )


def optimize_staged_ladder(
    target: CellOptimizationTarget,
    bounds_path: str | Path,
    seed_population: pd.DataFrame,
    screen_config: BiologicalScreenConfig,
    config: StagedLadderConfig | None = None,
    checkpoint_dir: str | Path | None = None,
) -> StagedLadderResult:
    """Run the fast-elite, slow-K branch, and joint-refinement ladder."""
    config = config or StagedLadderConfig()
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    fast_config = CellOptimizationConfig(
        population_size=config.fast_population_size,
        generations=config.fast_generations,
        workers=config.workers,
        seed=config.seed,
        gna_gk_factor=config.gna_gk_factor,
        passive_reanchoring=True,
        evaluation_profile="fast",
    )
    fast_result = optimize_cell(
        target,
        bounds_path,
        screen_config,
        fast_config,
        seed_population=seed_population,
        checkpoint_dir=(checkpoint / "step1_fast" if checkpoint else None),
    )
    lower, upper = load_cell_parameter_bounds(
        bounds_path,
        config.gna_gk_factor,
    )
    elites = _diverse_fast_elites(
        fast_result.history,
        lower,
        upper,
        config.fast_elites,
        config.fast_diversity_distance,
    )

    ladder_config = LadderOptimizationConfig(
        population_size=config.slow_population_size,
        generations=config.slow_generations,
        workers=config.workers,
        seed=config.seed + 100,
        sigma=config.sigma,
        gna_gk_factor=config.gna_gk_factor,
    )
    branches = []
    for elite_index, elite in elites.iterrows():
        base_vector = elite.loc[
            list(CELL_PARAMETER_NAMES)
        ].to_numpy(dtype=float)
        branch = optimize_ladder_variant(
            "k-slow",
            base_vector,
            target,
            screen_config,
            ladder_config,
            checkpoint_dir=(
                checkpoint / f"step2_kslow_elite_{elite_index}"
                if checkpoint
                else None
            ),
        )
        branches.append((base_vector, branch))
    base_vector, stage2 = min(
        branches,
        key=lambda pair: ladder_scalar_score(
            pair[1].evaluation.objectives
        ),
    )
    stage1 = baseline_ladder_result(
        base_vector,
        target,
        screen_config,
        ladder_config,
    )
    extension_center = stage2.best_parameters.loc[
        stage2.best_parameters.index[0],
        list(VARIANT_PARAMETER_NAMES["k-slow"]),
    ].to_numpy(dtype=float)
    joint_history, joint_best, joint_evaluation = _joint_refinement(
        base_vector,
        extension_center,
        target,
        bounds_path,
        screen_config,
        config,
    )
    stage3 = _fit_result_from_joint(
        joint_best,
        joint_history,
        joint_evaluation,
        target,
        screen_config,
        config,
    )
    metadata = {
        "version": 1,
        "algorithm": "NSGA-II fast elites + slow-K CMA-ES + joint CMA-ES",
        "config": asdict(config),
        "bounds_path": str(bounds_path),
        "screen_config": asdict(screen_config),
        "target": target_metadata(target),
        "n_fast_elites": len(elites),
        "winning_fast_elite_score": float(
            _fast_scalar(
                elites.loc[
                    (
                        elites.loc[:, list(CELL_PARAMETER_NAMES)]
                        == base_vector
                    ).all(axis=1)
                ]
            ).iloc[0]
        ),
        "slow_k_improved_over_fast_parent": (
            ladder_scalar_score(stage2.evaluation.objectives)
            < ladder_scalar_score(stage1.evaluation.objectives)
        ),
    }
    return StagedLadderResult(
        target=target,
        fast_history=fast_result.history,
        fast_elites=elites,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        joint_history=joint_history,
        metadata=metadata,
    )


def staged_summary(result: StagedLadderResult) -> pd.DataFrame:
    rows = []
    step1_score = ladder_scalar_score(result.stage1.evaluation.objectives)
    for label, stage in (
        ("step1_fast", result.stage1),
        ("step2_kslow", result.stage2),
        ("step3_joint", result.stage3),
    ):
        scalar_score = ladder_scalar_score(stage.evaluation.objectives)
        held_out = {
            f"held_out__{row.objective_group}": float(row.score)
            for row in stage.validation.itertuples()
        }
        rows.append(
            {
                "stage": label,
                "dataset": result.target.dataset,
                "cell_id": result.target.cell_id,
                **dict(zip(OBJECTIVE_NAMES, stage.evaluation.objectives)),
                "scalar_score": scalar_score,
                "delta_vs_step1": scalar_score - step1_score,
                "relative_improvement_vs_step1": (
                    (step1_score - scalar_score) / step1_score
                ),
                "improved_vs_step1": scalar_score < step1_score,
                "model_rheobase_pa": stage.evaluation.rheobase_upper_pa,
                **held_out,
            }
        )
    return pd.DataFrame(rows)


def _biological_sweep_map(target: CellOptimizationTarget):
    sweeps = read_current_clamp_sweeps(
        target.nwb_path,
        long_square_only=True,
    )
    return {sweep.sweep_number: sweep for sweep in sweeps}


def save_staged_trace_plot(
    result: StagedLadderResult,
    screen_config: BiologicalScreenConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_map = _biological_sweep_map(result.target)
    rows = [
        (
            "passive",
            result.target.passive.passive_sweep_number,
            result.target.passive.passive_current_pa,
            result.target.passive.passive_duration_ms,
        ),
        *[
            (
                protocol.name,
                protocol.sweep_number,
                protocol.current_pa,
                protocol.duration_ms,
            )
            for protocol in (
                *result.target.training_protocols,
                *result.target.validation_protocols,
            )
        ],
    ]
    stages = (
        ("Step 1 fast", result.stage1, "#e76f51"),
        ("Step 2 + slow K", result.stage2, "#457b9d"),
        ("Step 3 joint", result.stage3, "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        len(rows),
        1,
        figsize=(11.0, 9.5),
        constrained_layout=True,
    )
    for axis, (name, sweep_number, current_pa, duration_ms) in zip(
        axes,
        rows,
    ):
        sweep = sweep_map[sweep_number]
        biological_mask = (
            (sweep.time_ms >= sweep.stimulus_start_ms - 20.0)
            & (sweep.time_ms <= sweep.stimulus_end_ms)
        )
        axis.plot(
            sweep.time_ms[biological_mask] - sweep.stimulus_start_ms,
            sweep.voltage_mv[biological_mask],
            color="#147d7e",
            linewidth=1.15,
            label="Biological",
            zorder=4,
        )
        for label, stage, color in stages:
            trace = stage.evaluation.traces[name]
            model_time = trace.time_ms - screen_config.baseline_ms
            model_mask = (model_time >= -20.0) & (
                model_time <= duration_ms
            )
            axis.plot(
                model_time[model_mask],
                trace.voltage_mv[model_mask],
                color=color,
                linewidth=0.8,
                alpha=0.85,
                label=label,
            )
        axis.set_title(
            f"{name.replace('_', ' ').title()}: biological {current_pa:g} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"{result.target.dataset} / {result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _first_phase_cycle(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    stimulus: Stimulus,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    trace = voltage_feature_trace(time_ms, voltage_mv)
    detection = detect_spikes(trace, stimulus, FeatureConfig(min_spikes=1))
    if not len(detection.peak_indices):
        return None
    threshold = int(detection.threshold_indices[0])
    peak = int(detection.peak_indices[0])
    if len(detection.threshold_indices) > 1:
        stop = int(detection.threshold_indices[1])
    else:
        dt = float(np.median(np.diff(trace.time_ms)))
        stop = min(len(trace.time_ms) - 1, peak + int(round(12.0 / dt)))
    return (
        trace.voltage_mv[threshold : stop + 1],
        trace.dvdt_mv_ms[threshold : stop + 1],
        float(trace.voltage_mv[threshold]),
    )


def save_staged_phase_plot(
    result: StagedLadderResult,
    screen_config: BiologicalScreenConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_map = _biological_sweep_map(result.target)
    protocols = result.target.training_protocols
    stages = (
        ("Step 1 fast", result.stage1, "#e76f51"),
        ("Step 2 + slow K", result.stage2, "#457b9d"),
        ("Step 3 joint", result.stage3, "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        len(protocols),
        2,
        figsize=(10.5, 7.5),
        constrained_layout=True,
    )
    for row_index, protocol in enumerate(protocols):
        sweep = sweep_map[protocol.sweep_number]
        biological_stimulus = Stimulus(
            amplitude_pa=protocol.current_pa,
            amplitude_ua_cm2=None,
            start_ms=float(sweep.stimulus_start_ms),
            end_ms=float(sweep.stimulus_end_ms),
        )
        biological = _first_phase_cycle(
            sweep.time_ms,
            sweep.voltage_mv,
            biological_stimulus,
        )
        cycles = [("Biological", biological, "#147d7e")]
        for label, stage, color in stages:
            trace = stage.evaluation.traces[protocol.name]
            model_stimulus = Stimulus(
                amplitude_pa=(
                    protocol.rheobase_factor
                    * stage.evaluation.rheobase_upper_pa
                ),
                amplitude_ua_cm2=None,
                start_ms=screen_config.baseline_ms,
                end_ms=(
                    screen_config.baseline_ms + protocol.duration_ms
                ),
            )
            cycles.append(
                (
                    label,
                    _first_phase_cycle(
                        trace.time_ms,
                        trace.voltage_mv,
                        model_stimulus,
                    ),
                    color,
                )
            )
        for label, cycle, color in cycles:
            if cycle is None:
                continue
            voltage, dvdt, threshold = cycle
            axes[row_index, 0].plot(
                voltage,
                dvdt,
                color=color,
                linewidth=1.1,
                label=label,
            )
            axes[row_index, 1].plot(
                voltage - threshold,
                dvdt,
                color=color,
                linewidth=1.1,
                label=label,
            )
        axes[row_index, 0].set_title(
            f"{protocol.name.title()}: absolute V",
            loc="left",
            fontsize=10,
        )
        axes[row_index, 1].set_title(
            f"{protocol.name.title()}: threshold aligned",
            loc="left",
            fontsize=10,
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.5)
            axis.set_ylabel("dV/dt (mV/ms)")
            axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("V (mV)")
    axes[-1, 1].set_xlabel("V - threshold (mV)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"First-spike phase planes: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_staged_score_plot(
    summary: pd.DataFrame,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "passive",
        "excitability",
        "spike_shape",
        "spike_dynamics",
        "phase_geometry",
        "firing_pattern",
        "held_out__firing_pattern",
        "scalar_score",
    )
    labels = (
        "Passive",
        "Rheobase",
        "Waveform",
        "Dynamics",
        "Phase",
        "Train",
        "Held-out train",
        "Total",
    )
    stages = summary["stage"].tolist()
    colors = ("#e76f51", "#457b9d", "#7b2cbf")
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    for index, (stage, color) in enumerate(zip(stages, colors)):
        axis.bar(
            x + (index - 1) * width,
            summary.loc[
                summary["stage"].eq(stage),
                list(metrics),
            ].iloc[0],
            width=width,
            label=stage.replace("_", " ").title(),
            color=color,
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Normalized loss (lower is better)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    axis.set_title(
        f"Stage scores: {summary['dataset'].iloc[0]} / "
        f"{summary['cell_id'].iloc[0]}",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_staged_result(
    result: StagedLadderResult,
    output_dir: str | Path,
    screen_config: BiologicalScreenConfig,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.fast_history.to_csv(output / "step1_history.csv", index=False)
    result.fast_elites.to_csv(output / "step1_elites.csv", index=False)
    result.stage2.history.to_csv(output / "step2_history.csv", index=False)
    result.stage2.best_parameters.to_csv(
        output / "step2_best_model.csv",
        index=False,
    )
    result.joint_history.to_csv(output / "step3_history.csv", index=False)
    result.stage3.best_parameters.to_csv(
        output / "step3_best_model.csv",
        index=False,
    )
    summary = staged_summary(result)
    summary.to_csv(output / "stage_summary.csv", index=False)
    result.stage3.feature_comparison.to_csv(
        output / "final_feature_comparison.csv",
        index=False,
    )
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )
    save_staged_trace_plot(
        result,
        screen_config,
        output / "trace_comparison.png",
    )
    save_staged_phase_plot(
        result,
        screen_config,
        output / "phase_plane_comparison.png",
    )
    save_staged_score_plot(
        summary,
        output / "score_bar_plot.png",
    )
