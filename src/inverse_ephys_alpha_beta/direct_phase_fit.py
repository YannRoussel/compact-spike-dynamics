"""Staged direct x_inf/tau fitting of first-spike phase cycles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .cell_optimization import CELL_PARAMETER_NAMES
from .cell_targets import (
    CellOptimizationTarget,
    admittance_anchored_biophysics,
    target_metadata,
)
from .direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
    DirectKinetics,
    direct_kinetic_initial,
    direct_kinetic_parameter_bounds,
)
from .hh_model import SimulationError, Stimulus, Trace
from .kinetics import KineticParameters, PARAMETER_NAMES
from .model_ladder import (
    AIS_PARAMETER_NAMES,
    DecodedLadderModel,
    _bounded_parameters,
    _initial_state,
    _unbounded_parameters,
    extension_parameter_bounds,
    find_rheobase_rush_larsen,
    simulate_ladder_rush_larsen,
)
from .phase_experiment import (
    KineticDiagnostics,
    biological_phase_cycles,
    coupled_resting_state_is_stable,
    kinetic_diagnostics,
)
from .phase_shape import (
    PhaseCycle,
    PhaseShapeConfig,
    PhaseShapeScores,
    compare_phase_cycles,
    extract_phase_cycle,
    mean_phase_scores,
)
from .protocols import BiologicalScreenConfig, step_simulation_config
from .raw_patchseq import read_current_clamp_sweeps


DIRECT_STATIC_PARAMETER_NAMES = (
    "param__direct_static__log_gna_scale",
    "param__direct_static__log_gk_scale",
    "param__direct_static__ena_mv",
    "param__direct_static__ek_mv",
    *AIS_PARAMETER_NAMES,
)
DIRECT_ALL_PARAMETER_NAMES = (
    DIRECT_KINETIC_PARAMETER_NAMES + DIRECT_STATIC_PARAMETER_NAMES
)


@dataclass(frozen=True)
class DirectPhaseFitConfig:
    shape_population_size: int = 10
    shape_generations: int = 3
    shape_elites: int = 2
    scale_population_size: int = 8
    scale_generations: int = 2
    final_population_size: int = 10
    final_generations: int = 2
    workers: int = 1
    seed: int = 42
    sigma: float = 0.18
    gna_gk_factor: float = 1.8
    phase_step_duration_ms: float = 300.0
    rheobase_probe_duration_ms: float = 300.0
    rheobase_probe_tolerance_pa: float = 5.0
    minimum_resolved_tau_ms: float = 0.003
    minimum_tau_dt_fraction: float = 0.08
    kinetic_prior_weight: float = 0.05
    static_prior_weight: float = 0.05
    rheobase_weight: float = 0.05
    rheobase_constraint_weight: float = 10.0
    rheobase_tolerance_fraction: float = 0.15
    rheobase_tolerance_floor_pa: float = 10.0
    shape_scale_guard_weight: float = 0.0
    scale_shape_guard_weight: float = 100.0
    shape_guard_relative_tolerance: float = 0.20
    physical_constraint_weight: float = 20.0
    trust_fraction: float = 0.20
    voltage_trust_margin_mv: float = 4.0
    invalid_score: float = 10_000.0


@dataclass(frozen=True)
class DirectPhaseEvaluation:
    valid: bool
    reason: str
    objective_total: float
    shape_loss: float = float("nan")
    scale_loss: float = float("nan")
    regularity_loss: float = float("nan")
    constraints_satisfied: bool = False
    phase_scores: PhaseShapeScores | None = None
    protocol_scores: Mapping[str, PhaseShapeScores] | None = None
    cycles: Mapping[str, PhaseCycle] | None = None
    traces: Mapping[str, Trace] | None = None
    rheobase_lower_pa: float = float("nan")
    rheobase_upper_pa: float = float("nan")
    rheobase_loss: float = float("nan")
    rheobase_constraint_loss: float = float("nan")
    kinetic_prior: float = float("nan")
    static_prior: float = float("nan")
    diagnostics: KineticDiagnostics | None = None
    biophysics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class DirectPhaseFitResult:
    target: CellOptimizationTarget
    biological_cycles: Mapping[str, PhaseCycle]
    initial_parameters: np.ndarray
    shape_parameters: np.ndarray
    scale_parameters: np.ndarray
    final_parameters: np.ndarray
    initial_evaluation: DirectPhaseEvaluation
    shape_evaluation: DirectPhaseEvaluation
    scale_evaluation: DirectPhaseEvaluation
    final_evaluation: DirectPhaseEvaluation
    shape_history: pd.DataFrame
    scale_history: pd.DataFrame
    final_history: pd.DataFrame
    config: DirectPhaseFitConfig
    shape_config: PhaseShapeConfig


def _static_bounds(
    gna_gk_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if gna_gk_factor <= 1.0:
        raise ValueError("gna_gk_factor must be greater than one")
    ais_lower, ais_upper, _ = extension_parameter_bounds("soma-ais")
    nuisance_width = math.log(gna_gk_factor)
    return (
        np.concatenate(
            (
                (-nuisance_width, -nuisance_width, 40.0, -100.0),
                ais_lower,
            )
        ),
        np.concatenate(
            (
                (nuisance_width, nuisance_width, 70.0, -65.0),
                ais_upper,
            )
        ),
    )


def _baseline_centers(
    baseline: pd.DataFrame,
    config: DirectPhaseFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if baseline.empty:
        raise ValueError("Baseline model table is empty")
    required = set(CELL_PARAMETER_NAMES + AIS_PARAMETER_NAMES)
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(
            f"Baseline model is missing parameters: {sorted(missing)}"
        )
    row = baseline.iloc[0]
    alpha_beta = KineticParameters.from_vector(
        row.loc[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    )
    kinetic = direct_kinetic_initial(alpha_beta)
    static = np.asarray(
        (
            float(row[CELL_PARAMETER_NAMES[-2]]),
            float(row[CELL_PARAMETER_NAMES[-1]]),
            50.0,
            -77.0,
            *row.loc[list(AIS_PARAMETER_NAMES)].to_numpy(dtype=float),
        ),
        dtype=float,
    )
    lower, upper = _static_bounds(config.gna_gk_factor)
    static = np.clip(static, lower + 1e-9, upper - 1e-9)
    return kinetic, static


def _shape_bounds(
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    absolute_lower, absolute_upper = direct_kinetic_parameter_bounds()
    widths = np.asarray(
        [
            (
                3.0
                if name.endswith("start_logit")
                else (
                    math.log(3.0)
                    if "log_tau_knot" in name
                    else 1.0
                )
            )
            for name in DIRECT_KINETIC_PARAMETER_NAMES
        ],
        dtype=float,
    )
    return (
        np.maximum(absolute_lower, center - widths),
        np.minimum(absolute_upper, center + widths),
    )


def _ais_leak_reversal(
    kinetics: DirectKinetics,
    biophysics,
    ais_gna_multiplier: float,
    resting_voltage_mv: float,
) -> float:
    m_gate, h_gate, n_gate = kinetics.steady_state(resting_voltage_mv)
    sodium = (
        biophysics.gna_ms_cm2
        * ais_gna_multiplier
        * m_gate**3
        * h_gate
        * (resting_voltage_mv - biophysics.ena_mv)
    )
    potassium = (
        biophysics.gk_ms_cm2
        * n_gate**4
        * (resting_voltage_mv - biophysics.ek_mv)
    )
    return float(
        resting_voltage_mv
        + (sodium + potassium) / biophysics.gleak_ms_cm2
    )


def decode_direct_model(
    kinetic_parameters: Sequence[float],
    static_parameters: Sequence[float],
    target: CellOptimizationTarget,
) -> DecodedLadderModel:
    kinetics = DirectKinetics.from_parameters(kinetic_parameters)
    static = np.asarray(static_parameters, dtype=float)
    if static.shape != (len(DIRECT_STATIC_PARAMETER_NAMES),):
        raise ValueError(f"Invalid direct static vector: {static.shape}")
    biophysics = admittance_anchored_biophysics(
        target.passive,
        kinetics,
        log_gna_scale=float(static[0]),
        log_gk_scale=float(static[1]),
        ena_mv=float(static[2]),
        ek_mv=float(static[3]),
    )
    area_fraction = 0.05 * math.exp(float(static[4]))
    ais_gna_multiplier = 3.0 * math.exp(float(static[5]))
    coupling_ns = 5.0 * math.exp(float(static[6]))
    ais_eleak = _ais_leak_reversal(
        kinetics,
        biophysics,
        ais_gna_multiplier,
        target.passive.resting_voltage_mv,
    )
    return DecodedLadderModel(
        variant="soma-ais",
        kinetics=kinetics,
        biophysics=biophysics,
        extra_gate=None,
        gslow_ms_cm2=0.0,
        ais_area_fraction=float(area_fraction),
        ais_gna_multiplier=float(ais_gna_multiplier),
        coupling_ns=float(coupling_ns),
        ais_eleak_mv=ais_eleak,
        extension_prior=0.0,
        ais_kinetics=kinetics,
    )


def _normalized_prior(
    values: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    half_width = np.maximum((upper - lower) / 2.0, 1e-12)
    return float(np.mean(((values - center) / half_width) ** 2))


def evaluate_direct_candidate(
    kinetic_parameters: Sequence[float],
    static_parameters: Sequence[float],
    kinetic_prior_center: Sequence[float],
    static_prior_center: Sequence[float],
    kinetic_bounds: tuple[Sequence[float], Sequence[float]],
    static_bounds: tuple[Sequence[float], Sequence[float]],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    fit_config: DirectPhaseFitConfig,
    shape_config: PhaseShapeConfig,
    include_validation: bool = False,
    include_artifacts: bool = False,
) -> DirectPhaseEvaluation:
    try:
        kinetic_values = np.asarray(kinetic_parameters, dtype=float)
        static_values = np.asarray(static_parameters, dtype=float)
        kinetic_lower = np.asarray(kinetic_bounds[0], dtype=float)
        kinetic_upper = np.asarray(kinetic_bounds[1], dtype=float)
        static_lower = np.asarray(static_bounds[0], dtype=float)
        static_upper = np.asarray(static_bounds[1], dtype=float)
        if (
            np.any(kinetic_values < kinetic_lower)
            or np.any(kinetic_values > kinetic_upper)
            or np.any(static_values < static_lower)
            or np.any(static_values > static_upper)
        ):
            raise ValueError("Direct candidate is outside fitted bounds")
        model = decode_direct_model(kinetic_values, static_values, target)
        if (
            not -120.0 <= model.biophysics.eleak_mv <= -20.0
            or not -120.0 <= model.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        factors = model.biophysics.temperature_factors(
            screen_config.temperature_c
        )
        diagnostics = kinetic_diagnostics(model.kinetics, factors)
        resolved_tau_floor = max(
            fit_config.minimum_resolved_tau_ms,
            fit_config.minimum_tau_dt_fraction * screen_config.dt_ms,
        )
        if diagnostics.minimum_tau_ms < resolved_tau_floor:
            raise SimulationError(
                "Gate time constant is below the numerical resolution floor"
            )
        if diagnostics.monotonic_fraction < 0.999:
            raise SimulationError("Direct steady state is non-monotone")
        if diagnostics.maximum_rate_per_ms > 2_000.0:
            raise SimulationError("Gate rate exceeds the kinetic ceiling")
        state = _initial_state(model, target)
        if not coupled_resting_state_is_stable(
            model,
            state,
            screen_config,
        ):
            raise SimulationError("Anchored soma-AIS resting state is unstable")
        lower_rheobase, upper_rheobase, _ = (
            find_rheobase_rush_larsen(
                model,
                state,
                screen_config,
                duration_ms=fit_config.rheobase_probe_duration_ms,
                tolerance_pa=fit_config.rheobase_probe_tolerance_pa,
            )
        )
        protocols = list(target.training_protocols)
        if include_validation:
            protocols.extend(target.validation_protocols)
        traces: dict[str, Trace] = {}
        cycles: dict[str, PhaseCycle] = {}
        protocol_scores: dict[str, PhaseShapeScores] = {}
        for protocol in protocols:
            duration = min(
                protocol.duration_ms,
                fit_config.phase_step_duration_ms,
            )
            current_pa = protocol.rheobase_factor * upper_rheobase
            simulation = step_simulation_config(
                current_pa,
                duration,
                screen_config,
                model.biophysics,
                target.passive.resting_voltage_mv,
            )
            trace = simulate_ladder_rush_larsen(
                model,
                simulation,
                state,
            )
            stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=current_pa,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + duration,
            )
            cycle = extract_phase_cycle(
                trace.time_ms,
                trace.voltage_mv,
                stimulus,
                shape_config,
            )
            score = compare_phase_cycles(
                cycle,
                biological_cycles_target[protocol.name],
                shape_config,
            )
            protocol_scores[protocol.name] = score
            if include_artifacts:
                traces[protocol.name] = trace
                cycles[protocol.name] = cycle
        training = {
            protocol.name: protocol_scores[protocol.name]
            for protocol in target.training_protocols
        }
        phase_scores = mean_phase_scores(training)
        shape_loss = float(
            3.0 * phase_scores.normalized_shape
            + phase_scores.slope_shape
            + 2.5 * phase_scores.concavity
        )
        scale_loss = float(
            1.5 * phase_scores.physical_curve
            + 2.0 * phase_scores.velocity_extent
            + fit_config.physical_constraint_weight
            * phase_scores.physical_constraint_loss
        )
        rheobase_loss = abs(
            upper_rheobase - target.sampled_rheobase_pa
        ) / max(10.0, 0.15 * target.sampled_rheobase_pa)
        rheobase_tolerance = max(
            fit_config.rheobase_tolerance_floor_pa,
            fit_config.rheobase_tolerance_fraction
            * target.sampled_rheobase_pa,
        )
        rheobase_excess = max(
            0.0,
            abs(upper_rheobase - target.sampled_rheobase_pa)
            - rheobase_tolerance,
        ) / rheobase_tolerance
        rheobase_constraint_loss = float(rheobase_excess**2)
        kinetic_prior = _normalized_prior(
            kinetic_values,
            np.asarray(kinetic_prior_center, dtype=float),
            kinetic_lower,
            kinetic_upper,
        )
        static_prior = _normalized_prior(
            static_values,
            np.asarray(static_prior_center, dtype=float),
            static_lower,
            static_upper,
        )
        regularity = float(
            fit_config.rheobase_weight * rheobase_loss
            + fit_config.kinetic_prior_weight * kinetic_prior
            + fit_config.static_prior_weight * static_prior
            + 0.25 * diagnostics.pathology_score
        )
        scale_loss += (
            fit_config.rheobase_constraint_weight
            * rheobase_constraint_loss
        )
        objective = float(shape_loss + scale_loss + regularity)
        if not np.isfinite(objective):
            raise SimulationError("Direct candidate produced a non-finite score")
        biophysics = model.biophysics.to_physical_mapping()
        biophysics.update(
            {
                "extension__ais_area_fraction": model.ais_area_fraction,
                "extension__ais_gna_multiplier": model.ais_gna_multiplier,
                "extension__coupling_ns": model.coupling_ns,
                "extension__ais_eleak_mv": model.ais_eleak_mv,
            }
        )
        return DirectPhaseEvaluation(
            valid=True,
            reason="accepted",
            objective_total=objective,
            shape_loss=shape_loss,
            scale_loss=scale_loss,
            regularity_loss=regularity,
            constraints_satisfied=bool(
                phase_scores.physical_constraint_loss <= 1e-12
                and rheobase_constraint_loss <= 1e-12
            ),
            phase_scores=phase_scores,
            protocol_scores=protocol_scores,
            cycles=cycles if include_artifacts else None,
            traces=traces if include_artifacts else None,
            rheobase_lower_pa=float(lower_rheobase),
            rheobase_upper_pa=float(upper_rheobase),
            rheobase_loss=float(rheobase_loss),
            rheobase_constraint_loss=rheobase_constraint_loss,
            kinetic_prior=kinetic_prior,
            static_prior=static_prior,
            diagnostics=diagnostics,
            biophysics=biophysics,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return DirectPhaseEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            objective_total=fit_config.invalid_score,
        )


def _stage_score(
    evaluation: DirectPhaseEvaluation,
    stage: str,
    config: DirectPhaseFitConfig,
    reference_shape_loss: float,
) -> float:
    if not evaluation.valid:
        return config.invalid_score
    if stage == "shape":
        return float(
            evaluation.shape_loss
            + config.rheobase_constraint_weight
            * evaluation.rheobase_constraint_loss
            + config.shape_scale_guard_weight * evaluation.scale_loss
            + evaluation.regularity_loss
        )
    if stage == "scale":
        guard_limit = (
            (1.0 + config.shape_guard_relative_tolerance)
            * reference_shape_loss
        )
        guard_excess = max(
            0.0,
            evaluation.shape_loss / max(guard_limit, 1e-12) - 1.0,
        )
        return float(
            evaluation.scale_loss
            + config.scale_shape_guard_weight * guard_excess**2
            + evaluation.regularity_loss
        )
    if stage == "joint":
        guard_limit = (
            (1.0 + config.shape_guard_relative_tolerance)
            * reference_shape_loss
        )
        guard_excess = max(
            0.0,
            evaluation.shape_loss / max(guard_limit, 1e-12) - 1.0,
        )
        return float(
            evaluation.objective_total
            + config.scale_shape_guard_weight * guard_excess**2
        )
    raise ValueError(f"Unknown direct-fit stage: {stage}")


def _stage_payload(
    normalized: Sequence[float],
    stage: str,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    fixed_kinetic: np.ndarray,
    fixed_static: np.ndarray,
    kinetic_prior_center: np.ndarray,
    static_prior_center: np.ndarray,
    kinetic_bounds: tuple[np.ndarray, np.ndarray],
    static_bounds: tuple[np.ndarray, np.ndarray],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    fit_config: DirectPhaseFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
) -> tuple[np.ndarray, float, DirectPhaseEvaluation]:
    variables = _bounded_parameters(
        normalized,
        variable_lower,
        variable_upper,
    )
    if stage == "shape":
        kinetic = variables
        static = fixed_static
    elif stage == "scale":
        kinetic = fixed_kinetic
        static = variables
    elif stage == "joint":
        split = len(DIRECT_KINETIC_PARAMETER_NAMES)
        kinetic = variables[:split]
        static = variables[split:]
    else:
        raise ValueError(f"Unknown direct-fit stage: {stage}")
    evaluation = evaluate_direct_candidate(
        kinetic,
        static,
        kinetic_prior_center,
        static_prior_center,
        kinetic_bounds,
        static_bounds,
        target,
        biological_cycles_target,
        screen_config,
        fit_config,
        shape_config,
    )
    return (
        variables,
        _stage_score(
            evaluation,
            stage,
            fit_config,
            reference_shape_loss,
        ),
        evaluation,
    )


def _history_row(
    generation: int,
    individual: int,
    names: Sequence[str],
    variables: np.ndarray,
    score: float,
    evaluation: DirectPhaseEvaluation,
) -> dict[str, object]:
    row: dict[str, object] = {
        "generation": generation,
        "individual": individual,
        **dict(zip(names, map(float, variables))),
        "stage_score": float(score),
        "valid": evaluation.valid,
        "reason": evaluation.reason,
    }
    if evaluation.valid and evaluation.phase_scores is not None:
        row.update(
            {
                "shape_loss": evaluation.shape_loss,
                "scale_loss": evaluation.scale_loss,
                "regularity_loss": evaluation.regularity_loss,
                "constraints_satisfied": evaluation.constraints_satisfied,
                "model_rheobase_pa": evaluation.rheobase_upper_pa,
                "rheobase_loss": evaluation.rheobase_loss,
                "rheobase_constraint_loss": (
                    evaluation.rheobase_constraint_loss
                ),
                **{
                    f"phase__{name}": value
                    for name, value in (
                        evaluation.phase_scores.to_mapping().items()
                    )
                },
            }
        )
    return row


def _optimize_stage(
    stage: str,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    names: Sequence[str],
    fixed_kinetic: np.ndarray,
    fixed_static: np.ndarray,
    kinetic_prior_center: np.ndarray,
    static_prior_center: np.ndarray,
    kinetic_bounds: tuple[np.ndarray, np.ndarray],
    static_bounds: tuple[np.ndarray, np.ndarray],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    fit_config: DirectPhaseFitConfig,
    shape_config: PhaseShapeConfig,
    population_size: int,
    generations: int,
    seed: int,
) -> pd.DataFrame:
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Direct phase fitting requires DEAP") from error
    if population_size < 4:
        raise ValueError("CMA population size must be at least four")
    fitness_name = "FitnessDirectPhaseCMA"
    individual_name = "IndividualDirectPhaseCMA"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)
    np.random.seed(seed)
    strategy = cma.Strategy(
        centroid=_unbounded_parameters(center, lower, upper),
        sigma=fit_config.sigma,
        lambda_=population_size,
    )
    rows: list[dict[str, object]] = []
    center_payload = _stage_payload(
        _unbounded_parameters(center, lower, upper),
        stage,
        lower,
        upper,
        fixed_kinetic,
        fixed_static,
        kinetic_prior_center,
        static_prior_center,
        kinetic_bounds,
        static_bounds,
        target,
        biological_cycles_target,
        screen_config,
        fit_config,
        shape_config,
        float("inf"),
    )
    reference_shape_loss = center_payload[2].shape_loss
    center_score = _stage_score(
        center_payload[2],
        stage,
        fit_config,
        reference_shape_loss,
    )
    rows.append(
        _history_row(
            -1,
            -1,
            names,
            center_payload[0],
            center_score,
            center_payload[2],
        )
    )
    executor = None
    mapper = map
    if fit_config.workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=fit_config.workers)
            mapper = executor.map
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"Parallel evaluation unavailable ({error}); using one worker.",
                RuntimeWarning,
            )
    try:
        for generation in range(generations):
            population = strategy.generate(individual_type)
            payloads = list(
                mapper(
                    _stage_payload,
                    population,
                    [stage] * len(population),
                    [lower] * len(population),
                    [upper] * len(population),
                    [fixed_kinetic] * len(population),
                    [fixed_static] * len(population),
                    [kinetic_prior_center] * len(population),
                    [static_prior_center] * len(population),
                    [kinetic_bounds] * len(population),
                    [static_bounds] * len(population),
                    [target] * len(population),
                    [biological_cycles_target] * len(population),
                    [screen_config] * len(population),
                    [fit_config] * len(population),
                    [shape_config] * len(population),
                    [reference_shape_loss] * len(population),
                )
            )
            for index, (individual, payload) in enumerate(
                zip(population, payloads)
            ):
                variables, score, evaluation = payload
                individual.fitness.values = (score,)
                rows.append(
                    _history_row(
                        generation,
                        index,
                        names,
                        variables,
                        score,
                        evaluation,
                    )
                )
            strategy.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    history = pd.DataFrame(rows)
    if not history["valid"].any():
        raise RuntimeError(f"Direct {stage} stage produced no valid model")
    return history


def _diverse_shape_elites(
    history: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    minimum_distance: float = 0.04,
) -> list[np.ndarray]:
    valid = history.loc[history["valid"]].sort_values("stage_score")
    feasible = valid.loc[
        valid["rheobase_constraint_loss"].le(1e-12)
    ]
    if not feasible.empty:
        valid = feasible
    span = np.maximum(upper - lower, 1e-12)
    selected: list[np.ndarray] = []
    normalized_selected: list[np.ndarray] = []
    for _, row in valid.iterrows():
        values = row.loc[
            list(DIRECT_KINETIC_PARAMETER_NAMES)
        ].to_numpy(dtype=float)
        normalized = (values - lower) / span
        if all(
            np.linalg.norm(normalized - previous)
            / math.sqrt(len(normalized))
            >= minimum_distance
            for previous in normalized_selected
        ):
            selected.append(values)
            normalized_selected.append(normalized)
            if len(selected) >= count:
                break
    if not selected:
        raise RuntimeError("No valid direct shape elite was found")
    return selected


def _joint_trust_bounds(
    center: np.ndarray,
    absolute_lower: np.ndarray,
    absolute_upper: np.ndarray,
    config: DirectPhaseFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    widths: list[float] = []
    for name in DIRECT_ALL_PARAMETER_NAMES:
        if name.endswith("start_logit"):
            widths.append(1.0)
        elif name.endswith(("ena_mv", "ek_mv")):
            widths.append(config.voltage_trust_margin_mv)
        else:
            widths.append(math.log1p(config.trust_fraction))
    width = np.asarray(widths, dtype=float)
    return (
        np.maximum(absolute_lower, center - width),
        np.minimum(absolute_upper, center + width),
    )


def _reevaluate(
    kinetic: np.ndarray,
    static: np.ndarray,
    kinetic_prior_center: np.ndarray,
    static_prior_center: np.ndarray,
    kinetic_bounds: tuple[np.ndarray, np.ndarray],
    static_bounds: tuple[np.ndarray, np.ndarray],
    target: CellOptimizationTarget,
    biological: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    config: DirectPhaseFitConfig,
    shape_config: PhaseShapeConfig,
) -> DirectPhaseEvaluation:
    evaluation = evaluate_direct_candidate(
        kinetic,
        static,
        kinetic_prior_center,
        static_prior_center,
        kinetic_bounds,
        static_bounds,
        target,
        biological,
        screen_config,
        config,
        shape_config,
        include_validation=True,
        include_artifacts=True,
    )
    if not evaluation.valid:
        raise RuntimeError(
            f"Selected direct model failed reevaluation: {evaluation.reason}"
        )
    return evaluation


def _viable_static_seed(
    kinetic: np.ndarray,
    static: np.ndarray,
    kinetic_prior_center: np.ndarray,
    static_prior_center: np.ndarray,
    kinetic_bounds: tuple[np.ndarray, np.ndarray],
    static_bounds: tuple[np.ndarray, np.ndarray],
    target: CellOptimizationTarget,
    biological: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    config: DirectPhaseFitConfig,
    shape_config: PhaseShapeConfig,
) -> np.ndarray:
    """Find the smallest conductance adjustment that restores seed spiking."""
    for gna_factor, gk_factor in (
        (1.0, 1.0),
        (1.05, 1.0),
        (1.10, 1.0),
        (1.15, 0.95),
        (1.25, 0.90),
        (1.35, 0.85),
    ):
        candidate = static.copy()
        candidate[0] += math.log(gna_factor)
        candidate[1] += math.log(gk_factor)
        candidate = np.clip(
            candidate,
            static_bounds[0] + 1e-9,
            static_bounds[1] - 1e-9,
        )
        evaluation = evaluate_direct_candidate(
            kinetic,
            candidate,
            kinetic_prior_center,
            static_prior_center,
            kinetic_bounds,
            static_bounds,
            target,
            biological,
            screen_config,
            config,
            shape_config,
        )
        if evaluation.valid:
            return candidate
    raise RuntimeError(
        "No viable direct-kinetics seed was found within the "
        "conductance initialization ladder"
    )


def optimize_direct_phase_fit(
    target: CellOptimizationTarget,
    baseline: pd.DataFrame,
    screen_config: BiologicalScreenConfig,
    config: DirectPhaseFitConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
) -> DirectPhaseFitResult:
    """Run shape, physical-scale, and joint trust-region stages."""
    config = config or DirectPhaseFitConfig()
    shape_config = shape_config or PhaseShapeConfig()
    biological = biological_phase_cycles(
        target,
        shape_config,
        include_validation=True,
    )
    kinetic_initial, static_prior_center = _baseline_centers(
        baseline,
        config,
    )
    kinetic_absolute = direct_kinetic_parameter_bounds()
    static_absolute = _static_bounds(config.gna_gk_factor)
    kinetic_shape_bounds = _shape_bounds(kinetic_initial)
    static_initial = _viable_static_seed(
        kinetic_initial,
        static_prior_center,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
    )
    initial_evaluation = _reevaluate(
        kinetic_initial,
        static_initial,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
    )

    shape_history = _optimize_stage(
        "shape",
        kinetic_initial,
        kinetic_shape_bounds[0],
        kinetic_shape_bounds[1],
        DIRECT_KINETIC_PARAMETER_NAMES,
        kinetic_initial,
        static_initial,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
        config.shape_population_size,
        config.shape_generations,
        config.seed,
    )
    shape_elites = _diverse_shape_elites(
        shape_history,
        kinetic_shape_bounds[0],
        kinetic_shape_bounds[1],
        config.shape_elites,
    )

    scale_histories: list[pd.DataFrame] = []
    scale_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for elite_index, kinetic_elite in enumerate(shape_elites):
        branch_history = _optimize_stage(
            "scale",
            static_initial,
            static_absolute[0],
            static_absolute[1],
            DIRECT_STATIC_PARAMETER_NAMES,
            kinetic_elite,
            static_initial,
            kinetic_initial,
            static_prior_center,
            kinetic_absolute,
            static_absolute,
            target,
            biological,
            screen_config,
            config,
            shape_config,
            config.scale_population_size,
            config.scale_generations,
            config.seed + 100 + elite_index,
        )
        branch_history.insert(0, "shape_elite", elite_index)
        scale_histories.append(branch_history)
        valid_branch = branch_history.loc[
            branch_history["valid"]
        ]
        feasible_branch = valid_branch.loc[
            valid_branch["rheobase_constraint_loss"].le(1e-12)
        ]
        if not feasible_branch.empty:
            valid_branch = feasible_branch
        best = valid_branch.nsmallest(1, "stage_score").iloc[0]
        scale_candidates.append(
            (
                float(best["stage_score"]),
                kinetic_elite,
                best.loc[
                    list(DIRECT_STATIC_PARAMETER_NAMES)
                ].to_numpy(dtype=float),
            )
        )
    _, shape_selected, scale_selected = min(
        scale_candidates,
        key=lambda item: item[0],
    )
    scale_history = pd.concat(scale_histories, ignore_index=True)

    joint_center = np.concatenate((shape_selected, scale_selected))
    absolute_lower = np.concatenate(
        (kinetic_absolute[0], static_absolute[0])
    )
    absolute_upper = np.concatenate(
        (kinetic_absolute[1], static_absolute[1])
    )
    trust_lower, trust_upper = _joint_trust_bounds(
        joint_center,
        absolute_lower,
        absolute_upper,
        config,
    )
    final_history = _optimize_stage(
        "joint",
        joint_center,
        trust_lower,
        trust_upper,
        DIRECT_ALL_PARAMETER_NAMES,
        shape_selected,
        scale_selected,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
        config.final_population_size,
        config.final_generations,
        config.seed + 1000,
    )
    valid_final = final_history.loc[final_history["valid"]]
    feasible_final = valid_final.loc[
        valid_final["rheobase_constraint_loss"].le(1e-12)
    ]
    if not feasible_final.empty:
        valid_final = feasible_final
    final_row = valid_final.nsmallest(1, "stage_score").iloc[0]
    final_parameters = final_row.loc[
        list(DIRECT_ALL_PARAMETER_NAMES)
    ].to_numpy(dtype=float)
    split = len(DIRECT_KINETIC_PARAMETER_NAMES)
    final_kinetic = final_parameters[:split]
    final_static = final_parameters[split:]

    shape_evaluation = _reevaluate(
        shape_selected,
        static_initial,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
    )
    scale_evaluation = _reevaluate(
        shape_selected,
        scale_selected,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
    )
    final_evaluation = _reevaluate(
        final_kinetic,
        final_static,
        kinetic_initial,
        static_prior_center,
        kinetic_absolute,
        static_absolute,
        target,
        biological,
        screen_config,
        config,
        shape_config,
    )
    return DirectPhaseFitResult(
        target=target,
        biological_cycles=biological,
        initial_parameters=np.concatenate(
            (kinetic_initial, static_initial)
        ),
        shape_parameters=np.concatenate(
            (shape_selected, static_initial)
        ),
        scale_parameters=np.concatenate(
            (shape_selected, scale_selected)
        ),
        final_parameters=final_parameters,
        initial_evaluation=initial_evaluation,
        shape_evaluation=shape_evaluation,
        scale_evaluation=scale_evaluation,
        final_evaluation=final_evaluation,
        shape_history=shape_history,
        scale_history=scale_history,
        final_history=final_history,
        config=config,
        shape_config=shape_config,
    )


def direct_phase_summary(result: DirectPhaseFitResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stages = (
        ("initial_direct", result.initial_evaluation),
        ("shape", result.shape_evaluation),
        ("scale", result.scale_evaluation),
        ("joint", result.final_evaluation),
    )
    for stage, evaluation in stages:
        row: dict[str, object] = {
            "stage": stage,
            "dataset": result.target.dataset,
            "cell_id": result.target.cell_id,
            "objective_total": evaluation.objective_total,
            "shape_loss": evaluation.shape_loss,
            "scale_loss": evaluation.scale_loss,
            "regularity_loss": evaluation.regularity_loss,
            "constraints_satisfied": evaluation.constraints_satisfied,
            "model_rheobase_pa": evaluation.rheobase_upper_pa,
            "biological_rheobase_pa": result.target.sampled_rheobase_pa,
            "rheobase_constraint_loss": (
                evaluation.rheobase_constraint_loss
            ),
            **evaluation.phase_scores.to_mapping(),
        }
        for protocol, scores in evaluation.protocol_scores.items():
            row.update(
                {
                    f"{protocol}__{name}": value
                    for name, value in scores.to_mapping().items()
                }
            )
        if evaluation.biophysics is not None:
            row.update(evaluation.biophysics)
        rows.append(row)
    return pd.DataFrame(rows)


def _phase_voltage(cycle: PhaseCycle) -> tuple[np.ndarray, np.ndarray]:
    up = cycle.threshold_voltage_mv + cycle.grid * cycle.amplitude_mv
    down = (
        cycle.peak_voltage_mv
        - cycle.grid * cycle.repolarization_amplitude_mv
    )
    return up, down


def save_direct_phase_plot(
    result: DirectPhaseFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = (
        ("Biological", None, "#147d7e"),
        ("Initial direct", result.initial_evaluation, "#9c6644"),
        ("Shape", result.shape_evaluation, "#e76f51"),
        ("Scale", result.scale_evaluation, "#457b9d"),
        ("Joint", result.final_evaluation, "#7b2cbf"),
    )
    protocols = (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    )
    figure, axes = plt.subplots(
        len(protocols),
        3,
        figsize=(14.2, 10.2),
        constrained_layout=True,
    )
    for row_index, protocol in enumerate(protocols):
        biological = result.biological_cycles[protocol.name]
        for label, evaluation, color in stages:
            cycle = (
                biological
                if evaluation is None
                else evaluation.cycles[protocol.name]
            )
            up_voltage, down_voltage = _phase_voltage(cycle)
            axes[row_index, 0].plot(
                up_voltage,
                cycle.up_dvdt_mv_ms,
                color=color,
                linewidth=1.1,
                label=label,
            )
            axes[row_index, 0].plot(
                down_voltage,
                cycle.down_dvdt_mv_ms,
                color=color,
                linewidth=1.1,
            )
            axes[row_index, 1].plot(
                cycle.grid,
                cycle.up_normalized,
                color=color,
                linewidth=1.1,
            )
            axes[row_index, 1].plot(
                1.0 - cycle.grid,
                cycle.down_normalized,
                color=color,
                linewidth=1.1,
            )
            axes[row_index, 2].plot(
                down_voltage,
                cycle.down_concavity,
                color=color,
                linewidth=1.1,
            )
        low_boundary = (
            biological.peak_voltage_mv
            - result.shape_config.downstroke_low_voltage_start
            * biological.repolarization_amplitude_mv
        )
        axes[row_index, 2].axvspan(
            biological.down_end_voltage_mv,
            low_boundary,
            color="#eeeeee",
            zorder=-3,
        )
        role = "held out" if protocol.role == "validation" else "training"
        axes[row_index, 0].set_title(
            f"{protocol.name.title()} ({role}): absolute phase plane",
            loc="left",
            fontsize=9,
        )
        axes[row_index, 1].set_title(
            "Normalized branch shape",
            loc="left",
            fontsize=9,
        )
        axes[row_index, 2].set_title(
            "Downstroke concavity; low-V region shaded",
            loc="left",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylabel("dV/dt (mV/ms)")
    axes[-1, 0].set_xlabel("V (mV)")
    axes[-1, 1].set_xlabel("Normalized branch voltage")
    axes[-1, 2].set_xlabel("V (mV)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(
        f"Direct x_inf/tau phase fit: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_trace_plot(
    result: DirectPhaseFitResult,
    screen_config: BiologicalScreenConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = {
        sweep.sweep_number: sweep
        for sweep in read_current_clamp_sweeps(
            result.target.nwb_path,
            long_square_only=True,
        )
    }
    protocols = (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    )
    stages = (
        ("Biological", None, "#147d7e"),
        ("Initial direct", result.initial_evaluation, "#9c6644"),
        ("Shape", result.shape_evaluation, "#e76f51"),
        ("Scale", result.scale_evaluation, "#457b9d"),
        ("Joint", result.final_evaluation, "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        len(protocols),
        1,
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )
    for axis, protocol in zip(axes, protocols):
        sweep = sweeps[protocol.sweep_number]
        biological_mask = (
            (sweep.time_ms >= sweep.stimulus_start_ms - 20.0)
            & (
                sweep.time_ms
                <= min(
                    sweep.stimulus_end_ms,
                    sweep.stimulus_start_ms
                    + result.config.phase_step_duration_ms,
                )
            )
        )
        axis.plot(
            sweep.time_ms[biological_mask] - sweep.stimulus_start_ms,
            sweep.voltage_mv[biological_mask],
            color=stages[0][2],
            linewidth=1.1,
            label=stages[0][0],
        )
        for label, evaluation, color in stages[1:]:
            trace = evaluation.traces[protocol.name]
            model_time = trace.time_ms - screen_config.baseline_ms
            model_mask = (
                (model_time >= -20.0)
                & (
                    model_time
                    <= result.config.phase_step_duration_ms
                )
            )
            axis.plot(
                model_time[model_mask],
                trace.voltage_mv[model_mask],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                label=label,
            )
        role = "held out" if protocol.role == "validation" else "training"
        axis.set_title(
            f"{protocol.name.title()} ({role}), biological "
            f"{protocol.current_pa:g} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=5, fontsize=7)
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"Direct kinetics traces: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_score_plot(
    summary: pd.DataFrame,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "normalized_shape",
        "upstroke_onset_concavity",
        "downstroke_low_voltage_concavity",
        "velocity_extent",
        "physical_constraint_loss",
        "total",
    )
    labels = (
        "Normalized\nshape",
        "Early upstroke\nconcavity",
        "Low-V downstroke\nconcavity",
        "dV/dt\nextrema",
        "Physical\nconstraints",
        "Phase\ntotal",
    )
    colors = ("#9c6644", "#e76f51", "#457b9d", "#7b2cbf")
    x = np.arange(len(metrics))
    width = 0.20
    figure, axis = plt.subplots(
        figsize=(11.2, 4.9),
        constrained_layout=True,
    )
    for index, (_, row) in enumerate(summary.iterrows()):
        axis.bar(
            x + (index - 1.5) * width,
            row.loc[list(metrics)].to_numpy(dtype=float),
            width,
            color=colors[index],
            label=row["stage"].replace("_", " ").title(),
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized loss (lower is better)")
    axis.set_title(
        f"Staged direct fit: {summary['dataset'].iloc[0]} / "
        f"{summary['cell_id'].iloc[0]}",
        loc="left",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=4, fontsize=8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_kinetics_plot(
    result: DirectPhaseFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voltage = np.linspace(-100.0, 60.0, 321)
    split = len(DIRECT_KINETIC_PARAMETER_NAMES)
    stages = (
        ("Initial direct", result.initial_parameters[:split], "#9c6644"),
        ("Shape", result.shape_parameters[:split], "#e76f51"),
        ("Joint", result.final_parameters[:split], "#7b2cbf"),
    )
    factors = result.final_evaluation.biophysics
    q10 = (
        factors["physical__q10_m"],
        factors["physical__q10_h"],
        factors["physical__q10_n"],
    )
    exponent = (result.target.temperature_c - 6.3) / 10.0
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.8, 9.0),
        constrained_layout=True,
    )
    for row, gate in enumerate(("m", "h", "n")):
        for label, parameters, color in stages:
            kinetics = DirectKinetics.from_parameters(parameters)
            steady, tau = kinetics.gate_curves(gate, voltage)
            axes[row, 0].plot(
                voltage,
                steady,
                color=color,
                label=label,
            )
            axes[row, 1].plot(
                voltage,
                tau / (q10[row] ** exponent),
                color=color,
            )
        axes[row, 0].set_ylabel(f"{gate}_inf")
        axes[row, 1].set_ylabel(
            f"tau_{gate} at {result.target.temperature_c:g} C (ms)"
        )
        axes[row, 1].set_yscale("log")
        for axis in axes[row]:
            axis.set_xlabel("V (mV)")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Direct gate kinetics: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_phase_fit_result(
    result: DirectPhaseFitResult,
    screen_config: BiologicalScreenConfig,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = direct_phase_summary(result)
    summary.to_csv(output / "stage_summary.csv", index=False)
    pd.DataFrame(
        (
            {
                "stage": stage,
                **dict(zip(DIRECT_ALL_PARAMETER_NAMES, parameters)),
            }
            for stage, parameters in (
                ("initial_direct", result.initial_parameters),
                ("shape", result.shape_parameters),
                ("scale", result.scale_parameters),
                ("joint", result.final_parameters),
            )
        )
    ).to_csv(output / "stage_parameters.csv", index=False)
    result.shape_history.to_csv(output / "shape_history.csv", index=False)
    result.scale_history.to_csv(output / "scale_history.csv", index=False)
    result.final_history.to_csv(output / "joint_history.csv", index=False)
    metadata = {
        "version": 1,
        "experiment": "staged direct x_inf/tau phase fit",
        "target": target_metadata(result.target),
        "fit_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
        "training_protocols": [
            protocol.name for protocol in result.target.training_protocols
        ],
        "held_out_protocols": [
            protocol.name for protocol in result.target.validation_protocols
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_direct_phase_plot(
        result,
        output / "direct_phase_comparison.png",
    )
    save_direct_trace_plot(
        result,
        screen_config,
        output / "direct_trace_comparison.png",
    )
    save_direct_score_plot(
        summary,
        output / "direct_score_bars.png",
    )
    save_direct_kinetics_plot(
        result,
        output / "direct_gate_kinetics.png",
    )
