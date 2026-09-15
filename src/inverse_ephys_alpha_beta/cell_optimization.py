"""Cell-specific multi-objective fitting of HH alpha/beta kinetics."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .cell_targets import (
    admittance_anchored_biophysics,
    CellOptimizationTarget,
    ProtocolFeatureTarget,
    anchored_biophysics,
    resting_state_is_stable,
    target_metadata,
    target_protocol_table,
)
from .features import FeatureConfig, extract_observed_features
from .hh_model import SimulationError, Stimulus, Trace, elicits_spike, simulate
from .kinetics import KineticParameters, PARAMETER_NAMES
from .protocols import (
    BiologicalScreenConfig,
    find_rheobase,
    step_simulation_config,
    subthreshold_features,
)
from .raw_patchseq import read_current_clamp_sweeps


NUISANCE_PARAMETER_NAMES = (
    "param__nuisance__log_gna_scale",
    "param__nuisance__log_gk_scale",
)
CELL_PARAMETER_NAMES = PARAMETER_NAMES + NUISANCE_PARAMETER_NAMES

OBJECTIVE_NAMES = (
    "passive",
    "excitability",
    "spike_shape",
    "spike_dynamics",
    "phase_geometry",
    "firing_pattern",
    "conductance_prior",
)

FEATURE_GROUPS = {
    "spike_shape": (
        "first_threshold_voltage_mv",
        "first_peak_voltage_mv",
        "first_ap_amplitude_mv",
        "first_half_width_ms",
        "first_fast_trough_voltage_mv",
    ),
    "spike_dynamics": (
        "first_upstroke_mv_ms",
        "first_downstroke_mv_ms",
        "first_upstroke_downstroke_ratio",
        "first_ap_upstroke_time_ms",
        "first_ap_repolarization_time_ms",
        "first_ap_duration_ms",
        "first_onset_voltage_mv",
        "first_onset_rapidness_per_ms",
    ),
    "phase_geometry": (
        "upstroke_inflection_voltage_mv",
        "upstroke_inflection_dvdt_mv_ms",
        "downstroke_inflection_voltage_mv",
        "downstroke_inflection_dvdt_mv_ms",
        "upstroke_inflection_relative_to_threshold_mv",
        "downstroke_inflection_relative_to_peak_mv",
        "first_ap_phase_area_normalized",
        "first_cycle_phase_area_normalized",
        "first_cycle_phase_path_length_normalized",
    ),
    "firing_pattern": (
        "spike_count",
        "firing_rate_hz",
        "first_spike_latency_ms",
        "terminal_silence_ms",
        "late_spike_fraction",
        "mean_isi_ms",
        "isi_cv",
        "adaptation_index",
    ),
}

_ABSOLUTE_FEATURE_SCALES = {
    "first_threshold_voltage_mv": 4.0,
    "first_peak_voltage_mv": 8.0,
    "first_ap_amplitude_mv": 10.0,
    "first_half_width_ms": 0.25,
    "first_fast_trough_voltage_mv": 7.0,
    "first_upstroke_mv_ms": 40.0,
    "first_downstroke_mv_ms": 25.0,
    "first_upstroke_downstroke_ratio": 0.4,
    "first_ap_upstroke_time_ms": 0.35,
    "first_ap_repolarization_time_ms": 0.5,
    "first_ap_duration_ms": 0.6,
    "first_onset_voltage_mv": 5.0,
    "first_onset_rapidness_per_ms": 8.0,
    "upstroke_inflection_voltage_mv": 6.0,
    "upstroke_inflection_dvdt_mv_ms": 35.0,
    "downstroke_inflection_voltage_mv": 8.0,
    "downstroke_inflection_dvdt_mv_ms": 25.0,
    "upstroke_inflection_relative_to_threshold_mv": 5.0,
    "downstroke_inflection_relative_to_peak_mv": 7.0,
    "first_ap_phase_area_normalized": 0.15,
    "first_cycle_phase_area_normalized": 0.15,
    "first_cycle_phase_path_length_normalized": 0.4,
    "isi_cv": 0.15,
    "adaptation_index": 0.20,
    "late_spike_fraction": 0.15,
}


@dataclass(frozen=True)
class CellOptimizationConfig:
    population_size: int = 32
    generations: int = 10
    workers: int = 1
    seed: int = 42
    gna_gk_factor: float = 1.5
    crossover_probability: float = 0.9
    mutation_probability: float = 0.35
    mutation_eta: float = 20.0
    missing_feature_penalty: float = 25.0
    invalid_model_penalty: float = 100.0
    representative_prior_weight: float = 0.25
    viability_step_ms: float = 250.0
    passive_reanchoring: bool = False
    evaluation_profile: str = "full"
    fast_protocol_duration_ms: float = 250.0


@dataclass(frozen=True)
class CandidateEvaluation:
    objectives: tuple[float, ...]
    valid: bool
    reason: str
    rheobase_lower_pa: float = float("nan")
    rheobase_upper_pa: float = float("nan")
    biophysics: Mapping[str, float] | None = None
    passive_features: Mapping[str, float] | None = None
    protocol_features: Mapping[str, Mapping[str, float]] | None = None
    traces: Mapping[str, Trace] | None = None


@dataclass(frozen=True)
class CellOptimizationResult:
    target: CellOptimizationTarget
    history: pd.DataFrame
    population: pd.DataFrame
    pareto_front: pd.DataFrame
    representative: pd.DataFrame
    validation: pd.DataFrame
    feature_comparison: pd.DataFrame
    metadata: Mapping[str, object]


def load_cell_parameter_bounds(
    path: str | Path,
    gna_gk_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the 18 kinetic bounds and append narrow conductance bounds."""
    if gna_gk_factor <= 1.0:
        raise ValueError("gna_gk_factor must be greater than one")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = document.get("bounds", document)
    missing = set(PARAMETER_NAMES).difference(bounds)
    if missing:
        raise ValueError(f"Kinetic parameter bounds are missing: {sorted(missing)}")
    lower = [float(bounds[name][0]) for name in PARAMETER_NAMES]
    upper = [float(bounds[name][1]) for name in PARAMETER_NAMES]
    nuisance_width = math.log(gna_gk_factor)
    lower.extend((-nuisance_width, -nuisance_width))
    upper.extend((nuisance_width, nuisance_width))
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if np.any(lower_array >= upper_array):
        raise ValueError("All parameter lower bounds must be below upper bounds")
    return lower_array, upper_array


def _huber(residual: float) -> float:
    absolute = abs(residual)
    return 0.5 * residual**2 if absolute <= 1.0 else absolute - 0.5


def feature_scale(feature_name: str, target_value: float) -> float:
    """Return interpretable per-feature tolerances for objective normalization."""
    if feature_name in _ABSOLUTE_FEATURE_SCALES:
        return _ABSOLUTE_FEATURE_SCALES[feature_name]
    if feature_name == "spike_count":
        return max(1.0, 0.15 * abs(target_value))
    if feature_name == "firing_rate_hz":
        return max(2.0, 0.15 * abs(target_value))
    if feature_name in (
        "first_spike_latency_ms",
        "terminal_silence_ms",
        "mean_isi_ms",
    ):
        return max(2.0, 0.15 * abs(target_value))
    return max(1e-6, 0.15 * abs(target_value))


def _group_score(
    model_features: Mapping[str, float],
    target_features: Mapping[str, float],
    feature_names: Sequence[str],
    missing_penalty: float,
) -> float:
    losses = []
    for feature_name in feature_names:
        target_value = float(target_features.get(feature_name, float("nan")))
        if not np.isfinite(target_value):
            continue
        model_value = float(model_features.get(feature_name, float("nan")))
        if not np.isfinite(model_value):
            losses.append(missing_penalty)
            continue
        residual = (
            model_value - target_value
        ) / feature_scale(feature_name, target_value)
        losses.append(_huber(float(residual)))
    return float(np.mean(losses)) if losses else 0.0


def _mean_protocol_group_score(
    model_features: Mapping[str, Mapping[str, float]],
    protocols: Sequence[ProtocolFeatureTarget],
    group_name: str,
    missing_penalty: float,
) -> float:
    return float(
        np.mean(
            [
                _group_score(
                    model_features[protocol.name],
                    protocol.features,
                    FEATURE_GROUPS[group_name],
                    missing_penalty,
                )
                for protocol in protocols
            ]
        )
    )


_TRANSLATABLE_VOLTAGE_FEATURES = (
    "first_threshold_voltage_mv",
    "first_peak_voltage_mv",
    "first_fast_trough_voltage_mv",
    "first_onset_voltage_mv",
    "upstroke_inflection_voltage_mv",
    "downstroke_inflection_voltage_mv",
)


def _translation_aligned_features(
    model_features: Mapping[str, float],
    target_features: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    """Align spike voltage origin while preserving amplitude and velocity."""
    model_threshold = float(
        model_features.get("first_threshold_voltage_mv", float("nan"))
    )
    target_threshold = float(
        target_features.get("first_threshold_voltage_mv", float("nan"))
    )
    offset = (
        target_threshold - model_threshold
        if np.isfinite(model_threshold) and np.isfinite(target_threshold)
        else 0.0
    )
    aligned = dict(model_features)
    for feature_name in _TRANSLATABLE_VOLTAGE_FEATURES:
        value = float(aligned.get(feature_name, float("nan")))
        if np.isfinite(value):
            aligned[feature_name] = value + offset
    return aligned, float(offset)


def _fast_protocol_group_score(
    model_features: Mapping[str, Mapping[str, float]],
    protocols: Sequence[ProtocolFeatureTarget],
    group_name: str,
    missing_penalty: float,
) -> float:
    losses = []
    offsets = []
    for protocol in protocols:
        aligned, offset = _translation_aligned_features(
            model_features[protocol.name],
            protocol.features,
        )
        losses.append(
            _group_score(
                aligned,
                protocol.features,
                FEATURE_GROUPS[group_name],
                missing_penalty,
            )
        )
        offsets.append(_huber(offset / 10.0))
    if not losses:
        return 0.0
    score = float(np.mean(losses))
    return score + (
        0.1 * float(np.mean(offsets))
        if group_name == "spike_shape"
        else 0.0
    )


def _initial_state(
    target: CellOptimizationTarget,
    kinetics: KineticParameters,
) -> np.ndarray:
    voltage = target.passive.resting_voltage_mv
    return np.asarray((voltage, *kinetics.steady_state(voltage)), dtype=float)


def _simulate_protocol(
    kinetics: KineticParameters,
    target: CellOptimizationTarget,
    protocol: ProtocolFeatureTarget,
    model_rheobase_pa: float,
    screen_config: BiologicalScreenConfig,
    biophysics,
    initial_state: np.ndarray,
    feature_config: FeatureConfig,
) -> tuple[Trace, dict[str, float]]:
    current_pa = protocol.rheobase_factor * model_rheobase_pa
    simulation_config = step_simulation_config(
        current_pa=current_pa,
        step_duration_ms=protocol.duration_ms,
        config=screen_config,
        biophysics=biophysics,
        resting_voltage_mv=target.passive.resting_voltage_mv,
    )
    trace = simulate(kinetics, simulation_config, initial_state)
    features = extract_observed_features(
        trace,
        simulation_config.stimulus,
        feature_config,
    )
    return trace, features


def evaluate_cell_candidate(
    vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: CellOptimizationConfig,
    include_validation: bool = False,
    include_traces: bool = False,
) -> CandidateEvaluation:
    """Evaluate one parameter vector with training protocols and optional holdout."""
    invalid = tuple(config.invalid_model_penalty for _ in OBJECTIVE_NAMES)
    try:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(CELL_PARAMETER_NAMES),):
            raise ValueError(
                f"Expected {len(CELL_PARAMETER_NAMES)} parameters, got {values.shape}"
            )
        kinetics = KineticParameters.from_vector(values[: len(PARAMETER_NAMES)])
        anchor_builder = (
            admittance_anchored_biophysics
            if config.passive_reanchoring
            else anchored_biophysics
        )
        biophysics = anchor_builder(
            target.passive,
            kinetics,
            log_gna_scale=float(values[-2]),
            log_gk_scale=float(values[-1]),
        )
        if not -120.0 <= biophysics.eleak_mv <= -20.0:
            raise SimulationError("Derived ELeak is outside the accepted range")
        if not resting_state_is_stable(target.passive, kinetics, biophysics):
            raise SimulationError("Anchored resting state is unstable")
        state = _initial_state(target, kinetics)

        passive_simulation = step_simulation_config(
            current_pa=target.passive.passive_current_pa,
            step_duration_ms=target.passive.passive_duration_ms,
            config=screen_config,
            biophysics=biophysics,
            resting_voltage_mv=target.passive.resting_voltage_mv,
        )
        passive_trace = simulate(kinetics, passive_simulation, state)
        passive_observed = extract_observed_features(
            passive_trace,
            passive_simulation.stimulus,
            FeatureConfig(min_spikes=1),
        )
        if passive_observed.get("spike_count", 0.0) > 0.0:
            raise SimulationError(
                "Model spikes during the hyperpolarizing passive protocol"
            )
        passive = subthreshold_features(
            passive_trace,
            target.passive.resting_voltage_mv,
            target.passive.passive_current_pa,
            passive_simulation.stimulus,
        )
        if passive["input_resistance_mohm"] <= 0.0:
            raise SimulationError(
                "Passive response has the wrong voltage-deflection polarity"
            )
        viability_duration = min(
            config.viability_step_ms,
            screen_config.rheobase_step_ms,
        )
        viability_currents = screen_config.rheobase_currents_pa
        viable = any(
            elicits_spike(
                kinetics,
                step_simulation_config(
                    current_pa=current_pa,
                    step_duration_ms=viability_duration,
                    config=screen_config,
                    biophysics=biophysics,
                    resting_voltage_mv=target.passive.resting_voltage_mv,
                ),
                state,
                max_step_ms=screen_config.spike_test_max_step_ms,
            )
            for current_pa in viability_currents
        )
        if not viable:
            raise SimulationError(
                f"No spike within the {viability_duration:g} ms viability screen"
            )
        rheobase_lower, rheobase_upper, _ = find_rheobase(
            kinetics,
            biophysics,
            state,
            screen_config,
        )
        if config.evaluation_profile not in ("full", "fast"):
            raise ValueError(
                f"Unknown evaluation profile: {config.evaluation_profile}"
            )
        protocols = list(target.training_protocols)
        if config.evaluation_profile == "fast":
            protocols = [
                replace(
                    target.training_protocols[0],
                    duration_ms=min(
                        target.training_protocols[0].duration_ms,
                        config.fast_protocol_duration_ms,
                    ),
                )
            ]
        elif include_validation:
            protocols.extend(target.validation_protocols)
        protocol_features: dict[str, Mapping[str, float]] = {}
        traces: dict[str, Trace] = {}
        for protocol in protocols:
            trace, features = _simulate_protocol(
                kinetics,
                target,
                protocol,
                rheobase_upper,
                screen_config,
                biophysics,
                state,
                FeatureConfig(min_spikes=1),
            )
            protocol_features[protocol.name] = features
            if include_traces:
                traces[protocol.name] = trace
        if include_traces:
            traces["passive"] = passive_trace

        passive_losses = (
            _huber(
                (
                    float(passive["input_resistance_mohm"])
                    - target.passive.input_resistance_mohm
                )
                / max(10.0, 0.15 * target.passive.input_resistance_mohm)
            ),
            _huber(
                (
                    float(passive["membrane_tau_ms"])
                    - target.passive.membrane_tau_ms
                )
                / max(1.0, 0.15 * target.passive.membrane_tau_ms)
            ),
        )
        passive_score = float(np.mean(passive_losses))
        excitability_score = _huber(
            (rheobase_upper - target.sampled_rheobase_pa)
            / max(10.0, 0.15 * target.sampled_rheobase_pa)
        )
        training = protocols
        group_scorer = (
            _fast_protocol_group_score
            if config.evaluation_profile == "fast"
            else _mean_protocol_group_score
        )
        shape_score = group_scorer(
            protocol_features,
            training,
            "spike_shape",
            config.missing_feature_penalty,
        )
        dynamics_score = group_scorer(
            protocol_features,
            training,
            "spike_dynamics",
            config.missing_feature_penalty,
        )
        phase_score = group_scorer(
            protocol_features,
            training,
            "phase_geometry",
            config.missing_feature_penalty,
        )
        firing_score = (
            0.0
            if config.evaluation_profile == "fast"
            else _mean_protocol_group_score(
                protocol_features,
                training,
                "firing_pattern",
                config.missing_feature_penalty,
            )
        )
        nuisance_width = math.log(config.gna_gk_factor)
        conductance_prior = float(
            np.mean((values[-2:] / nuisance_width) ** 2)
        )
        objectives = (
            passive_score,
            float(excitability_score),
            shape_score,
            dynamics_score,
            phase_score,
            firing_score,
            conductance_prior,
        )
        if not np.all(np.isfinite(objectives)):
            raise SimulationError("Candidate produced non-finite objective values")
        return CandidateEvaluation(
            objectives=objectives,
            valid=True,
            reason="accepted",
            rheobase_lower_pa=float(rheobase_lower),
            rheobase_upper_pa=float(rheobase_upper),
            biophysics=biophysics.to_physical_mapping(),
            passive_features=passive,
            protocol_features=protocol_features,
            traces=traces if include_traces else None,
        )
    except (FloatingPointError, OverflowError, SimulationError, ValueError) as error:
        return CandidateEvaluation(
            objectives=invalid,
            valid=False,
            reason=f"{type(error).__name__}: {error}",
        )


def _evaluate_objectives(
    vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: CellOptimizationConfig,
) -> tuple[float, ...]:
    return evaluate_cell_candidate(
        vector,
        target,
        screen_config,
        config,
    ).objectives


def _seed_validity_payload(
    vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: CellOptimizationConfig,
) -> bool:
    return evaluate_cell_candidate(
        vector,
        target,
        screen_config,
        config,
    ).valid


def _seed_score(
    row: pd.Series,
    target: CellOptimizationTarget,
    missing_penalty: float,
) -> float:
    model_features = {
        column.removeprefix("feature__"): float(value)
        for column, value in row.items()
        if column.startswith("feature__")
        and pd.notna(value)
    }
    rheobase = float(model_features.get("rheobase_pa", float("nan")))
    scores = []
    if np.isfinite(rheobase):
        rheobase_score = _huber(
            (rheobase - target.sampled_rheobase_pa)
            / max(10.0, 0.15 * target.sampled_rheobase_pa)
        )
        scores.extend((rheobase_score, rheobase_score))
    rheobase_target = target.training_protocols[0]
    for group_name in FEATURE_GROUPS:
        scores.append(
            _group_score(
                model_features,
                rheobase_target.features,
                FEATURE_GROUPS[group_name],
                missing_penalty,
            )
        )
    return float(np.mean(scores)) if scores else float("inf")


def seed_vectors_from_population(
    population: pd.DataFrame | None,
    target: CellOptimizationTarget,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_seeds: int,
    missing_penalty: float,
) -> list[np.ndarray]:
    """Rank a previous screen against the target and convert it to search vectors."""
    if population is None or population.empty or maximum_seeds <= 0:
        return []
    missing = set(PARAMETER_NAMES).difference(population.columns)
    if missing:
        raise ValueError(f"Seed population is missing parameters: {sorted(missing)}")
    candidates = population.copy()
    candidates["_seed_score"] = candidates.apply(
        _seed_score,
        axis=1,
        args=(target, missing_penalty),
    )
    rheobase_column = "feature__rheobase_pa"
    if rheobase_column in candidates:
        candidates["_rheobase_distance"] = (
            pd.to_numeric(candidates[rheobase_column], errors="coerce")
            - target.sampled_rheobase_pa
        ).abs()
    else:
        candidates["_rheobase_distance"] = np.inf
    objective_ranked = candidates.sort_values("_seed_score")
    rheobase_ranked = candidates.sort_values("_rheobase_distance")
    ordered_indices = []
    for objective_index, rheobase_index in zip(
        objective_ranked.index,
        rheobase_ranked.index,
    ):
        for index in (objective_index, rheobase_index):
            if index not in ordered_indices:
                ordered_indices.append(index)
            if len(ordered_indices) >= maximum_seeds:
                break
        if len(ordered_indices) >= maximum_seeds:
            break
    ranked = candidates.loc[ordered_indices]
    seeds = []
    for _, row in ranked.iterrows():
        vector = row.loc[list(PARAMETER_NAMES)].to_numpy(dtype=float, copy=True)
        temperature_exponent = (target.temperature_c - 6.3) / 10.0
        for gate_name in ("m", "h", "n"):
            source_q10 = float(row.get(f"param__static__q10_{gate_name}", 3.0))
            if source_q10 <= 0.0:
                continue
            effective_rate_shift = temperature_exponent * math.log(
                source_q10 / 3.0
            )
            for rate_prefix in ("alpha", "beta"):
                parameter_name = (
                    f"param__{rate_prefix}_{gate_name}__log_rate_scale"
                )
                parameter_index = PARAMETER_NAMES.index(parameter_name)
                vector[parameter_index] += effective_rate_shift
        nuisance = np.asarray(
            (
                float(row.get("param__static__log_gna_scale", 0.0)),
                float(row.get("param__static__log_gk_scale", 0.0)),
            )
        )
        seeds.append(np.clip(np.concatenate((vector, nuisance)), lower, upper))
    return seeds


def _population_frame(
    population: Iterable,
    generation: int,
    invalid_penalty: float,
) -> pd.DataFrame:
    rows = []
    for index, individual in enumerate(population):
        row = {
            "generation": generation,
            "individual": index,
            **dict(zip(CELL_PARAMETER_NAMES, map(float, individual))),
            **dict(zip(OBJECTIVE_NAMES, map(float, individual.fitness.values))),
            "crowding_distance": float(
                getattr(individual.fitness, "crowding_dist", float("nan"))
            ),
        }
        objective_values = np.asarray(individual.fitness.values, dtype=float)
        row["objective_sum"] = float(np.sum(objective_values))
        row["valid"] = bool(
            not np.allclose(
                objective_values,
                invalid_penalty,
                rtol=0.0,
                atol=1e-12,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_checkpoint(
    output_dir: Path,
    population: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("checkpoint_population.csv", population),
        ("checkpoint_history.csv", history),
    ):
        destination = output_dir / name
        temporary = destination.with_suffix(".csv.part")
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)


def representative_feature_comparison(
    target: CellOptimizationTarget,
    evaluation: CandidateEvaluation,
    config: CellOptimizationConfig,
) -> pd.DataFrame:
    """Return measurement-matched feature residuals for one fitted model."""
    rows = []
    if evaluation.passive_features is not None:
        passive_targets = {
            "input_resistance_mohm": target.passive.input_resistance_mohm,
            "membrane_tau_ms": target.passive.membrane_tau_ms,
        }
        passive_scales = {
            "input_resistance_mohm": max(
                10.0,
                0.15 * target.passive.input_resistance_mohm,
            ),
            "membrane_tau_ms": max(
                1.0,
                0.15 * target.passive.membrane_tau_ms,
            ),
        }
        for feature_name, target_value in passive_targets.items():
            model_value = float(evaluation.passive_features[feature_name])
            scale = passive_scales[feature_name]
            residual = (model_value - target_value) / scale
            rows.append(
                {
                    "protocol": "passive",
                    "role": "training",
                    "feature": feature_name,
                    "biological_value": target_value,
                    "model_value": model_value,
                    "scale": scale,
                    "normalized_residual": residual,
                    "huber_loss": _huber(float(residual)),
                }
            )
    if np.isfinite(evaluation.rheobase_upper_pa):
        scale = max(10.0, 0.15 * target.sampled_rheobase_pa)
        residual = (
            evaluation.rheobase_upper_pa - target.sampled_rheobase_pa
        ) / scale
        rows.append(
            {
                "protocol": "rheobase_search",
                "role": "training",
                "feature": "rheobase_pa",
                "biological_value": target.sampled_rheobase_pa,
                "model_value": evaluation.rheobase_upper_pa,
                "scale": scale,
                "normalized_residual": residual,
                "huber_loss": _huber(float(residual)),
            }
        )
    if evaluation.protocol_features is None:
        return pd.DataFrame(rows)
    for protocol in (
        *target.training_protocols,
        *target.validation_protocols,
    ):
        if protocol.name not in evaluation.protocol_features:
            continue
        model_features = evaluation.protocol_features[protocol.name]
        feature_names = dict.fromkeys(
            feature_name
            for group_features in FEATURE_GROUPS.values()
            for feature_name in group_features
        )
        for feature_name in feature_names:
            target_value = float(
                protocol.features.get(feature_name, float("nan"))
            )
            model_value = float(
                model_features.get(feature_name, float("nan"))
            )
            if not np.isfinite(target_value):
                continue
            scale = feature_scale(feature_name, target_value)
            residual = (
                (model_value - target_value) / scale
                if np.isfinite(model_value)
                else float("nan")
            )
            rows.append(
                {
                    "protocol": protocol.name,
                    "role": protocol.role,
                    "feature": feature_name,
                    "biological_value": target_value,
                    "model_value": model_value,
                    "scale": scale,
                    "normalized_residual": residual,
                    "huber_loss": (
                        _huber(float(residual))
                        if np.isfinite(residual)
                        else config.missing_feature_penalty
                    ),
                }
            )
    return pd.DataFrame(rows)


def optimize_cell(
    target: CellOptimizationTarget,
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    config: CellOptimizationConfig | None = None,
    seed_population: pd.DataFrame | None = None,
    checkpoint_dir: str | Path | None = None,
) -> CellOptimizationResult:
    """Run an NSGA-II fit and retain a held-out sweep for final validation."""
    try:
        from deap import base, creator, tools
    except ImportError as error:
        raise ImportError(
            "Cell optimization requires the optional 'optim' dependencies"
        ) from error

    config = config or CellOptimizationConfig()
    if config.population_size < 4 or config.population_size % 4:
        raise ValueError("population_size must be at least four and divisible by four")
    if config.generations < 0:
        raise ValueError("generations cannot be negative")
    lower, upper = load_cell_parameter_bounds(
        bounds_path,
        config.gna_gk_factor,
    )
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    fitness_name = "FitnessCellAlphaBeta"
    individual_name = "IndividualCellAlphaBeta"
    if not hasattr(creator, fitness_name):
        creator.create(
            fitness_name,
            base.Fitness,
            weights=(-1.0,) * len(OBJECTIVE_NAMES),
        )
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)

    seeds = seed_vectors_from_population(
        seed_population,
        target,
        lower,
        upper,
        maximum_seeds=(
            len(seed_population)
            if config.passive_reanchoring
            else config.population_size // 2
        ),
        missing_penalty=config.missing_feature_penalty,
    )
    canonical_seed = np.clip(
        np.zeros(len(CELL_PARAMETER_NAMES)),
        lower,
        upper,
    )
    seeds.insert(0, canonical_seed)
    if config.passive_reanchoring:
        admissible_seeds = []
        for seed in seeds:
            try:
                admittance_anchored_biophysics(
                    target.passive,
                    KineticParameters.from_vector(
                        seed[: len(PARAMETER_NAMES)]
                    ),
                    log_gna_scale=float(seed[-2]),
                    log_gk_scale=float(seed[-1]),
                )
                admissible_seeds.append(seed)
            except ValueError:
                continue
        seeds = admissible_seeds
        viable_seeds = []
        target_count = config.population_size // 2
        batch_size = max(target_count * 4, config.workers * 4, 4)
        seed_executor = (
            ProcessPoolExecutor(max_workers=config.workers)
            if config.workers > 1
            else None
        )
        seed_mapper = seed_executor.map if seed_executor is not None else map
        try:
            for start in range(0, len(seeds), batch_size):
                batch = seeds[start : start + batch_size]
                validity = list(
                    seed_mapper(
                        _seed_validity_payload,
                        batch,
                        [target] * len(batch),
                        [screen_config] * len(batch),
                        [config] * len(batch),
                    )
                )
                viable_seeds.extend(
                    seed
                    for seed, valid in zip(batch, validity)
                    if valid
                )
                if len(viable_seeds) >= target_count:
                    break
        finally:
            if seed_executor is not None:
                seed_executor.shutdown(wait=True)
        viable_seeds = viable_seeds[:target_count]
        seeds = viable_seeds
    unique_seeds = []
    seen = set()
    for seed in seeds:
        key = tuple(np.round(seed, 10))
        if key not in seen:
            unique_seeds.append(seed)
            seen.add(key)
        if len(unique_seeds) >= config.population_size // 2:
            break
    random_count = config.population_size - len(unique_seeds)
    random_vectors = []
    attempts = 0
    while len(random_vectors) < random_count:
        attempts += 1
        if attempts > 100 * max(1, random_count):
            raise RuntimeError(
                "Could not sample enough passively admissible candidates"
            )
        vector = rng.uniform(lower, upper)
        if config.passive_reanchoring:
            try:
                admittance_anchored_biophysics(
                    target.passive,
                    KineticParameters.from_vector(
                        vector[: len(PARAMETER_NAMES)]
                    ),
                    log_gna_scale=float(vector[-2]),
                    log_gk_scale=float(vector[-1]),
                )
            except ValueError:
                continue
        random_vectors.append(vector)
    population = [
        individual_type(vector.tolist())
        for vector in (*unique_seeds, *random_vectors)
    ]

    toolbox = base.Toolbox()
    toolbox.register(
        "evaluate",
        _evaluate_objectives,
        target=target,
        screen_config=screen_config,
        config=config,
    )
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=lower.tolist(),
        up=upper.tolist(),
        eta=15.0,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=lower.tolist(),
        up=upper.tolist(),
        eta=config.mutation_eta,
        indpb=1.0 / len(CELL_PARAMETER_NAMES),
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("clone", copy.deepcopy)

    executor = None
    effective_workers = config.workers
    if config.workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=config.workers)
            toolbox.register("map", executor.map)
        except (OSError, PermissionError) as error:
            effective_workers = 1
            warnings.warn(
                f"Parallel evaluation is unavailable ({error}); using one worker.",
                RuntimeWarning,
            )
            toolbox.register("map", map)
    else:
        toolbox.register("map", map)

    history_frames = []
    pareto = tools.ParetoFront(
        similar=lambda left, right: np.allclose(left, right, rtol=0.0, atol=1e-10)
    )
    try:
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = fitness
        population = toolbox.select(population, len(population))
        pareto.update(population)
        history_frames.append(
            _population_frame(
                population,
                0,
                config.invalid_model_penalty,
            )
        )
        if checkpoint_dir is not None:
            _write_checkpoint(
                Path(checkpoint_dir),
                history_frames[-1],
                pd.concat(history_frames, ignore_index=True),
            )

        for generation in range(1, config.generations + 1):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for left, right in zip(offspring[::2], offspring[1::2]):
                if random.random() <= config.crossover_probability:
                    toolbox.mate(left, right)
                    del left.fitness.values
                    del right.fitness.values
            for individual in offspring:
                if random.random() <= config.mutation_probability:
                    toolbox.mutate(individual)
                    if individual.fitness.valid:
                        del individual.fitness.values
            invalid_individuals = [
                individual
                for individual in offspring
                if not individual.fitness.valid
            ]
            fitnesses = list(
                toolbox.map(toolbox.evaluate, invalid_individuals)
            )
            for individual, fitness in zip(invalid_individuals, fitnesses):
                individual.fitness.values = fitness
            population = toolbox.select(
                population + offspring,
                config.population_size,
            )
            pareto.update(population)
            history_frames.append(
                _population_frame(
                    population,
                    generation,
                    config.invalid_model_penalty,
                )
            )
            if checkpoint_dir is not None:
                _write_checkpoint(
                    Path(checkpoint_dir),
                    history_frames[-1],
                    pd.concat(history_frames, ignore_index=True),
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    history = pd.concat(history_frames, ignore_index=True)
    population_frame = _population_frame(
        population,
        config.generations,
        config.invalid_model_penalty,
    )
    pareto_frame = _population_frame(
        pareto,
        config.generations,
        config.invalid_model_penalty,
    )
    valid_pareto = pareto_frame.loc[pareto_frame["valid"]].copy()
    if valid_pareto.empty:
        raise RuntimeError(
            "Optimization produced no valid spiking model; inspect the checkpoint "
            "population or broaden the initialization."
        )
    source = valid_pareto
    representative_score = (
        source.loc[:, list(OBJECTIVE_NAMES[:-1])].sum(axis=1)
        + config.representative_prior_weight * source["conductance_prior"]
    )
    representative = source.loc[[representative_score.idxmin()]].copy()
    representative.insert(
        representative.columns.get_loc("objective_sum") + 1,
        "representative_score",
        float(representative_score.min()),
    )

    vector = representative.loc[
        representative.index[0],
        list(CELL_PARAMETER_NAMES),
    ].to_numpy(dtype=float)
    validation_evaluation = evaluate_cell_candidate(
        vector,
        target,
        screen_config,
        config,
        include_validation=True,
        include_traces=False,
    )
    validation_rows = []
    if validation_evaluation.protocol_features is not None:
        for protocol in target.validation_protocols:
            if protocol.name not in validation_evaluation.protocol_features:
                continue
            model_features = validation_evaluation.protocol_features[protocol.name]
            for group_name, feature_names in FEATURE_GROUPS.items():
                validation_rows.append(
                    {
                        "protocol": protocol.name,
                        "objective_group": group_name,
                        "score": _group_score(
                            model_features,
                            protocol.features,
                            feature_names,
                            config.missing_feature_penalty,
                        ),
                    }
                )
    validation = pd.DataFrame(validation_rows)
    feature_comparison = representative_feature_comparison(
        target,
        validation_evaluation,
        config,
    )
    metadata = {
        "version": 1,
        "algorithm": "DEAP NSGA-II",
        "objective_names": list(OBJECTIVE_NAMES),
        "parameter_names": list(CELL_PARAMETER_NAMES),
        "bounds_path": str(bounds_path),
        "screen_config": asdict(screen_config),
        "optimization_config": asdict(config),
        "effective_workers": effective_workers,
        "target": target_metadata(target),
        "n_pareto_models": len(pareto_frame),
        "n_valid_pareto_models": len(valid_pareto),
        "held_out_validation_valid": validation_evaluation.valid,
        "held_out_validation_reason": validation_evaluation.reason,
        "representative_rheobase_lower_pa": (
            validation_evaluation.rheobase_lower_pa
        ),
        "representative_rheobase_upper_pa": (
            validation_evaluation.rheobase_upper_pa
        ),
        "representative_biophysics": validation_evaluation.biophysics,
    }
    return CellOptimizationResult(
        target=target,
        history=history,
        population=population_frame,
        pareto_front=pareto_frame,
        representative=representative,
        validation=validation,
        feature_comparison=feature_comparison,
        metadata=metadata,
    )


def _experimental_sweep_map(
    target: CellOptimizationTarget,
) -> dict[int, tuple[np.ndarray, np.ndarray, Stimulus]]:
    sweeps = read_current_clamp_sweeps(target.nwb_path, long_square_only=True)
    requested = {
        target.passive.passive_sweep_number,
        *(
            protocol.sweep_number
            for protocol in (
                *target.training_protocols,
                *target.validation_protocols,
            )
        ),
    }
    result = {}
    for sweep in sweeps:
        if sweep.sweep_number not in requested:
            continue
        result[sweep.sweep_number] = (
            sweep.time_ms,
            sweep.voltage_mv,
            Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=float(sweep.stimulus_amplitude_pa),
                start_ms=float(sweep.stimulus_start_ms),
                end_ms=float(sweep.stimulus_end_ms),
            ),
        )
    return result


def save_representative_trace_plot(
    result: CellOptimizationResult,
    screen_config: BiologicalScreenConfig,
    config: CellOptimizationConfig,
    path: str | Path,
    evaluation: CandidateEvaluation | None = None,
) -> None:
    """Plot biological and representative-model traces for all four sweeps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vector = result.representative.loc[
        result.representative.index[0],
        list(CELL_PARAMETER_NAMES),
    ].to_numpy(dtype=float)
    if evaluation is None:
        evaluation = evaluate_cell_candidate(
            vector,
            result.target,
            screen_config,
            config,
            include_validation=True,
            include_traces=True,
        )
    if not evaluation.valid or evaluation.traces is None:
        raise ValueError(f"Representative model cannot be plotted: {evaluation.reason}")
    experimental = _experimental_sweep_map(result.target)
    protocol_rows = [
        (
            "passive",
            result.target.passive.passive_sweep_number,
            result.target.passive.passive_current_pa,
            result.target.passive.passive_current_pa,
            result.target.passive.passive_duration_ms,
        ),
        *[
            (
                protocol.name,
                protocol.sweep_number,
                protocol.current_pa,
                protocol.rheobase_factor * evaluation.rheobase_upper_pa,
                protocol.duration_ms,
            )
            for protocol in (
                *result.target.training_protocols,
                *result.target.validation_protocols,
            )
        ],
    ]
    figure, axes = plt.subplots(
        len(protocol_rows),
        1,
        figsize=(10.0, 9.0),
        sharex=False,
        constrained_layout=True,
    )
    for axis, (
        name,
        sweep_number,
        biological_current_pa,
        model_current_pa,
        duration_ms,
    ) in zip(
        axes,
        protocol_rows,
    ):
        bio_time, bio_voltage, bio_stimulus = experimental[sweep_number]
        bio_mask = (
            (bio_time >= bio_stimulus.start_ms - 20.0)
            & (bio_time <= bio_stimulus.end_ms)
        )
        model_trace = evaluation.traces[name]
        model_time = model_trace.time_ms - screen_config.baseline_ms
        model_mask = (
            (model_time >= -20.0)
            & (model_time <= duration_ms)
        )
        axis.plot(
            bio_time[bio_mask] - bio_stimulus.start_ms,
            bio_voltage[bio_mask],
            color="#127475",
            linewidth=1.1,
            label="biological",
        )
        axis.plot(
            model_time[model_mask],
            model_trace.voltage_mv[model_mask],
            color="#d1495b",
            linewidth=1.0,
            alpha=0.9,
            label="HH fit",
        )
        role = "held out" if name == "held_out" else "fit"
        axis.set_title(
            f"{name.replace('_', ' ').title()} ({role}): "
            f"bio {biological_current_pa:g} pA, model {model_current_pa:.1f} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"{result.target.dataset} / {result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_pareto_objective_plot(
    result: CellOptimizationResult,
    path: str | Path,
) -> None:
    """Plot the main objective trade-offs in the retained Pareto front."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto = result.pareto_front.loc[result.pareto_front["valid"]].copy()
    if pareto.empty:
        raise ValueError("No valid Pareto models are available to plot")
    representative = result.representative.iloc[0]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.2),
        constrained_layout=True,
    )
    plots = (
        (
            "excitability",
            "firing_pattern",
            "spike_shape",
            "Rheobase versus firing pattern",
        ),
        (
            "spike_dynamics",
            "phase_geometry",
            "passive",
            "Spike dynamics versus phase geometry",
        ),
    )
    for axis, (x_name, y_name, color_name, title) in zip(axes, plots):
        points = axis.scatter(
            pareto[x_name],
            pareto[y_name],
            c=pareto[color_name],
            cmap="viridis",
            s=36,
            alpha=0.8,
            edgecolors="none",
        )
        axis.scatter(
            representative[x_name],
            representative[y_name],
            marker="*",
            s=180,
            color="#d1495b",
            edgecolors="white",
            linewidths=0.8,
            label="balanced representative",
            zorder=3,
        )
        axis.set_xlabel(x_name.replace("_", " ").title() + " score")
        axis.set_ylabel(y_name.replace("_", " ").title() + " score")
        axis.set_title(title, loc="left", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        colorbar = figure.colorbar(points, ax=axis, pad=0.02)
        colorbar.set_label(color_name.replace("_", " ").title() + " score")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        f"Pareto objectives: {result.target.dataset} / {result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_cell_optimization(
    result: CellOptimizationResult,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(output / "optimization_history.csv", index=False)
    result.population.to_csv(output / "final_population.csv", index=False)
    result.pareto_front.to_csv(output / "pareto_front.csv", index=False)
    result.representative.to_csv(output / "representative_model.csv", index=False)
    result.validation.to_csv(output / "held_out_validation.csv", index=False)
    result.feature_comparison.to_csv(
        output / "representative_feature_comparison.csv",
        index=False,
    )
    target_protocol_table(result.target).to_csv(
        output / "biological_target_protocols.csv",
        index=False,
    )
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )
