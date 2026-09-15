"""Nested Pareto comparison of sodium-activation mechanisms."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .activation_kinetics import (
    FLEXIBLE_M_PARAMETER_NAMES,
    FlexibleActivationKinetics,
    ShiftedActivationKinetics,
    flexible_m_initial,
    flexible_m_parameter_bounds,
)
from .cell_optimization import (
    CELL_PARAMETER_NAMES,
    load_cell_parameter_bounds,
)
from .cell_targets import (
    CellOptimizationTarget,
    admittance_anchored_biophysics,
    target_metadata,
)
from .hh_model import SimulationError, Stimulus, Trace
from .kinetics import KineticParameters, PARAMETER_NAMES
from .model_ladder import (
    AIS_PARAMETER_NAMES,
    _find_rheobase,
    _initial_state,
    decode_ladder_model,
    extension_parameter_bounds,
    simulate_ladder,
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


NESTED_PHASE_VARIANTS = (
    "shared-alpha-beta",
    "ais-m-shift",
    "flexible-shared-m",
)
AIS_M_SHIFT_PARAMETER_NAME = "param__extension__ais_m_shift_mv"
PARETO_OBJECTIVE_NAMES = (
    "objective_concavity",
    "objective_velocity_extent",
    "objective_remaining_fit",
)
_NON_M_PARAMETER_NAMES = CELL_PARAMETER_NAMES[6:]


@dataclass(frozen=True)
class PhaseParetoConfig:
    population_size: int = 12
    generations: int = 4
    workers: int = 1
    seed: int = 42
    initialization_fraction: float = 0.08
    crossover_probability: float = 0.90
    mutation_probability: float = 0.40
    mutation_eta: float = 18.0
    phase_step_duration_ms: float = 300.0
    rheobase_weight: float = 0.20
    parameter_prior_weight: float = 0.15
    kinetic_pathology_weight: float = 0.50
    minimum_resolved_tau_ms: float = 0.003
    minimum_tau_dt_fraction: float = 0.08
    invalid_score: float = 10_000.0
    gna_gk_factor: float = 1.5


@dataclass(frozen=True)
class NestedPhaseEvaluation:
    valid: bool
    reason: str
    objectives: tuple[float, float, float]
    scalar_score: float
    phase_scores: PhaseShapeScores | None = None
    protocol_scores: Mapping[str, PhaseShapeScores] | None = None
    cycles: Mapping[str, PhaseCycle] | None = None
    traces: Mapping[str, Trace] | None = None
    rheobase_lower_pa: float = float("nan")
    rheobase_upper_pa: float = float("nan")
    rheobase_loss: float = float("nan")
    parameter_prior: float = float("nan")
    soma_diagnostics: KineticDiagnostics | None = None
    ais_diagnostics: KineticDiagnostics | None = None
    biophysics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class NestedVariantResult:
    variant: str
    parameter_names: tuple[str, ...]
    history: pd.DataFrame
    pareto_front: pd.DataFrame
    representative: pd.DataFrame
    evaluation: NestedPhaseEvaluation
    soma_kinetics: object
    ais_kinetics: object


@dataclass(frozen=True)
class NestedPhaseParetoResult:
    target: CellOptimizationTarget
    biological_cycles: Mapping[str, PhaseCycle]
    variants: tuple[NestedVariantResult, ...]
    config: PhaseParetoConfig
    shape_config: PhaseShapeConfig


def _baseline_vector(
    baseline: pd.DataFrame,
    names: Sequence[str],
) -> np.ndarray:
    if baseline.empty:
        raise ValueError("Baseline model table is empty")
    missing = set(names).difference(baseline.columns)
    if missing:
        raise ValueError(f"Baseline model is missing parameters: {sorted(missing)}")
    return baseline.iloc[0].loc[list(names)].to_numpy(dtype=float)


def nested_parameter_spec(
    variant: str,
    baseline: pd.DataFrame,
    bounds_path: str | Path,
    gna_gk_factor: float,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    if variant not in NESTED_PHASE_VARIANTS:
        raise ValueError(f"Unknown nested phase variant: {variant}")
    base_lower, base_upper = load_cell_parameter_bounds(
        bounds_path,
        gna_gk_factor,
    )
    ais_lower, ais_upper, _ = extension_parameter_bounds("soma-ais")
    shared_names = CELL_PARAMETER_NAMES + AIS_PARAMETER_NAMES
    shared_lower = np.concatenate((base_lower, ais_lower))
    shared_upper = np.concatenate((base_upper, ais_upper))
    shared_center = _baseline_vector(baseline, shared_names)
    shared_center = np.clip(
        shared_center,
        shared_lower + 1e-9,
        shared_upper - 1e-9,
    )
    if variant == "shared-alpha-beta":
        return shared_names, shared_lower, shared_upper, shared_center
    if variant == "ais-m-shift":
        return (
            shared_names + (AIS_M_SHIFT_PARAMETER_NAME,),
            np.concatenate((shared_lower, (-15.0,))),
            np.concatenate((shared_upper, (15.0,))),
            np.concatenate((shared_center, (0.0,))),
        )

    base_kinetics = KineticParameters.from_vector(
        shared_center[: len(PARAMETER_NAMES)]
    )
    flex_lower, flex_upper = flexible_m_parameter_bounds()
    names = (
        _NON_M_PARAMETER_NAMES
        + AIS_PARAMETER_NAMES
        + FLEXIBLE_M_PARAMETER_NAMES
    )
    lower = np.concatenate(
        (
            base_lower[6:],
            ais_lower,
            flex_lower,
        )
    )
    upper = np.concatenate(
        (
            base_upper[6:],
            ais_upper,
            flex_upper,
        )
    )
    center = np.concatenate(
        (
            shared_center[6 : len(CELL_PARAMETER_NAMES)],
            shared_center[len(CELL_PARAMETER_NAMES) :],
            flexible_m_initial(base_kinetics),
        )
    )
    return names, lower, upper, np.clip(center, lower + 1e-9, upper - 1e-9)


def _ais_leak_reversal(
    model,
    target: CellOptimizationTarget,
) -> float:
    voltage = target.passive.resting_voltage_mv
    kinetics = model.ais_kinetics or model.kinetics
    m_gate, h_gate, n_gate = kinetics.steady_state(voltage)
    sodium = (
        model.biophysics.gna_ms_cm2
        * model.ais_gna_multiplier
        * m_gate**3
        * h_gate
        * (voltage - model.biophysics.ena_mv)
    )
    potassium = (
        model.biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage - model.biophysics.ek_mv)
    )
    return float(
        voltage + (sodium + potassium) / model.biophysics.gleak_ms_cm2
    )


def decode_nested_model(
    variant: str,
    parameters: Sequence[float],
    target: CellOptimizationTarget,
    gna_gk_factor: float,
):
    values = np.asarray(parameters, dtype=float)
    if variant in ("shared-alpha-beta", "ais-m-shift"):
        shared_count = len(CELL_PARAMETER_NAMES) + len(AIS_PARAMETER_NAMES)
        expected = shared_count + (1 if variant == "ais-m-shift" else 0)
        if values.shape != (expected,):
            raise ValueError(f"Invalid {variant} vector: {values.shape}")
        model = decode_ladder_model(
            "soma-ais",
            values[: len(CELL_PARAMETER_NAMES)],
            values[
                len(CELL_PARAMETER_NAMES) : shared_count
            ],
            target,
            gna_gk_factor,
        )
        if variant == "ais-m-shift":
            model = replace(
                model,
                ais_kinetics=ShiftedActivationKinetics(
                    model.kinetics,
                    float(values[-1]),
                ),
            )
            model = replace(
                model,
                ais_eleak_mv=_ais_leak_reversal(model, target),
            )
        return model

    if variant != "flexible-shared-m":
        raise ValueError(f"Unknown nested phase variant: {variant}")
    non_m_count = len(_NON_M_PARAMETER_NAMES)
    ais_count = len(AIS_PARAMETER_NAMES)
    expected = non_m_count + ais_count + len(FLEXIBLE_M_PARAMETER_NAMES)
    if values.shape != (expected,):
        raise ValueError(f"Invalid flexible vector: {values.shape}")
    non_m = values[:non_m_count]
    extension = values[non_m_count : non_m_count + ais_count]
    flexible_values = values[non_m_count + ais_count :]
    full_base = np.concatenate((np.zeros(6), non_m))
    base_kinetics = KineticParameters.from_vector(
        full_base[: len(PARAMETER_NAMES)]
    )
    flexible = FlexibleActivationKinetics.from_parameters(
        base_kinetics,
        flexible_values,
    )
    model = decode_ladder_model(
        "soma-ais",
        full_base,
        extension,
        target,
        gna_gk_factor,
    )
    biophysics = admittance_anchored_biophysics(
        target.passive,
        flexible,
        log_gna_scale=float(full_base[-2]),
        log_gk_scale=float(full_base[-1]),
    )
    model = replace(
        model,
        kinetics=flexible,
        ais_kinetics=flexible,
        biophysics=biophysics,
    )
    return replace(
        model,
        ais_eleak_mv=_ais_leak_reversal(model, target),
    )


def _diagnostic_mapping(
    soma: KineticDiagnostics,
    ais: KineticDiagnostics,
) -> dict[str, float]:
    values = {}
    for prefix, diagnostics in (("soma", soma), ("ais", ais)):
        values.update(
            {
                f"{prefix}_kinetics__{key}": value
                for key, value in diagnostics.to_mapping().items()
            }
        )
    return values


def evaluate_nested_candidate(
    parameters: Sequence[float],
    variant: str,
    center: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseParetoConfig,
    shape_config: PhaseShapeConfig,
    include_validation: bool = False,
    include_traces: bool = False,
) -> NestedPhaseEvaluation:
    invalid = (
        experiment_config.invalid_score,
        experiment_config.invalid_score,
        experiment_config.invalid_score,
    )
    try:
        values = np.asarray(parameters, dtype=float)
        center_array = np.asarray(center, dtype=float)
        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        if values.shape != center_array.shape:
            raise ValueError("Candidate and center parameter shapes differ")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("Candidate is outside nested-fit bounds")
        model = decode_nested_model(
            variant,
            values,
            target,
            experiment_config.gna_gk_factor,
        )
        if (
            not -120.0 <= model.biophysics.eleak_mv <= -20.0
            or not -120.0 <= model.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        factors = model.biophysics.temperature_factors(
            screen_config.temperature_c
        )
        soma_diagnostics = kinetic_diagnostics(model.kinetics, factors)
        ais_diagnostics = kinetic_diagnostics(
            model.ais_kinetics or model.kinetics,
            factors,
        )
        minimum_tau = min(
            soma_diagnostics.minimum_tau_ms,
            ais_diagnostics.minimum_tau_ms,
        )
        resolved_tau_floor = max(
            experiment_config.minimum_resolved_tau_ms,
            experiment_config.minimum_tau_dt_fraction
            * screen_config.dt_ms,
        )
        if minimum_tau < resolved_tau_floor:
            raise SimulationError(
                "Gate time constant is below the numerical resolution floor"
            )
        if min(
            soma_diagnostics.monotonic_fraction,
            ais_diagnostics.monotonic_fraction,
        ) < 0.98:
            raise SimulationError("Gate steady state is non-monotone")
        if max(
            soma_diagnostics.maximum_rate_per_ms,
            ais_diagnostics.maximum_rate_per_ms,
        ) > 2_000.0:
            raise SimulationError("Gate rate exceeds the kinetic ceiling")
        state = _initial_state(model, target)
        if not coupled_resting_state_is_stable(
            model,
            state,
            screen_config,
        ):
            raise SimulationError("Anchored soma-AIS resting state is unstable")
        lower_rheobase, upper_rheobase, _ = _find_rheobase(
            model,
            state,
            screen_config,
        )
        protocols = list(target.training_protocols)
        if include_validation:
            protocols.extend(target.validation_protocols)
        traces = {}
        cycles = {}
        protocol_scores = {}
        for protocol in protocols:
            duration = min(
                protocol.duration_ms,
                experiment_config.phase_step_duration_ms,
            )
            simulation = step_simulation_config(
                protocol.rheobase_factor * upper_rheobase,
                duration,
                screen_config,
                model.biophysics,
                target.passive.resting_voltage_mv,
            )
            trace = simulate_ladder(model, simulation, state)
            traces[protocol.name] = trace
            stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=protocol.rheobase_factor * upper_rheobase,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + duration,
            )
            cycle = extract_phase_cycle(
                trace.time_ms,
                trace.voltage_mv,
                stimulus,
                shape_config,
            )
            cycles[protocol.name] = cycle
            protocol_scores[protocol.name] = compare_phase_cycles(
                cycle,
                biological_cycles_target[protocol.name],
                shape_config,
            )
        training = {
            protocol.name: protocol_scores[protocol.name]
            for protocol in target.training_protocols
        }
        phase_scores = mean_phase_scores(training)
        rheobase_loss = abs(
            upper_rheobase - target.sampled_rheobase_pa
        ) / max(10.0, 0.15 * target.sampled_rheobase_pa)
        half_width = np.maximum((upper - lower) / 2.0, 1e-12)
        parameter_prior = float(
            np.mean(((values - center_array) / half_width) ** 2)
        )
        pathology = max(
            soma_diagnostics.pathology_score,
            ais_diagnostics.pathology_score,
        )
        regularity = (
            experiment_config.rheobase_weight * rheobase_loss
            + experiment_config.parameter_prior_weight * parameter_prior
            + experiment_config.kinetic_pathology_weight * pathology
        )
        remaining = (
            1.5 * phase_scores.physical_curve
            + 3.0 * phase_scores.normalized_shape
            + phase_scores.slope_shape
            + 0.75 * phase_scores.landmarks
            + regularity
        )
        objectives = (
            float(phase_scores.concavity),
            float(phase_scores.velocity_extent),
            float(remaining),
        )
        scalar = float(phase_scores.total + regularity)
        if not np.all(np.isfinite((*objectives, scalar))):
            raise SimulationError("Nested candidate produced non-finite scores")
        biophysics = model.biophysics.to_physical_mapping()
        biophysics.update(
            {
                "extension__ais_area_fraction": model.ais_area_fraction,
                "extension__ais_gna_multiplier": model.ais_gna_multiplier,
                "extension__coupling_ns": model.coupling_ns,
                "extension__ais_eleak_mv": model.ais_eleak_mv,
            }
        )
        return NestedPhaseEvaluation(
            valid=True,
            reason="accepted",
            objectives=objectives,
            scalar_score=scalar,
            phase_scores=phase_scores,
            protocol_scores=protocol_scores,
            cycles=cycles,
            traces=traces if include_traces else None,
            rheobase_lower_pa=float(lower_rheobase),
            rheobase_upper_pa=float(upper_rheobase),
            rheobase_loss=float(rheobase_loss),
            parameter_prior=parameter_prior,
            soma_diagnostics=soma_diagnostics,
            ais_diagnostics=ais_diagnostics,
            biophysics=biophysics,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return NestedPhaseEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            objectives=invalid,
            scalar_score=experiment_config.invalid_score,
        )


def _evaluation_payload(
    parameters: Sequence[float],
    variant: str,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseParetoConfig,
    shape_config: PhaseShapeConfig,
) -> NestedPhaseEvaluation:
    return evaluate_nested_candidate(
        parameters,
        variant,
        center,
        lower,
        upper,
        target,
        biological_cycles_target,
        screen_config,
        experiment_config,
        shape_config,
    )


def _evaluation_row(
    generation: int,
    individual: int,
    parameter_names: Sequence[str],
    parameters: Sequence[float],
    evaluation: NestedPhaseEvaluation,
) -> dict[str, object]:
    row = {
        "generation": generation,
        "individual": individual,
        **dict(zip(parameter_names, map(float, parameters))),
        **dict(zip(PARETO_OBJECTIVE_NAMES, evaluation.objectives)),
        "scalar_score": evaluation.scalar_score,
        "valid": evaluation.valid,
        "reason": evaluation.reason,
    }
    if evaluation.phase_scores is not None:
        row.update(
            {
                f"phase__{key}": value
                for key, value in evaluation.phase_scores.to_mapping().items()
            }
        )
        row.update(
            {
                "rheobase_loss": evaluation.rheobase_loss,
                "model_rheobase_pa": evaluation.rheobase_upper_pa,
                "parameter_prior": evaluation.parameter_prior,
            }
        )
    if (
        evaluation.soma_diagnostics is not None
        and evaluation.ais_diagnostics is not None
    ):
        row.update(
            _diagnostic_mapping(
                evaluation.soma_diagnostics,
                evaluation.ais_diagnostics,
            )
        )
    return row


def _individual_key(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(values, dtype=float), 12))


def optimize_nested_variant(
    variant: str,
    baseline: pd.DataFrame,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseParetoConfig,
    shape_config: PhaseShapeConfig,
) -> NestedVariantResult:
    """Optimize one nested mechanism with three-objective NSGA-II."""
    try:
        from deap import base, creator, tools
    except ImportError as error:
        raise ImportError("Nested phase optimization requires DEAP") from error
    if (
        experiment_config.population_size < 4
        or experiment_config.population_size % 4
    ):
        raise ValueError("population_size must be at least four and divisible by four")

    names, lower, upper, center = nested_parameter_spec(
        variant,
        baseline,
        bounds_path,
        experiment_config.gna_gk_factor,
    )
    variant_index = NESTED_PHASE_VARIANTS.index(variant)
    seed = experiment_config.seed + 10_000 * variant_index
    random.seed(seed)
    rng = np.random.default_rng(seed)

    fitness_name = "FitnessNestedPhasePareto"
    individual_name = "IndividualNestedPhasePareto"
    if not hasattr(creator, fitness_name):
        creator.create(
            fitness_name,
            base.Fitness,
            weights=(-1.0,) * len(PARETO_OBJECTIVE_NAMES),
        )
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)
    population = [individual_type(center.tolist())]
    width = upper - lower
    while len(population) < experiment_config.population_size:
        fraction = experiment_config.initialization_fraction
        if len(population) > experiment_config.population_size // 2:
            fraction *= 2.0
        candidate = np.clip(
            center + rng.normal(0.0, fraction, len(center)) * width,
            lower + 1e-9,
            upper - 1e-9,
        )
        population.append(individual_type(candidate.tolist()))

    toolbox = base.Toolbox()
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
        eta=experiment_config.mutation_eta,
        indpb=1.0 / len(names),
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("clone", copy.deepcopy)

    executor = None
    mapper = map
    if experiment_config.workers > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=experiment_config.workers
            )
            mapper = executor.map
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"Parallel evaluation is unavailable ({error}); using one worker.",
                RuntimeWarning,
            )

    rows = []
    evaluation_lookup: dict[
        tuple[float, ...],
        NestedPhaseEvaluation,
    ] = {}

    def evaluate_population(
        individuals,
        generation: int,
    ) -> None:
        evaluations = list(
            mapper(
                _evaluation_payload,
                individuals,
                [variant] * len(individuals),
                [center] * len(individuals),
                [lower] * len(individuals),
                [upper] * len(individuals),
                [target] * len(individuals),
                [biological_cycles_target] * len(individuals),
                [screen_config] * len(individuals),
                [experiment_config] * len(individuals),
                [shape_config] * len(individuals),
            )
        )
        for index, (individual, evaluation) in enumerate(
            zip(individuals, evaluations)
        ):
            individual.fitness.values = evaluation.objectives
            evaluation_lookup[_individual_key(individual)] = evaluation
            rows.append(
                _evaluation_row(
                    generation,
                    index,
                    names,
                    individual,
                    evaluation,
                )
            )

    pareto = tools.ParetoFront(
        similar=lambda left, right: np.allclose(
            left,
            right,
            rtol=0.0,
            atol=1e-10,
        )
    )
    try:
        evaluate_population(population, 0)
        population = toolbox.select(population, len(population))
        pareto.update(population)
        for generation in range(1, experiment_config.generations + 1):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for left, right in zip(offspring[::2], offspring[1::2]):
                if random.random() <= experiment_config.crossover_probability:
                    toolbox.mate(left, right)
                    del left.fitness.values
                    del right.fitness.values
            for individual in offspring:
                if random.random() <= experiment_config.mutation_probability:
                    toolbox.mutate(individual)
                    if individual.fitness.valid:
                        del individual.fitness.values
            invalid = [
                individual
                for individual in offspring
                if not individual.fitness.valid
            ]
            evaluate_population(invalid, generation)
            population = toolbox.select(
                population + offspring,
                experiment_config.population_size,
            )
            pareto.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    history = pd.DataFrame(rows)
    pareto_rows = []
    for index, individual in enumerate(pareto):
        evaluation = evaluation_lookup[_individual_key(individual)]
        pareto_rows.append(
            _evaluation_row(
                experiment_config.generations,
                index,
                names,
                individual,
                evaluation,
            )
        )
    pareto_frame = pd.DataFrame(pareto_rows)
    valid_pareto = pareto_frame.loc[pareto_frame["valid"]].copy()
    if valid_pareto.empty:
        raise RuntimeError(f"{variant} produced no valid Pareto model")
    representative = valid_pareto.nsmallest(1, "scalar_score").copy()
    parameters = representative.iloc[0].loc[list(names)].to_numpy(dtype=float)
    evaluation = evaluate_nested_candidate(
        parameters,
        variant,
        center,
        lower,
        upper,
        target,
        biological_cycles_target,
        screen_config,
        experiment_config,
        shape_config,
        include_validation=True,
        include_traces=True,
    )
    if not evaluation.valid:
        raise RuntimeError(
            f"Best {variant} model failed reevaluation: {evaluation.reason}"
        )
    model = decode_nested_model(
        variant,
        parameters,
        target,
        experiment_config.gna_gk_factor,
    )
    return NestedVariantResult(
        variant=variant,
        parameter_names=names,
        history=history,
        pareto_front=pareto_frame,
        representative=representative,
        evaluation=evaluation,
        soma_kinetics=model.kinetics,
        ais_kinetics=model.ais_kinetics or model.kinetics,
    )


def run_nested_phase_pareto_experiment(
    target: CellOptimizationTarget,
    baseline: pd.DataFrame,
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseParetoConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
) -> NestedPhaseParetoResult:
    experiment_config = experiment_config or PhaseParetoConfig()
    shape_config = shape_config or PhaseShapeConfig()
    biological = biological_phase_cycles(
        target,
        shape_config,
        include_validation=True,
    )
    variants = tuple(
        optimize_nested_variant(
            variant,
            baseline,
            target,
            biological,
            bounds_path,
            screen_config,
            experiment_config,
            shape_config,
        )
        for variant in NESTED_PHASE_VARIANTS
    )
    return NestedPhaseParetoResult(
        target=target,
        biological_cycles=biological,
        variants=variants,
        config=experiment_config,
        shape_config=shape_config,
    )


def nested_phase_summary(
    result: NestedPhaseParetoResult,
) -> pd.DataFrame:
    rows = []
    for variant_result in result.variants:
        evaluation = variant_result.evaluation
        valid_pareto = variant_result.pareto_front.loc[
            variant_result.pareto_front["valid"]
        ]
        jointly_close = (
            valid_pareto["phase__concavity"].lt(0.25)
            & valid_pareto["phase__velocity_extent"].lt(1.0)
        )
        row = {
            "variant": variant_result.variant,
            "parameter_count": len(variant_result.parameter_names),
            "dataset": result.target.dataset,
            "cell_id": result.target.cell_id,
            "pareto_models": len(valid_pareto),
            "joint_target_models": int(jointly_close.sum()),
            "minimum_pareto_concavity": float(
                valid_pareto["phase__concavity"].min()
            ),
            "minimum_pareto_velocity_extent": float(
                valid_pareto["phase__velocity_extent"].min()
            ),
            "scalar_score": evaluation.scalar_score,
            "model_rheobase_pa": evaluation.rheobase_upper_pa,
            "biological_rheobase_pa": result.target.sampled_rheobase_pa,
            "rheobase_loss": evaluation.rheobase_loss,
            "parameter_prior": evaluation.parameter_prior,
            **evaluation.phase_scores.to_mapping(),
            **_diagnostic_mapping(
                evaluation.soma_diagnostics,
                evaluation.ais_diagnostics,
            ),
        }
        row["selected__ais_m_shift_mv"] = float(
            variant_result.representative.iloc[0].get(
                AIS_M_SHIFT_PARAMETER_NAME,
                0.0,
            )
        )
        if evaluation.biophysics is not None:
            row.update(evaluation.biophysics)
        for protocol_name, scores in evaluation.protocol_scores.items():
            row.update(
                {
                    f"{protocol_name}__{key}": value
                    for key, value in scores.to_mapping().items()
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _phase_xy(cycle: PhaseCycle) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.concatenate(
        (
            cycle.grid * cycle.amplitude_mv,
            cycle.amplitude_mv
            - cycle.grid[1:] * cycle.repolarization_amplitude_mv,
        )
    )
    dvdt = np.concatenate(
        (cycle.up_dvdt_mv_ms, cycle.down_dvdt_mv_ms[1:])
    )
    return voltage, dvdt


def _normalized_phase_xy(
    cycle: PhaseCycle,
) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.concatenate(
        (
            cycle.grid,
            1.0
            - cycle.grid[1:]
            * cycle.repolarization_amplitude_mv
            / cycle.amplitude_mv,
        )
    )
    dvdt = np.concatenate(
        (cycle.up_normalized, cycle.down_normalized[1:])
    )
    return voltage, dvdt


def save_nested_phase_plot(
    result: NestedPhaseParetoResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "shared-alpha-beta": "#e76f51",
        "ais-m-shift": "#7b2cbf",
        "flexible-shared-m": "#1976a3",
    }
    labels = {
        "shared-alpha-beta": "Shared alpha/beta",
        "ais-m-shift": "AIS m shift",
        "flexible-shared-m": "Flexible shared m",
    }
    protocols = (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    )
    figure, axes = plt.subplots(
        len(protocols),
        3,
        figsize=(13.5, 10.5),
        constrained_layout=True,
    )
    for row_index, protocol in enumerate(protocols):
        biological = result.biological_cycles[protocol.name]
        series = [
            ("Biological", biological, "#147d7e"),
            *(
                (
                    labels[variant.variant],
                    variant.evaluation.cycles[protocol.name],
                    colors[variant.variant],
                )
                for variant in result.variants
            ),
        ]
        for label, cycle, color in series:
            voltage, dvdt = _phase_xy(cycle)
            axes[row_index, 0].plot(
                voltage,
                dvdt,
                color=color,
                linewidth=1.25,
                label=label,
            )
            norm_voltage, norm_dvdt = _normalized_phase_xy(cycle)
            axes[row_index, 1].plot(
                norm_voltage,
                norm_dvdt,
                color=color,
                linewidth=1.25,
            )
            axes[row_index, 2].plot(
                cycle.grid,
                cycle.up_concavity,
                color=color,
                linewidth=1.25,
            )
        axes[row_index, 2].axvspan(
            0.0,
            biological.up_max_relative_voltage,
            color="#eeeeee",
            zorder=-3,
        )
        axes[row_index, 0].set_title(
            f"{protocol.name.title()}: threshold aligned",
            loc="left",
            fontsize=10,
        )
        axes[row_index, 1].set_title(
            "Normalized phase shape",
            loc="left",
            fontsize=10,
        )
        axes[row_index, 2].set_title(
            "Smoothed upstroke concavity",
            loc="left",
            fontsize=10,
        )
        axes[row_index, 0].set_ylabel("dV/dt (mV/ms)")
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.55)
            axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("V - threshold (mV)")
    axes[-1, 1].set_xlabel("Normalized AP voltage")
    axes[-1, 2].set_xlabel("Normalized upstroke voltage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Nested activation mechanisms: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=14,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_nested_pareto_plot(
    result: NestedPhaseParetoResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "shared-alpha-beta": ("#e76f51", "Shared alpha/beta"),
        "ais-m-shift": ("#7b2cbf", "AIS m shift"),
        "flexible-shared-m": ("#1976a3", "Flexible shared m"),
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
    )
    for variant_result in result.variants:
        color, label = styles[variant_result.variant]
        front = variant_result.pareto_front.loc[
            variant_result.pareto_front["valid"]
        ]
        axes[0].scatter(
            front["phase__velocity_extent"],
            np.maximum(front["phase__concavity"], 1e-5),
            color=color,
            edgecolors="white",
            linewidths=0.6,
            s=58,
            alpha=0.88,
            label=label,
        )
        evaluation = variant_result.evaluation
        axes[0].scatter(
            evaluation.phase_scores.velocity_extent,
            max(evaluation.phase_scores.concavity, 1e-5),
            marker="*",
            s=170,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        axes[1].scatter(
            front["objective_remaining_fit"],
            front["phase__normalized_shape"],
            color=color,
            s=48,
            alpha=0.75,
            label=label,
        )
        axes[1].scatter(
            evaluation.objectives[2],
            evaluation.phase_scores.normalized_shape,
            marker="*",
            s=170,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
    axes[0].axvline(1.0, color="#8d99ae", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.25, color="#8d99ae", linestyle=":", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Velocity-extent loss")
    axes[0].set_ylabel("Smoothed concavity loss")
    axes[0].set_title("Non-dominated concavity/velocity fronts", loc="left")
    axes[1].set_xlabel("Remaining fit + regularity objective")
    axes[1].set_ylabel("Normalized branch-shape loss")
    axes[1].set_title("Cost outside the two focal objectives", loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_nested_score_plot(
    result: NestedPhaseParetoResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "normalized_shape",
        "concavity",
        "velocity_extent",
        "landmarks",
        "total",
    )
    labels = (
        "Normalized\nshape",
        "Concavity",
        "Velocity\nextent",
        "Landmarks",
        "Phase total",
    )
    colors = ("#e76f51", "#7b2cbf", "#1976a3")
    names = ("Shared alpha/beta", "AIS m shift", "Flexible shared m")
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(
        figsize=(10.0, 4.8),
        constrained_layout=True,
    )
    for index, (variant, label, color) in enumerate(
        zip(result.variants, names, colors)
    ):
        values = variant.evaluation.phase_scores.to_mapping()
        axis.bar(
            x + (index - 1.0) * width,
            [values[metric] for metric in metrics],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized loss (lower is better)")
    axis.set_title("Representative Pareto models", loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_nested_activation_plot(
    result: NestedPhaseParetoResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voltage = np.linspace(-100.0, 60.0, 321)
    colors = {
        "shared-alpha-beta": "#e76f51",
        "ais-m-shift": "#7b2cbf",
        "flexible-shared-m": "#1976a3",
    }
    labels = {
        "shared-alpha-beta": "Shared alpha/beta",
        "ais-m-shift": "AIS m shift",
        "flexible-shared-m": "Flexible shared m",
    }
    exponent = (result.target.temperature_c - 6.3) / 10.0
    q10_m = float(
        result.variants[0].evaluation.biophysics["physical__q10_m"]
    )
    temperature_factor = q10_m**exponent
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        constrained_layout=True,
    )
    canonical = KineticParameters.canonical()
    canonical_rates = canonical.rates(voltage)
    canonical_total = (
        canonical_rates["alpha_m"] + canonical_rates["beta_m"]
    )
    axes[0].plot(
        voltage,
        canonical_rates["alpha_m"] / canonical_total,
        color="#777777",
        label="Canonical",
    )
    axes[1].plot(
        voltage,
        1.0 / (temperature_factor * canonical_total),
        color="#777777",
        label="Canonical",
    )
    for variant in result.variants:
        color = colors[variant.variant]
        soma_rates = variant.soma_kinetics.rates(voltage)
        soma_total = soma_rates["alpha_m"] + soma_rates["beta_m"]
        axes[0].plot(
            voltage,
            soma_rates["alpha_m"] / soma_total,
            color=color,
            label=labels[variant.variant],
        )
        axes[1].plot(
            voltage,
            1.0 / (temperature_factor * soma_total),
            color=color,
        )
        ais_rates = variant.ais_kinetics.rates(voltage)
        ais_total = ais_rates["alpha_m"] + ais_rates["beta_m"]
        if not (
            np.allclose(ais_rates["alpha_m"], soma_rates["alpha_m"])
            and np.allclose(ais_rates["beta_m"], soma_rates["beta_m"])
        ):
            axes[0].plot(
                voltage,
                ais_rates["alpha_m"] / ais_total,
                color=color,
                linestyle="--",
                label=f"{labels[variant.variant]} AIS",
            )
            axes[1].plot(
                voltage,
                1.0 / (temperature_factor * ais_total),
                color=color,
                linestyle="--",
            )
    axes[0].set_ylabel("m_inf")
    axes[1].set_ylabel("tau_m at recording temperature (ms)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Sodium activation steady state", loc="left")
    axes[1].set_title("Sodium activation time constant", loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_nested_phase_pareto_result(
    result: NestedPhaseParetoResult,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nested_phase_summary(result).to_csv(
        output / "nested_variant_summary.csv",
        index=False,
    )
    protocol_rows = []
    for variant_result in result.variants:
        variant_result.history.to_csv(
            output / f"{variant_result.variant}_history.csv",
            index=False,
        )
        variant_result.pareto_front.to_csv(
            output / f"{variant_result.variant}_pareto.csv",
            index=False,
        )
        variant_result.representative.to_csv(
            output / f"{variant_result.variant}_representative.csv",
            index=False,
        )
        for protocol, scores in (
            variant_result.evaluation.protocol_scores.items()
        ):
            protocol_rows.append(
                {
                    "variant": variant_result.variant,
                    "protocol": protocol,
                    **scores.to_mapping(),
                }
            )
    pd.DataFrame(protocol_rows).to_csv(
        output / "nested_protocol_scores.csv",
        index=False,
    )
    metadata = {
        "version": 1,
        "experiment": "nested sodium-activation Pareto adequacy",
        "variants": list(NESTED_PHASE_VARIANTS),
        "objectives": list(PARETO_OBJECTIVE_NAMES),
        "target": target_metadata(result.target),
        "experiment_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_nested_phase_plot(
        result,
        output / "nested_phase_comparison.png",
    )
    save_nested_pareto_plot(
        result,
        output / "nested_pareto_fronts.png",
    )
    save_nested_score_plot(
        result,
        output / "nested_score_bars.png",
    )
    save_nested_activation_plot(
        result,
        output / "nested_activation_kinetics.png",
    )
