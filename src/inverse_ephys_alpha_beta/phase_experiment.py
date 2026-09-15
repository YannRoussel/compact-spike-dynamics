"""Phase-focused model-adequacy experiment for one-compartment and soma-AIS HH."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cell_optimization import (
    CELL_PARAMETER_NAMES,
    load_cell_parameter_bounds,
)
from .cell_targets import (
    CellOptimizationTarget,
    admittance_anchored_biophysics,
    resting_state_is_stable,
    target_metadata,
)
from .hh_model import (
    SimulationConfig,
    SimulationError,
    Stimulus,
    Trace,
    simulate,
)
from .kinetics import KineticParameters, PARAMETER_NAMES, RATE_NAMES
from .model_ladder import (
    AIS_PARAMETER_NAMES,
    _bounded_parameters,
    _find_rheobase as find_ladder_rheobase,
    _initial_state as ladder_initial_state,
    _ladder_rhs,
    _unbounded_parameters,
    decode_ladder_model,
    extension_parameter_bounds,
    simulate_ladder,
)
from .phase_shape import (
    PhaseCycle,
    PhaseShapeConfig,
    PhaseShapeScores,
    compare_phase_cycles,
    extract_phase_cycle,
    mean_phase_scores,
)
from .protocols import (
    BiologicalScreenConfig,
    find_rheobase,
    step_simulation_config,
)
from .raw_patchseq import read_current_clamp_sweeps


PHASE_ARCHITECTURES = ("one-compartment", "soma-ais")
PHASE_SCORE_NAMES = (
    "physical_curve",
    "normalized_shape",
    "slope_shape",
    "concavity",
    "velocity_extent",
    "landmarks",
    "total",
    "onset_concavity_sign_agreement",
    "max_dvdt_residual_mv_ms",
    "min_dvdt_residual_mv_ms",
    "dvdt_span_residual_mv_ms",
)


@dataclass(frozen=True)
class PhaseExperimentConfig:
    population_size: int = 8
    generations: int = 3
    workers: int = 1
    seed: int = 42
    sigma: float = 0.40
    phase_step_duration_ms: float = 300.0
    rheobase_weight: float = 0.20
    parameter_prior_weight: float = 0.15
    kinetic_pathology_weight: float = 0.50
    invalid_score: float = 10_000.0
    gna_gk_factor: float = 1.5


@dataclass(frozen=True)
class KineticDiagnostics:
    monotonic_fraction: float
    minimum_tau_ms: float
    maximum_tau_ms: float
    maximum_rate_per_ms: float
    pathology_score: float

    def to_mapping(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class PhaseFitEvaluation:
    valid: bool
    reason: str
    scalar_score: float
    phase_scores: PhaseShapeScores | None = None
    protocol_scores: Mapping[str, PhaseShapeScores] | None = None
    cycles: Mapping[str, PhaseCycle] | None = None
    traces: Mapping[str, Trace] | None = None
    rheobase_lower_pa: float = float("nan")
    rheobase_upper_pa: float = float("nan")
    rheobase_loss: float = float("nan")
    parameter_prior: float = float("nan")
    bound_proximity_fraction: float = float("nan")
    maximum_parent_displacement: float = float("nan")
    kinetic_diagnostics: KineticDiagnostics | None = None
    biophysics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PhaseArchitectureResult:
    architecture: str
    parameter_names: tuple[str, ...]
    history: pd.DataFrame
    best_parameters: pd.DataFrame
    evaluation: PhaseFitEvaluation


@dataclass(frozen=True)
class PhaseAdequacyResult:
    target: CellOptimizationTarget
    biological_cycles: Mapping[str, PhaseCycle]
    one_compartment: PhaseArchitectureResult
    soma_ais: PhaseArchitectureResult
    config: PhaseExperimentConfig
    shape_config: PhaseShapeConfig


def architecture_parameter_bounds(
    architecture: str,
    bounds_path: str | Path,
    gna_gk_factor: float,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if architecture not in PHASE_ARCHITECTURES:
        raise ValueError(f"Unknown phase architecture: {architecture}")
    lower, upper = load_cell_parameter_bounds(
        bounds_path,
        gna_gk_factor,
    )
    names = CELL_PARAMETER_NAMES
    if architecture == "soma-ais":
        ais_lower, ais_upper, _ = extension_parameter_bounds("soma-ais")
        lower = np.concatenate((lower, ais_lower))
        upper = np.concatenate((upper, ais_upper))
        names = CELL_PARAMETER_NAMES + AIS_PARAMETER_NAMES
    return names, lower, upper


def _ais_phase_initial() -> np.ndarray:
    """Start from a modest AIS sodium enrichment that remains excitable."""
    lower, upper, initial = extension_parameter_bounds("soma-ais")
    initial = initial.copy()
    initial[1] = lower[1] + 0.10 * (upper[1] - lower[1])
    return initial


def biological_phase_cycles(
    target: CellOptimizationTarget,
    shape_config: PhaseShapeConfig,
    include_validation: bool = True,
) -> dict[str, PhaseCycle]:
    sweeps = {
        sweep.sweep_number: sweep
        for sweep in read_current_clamp_sweeps(
            target.nwb_path,
            long_square_only=True,
        )
    }
    protocols = list(target.training_protocols)
    if include_validation:
        protocols.extend(target.validation_protocols)
    cycles = {}
    for protocol in protocols:
        sweep = sweeps[protocol.sweep_number]
        stimulus = Stimulus(
            amplitude_ua_cm2=None,
            amplitude_pa=protocol.current_pa,
            start_ms=float(sweep.stimulus_start_ms),
            end_ms=float(sweep.stimulus_end_ms),
        )
        cycles[protocol.name] = extract_phase_cycle(
            sweep.time_ms,
            sweep.voltage_mv,
            stimulus,
            shape_config,
        )
    return cycles


def kinetic_diagnostics(
    kinetics: KineticParameters,
    temperature_factors: Sequence[float],
) -> KineticDiagnostics:
    voltage = np.linspace(-100.0, 60.0, 321)
    rates = kinetics.rates(voltage)
    factors = tuple(float(value) for value in temperature_factors)
    gate_specs = (
        ("m", "alpha_m", "beta_m", factors[0], 1.0),
        ("h", "alpha_h", "beta_h", factors[1], -1.0),
        ("n", "alpha_n", "beta_n", factors[2], 1.0),
    )
    taus = []
    monotonic = []
    maximum_rate = 0.0
    for _, alpha_name, beta_name, factor, direction in gate_specs:
        alpha = factor * rates[alpha_name]
        beta = factor * rates[beta_name]
        total = alpha + beta
        steady = alpha / total
        tau = 1.0 / total
        taus.append(tau)
        differences = direction * np.diff(steady)
        monotonic.append(float(np.mean(differences >= -1e-4)))
        maximum_rate = max(
            maximum_rate,
            float(np.max(alpha)),
            float(np.max(beta)),
        )
    all_tau = np.concatenate(taus)
    minimum_tau = float(np.min(all_tau))
    maximum_tau = float(np.max(all_tau))
    monotonic_fraction = float(np.mean(monotonic))
    pathology = (
        10.0 * max(0.0, 0.98 - monotonic_fraction)
        + max(0.0, math.log(0.005 / minimum_tau))
        + max(0.0, math.log(maximum_tau / 500.0))
        + max(0.0, math.log(maximum_rate / 2_000.0))
    )
    return KineticDiagnostics(
        monotonic_fraction=monotonic_fraction,
        minimum_tau_ms=minimum_tau,
        maximum_tau_ms=maximum_tau,
        maximum_rate_per_ms=maximum_rate,
        pathology_score=float(pathology),
    )


def _one_compartment_components(
    base_vector: np.ndarray,
    target: CellOptimizationTarget,
    config: PhaseExperimentConfig,
) -> tuple[KineticParameters, object, np.ndarray]:
    kinetics = KineticParameters.from_vector(
        base_vector[: len(PARAMETER_NAMES)]
    )
    biophysics = admittance_anchored_biophysics(
        target.passive,
        kinetics,
        log_gna_scale=float(base_vector[-2]),
        log_gk_scale=float(base_vector[-1]),
    )
    if not -120.0 <= biophysics.eleak_mv <= -20.0:
        raise SimulationError("Derived leak reversal is outside range")
    if not resting_state_is_stable(target.passive, kinetics, biophysics):
        raise SimulationError("Anchored resting state is unstable")
    state = np.asarray(
        (
            target.passive.resting_voltage_mv,
            *kinetics.steady_state(target.passive.resting_voltage_mv),
        ),
        dtype=float,
    )
    return kinetics, biophysics, state


def _simulate_one_compartment(
    base_vector: np.ndarray,
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig,
    protocols,
) -> tuple[
    float,
    float,
    dict[str, Trace],
    Mapping[str, float],
    KineticDiagnostics,
]:
    kinetics, biophysics, state = _one_compartment_components(
        base_vector,
        target,
        experiment_config,
    )
    lower, upper, _ = find_rheobase(
        kinetics,
        biophysics,
        state,
        screen_config,
    )
    traces = {}
    for protocol in protocols:
        duration = min(
            protocol.duration_ms,
            experiment_config.phase_step_duration_ms,
        )
        simulation = step_simulation_config(
            protocol.rheobase_factor * upper,
            duration,
            screen_config,
            biophysics,
            target.passive.resting_voltage_mv,
        )
        traces[protocol.name] = simulate(
            kinetics,
            simulation,
            state,
        )
    diagnostics = kinetic_diagnostics(
        kinetics,
        biophysics.temperature_factors(screen_config.temperature_c),
    )
    return (
        float(lower),
        float(upper),
        traces,
        biophysics.to_physical_mapping(),
        diagnostics,
    )


def _simulate_soma_ais(
    parameters: np.ndarray,
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig,
    protocols,
) -> tuple[
    float,
    float,
    dict[str, Trace],
    Mapping[str, float],
    KineticDiagnostics,
]:
    base_count = len(CELL_PARAMETER_NAMES)
    model = decode_ladder_model(
        "soma-ais",
        parameters[:base_count],
        parameters[base_count:],
        target,
        experiment_config.gna_gk_factor,
    )
    if (
        not -120.0 <= model.biophysics.eleak_mv <= -20.0
        or not -120.0 <= model.ais_eleak_mv <= -20.0
    ):
        raise SimulationError("Derived leak reversal is outside range")
    state = ladder_initial_state(model, target)
    if not coupled_resting_state_is_stable(
        model,
        state,
        screen_config,
    ):
        raise SimulationError("Anchored soma-AIS resting state is unstable")
    lower, upper, _ = find_ladder_rheobase(
        model,
        state,
        screen_config,
    )
    traces = {}
    for protocol in protocols:
        duration = min(
            protocol.duration_ms,
            experiment_config.phase_step_duration_ms,
        )
        simulation = step_simulation_config(
            protocol.rheobase_factor * upper,
            duration,
            screen_config,
            model.biophysics,
            target.passive.resting_voltage_mv,
        )
        traces[protocol.name] = simulate_ladder(
            model,
            simulation,
            state,
        )
    mapping = model.biophysics.to_physical_mapping()
    mapping.update(
        {
            "extension__ais_area_fraction": model.ais_area_fraction,
            "extension__ais_gna_multiplier": model.ais_gna_multiplier,
            "extension__coupling_ns": model.coupling_ns,
            "extension__ais_eleak_mv": model.ais_eleak_mv,
        }
    )
    diagnostics = kinetic_diagnostics(
        model.kinetics,
        model.biophysics.temperature_factors(
            screen_config.temperature_c
        ),
    )
    return (
        float(lower),
        float(upper),
        traces,
        mapping,
        diagnostics,
    )


def coupled_resting_state_is_stable(
    model,
    state: np.ndarray,
    screen_config: BiologicalScreenConfig,
    eigenvalue_tolerance: float = 1e-5,
) -> bool:
    """Test stability of the complete coupled soma-AIS resting state."""
    zero_stimulus = Stimulus(
        amplitude_ua_cm2=None,
        amplitude_pa=0.0,
        start_ms=0.0,
        end_ms=1.0,
    )
    simulation = SimulationConfig(
        duration_ms=1.0,
        dt_ms=screen_config.dt_ms,
        initial_voltage_mv=float(state[0]),
        stimulus=zero_stimulus,
        conductances=model.biophysics,
        temperature_c=screen_config.temperature_c,
    )
    state = np.asarray(state, dtype=float)
    jacobian = np.empty((len(state), len(state)), dtype=float)
    for column in range(len(state)):
        delta = 1e-3 if column in (0, 4) else 1e-6
        left = state.copy()
        right = state.copy()
        left[column] -= delta
        right[column] += delta
        jacobian[:, column] = (
            _ladder_rhs(0.0, right, model, simulation)
            - _ladder_rhs(0.0, left, model, simulation)
        ) / (2.0 * delta)
    eigenvalues = np.linalg.eigvals(jacobian)
    return bool(
        np.all(np.isfinite(eigenvalues))
        and float(np.max(np.real(eigenvalues))) < eigenvalue_tolerance
    )


def evaluate_phase_candidate(
    parameters: Sequence[float],
    architecture: str,
    center: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig,
    shape_config: PhaseShapeConfig,
    include_validation: bool = False,
    include_traces: bool = False,
) -> PhaseFitEvaluation:
    """Evaluate one model using branch shape as the dominant objective."""
    try:
        values = np.asarray(parameters, dtype=float)
        center_array = np.asarray(center, dtype=float)
        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        if values.shape != center_array.shape:
            raise ValueError("Candidate and center parameter shapes differ")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("Candidate is outside phase-fit bounds")
        protocols = list(target.training_protocols)
        if include_validation:
            protocols.extend(target.validation_protocols)
        if architecture == "one-compartment":
            simulation = _simulate_one_compartment(
                values,
                target,
                screen_config,
                experiment_config,
                protocols,
            )
        elif architecture == "soma-ais":
            simulation = _simulate_soma_ais(
                values,
                target,
                screen_config,
                experiment_config,
                protocols,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        (
            rheobase_lower,
            rheobase_upper,
            traces,
            biophysics,
            diagnostics,
        ) = simulation
        cycles = {}
        protocol_scores = {}
        for protocol in protocols:
            trace = traces[protocol.name]
            stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=(
                    protocol.rheobase_factor * rheobase_upper
                ),
                start_ms=screen_config.baseline_ms,
                end_ms=(
                    screen_config.baseline_ms
                    + min(
                        protocol.duration_ms,
                        experiment_config.phase_step_duration_ms,
                    )
                ),
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
        training_scores = {
            protocol.name: protocol_scores[protocol.name]
            for protocol in target.training_protocols
        }
        phase_scores = mean_phase_scores(training_scores)
        rheobase_loss = abs(
            rheobase_upper - target.sampled_rheobase_pa
        ) / max(10.0, 0.15 * target.sampled_rheobase_pa)
        half_width = np.maximum((upper - lower) / 2.0, 1e-12)
        parameter_prior = float(
            np.mean(((values - center_array) / half_width) ** 2)
        )
        unit_position = (values - lower) / np.maximum(upper - lower, 1e-12)
        bound_proximity = float(
            np.mean((unit_position <= 0.03) | (unit_position >= 0.97))
        )
        maximum_parent_displacement = float(
            np.max(np.abs((values - center_array) / half_width))
        )
        scalar = float(
            phase_scores.total
            + experiment_config.rheobase_weight * rheobase_loss
            + experiment_config.parameter_prior_weight * parameter_prior
            + experiment_config.kinetic_pathology_weight
            * diagnostics.pathology_score
        )
        if not np.isfinite(scalar):
            raise SimulationError("Phase candidate produced non-finite scores")
        return PhaseFitEvaluation(
            valid=True,
            reason="accepted",
            scalar_score=scalar,
            phase_scores=phase_scores,
            protocol_scores=protocol_scores,
            cycles=cycles,
            traces=traces if include_traces else None,
            rheobase_lower_pa=rheobase_lower,
            rheobase_upper_pa=rheobase_upper,
            rheobase_loss=float(rheobase_loss),
            parameter_prior=parameter_prior,
            bound_proximity_fraction=bound_proximity,
            maximum_parent_displacement=maximum_parent_displacement,
            kinetic_diagnostics=diagnostics,
            biophysics=biophysics,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return PhaseFitEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            scalar_score=experiment_config.invalid_score,
        )


def _phase_payload(
    unbounded: Sequence[float],
    architecture: str,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig,
    shape_config: PhaseShapeConfig,
) -> tuple[np.ndarray, PhaseFitEvaluation]:
    parameters = _bounded_parameters(unbounded, lower, upper)
    evaluation = evaluate_phase_candidate(
        parameters,
        architecture,
        center,
        lower,
        upper,
        target,
        biological_cycles_target,
        screen_config,
        experiment_config,
        shape_config,
    )
    return parameters, evaluation


def _evaluation_row(
    generation: int,
    individual: int,
    branch: int,
    parameter_names: Sequence[str],
    parameters: np.ndarray,
    evaluation: PhaseFitEvaluation,
) -> dict[str, object]:
    row: dict[str, object] = {
        "generation": generation,
        "individual": individual,
        "branch": branch,
        **dict(zip(parameter_names, parameters)),
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
        row["rheobase_loss"] = evaluation.rheobase_loss
        row["model_rheobase_pa"] = evaluation.rheobase_upper_pa
        row["parameter_prior"] = evaluation.parameter_prior
        row["bound_proximity_fraction"] = (
            evaluation.bound_proximity_fraction
        )
        row["maximum_parent_displacement"] = (
            evaluation.maximum_parent_displacement
        )
    if evaluation.kinetic_diagnostics is not None:
        row.update(
            {
                f"kinetics__{key}": value
                for key, value in (
                    evaluation.kinetic_diagnostics.to_mapping().items()
                )
            }
        )
    return row


def optimize_phase_architecture(
    architecture: str,
    parent_vectors: Sequence[Sequence[float]],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig,
    shape_config: PhaseShapeConfig,
) -> PhaseArchitectureResult:
    """Optimize one architecture from every retained fast-channel parent."""
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Phase-shape optimization requires DEAP") from error

    names, lower, upper = architecture_parameter_bounds(
        architecture,
        bounds_path,
        experiment_config.gna_gk_factor,
    )
    rows = []
    fitness_name = "FitnessPhaseShapeCMA"
    individual_name = "IndividualPhaseShapeCMA"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)

    executor = (
        ProcessPoolExecutor(max_workers=experiment_config.workers)
        if experiment_config.workers > 1
        else None
    )
    mapper = executor.map if executor is not None else map
    try:
        for branch, parent in enumerate(parent_vectors):
            base_center = np.asarray(parent, dtype=float)
            if base_center.shape != (len(CELL_PARAMETER_NAMES),):
                raise ValueError("Each phase parent must contain base parameters")
            center = base_center
            if architecture == "soma-ais":
                center = np.concatenate(
                    (
                        base_center,
                        _ais_phase_initial(),
                    )
                )
            center = np.clip(center, lower + 1e-9, upper - 1e-9)
            initial = _unbounded_parameters(center, lower, upper)
            np.random.seed(
                experiment_config.seed
                + branch
                + (10_000 if architecture == "soma-ais" else 0)
            )
            strategy = cma.Strategy(
                centroid=initial,
                sigma=experiment_config.sigma,
                lambda_=experiment_config.population_size,
            )
            initial_evaluation = evaluate_phase_candidate(
                center,
                architecture,
                center,
                lower,
                upper,
                target,
                biological_cycles_target,
                screen_config,
                experiment_config,
                shape_config,
            )
            rows.append(
                _evaluation_row(
                    -1,
                    -1,
                    branch,
                    names,
                    center,
                    initial_evaluation,
                )
            )
            for generation in range(experiment_config.generations):
                population = strategy.generate(individual_type)
                payloads = list(
                    mapper(
                        _phase_payload,
                        population,
                        [architecture] * len(population),
                        [center] * len(population),
                        [lower] * len(population),
                        [upper] * len(population),
                        [target] * len(population),
                        [biological_cycles_target] * len(population),
                        [screen_config] * len(population),
                        [experiment_config] * len(population),
                        [shape_config] * len(population),
                    )
                )
                for index, (individual, payload) in enumerate(
                    zip(population, payloads)
                ):
                    parameters, evaluation = payload
                    individual.fitness.values = (
                        evaluation.scalar_score,
                    )
                    rows.append(
                        _evaluation_row(
                            generation,
                            index,
                            branch,
                            names,
                            parameters,
                            evaluation,
                        )
                    )
                strategy.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    history = pd.DataFrame(rows)
    valid = history.loc[history["valid"]].copy()
    if valid.empty:
        raise RuntimeError(
            f"{architecture} phase optimization produced no valid model"
        )
    best = valid.nsmallest(1, "scalar_score").copy()
    best_row = best.iloc[0]
    best_parameters = best_row.loc[list(names)].to_numpy(dtype=float)
    parent_index = int(best_row["branch"])
    base_center = np.asarray(parent_vectors[parent_index], dtype=float)
    center = (
        base_center
        if architecture == "one-compartment"
        else np.concatenate(
            (
                base_center,
                _ais_phase_initial(),
            )
        )
    )
    center = np.clip(center, lower + 1e-9, upper - 1e-9)
    evaluation = evaluate_phase_candidate(
        best_parameters,
        architecture,
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
            f"Best {architecture} model failed reevaluation: "
            f"{evaluation.reason}"
        )
    return PhaseArchitectureResult(
        architecture=architecture,
        parameter_names=names,
        history=history,
        best_parameters=best,
        evaluation=evaluation,
    )


def run_phase_adequacy_experiment(
    target: CellOptimizationTarget,
    parent_vectors: Sequence[Sequence[float]],
    bounds_path: str | Path,
    screen_config: BiologicalScreenConfig,
    experiment_config: PhaseExperimentConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
) -> PhaseAdequacyResult:
    experiment_config = experiment_config or PhaseExperimentConfig()
    shape_config = shape_config or PhaseShapeConfig()
    biological = biological_phase_cycles(
        target,
        shape_config,
        include_validation=True,
    )
    one_compartment = optimize_phase_architecture(
        "one-compartment",
        parent_vectors,
        target,
        biological,
        bounds_path,
        screen_config,
        experiment_config,
        shape_config,
    )
    soma_ais = optimize_phase_architecture(
        "soma-ais",
        parent_vectors,
        target,
        biological,
        bounds_path,
        screen_config,
        experiment_config,
        shape_config,
    )
    return PhaseAdequacyResult(
        target=target,
        biological_cycles=biological,
        one_compartment=one_compartment,
        soma_ais=soma_ais,
        config=experiment_config,
        shape_config=shape_config,
    )


def phase_experiment_summary(
    result: PhaseAdequacyResult,
) -> pd.DataFrame:
    rows = []
    for architecture_result in (
        result.one_compartment,
        result.soma_ais,
    ):
        evaluation = architecture_result.evaluation
        initial_best = float(
            architecture_result.history.loc[
                architecture_result.history["generation"].eq(-1)
                & architecture_result.history["valid"],
                "scalar_score",
            ].min()
        )
        row = {
            "architecture": architecture_result.architecture,
            "dataset": result.target.dataset,
            "cell_id": result.target.cell_id,
            "scalar_score": evaluation.scalar_score,
            "model_rheobase_pa": evaluation.rheobase_upper_pa,
            "biological_rheobase_pa": result.target.sampled_rheobase_pa,
            "rheobase_loss": evaluation.rheobase_loss,
            "parameter_prior": evaluation.parameter_prior,
            "bound_proximity_fraction": (
                evaluation.bound_proximity_fraction
            ),
            "maximum_parent_displacement": (
                evaluation.maximum_parent_displacement
            ),
            "initial_best_scalar_score": initial_best,
            "relative_improvement": (
                (initial_best - evaluation.scalar_score) / initial_best
            ),
            **evaluation.phase_scores.to_mapping(),
            **{
                f"kinetics__{key}": value
                for key, value in (
                    evaluation.kinetic_diagnostics.to_mapping().items()
                )
            },
        }
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
        (
            cycle.up_dvdt_mv_ms,
            cycle.down_dvdt_mv_ms[1:],
        )
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


def save_phase_comparison_plot(
    result: PhaseAdequacyResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    )
    series = (
        ("Biological", result.biological_cycles, "#147d7e"),
        (
            "One compartment",
            result.one_compartment.evaluation.cycles,
            "#e76f51",
        ),
        ("Soma-AIS", result.soma_ais.evaluation.cycles, "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        len(protocols),
        3,
        figsize=(13.5, 10.5),
        constrained_layout=True,
    )
    for row_index, protocol in enumerate(protocols):
        biological = result.biological_cycles[protocol.name]
        for label, cycles, color in series:
            cycle = cycles[protocol.name]
            voltage, dvdt = _phase_xy(cycle)
            axes[row_index, 0].plot(
                voltage,
                dvdt,
                color=color,
                linewidth=1.25,
                label=label,
            )
            normalized_voltage, normalized_dvdt = (
                _normalized_phase_xy(cycle)
            )
            axes[row_index, 1].plot(
                normalized_voltage,
                normalized_dvdt,
                color=color,
                linewidth=1.25,
                label=label,
            )
            axes[row_index, 2].plot(
                cycle.grid,
                cycle.up_concavity,
                color=color,
                linewidth=1.25,
                label=label,
            )
        axes[row_index, 2].axvspan(
            0.0,
            biological.up_max_relative_voltage,
            color="#eeeeee",
            zorder=-3,
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.55)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylabel("dV/dt (mV/ms)")
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
    axes[-1, 0].set_xlabel("V - threshold (mV)")
    axes[-1, 1].set_xlabel("Normalized AP voltage")
    axes[-1, 2].set_xlabel("Normalized upstroke voltage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Phase-shape adequacy: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=14,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_phase_score_plot(
    result: PhaseAdequacyResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "physical_curve",
        "normalized_shape",
        "slope_shape",
        "concavity",
        "velocity_extent",
        "landmarks",
        "total",
    )
    labels = (
        "Physical\ncurve",
        "Normalized\nshape",
        "Slope",
        "Concavity",
        "Velocity\nextent",
        "Landmarks",
        "Phase total",
    )
    evaluations = (
        result.one_compartment.evaluation,
        result.soma_ais.evaluation,
    )
    names = ("One compartment", "Soma-AIS")
    colors = ("#e76f51", "#7b2cbf")
    x = np.arange(len(metrics))
    width = 0.34
    figure, axis = plt.subplots(
        figsize=(10.0, 4.8),
        constrained_layout=True,
    )
    for index, (name, evaluation, color) in enumerate(
        zip(names, evaluations, colors)
    ):
        values = evaluation.phase_scores.to_mapping()
        axis.bar(
            x + (index - 0.5) * width,
            [values[metric] for metric in metrics],
            width,
            label=name,
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized loss (lower is better)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_title(
        f"Phase-focused scores: {result.target.dataset} / "
        f"{result.target.cell_id}",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _best_kinetics(
    architecture_result: PhaseArchitectureResult,
) -> KineticParameters:
    row = architecture_result.best_parameters.iloc[0]
    return KineticParameters.from_vector(
        row.loc[list(PARAMETER_NAMES)].to_numpy(dtype=float)
    )


def save_gate_kinetics_plot(
    result: PhaseAdequacyResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voltage = np.linspace(-100.0, 60.0, 321)
    kinetics_series = (
        ("Canonical", KineticParameters.canonical(), "#777777"),
        (
            "One compartment",
            _best_kinetics(result.one_compartment),
            "#e76f51",
        ),
        ("Soma-AIS", _best_kinetics(result.soma_ais), "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 9.0),
        constrained_layout=True,
    )
    factors = result.one_compartment.evaluation.biophysics
    exponent = (result.target.temperature_c - 6.3) / 10.0
    factor_values = (
        float(factors["physical__q10_m"]) ** exponent,
        float(factors["physical__q10_h"]) ** exponent,
        float(factors["physical__q10_n"]) ** exponent,
    )
    for row_index, gate in enumerate(("m", "h", "n")):
        for label, kinetics, color in kinetics_series:
            rates = kinetics.rates(voltage)
            alpha = rates[f"alpha_{gate}"]
            beta = rates[f"beta_{gate}"]
            steady = alpha / (alpha + beta)
            tau = 1.0 / (
                factor_values[row_index] * (alpha + beta)
            )
            axes[row_index, 0].plot(
                voltage,
                steady,
                color=color,
                label=label,
            )
            axes[row_index, 1].plot(
                voltage,
                tau,
                color=color,
                label=label,
            )
        axes[row_index, 0].set_ylabel(f"{gate}_inf")
        axes[row_index, 1].set_ylabel(f"tau_{gate} (ms)")
        axes[row_index, 1].set_yscale("log")
        for axis in axes[row_index]:
            axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("V (mV)")
    axes[-1, 1].set_xlabel("V (mV)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Optimized gate kinetics at {result.target.temperature_c:g} C",
        fontsize=14,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_alpha_beta_plot(
    result: PhaseAdequacyResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voltage = np.linspace(-100.0, 60.0, 321)
    kinetics_series = (
        ("Canonical", KineticParameters.canonical(), "#777777"),
        (
            "One compartment",
            _best_kinetics(result.one_compartment),
            "#e76f51",
        ),
        ("Soma-AIS", _best_kinetics(result.soma_ais), "#7b2cbf"),
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 9.0),
        constrained_layout=True,
    )
    for axis, rate_name in zip(axes.flat, RATE_NAMES):
        for label, kinetics, color in kinetics_series:
            axis.plot(
                voltage,
                kinetics.rates(voltage)[rate_name],
                color=color,
                label=label,
            )
        axis.set_title(rate_name.replace("_", " "), loc="left")
        axis.set_yscale("log")
        axis.set_xlabel("V (mV)")
        axis.set_ylabel("Rate at 6.3 C (1/ms)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Alpha/beta rate families: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=14,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_phase_adequacy_result(
    result: PhaseAdequacyResult,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = phase_experiment_summary(result)
    summary.to_csv(output / "architecture_summary.csv", index=False)
    protocol_rows = []
    for architecture_result in (
        result.one_compartment,
        result.soma_ais,
    ):
        for protocol, scores in (
            architecture_result.evaluation.protocol_scores.items()
        ):
            protocol_rows.append(
                {
                    "architecture": architecture_result.architecture,
                    "protocol": protocol,
                    **scores.to_mapping(),
                }
            )
    pd.DataFrame(protocol_rows).to_csv(
        output / "protocol_phase_scores.csv",
        index=False,
    )
    for architecture_result in (
        result.one_compartment,
        result.soma_ais,
    ):
        prefix = architecture_result.architecture.replace("-", "_")
        architecture_result.history.to_csv(
            output / f"{prefix}_history.csv",
            index=False,
        )
        architecture_result.best_parameters.to_csv(
            output / f"{prefix}_best_model.csv",
            index=False,
        )
    metadata = {
        "version": 1,
        "experiment": "phase-shape model adequacy",
        "target": target_metadata(result.target),
        "experiment_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_phase_comparison_plot(
        result,
        output / "phase_shape_comparison.png",
    )
    save_phase_score_plot(
        result,
        output / "phase_score_bar_plot.png",
    )
    save_gate_kinetics_plot(
        result,
        output / "gate_kinetics.png",
    )
    save_alpha_beta_plot(
        result,
        output / "alpha_beta_rates.png",
    )
