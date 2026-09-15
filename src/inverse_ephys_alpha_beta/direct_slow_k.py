"""A direct, one-state slow-potassium extension of the fast phase model."""

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

from .cell_optimization import FEATURE_GROUPS, _group_score
from .cell_targets import (
    CellOptimizationTarget,
    passive_leak_from_active_current,
    target_metadata,
)
from .direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
    DIRECT_SLOW_K_PARAMETER_NAMES,
    DirectSlowKinetics,
    direct_slow_k_initial,
    direct_slow_k_parameter_bounds,
)
from .direct_phase_fit import (
    DIRECT_STATIC_PARAMETER_NAMES,
    decode_direct_model,
)
from .features import (
    FeatureConfig,
    extract_observed_features,
    extract_voltage_features,
)
from .hh_model import SimulationConfig, SimulationError, Stimulus, Trace
from .model_ladder import (
    DecodedLadderModel,
    _bounded_parameters,
    _rush_larsen_gate_update,
    _unbounded_parameters,
)
from .phase_experiment import biological_phase_cycles
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


DIRECT_SLOW_K_ALL_PARAMETER_NAMES = (
    DIRECT_SLOW_K_PARAMETER_NAMES
    + ("param__slow_k__log_gslow_ms_cm2",)
)


@dataclass(frozen=True)
class DirectSlowKModel:
    base: DecodedLadderModel
    slow_kinetics: DirectSlowKinetics
    gslow_ms_cm2: float


@dataclass(frozen=True)
class DirectSlowKFitConfig:
    population_size: int = 10
    generations: int = 3
    workers: int = 1
    seed: int = 42
    sigma: float = 0.35
    train_duration_ms: float = 500.0
    phase_duration_ms: float = 300.0
    rheobase_probe_duration_ms: float = 300.0
    rheobase_probe_tolerance_pa: float = 5.0
    rheobase_constraint_weight: float = 10.0
    rheobase_tolerance_fraction: float = 0.15
    rheobase_tolerance_floor_pa: float = 10.0
    physical_constraint_weight: float = 20.0
    firing_pattern_weight: float = 4.0
    spike_count_weight: float = 4.0
    waveform_weight: float = 0.5
    shape_guard_weight: float = 100.0
    shape_guard_relative_tolerance: float = 0.25
    smoothness_weight: float = 0.05
    invalid_score: float = 10_000.0


@dataclass(frozen=True)
class DirectSlowKEvaluation:
    valid: bool
    reason: str
    objective_total: float
    shape_loss: float = float("nan")
    scale_loss: float = float("nan")
    firing_pattern_loss: float = float("nan")
    spike_count_loss: float = float("nan")
    waveform_loss: float = float("nan")
    regularity_loss: float = float("nan")
    rheobase_constraint_loss: float = float("nan")
    model_rheobase_pa: float = float("nan")
    phase_scores: PhaseShapeScores | None = None
    protocol_scores: Mapping[str, PhaseShapeScores] | None = None
    protocol_features: Mapping[str, Mapping[str, float]] | None = None
    traces: Mapping[str, Trace] | None = None
    cycles: Mapping[str, PhaseCycle] | None = None
    physical: Mapping[str, float] | None = None


@dataclass(frozen=True)
class DirectSlowKFitResult:
    target: CellOptimizationTarget
    base_model: DecodedLadderModel
    biological_cycles: Mapping[str, PhaseCycle]
    biological_features: Mapping[str, Mapping[str, float]]
    parent_evaluation: DirectSlowKEvaluation
    best_evaluation: DirectSlowKEvaluation
    best_parameters: np.ndarray
    history: pd.DataFrame
    config: DirectSlowKFitConfig
    shape_config: PhaseShapeConfig


def direct_slow_k_bounds() -> tuple[np.ndarray, np.ndarray]:
    kinetic_lower, kinetic_upper = direct_slow_k_parameter_bounds()
    return (
        np.concatenate((kinetic_lower, (math.log(0.001),))),
        np.concatenate((kinetic_upper, (math.log(20.0),))),
    )


def direct_slow_k_fit_initial() -> np.ndarray:
    return np.concatenate((direct_slow_k_initial(), (math.log(0.05),)))


def load_direct_parent(
    parameter_path: str | Path,
    stage: str,
    target: CellOptimizationTarget,
) -> DecodedLadderModel:
    parameters = pd.read_csv(parameter_path)
    selected = parameters.loc[parameters["stage"].eq(stage)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {stage!r} parent row in {parameter_path}"
        )
    row = selected.iloc[0]
    kinetic = row.loc[
        list(DIRECT_KINETIC_PARAMETER_NAMES)
    ].to_numpy(dtype=float)
    static = row.loc[
        list(DIRECT_STATIC_PARAMETER_NAMES)
    ].to_numpy(dtype=float)
    return decode_direct_model(kinetic, static, target)


def _fast_active_current(
    voltage_mv: float,
    base: DecodedLadderModel,
) -> tuple[float, float]:
    m_gate, h_gate, n_gate = base.kinetics.steady_state(voltage_mv)
    sodium = (
        base.biophysics.gna_ms_cm2
        * m_gate**3
        * h_gate
        * (voltage_mv - base.biophysics.ena_mv)
    )
    potassium = (
        base.biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage_mv - base.biophysics.ek_mv)
    )
    return float(sodium), float(potassium)


def decode_direct_slow_k_model(
    base: DecodedLadderModel,
    parameters: Sequence[float],
    target: CellOptimizationTarget,
    gslow_override_ms_cm2: float | None = None,
) -> DirectSlowKModel:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(DIRECT_SLOW_K_ALL_PARAMETER_NAMES),):
        raise ValueError(f"Invalid direct slow-K vector: {values.shape}")
    slow = DirectSlowKinetics.from_parameters(values[:-1])
    gslow = (
        math.exp(float(values[-1]))
        if gslow_override_ms_cm2 is None
        else float(gslow_override_ms_cm2)
    )

    def active_current(voltage_mv: float) -> float:
        sodium, potassium = _fast_active_current(voltage_mv, base)
        slow_gate, _ = slow.gate_curves_scalar(voltage_mv)
        return float(
            sodium
            + potassium
            + gslow
            * slow_gate
            * (voltage_mv - base.biophysics.ek_mv)
        )

    gleak, eleak = passive_leak_from_active_current(
        target.passive,
        active_current,
    )
    biophysics = replace(
        base.biophysics,
        gleak_ms_cm2=gleak,
        eleak_mv=eleak,
    )
    resting_voltage = target.passive.resting_voltage_mv
    ais_kinetics = base.ais_kinetics or base.kinetics
    am_gate, ah_gate, an_gate = ais_kinetics.steady_state(resting_voltage)
    ais_active = (
        biophysics.gna_ms_cm2
        * base.ais_gna_multiplier
        * am_gate**3
        * ah_gate
        * (resting_voltage - biophysics.ena_mv)
        + biophysics.gk_ms_cm2
        * an_gate**4
        * (resting_voltage - biophysics.ek_mv)
    )
    ais_eleak = resting_voltage + ais_active / gleak
    anchored_base = replace(
        base,
        biophysics=biophysics,
        ais_eleak_mv=float(ais_eleak),
    )
    return DirectSlowKModel(
        base=anchored_base,
        slow_kinetics=slow,
        gslow_ms_cm2=gslow,
    )


def direct_slow_k_initial_state(
    model: DirectSlowKModel,
    target: CellOptimizationTarget,
) -> np.ndarray:
    voltage = target.passive.resting_voltage_mv
    sm, sh, sn = model.base.kinetics.steady_state(voltage)
    ais_kinetics = model.base.ais_kinetics or model.base.kinetics
    am, ah, an = ais_kinetics.steady_state(voltage)
    slow, _ = model.slow_kinetics.gate_curves_scalar(voltage)
    return np.asarray(
        (voltage, sm, sh, sn, slow, voltage, am, ah, an),
        dtype=float,
    )


def _currents(
    state: np.ndarray,
    model: DirectSlowKModel,
) -> tuple[float, float, float, float, float]:
    voltage, m_gate, h_gate, n_gate, slow_gate = state[:5]
    biophysics = model.base.biophysics
    sodium = (
        biophysics.gna_ms_cm2
        * m_gate**3
        * h_gate
        * (voltage - biophysics.ena_mv)
    )
    fast_k = (
        biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage - biophysics.ek_mv)
    )
    slow_k = (
        model.gslow_ms_cm2
        * slow_gate
        * (voltage - biophysics.ek_mv)
    )
    leak = biophysics.gleak_ms_cm2 * (
        voltage - biophysics.eleak_mv
    )
    return (
        float(sodium),
        float(fast_k),
        float(slow_k),
        float(leak),
        float(sodium + fast_k + slow_k + leak),
    )


def _ais_currents(
    state: np.ndarray,
    model: DirectSlowKModel,
) -> tuple[float, float, float, float]:
    voltage, m_gate, h_gate, n_gate = (
        state[5],
        state[6],
        state[7],
        state[8],
    )
    base = model.base
    biophysics = base.biophysics
    sodium = (
        biophysics.gna_ms_cm2
        * base.ais_gna_multiplier
        * m_gate**3
        * h_gate
        * (voltage - biophysics.ena_mv)
    )
    potassium = (
        biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage - biophysics.ek_mv)
    )
    leak = biophysics.gleak_ms_cm2 * (
        voltage - base.ais_eleak_mv
    )
    return (
        float(sodium),
        float(potassium),
        float(leak),
        float(sodium + potassium + leak),
    )


def _voltage_rhs(
    time_ms: float,
    state: np.ndarray,
    model: DirectSlowKModel,
    config: SimulationConfig,
) -> np.ndarray:
    base = model.base
    biophysics = base.biophysics
    current_pa = config.stimulus.pa_value(time_ms, biophysics)
    soma_ion = _currents(state, model)[-1]
    ais_ion = _ais_currents(state, model)[-1]
    soma_area_um2 = biophysics.membrane_area_um2 * (
        1.0 - base.ais_area_fraction
    )
    ais_area_um2 = (
        biophysics.membrane_area_um2 * base.ais_area_fraction
    )
    coupling_pa = base.coupling_ns * (state[0] - state[5])
    soma_density = current_pa / (soma_area_um2 * 0.01)
    soma_coupling = coupling_pa / (soma_area_um2 * 0.01)
    ais_coupling = coupling_pa / (ais_area_um2 * 0.01)
    return np.asarray(
        (
            (
                soma_density - soma_ion - soma_coupling
            )
            / biophysics.capacitance_uf_cm2,
            (
                -ais_ion + ais_coupling
            )
            / biophysics.capacitance_uf_cm2,
        ),
        dtype=float,
    )


def _gate_half_step(
    state: np.ndarray,
    model: DirectSlowKModel,
    config: SimulationConfig,
    duration_ms: float,
) -> np.ndarray:
    updated = state.copy()
    updated[1:4] = _rush_larsen_gate_update(
        float(state[0]),
        state[1:4],
        model.base.kinetics,
        model.base,
        config.temperature_c,
        duration_ms,
    )
    slow_steady, slow_tau = model.slow_kinetics.gate_curves_scalar(
        float(state[0]),
        config.temperature_c,
    )
    slow_decay = math.exp(
        max(-80.0, -duration_ms / max(slow_tau, 1e-12))
    )
    updated[4] = slow_steady + (
        float(state[4]) - slow_steady
    ) * slow_decay
    updated[6:9] = _rush_larsen_gate_update(
        float(state[5]),
        state[6:9],
        model.base.ais_kinetics or model.base.kinetics,
        model.base,
        config.temperature_c,
        duration_ms,
    )
    return updated


def _step(
    time_ms: float,
    state: np.ndarray,
    model: DirectSlowKModel,
    config: SimulationConfig,
) -> np.ndarray:
    dt = config.dt_ms
    half = _gate_half_step(state, model, config, dt / 2.0)
    voltage_indices = np.asarray((0, 5), dtype=int)

    def rhs(rhs_time: float, voltages: np.ndarray) -> np.ndarray:
        trial = half.copy()
        trial[voltage_indices] = voltages
        return _voltage_rhs(rhs_time, trial, model, config)

    voltage = half[voltage_indices]
    k1 = rhs(time_ms, voltage)
    k2 = rhs(time_ms + dt / 2.0, voltage + dt * k1 / 2.0)
    k3 = rhs(time_ms + dt / 2.0, voltage + dt * k2 / 2.0)
    k4 = rhs(time_ms + dt, voltage + dt * k3)
    half[voltage_indices] = voltage + dt * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0
    updated = _gate_half_step(half, model, config, dt / 2.0)
    if (
        not np.all(np.isfinite(updated))
        or np.any(np.abs(updated[voltage_indices]) > 200.0)
        or np.any(updated[[1, 2, 3, 4, 6, 7, 8]] < -0.05)
        or np.any(updated[[1, 2, 3, 4, 6, 7, 8]] > 1.05)
    ):
        raise SimulationError(
            f"Direct slow-K integration diverged at {time_ms + dt:.4f} ms"
        )
    updated[[1, 2, 3, 4, 6, 7, 8]] = np.clip(
        updated[[1, 2, 3, 4, 6, 7, 8]],
        0.0,
        1.0,
    )
    return updated


def simulate_direct_slow_k(
    model: DirectSlowKModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> Trace:
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, 9), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for index in range(n_steps):
        states[index + 1] = _step(
            time_ms[index],
            states[index],
            model,
            config,
        )
    biophysics = model.base.biophysics
    applied_pa = np.asarray(
        [
            config.stimulus.pa_value(time, biophysics)
            for time in time_ms
        ],
        dtype=float,
    )
    soma_area_um2 = biophysics.membrane_area_um2 * (
        1.0 - model.base.ais_area_fraction
    )
    applied_density = applied_pa / (soma_area_um2 * 0.01)
    current_rows = np.asarray(
        [_currents(state, model) for state in states],
        dtype=float,
    )
    sodium = current_rows[:, 0]
    potassium = current_rows[:, 1] + current_rows[:, 2]
    leak = current_rows[:, 3]
    ionic = current_rows[:, 4]
    dvdt = np.asarray(
        [
            _voltage_rhs(time, state, model, config)[0]
            for time, state in zip(time_ms, states)
        ],
        dtype=float,
    )
    return Trace(
        time_ms=time_ms,
        voltage_mv=states[:, 0],
        m=states[:, 1],
        h=states[:, 2],
        n=states[:, 3],
        applied_current_ua_cm2=applied_density,
        applied_current_pa=applied_pa,
        ionic_current_ua_cm2=ionic,
        ionic_current_pa=biophysics.current_density_to_pa(ionic),
        sodium_current_ua_cm2=sodium,
        potassium_current_ua_cm2=potassium,
        leak_current_ua_cm2=leak,
        dvdt_mv_ms=dvdt,
    )


def elicits_spike_direct_slow_k(
    model: DirectSlowKModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> bool:
    state = np.asarray(initial_state, dtype=float)
    n_steps = int(round(config.duration_ms / config.dt_ms))
    for index in range(n_steps):
        previous_voltage = float(state[0])
        state = _step(
            index * config.dt_ms,
            state,
            model,
            config,
        )
        if previous_voltage < 0.0 <= state[0]:
            return True
    return False


def find_rheobase_direct_slow_k(
    model: DirectSlowKModel,
    initial_state: np.ndarray,
    screen_config: BiologicalScreenConfig,
    duration_ms: float = 300.0,
    tolerance_pa: float = 5.0,
) -> tuple[float, float, int]:
    maximum_current = float(screen_config.rheobase_currents_pa[-1])

    def spikes(current_pa: float) -> bool:
        simulation = step_simulation_config(
            current_pa,
            min(duration_ms, screen_config.rheobase_step_ms),
            screen_config,
            model.base.biophysics,
            float(initial_state[0]),
        )
        return elicits_spike_direct_slow_k(
            model,
            simulation,
            initial_state,
        )

    if spikes(0.0):
        return 0.0, 0.0, 0
    if not spikes(maximum_current):
        raise SimulationError("No spike within the rheobase search range")
    lower_pa = 0.0
    upper_pa = maximum_current
    iterations = 0
    while upper_pa - lower_pa > tolerance_pa and iterations < 20:
        midpoint = (lower_pa + upper_pa) / 2.0
        if spikes(midpoint):
            upper_pa = midpoint
        else:
            lower_pa = midpoint
        iterations += 1
    return float(lower_pa), float(upper_pa), iterations


def biological_window_features(
    target: CellOptimizationTarget,
    duration_ms: float,
) -> dict[str, dict[str, float]]:
    sweeps = {
        sweep.sweep_number: sweep
        for sweep in read_current_clamp_sweeps(
            target.nwb_path,
            long_square_only=True,
        )
    }
    output: dict[str, dict[str, float]] = {}
    feature_config = FeatureConfig(min_spikes=1)
    for protocol in (
        *target.training_protocols,
        *target.validation_protocols,
    ):
        sweep = sweeps[protocol.sweep_number]
        start = float(sweep.stimulus_start_ms)
        end = min(
            float(sweep.stimulus_end_ms),
            start + duration_ms,
        )
        output[protocol.name] = extract_voltage_features(
            sweep.time_ms,
            sweep.voltage_mv,
            Stimulus(
                amplitude_pa=protocol.current_pa,
                amplitude_ua_cm2=None,
                start_ms=start,
                end_ms=end,
            ),
            feature_config,
        )
    return output


def _mean_feature_group(
    model_features: Mapping[str, Mapping[str, float]],
    biological_features: Mapping[str, Mapping[str, float]],
    protocols,
    group: str,
) -> float:
    return float(
        np.mean(
            [
                _group_score(
                    model_features[protocol.name],
                    biological_features[protocol.name],
                    FEATURE_GROUPS[group],
                    missing_penalty=25.0,
                )
                for protocol in protocols
            ]
        )
    )


def _mean_spike_count_loss(
    model_features: Mapping[str, Mapping[str, float]],
    biological_features: Mapping[str, Mapping[str, float]],
    protocols,
) -> float:
    return float(
        np.mean(
            [
                _group_score(
                    model_features[protocol.name],
                    biological_features[protocol.name],
                    ("spike_count", "firing_rate_hz"),
                    missing_penalty=25.0,
                )
                for protocol in protocols
            ]
        )
    )


def evaluate_direct_slow_k_candidate(
    parameters: Sequence[float],
    base_model: DecodedLadderModel,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: DirectSlowKFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
    include_validation: bool = False,
    include_artifacts: bool = False,
    gslow_override_ms_cm2: float | None = None,
) -> DirectSlowKEvaluation:
    try:
        values = np.asarray(parameters, dtype=float)
        lower, upper = direct_slow_k_bounds()
        if (
            values.shape != lower.shape
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError("Slow-K candidate is outside fitted bounds")
        model = decode_direct_slow_k_model(
            base_model,
            values,
            target,
            gslow_override_ms_cm2=gslow_override_ms_cm2,
        )
        if (
            not -120.0 <= model.base.biophysics.eleak_mv <= -20.0
            or not -120.0 <= model.base.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        voltage = np.linspace(-100.0, 60.0, 161)
        _, tau = model.slow_kinetics.gate_curves(
            voltage,
            target.temperature_c,
        )
        if float(np.min(tau)) < 1.0:
            raise SimulationError("Slow-K tau fell below 1 ms")
        state = direct_slow_k_initial_state(model, target)
        lower_rheobase, upper_rheobase, _ = (
            find_rheobase_direct_slow_k(
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
        protocol_features: dict[str, Mapping[str, float]] = {}
        for protocol in protocols:
            phase_duration = min(
                protocol.duration_ms,
                fit_config.phase_duration_ms,
            )
            phase_current_pa = (
                protocol.rheobase_factor * upper_rheobase
            )
            phase_simulation = step_simulation_config(
                phase_current_pa,
                phase_duration,
                screen_config,
                model.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            phase_trace = simulate_direct_slow_k(
                model,
                phase_simulation,
                state,
            )
            phase_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=phase_current_pa,
                start_ms=screen_config.baseline_ms,
                end_ms=(
                    screen_config.baseline_ms + phase_duration
                ),
            )
            cycle = extract_phase_cycle(
                phase_trace.time_ms,
                phase_trace.voltage_mv,
                phase_stimulus,
                shape_config,
            )
            protocol_scores[protocol.name] = compare_phase_cycles(
                cycle,
                biological_cycles_target[protocol.name],
                shape_config,
            )
            train_duration = min(
                protocol.duration_ms,
                fit_config.train_duration_ms,
            )
            train_simulation = step_simulation_config(
                protocol.current_pa,
                train_duration,
                screen_config,
                model.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            train_trace = simulate_direct_slow_k(
                model,
                train_simulation,
                state,
            )
            train_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=protocol.current_pa,
                start_ms=screen_config.baseline_ms,
                end_ms=(
                    screen_config.baseline_ms + train_duration
                ),
            )
            protocol_features[protocol.name] = (
                extract_observed_features(
                    train_trace,
                    train_stimulus,
                    FeatureConfig(min_spikes=1),
                )
            )
            if include_artifacts:
                traces[protocol.name] = train_trace
                cycles[protocol.name] = cycle
        training_scores = {
            protocol.name: protocol_scores[protocol.name]
            for protocol in target.training_protocols
        }
        phase_scores = mean_phase_scores(training_scores)
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
        firing_pattern_loss = _mean_feature_group(
            protocol_features,
            biological_features_target,
            target.training_protocols,
            "firing_pattern",
        )
        spike_count_loss = _mean_spike_count_loss(
            protocol_features,
            biological_features_target,
            target.training_protocols,
        )
        waveform_loss = 0.5 * (
            _mean_feature_group(
                protocol_features,
                biological_features_target,
                target.training_protocols,
                "spike_shape",
            )
            + _mean_feature_group(
                protocol_features,
                biological_features_target,
                target.training_protocols,
                "spike_dynamics",
            )
        )
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
        shape_guard_limit = (
            (1.0 + fit_config.shape_guard_relative_tolerance)
            * max(reference_shape_loss, 1e-12)
        )
        shape_guard = max(
            0.0,
            shape_loss / shape_guard_limit - 1.0,
        ) ** 2
        regularity = float(
            fit_config.smoothness_weight
            * model.slow_kinetics.smoothness_penalty()
        )
        objective = float(
            shape_loss
            + scale_loss
            + fit_config.firing_pattern_weight
            * firing_pattern_loss
            + fit_config.spike_count_weight * spike_count_loss
            + fit_config.waveform_weight * waveform_loss
            + fit_config.rheobase_constraint_weight
            * rheobase_constraint_loss
            + fit_config.shape_guard_weight * shape_guard
            + regularity
        )
        physical = model.base.biophysics.to_physical_mapping()
        physical.update(
            {
                "extension__gslow_ms_cm2": model.gslow_ms_cm2,
                "extension__slow_k_q10": model.slow_kinetics.q10,
                "extension__ais_eleak_mv": model.base.ais_eleak_mv,
            }
        )
        return DirectSlowKEvaluation(
            valid=True,
            reason="accepted",
            objective_total=objective,
            shape_loss=shape_loss,
            scale_loss=scale_loss,
            firing_pattern_loss=firing_pattern_loss,
            spike_count_loss=spike_count_loss,
            waveform_loss=waveform_loss,
            regularity_loss=regularity,
            rheobase_constraint_loss=rheobase_constraint_loss,
            model_rheobase_pa=float(upper_rheobase),
            phase_scores=phase_scores,
            protocol_scores=protocol_scores,
            protocol_features=protocol_features,
            traces=traces if include_artifacts else None,
            cycles=cycles if include_artifacts else None,
            physical=physical,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return DirectSlowKEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            objective_total=fit_config.invalid_score,
        )


def _payload(
    unbounded: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
    base_model: DecodedLadderModel,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: DirectSlowKFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
) -> tuple[np.ndarray, DirectSlowKEvaluation]:
    parameters = _bounded_parameters(unbounded, lower, upper)
    evaluation = evaluate_direct_slow_k_candidate(
        parameters,
        base_model,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        fit_config,
        shape_config,
        reference_shape_loss,
    )
    return parameters, evaluation


def _history_row(
    generation: int,
    individual: int,
    parameters: np.ndarray,
    evaluation: DirectSlowKEvaluation,
) -> dict[str, object]:
    row: dict[str, object] = {
        "generation": generation,
        "individual": individual,
        **dict(zip(DIRECT_SLOW_K_ALL_PARAMETER_NAMES, parameters)),
        "valid": evaluation.valid,
        "reason": evaluation.reason,
        "objective_total": evaluation.objective_total,
    }
    if evaluation.valid:
        row.update(
            {
                "shape_loss": evaluation.shape_loss,
                "scale_loss": evaluation.scale_loss,
                "firing_pattern_loss": evaluation.firing_pattern_loss,
                "spike_count_loss": evaluation.spike_count_loss,
                "waveform_loss": evaluation.waveform_loss,
                "regularity_loss": evaluation.regularity_loss,
                "model_rheobase_pa": evaluation.model_rheobase_pa,
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


def optimize_direct_slow_k(
    target: CellOptimizationTarget,
    base_model: DecodedLadderModel,
    screen_config: BiologicalScreenConfig,
    config: DirectSlowKFitConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
) -> DirectSlowKFitResult:
    config = config or DirectSlowKFitConfig()
    shape_config = shape_config or PhaseShapeConfig()
    biological_cycles_target = biological_phase_cycles(
        target,
        shape_config,
        include_validation=True,
    )
    biological_features_target = biological_window_features(
        target,
        config.train_duration_ms,
    )
    center = direct_slow_k_fit_initial()
    lower, upper = direct_slow_k_bounds()
    provisional_parent = evaluate_direct_slow_k_candidate(
        center,
        base_model,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        reference_shape_loss=1.0,
        include_validation=True,
        include_artifacts=True,
        gslow_override_ms_cm2=0.0,
    )
    if not provisional_parent.valid:
        raise RuntimeError(
            "Frozen direct parent failed slow-K reevaluation: "
            f"{provisional_parent.reason}"
        )
    reference_shape_loss = provisional_parent.shape_loss
    parent_evaluation = evaluate_direct_slow_k_candidate(
        center,
        base_model,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        reference_shape_loss=reference_shape_loss,
        include_validation=True,
        include_artifacts=True,
        gslow_override_ms_cm2=0.0,
    )
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Direct slow-K fitting requires DEAP") from error
    if config.population_size < 4:
        raise ValueError("CMA population size must be at least four")
    fitness_name = "FitnessDirectSlowKCMA"
    individual_name = "IndividualDirectSlowKCMA"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)
    np.random.seed(config.seed)
    strategy = cma.Strategy(
        centroid=_unbounded_parameters(center, lower, upper),
        sigma=config.sigma,
        lambda_=config.population_size,
    )
    initial_evaluation = evaluate_direct_slow_k_candidate(
        center,
        base_model,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        reference_shape_loss,
    )
    rows = [
        _history_row(-1, -1, center, initial_evaluation)
    ]
    executor = None
    mapper = map
    if config.workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=config.workers)
            mapper = executor.map
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"Parallel evaluation unavailable ({error}); using one worker.",
                RuntimeWarning,
            )
    try:
        for generation in range(config.generations):
            population = strategy.generate(individual_type)
            payloads = list(
                mapper(
                    _payload,
                    population,
                    [lower] * len(population),
                    [upper] * len(population),
                    [base_model] * len(population),
                    [target] * len(population),
                    [biological_cycles_target] * len(population),
                    [biological_features_target] * len(population),
                    [screen_config] * len(population),
                    [config] * len(population),
                    [shape_config] * len(population),
                    [reference_shape_loss] * len(population),
                )
            )
            for index, (individual, payload) in enumerate(
                zip(population, payloads)
            ):
                parameters, evaluation = payload
                individual.fitness.values = (
                    evaluation.objective_total,
                )
                rows.append(
                    _history_row(
                        generation,
                        index,
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
    feasible = valid.loc[
        valid["rheobase_constraint_loss"].le(1e-12)
    ]
    if not feasible.empty:
        valid = feasible
    if valid.empty:
        raise RuntimeError("Slow-K optimization produced no valid model")
    best_row = valid.nsmallest(1, "objective_total").iloc[0]
    best_parameters = best_row.loc[
        list(DIRECT_SLOW_K_ALL_PARAMETER_NAMES)
    ].to_numpy(dtype=float)
    best_evaluation = evaluate_direct_slow_k_candidate(
        best_parameters,
        base_model,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        reference_shape_loss,
        include_validation=True,
        include_artifacts=True,
    )
    if not best_evaluation.valid:
        raise RuntimeError(
            "Selected slow-K model failed reevaluation: "
            f"{best_evaluation.reason}"
        )
    return DirectSlowKFitResult(
        target=target,
        base_model=base_model,
        biological_cycles=biological_cycles_target,
        biological_features=biological_features_target,
        parent_evaluation=parent_evaluation,
        best_evaluation=best_evaluation,
        best_parameters=best_parameters,
        history=history,
        config=config,
        shape_config=shape_config,
    )


def direct_slow_k_summary(
    result: DirectSlowKFitResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, evaluation in (
        ("frozen_fast_parent", result.parent_evaluation),
        ("direct_slow_k", result.best_evaluation),
    ):
        row: dict[str, object] = {
            "stage": stage,
            "dataset": result.target.dataset,
            "cell_id": result.target.cell_id,
            "objective_total": evaluation.objective_total,
            "shape_loss": evaluation.shape_loss,
            "scale_loss": evaluation.scale_loss,
            "firing_pattern_loss": evaluation.firing_pattern_loss,
            "spike_count_loss": evaluation.spike_count_loss,
            "waveform_loss": evaluation.waveform_loss,
            "regularity_loss": evaluation.regularity_loss,
            "model_rheobase_pa": evaluation.model_rheobase_pa,
            "biological_rheobase_pa": result.target.sampled_rheobase_pa,
            "rheobase_constraint_loss": (
                evaluation.rheobase_constraint_loss
            ),
            **evaluation.phase_scores.to_mapping(),
        }
        if evaluation.physical is not None:
            row.update(evaluation.physical)
        rows.append(row)
    return pd.DataFrame(rows)


def direct_slow_k_feature_comparison(
    result: DirectSlowKFitResult,
) -> pd.DataFrame:
    rows = []
    for protocol in (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    ):
        for feature in FEATURE_GROUPS["firing_pattern"]:
            rows.append(
                {
                    "protocol": protocol.name,
                    "role": protocol.role,
                    "feature": feature,
                    "biological": result.biological_features[
                        protocol.name
                    ].get(feature, float("nan")),
                    "frozen_fast_parent": (
                        result.parent_evaluation.protocol_features[
                            protocol.name
                        ].get(feature, float("nan"))
                    ),
                    "direct_slow_k": (
                        result.best_evaluation.protocol_features[
                            protocol.name
                        ].get(feature, float("nan"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _phase_voltage(
    cycle: PhaseCycle,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        cycle.threshold_voltage_mv + cycle.grid * cycle.amplitude_mv,
        cycle.peak_voltage_mv
        - cycle.grid * cycle.repolarization_amplitude_mv,
    )


def save_direct_slow_k_phase_plot(
    result: DirectSlowKFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = (
        ("Biological", None, "#147d7e"),
        ("Frozen fast", result.parent_evaluation, "#e76f51"),
        ("+ direct slow K", result.best_evaluation, "#457b9d"),
    )
    protocols = (
        *result.target.training_protocols,
        *result.target.validation_protocols,
    )
    figure, axes = plt.subplots(
        len(protocols),
        2,
        figsize=(11.2, 9.2),
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
        role = "held out" if protocol.role == "validation" else "training"
        axes[row_index, 0].set_title(
            f"{protocol.name.title()} ({role}): physical",
            loc="left",
            fontsize=9,
        )
        axes[row_index, 1].set_title(
            "Normalized branch shape",
            loc="left",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylabel("dV/dt (mV/ms)")
    axes[-1, 0].set_xlabel("V (mV)")
    axes[-1, 1].set_xlabel("Normalized branch voltage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Direct slow-K phase test: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_slow_k_trace_plot(
    result: DirectSlowKFitResult,
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
        ("Frozen fast", result.parent_evaluation, "#e76f51"),
        ("+ direct slow K", result.best_evaluation, "#457b9d"),
    )
    figure, axes = plt.subplots(
        len(protocols),
        1,
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )
    for axis, protocol in zip(axes, protocols):
        sweep = sweeps[protocol.sweep_number]
        end = min(
            float(sweep.stimulus_end_ms),
            float(sweep.stimulus_start_ms)
            + result.config.train_duration_ms,
        )
        mask = (
            (sweep.time_ms >= sweep.stimulus_start_ms - 20.0)
            & (sweep.time_ms <= end)
        )
        axis.plot(
            sweep.time_ms[mask] - sweep.stimulus_start_ms,
            sweep.voltage_mv[mask],
            color="#147d7e",
            linewidth=1.1,
            label="Biological",
        )
        for label, evaluation, color in stages:
            trace = evaluation.traces[protocol.name]
            model_time = trace.time_ms - screen_config.baseline_ms
            model_mask = (
                (model_time >= -20.0)
                & (
                    model_time
                    <= result.config.train_duration_ms
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
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"Slow-K spike-train test: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=13,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_slow_k_score_plot(
    summary: pd.DataFrame,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "shape_loss",
        "physical_constraint_loss",
        "firing_pattern_loss",
        "spike_count_loss",
        "waveform_loss",
        "objective_total",
    )
    labels = (
        "Phase shape",
        "Physical constraints",
        "Spike train",
        "Spike count",
        "Waveform",
        "Total",
    )
    colors = ("#e76f51", "#457b9d")
    x = np.arange(len(metrics))
    width = 0.34
    figure, axis = plt.subplots(
        figsize=(10.4, 4.8),
        constrained_layout=True,
    )
    for index, (_, row) in enumerate(summary.iterrows()):
        values = np.maximum(
            row.loc[list(metrics)].to_numpy(dtype=float),
            1e-3,
        )
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=colors[index],
            label=row["stage"].replace("_", " ").title(),
        )
    axis.set_xticks(x, labels)
    axis.set_yscale("log")
    axis.set_ylabel("Normalized loss (log scale)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_title(
        f"Slow-K extension: {summary['dataset'].iloc[0]} / "
        f"{summary['cell_id'].iloc[0]}",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_slow_k_kinetics_plot(
    result: DirectSlowKFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinetics = DirectSlowKinetics.from_parameters(
        result.best_parameters[:-1]
    )
    voltage = np.linspace(-100.0, 60.0, 321)
    steady, tau = kinetics.gate_curves(
        voltage,
        result.target.temperature_c,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 4.0),
        constrained_layout=True,
    )
    axes[0].plot(voltage, steady, color="#457b9d")
    axes[1].plot(voltage, tau, color="#457b9d")
    axes[0].set_ylabel("p_inf")
    axes[1].set_ylabel(
        f"tau_p at {result.target.temperature_c:g} C (ms)"
    )
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Optimized direct slow-K gate, "
        f"g={math.exp(result.best_parameters[-1]):.3g} mS/cm2",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_direct_slow_k_result(
    result: DirectSlowKFitResult,
    screen_config: BiologicalScreenConfig,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = direct_slow_k_summary(result)
    summary.to_csv(output / "stage_summary.csv", index=False)
    result.history.to_csv(output / "optimization_history.csv", index=False)
    pd.DataFrame(
        [
            {
                **dict(
                    zip(
                        DIRECT_SLOW_K_ALL_PARAMETER_NAMES,
                        result.best_parameters,
                    )
                )
            }
        ]
    ).to_csv(output / "best_slow_k_parameters.csv", index=False)
    direct_slow_k_feature_comparison(result).to_csv(
        output / "firing_pattern_comparison.csv",
        index=False,
    )
    metadata = {
        "version": 1,
        "experiment": "frozen direct fast model plus direct slow K",
        "target": target_metadata(result.target),
        "fit_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
        "screen_config": asdict(screen_config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_direct_slow_k_phase_plot(
        result,
        output / "slow_k_phase_comparison.png",
    )
    save_direct_slow_k_trace_plot(
        result,
        screen_config,
        output / "slow_k_trace_comparison.png",
    )
    save_direct_slow_k_score_plot(
        summary,
        output / "slow_k_score_bars.png",
    )
    save_direct_slow_k_kinetics_plot(
        result,
        output / "slow_k_kinetics.png",
    )
