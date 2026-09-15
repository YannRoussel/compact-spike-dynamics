"""Discrete gate topology and effective slow-current model selection.

The model deliberately separates two ideas:

* integer powers change the cooperativity of an existing gate;
* an independent gate adds a dynamical state, and therefore memory.

The first rung contains fast Na and K currents plus optional slow Na and K
components. Slow Na shares the fast activation gate but has its own recovery
state. Slow K has its own activation state. This gives four current components
without introducing unidentifiable unused gates.
"""

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

from .cell_optimization import FEATURE_GROUPS
from .cell_targets import (
    CellOptimizationTarget,
    passive_leak_from_active_current,
    target_metadata,
)
from .direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
    DIRECT_SLOW_K_PARAMETER_NAMES,
    DIRECT_VOLTAGE_KNOTS_MV,
    DirectSlowKinetics,
    direct_kinetic_parameter_bounds,
    direct_slow_k_initial,
    direct_slow_k_parameter_bounds,
)
from .direct_phase_fit import (
    DIRECT_STATIC_PARAMETER_NAMES,
    _static_bounds,
    decode_direct_model,
)
from .direct_slow_k import (
    _mean_feature_group,
    _mean_spike_count_loss,
    biological_window_features,
)
from .features import FeatureConfig, extract_observed_features
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


SLOW_NA_PARAMETER_NAMES = tuple(
    name.replace("param__slow_k__", "param__slow_na__")
    for name in DIRECT_SLOW_K_PARAMETER_NAMES
)
EFFECTIVE_PARAMETER_NAMES = (
    DIRECT_KINETIC_PARAMETER_NAMES
    + DIRECT_STATIC_PARAMETER_NAMES
    + SLOW_NA_PARAMETER_NAMES
    + ("param__slow_na__log_gna_ms_cm2",)
    + DIRECT_SLOW_K_PARAMETER_NAMES
    + ("param__slow_k__log_gk_ms_cm2",)
)

_FAST_KINETIC_SLICE = slice(0, len(DIRECT_KINETIC_PARAMETER_NAMES))
_FAST_STATIC_START = _FAST_KINETIC_SLICE.stop
_FAST_STATIC_SLICE = slice(
    _FAST_STATIC_START,
    _FAST_STATIC_START + len(DIRECT_STATIC_PARAMETER_NAMES),
)
_SLOW_NA_START = _FAST_STATIC_SLICE.stop
_SLOW_NA_SLICE = slice(
    _SLOW_NA_START,
    _SLOW_NA_START + len(DIRECT_SLOW_K_PARAMETER_NAMES),
)
_SLOW_NA_G_INDEX = _SLOW_NA_SLICE.stop
_SLOW_K_START = _SLOW_NA_G_INDEX + 1
_SLOW_K_SLICE = slice(
    _SLOW_K_START,
    _SLOW_K_START + len(DIRECT_SLOW_K_PARAMETER_NAMES),
)
_SLOW_K_G_INDEX = _SLOW_K_SLICE.stop
_FAST_G_INDICES = (
    _FAST_STATIC_START,
    _FAST_STATIC_START + 1,
)


@dataclass(frozen=True, order=True)
class EffectiveTopology:
    """Integer powers for the four effective current components."""

    fast_na_activation: int = 3
    fast_na_inactivation: int = 1
    fast_k_activation: int = 4
    slow_na_activation: int = 0
    slow_na_inactivation: int = 0
    slow_k_activation: int = 0
    slow_k_inactivation: int = 0

    def __post_init__(self) -> None:
        powers = asdict(self)
        if any(value not in (0, 1, 2, 3, 4) for value in powers.values()):
            raise ValueError("Gate powers must belong to {0, 1, 2, 3, 4}")
        if self.fast_na_activation == 0:
            raise ValueError("Fast Na requires an activation gate")
        if self.fast_k_activation == 0:
            raise ValueError("Fast K requires an activation gate")
        if bool(self.slow_na_activation) != bool(
            self.slow_na_inactivation
        ):
            raise ValueError(
                "Slow Na requires both shared activation and recovery"
            )
        if self.slow_k_inactivation != 0:
            raise ValueError(
                "This model rung has no independent slow-K inactivation state"
            )

    @property
    def has_slow_na(self) -> bool:
        return self.slow_na_activation > 0

    @property
    def has_slow_k(self) -> bool:
        return self.slow_k_activation > 0

    @property
    def active_state_count(self) -> int:
        return 3 + int(self.has_slow_na) + int(self.has_slow_k)

    @property
    def active_component_count(self) -> int:
        return 2 + int(self.has_slow_na) + int(self.has_slow_k)

    @property
    def label(self) -> str:
        label = (
            f"NaF_m{self.fast_na_activation}"
            f"h{self.fast_na_inactivation}"
            f"_KF_n{self.fast_k_activation}"
        )
        if self.has_slow_na:
            label += (
                f"_NaS_m{self.slow_na_activation}"
                f"h{self.slow_na_inactivation}"
            )
        if self.has_slow_k:
            label += f"_KS_n{self.slow_k_activation}"
        return label

    def to_mapping(self) -> dict[str, int]:
        return {
            f"topology__{name}": int(value)
            for name, value in asdict(self).items()
        }


CANONICAL_TOPOLOGY = EffectiveTopology()


def fast_topology_candidates() -> tuple[EffectiveTopology, ...]:
    """Enumerate the small discrete fast-current topology family."""
    return tuple(
        EffectiveTopology(
            fast_na_activation=na_activation,
            fast_na_inactivation=na_inactivation,
            fast_k_activation=k_activation,
        )
        for na_activation in (1, 2, 3, 4)
        for na_inactivation in (1, 2)
        for k_activation in (1, 2, 3, 4)
    )


def representative_fast_topologies() -> tuple[EffectiveTopology, ...]:
    """Return a factorial spine plus cross-terms around canonical HH."""
    candidates = {
        *(
            EffectiveTopology(
                fast_na_activation=activation,
                fast_na_inactivation=1,
                fast_k_activation=4,
            )
            for activation in (1, 2, 3, 4)
        ),
        *(
            EffectiveTopology(
                fast_na_activation=activation,
                fast_na_inactivation=2,
                fast_k_activation=4,
            )
            for activation in (2, 3, 4)
        ),
        *(
            EffectiveTopology(
                fast_na_activation=3,
                fast_na_inactivation=1,
                fast_k_activation=activation,
            )
            for activation in (1, 2, 3, 4)
        ),
        *(
            EffectiveTopology(
                fast_na_activation=na_activation,
                fast_na_inactivation=na_inactivation,
                fast_k_activation=3,
            )
            for na_activation in (2, 4)
            for na_inactivation in (1, 2)
        ),
    }
    return tuple(sorted(candidates))


def neighboring_fast_topologies(
    topology: EffectiveTopology,
) -> tuple[EffectiveTopology, ...]:
    """Return one-integer moves in the three fast topology coordinates."""
    output = set()
    for field in (
        "fast_na_activation",
        "fast_na_inactivation",
        "fast_k_activation",
    ):
        value = getattr(topology, field)
        allowed = (
            (1, 2)
            if field == "fast_na_inactivation"
            else (1, 2, 3, 4)
        )
        for neighbor in (value - 1, value + 1):
            if neighbor in allowed:
                output.add(replace(topology, **{field: neighbor}))
    return tuple(sorted(output))


def slow_na_topology_candidates(
    fast: EffectiveTopology,
) -> tuple[EffectiveTopology, ...]:
    return tuple(
        replace(
            fast,
            slow_na_activation=activation,
            slow_na_inactivation=inactivation,
        )
        for activation in (1, 2, 3, 4)
        for inactivation in (1, 2)
    )


def slow_k_topology_candidates(
    fast: EffectiveTopology,
) -> tuple[EffectiveTopology, ...]:
    return tuple(
        replace(fast, slow_k_activation=activation)
        for activation in (1, 2, 3, 4)
    )


@dataclass(frozen=True)
class EffectiveComponentModel:
    base: DecodedLadderModel
    topology: EffectiveTopology
    slow_na_kinetics: DirectSlowKinetics
    slow_k_kinetics: DirectSlowKinetics
    gslow_na_ms_cm2: float
    gslow_k_ms_cm2: float


@dataclass(frozen=True)
class EffectiveFitConfig:
    fast_parent_count: int = 3
    mechanism_parent_count: int = 2
    joint_seed_count: int = 2
    fast_population_size: int = 6
    fast_generations: int = 2
    mechanism_population_size: int = 8
    mechanism_generations: int = 2
    joint_population_size: int = 10
    joint_generations: int = 2
    workers: int = 1
    seed: int = 42
    sigma: float = 0.30
    train_duration_ms: float = 500.0
    phase_duration_ms: float = 300.0
    screen_train_duration_ms: float = 250.0
    screen_phase_duration_ms: float = 220.0
    screen_protocol_count: int = 1
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
    validation_weight: float = 0.25
    extra_state_penalty: float = 2.0
    mechanism_minimum_relative_gain: float = 0.05
    trust_fraction: float = 0.20
    exhaustive_fast_topologies: bool = False
    invalid_score: float = 10_000.0


@dataclass(frozen=True)
class EffectiveEvaluation:
    valid: bool
    reason: str
    objective_total: float
    selection_score: float
    shape_loss: float = float("nan")
    scale_loss: float = float("nan")
    firing_pattern_loss: float = float("nan")
    spike_count_loss: float = float("nan")
    waveform_loss: float = float("nan")
    validation_loss: float = float("nan")
    regularity_loss: float = float("nan")
    complexity_penalty: float = float("nan")
    rheobase_constraint_loss: float = float("nan")
    model_rheobase_pa: float = float("nan")
    phase_scores: PhaseShapeScores | None = None
    validation_phase_scores: PhaseShapeScores | None = None
    protocol_scores: Mapping[str, PhaseShapeScores] | None = None
    protocol_features: Mapping[str, Mapping[str, float]] | None = None
    traces: Mapping[str, Trace] | None = None
    cycles: Mapping[str, PhaseCycle] | None = None
    physical: Mapping[str, float] | None = None


@dataclass(frozen=True)
class EffectiveCandidate:
    label: str
    stage: str
    parent_label: str
    topology: EffectiveTopology
    parameters: np.ndarray
    evaluation: EffectiveEvaluation
    eligible: bool = True


@dataclass(frozen=True)
class EffectiveFitResult:
    target: CellOptimizationTarget
    biological_cycles: Mapping[str, PhaseCycle]
    biological_features: Mapping[str, Mapping[str, float]]
    parent: EffectiveCandidate
    candidates: tuple[EffectiveCandidate, ...]
    selected: EffectiveCandidate
    topology_screen: pd.DataFrame
    history: pd.DataFrame
    config: EffectiveFitConfig
    shape_config: PhaseShapeConfig


def _slow_na_initial() -> np.ndarray:
    voltage = np.asarray(DIRECT_VOLTAGE_KNOTS_MV, dtype=float)
    unavailable_logits = (voltage + 48.0) / 9.0
    increments = np.maximum(np.diff(unavailable_logits), 1e-6)
    values = np.asarray(
        (
            unavailable_logits[0],
            *np.log(increments),
            *([math.log(800.0)] * len(voltage)),
        ),
        dtype=float,
    )
    lower, upper = direct_slow_k_parameter_bounds()
    return np.clip(values, lower + 1e-9, upper - 1e-9)


def effective_parameter_bounds(
    gna_gk_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    kinetic_lower, kinetic_upper = direct_kinetic_parameter_bounds()
    static_lower, static_upper = _static_bounds(gna_gk_factor)
    slow_lower, slow_upper = direct_slow_k_parameter_bounds()
    return (
        np.concatenate(
            (
                kinetic_lower,
                static_lower,
                slow_lower,
                (math.log(0.001),),
                slow_lower,
                (math.log(0.001),),
            )
        ),
        np.concatenate(
            (
                kinetic_upper,
                static_upper,
                slow_upper,
                (math.log(20.0),),
                slow_upper,
                (math.log(20.0),),
            )
        ),
    )


def load_effective_parent_vector(
    parameter_path: str | Path,
    stage: str,
) -> np.ndarray:
    table = pd.read_csv(parameter_path)
    selected = table.loc[table["stage"].eq(stage)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {stage!r} parent row in {parameter_path}"
        )
    row = selected.iloc[0]
    return np.concatenate(
        (
            row.loc[
                list(DIRECT_KINETIC_PARAMETER_NAMES)
            ].to_numpy(dtype=float),
            row.loc[
                list(DIRECT_STATIC_PARAMETER_NAMES)
            ].to_numpy(dtype=float),
            _slow_na_initial(),
            (math.log(0.05),),
            direct_slow_k_initial(),
            (math.log(0.05),),
        )
    )


def _pow(gate: float, exponent: int) -> float:
    return 1.0 if exponent == 0 else float(gate) ** exponent


def _component_currents(
    voltage_mv: float,
    m_gate: float,
    h_gate: float,
    n_gate: float,
    slow_na_availability: float,
    slow_k_activation: float,
    model: EffectiveComponentModel,
    compartment: str,
) -> tuple[float, float, float, float, float]:
    base = model.base
    biophysics = base.biophysics
    topology = model.topology
    gna = biophysics.gna_ms_cm2
    eleak = biophysics.eleak_mv
    if compartment == "ais":
        gna *= base.ais_gna_multiplier
        eleak = base.ais_eleak_mv
    fast_na = (
        gna
        * _pow(m_gate, topology.fast_na_activation)
        * _pow(h_gate, topology.fast_na_inactivation)
        * (voltage_mv - biophysics.ena_mv)
    )
    fast_k = (
        biophysics.gk_ms_cm2
        * _pow(n_gate, topology.fast_k_activation)
        * (voltage_mv - biophysics.ek_mv)
    )
    slow_na = 0.0
    slow_k = 0.0
    if compartment == "soma" and topology.has_slow_na:
        slow_na = (
            model.gslow_na_ms_cm2
            * _pow(m_gate, topology.slow_na_activation)
            * _pow(
                slow_na_availability,
                topology.slow_na_inactivation,
            )
            * (voltage_mv - biophysics.ena_mv)
        )
    if compartment == "soma" and topology.has_slow_k:
        slow_k = (
            model.gslow_k_ms_cm2
            * _pow(slow_k_activation, topology.slow_k_activation)
            * (voltage_mv - biophysics.ek_mv)
        )
    leak = biophysics.gleak_ms_cm2 * (voltage_mv - eleak)
    return (
        float(fast_na + slow_na),
        float(fast_k + slow_k),
        float(leak),
        float(fast_na + fast_k + slow_na + slow_k + leak),
        float(slow_na + slow_k),
    )


def decode_effective_model(
    parameters: Sequence[float],
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
) -> EffectiveComponentModel:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(EFFECTIVE_PARAMETER_NAMES),):
        raise ValueError(f"Invalid effective-component vector: {values.shape}")
    base = decode_direct_model(
        values[_FAST_KINETIC_SLICE],
        values[_FAST_STATIC_SLICE],
        target,
    )
    slow_na = DirectSlowKinetics.from_parameters(
        values[_SLOW_NA_SLICE]
    )
    slow_k = DirectSlowKinetics.from_parameters(values[_SLOW_K_SLICE])
    gslow_na = math.exp(float(values[_SLOW_NA_G_INDEX]))
    gslow_k = math.exp(float(values[_SLOW_K_G_INDEX]))
    temporary = EffectiveComponentModel(
        base=base,
        topology=topology,
        slow_na_kinetics=slow_na,
        slow_k_kinetics=slow_k,
        gslow_na_ms_cm2=gslow_na,
        gslow_k_ms_cm2=gslow_k,
    )

    def active_current(voltage_mv: float) -> float:
        m_gate, h_gate, n_gate = base.kinetics.steady_state(voltage_mv)
        unavailable, _ = slow_na.gate_curves_scalar(voltage_mv)
        slow_k_gate, _ = slow_k.gate_curves_scalar(voltage_mv)
        currents = _component_currents(
            voltage_mv,
            m_gate,
            h_gate,
            n_gate,
            1.0 - unavailable,
            slow_k_gate,
            temporary,
            "soma",
        )
        return float(currents[3] - currents[2])

    gleak, eleak = passive_leak_from_active_current(
        target.passive,
        active_current,
    )
    biophysics = replace(
        base.biophysics,
        gleak_ms_cm2=float(gleak),
        eleak_mv=float(eleak),
    )
    base = replace(base, biophysics=biophysics)
    temporary = replace(temporary, base=base)
    resting_voltage = target.passive.resting_voltage_mv
    ais_kinetics = base.ais_kinetics or base.kinetics
    am_gate, ah_gate, an_gate = ais_kinetics.steady_state(
        resting_voltage
    )
    ais_active = _component_currents(
        resting_voltage,
        am_gate,
        ah_gate,
        an_gate,
        1.0,
        0.0,
        temporary,
        "ais",
    )[3]
    ais_active -= biophysics.gleak_ms_cm2 * (
        resting_voltage - base.ais_eleak_mv
    )
    ais_eleak = (
        resting_voltage + ais_active / biophysics.gleak_ms_cm2
    )
    return replace(
        temporary,
        base=replace(base, ais_eleak_mv=float(ais_eleak)),
    )


def effective_initial_state(
    model: EffectiveComponentModel,
    target: CellOptimizationTarget,
) -> np.ndarray:
    voltage = target.passive.resting_voltage_mv
    sm, sh, sn = model.base.kinetics.steady_state(voltage)
    unavailable, _ = model.slow_na_kinetics.gate_curves_scalar(voltage)
    slow_k, _ = model.slow_k_kinetics.gate_curves_scalar(voltage)
    ais_kinetics = model.base.ais_kinetics or model.base.kinetics
    am, ah, an = ais_kinetics.steady_state(voltage)
    return np.asarray(
        (
            voltage,
            sm,
            sh,
            sn,
            1.0 - unavailable,
            slow_k,
            voltage,
            am,
            ah,
            an,
        ),
        dtype=float,
    )


def _voltage_rhs(
    time_ms: float,
    state: np.ndarray,
    model: EffectiveComponentModel,
    config: SimulationConfig,
) -> np.ndarray:
    base = model.base
    biophysics = base.biophysics
    soma_current = _component_currents(
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
        model,
        "soma",
    )[3]
    ais_current = _component_currents(
        state[6],
        state[7],
        state[8],
        state[9],
        1.0,
        0.0,
        model,
        "ais",
    )[3]
    soma_area = biophysics.membrane_area_um2 * (
        1.0 - base.ais_area_fraction
    )
    ais_area = (
        biophysics.membrane_area_um2 * base.ais_area_fraction
    )
    applied_pa = config.stimulus.pa_value(time_ms, biophysics)
    coupling_pa = base.coupling_ns * (state[0] - state[6])
    return np.asarray(
        (
            (
                applied_pa / (soma_area * 0.01)
                - soma_current
                - coupling_pa / (soma_area * 0.01)
            )
            / biophysics.capacitance_uf_cm2,
            (
                -ais_current + coupling_pa / (ais_area * 0.01)
            )
            / biophysics.capacitance_uf_cm2,
        ),
        dtype=float,
    )


def _gate_half_step(
    state: np.ndarray,
    model: EffectiveComponentModel,
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
    unavailable_inf, slow_na_tau = (
        model.slow_na_kinetics.gate_curves_scalar(
            float(state[0]),
            config.temperature_c,
        )
    )
    slow_na_inf = 1.0 - unavailable_inf
    slow_na_decay = math.exp(
        max(-80.0, -duration_ms / max(slow_na_tau, 1e-12))
    )
    updated[4] = slow_na_inf + (
        float(state[4]) - slow_na_inf
    ) * slow_na_decay
    slow_k_inf, slow_k_tau = model.slow_k_kinetics.gate_curves_scalar(
        float(state[0]),
        config.temperature_c,
    )
    slow_k_decay = math.exp(
        max(-80.0, -duration_ms / max(slow_k_tau, 1e-12))
    )
    updated[5] = slow_k_inf + (
        float(state[5]) - slow_k_inf
    ) * slow_k_decay
    updated[7:10] = _rush_larsen_gate_update(
        float(state[6]),
        state[7:10],
        model.base.ais_kinetics or model.base.kinetics,
        model.base,
        config.temperature_c,
        duration_ms,
    )
    return updated


def _step(
    time_ms: float,
    state: np.ndarray,
    model: EffectiveComponentModel,
    config: SimulationConfig,
) -> np.ndarray:
    dt_ms = config.dt_ms
    half = _gate_half_step(state, model, config, dt_ms / 2.0)
    voltage_indices = np.asarray((0, 6), dtype=int)

    def rhs(rhs_time: float, voltages: np.ndarray) -> np.ndarray:
        trial = half.copy()
        trial[voltage_indices] = voltages
        return _voltage_rhs(rhs_time, trial, model, config)

    voltage = half[voltage_indices]
    k1 = rhs(time_ms, voltage)
    k2 = rhs(time_ms + dt_ms / 2.0, voltage + dt_ms * k1 / 2.0)
    k3 = rhs(time_ms + dt_ms / 2.0, voltage + dt_ms * k2 / 2.0)
    k4 = rhs(time_ms + dt_ms, voltage + dt_ms * k3)
    half[voltage_indices] = voltage + dt_ms * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0
    updated = _gate_half_step(half, model, config, dt_ms / 2.0)
    gate_indices = np.asarray((1, 2, 3, 4, 5, 7, 8, 9), dtype=int)
    if (
        not np.all(np.isfinite(updated))
        or np.any(np.abs(updated[voltage_indices]) > 200.0)
        or np.any(updated[gate_indices] < -0.05)
        or np.any(updated[gate_indices] > 1.05)
    ):
        raise SimulationError(
            f"Effective-component integration diverged at "
            f"{time_ms + dt_ms:.4f} ms"
        )
    updated[gate_indices] = np.clip(
        updated[gate_indices],
        0.0,
        1.0,
    )
    return updated


def simulate_effective_components(
    model: EffectiveComponentModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> Trace:
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, 10), dtype=float)
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
    soma_area = biophysics.membrane_area_um2 * (
        1.0 - model.base.ais_area_fraction
    )
    applied_density = applied_pa / (soma_area * 0.01)
    current_rows = np.asarray(
        [
            _component_currents(
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                state[5],
                model,
                "soma",
            )
            for state in states
        ],
        dtype=float,
    )
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
        ionic_current_ua_cm2=current_rows[:, 3],
        ionic_current_pa=biophysics.current_density_to_pa(
            current_rows[:, 3]
        ),
        sodium_current_ua_cm2=current_rows[:, 0],
        potassium_current_ua_cm2=current_rows[:, 1],
        leak_current_ua_cm2=current_rows[:, 2],
        dvdt_mv_ms=dvdt,
    )


def _elicits_spike(
    model: EffectiveComponentModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> bool:
    state = np.asarray(initial_state, dtype=float)
    n_steps = int(round(config.duration_ms / config.dt_ms))
    for index in range(n_steps):
        previous = float(state[0])
        state = _step(
            index * config.dt_ms,
            state,
            model,
            config,
        )
        if previous < 0.0 <= state[0]:
            return True
    return False


def find_rheobase_effective_components(
    model: EffectiveComponentModel,
    initial_state: np.ndarray,
    screen_config: BiologicalScreenConfig,
    duration_ms: float,
    tolerance_pa: float,
) -> tuple[float, float, int]:
    maximum = float(screen_config.rheobase_currents_pa[-1])

    def spikes(current_pa: float) -> bool:
        simulation = step_simulation_config(
            current_pa,
            min(duration_ms, screen_config.rheobase_step_ms),
            screen_config,
            model.base.biophysics,
            float(initial_state[0]),
        )
        return _elicits_spike(model, simulation, initial_state)

    if spikes(0.0):
        return 0.0, 0.0, 0
    if not spikes(maximum):
        raise SimulationError("No spike within the rheobase search range")
    lower = 0.0
    upper = maximum
    iterations = 0
    while upper - lower > tolerance_pa and iterations < 20:
        midpoint = 0.5 * (lower + upper)
        if spikes(midpoint):
            upper = midpoint
        else:
            lower = midpoint
        iterations += 1
    return float(lower), float(upper), iterations


def _phase_losses(
    scores: PhaseShapeScores,
    config: EffectiveFitConfig,
) -> tuple[float, float]:
    shape = float(
        3.0 * scores.normalized_shape
        + scores.slope_shape
        + 2.5 * scores.concavity
    )
    scale = float(
        1.5 * scores.physical_curve
        + 2.0 * scores.velocity_extent
        + config.physical_constraint_weight
        * scores.physical_constraint_loss
    )
    return shape, scale


def _protocol_behavior_loss(
    features: Mapping[str, Mapping[str, float]],
    biological: Mapping[str, Mapping[str, float]],
    protocols,
    config: EffectiveFitConfig,
) -> tuple[float, float, float]:
    firing = _mean_feature_group(
        features,
        biological,
        protocols,
        "firing_pattern",
    )
    count = _mean_spike_count_loss(features, biological, protocols)
    waveform = 0.5 * (
        _mean_feature_group(
            features,
            biological,
            protocols,
            "spike_shape",
        )
        + _mean_feature_group(
            features,
            biological,
            protocols,
            "spike_dynamics",
        )
    )
    return firing, count, waveform


def evaluate_effective_candidate(
    parameters: Sequence[float],
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float = float("inf"),
    include_validation: bool = False,
    include_artifacts: bool = False,
    protocol_limit: int | None = None,
    train_duration_ms: float | None = None,
    phase_duration_ms: float | None = None,
) -> EffectiveEvaluation:
    try:
        values = np.asarray(parameters, dtype=float)
        lower, upper = effective_parameter_bounds()
        if (
            values.shape != lower.shape
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError("Effective candidate is outside fitted bounds")
        model = decode_effective_model(values, topology, target)
        if (
            not -120.0 <= model.base.biophysics.eleak_mv <= -20.0
            or not -120.0 <= model.base.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        voltage = np.linspace(-100.0, 60.0, 161)
        for enabled, kinetics in (
            (topology.has_slow_na, model.slow_na_kinetics),
            (topology.has_slow_k, model.slow_k_kinetics),
        ):
            if not enabled:
                continue
            _, tau = kinetics.gate_curves(
                voltage,
                target.temperature_c,
            )
            if float(np.min(tau)) < 1.0:
                raise SimulationError("Slow gate tau fell below 1 ms")
        state = effective_initial_state(model, target)
        rest_rhs = _voltage_rhs(0.0, state, model, SimulationConfig(
            duration_ms=1.0,
            dt_ms=screen_config.dt_ms,
            initial_voltage_mv=target.passive.resting_voltage_mv,
            stimulus=Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=0.0,
                start_ms=0.0,
                end_ms=1.0,
            ),
            conductances=model.base.biophysics,
            temperature_c=target.temperature_c,
        ))
        if float(np.max(np.abs(rest_rhs))) > 1e-6:
            raise SimulationError("Passive reanchoring did not preserve rest")
        _, upper_rheobase, _ = find_rheobase_effective_components(
            model,
            state,
            screen_config,
            fit_config.rheobase_probe_duration_ms,
            fit_config.rheobase_probe_tolerance_pa,
        )
        training_protocols = list(target.training_protocols)
        if protocol_limit is not None:
            training_protocols = training_protocols[:protocol_limit]
        protocols = list(training_protocols)
        if include_validation:
            protocols.extend(target.validation_protocols)
        traces: dict[str, Trace] = {}
        cycles: dict[str, PhaseCycle] = {}
        scores: dict[str, PhaseShapeScores] = {}
        features: dict[str, Mapping[str, float]] = {}
        for protocol in protocols:
            phase_duration = min(
                protocol.duration_ms,
                phase_duration_ms or fit_config.phase_duration_ms,
            )
            phase_current = protocol.rheobase_factor * upper_rheobase
            phase_simulation = step_simulation_config(
                phase_current,
                phase_duration,
                screen_config,
                model.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            phase_trace = simulate_effective_components(
                model,
                phase_simulation,
                state,
            )
            phase_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=phase_current,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + phase_duration,
            )
            cycle = extract_phase_cycle(
                phase_trace.time_ms,
                phase_trace.voltage_mv,
                phase_stimulus,
                shape_config,
            )
            scores[protocol.name] = compare_phase_cycles(
                cycle,
                biological_cycles_target[protocol.name],
                shape_config,
            )
            train_duration = min(
                protocol.duration_ms,
                train_duration_ms or fit_config.train_duration_ms,
            )
            train_simulation = step_simulation_config(
                protocol.current_pa,
                train_duration,
                screen_config,
                model.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            train_trace = simulate_effective_components(
                model,
                train_simulation,
                state,
            )
            train_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=protocol.current_pa,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + train_duration,
            )
            features[protocol.name] = extract_observed_features(
                train_trace,
                train_stimulus,
                FeatureConfig(min_spikes=1),
            )
            if include_artifacts:
                traces[protocol.name] = train_trace
                cycles[protocol.name] = cycle
        training_scores = {
            protocol.name: scores[protocol.name]
            for protocol in training_protocols
        }
        phase_scores = mean_phase_scores(training_scores)
        shape_loss, scale_loss = _phase_losses(
            phase_scores,
            fit_config,
        )
        firing, count, waveform = _protocol_behavior_loss(
            features,
            biological_features_target,
            training_protocols,
            fit_config,
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
        rheobase_constraint = float(rheobase_excess**2)
        guard_limit = (
            (1.0 + fit_config.shape_guard_relative_tolerance)
            * reference_shape_loss
        )
        shape_guard = (
            max(0.0, shape_loss / max(guard_limit, 1e-12) - 1.0)
            ** 2
            if np.isfinite(reference_shape_loss)
            else 0.0
        )
        smoothness = 0.0
        if topology.has_slow_na:
            smoothness += model.slow_na_kinetics.smoothness_penalty()
        if topology.has_slow_k:
            smoothness += model.slow_k_kinetics.smoothness_penalty()
        regularity = fit_config.smoothness_weight * smoothness
        objective = float(
            shape_loss
            + scale_loss
            + fit_config.firing_pattern_weight * firing
            + fit_config.spike_count_weight * count
            + fit_config.waveform_weight * waveform
            + fit_config.rheobase_constraint_weight
            * rheobase_constraint
            + fit_config.shape_guard_weight * shape_guard
            + regularity
        )
        validation_loss = 0.0
        validation_phase = None
        if include_validation and target.validation_protocols:
            validation_phase = mean_phase_scores(
                {
                    protocol.name: scores[protocol.name]
                    for protocol in target.validation_protocols
                }
            )
            validation_shape, validation_scale = _phase_losses(
                validation_phase,
                fit_config,
            )
            val_firing, val_count, val_waveform = (
                _protocol_behavior_loss(
                    features,
                    biological_features_target,
                    target.validation_protocols,
                    fit_config,
                )
            )
            validation_loss = float(
                validation_shape
                + validation_scale
                + fit_config.firing_pattern_weight * val_firing
                + fit_config.spike_count_weight * val_count
                + fit_config.waveform_weight * val_waveform
            )
        complexity = fit_config.extra_state_penalty * (
            topology.active_state_count
            - CANONICAL_TOPOLOGY.active_state_count
        )
        selection_score = float(
            objective
            + fit_config.validation_weight * validation_loss
            + complexity
        )
        physical = model.base.biophysics.to_physical_mapping()
        physical.update(
            {
                "extension__gslow_na_ms_cm2": (
                    model.gslow_na_ms_cm2
                    if topology.has_slow_na
                    else 0.0
                ),
                "extension__gslow_k_ms_cm2": (
                    model.gslow_k_ms_cm2
                    if topology.has_slow_k
                    else 0.0
                ),
                "extension__active_state_count": (
                    topology.active_state_count
                ),
                "extension__active_component_count": (
                    topology.active_component_count
                ),
            }
        )
        return EffectiveEvaluation(
            valid=True,
            reason="accepted",
            objective_total=objective,
            selection_score=selection_score,
            shape_loss=shape_loss,
            scale_loss=scale_loss,
            firing_pattern_loss=firing,
            spike_count_loss=count,
            waveform_loss=waveform,
            validation_loss=validation_loss,
            regularity_loss=regularity,
            complexity_penalty=float(complexity),
            rheobase_constraint_loss=rheobase_constraint,
            model_rheobase_pa=float(upper_rheobase),
            phase_scores=phase_scores,
            validation_phase_scores=validation_phase,
            protocol_scores=scores,
            protocol_features=features,
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
        return EffectiveEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            objective_total=fit_config.invalid_score,
            selection_score=fit_config.invalid_score,
        )


def _candidate_evaluation_payload(
    parameters: np.ndarray,
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
    protocol_limit: int | None,
    train_duration_ms: float | None,
    phase_duration_ms: float | None,
) -> EffectiveEvaluation:
    return evaluate_effective_candidate(
        parameters,
        topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        fit_config,
        shape_config,
        reference_shape_loss,
        protocol_limit=protocol_limit,
        train_duration_ms=train_duration_ms,
        phase_duration_ms=phase_duration_ms,
    )


def _evaluate_candidate_batch(
    parameter_vectors: Sequence[np.ndarray],
    topologies: Sequence[EffectiveTopology],
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_losses: Sequence[float],
    protocol_limit: int | None,
    train_duration_ms: float | None,
    phase_duration_ms: float | None,
) -> list[EffectiveEvaluation]:
    count = len(parameter_vectors)
    if len(topologies) != count or len(reference_shape_losses) != count:
        raise ValueError("Candidate batch columns have inconsistent lengths")
    if fit_config.workers <= 1 or count <= 1:
        return [
            _candidate_evaluation_payload(
                parameters,
                topology,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                fit_config,
                shape_config,
                reference_shape_loss,
                protocol_limit,
                train_duration_ms,
                phase_duration_ms,
            )
            for parameters, topology, reference_shape_loss in zip(
                parameter_vectors,
                topologies,
                reference_shape_losses,
            )
        ]
    try:
        with ProcessPoolExecutor(
            max_workers=fit_config.workers
        ) as executor:
            return list(
                executor.map(
                    _candidate_evaluation_payload,
                    parameter_vectors,
                    topologies,
                    [target] * count,
                    [biological_cycles_target] * count,
                    [biological_features_target] * count,
                    [screen_config] * count,
                    [fit_config] * count,
                    [shape_config] * count,
                    reference_shape_losses,
                    [protocol_limit] * count,
                    [train_duration_ms] * count,
                    [phase_duration_ms] * count,
                )
            )
    except (OSError, PermissionError) as error:
        warnings.warn(
            f"Parallel screen unavailable ({error}); using one worker.",
            RuntimeWarning,
        )
        serial_config = replace(fit_config, workers=1)
        return _evaluate_candidate_batch(
            parameter_vectors,
            topologies,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            serial_config,
            shape_config,
            reference_shape_losses,
            protocol_limit,
            train_duration_ms,
            phase_duration_ms,
        )


def _conductance_normalized_topology(
    parent_parameters: np.ndarray,
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
    reference_voltage_mv: float = -20.0,
) -> np.ndarray:
    values = np.asarray(parent_parameters, dtype=float).copy()
    parent = decode_direct_model(
        values[_FAST_KINETIC_SLICE],
        values[_FAST_STATIC_SLICE],
        target,
    )
    m_gate, h_gate, n_gate = parent.kinetics.steady_state(
        reference_voltage_mv
    )
    canonical_na = _pow(m_gate, 3) * h_gate
    candidate_na = (
        _pow(m_gate, topology.fast_na_activation)
        * _pow(h_gate, topology.fast_na_inactivation)
    )
    canonical_k = _pow(n_gate, 4)
    candidate_k = _pow(n_gate, topology.fast_k_activation)
    values[_FAST_G_INDICES[0]] += math.log(
        max(canonical_na, 1e-12) / max(candidate_na, 1e-12)
    )
    values[_FAST_G_INDICES[1]] += math.log(
        max(canonical_k, 1e-12) / max(candidate_k, 1e-12)
    )
    lower, upper = effective_parameter_bounds()
    return np.clip(values, lower + 1e-9, upper - 1e-9)


def _evaluation_row(
    candidate: EffectiveCandidate,
) -> dict[str, object]:
    evaluation = candidate.evaluation
    row: dict[str, object] = {
        "label": candidate.label,
        "stage": candidate.stage,
        "parent_label": candidate.parent_label,
        "eligible": candidate.eligible,
        **candidate.topology.to_mapping(),
        "topology": candidate.topology.label,
        "valid": evaluation.valid,
        "reason": evaluation.reason,
        "objective_total": evaluation.objective_total,
        "selection_score": evaluation.selection_score,
        "shape_loss": evaluation.shape_loss,
        "scale_loss": evaluation.scale_loss,
        "firing_pattern_loss": evaluation.firing_pattern_loss,
        "spike_count_loss": evaluation.spike_count_loss,
        "waveform_loss": evaluation.waveform_loss,
        "validation_loss": evaluation.validation_loss,
        "complexity_penalty": evaluation.complexity_penalty,
        "model_rheobase_pa": evaluation.model_rheobase_pa,
        "rheobase_constraint_loss": (
            evaluation.rheobase_constraint_loss
        ),
    }
    if evaluation.phase_scores is not None:
        row.update(
            {
                f"phase__{name}": value
                for name, value in (
                    evaluation.phase_scores.to_mapping().items()
                )
            }
        )
    if evaluation.physical is not None:
        row.update(evaluation.physical)
    return row


def _optimization_payload(
    unbounded: Sequence[float],
    variable_indices: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    fixed_parameters: np.ndarray,
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
    protocol_limit: int | None,
    train_duration_ms: float | None,
    phase_duration_ms: float | None,
) -> tuple[np.ndarray, EffectiveEvaluation]:
    variables = _bounded_parameters(
        unbounded,
        variable_lower,
        variable_upper,
    )
    parameters = np.asarray(fixed_parameters, dtype=float).copy()
    parameters[variable_indices] = variables
    evaluation = evaluate_effective_candidate(
        parameters,
        topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        fit_config,
        shape_config,
        reference_shape_loss,
        include_validation=False,
        include_artifacts=False,
        protocol_limit=protocol_limit,
        train_duration_ms=train_duration_ms,
        phase_duration_ms=phase_duration_ms,
    )
    return parameters, evaluation


def _optimize_variables(
    center: np.ndarray,
    topology: EffectiveTopology,
    variable_indices: Sequence[int],
    lower: np.ndarray,
    upper: np.ndarray,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float,
    population_size: int,
    generations: int,
    seed: int,
    stage: str,
    protocol_limit: int | None = None,
    train_duration_ms: float | None = None,
    phase_duration_ms: float | None = None,
) -> tuple[np.ndarray, EffectiveEvaluation, pd.DataFrame]:
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError(
            "Effective-component fitting requires DEAP"
        ) from error
    if population_size < 4:
        raise ValueError("CMA population size must be at least four")
    indices = np.asarray(variable_indices, dtype=int)
    variable_lower = np.asarray(lower, dtype=float)[indices]
    variable_upper = np.asarray(upper, dtype=float)[indices]
    clipped_center = np.asarray(center, dtype=float).copy()
    clipped_center[indices] = np.clip(
        clipped_center[indices],
        variable_lower + 1e-9,
        variable_upper - 1e-9,
    )
    fitness_name = "FitnessEffectiveComponentsCMA"
    individual_name = "IndividualEffectiveComponentsCMA"
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
        centroid=_unbounded_parameters(
            clipped_center[indices],
            variable_lower,
            variable_upper,
        ),
        sigma=fit_config.sigma,
        lambda_=population_size,
    )
    center_evaluation = evaluate_effective_candidate(
        clipped_center,
        topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        fit_config,
        shape_config,
        reference_shape_loss,
        protocol_limit=protocol_limit,
        train_duration_ms=train_duration_ms,
        phase_duration_ms=phase_duration_ms,
    )
    evaluated: list[tuple[np.ndarray, EffectiveEvaluation]] = [
        (clipped_center, center_evaluation)
    ]
    rows = [
        {
            "stage": stage,
            "generation": -1,
            "individual": -1,
            "topology": topology.label,
            "valid": center_evaluation.valid,
            "objective_total": center_evaluation.objective_total,
            "reason": center_evaluation.reason,
        }
    ]
    executor = None
    mapper = map
    if fit_config.workers > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=fit_config.workers
            )
            mapper = executor.map
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"Parallel evaluation unavailable ({error}); "
                "using one worker.",
                RuntimeWarning,
            )
    try:
        for generation in range(generations):
            population = strategy.generate(individual_type)
            payloads = list(
                mapper(
                    _optimization_payload,
                    population,
                    [indices] * len(population),
                    [variable_lower] * len(population),
                    [variable_upper] * len(population),
                    [clipped_center] * len(population),
                    [topology] * len(population),
                    [target] * len(population),
                    [biological_cycles_target] * len(population),
                    [biological_features_target] * len(population),
                    [screen_config] * len(population),
                    [fit_config] * len(population),
                    [shape_config] * len(population),
                    [reference_shape_loss] * len(population),
                    [protocol_limit] * len(population),
                    [train_duration_ms] * len(population),
                    [phase_duration_ms] * len(population),
                )
            )
            for individual_index, (individual, payload) in enumerate(
                zip(population, payloads)
            ):
                parameters, evaluation = payload
                individual.fitness.values = (
                    evaluation.objective_total,
                )
                evaluated.append((parameters, evaluation))
                rows.append(
                    {
                        "stage": stage,
                        "generation": generation,
                        "individual": individual_index,
                        "topology": topology.label,
                        "valid": evaluation.valid,
                        "objective_total": evaluation.objective_total,
                        "reason": evaluation.reason,
                    }
                )
            strategy.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    valid = [
        item
        for item in evaluated
        if item[1].valid
    ]
    feasible = [
        item
        for item in valid
        if item[1].rheobase_constraint_loss <= 1e-12
    ]
    pool = feasible or valid
    if not pool:
        raise RuntimeError(
            f"Effective {stage} optimization produced no valid model"
        )
    parameters, evaluation = min(
        pool,
        key=lambda item: item[1].objective_total,
    )
    return parameters, evaluation, pd.DataFrame(rows)


def _full_evaluation(
    parameters: np.ndarray,
    topology: EffectiveTopology,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float = float("inf"),
    artifacts: bool = False,
) -> EffectiveEvaluation:
    return evaluate_effective_candidate(
        parameters,
        topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        fit_config,
        shape_config,
        reference_shape_loss,
        include_validation=True,
        include_artifacts=artifacts,
    )


def _candidate_sort_key(
    candidate: EffectiveCandidate,
) -> tuple[float, float]:
    infeasible = float(
        candidate.evaluation.rheobase_constraint_loss > 1e-12
    )
    return infeasible, candidate.evaluation.selection_score


def _extension_is_eligible(
    child: EffectiveEvaluation,
    parent: EffectiveEvaluation,
    minimum_gain: float,
) -> bool:
    if not child.valid or child.objective_total >= parent.objective_total:
        return False
    firing_gain = (
        parent.firing_pattern_loss - child.firing_pattern_loss
    ) / max(parent.firing_pattern_loss, 1e-12)
    count_gain = (
        parent.spike_count_loss - child.spike_count_loss
    ) / max(parent.spike_count_loss, 1e-12)
    return bool(
        firing_gain >= minimum_gain or count_gain >= minimum_gain
    )


def _joint_is_eligible(
    child: EffectiveEvaluation,
    seed: EffectiveEvaluation,
) -> bool:
    """Prevent narrow refinement from destroying train behavior."""
    return bool(
        child.valid
        and child.objective_total <= 1.02 * seed.objective_total
        and child.firing_pattern_loss
        <= 1.10 * seed.firing_pattern_loss
        and child.spike_count_loss <= 1.10 * seed.spike_count_loss
    )


def _mechanism_indices(mechanism: str) -> np.ndarray:
    if mechanism == "slow_na":
        return np.asarray(
            (*range(_SLOW_NA_SLICE.start, _SLOW_NA_SLICE.stop),
             _SLOW_NA_G_INDEX),
            dtype=int,
        )
    if mechanism == "slow_k":
        return np.asarray(
            (*range(_SLOW_K_SLICE.start, _SLOW_K_SLICE.stop),
             _SLOW_K_G_INDEX),
            dtype=int,
        )
    raise ValueError(f"Unknown mechanism: {mechanism}")


def _trust_bounds(
    center: np.ndarray,
    active_indices: Sequence[int],
    config: EffectiveFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    absolute_lower, absolute_upper = effective_parameter_bounds()
    lower = np.asarray(center, dtype=float).copy()
    upper = np.asarray(center, dtype=float).copy()
    log_margin = math.log1p(config.trust_fraction)
    for index in active_indices:
        name = EFFECTIVE_PARAMETER_NAMES[index]
        value = float(center[index])
        if "start_logit" in name:
            margin = max(0.5, config.trust_fraction * abs(value))
        elif "ena_mv" in name or "ek_mv" in name:
            margin = 4.0
        elif "log" in name:
            margin = log_margin
        else:
            margin = config.trust_fraction * max(abs(value), 1.0)
        lower[index] = max(absolute_lower[index], value - margin)
        upper[index] = min(absolute_upper[index], value + margin)
        if upper[index] - lower[index] < 1e-8:
            lower[index] = absolute_lower[index]
            upper[index] = absolute_upper[index]
    return lower, upper


def fit_effective_component_ladder(
    target: CellOptimizationTarget,
    parent_parameters: Sequence[float],
    screen_config: BiologicalScreenConfig,
    config: EffectiveFitConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
    slow_k_seed_parameters: Sequence[float] | None = None,
) -> EffectiveFitResult:
    """Run topology, mechanism, and trust-region selection for one cell."""
    config = config or EffectiveFitConfig()
    shape_config = shape_config or PhaseShapeConfig()
    parent_values = np.asarray(parent_parameters, dtype=float)
    lower, upper = effective_parameter_bounds()
    if parent_values.shape != lower.shape:
        raise ValueError("Parent vector has the wrong shape")
    slow_k_seed = (
        None
        if slow_k_seed_parameters is None
        else np.asarray(slow_k_seed_parameters, dtype=float)
    )
    if slow_k_seed is not None and slow_k_seed.shape != (
        len(DIRECT_SLOW_K_PARAMETER_NAMES) + 1,
    ):
        raise ValueError("Slow-K warm start has the wrong shape")
    biological_cycles_target = biological_phase_cycles(
        target,
        shape_config,
        include_validation=True,
    )
    biological_features_target = biological_window_features(
        target,
        config.train_duration_ms,
    )
    parent_evaluation = _full_evaluation(
        parent_values,
        CANONICAL_TOPOLOGY,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        artifacts=True,
    )
    if not parent_evaluation.valid:
        raise RuntimeError(
            f"Canonical parent is invalid: {parent_evaluation.reason}"
        )
    parent = EffectiveCandidate(
        label="canonical_parent",
        stage="parent",
        parent_label="",
        topology=CANONICAL_TOPOLOGY,
        parameters=parent_values,
        evaluation=parent_evaluation,
    )
    candidates: list[EffectiveCandidate] = [parent]
    histories: list[pd.DataFrame] = []

    topology_rows = []
    screened: list[tuple[EffectiveTopology, np.ndarray, EffectiveEvaluation]] = []
    initial_topologies = (
        fast_topology_candidates()
        if config.exhaustive_fast_topologies
        else representative_fast_topologies()
    )

    def screen_fast_topologies(
        topologies: Sequence[EffectiveTopology],
        screen_stage: str,
    ) -> None:
        new_topologies = [
            topology
            for topology in topologies
            if topology not in {
                existing[0]
                for existing in screened
            }
        ]
        if not new_topologies:
            return
        vectors = [
            _conductance_normalized_topology(
                parent_values,
                topology,
                target,
            )
            for topology in new_topologies
        ]
        evaluations = _evaluate_candidate_batch(
            vectors,
            new_topologies,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            [float("inf")] * len(vectors),
            config.screen_protocol_count,
            config.screen_train_duration_ms,
            config.screen_phase_duration_ms,
        )
        for topology, values, evaluation in zip(
            new_topologies,
            vectors,
            evaluations,
        ):
            screened.append((topology, values, evaluation))
            topology_rows.append(
                {
                    "screen_stage": screen_stage,
                    **topology.to_mapping(),
                    "topology": topology.label,
                    "valid": evaluation.valid,
                    "reason": evaluation.reason,
                    "objective_total": evaluation.objective_total,
                    "shape_loss": evaluation.shape_loss,
                    "scale_loss": evaluation.scale_loss,
                    "firing_pattern_loss": (
                        evaluation.firing_pattern_loss
                    ),
                    "spike_count_loss": evaluation.spike_count_loss,
                    "model_rheobase_pa": (
                        evaluation.model_rheobase_pa
                    ),
                    "rheobase_constraint_loss": (
                        evaluation.rheobase_constraint_loss
                    ),
                }
            )

    screen_fast_topologies(initial_topologies, "representative")
    if not config.exhaustive_fast_topologies:
        initial_valid = [
            item for item in screened if item[2].valid
        ]
        initial_valid.sort(
            key=lambda item: (
                item[2].rheobase_constraint_loss > 1e-12,
                item[2].objective_total,
            )
        )
        local_neighbors = {
            neighbor
            for topology, _, _ in initial_valid[:3]
            for neighbor in neighboring_fast_topologies(topology)
        }
        screen_fast_topologies(
            tuple(sorted(local_neighbors)),
            "winner_neighbors",
        )
    valid_screened = [
        item for item in screened if item[2].valid
    ]
    if not valid_screened:
        raise RuntimeError("No valid fast topology survived screening")
    valid_screened.sort(
        key=lambda item: (
            item[2].rheobase_constraint_loss > 1e-12,
            item[2].objective_total,
        )
    )
    fast_seeds = valid_screened[: config.fast_parent_count]
    fast_candidates: list[EffectiveCandidate] = []
    for index, (topology, values, evaluation) in enumerate(fast_seeds):
        refined_values, _, history = _optimize_variables(
            values,
            topology,
            _FAST_G_INDICES,
            lower,
            upper,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            reference_shape_loss=evaluation.shape_loss,
            population_size=config.fast_population_size,
            generations=config.fast_generations,
            seed=config.seed + 100 + index,
            stage=f"fast_{index}",
            protocol_limit=config.screen_protocol_count,
            train_duration_ms=config.screen_train_duration_ms,
            phase_duration_ms=config.screen_phase_duration_ms,
        )
        histories.append(history)
        full_evaluation = _full_evaluation(
            refined_values,
            topology,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
        )
        fast_candidate = EffectiveCandidate(
            label=f"fast_parent_{index + 1}",
            stage="fast_topology",
            parent_label=parent.label,
            topology=topology,
            parameters=refined_values,
            evaluation=full_evaluation,
        )
        fast_candidates.append(fast_candidate)
        candidates.append(fast_candidate)
    retained_fast = sorted(
        [
            candidate
            for candidate in (parent, *fast_candidates)
            if candidate.evaluation.valid
        ],
        key=_candidate_sort_key,
    )[: config.fast_parent_count]
    if not retained_fast:
        retained_fast = [parent]

    extension_candidates: list[EffectiveCandidate] = []
    for parent_index, fast_parent in enumerate(
        retained_fast[: config.mechanism_parent_count]
    ):
        mechanism_best: dict[str, EffectiveCandidate] = {}
        for mechanism in ("slow_na", "slow_k"):
            topologies = (
                slow_na_topology_candidates(fast_parent.topology)
                if mechanism == "slow_na"
                else slow_k_topology_candidates(fast_parent.topology)
            )
            conductance_index = (
                _SLOW_NA_G_INDEX
                if mechanism == "slow_na"
                else _SLOW_K_G_INDEX
            )
            center_vectors = []
            for topology in topologies:
                values = fast_parent.parameters.copy()
                values[conductance_index] = math.log(0.2)
                center_vectors.append(values)
            center_evaluations = _evaluate_candidate_batch(
                center_vectors,
                topologies,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                [fast_parent.evaluation.shape_loss]
                * len(center_vectors),
                config.screen_protocol_count,
                config.screen_train_duration_ms,
                config.screen_phase_duration_ms,
            )
            center_candidates = [
                (topology, values, evaluation)
                for topology, values, evaluation in zip(
                    topologies,
                    center_vectors,
                    center_evaluations,
                )
                if evaluation.valid
            ]
            center_candidates.sort(
                key=lambda item: (
                    item[2].rheobase_constraint_loss > 1e-12,
                    item[2].objective_total,
                )
            )
            finalist_topologies = [
                item[0] for item in center_candidates[:2]
            ]
            edge_vectors = []
            edge_topologies = []
            for topology in finalist_topologies:
                for conductance in (0.02, 2.0):
                    values = fast_parent.parameters.copy()
                    values[conductance_index] = math.log(conductance)
                    edge_vectors.append(values)
                    edge_topologies.append(topology)
            edge_evaluations = _evaluate_candidate_batch(
                edge_vectors,
                edge_topologies,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                [fast_parent.evaluation.shape_loss]
                * len(edge_vectors),
                config.screen_protocol_count,
                config.screen_train_duration_ms,
                config.screen_phase_duration_ms,
            )
            grid_candidates = center_candidates + [
                (topology, values, evaluation)
                for topology, values, evaluation in zip(
                    edge_topologies,
                    edge_vectors,
                    edge_evaluations,
                )
                if evaluation.valid
            ]
            if mechanism == "slow_k" and slow_k_seed is not None:
                warm_values = fast_parent.parameters.copy()
                warm_values[
                    _SLOW_K_SLICE.start : _SLOW_K_G_INDEX + 1
                ] = slow_k_seed
                warm_topology = replace(
                    fast_parent.topology,
                    slow_k_activation=1,
                )
                warm_evaluation = evaluate_effective_candidate(
                    warm_values,
                    warm_topology,
                    target,
                    biological_cycles_target,
                    biological_features_target,
                    screen_config,
                    config,
                    shape_config,
                    fast_parent.evaluation.shape_loss,
                    protocol_limit=config.screen_protocol_count,
                    train_duration_ms=config.screen_train_duration_ms,
                    phase_duration_ms=config.screen_phase_duration_ms,
                )
                if warm_evaluation.valid:
                    grid_candidates.append(
                        (
                            warm_topology,
                            warm_values,
                            warm_evaluation,
                        )
                    )
            if not grid_candidates:
                continue
            topology, values, screen_evaluation = min(
                grid_candidates,
                key=lambda item: (
                    item[2].rheobase_constraint_loss > 1e-12,
                    item[2].objective_total,
                ),
            )
            refined_values, _, history = _optimize_variables(
                values,
                topology,
                _mechanism_indices(mechanism),
                lower,
                upper,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                fast_parent.evaluation.shape_loss,
                config.mechanism_population_size,
                config.mechanism_generations,
                config.seed + 1000 + 100 * parent_index
                + (0 if mechanism == "slow_na" else 50),
                f"{mechanism}_parent_{parent_index + 1}",
            )
            histories.append(history)
            full_evaluation = _full_evaluation(
                refined_values,
                topology,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                fast_parent.evaluation.shape_loss,
            )
            eligible = _extension_is_eligible(
                full_evaluation,
                fast_parent.evaluation,
                config.mechanism_minimum_relative_gain,
            )
            candidate = EffectiveCandidate(
                label=f"{mechanism}_parent_{parent_index + 1}",
                stage=mechanism,
                parent_label=fast_parent.label,
                topology=topology,
                parameters=refined_values,
                evaluation=full_evaluation,
                eligible=eligible,
            )
            mechanism_best[mechanism] = candidate
            extension_candidates.append(candidate)
            candidates.append(candidate)
        if set(mechanism_best) == {"slow_na", "slow_k"}:
            slow_na_candidate = mechanism_best["slow_na"]
            slow_k_candidate = mechanism_best["slow_k"]
            combined_values = fast_parent.parameters.copy()
            combined_values[
                _SLOW_NA_SLICE.start : _SLOW_NA_G_INDEX + 1
            ] = slow_na_candidate.parameters[
                _SLOW_NA_SLICE.start : _SLOW_NA_G_INDEX + 1
            ]
            combined_values[
                _SLOW_K_SLICE.start : _SLOW_K_G_INDEX + 1
            ] = slow_k_candidate.parameters[
                _SLOW_K_SLICE.start : _SLOW_K_G_INDEX + 1
            ]
            combined_topology = replace(
                fast_parent.topology,
                slow_na_activation=(
                    slow_na_candidate.topology.slow_na_activation
                ),
                slow_na_inactivation=(
                    slow_na_candidate.topology.slow_na_inactivation
                ),
                slow_k_activation=(
                    slow_k_candidate.topology.slow_k_activation
                ),
            )
            combined_evaluation = _full_evaluation(
                combined_values,
                combined_topology,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                fast_parent.evaluation.shape_loss,
            )
            eligible = _extension_is_eligible(
                combined_evaluation,
                fast_parent.evaluation,
                config.mechanism_minimum_relative_gain,
            )
            combined_candidate = EffectiveCandidate(
                label=f"both_slow_parent_{parent_index + 1}",
                stage="slow_na_and_k",
                parent_label=fast_parent.label,
                topology=combined_topology,
                parameters=combined_values,
                evaluation=combined_evaluation,
                eligible=eligible,
            )
            extension_candidates.append(combined_candidate)
            candidates.append(combined_candidate)

    joint_pool = [
        candidate
        for candidate in (*retained_fast, *extension_candidates)
        if candidate.evaluation.valid and candidate.eligible
    ]
    joint_pool = sorted(joint_pool, key=_candidate_sort_key)[
        : config.joint_seed_count
    ]
    for joint_index, seed_candidate in enumerate(joint_pool):
        active_indices = list(range(_FAST_KINETIC_SLICE.stop))
        active_indices.extend(
            range(_FAST_STATIC_SLICE.start, _FAST_STATIC_SLICE.stop)
        )
        if seed_candidate.topology.has_slow_na:
            active_indices.extend(_mechanism_indices("slow_na"))
        if seed_candidate.topology.has_slow_k:
            active_indices.extend(_mechanism_indices("slow_k"))
        active_indices = sorted(set(active_indices))
        trust_lower, trust_upper = _trust_bounds(
            seed_candidate.parameters,
            active_indices,
            config,
        )
        refined_values, _, history = _optimize_variables(
            seed_candidate.parameters,
            seed_candidate.topology,
            active_indices,
            trust_lower,
            trust_upper,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            seed_candidate.evaluation.shape_loss,
            config.joint_population_size,
            config.joint_generations,
            config.seed + 5000 + joint_index,
            f"joint_{joint_index + 1}",
        )
        histories.append(history)
        full_evaluation = _full_evaluation(
            refined_values,
            seed_candidate.topology,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            seed_candidate.evaluation.shape_loss,
        )
        joint_candidate = EffectiveCandidate(
            label=f"joint_{joint_index + 1}",
            stage="joint_20_percent",
            parent_label=seed_candidate.label,
            topology=seed_candidate.topology,
            parameters=refined_values,
            evaluation=full_evaluation,
            eligible=_joint_is_eligible(
                full_evaluation,
                seed_candidate.evaluation,
            ),
        )
        candidates.append(joint_candidate)

    selection_pool = [
        candidate
        for candidate in candidates
        if candidate.evaluation.valid and candidate.eligible
    ]
    feasible = [
        candidate
        for candidate in selection_pool
        if candidate.evaluation.rheobase_constraint_loss <= 1e-12
    ]
    selected = min(
        feasible or selection_pool,
        key=lambda candidate: candidate.evaluation.selection_score,
    )
    selected_evaluation = _full_evaluation(
        selected.parameters,
        selected.topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        artifacts=True,
    )
    selected = replace(selected, evaluation=selected_evaluation)
    candidates = [
        selected if candidate.label == selected.label else candidate
        for candidate in candidates
    ]
    parent_evaluation = _full_evaluation(
        parent.parameters,
        parent.topology,
        target,
        biological_cycles_target,
        biological_features_target,
        screen_config,
        config,
        shape_config,
        artifacts=True,
    )
    parent = replace(parent, evaluation=parent_evaluation)
    candidates = [
        parent if candidate.label == parent.label else candidate
        for candidate in candidates
    ]
    return EffectiveFitResult(
        target=target,
        biological_cycles=biological_cycles_target,
        biological_features=biological_features_target,
        parent=parent,
        candidates=tuple(candidates),
        selected=selected,
        topology_screen=pd.DataFrame(topology_rows),
        history=(
            pd.concat(histories, ignore_index=True)
            if histories
            else pd.DataFrame()
        ),
        config=config,
        shape_config=shape_config,
    )


def effective_candidate_summary(
    result: EffectiveFitResult,
) -> pd.DataFrame:
    rows = [_evaluation_row(candidate) for candidate in result.candidates]
    table = pd.DataFrame(rows)
    table.insert(0, "selected", table["label"].eq(result.selected.label))
    table.insert(0, "cell_id", result.target.cell_id)
    table.insert(0, "dataset", result.target.dataset)
    table["biological_rheobase_pa"] = result.target.sampled_rheobase_pa
    return table


def effective_feature_comparison(
    result: EffectiveFitResult,
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
                    "canonical_parent": (
                        result.parent.evaluation.protocol_features[
                            protocol.name
                        ].get(feature, float("nan"))
                    ),
                    "selected": (
                        result.selected.evaluation.protocol_features[
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


def save_effective_phase_plot(
    result: EffectiveFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    stages = (
        ("Biological", None, "#147d7e"),
        ("Canonical parent", result.parent.evaluation, "#e76f51"),
        ("Selected", result.selected.evaluation, "#457b9d"),
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
        role = "validation" if protocol.role == "validation" else "training"
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
        f"Effective-current phase fit: {result.target.dataset} / "
        f"{result.target.cell_id}\n{result.selected.topology.label}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_effective_trace_plot(
    result: EffectiveFitResult,
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
    figure, axes = plt.subplots(
        len(protocols),
        1,
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )
    stages = (
        ("Canonical parent", result.parent.evaluation, "#e76f51"),
        ("Selected", result.selected.evaluation, "#457b9d"),
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
                & (model_time <= result.config.train_duration_ms)
            )
            axis.plot(
                model_time[model_mask],
                trace.voltage_mv[model_mask],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                label=label,
            )
        role = "validation" if protocol.role == "validation" else "training"
        axis.set_title(
            f"{protocol.name.title()} ({role}), "
            f"{protocol.current_pa:g} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"Effective-current train fit: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_effective_score_plot(
    result: EffectiveFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = effective_candidate_summary(result)
    summary = summary.loc[
        summary["label"].isin(
            (
                result.parent.label,
                result.selected.parent_label,
                result.selected.label,
            )
        )
    ].drop_duplicates("label")
    metrics = (
        "shape_loss",
        "phase__physical_constraint_loss",
        "firing_pattern_loss",
        "spike_count_loss",
        "validation_loss",
        "selection_score",
    )
    labels = (
        "Phase shape",
        "Physical",
        "Spike train",
        "Spike count",
        "Validation",
        "Selection",
    )
    colors = ("#e76f51", "#d4a373", "#457b9d")
    x = np.arange(len(metrics))
    width = 0.8 / max(len(summary), 1)
    figure, axis = plt.subplots(
        figsize=(10.8, 4.8),
        constrained_layout=True,
    )
    for index, (_, row) in enumerate(summary.iterrows()):
        values = np.maximum(
            row.loc[list(metrics)].to_numpy(dtype=float),
            1e-3,
        )
        offset = (index - (len(summary) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width,
            color=colors[index % len(colors)],
            label=row["label"].replace("_", " ").title(),
        )
    axis.set_xticks(x, labels)
    axis.set_yscale("log")
    axis.set_ylabel("Normalized loss (log scale)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(
        f"Topology/mechanism selection: {result.target.cell_id}",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_topology_screen_plot(
    result: EffectiveFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result.topology_screen.loc[
        result.topology_screen["valid"]
    ].sort_values("objective_total")
    figure, axis = plt.subplots(
        figsize=(11.0, 5.2),
        constrained_layout=True,
    )
    colors = table[
        "topology__fast_na_inactivation"
    ].map({1: "#457b9d", 2: "#e76f51"})
    axis.scatter(
        np.arange(len(table)),
        table["objective_total"],
        c=colors,
        s=35,
    )
    best = table.head(min(8, len(table)))
    for x_position, (_, row) in enumerate(table.iterrows()):
        if row["topology"] in set(best["topology"]):
            axis.annotate(
                row["topology"].replace("_", "\n", 2),
                (x_position, row["objective_total"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
    axis.set_yscale("log")
    axis.set_xlabel("Valid topology, sorted by pilot objective")
    axis.set_ylabel("Screen objective")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        "Discrete fast-gate topology screen "
        "(blue: h^1, red: h^2)",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_effective_kinetics_plot(
    result: EffectiveFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = decode_effective_model(
        result.selected.parameters,
        result.selected.topology,
        result.target,
    )
    voltage = np.linspace(-100.0, 60.0, 321)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.2),
        constrained_layout=True,
    )
    for gate, color in zip(
        ("m", "h", "n"),
        ("#d62828", "#457b9d", "#2a9d8f"),
    ):
        steady, tau = model.base.kinetics.gate_curves(gate, voltage)
        axes[0, 0].plot(voltage, steady, color=color, label=gate)
        axes[0, 1].plot(voltage, tau, color=color, label=gate)
    unavailable, slow_na_tau = (
        model.slow_na_kinetics.gate_curves(
            voltage,
            result.target.temperature_c,
        )
    )
    slow_k, slow_k_tau = model.slow_k_kinetics.gate_curves(
        voltage,
        result.target.temperature_c,
    )
    axes[1, 0].plot(
        voltage,
        1.0 - unavailable,
        color="#7b2cbf",
        label="slow Na availability",
    )
    axes[1, 0].plot(
        voltage,
        slow_k,
        color="#f4a261",
        label="slow K activation",
    )
    axes[1, 1].plot(
        voltage,
        slow_na_tau,
        color="#7b2cbf",
        label="slow Na",
    )
    axes[1, 1].plot(
        voltage,
        slow_k_tau,
        color="#f4a261",
        label="slow K",
    )
    axes[0, 0].set_ylabel("Fast steady state")
    axes[0, 1].set_ylabel("Fast tau at 6.3 C (ms)")
    axes[1, 0].set_ylabel("Slow steady state")
    axes[1, 1].set_ylabel(
        f"Slow tau at {result.target.temperature_c:g} C (ms)"
    )
    axes[0, 1].set_yscale("log")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xlabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Selected effective kinetics: {result.selected.topology.label}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_effective_fit_result(
    result: EffectiveFitResult,
    screen_config: BiologicalScreenConfig,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effective_candidate_summary(result).to_csv(
        output / "candidate_summary.csv",
        index=False,
    )
    result.topology_screen.to_csv(
        output / "topology_screen.csv",
        index=False,
    )
    result.history.to_csv(output / "optimization_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": result.selected.label,
                **result.selected.topology.to_mapping(),
                **dict(
                    zip(
                        EFFECTIVE_PARAMETER_NAMES,
                        result.selected.parameters,
                    )
                ),
            }
        ]
    ).to_csv(output / "selected_parameters.csv", index=False)
    effective_feature_comparison(result).to_csv(
        output / "firing_pattern_comparison.csv",
        index=False,
    )
    metadata = {
        "version": 1,
        "experiment": (
            "discrete gate topology plus conditional slow Na/K states"
        ),
        "target": target_metadata(result.target),
        "selected_label": result.selected.label,
        "selected_topology": asdict(result.selected.topology),
        "fit_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
        "screen_config": asdict(screen_config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_effective_phase_plot(
        result,
        output / "effective_phase_comparison.png",
    )
    save_effective_trace_plot(
        result,
        screen_config,
        output / "effective_trace_comparison.png",
    )
    save_effective_score_plot(
        result,
        output / "effective_score_bars.png",
    )
    save_topology_screen_plot(
        result,
        output / "fast_topology_screen.png",
    )
    save_effective_kinetics_plot(
        result,
        output / "effective_kinetics.png",
    )
