"""Soma-AIS effective-current model with compartment-specific kinetics.

Both compartments use the same generic current family:

    I_NaF = g_NaF m^a h^b (V - E_Na)
    I_KF  = g_KF n^c (V - E_K)
    I_NaS = g_NaS m^d s^e (V - E_Na)
    I_KS  = g_KS p^f (V - E_K)

The soma parameters are initialized from the selected effective-component
model. The AIS starts as an exact representation of the old shared-kinetics
AIS, then receives its own direct m/h/n curves, conductances, integer gate
powers, and optional slow sodium and potassium states.
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
from .cell_targets import CellOptimizationTarget, target_metadata
from .direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
    DIRECT_SLOW_K_PARAMETER_NAMES,
    DIRECT_VOLTAGE_KNOTS_MV,
    DirectKinetics,
    DirectSlowKinetics,
    direct_kinetic_parameter_bounds,
    direct_slow_k_parameter_bounds,
)
from .direct_slow_k import biological_window_features
from .effective_components import (
    EFFECTIVE_PARAMETER_NAMES,
    _FAST_KINETIC_SLICE,
    _SLOW_K_G_INDEX,
    _SLOW_K_SLICE,
    _SLOW_NA_G_INDEX,
    _SLOW_NA_SLICE,
    CANONICAL_TOPOLOGY,
    EffectiveEvaluation,
    EffectiveFitConfig,
    EffectiveTopology,
    _extension_is_eligible,
    _joint_is_eligible,
    _phase_losses,
    _pow,
    _protocol_behavior_loss,
    decode_effective_model,
    effective_parameter_bounds,
    neighboring_fast_topologies,
    representative_fast_topologies,
    slow_k_topology_candidates,
    slow_na_topology_candidates,
)
from .features import FeatureConfig, extract_observed_features
from .hh_model import SimulationConfig, SimulationError, Stimulus, Trace
from .model_ladder import (
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


def _ais_name(name: str) -> str:
    return name.replace("param__", "param__ais__", 1)


AIS_FAST_PARAMETER_NAMES = tuple(
    _ais_name(name) for name in DIRECT_KINETIC_PARAMETER_NAMES
)
AIS_CONDUCTANCE_PARAMETER_NAMES = (
    "param__ais__log_gna_fast_ms_cm2",
    "param__ais__log_gk_fast_ms_cm2",
)
AIS_SLOW_NA_PARAMETER_NAMES = tuple(
    _ais_name(name.replace("slow_k", "slow_na"))
    for name in DIRECT_SLOW_K_PARAMETER_NAMES
)
AIS_SLOW_K_PARAMETER_NAMES = tuple(
    _ais_name(name) for name in DIRECT_SLOW_K_PARAMETER_NAMES
)
TWO_COMPARTMENT_PARAMETER_NAMES = (
    tuple(f"param__soma__{name.removeprefix('param__')}" for name in
          EFFECTIVE_PARAMETER_NAMES)
    + AIS_FAST_PARAMETER_NAMES
    + AIS_CONDUCTANCE_PARAMETER_NAMES
    + AIS_SLOW_NA_PARAMETER_NAMES
    + ("param__ais__slow_na__log_gna_ms_cm2",)
    + AIS_SLOW_K_PARAMETER_NAMES
    + ("param__ais__slow_k__log_gk_ms_cm2",)
)

_SOMA_SLICE = slice(0, len(EFFECTIVE_PARAMETER_NAMES))
_AIS_FAST_START = _SOMA_SLICE.stop
_AIS_FAST_SLICE = slice(
    _AIS_FAST_START,
    _AIS_FAST_START + len(DIRECT_KINETIC_PARAMETER_NAMES),
)
_AIS_GNA_INDEX = _AIS_FAST_SLICE.stop
_AIS_GK_INDEX = _AIS_GNA_INDEX + 1
_AIS_SLOW_NA_START = _AIS_GK_INDEX + 1
_AIS_SLOW_NA_SLICE = slice(
    _AIS_SLOW_NA_START,
    _AIS_SLOW_NA_START + len(DIRECT_SLOW_K_PARAMETER_NAMES),
)
_AIS_SLOW_NA_G_INDEX = _AIS_SLOW_NA_SLICE.stop
_AIS_SLOW_K_START = _AIS_SLOW_NA_G_INDEX + 1
_AIS_SLOW_K_SLICE = slice(
    _AIS_SLOW_K_START,
    _AIS_SLOW_K_START + len(DIRECT_SLOW_K_PARAMETER_NAMES),
)
_AIS_SLOW_K_G_INDEX = _AIS_SLOW_K_SLICE.stop
_AIS_FAST_INDICES = np.asarray(
    (
        *range(_AIS_FAST_SLICE.start, _AIS_FAST_SLICE.stop),
        _AIS_GNA_INDEX,
        _AIS_GK_INDEX,
    ),
    dtype=int,
)


@dataclass(frozen=True)
class CompartmentTopologies:
    """Independent integer gate powers in the soma and AIS."""

    soma: EffectiveTopology
    ais: EffectiveTopology

    @property
    def active_state_count(self) -> int:
        return self.soma.active_state_count + self.ais.active_state_count

    @property
    def active_component_count(self) -> int:
        return (
            self.soma.active_component_count
            + self.ais.active_component_count
        )

    @property
    def label(self) -> str:
        return f"soma[{self.soma.label}]__ais[{self.ais.label}]"

    def to_mapping(self) -> dict[str, int]:
        values = {}
        for compartment, topology in (
            ("soma", self.soma),
            ("ais", self.ais),
        ):
            values.update(
                {
                    f"topology__{compartment}__{name}": int(value)
                    for name, value in asdict(topology).items()
                }
            )
        return values


@dataclass(frozen=True)
class TwoCompartmentEffectiveModel:
    """Decoded soma-AIS model with four possible currents per compartment."""

    soma: object
    ais_topology: EffectiveTopology
    ais_kinetics: DirectKinetics
    ais_slow_na_kinetics: DirectSlowKinetics
    ais_slow_k_kinetics: DirectSlowKinetics
    ais_gna_fast_ms_cm2: float
    ais_gk_fast_ms_cm2: float
    ais_gslow_na_ms_cm2: float
    ais_gslow_k_ms_cm2: float
    ais_eleak_mv: float

    @property
    def topologies(self) -> CompartmentTopologies:
        return CompartmentTopologies(
            soma=self.soma.topology,
            ais=self.ais_topology,
        )


@dataclass(frozen=True)
class TwoCompartmentSimulation:
    soma: Trace
    ais_voltage_mv: np.ndarray
    ais_dvdt_mv_ms: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class TwoCompartmentEvaluation(EffectiveEvaluation):
    ais_voltage_mv: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class TwoCompartmentCandidate:
    label: str
    stage: str
    parent_label: str
    topologies: CompartmentTopologies
    parameters: np.ndarray
    evaluation: TwoCompartmentEvaluation
    eligible: bool = True


@dataclass(frozen=True)
class TwoCompartmentFitResult:
    target: CellOptimizationTarget
    biological_cycles: Mapping[str, PhaseCycle]
    biological_features: Mapping[str, Mapping[str, float]]
    parent: TwoCompartmentCandidate
    candidates: tuple[TwoCompartmentCandidate, ...]
    selected: TwoCompartmentCandidate
    topology_screen: pd.DataFrame
    history: pd.DataFrame
    config: EffectiveFitConfig
    shape_config: PhaseShapeConfig


def _fast_only(topology: EffectiveTopology) -> EffectiveTopology:
    return EffectiveTopology(
        fast_na_activation=topology.fast_na_activation,
        fast_na_inactivation=topology.fast_na_inactivation,
        fast_k_activation=topology.fast_k_activation,
    )


def load_selected_effective_parent(
    path: str | Path,
    target: CellOptimizationTarget,
) -> tuple[np.ndarray, CompartmentTopologies]:
    """Load a selected soma model and construct its exact old AIS parent."""
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError(f"Expected one selected row in {path}")
    row = table.iloc[0]
    soma_values = row.loc[list(EFFECTIVE_PARAMETER_NAMES)].to_numpy(
        dtype=float
    )
    soma_topology = EffectiveTopology(
        **{
            field: int(row[f"topology__{field}"])
            for field in asdict(CANONICAL_TOPOLOGY)
        }
    )
    return initialize_two_compartment_parent(
        soma_values,
        soma_topology,
        target,
    )


def initialize_two_compartment_parent(
    soma_values: Sequence[float],
    soma_topology: EffectiveTopology,
    target: CellOptimizationTarget,
) -> tuple[np.ndarray, CompartmentTopologies]:
    """Lift an effective-component model into the explicit soma-AIS model."""
    soma_values = np.asarray(soma_values, dtype=float)
    if soma_values.shape != (len(EFFECTIVE_PARAMETER_NAMES),):
        raise ValueError("Soma effective-component vector has wrong shape")
    soma = decode_effective_model(soma_values, soma_topology, target)
    ais_topology = _fast_only(soma_topology)
    values = np.concatenate(
        (
            soma_values,
            soma_values[_FAST_KINETIC_SLICE],
            (
                math.log(
                    soma.base.biophysics.gna_ms_cm2
                    * soma.base.ais_gna_multiplier
                ),
                math.log(soma.base.biophysics.gk_ms_cm2),
            ),
            soma_values[_SLOW_NA_SLICE],
            (soma_values[_SLOW_NA_G_INDEX],),
            soma_values[_SLOW_K_SLICE],
            (soma_values[_SLOW_K_G_INDEX],),
        )
    )
    return values, CompartmentTopologies(soma_topology, ais_topology)


def two_compartment_parameter_bounds(
) -> tuple[np.ndarray, np.ndarray]:
    soma_lower, soma_upper = effective_parameter_bounds()
    fast_lower, fast_upper = direct_kinetic_parameter_bounds()
    slow_lower, slow_upper = direct_slow_k_parameter_bounds()
    conductance_lower = math.log(0.001)
    conductance_upper = math.log(500.0)
    slow_g_lower = math.log(0.001)
    slow_g_upper = math.log(20.0)
    return (
        np.concatenate(
            (
                soma_lower,
                fast_lower,
                (conductance_lower, conductance_lower),
                slow_lower,
                (slow_g_lower,),
                slow_lower,
                (slow_g_lower,),
            )
        ),
        np.concatenate(
            (
                soma_upper,
                fast_upper,
                (conductance_upper, conductance_upper),
                slow_upper,
                (slow_g_upper,),
                slow_upper,
                (slow_g_upper,),
            )
        ),
    )


def _ais_currents(
    voltage_mv: float,
    m_gate: float,
    h_gate: float,
    n_gate: float,
    slow_na_availability: float,
    slow_k_activation: float,
    model: TwoCompartmentEffectiveModel,
) -> tuple[float, float, float, float, float]:
    topology = model.ais_topology
    biophysics = model.soma.base.biophysics
    fast_na = (
        model.ais_gna_fast_ms_cm2
        * _pow(m_gate, topology.fast_na_activation)
        * _pow(h_gate, topology.fast_na_inactivation)
        * (voltage_mv - biophysics.ena_mv)
    )
    fast_k = (
        model.ais_gk_fast_ms_cm2
        * _pow(n_gate, topology.fast_k_activation)
        * (voltage_mv - biophysics.ek_mv)
    )
    slow_na = 0.0
    if topology.has_slow_na:
        slow_na = (
            model.ais_gslow_na_ms_cm2
            * _pow(m_gate, topology.slow_na_activation)
            * _pow(
                slow_na_availability,
                topology.slow_na_inactivation,
            )
            * (voltage_mv - biophysics.ena_mv)
        )
    slow_k = 0.0
    if topology.has_slow_k:
        slow_k = (
            model.ais_gslow_k_ms_cm2
            * _pow(slow_k_activation, topology.slow_k_activation)
            * (voltage_mv - biophysics.ek_mv)
        )
    leak = biophysics.gleak_ms_cm2 * (
        voltage_mv - model.ais_eleak_mv
    )
    return (
        float(fast_na + slow_na),
        float(fast_k + slow_k),
        float(leak),
        float(fast_na + fast_k + slow_na + slow_k + leak),
        float(slow_na + slow_k),
    )


def decode_two_compartment_model(
    parameters: Sequence[float],
    topologies: CompartmentTopologies,
    target: CellOptimizationTarget,
) -> TwoCompartmentEffectiveModel:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(TWO_COMPARTMENT_PARAMETER_NAMES),):
        raise ValueError(f"Invalid soma-AIS vector: {values.shape}")
    soma = decode_effective_model(
        values[_SOMA_SLICE],
        topologies.soma,
        target,
    )
    temporary = TwoCompartmentEffectiveModel(
        soma=soma,
        ais_topology=topologies.ais,
        ais_kinetics=DirectKinetics.from_parameters(
            values[_AIS_FAST_SLICE]
        ),
        ais_slow_na_kinetics=DirectSlowKinetics.from_parameters(
            values[_AIS_SLOW_NA_SLICE]
        ),
        ais_slow_k_kinetics=DirectSlowKinetics.from_parameters(
            values[_AIS_SLOW_K_SLICE]
        ),
        ais_gna_fast_ms_cm2=math.exp(float(values[_AIS_GNA_INDEX])),
        ais_gk_fast_ms_cm2=math.exp(float(values[_AIS_GK_INDEX])),
        ais_gslow_na_ms_cm2=math.exp(
            float(values[_AIS_SLOW_NA_G_INDEX])
        ),
        ais_gslow_k_ms_cm2=math.exp(
            float(values[_AIS_SLOW_K_G_INDEX])
        ),
        ais_eleak_mv=soma.base.ais_eleak_mv,
    )
    rest = target.passive.resting_voltage_mv
    m_gate, h_gate, n_gate = temporary.ais_kinetics.steady_state(rest)
    unavailable, _ = (
        temporary.ais_slow_na_kinetics.gate_curves_scalar(rest)
    )
    slow_k, _ = temporary.ais_slow_k_kinetics.gate_curves_scalar(rest)
    currents = _ais_currents(
        rest,
        m_gate,
        h_gate,
        n_gate,
        1.0 - unavailable,
        slow_k,
        temporary,
    )
    active = currents[3] - currents[2]
    ais_eleak = (
        rest
        + active / soma.base.biophysics.gleak_ms_cm2
    )
    return replace(temporary, ais_eleak_mv=float(ais_eleak))


def two_compartment_initial_state(
    model: TwoCompartmentEffectiveModel,
    target: CellOptimizationTarget,
) -> np.ndarray:
    voltage = target.passive.resting_voltage_mv
    sm, sh, sn = model.soma.base.kinetics.steady_state(voltage)
    soma_unavailable, _ = (
        model.soma.slow_na_kinetics.gate_curves_scalar(voltage)
    )
    soma_slow_k, _ = model.soma.slow_k_kinetics.gate_curves_scalar(
        voltage
    )
    am, ah, an = model.ais_kinetics.steady_state(voltage)
    ais_unavailable, _ = (
        model.ais_slow_na_kinetics.gate_curves_scalar(voltage)
    )
    ais_slow_k, _ = model.ais_slow_k_kinetics.gate_curves_scalar(
        voltage
    )
    return np.asarray(
        (
            voltage,
            sm,
            sh,
            sn,
            1.0 - soma_unavailable,
            soma_slow_k,
            voltage,
            am,
            ah,
            an,
            1.0 - ais_unavailable,
            ais_slow_k,
        ),
        dtype=float,
    )


def _soma_currents(
    state: np.ndarray,
    model: TwoCompartmentEffectiveModel,
) -> tuple[float, float, float, float, float]:
    from .effective_components import _component_currents

    return _component_currents(
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
        model.soma,
        "soma",
    )


def _two_compartment_voltage_rhs(
    time_ms: float,
    state: np.ndarray,
    model: TwoCompartmentEffectiveModel,
    config: SimulationConfig,
) -> np.ndarray:
    base = model.soma.base
    biophysics = base.biophysics
    soma_current = _soma_currents(state, model)[3]
    ais_current = _ais_currents(
        state[6],
        state[7],
        state[8],
        state[9],
        state[10],
        state[11],
        model,
    )[3]
    soma_area = biophysics.membrane_area_um2 * (
        1.0 - base.ais_area_fraction
    )
    ais_area = biophysics.membrane_area_um2 * base.ais_area_fraction
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


def _slow_gate_update(
    value: float,
    steady: float,
    tau_ms: float,
    duration_ms: float,
) -> float:
    decay = math.exp(max(-80.0, -duration_ms / max(tau_ms, 1e-12)))
    return steady + (value - steady) * decay


def _two_compartment_gate_half_step(
    state: np.ndarray,
    model: TwoCompartmentEffectiveModel,
    config: SimulationConfig,
    duration_ms: float,
) -> np.ndarray:
    updated = state.copy()
    updated[1:4] = _rush_larsen_gate_update(
        float(state[0]),
        state[1:4],
        model.soma.base.kinetics,
        model.soma.base,
        config.temperature_c,
        duration_ms,
    )
    unavailable, tau = model.soma.slow_na_kinetics.gate_curves_scalar(
        float(state[0]),
        config.temperature_c,
    )
    updated[4] = _slow_gate_update(
        float(state[4]), 1.0 - unavailable, tau, duration_ms
    )
    steady, tau = model.soma.slow_k_kinetics.gate_curves_scalar(
        float(state[0]),
        config.temperature_c,
    )
    updated[5] = _slow_gate_update(
        float(state[5]), steady, tau, duration_ms
    )
    updated[7:10] = _rush_larsen_gate_update(
        float(state[6]),
        state[7:10],
        model.ais_kinetics,
        model.soma.base,
        config.temperature_c,
        duration_ms,
    )
    unavailable, tau = (
        model.ais_slow_na_kinetics.gate_curves_scalar(
            float(state[6]),
            config.temperature_c,
        )
    )
    updated[10] = _slow_gate_update(
        float(state[10]), 1.0 - unavailable, tau, duration_ms
    )
    steady, tau = model.ais_slow_k_kinetics.gate_curves_scalar(
        float(state[6]),
        config.temperature_c,
    )
    updated[11] = _slow_gate_update(
        float(state[11]), steady, tau, duration_ms
    )
    return updated


def _two_compartment_step(
    time_ms: float,
    state: np.ndarray,
    model: TwoCompartmentEffectiveModel,
    config: SimulationConfig,
) -> np.ndarray:
    dt_ms = config.dt_ms
    half = _two_compartment_gate_half_step(
        state, model, config, dt_ms / 2.0
    )
    voltage_indices = np.asarray((0, 6), dtype=int)

    def rhs(rhs_time: float, voltages: np.ndarray) -> np.ndarray:
        trial = half.copy()
        trial[voltage_indices] = voltages
        return _two_compartment_voltage_rhs(
            rhs_time, trial, model, config
        )

    voltage = half[voltage_indices]
    k1 = rhs(time_ms, voltage)
    k2 = rhs(time_ms + dt_ms / 2.0, voltage + dt_ms * k1 / 2.0)
    k3 = rhs(time_ms + dt_ms / 2.0, voltage + dt_ms * k2 / 2.0)
    k4 = rhs(time_ms + dt_ms, voltage + dt_ms * k3)
    half[voltage_indices] = voltage + dt_ms * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0
    updated = _two_compartment_gate_half_step(
        half, model, config, dt_ms / 2.0
    )
    gate_indices = np.asarray(
        (1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
        dtype=int,
    )
    if (
        not np.all(np.isfinite(updated))
        or np.any(np.abs(updated[voltage_indices]) > 200.0)
        or np.any(updated[gate_indices] < -0.05)
        or np.any(updated[gate_indices] > 1.05)
    ):
        raise SimulationError(
            f"Soma-AIS integration diverged at "
            f"{time_ms + dt_ms:.4f} ms"
        )
    updated[gate_indices] = np.clip(updated[gate_indices], 0.0, 1.0)
    return updated


def simulate_two_compartment_effective(
    model: TwoCompartmentEffectiveModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> TwoCompartmentSimulation:
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, 12), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for index in range(n_steps):
        states[index + 1] = _two_compartment_step(
            time_ms[index], states[index], model, config
        )
    biophysics = model.soma.base.biophysics
    applied_pa = np.asarray(
        [
            config.stimulus.pa_value(time, biophysics)
            for time in time_ms
        ],
        dtype=float,
    )
    soma_area = biophysics.membrane_area_um2 * (
        1.0 - model.soma.base.ais_area_fraction
    )
    applied_density = applied_pa / (soma_area * 0.01)
    current_rows = np.asarray(
        [_soma_currents(state, model) for state in states],
        dtype=float,
    )
    derivatives = np.asarray(
        [
            _two_compartment_voltage_rhs(
                time, state, model, config
            )
            for time, state in zip(time_ms, states)
        ],
        dtype=float,
    )
    trace = Trace(
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
        dvdt_mv_ms=derivatives[:, 0],
    )
    return TwoCompartmentSimulation(
        soma=trace,
        ais_voltage_mv=states[:, 6],
        ais_dvdt_mv_ms=derivatives[:, 1],
        states=states,
    )


def _elicits_spike(
    model: TwoCompartmentEffectiveModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> bool:
    state = np.asarray(initial_state, dtype=float)
    n_steps = int(round(config.duration_ms / config.dt_ms))
    for index in range(n_steps):
        previous = float(state[0])
        state = _two_compartment_step(
            index * config.dt_ms, state, model, config
        )
        if previous < 0.0 <= state[0]:
            return True
    return False


def find_rheobase_two_compartment(
    model: TwoCompartmentEffectiveModel,
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
            model.soma.base.biophysics,
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


def evaluate_two_compartment_candidate(
    parameters: Sequence[float],
    topologies: CompartmentTopologies,
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
) -> TwoCompartmentEvaluation:
    """Evaluate phase shape and absolute-current spike trains at the soma."""
    try:
        values = np.asarray(parameters, dtype=float)
        lower, upper = two_compartment_parameter_bounds()
        if (
            values.shape != lower.shape
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError("Soma-AIS candidate is outside fitted bounds")
        model = decode_two_compartment_model(values, topologies, target)
        if (
            not -120.0
            <= model.soma.base.biophysics.eleak_mv
            <= -20.0
            or not -120.0 <= model.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        voltage = np.linspace(-100.0, 60.0, 161)
        slow_kinetics = (
            (
                topologies.soma.has_slow_na,
                model.soma.slow_na_kinetics,
            ),
            (
                topologies.soma.has_slow_k,
                model.soma.slow_k_kinetics,
            ),
            (
                topologies.ais.has_slow_na,
                model.ais_slow_na_kinetics,
            ),
            (
                topologies.ais.has_slow_k,
                model.ais_slow_k_kinetics,
            ),
        )
        for enabled, kinetics in slow_kinetics:
            if enabled:
                _, tau = kinetics.gate_curves(
                    voltage, target.temperature_c
                )
                if float(np.min(tau)) < 1.0:
                    raise SimulationError("Slow gate tau fell below 1 ms")
        state = two_compartment_initial_state(model, target)
        zero_config = SimulationConfig(
            duration_ms=1.0,
            dt_ms=screen_config.dt_ms,
            initial_voltage_mv=target.passive.resting_voltage_mv,
            stimulus=Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=0.0,
                start_ms=0.0,
                end_ms=1.0,
            ),
            conductances=model.soma.base.biophysics,
            temperature_c=target.temperature_c,
        )
        rest_rhs = _two_compartment_voltage_rhs(
            0.0, state, model, zero_config
        )
        if float(np.max(np.abs(rest_rhs))) > 1e-6:
            raise SimulationError("Reanchoring did not preserve coupled rest")
        _, upper_rheobase, _ = find_rheobase_two_compartment(
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
        ais_voltages: dict[str, np.ndarray] = {}
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
                model.soma.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            phase_result = simulate_two_compartment_effective(
                model, phase_simulation, state
            )
            phase_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=phase_current,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + phase_duration,
            )
            cycle = extract_phase_cycle(
                phase_result.soma.time_ms,
                phase_result.soma.voltage_mv,
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
                model.soma.base.biophysics,
                target.passive.resting_voltage_mv,
            )
            train_result = simulate_two_compartment_effective(
                model, train_simulation, state
            )
            train_stimulus = Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=protocol.current_pa,
                start_ms=screen_config.baseline_ms,
                end_ms=screen_config.baseline_ms + train_duration,
            )
            features[protocol.name] = extract_observed_features(
                train_result.soma,
                train_stimulus,
                FeatureConfig(min_spikes=1),
            )
            if include_artifacts:
                traces[protocol.name] = train_result.soma
                ais_voltages[protocol.name] = (
                    train_result.ais_voltage_mv
                )
                cycles[protocol.name] = cycle
        training_scores = {
            protocol.name: scores[protocol.name]
            for protocol in training_protocols
        }
        phase_scores = mean_phase_scores(training_scores)
        shape_loss, scale_loss = _phase_losses(
            phase_scores, fit_config
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
        smoothness = (
            model.soma.base.kinetics.smoothness_penalty()
            + model.ais_kinetics.smoothness_penalty()
        )
        for enabled, kinetics in slow_kinetics:
            if enabled:
                smoothness += kinetics.smoothness_penalty()
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
                validation_phase, fit_config
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
        extra_ais_states = (
            topologies.ais.active_state_count
            - CANONICAL_TOPOLOGY.active_state_count
        )
        complexity = fit_config.extra_state_penalty * extra_ais_states
        selection_score = float(
            objective
            + fit_config.validation_weight * validation_loss
            + complexity
        )
        physical = model.soma.base.biophysics.to_physical_mapping()
        physical.update(
            {
                "extension__ais_area_fraction": (
                    model.soma.base.ais_area_fraction
                ),
                "extension__coupling_ns": (
                    model.soma.base.coupling_ns
                ),
                "extension__ais_gna_fast_ms_cm2": (
                    model.ais_gna_fast_ms_cm2
                ),
                "extension__ais_gk_fast_ms_cm2": (
                    model.ais_gk_fast_ms_cm2
                ),
                "extension__ais_gslow_na_ms_cm2": (
                    model.ais_gslow_na_ms_cm2
                    if topologies.ais.has_slow_na
                    else 0.0
                ),
                "extension__ais_gslow_k_ms_cm2": (
                    model.ais_gslow_k_ms_cm2
                    if topologies.ais.has_slow_k
                    else 0.0
                ),
                "extension__active_state_count": (
                    topologies.active_state_count
                ),
                "extension__active_component_count": (
                    topologies.active_component_count
                ),
            }
        )
        return TwoCompartmentEvaluation(
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
            ais_voltage_mv=ais_voltages if include_artifacts else None,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return TwoCompartmentEvaluation(
            valid=False,
            reason=f"{type(error).__name__}: {error}",
            objective_total=fit_config.invalid_score,
            selection_score=fit_config.invalid_score,
        )


def _full_evaluation(
    parameters: np.ndarray,
    topologies: CompartmentTopologies,
    target: CellOptimizationTarget,
    biological_cycles_target: Mapping[str, PhaseCycle],
    biological_features_target: Mapping[str, Mapping[str, float]],
    screen_config: BiologicalScreenConfig,
    fit_config: EffectiveFitConfig,
    shape_config: PhaseShapeConfig,
    reference_shape_loss: float = float("inf"),
    artifacts: bool = False,
) -> TwoCompartmentEvaluation:
    return evaluate_two_compartment_candidate(
        parameters,
        topologies,
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


def _screen_evaluation_payload(
    parameters: np.ndarray,
    topologies: CompartmentTopologies,
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
) -> TwoCompartmentEvaluation:
    return evaluate_two_compartment_candidate(
        parameters,
        topologies,
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


def _evaluate_screen_batch(
    parameter_vectors: Sequence[np.ndarray],
    topologies: Sequence[CompartmentTopologies],
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
) -> list[TwoCompartmentEvaluation]:
    count = len(parameter_vectors)
    if (
        len(topologies) != count
        or len(reference_shape_losses) != count
    ):
        raise ValueError("Soma-AIS screen columns have inconsistent lengths")
    arguments = zip(
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
    if fit_config.workers <= 1 or count <= 1:
        return [
            _screen_evaluation_payload(*argument)
            for argument in arguments
        ]
    try:
        with ProcessPoolExecutor(
            max_workers=fit_config.workers
        ) as executor:
            columns = list(
                zip(
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
            return list(
                executor.map(
                    _screen_evaluation_payload,
                    *(list(column) for column in zip(*columns)),
                )
            )
    except (OSError, PermissionError) as error:
        warnings.warn(
            f"Parallel soma-AIS screen unavailable ({error}); "
            "using one worker.",
            RuntimeWarning,
        )
        return _evaluate_screen_batch(
            parameter_vectors,
            topologies,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            replace(fit_config, workers=1),
            shape_config,
            reference_shape_losses,
            protocol_limit,
            train_duration_ms,
            phase_duration_ms,
        )


def _evaluation_payload(
    unbounded: Sequence[float],
    variable_indices: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    fixed_parameters: np.ndarray,
    topologies: CompartmentTopologies,
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
) -> tuple[np.ndarray, TwoCompartmentEvaluation]:
    variables = _bounded_parameters(
        unbounded, variable_lower, variable_upper
    )
    parameters = np.asarray(fixed_parameters, dtype=float).copy()
    parameters[variable_indices] = variables
    evaluation = evaluate_two_compartment_candidate(
        parameters,
        topologies,
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
    return parameters, evaluation


def _optimize_variables(
    center: np.ndarray,
    topologies: CompartmentTopologies,
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
) -> tuple[np.ndarray, TwoCompartmentEvaluation, pd.DataFrame]:
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Soma-AIS fitting requires DEAP") from error
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
    fitness_name = "FitnessTwoCompartmentCMA"
    individual_name = "IndividualTwoCompartmentCMA"
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
    center_evaluation = evaluate_two_compartment_candidate(
        clipped_center,
        topologies,
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
    evaluated = [(clipped_center, center_evaluation)]
    rows = [
        {
            "stage": stage,
            "generation": -1,
            "individual": -1,
            "topology": topologies.label,
            "valid": center_evaluation.valid,
            "objective_total": center_evaluation.objective_total,
            "reason": center_evaluation.reason,
        }
    ]
    executor = None
    mapper = map
    if fit_config.workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=fit_config.workers)
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
                    _evaluation_payload,
                    population,
                    [indices] * len(population),
                    [variable_lower] * len(population),
                    [variable_upper] * len(population),
                    [clipped_center] * len(population),
                    [topologies] * len(population),
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
                        "topology": topologies.label,
                        "valid": evaluation.valid,
                        "objective_total": evaluation.objective_total,
                        "reason": evaluation.reason,
                    }
                )
            strategy.update(population)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    valid = [item for item in evaluated if item[1].valid]
    feasible = [
        item
        for item in valid
        if item[1].rheobase_constraint_loss <= 1e-12
    ]
    pool = feasible or valid
    if not pool:
        raise RuntimeError(
            f"Soma-AIS {stage} optimization produced no valid model"
        )
    parameters, evaluation = min(
        pool, key=lambda item: item[1].objective_total
    )
    return parameters, evaluation, pd.DataFrame(rows)


def _ais_conductance_normalized_topology(
    parent_parameters: np.ndarray,
    parent_topology: EffectiveTopology,
    candidate_topology: EffectiveTopology,
    target: CellOptimizationTarget,
    soma_topology: EffectiveTopology,
    reference_voltage_mv: float = -20.0,
) -> np.ndarray:
    values = np.asarray(parent_parameters, dtype=float).copy()
    topologies = CompartmentTopologies(
        soma=soma_topology,
        ais=parent_topology,
    )
    model = decode_two_compartment_model(values, topologies, target)
    m_gate, h_gate, n_gate = model.ais_kinetics.steady_state(
        reference_voltage_mv
    )
    parent_na = (
        _pow(m_gate, parent_topology.fast_na_activation)
        * _pow(h_gate, parent_topology.fast_na_inactivation)
    )
    candidate_na = (
        _pow(m_gate, candidate_topology.fast_na_activation)
        * _pow(h_gate, candidate_topology.fast_na_inactivation)
    )
    parent_k = _pow(n_gate, parent_topology.fast_k_activation)
    candidate_k = _pow(
        n_gate, candidate_topology.fast_k_activation
    )
    values[_AIS_GNA_INDEX] += math.log(
        max(parent_na, 1e-12) / max(candidate_na, 1e-12)
    )
    values[_AIS_GK_INDEX] += math.log(
        max(parent_k, 1e-12) / max(candidate_k, 1e-12)
    )
    lower, upper = two_compartment_parameter_bounds()
    return np.clip(values, lower + 1e-9, upper - 1e-9)


def _mechanism_indices(mechanism: str) -> np.ndarray:
    if mechanism == "slow_na":
        return np.asarray(
            (
                *range(
                    _AIS_SLOW_NA_SLICE.start,
                    _AIS_SLOW_NA_SLICE.stop,
                ),
                _AIS_SLOW_NA_G_INDEX,
            ),
            dtype=int,
        )
    if mechanism == "slow_k":
        return np.asarray(
            (
                *range(
                    _AIS_SLOW_K_SLICE.start,
                    _AIS_SLOW_K_SLICE.stop,
                ),
                _AIS_SLOW_K_G_INDEX,
            ),
            dtype=int,
        )
    raise ValueError(f"Unknown AIS mechanism: {mechanism}")


def _trust_bounds(
    center: np.ndarray,
    active_indices: Sequence[int],
    config: EffectiveFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    absolute_lower, absolute_upper = two_compartment_parameter_bounds()
    lower = np.asarray(center, dtype=float).copy()
    upper = np.asarray(center, dtype=float).copy()
    log_margin = math.log1p(config.trust_fraction)
    for index in active_indices:
        name = TWO_COMPARTMENT_PARAMETER_NAMES[index]
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


def _candidate_sort_key(
    candidate: TwoCompartmentCandidate,
) -> tuple[float, float]:
    return (
        float(candidate.evaluation.rheobase_constraint_loss > 1e-12),
        candidate.evaluation.selection_score,
    )


def fit_two_compartment_ladder(
    target: CellOptimizationTarget,
    parent_parameters: Sequence[float],
    parent_topologies: CompartmentTopologies,
    screen_config: BiologicalScreenConfig,
    config: EffectiveFitConfig | None = None,
    shape_config: PhaseShapeConfig | None = None,
) -> TwoCompartmentFitResult:
    """Fit independent AIS topology, kinetics, memory, and joint parameters."""
    config = config or EffectiveFitConfig()
    shape_config = shape_config or PhaseShapeConfig()
    parent_values = np.asarray(parent_parameters, dtype=float)
    lower, upper = two_compartment_parameter_bounds()
    if parent_values.shape != lower.shape:
        raise ValueError("Soma-AIS parent vector has the wrong shape")
    biological_cycles_target = biological_phase_cycles(
        target, shape_config, include_validation=True
    )
    biological_features_target = biological_window_features(
        target, config.train_duration_ms
    )
    parent_evaluation = _full_evaluation(
        parent_values,
        parent_topologies,
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
            f"Soma-AIS parent is invalid: {parent_evaluation.reason}"
        )
    parent = TwoCompartmentCandidate(
        label="shared_ais_parent",
        stage="parent",
        parent_label="",
        topologies=parent_topologies,
        parameters=parent_values,
        evaluation=parent_evaluation,
    )
    candidates = [parent]
    histories: list[pd.DataFrame] = []
    topology_rows: list[dict[str, object]] = []
    screened: list[
        tuple[
            EffectiveTopology,
            np.ndarray,
            TwoCompartmentEvaluation,
        ]
    ] = []

    def screen(topologies: Sequence[EffectiveTopology], stage: str) -> None:
        new_topologies = [
            topology
            for topology in topologies
            if topology not in {item[0] for item in screened}
        ]
        vectors = [
            _ais_conductance_normalized_topology(
                parent_values,
                parent_topologies.ais,
                ais_topology,
                target,
                parent_topologies.soma,
            )
            for ais_topology in new_topologies
        ]
        pairs = [
            CompartmentTopologies(
                parent_topologies.soma, ais_topology
            )
            for ais_topology in new_topologies
        ]
        evaluations = _evaluate_screen_batch(
            vectors,
            pairs,
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
        for ais_topology, values, pair, evaluation in zip(
            new_topologies, vectors, pairs, evaluations
        ):
            screened.append((ais_topology, values, evaluation))
            topology_rows.append(
                {
                    "screen_stage": stage,
                    **pair.to_mapping(),
                    "topology": pair.label,
                    "valid": evaluation.valid,
                    "reason": evaluation.reason,
                    "objective_total": evaluation.objective_total,
                    "shape_loss": evaluation.shape_loss,
                    "scale_loss": evaluation.scale_loss,
                    "firing_pattern_loss": (
                        evaluation.firing_pattern_loss
                    ),
                    "spike_count_loss": evaluation.spike_count_loss,
                    "model_rheobase_pa": evaluation.model_rheobase_pa,
                }
            )

    screen(representative_fast_topologies(), "representative")
    valid_screened = [item for item in screened if item[2].valid]
    valid_screened.sort(
        key=lambda item: (
            item[2].rheobase_constraint_loss > 1e-12,
            item[2].objective_total,
        )
    )
    neighbor_topologies = {
        neighbor
        for topology, _, _ in valid_screened[:3]
        for neighbor in neighboring_fast_topologies(topology)
    }
    screen(tuple(sorted(neighbor_topologies)), "winner_neighbors")
    valid_screened = [item for item in screened if item[2].valid]
    valid_screened.sort(
        key=lambda item: (
            item[2].rheobase_constraint_loss > 1e-12,
            item[2].objective_total,
        )
    )
    if not valid_screened:
        raise RuntimeError("No valid AIS fast topology survived screening")
    fast_candidates: list[TwoCompartmentCandidate] = []
    for index, (ais_topology, values, evaluation) in enumerate(
        valid_screened[: config.fast_parent_count]
    ):
        pair = CompartmentTopologies(
            parent_topologies.soma, ais_topology
        )
        refined, _, history = _optimize_variables(
            values,
            pair,
            _AIS_FAST_INDICES,
            lower,
            upper,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            evaluation.shape_loss,
            config.fast_population_size,
            config.fast_generations,
            config.seed + 100 + index,
            f"ais_fast_{index + 1}",
            config.screen_protocol_count,
            config.screen_train_duration_ms,
            config.screen_phase_duration_ms,
        )
        histories.append(history)
        full = _full_evaluation(
            refined,
            pair,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
        )
        candidate = TwoCompartmentCandidate(
            label=f"ais_fast_parent_{index + 1}",
            stage="ais_fast",
            parent_label=parent.label,
            topologies=pair,
            parameters=refined,
            evaluation=full,
        )
        fast_candidates.append(candidate)
        candidates.append(candidate)
    retained_fast = sorted(
        [
            candidate
            for candidate in (parent, *fast_candidates)
            if candidate.evaluation.valid
        ],
        key=_candidate_sort_key,
    )[: config.fast_parent_count]
    extension_candidates: list[TwoCompartmentCandidate] = []
    for parent_index, fast_parent in enumerate(
        retained_fast[: config.mechanism_parent_count]
    ):
        mechanism_best = {}
        for mechanism in ("slow_na", "slow_k"):
            ais_topologies = (
                slow_na_topology_candidates(fast_parent.topologies.ais)
                if mechanism == "slow_na"
                else slow_k_topology_candidates(
                    fast_parent.topologies.ais
                )
            )
            conductance_index = (
                _AIS_SLOW_NA_G_INDEX
                if mechanism == "slow_na"
                else _AIS_SLOW_K_G_INDEX
            )
            grid_vectors = []
            grid_pairs = []
            for topology in ais_topologies:
                for conductance in (0.02, 0.2, 2.0):
                    values = fast_parent.parameters.copy()
                    values[conductance_index] = math.log(conductance)
                    pair = CompartmentTopologies(
                        fast_parent.topologies.soma, topology
                    )
                    grid_vectors.append(values)
                    grid_pairs.append(pair)
            grid_evaluations = _evaluate_screen_batch(
                grid_vectors,
                grid_pairs,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                [fast_parent.evaluation.shape_loss]
                * len(grid_vectors),
                config.screen_protocol_count,
                config.screen_train_duration_ms,
                config.screen_phase_duration_ms,
            )
            grid = [
                (pair, values, evaluation)
                for pair, values, evaluation in zip(
                    grid_pairs, grid_vectors, grid_evaluations
                )
                if evaluation.valid
            ]
            if not grid:
                continue
            pair, values, screen_evaluation = min(
                grid,
                key=lambda item: (
                    item[2].rheobase_constraint_loss > 1e-12,
                    item[2].objective_total,
                ),
            )
            refined, _, history = _optimize_variables(
                values,
                pair,
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
                config.seed
                + 1000
                + 100 * parent_index
                + (0 if mechanism == "slow_na" else 50),
                f"ais_{mechanism}_{parent_index + 1}",
                config.screen_protocol_count,
                config.screen_train_duration_ms,
                config.screen_phase_duration_ms,
            )
            histories.append(history)
            full = _full_evaluation(
                refined,
                pair,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                fast_parent.evaluation.shape_loss,
            )
            candidate = TwoCompartmentCandidate(
                label=f"ais_{mechanism}_parent_{parent_index + 1}",
                stage=f"ais_{mechanism}",
                parent_label=fast_parent.label,
                topologies=pair,
                parameters=refined,
                evaluation=full,
                eligible=_extension_is_eligible(
                    full,
                    fast_parent.evaluation,
                    config.mechanism_minimum_relative_gain,
                ),
            )
            mechanism_best[mechanism] = candidate
            extension_candidates.append(candidate)
            candidates.append(candidate)
        if set(mechanism_best) == {"slow_na", "slow_k"}:
            slow_na = mechanism_best["slow_na"]
            slow_k = mechanism_best["slow_k"]
            combined = fast_parent.parameters.copy()
            combined[
                _AIS_SLOW_NA_SLICE.start:
                _AIS_SLOW_NA_G_INDEX + 1
            ] = slow_na.parameters[
                _AIS_SLOW_NA_SLICE.start:
                _AIS_SLOW_NA_G_INDEX + 1
            ]
            combined[
                _AIS_SLOW_K_SLICE.start:
                _AIS_SLOW_K_G_INDEX + 1
            ] = slow_k.parameters[
                _AIS_SLOW_K_SLICE.start:
                _AIS_SLOW_K_G_INDEX + 1
            ]
            ais_topology = replace(
                fast_parent.topologies.ais,
                slow_na_activation=(
                    slow_na.topologies.ais.slow_na_activation
                ),
                slow_na_inactivation=(
                    slow_na.topologies.ais.slow_na_inactivation
                ),
                slow_k_activation=(
                    slow_k.topologies.ais.slow_k_activation
                ),
            )
            pair = CompartmentTopologies(
                fast_parent.topologies.soma, ais_topology
            )
            full = _full_evaluation(
                combined,
                pair,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                fast_parent.evaluation.shape_loss,
            )
            combined_candidate = TwoCompartmentCandidate(
                label=f"ais_both_slow_parent_{parent_index + 1}",
                stage="ais_slow_na_and_k",
                parent_label=fast_parent.label,
                topologies=pair,
                parameters=combined,
                evaluation=full,
                eligible=_extension_is_eligible(
                    full,
                    fast_parent.evaluation,
                    config.mechanism_minimum_relative_gain,
                ),
            )
            extension_candidates.append(combined_candidate)
            candidates.append(combined_candidate)
    joint_pool = sorted(
        [
            candidate
            for candidate in (*retained_fast, *extension_candidates)
            if candidate.evaluation.valid and candidate.eligible
        ],
        key=_candidate_sort_key,
    )[: config.joint_seed_count]
    for joint_index, seed_candidate in enumerate(joint_pool):
        active_indices = list(range(_SOMA_SLICE.stop))
        active_indices.extend(_AIS_FAST_INDICES)
        if seed_candidate.topologies.ais.has_slow_na:
            active_indices.extend(_mechanism_indices("slow_na"))
        if seed_candidate.topologies.ais.has_slow_k:
            active_indices.extend(_mechanism_indices("slow_k"))
        active_indices = sorted(set(int(index) for index in active_indices))
        trust_lower, trust_upper = _trust_bounds(
            seed_candidate.parameters, active_indices, config
        )
        refined, _, history = _optimize_variables(
            seed_candidate.parameters,
            seed_candidate.topologies,
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
        full = _full_evaluation(
            refined,
            seed_candidate.topologies,
            target,
            biological_cycles_target,
            biological_features_target,
            screen_config,
            config,
            shape_config,
            seed_candidate.evaluation.shape_loss,
        )
        candidates.append(
            TwoCompartmentCandidate(
                label=f"joint_{joint_index + 1}",
                stage="joint_20_percent",
                parent_label=seed_candidate.label,
                topologies=seed_candidate.topologies,
                parameters=refined,
                evaluation=full,
                eligible=_joint_is_eligible(
                    full, seed_candidate.evaluation
                ),
            )
        )
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
    if selected.label == parent.label:
        selected = parent
    else:
        selected = replace(
            selected,
            evaluation=_full_evaluation(
                selected.parameters,
                selected.topologies,
                target,
                biological_cycles_target,
                biological_features_target,
                screen_config,
                config,
                shape_config,
                artifacts=True,
            ),
        )
    candidates = [
        (
            selected
            if candidate.label == selected.label
            else parent
            if candidate.label == parent.label
            else candidate
        )
        for candidate in candidates
    ]
    return TwoCompartmentFitResult(
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


def _evaluation_row(
    candidate: TwoCompartmentCandidate,
) -> dict[str, object]:
    evaluation = candidate.evaluation
    row = {
        "label": candidate.label,
        "stage": candidate.stage,
        "parent_label": candidate.parent_label,
        "eligible": candidate.eligible,
        **candidate.topologies.to_mapping(),
        "topology": candidate.topologies.label,
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
                for name, value in evaluation.phase_scores.to_mapping().items()
            }
        )
    if evaluation.physical is not None:
        row.update(evaluation.physical)
    return row


def two_compartment_candidate_summary(
    result: TwoCompartmentFitResult,
) -> pd.DataFrame:
    table = pd.DataFrame(
        [_evaluation_row(candidate) for candidate in result.candidates]
    )
    table.insert(0, "selected", table["label"].eq(result.selected.label))
    table.insert(0, "cell_id", result.target.cell_id)
    table.insert(0, "dataset", result.target.dataset)
    table["biological_rheobase_pa"] = result.target.sampled_rheobase_pa
    return table


def two_compartment_feature_comparison(
    result: TwoCompartmentFitResult,
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
                    "shared_ais_parent": (
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


def save_two_compartment_phase_plot(
    result: TwoCompartmentFitResult,
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
        squeeze=False,
    )
    stages = (
        ("Biological", None, "#147d7e"),
        ("Shared AIS parent", result.parent.evaluation, "#e76f51"),
        ("Selected soma-AIS", result.selected.evaluation, "#457b9d"),
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
        role = (
            "validation"
            if protocol.role == "validation"
            else "training"
        )
        axes[row_index, 0].set_title(
            f"{protocol.name.title()} ({role}): physical",
            loc="left",
            fontsize=9,
        )
        axes[row_index, 1].set_title(
            "Normalized branch shape", loc="left", fontsize=9
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="#999999", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylabel("dV/dt (mV/ms)")
    axes[-1, 0].set_xlabel("Soma V (mV)")
    axes[-1, 1].set_xlabel("Normalized branch voltage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Soma-AIS phase fit: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_two_compartment_trace_plot(
    result: TwoCompartmentFitResult,
    screen_config: BiologicalScreenConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = {
        sweep.sweep_number: sweep
        for sweep in read_current_clamp_sweeps(
            result.target.nwb_path, long_square_only=True
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
        squeeze=False,
    )
    for axis, protocol in zip(axes[:, 0], protocols):
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
            label="Biological soma",
        )
        for label, evaluation, color in (
            ("Shared AIS parent", result.parent.evaluation, "#e76f51"),
            ("Selected soma", result.selected.evaluation, "#457b9d"),
        ):
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
        selected_trace = result.selected.evaluation.traces[protocol.name]
        model_time = selected_trace.time_ms - screen_config.baseline_ms
        model_mask = (
            (model_time >= -20.0)
            & (model_time <= result.config.train_duration_ms)
        )
        axis.plot(
            model_time[model_mask],
            result.selected.evaluation.ais_voltage_mv[
                protocol.name
            ][model_mask],
            color="#9c2c77",
            linewidth=0.75,
            linestyle="--",
            alpha=0.8,
            label="Selected AIS",
        )
        role = (
            "validation"
            if protocol.role == "validation"
            else "training"
        )
        axis.set_title(
            f"{protocol.name.title()} ({role}), "
            f"{protocol.current_pa:g} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"Independent soma-AIS train fit: {result.target.dataset} / "
        f"{result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_two_compartment_score_plot(
    result: TwoCompartmentFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = two_compartment_candidate_summary(result)
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
        figsize=(10.8, 4.8), constrained_layout=True
    )
    for index, (_, row) in enumerate(summary.iterrows()):
        values = np.maximum(
            row.loc[list(metrics)].to_numpy(dtype=float), 1e-3
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
        f"Soma-AIS model selection: {result.target.cell_id}",
        loc="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_two_compartment_kinetics_plot(
    result: TwoCompartmentFitResult,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = decode_two_compartment_model(
        result.selected.parameters,
        result.selected.topologies,
        result.target,
    )
    voltage = np.linspace(-100.0, 60.0, 321)
    figure, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.2), constrained_layout=True
    )
    colors = {"m": "#d62828", "h": "#457b9d", "n": "#2a9d8f"}
    for gate in ("m", "h", "n"):
        soma_inf, soma_tau = model.soma.base.kinetics.gate_curves(
            gate, voltage
        )
        ais_inf, ais_tau = model.ais_kinetics.gate_curves(
            gate, voltage
        )
        axes[0, 0].plot(
            voltage,
            soma_inf,
            color=colors[gate],
            label=f"soma {gate}",
        )
        axes[0, 0].plot(
            voltage,
            ais_inf,
            color=colors[gate],
            linestyle="--",
            label=f"AIS {gate}",
        )
        axes[0, 1].plot(voltage, soma_tau, color=colors[gate])
        axes[0, 1].plot(
            voltage, ais_tau, color=colors[gate], linestyle="--"
        )
    for compartment, slow_na, slow_k, linestyle in (
        (
            "soma",
            model.soma.slow_na_kinetics,
            model.soma.slow_k_kinetics,
            "-",
        ),
        (
            "AIS",
            model.ais_slow_na_kinetics,
            model.ais_slow_k_kinetics,
            "--",
        ),
    ):
        unavailable, slow_na_tau = slow_na.gate_curves(
            voltage, result.target.temperature_c
        )
        slow_k_inf, slow_k_tau = slow_k.gate_curves(
            voltage, result.target.temperature_c
        )
        axes[1, 0].plot(
            voltage,
            1.0 - unavailable,
            color="#7b2cbf",
            linestyle=linestyle,
            label=f"{compartment} slow Na",
        )
        axes[1, 0].plot(
            voltage,
            slow_k_inf,
            color="#f4a261",
            linestyle=linestyle,
            label=f"{compartment} slow K",
        )
        axes[1, 1].plot(
            voltage,
            slow_na_tau,
            color="#7b2cbf",
            linestyle=linestyle,
        )
        axes[1, 1].plot(
            voltage,
            slow_k_tau,
            color="#f4a261",
            linestyle=linestyle,
        )
    model_names = ("Soma", "AIS")
    topology_values = (
        result.selected.topologies.soma,
        result.selected.topologies.ais,
    )
    state_counts = [
        topology.active_state_count for topology in topology_values
    ]
    axes[0, 2].bar(
        model_names, state_counts, color=("#147d7e", "#9c2c77")
    )
    axes[0, 2].set_ylabel("Independent gate states")
    for index, topology in enumerate(topology_values):
        axes[0, 2].text(
            index,
            state_counts[index] + 0.08,
            topology.label.replace("_", "\n"),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    conductance_labels = ("NaF", "KF", "NaS", "KS")
    soma_g = (
        model.soma.base.biophysics.gna_ms_cm2,
        model.soma.base.biophysics.gk_ms_cm2,
        (
            model.soma.gslow_na_ms_cm2
            if result.selected.topologies.soma.has_slow_na
            else 0.0
        ),
        (
            model.soma.gslow_k_ms_cm2
            if result.selected.topologies.soma.has_slow_k
            else 0.0
        ),
    )
    ais_g = (
        model.ais_gna_fast_ms_cm2,
        model.ais_gk_fast_ms_cm2,
        (
            model.ais_gslow_na_ms_cm2
            if result.selected.topologies.ais.has_slow_na
            else 0.0
        ),
        (
            model.ais_gslow_k_ms_cm2
            if result.selected.topologies.ais.has_slow_k
            else 0.0
        ),
    )
    x = np.arange(4)
    axes[1, 2].bar(
        x - 0.18, np.maximum(soma_g, 1e-4), 0.36,
        color="#147d7e", label="Soma"
    )
    axes[1, 2].bar(
        x + 0.18, np.maximum(ais_g, 1e-4), 0.36,
        color="#9c2c77", label="AIS"
    )
    axes[1, 2].set_xticks(x, conductance_labels)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("Conductance density (mS/cm2)")
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Fast steady state")
    axes[0, 1].set_ylabel("Fast tau at 6.3 C (ms)")
    axes[1, 0].set_ylabel("Slow steady state")
    axes[1, 1].set_ylabel(
        f"Slow tau at {result.target.temperature_c:g} C (ms)"
    )
    axes[0, 1].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 0].legend(frameon=False, fontsize=7)
    for axis in axes[:, :2].flat:
        axis.set_xlabel("V (mV)")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Selected soma-AIS kinetics: {result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_two_compartment_fit_result(
    result: TwoCompartmentFitResult,
    screen_config: BiologicalScreenConfig,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    two_compartment_candidate_summary(result).to_csv(
        output / "candidate_summary.csv", index=False
    )
    result.topology_screen.to_csv(
        output / "ais_topology_screen.csv", index=False
    )
    result.history.to_csv(
        output / "optimization_history.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "label": result.selected.label,
                **result.selected.topologies.to_mapping(),
                **dict(
                    zip(
                        TWO_COMPARTMENT_PARAMETER_NAMES,
                        result.selected.parameters,
                    )
                ),
            }
        ]
    ).to_csv(output / "selected_parameters.csv", index=False)
    two_compartment_feature_comparison(result).to_csv(
        output / "firing_pattern_comparison.csv", index=False
    )
    metadata = {
        "version": 1,
        "experiment": (
            "independent soma-AIS generic fast/slow Na/K currents"
        ),
        "target": target_metadata(result.target),
        "selected_label": result.selected.label,
        "selected_topologies": {
            "soma": asdict(result.selected.topologies.soma),
            "ais": asdict(result.selected.topologies.ais),
        },
        "fit_config": asdict(result.config),
        "phase_shape_config": asdict(result.shape_config),
        "screen_config": asdict(screen_config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_two_compartment_phase_plot(
        result, output / "soma_ais_phase_comparison.png"
    )
    save_two_compartment_trace_plot(
        result,
        screen_config,
        output / "soma_ais_trace_comparison.png",
    )
    save_two_compartment_score_plot(
        result, output / "soma_ais_score_bars.png"
    )
    save_two_compartment_kinetics_plot(
        result, output / "soma_ais_kinetics.png"
    )
