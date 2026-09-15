"""Minimal HH model extensions and CMA-ES ladder comparisons."""

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
from scipy.integrate import solve_ivp

from .cell_optimization import (
    CELL_PARAMETER_NAMES,
    FEATURE_GROUPS,
    OBJECTIVE_NAMES,
    CandidateEvaluation,
    CellOptimizationConfig,
    _group_score,
    _huber,
    _mean_protocol_group_score,
    evaluate_cell_candidate,
    representative_feature_comparison,
)
from .cell_targets import (
    anchored_biophysics,
    CellOptimizationTarget,
    passive_leak_from_active_current,
    target_metadata,
)
from .features import FeatureConfig, extract_observed_features
from .hh_model import SimulationConfig, SimulationError, Stimulus, Trace
from .kinetics import KineticParameters, PARAMETER_NAMES
from .protocols import (
    BiologicalScreenConfig,
    step_simulation_config,
    subthreshold_features,
)
from .raw_patchseq import read_current_clamp_sweeps


LADDER_VARIANTS = ("na-slow", "k-slow", "soma-ais")

SLOW_GATE_PARAMETER_NAMES = tuple(
    f"param__extension__{rate}__{field}"
    for rate in ("alpha", "beta")
    for field in ("log_rate_scale", "voltage_shift_mv", "log_slope_scale")
)
K_SLOW_PARAMETER_NAMES = (
    *SLOW_GATE_PARAMETER_NAMES,
    "param__extension__log_gslow_ms_cm2",
)
AIS_PARAMETER_NAMES = (
    "param__extension__log_ais_area_scale",
    "param__extension__log_ais_gna_multiplier_scale",
    "param__extension__log_coupling_scale",
)

VARIANT_PARAMETER_NAMES = {
    "na-slow": SLOW_GATE_PARAMETER_NAMES,
    "k-slow": K_SLOW_PARAMETER_NAMES,
    "soma-ais": AIS_PARAMETER_NAMES,
}

_SCALAR_WEIGHTS = np.asarray((1.0, 1.5, 1.0, 1.0, 1.0, 2.0, 0.25))


@dataclass(frozen=True)
class EffectiveGate:
    """Complementary alpha/beta curves with transformed rates."""

    mode: str
    alpha: tuple[float, float, float]
    beta: tuple[float, float, float]
    vhalf_mv: float
    slope_mv: float
    tau_reference_ms: float
    q10: float = 1.5
    reference_temperature_c: float = 6.3

    def _base_rate(
        self,
        branch: str,
        voltage_mv: float,
    ) -> float:
        alpha_activation = self.mode == "activation"
        increasing = alpha_activation if branch == "alpha" else not alpha_activation
        signed = (
            (voltage_mv - self.vhalf_mv) / self.slope_mv
            if increasing
            else -(voltage_mv - self.vhalf_mv) / self.slope_mv
        )
        sigmoid = 1.0 / (1.0 + math.exp(max(-80.0, min(80.0, -signed))))
        return sigmoid / self.tau_reference_ms

    def rate(self, branch: str, voltage_mv: float) -> float:
        transform = self.alpha if branch == "alpha" else self.beta
        log_rate_scale, voltage_shift_mv, log_slope_scale = transform
        transformed_voltage = self.vhalf_mv + (
            voltage_mv - self.vhalf_mv - voltage_shift_mv
        ) / math.exp(log_slope_scale)
        return math.exp(log_rate_scale) * self._base_rate(
            branch,
            transformed_voltage,
        )

    def rates(self, voltage_mv: float, temperature_c: float) -> tuple[float, float]:
        temperature_factor = self.q10 ** (
            (temperature_c - self.reference_temperature_c) / 10.0
        )
        return (
            temperature_factor * self.rate("alpha", voltage_mv),
            temperature_factor * self.rate("beta", voltage_mv),
        )

    def steady_state(self, voltage_mv: float) -> float:
        alpha = self.rate("alpha", voltage_mv)
        beta = self.rate("beta", voltage_mv)
        return alpha / (alpha + beta)


@dataclass(frozen=True)
class DecodedLadderModel:
    variant: str
    kinetics: KineticParameters
    biophysics: object
    extra_gate: EffectiveGate | None
    gslow_ms_cm2: float
    ais_area_fraction: float
    ais_gna_multiplier: float
    coupling_ns: float
    ais_eleak_mv: float
    extension_prior: float
    ais_kinetics: object | None = None


@dataclass(frozen=True)
class LadderOptimizationConfig:
    population_size: int = 8
    generations: int = 4
    workers: int = 1
    seed: int = 42
    sigma: float = 0.55
    invalid_score: float = 10_000.0
    gna_gk_factor: float = 1.5
    viability_step_ms: float = 250.0


@dataclass(frozen=True)
class LadderFitResult:
    variant: str
    target: CellOptimizationTarget
    parameter_names: tuple[str, ...]
    history: pd.DataFrame
    best_parameters: pd.DataFrame
    evaluation: CandidateEvaluation
    feature_comparison: pd.DataFrame
    validation: pd.DataFrame
    metadata: Mapping[str, object]


def extension_parameter_bounds(
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant not in LADDER_VARIANTS:
        raise ValueError(f"Unknown ladder variant: {variant}")
    if variant in ("na-slow", "k-slow"):
        lower = []
        upper = []
        for _ in ("alpha", "beta"):
            lower.extend((-math.log(4.0), -15.0, -math.log(2.0)))
            upper.extend((math.log(4.0), 15.0, math.log(2.0)))
        initial = [0.0] * 6
        if variant == "k-slow":
            lower.append(math.log(0.05))
            upper.append(math.log(20.0))
            initial.append(0.0)
        return (
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            np.asarray(initial, dtype=float),
        )
    lower = np.log(np.asarray((0.4, 0.5, 0.1)))
    upper = np.log(np.asarray((3.0, 10.0 / 3.0, 4.0)))
    initial = np.zeros(3, dtype=float)
    return lower, upper, initial


def _slow_gate(variant: str, values: np.ndarray) -> EffectiveGate:
    if variant == "na-slow":
        return EffectiveGate(
            mode="inactivation",
            alpha=tuple(values[:3]),
            beta=tuple(values[3:6]),
            vhalf_mv=-45.0,
            slope_mv=8.0,
            tau_reference_ms=500.0,
        )
    return EffectiveGate(
        mode="activation",
        alpha=tuple(values[:3]),
        beta=tuple(values[3:6]),
        vhalf_mv=-35.0,
        slope_mv=10.0,
        tau_reference_ms=200.0,
    )


def _steady_currents(
    voltage_mv: float,
    model: DecodedLadderModel,
    compartment: str = "soma",
) -> tuple[float, float, float]:
    kinetics = (
        model.ais_kinetics
        if compartment == "ais" and model.ais_kinetics is not None
        else model.kinetics
    )
    m_gate, h_gate, n_gate = kinetics.steady_state(voltage_mv)
    gna = model.biophysics.gna_ms_cm2
    eleak = model.biophysics.eleak_mv
    if compartment == "ais":
        gna *= model.ais_gna_multiplier
        eleak = model.ais_eleak_mv
    sodium_availability = 1.0
    if model.variant == "na-slow":
        sodium_availability = model.extra_gate.steady_state(voltage_mv)
    i_na = (
        gna
        * m_gate**3
        * h_gate
        * sodium_availability
        * (voltage_mv - model.biophysics.ena_mv)
    )
    i_k = (
        model.biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage_mv - model.biophysics.ek_mv)
    )
    if model.variant == "k-slow":
        slow_gate = model.extra_gate.steady_state(voltage_mv)
        i_k += (
            model.gslow_ms_cm2
            * slow_gate
            * (voltage_mv - model.biophysics.ek_mv)
        )
    i_leak = model.biophysics.gleak_ms_cm2 * (voltage_mv - eleak)
    return i_na, i_k, i_leak


def decode_ladder_model(
    variant: str,
    base_vector: Sequence[float],
    extension_values: Sequence[float],
    target: CellOptimizationTarget,
    gna_gk_factor: float = 1.5,
) -> DecodedLadderModel:
    values = np.asarray(extension_values, dtype=float)
    parameter_names = VARIANT_PARAMETER_NAMES.get(variant)
    if parameter_names is None or values.shape != (len(parameter_names),):
        raise ValueError(f"Invalid extension vector for {variant}: {values.shape}")
    base = np.asarray(base_vector, dtype=float)
    if base.shape != (len(CELL_PARAMETER_NAMES),):
        raise ValueError("Base vector must contain the 20 fitted cell parameters")
    kinetics = KineticParameters.from_vector(base[: len(PARAMETER_NAMES)])
    biophysics = anchored_biophysics(
        target.passive,
        kinetics,
        log_gna_scale=float(base[-2]),
        log_gk_scale=float(base[-1]),
    )
    gate = _slow_gate(variant, values) if variant != "soma-ais" else None
    gslow = math.exp(values[6]) if variant == "k-slow" else 0.0
    area_fraction = 0.05
    gna_multiplier = 3.0
    coupling_ns = 5.0
    if variant == "soma-ais":
        area_fraction *= math.exp(values[0])
        gna_multiplier *= math.exp(values[1])
        coupling_ns *= math.exp(values[2])

    temporary = DecodedLadderModel(
        variant=variant,
        kinetics=kinetics,
        biophysics=biophysics,
        extra_gate=gate,
        gslow_ms_cm2=gslow,
        ais_area_fraction=area_fraction,
        ais_gna_multiplier=gna_multiplier,
        coupling_ns=coupling_ns,
        ais_eleak_mv=biophysics.eleak_mv,
        extension_prior=0.0,
    )
    voltage = target.passive.resting_voltage_mv
    def active_current(test_voltage: float) -> float:
        i_na, i_k, _ = _steady_currents(test_voltage, temporary)
        return i_na + i_k

    gleak, soma_eleak = passive_leak_from_active_current(
        target.passive,
        active_current,
    )
    biophysics = replace(
        biophysics,
        gleak_ms_cm2=float(gleak),
        eleak_mv=float(soma_eleak),
    )
    temporary = replace(temporary, biophysics=biophysics)
    ais_eleak = soma_eleak
    if variant == "soma-ais":
        ais_na, ais_k, _ = _steady_currents(
            voltage,
            temporary,
            compartment="ais",
        )
        ais_eleak = voltage + (
            ais_na + ais_k
        ) / biophysics.gleak_ms_cm2

    lower, upper, initial = extension_parameter_bounds(variant)
    half_width = (upper - lower) / 2.0
    normalized = (values - initial) / half_width
    extension_prior = float(np.mean(normalized**2))
    nuisance_width = math.log(gna_gk_factor)
    base_prior = float(np.mean((base[-2:] / nuisance_width) ** 2))
    return replace(
        temporary,
        ais_eleak_mv=float(ais_eleak),
        extension_prior=base_prior + 0.25 * extension_prior,
    )


def _model_is_stable(
    model: DecodedLadderModel,
    target: CellOptimizationTarget,
) -> bool:
    voltage = target.passive.resting_voltage_mv
    delta = 0.05
    compartments = (
        ("soma", "ais")
        if model.variant == "soma-ais"
        else ("soma",)
    )
    for compartment in compartments:
        def current(test_voltage: float) -> float:
            return float(
                sum(_steady_currents(test_voltage, model, compartment))
            )

        slope = (
            current(voltage + delta) - current(voltage - delta)
        ) / (2.0 * delta)
        if not np.isfinite(slope) or slope <= 0.0:
            return False
    return True


def _initial_state(
    model: DecodedLadderModel,
    target: CellOptimizationTarget,
) -> np.ndarray:
    voltage = target.passive.resting_voltage_mv
    m_gate, h_gate, n_gate = model.kinetics.steady_state(voltage)
    if model.variant == "soma-ais":
        ais_kinetics = model.ais_kinetics or model.kinetics
        am_gate, ah_gate, an_gate = ais_kinetics.steady_state(voltage)
        return np.asarray(
            (
                voltage,
                m_gate,
                h_gate,
                n_gate,
                voltage,
                am_gate,
                ah_gate,
                an_gate,
            ),
            dtype=float,
        )
    slow_gate = model.extra_gate.steady_state(voltage)
    return np.asarray(
        (voltage, m_gate, h_gate, n_gate, slow_gate),
        dtype=float,
    )


def _one_compartment_currents(
    state: np.ndarray,
    model: DecodedLadderModel,
) -> tuple[float, float, float, float]:
    voltage, m_gate, h_gate, n_gate, slow_gate = state
    sodium_availability = slow_gate if model.variant == "na-slow" else 1.0
    i_na = (
        model.biophysics.gna_ms_cm2
        * m_gate**3
        * h_gate
        * sodium_availability
        * (voltage - model.biophysics.ena_mv)
    )
    i_k = (
        model.biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage - model.biophysics.ek_mv)
    )
    if model.variant == "k-slow":
        i_k += (
            model.gslow_ms_cm2
            * slow_gate
            * (voltage - model.biophysics.ek_mv)
        )
    i_leak = (
        model.biophysics.gleak_ms_cm2
        * (voltage - model.biophysics.eleak_mv)
    )
    return i_na, i_k, i_leak, i_na + i_k + i_leak


def _fast_gate_derivatives(
    voltage_mv: float,
    m_gate: float,
    h_gate: float,
    n_gate: float,
    model: DecodedLadderModel,
    temperature_c: float,
    kinetics: object | None = None,
) -> tuple[float, float, float]:
    rates = (kinetics or model.kinetics).rates_scalar(voltage_mv)
    factor_m, factor_h, factor_n = model.biophysics.temperature_factors(
        temperature_c
    )
    return (
        factor_m * (rates[0] * (1.0 - m_gate) - rates[1] * m_gate),
        factor_h * (rates[2] * (1.0 - h_gate) - rates[3] * h_gate),
        factor_n * (rates[4] * (1.0 - n_gate) - rates[5] * n_gate),
    )


def _ladder_rhs(
    time_ms: float,
    state: np.ndarray,
    model: DecodedLadderModel,
    config: SimulationConfig,
) -> np.ndarray:
    current_pa = config.stimulus.pa_value(time_ms, model.biophysics)
    if model.variant != "soma-ais":
        voltage, m_gate, h_gate, n_gate, slow_gate = state
        _, _, _, i_ion = _one_compartment_currents(state, model)
        applied_density = float(
            model.biophysics.current_pa_to_density(current_pa)
        )
        d_voltage = (
            applied_density - i_ion
        ) / model.biophysics.capacitance_uf_cm2
        d_m, d_h, d_n = _fast_gate_derivatives(
            voltage,
            m_gate,
            h_gate,
            n_gate,
            model,
            config.temperature_c,
        )
        alpha_slow, beta_slow = model.extra_gate.rates(
            voltage,
            config.temperature_c,
        )
        d_slow = alpha_slow * (1.0 - slow_gate) - beta_slow * slow_gate
        return np.asarray((d_voltage, d_m, d_h, d_n, d_slow))

    soma_voltage, sm, sh, sn, ais_voltage, am, ah, an = state
    soma_state = np.asarray((soma_voltage, sm, sh, sn, 1.0))
    i_na_s, i_k_s, i_leak_s, i_ion_s = _one_compartment_currents(
        soma_state,
        replace(model, variant="na-slow", extra_gate=None),
    )
    i_na_a = (
        model.biophysics.gna_ms_cm2
        * model.ais_gna_multiplier
        * am**3
        * ah
        * (ais_voltage - model.biophysics.ena_mv)
    )
    i_k_a = (
        model.biophysics.gk_ms_cm2
        * an**4
        * (ais_voltage - model.biophysics.ek_mv)
    )
    i_leak_a = (
        model.biophysics.gleak_ms_cm2
        * (ais_voltage - model.ais_eleak_mv)
    )
    i_ion_a = i_na_a + i_k_a + i_leak_a
    soma_fraction = 1.0 - model.ais_area_fraction
    soma_area_um2 = model.biophysics.membrane_area_um2 * soma_fraction
    ais_area_um2 = (
        model.biophysics.membrane_area_um2 * model.ais_area_fraction
    )
    coupling_pa = model.coupling_ns * (soma_voltage - ais_voltage)
    soma_density = current_pa / (soma_area_um2 * 0.01)
    soma_coupling_density = coupling_pa / (soma_area_um2 * 0.01)
    ais_coupling_density = coupling_pa / (ais_area_um2 * 0.01)
    d_soma = (
        soma_density - i_ion_s - soma_coupling_density
    ) / model.biophysics.capacitance_uf_cm2
    d_ais = (
        -i_ion_a + ais_coupling_density
    ) / model.biophysics.capacitance_uf_cm2
    d_sm, d_sh, d_sn = _fast_gate_derivatives(
        soma_voltage,
        sm,
        sh,
        sn,
        model,
        config.temperature_c,
    )
    d_am, d_ah, d_an = _fast_gate_derivatives(
        ais_voltage,
        am,
        ah,
        an,
        model,
        config.temperature_c,
        kinetics=model.ais_kinetics,
    )
    return np.asarray(
        (d_soma, d_sm, d_sh, d_sn, d_ais, d_am, d_ah, d_an)
    )


def _validate_state(
    state: np.ndarray,
    variant: str,
    time_ms: float,
) -> None:
    voltage_indices = (0, 4) if variant == "soma-ais" else (0,)
    gate_indices = [
        index
        for index in range(len(state))
        if index not in voltage_indices
    ]
    if (
        not np.all(np.isfinite(state))
        or any(abs(state[index]) > 200.0 for index in voltage_indices)
        or np.any(state[gate_indices] < -0.05)
        or np.any(state[gate_indices] > 1.05)
    ):
        raise SimulationError(f"Integration diverged at t={time_ms:.4f} ms")


def simulate_ladder(
    model: DecodedLadderModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> Trace:
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, len(initial_state)), dtype=float)
    states[0] = initial_state
    dt = config.dt_ms
    for index in range(n_steps):
        time = time_ms[index]
        state = states[index]
        k1 = _ladder_rhs(time, state, model, config)
        k2 = _ladder_rhs(
            time + dt / 2.0,
            state + dt * k1 / 2.0,
            model,
            config,
        )
        k3 = _ladder_rhs(
            time + dt / 2.0,
            state + dt * k2 / 2.0,
            model,
            config,
        )
        k4 = _ladder_rhs(
            time + dt,
            state + dt * k3,
            model,
            config,
        )
        next_state = state + dt * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        _validate_state(next_state, model.variant, time + dt)
        voltage_indices = (0, 4) if model.variant == "soma-ais" else (0,)
        gate_indices = [
            gate_index
            for gate_index in range(len(next_state))
            if gate_index not in voltage_indices
        ]
        next_state[gate_indices] = np.clip(
            next_state[gate_indices],
            0.0,
            1.0,
        )
        states[index + 1] = next_state

    applied_pa = np.asarray(
        [
            config.stimulus.pa_value(time, model.biophysics)
            for time in time_ms
        ]
    )
    if model.variant == "soma-ais":
        soma_area_um2 = model.biophysics.membrane_area_um2 * (
            1.0 - model.ais_area_fraction
        )
        applied_density = applied_pa / (soma_area_um2 * 0.01)
        voltage = states[:, 0]
        m_gate, h_gate, n_gate = states[:, 1], states[:, 2], states[:, 3]
        current_rows = []
        for state in states:
            soma_state = np.asarray((state[0], state[1], state[2], state[3], 1.0))
            current_rows.append(
                _one_compartment_currents(
                    soma_state,
                    replace(model, variant="na-slow", extra_gate=None),
                )
            )
    else:
        applied_density = model.biophysics.current_pa_to_density(applied_pa)
        voltage = states[:, 0]
        m_gate, h_gate, n_gate = states[:, 1], states[:, 2], states[:, 3]
        current_rows = [
            _one_compartment_currents(state, model)
            for state in states
        ]
    i_na, i_k, i_leak, i_ion = np.asarray(current_rows).T
    dvdt = np.asarray(
        [
            _ladder_rhs(time, state, model, config)[0]
            for time, state in zip(time_ms, states)
        ]
    )
    return Trace(
        time_ms=time_ms,
        voltage_mv=voltage,
        m=m_gate,
        h=h_gate,
        n=n_gate,
        applied_current_ua_cm2=np.asarray(applied_density),
        applied_current_pa=applied_pa,
        ionic_current_ua_cm2=i_ion,
        ionic_current_pa=model.biophysics.current_density_to_pa(i_ion),
        sodium_current_ua_cm2=i_na,
        potassium_current_ua_cm2=i_k,
        leak_current_ua_cm2=i_leak,
        dvdt_mv_ms=dvdt,
    )


def simulate_ladder_adaptive(
    model: DecodedLadderModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
    max_step_ms: float = 0.50,
) -> Trace:
    """Simulate a ladder model with adaptive integration on a fixed output grid."""
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    solution = solve_ivp(
        lambda time, state: _ladder_rhs(
            time,
            state,
            model,
            config,
        ),
        (0.0, config.duration_ms),
        np.asarray(initial_state, dtype=float),
        method="LSODA",
        t_eval=time_ms,
        rtol=1e-6,
        atol=np.full(len(initial_state), 1e-8),
        max_step=max_step_ms,
    )
    if (
        not solution.success
        or solution.y.shape != (len(initial_state), len(time_ms))
        or not np.all(np.isfinite(solution.y))
    ):
        raise SimulationError(
            f"Adaptive ladder integration failed: {solution.message}"
        )
    states = np.asarray(solution.y.T, dtype=float)
    voltage_indices = (0, 4) if model.variant == "soma-ais" else (0,)
    gate_indices = [
        index
        for index in range(states.shape[1])
        if index not in voltage_indices
    ]
    if (
        np.any(np.abs(states[:, list(voltage_indices)]) > 200.0)
        or np.any(states[:, gate_indices] < -0.05)
        or np.any(states[:, gate_indices] > 1.05)
    ):
        raise SimulationError(
            "Adaptive ladder integration left the physiological range"
        )
    states[:, gate_indices] = np.clip(
        states[:, gate_indices],
        0.0,
        1.0,
    )

    applied_pa = np.asarray(
        [
            config.stimulus.pa_value(time, model.biophysics)
            for time in time_ms
        ]
    )
    if model.variant == "soma-ais":
        soma_area_um2 = model.biophysics.membrane_area_um2 * (
            1.0 - model.ais_area_fraction
        )
        applied_density = applied_pa / (soma_area_um2 * 0.01)
        voltage = states[:, 0]
        m_gate, h_gate, n_gate = states[:, 1], states[:, 2], states[:, 3]
        current_rows = []
        for state in states:
            soma_state = np.asarray(
                (state[0], state[1], state[2], state[3], 1.0)
            )
            current_rows.append(
                _one_compartment_currents(
                    soma_state,
                    replace(model, variant="na-slow", extra_gate=None),
                )
            )
    else:
        applied_density = model.biophysics.current_pa_to_density(applied_pa)
        voltage = states[:, 0]
        m_gate, h_gate, n_gate = states[:, 1], states[:, 2], states[:, 3]
        current_rows = [
            _one_compartment_currents(state, model)
            for state in states
        ]
    i_na, i_k, i_leak, i_ion = np.asarray(current_rows).T
    dvdt = np.asarray(
        [
            _ladder_rhs(time, state, model, config)[0]
            for time, state in zip(time_ms, states)
        ]
    )
    return Trace(
        time_ms=time_ms,
        voltage_mv=voltage,
        m=m_gate,
        h=h_gate,
        n=n_gate,
        applied_current_ua_cm2=np.asarray(applied_density),
        applied_current_pa=applied_pa,
        ionic_current_ua_cm2=i_ion,
        ionic_current_pa=model.biophysics.current_density_to_pa(i_ion),
        sodium_current_ua_cm2=i_na,
        potassium_current_ua_cm2=i_k,
        leak_current_ua_cm2=i_leak,
        dvdt_mv_ms=dvdt,
    )


def _rush_larsen_gate_update(
    voltage_mv: float,
    gates: np.ndarray,
    kinetics: object,
    model: DecodedLadderModel,
    temperature_c: float,
    duration_ms: float,
) -> np.ndarray:
    rates = kinetics.rates_scalar(voltage_mv)
    factors = model.biophysics.temperature_factors(temperature_c)
    updated = np.empty(3, dtype=float)
    for index, (gate, factor) in enumerate(zip(gates, factors)):
        alpha = float(rates[2 * index])
        beta = float(rates[2 * index + 1])
        total = max(alpha + beta, 1e-12)
        steady = alpha / total
        decay = math.exp(
            max(-80.0, -float(factor) * total * duration_ms)
        )
        updated[index] = steady + (float(gate) - steady) * decay
    return np.clip(updated, 0.0, 1.0)


def _rush_larsen_half_step(
    state: np.ndarray,
    model: DecodedLadderModel,
    config: SimulationConfig,
    duration_ms: float,
) -> np.ndarray:
    if model.variant != "soma-ais":
        raise ValueError(
            "Rush-Larsen ladder integration currently requires soma-ais"
        )
    updated = state.copy()
    updated[1:4] = _rush_larsen_gate_update(
        float(state[0]),
        state[1:4],
        model.kinetics,
        model,
        config.temperature_c,
        duration_ms,
    )
    updated[5:8] = _rush_larsen_gate_update(
        float(state[4]),
        state[5:8],
        model.ais_kinetics or model.kinetics,
        model,
        config.temperature_c,
        duration_ms,
    )
    return updated


def _rush_larsen_step(
    time_ms: float,
    state: np.ndarray,
    model: DecodedLadderModel,
    config: SimulationConfig,
) -> np.ndarray:
    dt = config.dt_ms
    voltage_indices = np.asarray((0, 4), dtype=int)
    half = _rush_larsen_half_step(
        state,
        model,
        config,
        dt / 2.0,
    )

    def voltage_rhs(
        rhs_time: float,
        voltages: np.ndarray,
    ) -> np.ndarray:
        trial = half.copy()
        trial[voltage_indices] = voltages
        return _ladder_rhs(
            rhs_time,
            trial,
            model,
            config,
        )[voltage_indices]

    voltage = half[voltage_indices]
    k1 = voltage_rhs(time_ms, voltage)
    k2 = voltage_rhs(
        time_ms + dt / 2.0,
        voltage + dt * k1 / 2.0,
    )
    k3 = voltage_rhs(
        time_ms + dt / 2.0,
        voltage + dt * k2 / 2.0,
    )
    k4 = voltage_rhs(time_ms + dt, voltage + dt * k3)
    half[voltage_indices] = voltage + dt * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0
    next_state = _rush_larsen_half_step(
        half,
        model,
        config,
        dt / 2.0,
    )
    _validate_state(next_state, model.variant, time_ms + dt)
    return next_state


def simulate_ladder_rush_larsen(
    model: DecodedLadderModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> Trace:
    """Use analytic gate steps and RK4 voltage steps for soma-AIS models."""
    if model.variant != "soma-ais":
        raise ValueError(
            "Rush-Larsen ladder integration currently requires soma-ais"
        )
    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, len(initial_state)), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for index in range(n_steps):
        states[index + 1] = _rush_larsen_step(
            time_ms[index],
            states[index],
            model,
            config,
        )

    applied_pa = np.asarray(
        [
            config.stimulus.pa_value(time, model.biophysics)
            for time in time_ms
        ]
    )
    soma_area_um2 = model.biophysics.membrane_area_um2 * (
        1.0 - model.ais_area_fraction
    )
    applied_density = applied_pa / (soma_area_um2 * 0.01)
    voltage = states[:, 0]
    m_gate, h_gate, n_gate = states[:, 1], states[:, 2], states[:, 3]
    current_rows = []
    for state in states:
        soma_state = np.asarray(
            (state[0], state[1], state[2], state[3], 1.0)
        )
        current_rows.append(
            _one_compartment_currents(
                soma_state,
                replace(model, variant="na-slow", extra_gate=None),
            )
        )
    i_na, i_k, i_leak, i_ion = np.asarray(current_rows).T
    dvdt = np.asarray(
        [
            _ladder_rhs(time, state, model, config)[0]
            for time, state in zip(time_ms, states)
        ]
    )
    return Trace(
        time_ms=time_ms,
        voltage_mv=voltage,
        m=m_gate,
        h=h_gate,
        n=n_gate,
        applied_current_ua_cm2=np.asarray(applied_density),
        applied_current_pa=applied_pa,
        ionic_current_ua_cm2=i_ion,
        ionic_current_pa=model.biophysics.current_density_to_pa(i_ion),
        sodium_current_ua_cm2=i_na,
        potassium_current_ua_cm2=i_k,
        leak_current_ua_cm2=i_leak,
        dvdt_mv_ms=dvdt,
    )


def elicits_spike_ladder_rush_larsen(
    model: DecodedLadderModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
) -> bool:
    """Return on the first soma zero crossing using Rush-Larsen steps."""
    state = np.asarray(initial_state, dtype=float)
    n_steps = int(round(config.duration_ms / config.dt_ms))
    for index in range(n_steps):
        previous_voltage = float(state[0])
        state = _rush_larsen_step(
            index * config.dt_ms,
            state,
            model,
            config,
        )
        if previous_voltage < 0.0 <= state[0]:
            return True
    return False


def find_rheobase_rush_larsen(
    model: DecodedLadderModel,
    initial_state: np.ndarray,
    screen_config: BiologicalScreenConfig,
    duration_ms: float = 300.0,
    tolerance_pa: float = 5.0,
) -> tuple[float, float, int]:
    """Estimate rheobase with the same stable integrator used for phase fits."""
    maximum_current = float(screen_config.rheobase_currents_pa[-1])

    def spikes(current_pa: float) -> bool:
        simulation = step_simulation_config(
            current_pa,
            min(duration_ms, screen_config.rheobase_step_ms),
            screen_config,
            model.biophysics,
            float(initial_state[0]),
        )
        return elicits_spike_ladder_rush_larsen(
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


def elicits_spike_ladder(
    model: DecodedLadderModel,
    config: SimulationConfig,
    initial_state: np.ndarray,
    max_step_ms: float = 0.5,
) -> bool:
    def rhs(time_ms: float, state: np.ndarray) -> np.ndarray:
        return _ladder_rhs(time_ms, state, model, config)

    def spike_crossing(_time_ms: float, state: np.ndarray) -> float:
        return float(state[0])

    spike_crossing.terminal = True
    spike_crossing.direction = 1.0
    solution = solve_ivp(
        rhs,
        (config.stimulus.start_ms, config.stimulus.end_ms),
        initial_state,
        events=spike_crossing,
        rtol=1e-5,
        atol=np.full(len(initial_state), 1e-7),
        max_step=max_step_ms,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise SimulationError(f"Adaptive spike test failed: {solution.message}")
    if np.any(np.abs(solution.y[[0] if model.variant != "soma-ais" else [0, 4]]) > 200.0):
        raise SimulationError("Adaptive spike test left the physiological range")
    return bool(len(solution.t_events[0]))


def _find_rheobase(
    model: DecodedLadderModel,
    initial_state: np.ndarray,
    screen_config: BiologicalScreenConfig,
) -> tuple[float, float, int]:
    currents = screen_config.rheobase_currents_pa
    spike_cache: dict[int, bool] = {}

    def spikes_at(index: int) -> bool:
        if index not in spike_cache:
            simulation = step_simulation_config(
                currents[index],
                screen_config.rheobase_step_ms,
                screen_config,
                model.biophysics,
                float(initial_state[0]),
            )
            spike_cache[index] = elicits_spike_ladder(
                model,
                simulation,
                initial_state,
                screen_config.spike_test_max_step_ms,
            )
        return spike_cache[index]

    lower_index = -1
    upper_index = 0
    if not spikes_at(0):
        upper_index = len(currents) - 1
        if not spikes_at(upper_index):
            raise SimulationError("No spike within the rheobase search range")
        lower_index = 0
        while upper_index - lower_index > 1:
            midpoint = (lower_index + upper_index) // 2
            if spikes_at(midpoint):
                upper_index = midpoint
            else:
                lower_index = midpoint
    lower_pa = 0.0 if lower_index < 0 else currents[lower_index]
    upper_pa = currents[upper_index]
    iterations = 0
    while (
        upper_pa - lower_pa > screen_config.rheobase_tolerance_pa
        and iterations < screen_config.rheobase_max_iterations
    ):
        midpoint_pa = (lower_pa + upper_pa) / 2.0
        simulation = step_simulation_config(
            midpoint_pa,
            screen_config.rheobase_step_ms,
            screen_config,
            model.biophysics,
            float(initial_state[0]),
        )
        if elicits_spike_ladder(
            model,
            simulation,
            initial_state,
            screen_config.spike_test_max_step_ms,
        ):
            upper_pa = midpoint_pa
        else:
            lower_pa = midpoint_pa
        iterations += 1
    return float(lower_pa), float(upper_pa), iterations


def _objective_config(config: LadderOptimizationConfig) -> CellOptimizationConfig:
    return CellOptimizationConfig(
        population_size=max(4, config.population_size),
        generations=config.generations,
        workers=config.workers,
        seed=config.seed,
        gna_gk_factor=config.gna_gk_factor,
        invalid_model_penalty=100.0,
        viability_step_ms=config.viability_step_ms,
        passive_reanchoring=True,
    )


def evaluate_ladder_candidate(
    extension_values: Sequence[float],
    variant: str,
    base_vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: LadderOptimizationConfig,
    include_validation: bool = False,
    include_traces: bool = False,
) -> CandidateEvaluation:
    objective_config = _objective_config(config)
    invalid = tuple(
        objective_config.invalid_model_penalty
        for _ in OBJECTIVE_NAMES
    )
    try:
        model = decode_ladder_model(
            variant,
            base_vector,
            extension_values,
            target,
            config.gna_gk_factor,
        )
        if (
            not -120.0 <= model.biophysics.eleak_mv <= -20.0
            or not -120.0 <= model.ais_eleak_mv <= -20.0
        ):
            raise SimulationError("Derived leak reversal is outside range")
        if not _model_is_stable(model, target):
            raise SimulationError("Anchored resting state is unstable")
        initial_state = _initial_state(model, target)
        passive_simulation = step_simulation_config(
            target.passive.passive_current_pa,
            target.passive.passive_duration_ms,
            screen_config,
            model.biophysics,
            target.passive.resting_voltage_mv,
        )
        passive_trace = simulate_ladder(
            model,
            passive_simulation,
            initial_state,
        )
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
        viable = False
        for current_pa in screen_config.rheobase_currents_pa:
            viability_simulation = step_simulation_config(
                current_pa,
                viability_duration,
                screen_config,
                model.biophysics,
                target.passive.resting_voltage_mv,
            )
            if elicits_spike_ladder(
                model,
                viability_simulation,
                initial_state,
                screen_config.spike_test_max_step_ms,
            ):
                viable = True
                break
        if not viable:
            raise SimulationError("No spike within the viability screen")
        rheobase_lower, rheobase_upper, _ = _find_rheobase(
            model,
            initial_state,
            screen_config,
        )

        protocols = list(target.training_protocols)
        if include_validation:
            protocols.extend(target.validation_protocols)
        protocol_features = {}
        traces = {}
        feature_config = FeatureConfig(min_spikes=1)
        for protocol in protocols:
            simulation = step_simulation_config(
                protocol.rheobase_factor * rheobase_upper,
                protocol.duration_ms,
                screen_config,
                model.biophysics,
                target.passive.resting_voltage_mv,
            )
            trace = simulate_ladder(model, simulation, initial_state)
            protocol_features[protocol.name] = extract_observed_features(
                trace,
                simulation.stimulus,
                feature_config,
            )
            if include_traces:
                traces[protocol.name] = trace
        if include_traces:
            traces["passive"] = passive_trace

        passive_score = float(
            np.mean(
                (
                    _huber(
                        (
                            passive["input_resistance_mohm"]
                            - target.passive.input_resistance_mohm
                        )
                        / max(
                            10.0,
                            0.15 * target.passive.input_resistance_mohm,
                        )
                    ),
                    _huber(
                        (
                            passive["membrane_tau_ms"]
                            - target.passive.membrane_tau_ms
                        )
                        / max(1.0, 0.15 * target.passive.membrane_tau_ms)
                    ),
                )
            )
        )
        excitability_score = _huber(
            (rheobase_upper - target.sampled_rheobase_pa)
            / max(10.0, 0.15 * target.sampled_rheobase_pa)
        )
        group_scores = [
            _mean_protocol_group_score(
                protocol_features,
                target.training_protocols,
                group_name,
                objective_config.missing_feature_penalty,
            )
            for group_name in (
                "spike_shape",
                "spike_dynamics",
                "phase_geometry",
                "firing_pattern",
            )
        ]
        objectives = (
            passive_score,
            float(excitability_score),
            *group_scores,
            model.extension_prior,
        )
        if not np.all(np.isfinite(objectives)):
            raise SimulationError("Non-finite ladder objectives")
        mapping = model.biophysics.to_physical_mapping()
        mapping.update(
            {
                "extension__gslow_ms_cm2": model.gslow_ms_cm2,
                "extension__ais_area_fraction": model.ais_area_fraction,
                "extension__ais_gna_multiplier": model.ais_gna_multiplier,
                "extension__coupling_ns": model.coupling_ns,
                "extension__ais_eleak_mv": model.ais_eleak_mv,
            }
        )
        return CandidateEvaluation(
            objectives=tuple(float(value) for value in objectives),
            valid=True,
            reason="accepted",
            rheobase_lower_pa=rheobase_lower,
            rheobase_upper_pa=rheobase_upper,
            biophysics=mapping,
            passive_features=passive,
            protocol_features=protocol_features,
            traces=traces if include_traces else None,
        )
    except (
        FloatingPointError,
        OverflowError,
        SimulationError,
        ValueError,
    ) as error:
        return CandidateEvaluation(
            objectives=invalid,
            valid=False,
            reason=f"{type(error).__name__}: {error}",
        )


def ladder_scalar_score(objectives: Sequence[float]) -> float:
    return float(np.dot(_SCALAR_WEIGHTS, np.asarray(objectives, dtype=float)))


def _bounded_parameters(
    normalized: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    return midpoint + half_width * np.tanh(np.asarray(normalized, dtype=float))


def _unbounded_parameters(
    parameters: Sequence[float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    ratio = np.clip(
        (np.asarray(parameters, dtype=float) - midpoint) / half_width,
        -0.999999,
        0.999999,
    )
    return np.arctanh(ratio)


def _cma_payload(
    normalized: Sequence[float],
    variant: str,
    base_vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: LadderOptimizationConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, tuple[float, ...], bool, str, np.ndarray]:
    parameters = _bounded_parameters(normalized, lower, upper)
    evaluation = evaluate_ladder_candidate(
        parameters,
        variant,
        base_vector,
        target,
        screen_config,
        config,
    )
    score = (
        ladder_scalar_score(evaluation.objectives)
        if evaluation.valid
        else config.invalid_score
    )
    return (
        score,
        evaluation.objectives,
        evaluation.valid,
        evaluation.reason,
        parameters,
    )


def _validation_table(
    evaluation: CandidateEvaluation,
    target: CellOptimizationTarget,
    config: LadderOptimizationConfig,
) -> pd.DataFrame:
    rows = []
    if evaluation.protocol_features is None:
        return pd.DataFrame()
    for protocol in target.validation_protocols:
        model_features = evaluation.protocol_features[protocol.name]
        for group_name, feature_names in FEATURE_GROUPS.items():
            rows.append(
                {
                    "protocol": protocol.name,
                    "objective_group": group_name,
                    "score": _group_score(
                        model_features,
                        protocol.features,
                        feature_names,
                        _objective_config(config).missing_feature_penalty,
                    ),
                }
            )
    return pd.DataFrame(rows)


def optimize_ladder_variant(
    variant: str,
    base_vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: LadderOptimizationConfig | None = None,
    checkpoint_dir: str | Path | None = None,
) -> LadderFitResult:
    try:
        from deap import base, cma, creator
    except ImportError as error:
        raise ImportError("Ladder optimization requires DEAP") from error
    config = config or LadderOptimizationConfig()
    np.random.seed(config.seed)
    lower, upper, initial = extension_parameter_bounds(variant)
    centroid = _unbounded_parameters(initial, lower, upper)

    fitness_name = "FitnessLadderCMA"
    individual_name = "IndividualLadderCMA"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
        )
    individual_type = getattr(creator, individual_name)
    strategy = cma.Strategy(
        centroid=centroid,
        sigma=config.sigma,
        lambda_=config.population_size,
    )

    executor = None
    effective_workers = config.workers
    if config.workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=config.workers)
            mapper = executor.map
        except (OSError, PermissionError) as error:
            effective_workers = 1
            warnings.warn(
                f"Parallel CMA evaluation unavailable ({error}); using one worker.",
                RuntimeWarning,
            )
            mapper = map
    else:
        mapper = map

    rows = []
    try:
        initial_evaluation = evaluate_ladder_candidate(
            initial,
            variant,
            base_vector,
            target,
            screen_config,
            config,
        )
        rows.append(
            {
                "generation": -1,
                "individual": -1,
                **dict(zip(VARIANT_PARAMETER_NAMES[variant], initial)),
                **dict(zip(OBJECTIVE_NAMES, initial_evaluation.objectives)),
                "scalar_score": (
                    ladder_scalar_score(initial_evaluation.objectives)
                    if initial_evaluation.valid
                    else config.invalid_score
                ),
                "valid": initial_evaluation.valid,
                "reason": initial_evaluation.reason,
            }
        )
        for generation in range(config.generations):
            population = strategy.generate(individual_type)
            payloads = list(
                mapper(
                    _cma_payload,
                    population,
                    [variant] * len(population),
                    [base_vector] * len(population),
                    [target] * len(population),
                    [screen_config] * len(population),
                    [config] * len(population),
                    [lower] * len(population),
                    [upper] * len(population),
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
                        **dict(
                            zip(
                                VARIANT_PARAMETER_NAMES[variant],
                                parameters,
                            )
                        ),
                        **dict(zip(OBJECTIVE_NAMES, objectives)),
                        "scalar_score": score,
                        "valid": valid,
                        "reason": reason,
                    }
                )
            strategy.update(population)
            if checkpoint_dir is not None:
                output = Path(checkpoint_dir)
                output.mkdir(parents=True, exist_ok=True)
                history = pd.DataFrame(rows)
                destination = output / "checkpoint_history.csv"
                temporary = destination.with_suffix(".csv.part")
                history.to_csv(temporary, index=False)
                temporary.replace(destination)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    history = pd.DataFrame(rows)
    valid_history = history.loc[history["valid"]].copy()
    if valid_history.empty:
        raise RuntimeError(f"{variant} produced no valid model")
    best = valid_history.nsmallest(1, "scalar_score").copy()
    parameters = best.loc[
        best.index[0],
        list(VARIANT_PARAMETER_NAMES[variant]),
    ].to_numpy(dtype=float)
    evaluation = evaluate_ladder_candidate(
        parameters,
        variant,
        base_vector,
        target,
        screen_config,
        config,
        include_validation=True,
        include_traces=True,
    )
    if not evaluation.valid:
        raise RuntimeError(
            f"Best {variant} model failed deterministic reevaluation: "
            f"{evaluation.reason}"
        )
    feature_comparison = representative_feature_comparison(
        target,
        evaluation,
        _objective_config(config),
    )
    validation = _validation_table(evaluation, target, config)
    metadata = {
        "version": 1,
        "algorithm": "DEAP CMA-ES",
        "variant": variant,
        "frozen_base_parameter_count": len(CELL_PARAMETER_NAMES),
        "optimized_extension_parameter_count": len(parameters),
        "total_model_parameter_count": len(CELL_PARAMETER_NAMES) + len(parameters),
        "parameter_names": list(VARIANT_PARAMETER_NAMES[variant]),
        "optimization_config": asdict(config),
        "effective_workers": effective_workers,
        "screen_config": asdict(screen_config),
        "target": target_metadata(target),
        "best_biophysics": evaluation.biophysics,
        "best_rheobase_lower_pa": evaluation.rheobase_lower_pa,
        "best_rheobase_upper_pa": evaluation.rheobase_upper_pa,
        "scalar_weights": dict(zip(OBJECTIVE_NAMES, _SCALAR_WEIGHTS)),
    }
    return LadderFitResult(
        variant=variant,
        target=target,
        parameter_names=VARIANT_PARAMETER_NAMES[variant],
        history=history,
        best_parameters=best,
        evaluation=evaluation,
        feature_comparison=feature_comparison,
        validation=validation,
        metadata=metadata,
    )


def baseline_ladder_result(
    base_vector: Sequence[float],
    target: CellOptimizationTarget,
    screen_config: BiologicalScreenConfig,
    config: LadderOptimizationConfig,
) -> LadderFitResult:
    evaluation = evaluate_cell_candidate(
        base_vector,
        target,
        screen_config,
        _objective_config(config),
        include_validation=True,
        include_traces=True,
    )
    if not evaluation.valid:
        raise RuntimeError(f"Frozen baseline is invalid: {evaluation.reason}")
    scalar = ladder_scalar_score(evaluation.objectives)
    best = pd.DataFrame(
        [
            {
                **dict(zip(CELL_PARAMETER_NAMES, base_vector)),
                **dict(zip(OBJECTIVE_NAMES, evaluation.objectives)),
                "scalar_score": scalar,
                "valid": True,
                "reason": "accepted",
            }
        ]
    )
    return LadderFitResult(
        variant="base",
        target=target,
        parameter_names=CELL_PARAMETER_NAMES,
        history=best.copy(),
        best_parameters=best,
        evaluation=evaluation,
        feature_comparison=representative_feature_comparison(
            target,
            evaluation,
            _objective_config(config),
        ),
        validation=_validation_table(evaluation, target, config),
        metadata={
            "version": 1,
            "algorithm": "frozen NSGA-II representative",
            "variant": "base",
            "total_model_parameter_count": len(CELL_PARAMETER_NAMES),
            "screen_config": asdict(screen_config),
            "target": target_metadata(target),
            "best_biophysics": evaluation.biophysics,
            "best_rheobase_lower_pa": evaluation.rheobase_lower_pa,
            "best_rheobase_upper_pa": evaluation.rheobase_upper_pa,
            "scalar_weights": dict(zip(OBJECTIVE_NAMES, _SCALAR_WEIGHTS)),
        },
    )


def save_ladder_trace_plot(
    result: LadderFitResult,
    screen_config: BiologicalScreenConfig,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.evaluation.traces is None:
        raise ValueError("Ladder evaluation does not contain traces")
    sweeps = read_current_clamp_sweeps(
        result.target.nwb_path,
        long_square_only=True,
    )
    sweep_map = {sweep.sweep_number: sweep for sweep in sweeps}
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
                (
                    protocol.rheobase_factor
                    * result.evaluation.rheobase_upper_pa
                ),
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
        constrained_layout=True,
    )
    for axis, row in zip(axes, protocol_rows):
        (
            name,
            sweep_number,
            biological_current,
            model_current,
            duration_ms,
        ) = row
        sweep = sweep_map[sweep_number]
        bio_mask = (
            (sweep.time_ms >= sweep.stimulus_start_ms - 20.0)
            & (sweep.time_ms <= sweep.stimulus_end_ms)
        )
        trace = result.evaluation.traces[name]
        model_time = trace.time_ms - screen_config.baseline_ms
        model_mask = (model_time >= -20.0) & (model_time <= duration_ms)
        axis.plot(
            sweep.time_ms[bio_mask] - sweep.stimulus_start_ms,
            sweep.voltage_mv[bio_mask],
            color="#127475",
            linewidth=1.0,
            label="biological",
        )
        axis.plot(
            model_time[model_mask],
            trace.voltage_mv[model_mask],
            color="#d1495b",
            linewidth=0.95,
            label=result.variant,
        )
        role = "held out" if name == "held_out" else "fit"
        axis.set_title(
            f"{name.replace('_', ' ').title()} ({role}): "
            f"bio {biological_current:g} pA, model {model_current:.1f} pA",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("V (mV)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time from current onset (ms)")
    figure.suptitle(
        f"{result.variant}: {result.target.dataset} / {result.target.cell_id}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_ladder_fit(
    result: LadderFitResult,
    output_dir: str | Path,
    screen_config: BiologicalScreenConfig,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(output / "optimization_history.csv", index=False)
    result.best_parameters.to_csv(output / "best_model.csv", index=False)
    result.feature_comparison.to_csv(
        output / "feature_comparison.csv",
        index=False,
    )
    result.validation.to_csv(output / "held_out_validation.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )
    save_ladder_trace_plot(
        result,
        screen_config,
        output / "trace_comparison.png",
    )


def ladder_summary(results: Sequence[LadderFitResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        validation_scores = {
            f"held_out__{row.objective_group}": float(row.score)
            for row in result.validation.itertuples()
        }
        comparison = result.feature_comparison
        spike_rows = comparison.loc[
            comparison["feature"].eq("spike_count")
        ]
        spike_values = {
            f"{row.protocol}__biological_spike_count": row.biological_value
            for row in spike_rows.itertuples()
        }
        spike_values.update(
            {
                f"{row.protocol}__model_spike_count": row.model_value
                for row in spike_rows.itertuples()
            }
        )
        row = {
            "variant": result.variant,
            "dataset": result.target.dataset,
            "cell_id": result.target.cell_id,
            "total_parameter_count": result.metadata[
                "total_model_parameter_count"
            ],
            **dict(zip(OBJECTIVE_NAMES, result.evaluation.objectives)),
            "scalar_score": ladder_scalar_score(
                result.evaluation.objectives
            ),
            "model_rheobase_pa": result.evaluation.rheobase_upper_pa,
            **validation_scores,
            **spike_values,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_ladder_summary_plot(
    summary: pd.DataFrame,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = summary["variant"].tolist()
    x = np.arange(len(variants))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.0, 4.2),
        constrained_layout=True,
    )
    metrics = (
        ("firing_pattern", "Training firing-pattern score"),
        ("held_out__firing_pattern", "Held-out firing-pattern score"),
        ("scalar_score", "Weighted total score"),
    )
    colors = ("#6b9080", "#e07a5f", "#457b9d", "#8d5a97")
    for axis, (metric, title) in zip(axes, metrics):
        axis.bar(x, summary[metric], color=colors[: len(variants)])
        axis.set_xticks(x, variants, rotation=25, ha="right")
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel("Lower is better")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Minimal model ladder: {summary['dataset'].iloc[0]} / "
        f"{summary['cell_id'].iloc[0]}",
        fontsize=12,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
