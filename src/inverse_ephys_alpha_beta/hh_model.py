"""Current-clamp simulation of a parameterized Hodgkin-Huxley neuron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .kinetics import KineticParameters
from .static_parameters import BiophysicalParameters


class SimulationError(RuntimeError):
    """Raised when a sampled kinetic model cannot be simulated safely."""


Conductances = BiophysicalParameters


@dataclass(frozen=True)
class Stimulus:
    amplitude_ua_cm2: float | None = 10.0
    amplitude_pa: float | None = None
    start_ms: float = 10.0
    end_ms: float = 90.0

    def density_value(
        self, time_ms: float, biophysics: BiophysicalParameters
    ) -> float:
        if not self.start_ms <= time_ms < self.end_ms:
            return 0.0
        if self.amplitude_pa is not None:
            return float(biophysics.current_pa_to_density(self.amplitude_pa))
        return float(self.amplitude_ua_cm2 or 0.0)

    def pa_value(self, time_ms: float, biophysics: BiophysicalParameters) -> float:
        if not self.start_ms <= time_ms < self.end_ms:
            return 0.0
        if self.amplitude_pa is not None:
            return self.amplitude_pa
        return float(biophysics.current_density_to_pa(self.amplitude_ua_cm2 or 0.0))

    def value(self, time_ms: float) -> float:
        """Return density for compatibility with density-defined stimuli."""
        if self.amplitude_pa is not None:
            raise ValueError("A pA stimulus requires membrane-area conversion")
        return float(self.amplitude_ua_cm2 or 0.0) if self.start_ms <= time_ms < self.end_ms else 0.0


@dataclass(frozen=True)
class SimulationConfig:
    duration_ms: float = 100.0
    dt_ms: float = 0.025
    initial_voltage_mv: float = -65.0
    stimulus: Stimulus = Stimulus()
    conductances: Conductances = Conductances()
    temperature_c: float = 6.3


@dataclass(frozen=True)
class Trace:
    time_ms: NDArray[np.float64]
    voltage_mv: NDArray[np.float64]
    m: NDArray[np.float64]
    h: NDArray[np.float64]
    n: NDArray[np.float64]
    applied_current_ua_cm2: NDArray[np.float64]
    applied_current_pa: NDArray[np.float64]
    ionic_current_ua_cm2: NDArray[np.float64]
    ionic_current_pa: NDArray[np.float64]
    sodium_current_ua_cm2: NDArray[np.float64]
    potassium_current_ua_cm2: NDArray[np.float64]
    leak_current_ua_cm2: NDArray[np.float64]
    dvdt_mv_ms: NDArray[np.float64]


def _currents(
    state: NDArray[np.float64], conductances: Conductances
) -> tuple[float, float, float, float]:
    voltage, m_gate, h_gate, n_gate = state
    i_na = (
        conductances.gna_ms_cm2
        * m_gate**3
        * h_gate
        * (voltage - conductances.ena_mv)
    )
    i_k = conductances.gk_ms_cm2 * n_gate**4 * (voltage - conductances.ek_mv)
    i_leak = conductances.gleak_ms_cm2 * (voltage - conductances.eleak_mv)
    return i_na, i_k, i_leak, i_na + i_k + i_leak


def equilibrium_state(
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
    voltage_min_mv: float = -110.0,
    voltage_max_mv: float = -25.0,
) -> NDArray[np.float64]:
    """Find a stable zero-current fixed point near the canonical resting voltage."""

    def steady_current(voltage_mv: float) -> float:
        m_gate, h_gate, n_gate = kinetics.steady_state(voltage_mv)
        return _currents(
            np.asarray((voltage_mv, m_gate, h_gate, n_gate)),
            biophysics,
        )[3]

    voltage_grid = np.linspace(voltage_min_mv, voltage_max_mv, 341)
    currents = np.asarray([steady_current(voltage) for voltage in voltage_grid])
    roots = []
    for left, right, current_left, current_right in zip(
        voltage_grid[:-1],
        voltage_grid[1:],
        currents[:-1],
        currents[1:],
    ):
        if current_left == 0.0:
            roots.append(float(left))
        elif current_left * current_right < 0.0:
            roots.append(float(brentq(steady_current, left, right)))
    if not roots:
        raise SimulationError("No zero-current equilibrium in the physiological voltage range")

    stable_roots = []
    for voltage in roots:
        delta = 0.05
        slope = (
            steady_current(voltage + delta) - steady_current(voltage - delta)
        ) / (2.0 * delta)
        if slope > 0.0:
            stable_roots.append(voltage)
    candidates = stable_roots or roots
    resting_voltage = min(candidates, key=lambda voltage: abs(voltage + 65.0))
    m_gate, h_gate, n_gate = kinetics.steady_state(resting_voltage)
    return np.asarray((resting_voltage, m_gate, h_gate, n_gate), dtype=float)


def _rhs(
    time_ms: float,
    state: NDArray[np.float64],
    kinetics: KineticParameters,
    config: SimulationConfig,
    temperature_factors: tuple[float, float, float],
) -> NDArray[np.float64]:
    voltage, m_gate, h_gate, n_gate = state
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = kinetics.rates_scalar(
        float(voltage)
    )
    factor_m, factor_h, factor_n = temperature_factors
    _, _, _, i_ion = _currents(state, config.conductances)
    d_voltage = (
        config.stimulus.density_value(time_ms, config.conductances) - i_ion
    ) / config.conductances.capacitance_uf_cm2
    d_m = factor_m * (alpha_m * (1.0 - m_gate) - beta_m * m_gate)
    d_h = factor_h * (alpha_h * (1.0 - h_gate) - beta_h * h_gate)
    d_n = factor_n * (alpha_n * (1.0 - n_gate) - beta_n * n_gate)
    return np.asarray((d_voltage, d_m, d_h, d_n), dtype=float)


def elicits_spike(
    kinetics: KineticParameters,
    config: SimulationConfig,
    state_at_stimulus_onset: NDArray[np.float64],
    spike_voltage_mv: float = 0.0,
    max_step_ms: float = 0.5,
) -> bool:
    """Test for a spike with adaptive integration and stop at the first crossing."""
    initial_state = np.asarray(state_at_stimulus_onset, dtype=float)
    if initial_state.shape != (4,):
        raise ValueError("state_at_stimulus_onset must contain V, m, h, and n")
    if max_step_ms <= 0.0:
        raise ValueError("max_step_ms must be positive")

    temperature_factors = config.conductances.temperature_factors(
        config.temperature_c
    )

    def rhs(time_ms: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return _rhs(time_ms, state, kinetics, config, temperature_factors)

    def spike_crossing(_time_ms: float, state: NDArray[np.float64]) -> float:
        return float(state[0] - spike_voltage_mv)

    spike_crossing.terminal = True
    spike_crossing.direction = 1.0
    start_ms = config.stimulus.start_ms
    end_ms = min(config.duration_ms, config.stimulus.end_ms)
    solution = solve_ivp(
        rhs,
        (start_ms, end_ms),
        initial_state,
        events=spike_crossing,
        rtol=1e-5,
        atol=(1e-6, 1e-8, 1e-8, 1e-8),
        max_step=max_step_ms,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise SimulationError(f"Adaptive spike test failed: {solution.message}")
    if np.any(np.abs(solution.y[0]) > 200.0):
        raise SimulationError("Adaptive spike test left the physiological voltage range")
    return bool(len(solution.t_events[0]))


def simulate(
    kinetics: KineticParameters | None = None,
    config: SimulationConfig | None = None,
    initial_state: NDArray[np.float64] | None = None,
) -> Trace:
    """Simulate one model with a fixed-step fourth-order Runge-Kutta method."""
    kinetics = kinetics or KineticParameters.canonical()
    config = config or SimulationConfig()
    if config.dt_ms <= 0.0 or config.duration_ms <= 0.0:
        raise ValueError("Simulation duration and time step must be positive")

    n_steps = int(round(config.duration_ms / config.dt_ms))
    time_ms = np.linspace(0.0, config.duration_ms, n_steps + 1)
    states = np.empty((n_steps + 1, 4), dtype=float)
    if initial_state is None:
        m0, h0, n0 = kinetics.steady_state(config.initial_voltage_mv)
        states[0] = (config.initial_voltage_mv, m0, h0, n0)
    else:
        if np.asarray(initial_state).shape != (4,):
            raise ValueError("initial_state must contain V, m, h, and n")
        states[0] = np.asarray(initial_state, dtype=float)

    dt = config.dt_ms
    temperature_factors = config.conductances.temperature_factors(config.temperature_c)
    for index in range(n_steps):
        time = time_ms[index]
        state = states[index]
        k1 = _rhs(time, state, kinetics, config, temperature_factors)
        k2 = _rhs(
            time + dt / 2.0,
            state + dt * k1 / 2.0,
            kinetics,
            config,
            temperature_factors,
        )
        k3 = _rhs(
            time + dt / 2.0,
            state + dt * k2 / 2.0,
            kinetics,
            config,
            temperature_factors,
        )
        k4 = _rhs(
            time + dt,
            state + dt * k3,
            kinetics,
            config,
            temperature_factors,
        )
        next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        if (
            not np.all(np.isfinite(next_state))
            or abs(next_state[0]) > 200.0
            or np.any(next_state[1:] < -0.05)
            or np.any(next_state[1:] > 1.05)
        ):
            raise SimulationError(f"Integration diverged at t={time + dt:.4f} ms")
        next_state[1:] = np.clip(next_state[1:], 0.0, 1.0)
        states[index + 1] = next_state

    applied_current = np.asarray(
        [
            config.stimulus.density_value(time, config.conductances)
            for time in time_ms
        ]
    )
    applied_current_pa = np.asarray(
        [config.stimulus.pa_value(time, config.conductances) for time in time_ms]
    )
    current_components = np.asarray(
        [_currents(state, config.conductances) for state in states], dtype=float
    )
    i_na, i_k, i_leak, i_ion = current_components.T
    dvdt = (
        applied_current - i_ion
    ) / config.conductances.capacitance_uf_cm2

    return Trace(
        time_ms=time_ms,
        voltage_mv=states[:, 0],
        m=states[:, 1],
        h=states[:, 2],
        n=states[:, 3],
        applied_current_ua_cm2=applied_current,
        applied_current_pa=applied_current_pa,
        ionic_current_ua_cm2=i_ion,
        ionic_current_pa=config.conductances.current_density_to_pa(i_ion),
        sodium_current_ua_cm2=i_na,
        potassium_current_ua_cm2=i_k,
        leak_current_ua_cm2=i_leak,
        dvdt_mv_ms=dvdt,
    )
