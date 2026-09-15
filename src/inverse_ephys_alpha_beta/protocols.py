"""Protocol-matched screening simulations for biological feature coverage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureConfig, extract_observed_features
from .hh_model import (
    SimulationConfig,
    SimulationError,
    Stimulus,
    Trace,
    elicits_spike,
    equilibrium_state,
    simulate,
)
from .kinetics import KineticParameters
from .static_parameters import BiophysicalParameters


@dataclass(frozen=True)
class BiologicalScreenConfig:
    protocol_name: str = "scala"
    temperature_c: float = 22.0
    dt_ms: float = 0.025
    baseline_ms: float = 20.0
    rheobase_step_ms: float = 600.0
    waveform_step_ms: float = 600.0
    hyperpolarizing_step_ms: float = 600.0
    rheobase_currents_pa: tuple[float, ...] = (
        10.0,
        20.0,
        40.0,
        80.0,
        140.0,
        220.0,
        320.0,
        450.0,
        600.0,
        800.0,
        1000.0,
    )
    rheobase_tolerance_pa: float = 2.0
    rheobase_max_iterations: int = 12
    waveform_current_offset_pa: float = 0.0
    spike_test_max_step_ms: float = 0.5
    hyperpolarizing_current_pa: float = -40.0


SCREEN_PRESETS = ("fast", "scala", "scala-phys", "gouwens")
SCREEN_DEFAULT_TEMPERATURE_C = {
    "fast": 22.0,
    "scala": 22.0,
    "scala-phys": 34.0,
    "gouwens": 34.0,
}


def biological_screen_config(
    preset: str = "scala",
    temperature_c: float | None = None,
    dt_ms: float = 0.025,
) -> BiologicalScreenConfig:
    """Construct a named screening protocol while keeping temperature explicit."""
    if preset not in SCREEN_PRESETS:
        raise ValueError(f"Unknown biological screen preset: {preset}")
    resolved_temperature = (
        SCREEN_DEFAULT_TEMPERATURE_C[preset]
        if temperature_c is None
        else temperature_c
    )
    if preset == "fast":
        return BiologicalScreenConfig(
            protocol_name=preset,
            temperature_c=resolved_temperature,
            dt_ms=dt_ms,
            rheobase_step_ms=100.0,
            waveform_step_ms=300.0,
            hyperpolarizing_step_ms=200.0,
            rheobase_tolerance_pa=10.0,
        )
    if preset in ("scala", "scala-phys"):
        return BiologicalScreenConfig(
            protocol_name=preset,
            temperature_c=resolved_temperature,
            dt_ms=dt_ms,
            rheobase_step_ms=600.0,
            waveform_step_ms=600.0,
            hyperpolarizing_step_ms=600.0,
        )
    if preset == "gouwens":
        return BiologicalScreenConfig(
            protocol_name=preset,
            temperature_c=resolved_temperature,
            dt_ms=dt_ms,
            rheobase_step_ms=1000.0,
            waveform_step_ms=1000.0,
            hyperpolarizing_step_ms=1000.0,
        )
    raise AssertionError("Unreachable biological screen preset")


@dataclass(frozen=True)
class BiologicalScreenResult:
    waveform_trace: Trace
    hyperpolarizing_trace: Trace
    features: dict[str, float]
    rheobase_lower_pa: float
    rheobase_upper_pa: float
    bisection_iterations: int


def _step_config(
    current_pa: float,
    step_duration_ms: float,
    config: BiologicalScreenConfig,
    biophysics: BiophysicalParameters,
    resting_voltage_mv: float,
) -> SimulationConfig:
    end_ms = config.baseline_ms + step_duration_ms
    return SimulationConfig(
        duration_ms=end_ms,
        dt_ms=config.dt_ms,
        initial_voltage_mv=resting_voltage_mv,
        stimulus=Stimulus(
            amplitude_ua_cm2=None,
            amplitude_pa=current_pa,
            start_ms=config.baseline_ms,
            end_ms=end_ms,
        ),
        conductances=biophysics,
        temperature_c=config.temperature_c,
    )


def step_simulation_config(
    current_pa: float,
    step_duration_ms: float,
    config: BiologicalScreenConfig,
    biophysics: BiophysicalParameters,
    resting_voltage_mv: float,
) -> SimulationConfig:
    """Construct a protocol-matched current-step simulation."""
    return _step_config(
        current_pa,
        step_duration_ms,
        config,
        biophysics,
        resting_voltage_mv,
    )


def _find_rheobase(
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
    resting_state: np.ndarray,
    config: BiologicalScreenConfig,
) -> tuple[float, float, int]:
    currents = config.rheobase_currents_pa
    if not currents or currents[0] <= 0.0:
        raise ValueError("Rheobase current grid must contain positive currents")
    if any(right <= left for left, right in zip(currents[:-1], currents[1:])):
        raise ValueError("Rheobase current grid must be strictly increasing")
    if config.rheobase_tolerance_pa <= 0.0:
        raise ValueError("Rheobase tolerance must be positive")

    resting_voltage = float(resting_state[0])
    spike_cache: dict[int, bool] = {}

    def spikes_at(index: int) -> bool:
        if index in spike_cache:
            return spike_cache[index]
        current_pa = currents[index]
        search_config = _step_config(
            current_pa,
            config.rheobase_step_ms,
            config,
            biophysics,
            resting_voltage,
        )
        result = elicits_spike(
            kinetics,
            search_config,
            resting_state,
            max_step_ms=config.spike_test_max_step_ms,
        )
        spike_cache[index] = bool(result)
        return spike_cache[index]

    # Very excitable models can spike at the first grid current but enter
    # depolarization block at the largest one. Preserve that case explicitly,
    # then use logarithmic bracketing for the usual monotone onset regime.
    lower_index = -1
    upper_index = 0
    if not spikes_at(upper_index):
        upper_index = len(currents) - 1
        if not spikes_at(upper_index):
            raise SimulationError("No spike within the rheobase search range")
        lower_index = 0
        while upper_index - lower_index > 1:
            midpoint_index = (lower_index + upper_index) // 2
            if spikes_at(midpoint_index):
                upper_index = midpoint_index
            else:
                lower_index = midpoint_index

    lower_pa = 0.0 if lower_index < 0 else currents[lower_index]
    upper_pa = currents[upper_index]

    iterations = 0
    while (
        upper_pa - lower_pa > config.rheobase_tolerance_pa
        and iterations < config.rheobase_max_iterations
    ):
        midpoint_pa = (lower_pa + upper_pa) / 2.0
        midpoint_config = _step_config(
            midpoint_pa,
            config.rheobase_step_ms,
            config,
            biophysics,
            resting_voltage,
        )
        if elicits_spike(
            kinetics,
            midpoint_config,
            resting_state,
            max_step_ms=config.spike_test_max_step_ms,
        ):
            upper_pa = midpoint_pa
        else:
            lower_pa = midpoint_pa
        iterations += 1
    return lower_pa, upper_pa, iterations


def find_rheobase(
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
    resting_state: np.ndarray,
    config: BiologicalScreenConfig,
) -> tuple[float, float, int]:
    """Bracket and bisect model rheobase under a biological screen protocol."""
    return _find_rheobase(
        kinetics,
        biophysics,
        resting_state,
        config,
    )


def _subthreshold_features(
    trace: Trace,
    resting_voltage_mv: float,
    current_pa: float,
    stimulus: Stimulus,
) -> dict[str, float]:
    during = (trace.time_ms >= stimulus.start_ms) & (trace.time_ms < stimulus.end_ms)
    indices = np.flatnonzero(during)
    if len(indices) < 4:
        return {}
    final_count = max(2, int(round(20.0 / np.median(np.diff(trace.time_ms)))))
    steady_voltage = float(np.mean(trace.voltage_mv[indices[-final_count:]]))
    deflection = steady_voltage - resting_voltage_mv
    input_resistance = 1000.0 * deflection / current_pa

    target_voltage = resting_voltage_mv + 0.632 * deflection
    response = trace.voltage_mv[indices]
    if deflection < 0.0:
        crossing = np.flatnonzero(response <= target_voltage)
    else:
        crossing = np.flatnonzero(response >= target_voltage)
    membrane_tau = (
        float(trace.time_ms[indices[crossing[0]]] - stimulus.start_ms)
        if len(crossing)
        else float("nan")
    )

    minimum_voltage = float(np.min(response))
    peak_deflection = minimum_voltage - resting_voltage_mv
    sag_ratio = (
        float((steady_voltage - minimum_voltage) / abs(peak_deflection))
        if peak_deflection < -1e-6
        else 0.0
    )
    return {
        "input_resistance_mohm": input_resistance,
        "membrane_tau_ms": membrane_tau,
        "sag_ratio": sag_ratio,
    }


def subthreshold_features(
    trace: Trace,
    resting_voltage_mv: float,
    current_pa: float,
    stimulus: Stimulus,
) -> dict[str, float]:
    """Extract passive step-response features from a simulated trace."""
    return _subthreshold_features(
        trace,
        resting_voltage_mv,
        current_pa,
        stimulus,
    )


def run_biological_screen(
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
    config: BiologicalScreenConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> BiologicalScreenResult:
    """Refine rheobase and extract long-sweep waveform/subthreshold features."""
    config = config or BiologicalScreenConfig()
    feature_config = feature_config or FeatureConfig(min_spikes=1)
    resting_state = equilibrium_state(kinetics, biophysics)
    resting_voltage = float(resting_state[0])
    if not -100.0 <= resting_voltage <= -35.0:
        raise SimulationError(f"Resting voltage is outside bounds: {resting_voltage:.2f} mV")

    rheobase_lower_pa, rheobase_pa, bisection_iterations = _find_rheobase(
        kinetics,
        biophysics,
        resting_state,
        config,
    )
    waveform_current_pa = rheobase_pa + config.waveform_current_offset_pa

    waveform_config = _step_config(
        waveform_current_pa,
        config.waveform_step_ms,
        config,
        biophysics,
        resting_voltage,
    )
    waveform_trace = simulate(kinetics, waveform_config, initial_state=resting_state)
    waveform_features = extract_observed_features(
        waveform_trace,
        waveform_config.stimulus,
        feature_config,
    )
    if not bool(waveform_features["is_spiking"]):
        raise SimulationError("Rheobase waveform sweep did not spike")

    hyperpolarizing_config = _step_config(
        config.hyperpolarizing_current_pa,
        config.hyperpolarizing_step_ms,
        config,
        biophysics,
        resting_voltage,
    )
    hyperpolarizing_trace = simulate(
        kinetics,
        hyperpolarizing_config,
        initial_state=resting_state,
    )
    protocol_features = _subthreshold_features(
        hyperpolarizing_trace,
        resting_voltage,
        config.hyperpolarizing_current_pa,
        hyperpolarizing_config.stimulus,
    )
    waveform_features.update(protocol_features)
    waveform_features.update(
        {
            "resting_voltage_mv": resting_voltage,
            "rheobase_pa": rheobase_pa,
            "rheobase_lower_bound_pa": rheobase_lower_pa,
            "rheobase_upper_bound_pa": rheobase_pa,
            "rheobase_precision_pa": rheobase_pa - rheobase_lower_pa,
            "rheobase_bisection_iterations": float(bisection_iterations),
            "waveform_current_pa": waveform_current_pa,
            "rheobase_step_ms": config.rheobase_step_ms,
            "waveform_step_ms": config.waveform_step_ms,
            "temperature_c": config.temperature_c,
        }
    )
    return BiologicalScreenResult(
        waveform_trace=waveform_trace,
        hyperpolarizing_trace=hyperpolarizing_trace,
        features=waveform_features,
        rheobase_lower_pa=rheobase_lower_pa,
        rheobase_upper_pa=rheobase_pa,
        bisection_iterations=bisection_iterations,
    )
