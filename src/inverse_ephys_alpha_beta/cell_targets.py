"""Cell-specific multi-sweep targets and passive parameter anchoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .features import (
    FeatureConfig,
    extract_voltage_features,
    voltage_feature_trace,
)
from .hh_model import Stimulus
from .kinetics import KineticParameters
from .raw_patchseq import (
    CurrentClampSweep,
    count_sweep_spikes,
    read_current_clamp_sweeps,
    select_rheobase_sweep,
)
from .static_parameters import BiophysicalParameters


@dataclass(frozen=True)
class PassiveAnchor:
    resting_voltage_mv: float
    input_resistance_mohm: float
    membrane_tau_ms: float
    total_capacitance_pf: float
    membrane_area_um2: float
    capacitance_uf_cm2: float
    gleak_ms_cm2: float
    passive_current_pa: float
    passive_duration_ms: float
    passive_sweep_number: int
    sag_recovery_mv: float = 0.0
    peak_input_resistance_mohm: float = float("nan")
    estimation_method: str = "steady_state_crossing"


@dataclass(frozen=True)
class ProtocolFeatureTarget:
    name: str
    role: str
    sweep_number: int
    current_pa: float
    rheobase_factor: float
    duration_ms: float
    features: Mapping[str, float]


@dataclass(frozen=True)
class CellOptimizationTarget:
    dataset: str
    cell_id: str
    nwb_path: str
    sampled_rheobase_pa: float
    temperature_c: float
    passive: PassiveAnchor
    training_protocols: tuple[ProtocolFeatureTarget, ...]
    validation_protocols: tuple[ProtocolFeatureTarget, ...]


def _stimulus_for_sweep(sweep: CurrentClampSweep) -> Stimulus:
    if (
        sweep.stimulus_start_ms is None
        or sweep.stimulus_end_ms is None
        or sweep.stimulus_amplitude_pa is None
    ):
        raise ValueError(f"Sweep {sweep.sweep_number} has incomplete stimulus metadata")
    return Stimulus(
        amplitude_ua_cm2=None,
        amplitude_pa=float(sweep.stimulus_amplitude_pa),
        start_ms=float(sweep.stimulus_start_ms),
        end_ms=float(sweep.stimulus_end_ms),
    )


def _baseline_voltage(sweep: CurrentClampSweep) -> float:
    stimulus = _stimulus_for_sweep(sweep)
    baseline = sweep.voltage_mv[sweep.time_ms < stimulus.start_ms]
    if not len(baseline):
        raise ValueError(f"Sweep {sweep.sweep_number} has no pre-stimulus baseline")
    return float(np.median(baseline))


def _passive_features(
    sweep: CurrentClampSweep,
    feature_config: FeatureConfig,
) -> dict[str, float]:
    stimulus = _stimulus_for_sweep(sweep)
    current_pa = float(stimulus.amplitude_pa or 0.0)
    if current_pa >= 0.0:
        raise ValueError("Passive calibration requires a negative current step")
    trace = voltage_feature_trace(
        sweep.time_ms,
        sweep.voltage_mv,
        filter_window_ms=feature_config.voltage_filter_window_ms,
        polynomial_order=feature_config.voltage_filter_polynomial_order,
    )
    resting_voltage = _baseline_voltage(sweep)
    during = (trace.time_ms >= stimulus.start_ms) & (
        trace.time_ms < stimulus.end_ms
    )
    indices = np.flatnonzero(during)
    if len(indices) < 4:
        raise ValueError("Passive sweep is too short")
    dt_ms = float(np.median(np.diff(trace.time_ms)))
    final_count = max(2, int(round(20.0 / dt_ms)))
    steady_voltage = float(np.mean(trace.voltage_mv[indices[-final_count:]]))
    deflection = steady_voltage - resting_voltage
    input_resistance = 1000.0 * deflection / current_pa
    target_voltage = resting_voltage + 0.632 * deflection
    response = trace.voltage_mv[indices]
    crossing = np.flatnonzero(response <= target_voltage)
    membrane_tau = (
        float(trace.time_ms[indices[crossing[0]]] - stimulus.start_ms)
        if len(crossing)
        else float("nan")
    )
    return {
        "baseline_voltage_mv": resting_voltage,
        "input_resistance_mohm": input_resistance,
        "membrane_tau_ms": membrane_tau,
    }


def _local_passive_features(
    sweep: CurrentClampSweep,
    feature_config: FeatureConfig,
    early_fit_window_ms: float = 50.0,
) -> dict[str, float]:
    """Estimate near-rest passive properties before slow sag dominates."""
    stimulus = _stimulus_for_sweep(sweep)
    current_pa = float(stimulus.amplitude_pa or 0.0)
    if current_pa >= 0.0:
        raise ValueError("Passive calibration requires a negative current step")
    trace = voltage_feature_trace(
        sweep.time_ms,
        sweep.voltage_mv,
        filter_window_ms=feature_config.voltage_filter_window_ms,
        polynomial_order=feature_config.voltage_filter_polynomial_order,
    )
    resting_voltage = _baseline_voltage(sweep)
    during = (trace.time_ms >= stimulus.start_ms) & (
        trace.time_ms < stimulus.end_ms
    )
    indices = np.flatnonzero(during)
    if len(indices) < 8:
        raise ValueError("Passive sweep is too short")
    dt_ms = float(np.median(np.diff(trace.time_ms)))
    final_count = max(2, int(round(20.0 / dt_ms)))
    steady_voltage = float(np.mean(trace.voltage_mv[indices[-final_count:]]))
    trough_index = indices[int(np.argmin(trace.voltage_mv[indices]))]
    trough_voltage = float(trace.voltage_mv[trough_index])
    input_resistance = (
        1000.0 * (steady_voltage - resting_voltage) / current_pa
    )
    peak_input_resistance = (
        1000.0 * (trough_voltage - resting_voltage) / current_pa
    )

    fit = indices[
        (trace.time_ms[indices] >= stimulus.start_ms + max(dt_ms, 0.5))
        & (
            trace.time_ms[indices]
            <= stimulus.start_ms + early_fit_window_ms
        )
    ]
    if len(fit) < 5:
        raise ValueError("Passive sweep has too few early-response samples")
    fit_time = trace.time_ms[fit] - stimulus.start_ms
    fit_voltage = trace.voltage_mv[fit]
    initial_amplitude = min(
        -0.05,
        float(fit_voltage[-1] - resting_voltage),
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, tau_ms = parameters
        predicted = resting_voltage + amplitude * (
            1.0 - np.exp(-fit_time / tau_ms)
        )
        return predicted - fit_voltage

    fitted = least_squares(
        residual,
        x0=np.asarray((initial_amplitude, 10.0)),
        bounds=(
            np.asarray((-100.0, 0.2)),
            np.asarray((-0.01, 200.0)),
        ),
        loss="soft_l1",
    )
    membrane_tau = float(fitted.x[1])
    return {
        "baseline_voltage_mv": resting_voltage,
        "input_resistance_mohm": float(input_resistance),
        "membrane_tau_ms": membrane_tau,
        "sag_recovery_mv": float(steady_voltage - trough_voltage),
        "peak_input_resistance_mohm": float(peak_input_resistance),
        "steady_voltage_mv": steady_voltage,
        "trough_voltage_mv": trough_voltage,
    }


def derive_passive_anchor(
    resting_voltage_mv: float,
    input_resistance_mohm: float,
    membrane_tau_ms: float,
    passive_current_pa: float,
    passive_duration_ms: float,
    passive_sweep_number: int,
    capacitance_uf_cm2: float = 1.0,
    sag_recovery_mv: float = 0.0,
    peak_input_resistance_mohm: float = float("nan"),
    estimation_method: str = "steady_state_crossing",
) -> PassiveAnchor:
    """Derive single-compartment area and leak density from passive features."""
    if not -100.0 <= resting_voltage_mv <= -35.0:
        raise ValueError("Resting voltage is outside the physiological range")
    if input_resistance_mohm <= 0.0 or not np.isfinite(input_resistance_mohm):
        raise ValueError("Input resistance must be positive and finite")
    if membrane_tau_ms <= 0.0 or not np.isfinite(membrane_tau_ms):
        raise ValueError("Membrane time constant must be positive and finite")
    if capacitance_uf_cm2 <= 0.0:
        raise ValueError("Specific membrane capacitance must be positive")

    total_capacitance_pf = (
        1000.0 * membrane_tau_ms / input_resistance_mohm
    )
    membrane_area_um2 = (
        100.0 * total_capacitance_pf / capacitance_uf_cm2
    )
    total_conductance_ns = 1000.0 / input_resistance_mohm
    gleak_ms_cm2 = total_conductance_ns / (0.01 * membrane_area_um2)
    if not 100.0 <= membrane_area_um2 <= 200_000.0:
        raise ValueError(
            f"Passive estimate gives an implausible area: {membrane_area_um2:.1f} um2"
        )
    if not 0.001 <= gleak_ms_cm2 <= 10.0:
        raise ValueError(
            f"Passive estimate gives an implausible leak: {gleak_ms_cm2:.4g} mS/cm2"
        )
    return PassiveAnchor(
        resting_voltage_mv=float(resting_voltage_mv),
        input_resistance_mohm=float(input_resistance_mohm),
        membrane_tau_ms=float(membrane_tau_ms),
        total_capacitance_pf=float(total_capacitance_pf),
        membrane_area_um2=float(membrane_area_um2),
        capacitance_uf_cm2=float(capacitance_uf_cm2),
        gleak_ms_cm2=float(gleak_ms_cm2),
        passive_current_pa=float(passive_current_pa),
        passive_duration_ms=float(passive_duration_ms),
        passive_sweep_number=int(passive_sweep_number),
        sag_recovery_mv=float(sag_recovery_mv),
        peak_input_resistance_mohm=float(peak_input_resistance_mohm),
        estimation_method=str(estimation_method),
    )


def anchored_biophysics(
    anchor: PassiveAnchor,
    kinetics: KineticParameters,
    log_gna_scale: float = 0.0,
    log_gk_scale: float = 0.0,
    ena_mv: float = 50.0,
    ek_mv: float = -77.0,
    q10: float = 3.0,
) -> BiophysicalParameters:
    """Construct biophysics and solve ELeak for the measured resting voltage."""
    gna = 120.0 * np.exp(log_gna_scale)
    gk = 36.0 * np.exp(log_gk_scale)
    voltage = anchor.resting_voltage_mv
    m_gate, h_gate, n_gate = kinetics.steady_state(voltage)
    sodium_current = gna * m_gate**3 * h_gate * (voltage - ena_mv)
    potassium_current = gk * n_gate**4 * (voltage - ek_mv)
    eleak = voltage + (
        sodium_current + potassium_current
    ) / anchor.gleak_ms_cm2
    return BiophysicalParameters(
        membrane_area_um2=anchor.membrane_area_um2,
        capacitance_uf_cm2=anchor.capacitance_uf_cm2,
        gna_ms_cm2=float(gna),
        gk_ms_cm2=float(gk),
        gleak_ms_cm2=anchor.gleak_ms_cm2,
        ena_mv=ena_mv,
        ek_mv=ek_mv,
        eleak_mv=float(eleak),
        q10_m=q10,
        q10_h=q10,
        q10_n=q10,
    )


def admittance_anchored_biophysics(
    anchor: PassiveAnchor,
    kinetics: KineticParameters,
    log_gna_scale: float = 0.0,
    log_gk_scale: float = 0.0,
    ena_mv: float = 50.0,
    ek_mv: float = -77.0,
    q10: float = 3.0,
    minimum_leak_ms_cm2: float = 0.001,
) -> BiophysicalParameters:
    """Preserve rest and the observed finite-step passive endpoint."""
    gna = 120.0 * np.exp(log_gna_scale)
    gk = 36.0 * np.exp(log_gk_scale)
    voltage = anchor.resting_voltage_mv

    def active_current(test_voltage: float) -> float:
        m_gate, h_gate, n_gate = kinetics.steady_state(test_voltage)
        return float(
            gna
            * m_gate**3
            * h_gate
            * (test_voltage - ena_mv)
            + gk * n_gate**4 * (test_voltage - ek_mv)
        )

    gleak, eleak = passive_leak_from_active_current(
        anchor,
        active_current,
        minimum_leak_ms_cm2=minimum_leak_ms_cm2,
    )
    return BiophysicalParameters(
        membrane_area_um2=anchor.membrane_area_um2,
        capacitance_uf_cm2=anchor.capacitance_uf_cm2,
        gna_ms_cm2=float(gna),
        gk_ms_cm2=float(gk),
        gleak_ms_cm2=float(gleak),
        ena_mv=ena_mv,
        ek_mv=ek_mv,
        eleak_mv=float(eleak),
        q10_m=q10,
        q10_h=q10,
        q10_n=q10,
    )


def passive_leak_from_active_current(
    anchor: PassiveAnchor,
    active_current,
    minimum_leak_ms_cm2: float = 0.001,
    stability_delta_mv: float = 0.05,
) -> tuple[float, float]:
    """Solve leak parameters from rest and one observed passive endpoint."""
    resting_voltage = anchor.resting_voltage_mv
    passive_voltage = resting_voltage + (
        anchor.passive_current_pa
        * anchor.input_resistance_mohm
        / 1000.0
    )
    if passive_voltage == resting_voltage:
        raise ValueError("Passive anchor has zero voltage deflection")
    area_cm2 = anchor.membrane_area_um2 * 1e-8
    applied_density = anchor.passive_current_pa / (area_cm2 * 1e6)
    active_rest = float(active_current(resting_voltage))
    active_passive = float(active_current(passive_voltage))
    gleak = (
        applied_density - (active_passive - active_rest)
    ) / (passive_voltage - resting_voltage)
    if not np.isfinite(gleak) or gleak < minimum_leak_ms_cm2:
        raise ValueError(
            "Active passive-step conductance leaves no positive leak conductance"
        )
    eleak = resting_voltage + active_rest / gleak
    for label, test_voltage in (
        ("rest", resting_voltage),
        ("passive endpoint", passive_voltage),
    ):
        total_slope = gleak + (
            float(active_current(test_voltage + stability_delta_mv))
            - float(active_current(test_voltage - stability_delta_mv))
        ) / (2.0 * stability_delta_mv)
        if not np.isfinite(total_slope) or total_slope <= 0.0:
            raise ValueError(f"{label} is unstable after passive anchoring")
    return float(gleak), float(eleak)


def steady_current_density(
    voltage_mv: float,
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
) -> float:
    """Return steady-state ionic current density at one voltage."""
    m_gate, h_gate, n_gate = kinetics.steady_state(voltage_mv)
    return float(
        biophysics.gna_ms_cm2
        * m_gate**3
        * h_gate
        * (voltage_mv - biophysics.ena_mv)
        + biophysics.gk_ms_cm2
        * n_gate**4
        * (voltage_mv - biophysics.ek_mv)
        + biophysics.gleak_ms_cm2
        * (voltage_mv - biophysics.eleak_mv)
    )


def resting_state_is_stable(
    anchor: PassiveAnchor,
    kinetics: KineticParameters,
    biophysics: BiophysicalParameters,
    delta_mv: float = 0.05,
) -> bool:
    """Check the local steady-current slope at the anchored resting voltage."""
    voltage = anchor.resting_voltage_mv
    slope = (
        steady_current_density(voltage + delta_mv, kinetics, biophysics)
        - steady_current_density(voltage - delta_mv, kinetics, biophysics)
    ) / (2.0 * delta_mv)
    return bool(np.isfinite(slope) and slope > 0.0)


def _protocol_target(
    name: str,
    role: str,
    sweep: CurrentClampSweep,
    rheobase_pa: float,
    feature_config: FeatureConfig,
) -> ProtocolFeatureTarget:
    stimulus = _stimulus_for_sweep(sweep)
    features = extract_voltage_features(
        sweep.time_ms,
        sweep.voltage_mv,
        stimulus,
        feature_config,
    )
    current_pa = float(stimulus.amplitude_pa or 0.0)
    return ProtocolFeatureTarget(
        name=name,
        role=role,
        sweep_number=sweep.sweep_number,
        current_pa=current_pa,
        rheobase_factor=current_pa / rheobase_pa,
        duration_ms=stimulus.end_ms - stimulus.start_ms,
        features=features,
    )


def _nearest_unused_sweep(
    sweeps: Sequence[CurrentClampSweep],
    target_current_pa: float,
    used_sweep_numbers: set[int],
) -> CurrentClampSweep:
    available = [
        sweep
        for sweep in sweeps
        if sweep.sweep_number not in used_sweep_numbers
    ]
    if not available:
        raise ValueError("No unused spiking sweep is available")
    return min(
        available,
        key=lambda sweep: (
            abs(float(sweep.stimulus_amplitude_pa or 0.0) - target_current_pa),
            sweep.sweep_number,
        ),
    )


def build_cell_optimization_target(
    raw_spike_cycles: pd.DataFrame,
    dataset: str,
    cell_id: str,
    temperature_c: float,
    training_factor: float = 1.5,
    validation_factor: float = 2.0,
    passive_target_pa: float = -40.0,
    local_passive: bool = False,
    feature_config: FeatureConfig | None = None,
) -> CellOptimizationTarget:
    """Build passive, training, and held-out targets from one local NWB cell."""
    feature_config = feature_config or FeatureConfig(min_spikes=1)
    rows = raw_spike_cycles.loc[
        raw_spike_cycles["dataset"].astype(str).eq(dataset)
        & raw_spike_cycles["cell_id"].astype(str).eq(str(cell_id))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one raw feature row for {dataset}/{cell_id}, found {len(rows)}"
        )
    nwb_path = str(rows.iloc[0]["nwb_path"])
    sweeps = read_current_clamp_sweeps(nwb_path, long_square_only=True)
    rheobase_sweep, _ = select_rheobase_sweep(
        sweeps,
        config=feature_config,
    )
    if rheobase_sweep is None or rheobase_sweep.stimulus_amplitude_pa is None:
        raise ValueError(f"No sampled-rheobase sweep found for {cell_id}")
    rheobase_pa = float(rheobase_sweep.stimulus_amplitude_pa)

    negative_sweeps = [
        sweep
        for sweep in sweeps
        if sweep.stimulus_amplitude_pa is not None
        and sweep.stimulus_amplitude_pa < 0.0
    ]
    if not negative_sweeps:
        raise ValueError(f"No hyperpolarizing long-square sweep found for {cell_id}")
    passive_sweep = min(
        negative_sweeps,
        key=(
            (
                lambda sweep: (
                    abs(float(sweep.stimulus_amplitude_pa)),
                    sweep.sweep_number,
                )
            )
            if local_passive
            else (
                lambda sweep: (
                    abs(
                        float(sweep.stimulus_amplitude_pa)
                        - passive_target_pa
                    ),
                    sweep.sweep_number,
                )
            )
        ),
    )
    passive_values = (
        _local_passive_features(passive_sweep, feature_config)
        if local_passive
        else _passive_features(passive_sweep, feature_config)
    )
    passive_anchor = derive_passive_anchor(
        resting_voltage_mv=passive_values["baseline_voltage_mv"],
        input_resistance_mohm=passive_values["input_resistance_mohm"],
        membrane_tau_ms=passive_values["membrane_tau_ms"],
        passive_current_pa=float(passive_sweep.stimulus_amplitude_pa),
        passive_duration_ms=(
            float(passive_sweep.stimulus_end_ms)
            - float(passive_sweep.stimulus_start_ms)
        ),
        passive_sweep_number=passive_sweep.sweep_number,
        sag_recovery_mv=float(
            passive_values.get("sag_recovery_mv", 0.0)
        ),
        peak_input_resistance_mohm=float(
            passive_values.get(
                "peak_input_resistance_mohm",
                float("nan"),
            )
        ),
        estimation_method=(
            "smallest_step_early_exponential"
            if local_passive
            else "steady_state_crossing"
        ),
    )

    spiking_sweeps = [
        sweep
        for sweep in sweeps
        if sweep.stimulus_amplitude_pa is not None
        and sweep.stimulus_amplitude_pa > 0.0
        and count_sweep_spikes(sweep, feature_config) > 0
    ]
    used = {rheobase_sweep.sweep_number}
    training_sweep = _nearest_unused_sweep(
        spiking_sweeps,
        training_factor * rheobase_pa,
        used,
    )
    used.add(training_sweep.sweep_number)
    validation_sweep = _nearest_unused_sweep(
        spiking_sweeps,
        validation_factor * rheobase_pa,
        used,
    )

    training_protocols = (
        _protocol_target(
            "rheobase",
            "training",
            rheobase_sweep,
            rheobase_pa,
            feature_config,
        ),
        _protocol_target(
            "suprathreshold",
            "training",
            training_sweep,
            rheobase_pa,
            feature_config,
        ),
    )
    validation_protocols = (
        _protocol_target(
            "held_out",
            "validation",
            validation_sweep,
            rheobase_pa,
            feature_config,
        ),
    )
    return CellOptimizationTarget(
        dataset=dataset,
        cell_id=str(cell_id),
        nwb_path=nwb_path,
        sampled_rheobase_pa=rheobase_pa,
        temperature_c=float(temperature_c),
        passive=passive_anchor,
        training_protocols=training_protocols,
        validation_protocols=validation_protocols,
    )


def target_protocol_table(target: CellOptimizationTarget) -> pd.DataFrame:
    rows = []
    for protocol in (*target.training_protocols, *target.validation_protocols):
        row = {
            "dataset": target.dataset,
            "cell_id": target.cell_id,
            "protocol": protocol.name,
            "role": protocol.role,
            "sweep_number": protocol.sweep_number,
            "current_pa": protocol.current_pa,
            "rheobase_factor": protocol.rheobase_factor,
            "duration_ms": protocol.duration_ms,
        }
        row.update(
            {
                f"feature__{name}": value
                for name, value in protocol.features.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def target_metadata(target: CellOptimizationTarget) -> dict[str, object]:
    return {
        "dataset": target.dataset,
        "cell_id": target.cell_id,
        "nwb_path": target.nwb_path,
        "sampled_rheobase_pa": target.sampled_rheobase_pa,
        "temperature_c": target.temperature_c,
        "passive": asdict(target.passive),
        "training_protocols": [
            {
                key: value
                for key, value in asdict(protocol).items()
                if key != "features"
            }
            for protocol in target.training_protocols
        ],
        "validation_protocols": [
            {
                key: value
                for key, value in asdict(protocol).items()
                if key != "features"
            }
            for protocol in target.validation_protocols
        ],
    }


def load_raw_spike_cycle_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"cell_id": "string"})
