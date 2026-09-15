"""Compact phase timing with continuous recovery and a fitted onset branch."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.signal import find_peaks

from .abstract_phase_model import AbstractObservedTrace
from .phase_template_model import (
    PhaseCycleSet,
    PhaseTemplateModel,
    PhaseTemplateSimulation,
)
from .phase_timing import _advance_voltage_state, _periodic_spline


@dataclass(frozen=True)
class RecoveryOnsetConfig:
    """Fitting bounds for the recovery clock and first-spike onset path."""

    tau_candidates_ms: tuple[float, ...] = (
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    )
    late_validation_fraction: float = 0.3
    state_ridge: float = 1e-3
    voltage_scale_mv: float = 100.0
    voltage_drive_bound: float = 5.0
    current_drive_bound: float = 5.0
    spike_jump_bound: float = 2.0
    period_lower_factor: float = 0.5
    period_upper_factor: float = 2.0
    onset_velocity_mv_ms: float = 20.0
    onset_tau_min_ms: float = 0.05
    onset_tau_duration_factor: float = 10.0


@dataclass(frozen=True)
class RecoveryTimingModel:
    """Current spline plus one voltage/current/spike-driven recovery state."""

    current_levels: tuple[float, ...]
    baseline_log_period: tuple[float, ...]
    tau_ms: float
    voltage_drive: float
    current_drive: float
    spike_jump: float
    voltage_reference_mv: float
    voltage_scale_mv: float
    current_center: float
    current_scale: float
    minimum_period_ms: float
    maximum_period_ms: float

    def normalized_current(self, current_value: float) -> float:
        return (
            float(current_value) - self.current_center
        ) / self.current_scale

    def baseline(self, current_value: float) -> float:
        levels = np.asarray(self.current_levels, dtype=float)
        values = np.asarray(self.baseline_log_period, dtype=float)
        current = float(np.clip(current_value, levels[0], levels[-1]))
        return float(np.interp(current, levels, values))

    def period(self, current_value: float, state: float) -> float:
        log_period = self.baseline(current_value) + float(state)
        return float(
            math.exp(
                float(
                    np.clip(
                        log_period,
                        math.log(self.minimum_period_ms),
                        math.log(self.maximum_period_ms),
                    )
                )
            )
        )

    def target(self, voltage_mv: float, current_value: float) -> float:
        return (
            self.voltage_drive
            * (float(voltage_mv) - self.voltage_reference_mv)
            / self.voltage_scale_mv
            + self.current_drive * self.normalized_current(current_value)
        )


@dataclass(frozen=True)
class OnsetModel:
    """Interpolated time constant for the rest-to-cycle onset trajectory."""

    current_levels: tuple[float, ...]
    tau_ms: tuple[float, ...]
    fit_rmse_mv: tuple[float, ...]

    def tau(self, current_value: float) -> float:
        levels = np.asarray(self.current_levels, dtype=float)
        values = np.asarray(self.tau_ms, dtype=float)
        current = float(np.clip(current_value, levels[0], levels[-1]))
        return float(np.interp(current, levels, values))


@dataclass(frozen=True)
class RecoveryOnsetFit:
    timing: RecoveryTimingModel
    onset: OnsetModel
    candidate_table: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _RecoveryRows:
    currents: np.ndarray
    periods_ms: np.ndarray
    validation: np.ndarray
    states: np.ndarray


def _current_scale(cycles: Sequence[PhaseCycleSet]) -> tuple[float, float]:
    values = np.asarray(
        sorted({float(cycle.current_value) for cycle in cycles}),
        dtype=float,
    )
    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale <= 1e-9:
        scale = max(1.0, float(np.ptp(values)))
    return center, scale


def _recovery_rows(
    cycles: Sequence[PhaseCycleSet],
    tau_ms: float,
    voltage_reference_mv: float,
    current_center: float,
    current_scale: float,
    config: RecoveryOnsetConfig,
) -> _RecoveryRows:
    currents = []
    periods = []
    validation = []
    states = []
    for cycle_set in cycles:
        count = len(cycle_set.periods_ms)
        validation_start = max(
            1,
            int(
                math.floor(
                    (1.0 - config.late_validation_fraction) * count
                )
            ),
        )
        normalized_current = (
            float(cycle_set.current_value) - current_center
        ) / current_scale
        latency_decay = math.exp(
            -float(cycle_set.latency_ms) / tau_ms
        )
        voltage_state = 0.0
        current_state = normalized_current * (1.0 - latency_decay)
        # Period zero begins after the first observed spike.
        spike_state = 1.0
        for index, period in enumerate(cycle_set.periods_ms):
            currents.append(float(cycle_set.current_value))
            periods.append(float(period))
            validation.append(index >= validation_start)
            states.append((voltage_state, current_state, spike_state))
            voltage_state = _advance_voltage_state(
                voltage_state,
                cycle_set.voltage_cycles_mv[index],
                float(period),
                float(cycle_set.downstroke_durations_ms[index]),
                float(cycle_set.upstroke_durations_ms[index]),
                tau_ms,
                voltage_reference_mv,
                config.voltage_scale_mv,
            )
            decay = math.exp(-float(period) / tau_ms)
            current_state = (
                normalized_current
                + (current_state - normalized_current) * decay
            )
            spike_state = spike_state * decay + 1.0
    return _RecoveryRows(
        currents=np.asarray(currents, dtype=float),
        periods_ms=np.asarray(periods, dtype=float),
        validation=np.asarray(validation, dtype=bool),
        states=np.asarray(states, dtype=float),
    )


def _fit_fixed_current_effects(
    rows: _RecoveryRows,
    fit_mask: np.ndarray,
    config: RecoveryOnsetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = np.asarray(sorted(set(rows.currents)), dtype=float)
    target = np.log(np.maximum(rows.periods_ms, 1e-6))
    centered_states = []
    centered_target = []
    for level in levels:
        group = fit_mask & np.isclose(rows.currents, level)
        if not np.any(group):
            continue
        centered_states.append(
            rows.states[group] - np.mean(rows.states[group], axis=0)
        )
        centered_target.append(target[group] - np.mean(target[group]))
    design = np.concatenate(centered_states, axis=0)
    response = np.concatenate(centered_target)
    scales = np.maximum(np.std(design, axis=0), 1e-8)
    normalized = design / scales
    augmented_design = np.vstack(
        (
            normalized,
            math.sqrt(config.state_ridge)
            * np.eye(normalized.shape[1]),
        )
    )
    augmented_response = np.concatenate(
        (response, np.zeros(normalized.shape[1]))
    )
    physical_bounds = np.asarray(
        (
            config.voltage_drive_bound,
            config.current_drive_bound,
            config.spike_jump_bound,
        ),
        dtype=float,
    )
    normalized_coefficients = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(-physical_bounds * scales, physical_bounds * scales),
    ).x
    coefficients = normalized_coefficients / scales
    baseline = np.asarray(
        [
            np.mean(
                target[fit_mask & np.isclose(rows.currents, level)]
                - rows.states[
                    fit_mask & np.isclose(rows.currents, level)
                ]
                @ coefficients
            )
            for level in levels
        ],
        dtype=float,
    )
    return levels, baseline, coefficients


def _score_rows(
    rows: _RecoveryRows,
    config: RecoveryOnsetConfig,
) -> float:
    training = ~rows.validation
    levels, baseline, coefficients = _fit_fixed_current_effects(
        rows,
        training,
        config,
    )
    prediction = np.exp(
        np.interp(rows.currents, levels, baseline)
        + rows.states @ coefficients
    )
    target = rows.periods_ms[rows.validation]
    scale = max(1.0, float(np.std(target)))
    return float(
        np.sqrt(
            np.mean(
                np.square(prediction[rows.validation] - target)
            )
        )
        / scale
    )


def fit_recovery_timing(
    cycles: Sequence[PhaseCycleSet],
    voltage_reference_mv: float,
    config: RecoveryOnsetConfig | None = None,
) -> tuple[RecoveryTimingModel, tuple[dict[str, object], ...]]:
    """Fit the best recovery time constant by late-cycle prediction."""
    config = config or RecoveryOnsetConfig()
    if not cycles:
        raise ValueError("At least one phase-cycle set is required")
    current_center, current_scale = _current_scale(cycles)
    candidates = []
    best = None
    best_score = float("inf")
    for tau_ms in config.tau_candidates_ms:
        rows = _recovery_rows(
            cycles,
            tau_ms,
            voltage_reference_mv,
            current_center,
            current_scale,
            config,
        )
        score = _score_rows(rows, config)
        levels, baseline, coefficients = _fit_fixed_current_effects(
            rows,
            np.ones(len(rows.periods_ms), dtype=bool),
            config,
        )
        model = RecoveryTimingModel(
            current_levels=tuple(float(value) for value in levels),
            baseline_log_period=tuple(float(value) for value in baseline),
            tau_ms=float(tau_ms),
            voltage_drive=float(coefficients[0]),
            current_drive=float(coefficients[1]),
            spike_jump=float(coefficients[2]),
            voltage_reference_mv=float(voltage_reference_mv),
            voltage_scale_mv=config.voltage_scale_mv,
            current_center=current_center,
            current_scale=current_scale,
            minimum_period_ms=(
                config.period_lower_factor * float(np.min(rows.periods_ms))
            ),
            maximum_period_ms=(
                config.period_upper_factor * float(np.max(rows.periods_ms))
            ),
        )
        candidates.append(
            {
                "tau_ms": float(tau_ms),
                "late_period_nrmse": score,
                "voltage_drive": model.voltage_drive,
                "current_drive": model.current_drive,
                "spike_jump": model.spike_jump,
            }
        )
        if score < best_score:
            best = model
            best_score = score
    if best is None:
        raise RuntimeError("No recovery timing model was fitted")
    return (
        best,
        tuple(
            {
                **row,
                "tau_selected": row["tau_ms"] == best.tau_ms,
            }
            for row in candidates
        ),
    )


def _normalized_exponential(
    elapsed_ms: np.ndarray,
    duration_ms: float,
    tau_ms: float,
) -> np.ndarray:
    duration = max(float(duration_ms), 1e-9)
    tau = max(float(tau_ms), 1e-9)
    denominator = -math.expm1(-duration / tau)
    return -np.expm1(-np.asarray(elapsed_ms, dtype=float) / tau) / denominator


def _fit_trace_onset(
    trace: AbstractObservedTrace,
    config: RecoveryOnsetConfig,
) -> tuple[float, float, float] | None:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    indices = np.flatnonzero(during)
    if len(indices) < 5:
        return None
    peaks, _ = find_peaks(trace.voltage_mv[indices], height=0.0)
    if not len(peaks):
        return None
    peak = int(indices[peaks[0]])
    candidates = np.flatnonzero(
        (np.arange(len(trace.time_ms)) >= indices[0])
        & (np.arange(len(trace.time_ms)) <= peak)
        & (trace.velocity_mv_ms >= config.onset_velocity_mv_ms)
    )
    if not len(candidates):
        return None
    onset = int(candidates[0])
    segment = np.arange(indices[0], onset + 1)
    if len(segment) < 4:
        return None
    elapsed = trace.time_ms[segment] - trace.stimulus_start_ms
    duration = float(elapsed[-1])
    if duration <= 0.1:
        return None
    before = trace.voltage_mv[
        trace.time_ms < trace.stimulus_start_ms
    ]
    start_voltage = (
        float(np.median(before))
        if len(before)
        else float(trace.voltage_mv[segment[0]])
    )
    stop_voltage = float(trace.voltage_mv[onset])
    amplitude = stop_voltage - start_voltage
    if amplitude <= 1.0:
        return None
    observed_fraction = (
        trace.voltage_mv[segment] - start_voltage
    ) / amplitude

    def objective(log_tau: float) -> float:
        prediction = _normalized_exponential(
            elapsed,
            duration,
            math.exp(log_tau),
        )
        return float(np.mean(np.square(prediction - observed_fraction)))

    upper_tau = max(
        1.0,
        config.onset_tau_duration_factor * duration,
    )
    result = minimize_scalar(
        objective,
        bounds=(
            math.log(config.onset_tau_min_ms),
            math.log(upper_tau),
        ),
        method="bounded",
    )
    tau = float(math.exp(result.x))
    rmse = float(math.sqrt(result.fun) * amplitude)
    current = float(np.median(trace.input_value[during]))
    return current, tau, rmse


def fit_onset_model(
    traces: Sequence[AbstractObservedTrace],
    config: RecoveryOnsetConfig | None = None,
) -> OnsetModel:
    """Fit an exponential rest-to-upstroke trajectory at each current."""
    config = config or RecoveryOnsetConfig()
    rows = [
        row
        for trace in traces
        if (row := _fit_trace_onset(trace, config)) is not None
    ]
    if not rows:
        raise ValueError("No first-spike onset trajectory could be fitted")
    rows.sort(key=lambda row: row[0])
    return OnsetModel(
        current_levels=tuple(float(row[0]) for row in rows),
        tau_ms=tuple(float(row[1]) for row in rows),
        fit_rmse_mv=tuple(float(row[2]) for row in rows),
    )


def fit_recovery_onset_model(
    traces: Sequence[AbstractObservedTrace],
    cycles: Sequence[PhaseCycleSet],
    voltage_reference_mv: float,
    config: RecoveryOnsetConfig | None = None,
) -> RecoveryOnsetFit:
    """Fit the continuous recovery clock and onset trajectory."""
    config = config or RecoveryOnsetConfig()
    timing, candidates = fit_recovery_timing(
        cycles,
        voltage_reference_mv,
        config,
    )
    onset = fit_onset_model(traces, config)
    return RecoveryOnsetFit(
        timing=timing,
        onset=onset,
        candidate_table=candidates,
    )


def simulate_recovery_onset_model(
    waveform: PhaseTemplateModel,
    timing: RecoveryTimingModel,
    onset: OnsetModel,
    time_ms: Sequence[float],
    input_value: Sequence[float],
    initial_voltage_mv: float,
    use_onset_branch: bool = True,
) -> PhaseTemplateSimulation:
    """Simulate recovery timing with an optional rest-to-cycle onset branch."""
    time = np.asarray(time_ms, dtype=float)
    current = np.asarray(input_value, dtype=float)
    if time.shape != current.shape or len(time) < 2:
        raise ValueError("Time and input must be equal nontrivial arrays")
    dt = float(np.median(np.diff(time)))
    active_threshold = max(
        1e-12,
        0.1 * float(np.min(np.asarray(waveform.current_levels))),
    )
    voltage = np.empty_like(time)
    velocity = np.empty_like(time)
    phase_values = np.zeros_like(time)
    state_values = np.zeros_like(time)
    periods = np.full_like(time, np.nan)
    voltage[0] = initial_voltage_mv
    velocity[0] = 0.0
    phase = 0.0
    state = 0.0
    active = False
    in_onset = False
    onset_elapsed = 0.0
    onset_duration = 0.0
    onset_start_voltage = float(initial_voltage_mv)
    onset_stop_voltage = float(initial_voltage_mv)
    onset_tau = onset.tau(float(np.max(current)))
    current_period = timing.maximum_period_ms
    spline = None
    spline_current = float("nan")
    downstroke_ms = float("nan")
    upstroke_ms = float("nan")
    first_recovery_scale = 1.0
    for index in range(1, len(time)):
        applied = float(current[index])
        if applied < active_threshold:
            active = False
            in_onset = False
            state *= math.exp(-dt / timing.tau_ms)
            velocity[index] = -(
                voltage[index - 1] - waveform.resting_voltage_mv
            ) / waveform.rest_relaxation_ms
            voltage[index] = voltage[index - 1] + dt * velocity[index]
            phase_values[index] = phase
            state_values[index] = state
            continue
        if not active:
            active = True
            spline = _periodic_spline(waveform, applied)
            spline_current = applied
            current_period = timing.period(applied, state)
            downstroke_ms, recovery_ms, upstroke_ms = (
                waveform.segment_durations(applied, current_period)
            )
            if use_onset_branch:
                phase = 0.75
                in_onset = True
                onset_elapsed = 0.0
                onset_duration = max(
                    0.1,
                    waveform.latency(applied) - upstroke_ms,
                )
                onset_start_voltage = float(voltage[index - 1])
                onset_stop_voltage = float(spline(phase))
                onset_tau = onset.tau(applied)
            else:
                phase_grid = np.asarray(waveform.phase_grid)
                template = waveform.voltage_template(applied)
                derivative = spline(phase_grid, 1)
                candidates = np.flatnonzero(
                    (phase_grid >= 0.25)
                    & (phase_grid < 0.75)
                    & (derivative > 0.0)
                )
                if not len(candidates):
                    candidates = np.flatnonzero(derivative > 0.0)
                phase_index = int(
                    candidates[
                        np.argmin(
                            np.abs(
                                template[candidates]
                                - voltage[index - 1]
                            )
                        )
                    ]
                )
                phase = float(phase_grid[phase_index])
                remaining = max(
                    0.1,
                    (0.75 - phase) / 0.5 * recovery_ms,
                )
                first_recovery_scale = max(
                    0.1,
                    (waveform.latency(applied) - upstroke_ms) / remaining,
                )
        elif abs(applied - spline_current) > max(
            1e-9,
            1e-6 * abs(spline_current),
        ):
            spline = _periodic_spline(waveform, applied)
            spline_current = applied
        target = timing.target(voltage[index - 1], applied)
        decay = math.exp(-dt / timing.tau_ms)
        state = target + (state - target) * decay
        if in_onset:
            onset_elapsed = min(
                onset_duration,
                onset_elapsed + dt,
            )
            fraction = float(
                _normalized_exponential(
                    np.asarray((onset_elapsed,)),
                    onset_duration,
                    onset_tau,
                )[0]
            )
            voltage[index] = (
                onset_start_voltage
                + fraction * (onset_stop_voltage - onset_start_voltage)
            )
            velocity[index] = (
                voltage[index] - voltage[index - 1]
            ) / dt
            if onset_elapsed >= onset_duration:
                in_onset = False
            phase_values[index] = phase
            state_values[index] = state
            periods[index] = current_period
            continue
        recovery_ms = max(
            0.1,
            current_period - downstroke_ms - upstroke_ms,
        )
        if phase < 0.25:
            phase_rate = 0.25 / downstroke_ms
        elif phase < 0.75:
            phase_rate = 0.5 / (
                recovery_ms * first_recovery_scale
            )
        else:
            phase_rate = 0.25 / upstroke_ms
        phase += dt * phase_rate
        if phase >= 1.0:
            phase -= math.floor(phase)
            state += timing.spike_jump
            current_period = timing.period(applied, state)
            downstroke_ms, _, upstroke_ms = waveform.segment_durations(
                applied,
                current_period,
            )
            first_recovery_scale = 1.0
            phase_rate = 0.25 / downstroke_ms
        voltage[index] = float(spline(phase))
        velocity[index] = float(spline(phase, 1) * phase_rate)
        phase_values[index] = phase
        state_values[index] = state
        periods[index] = current_period
    return PhaseTemplateSimulation(
        time_ms=time,
        voltage_mv=voltage,
        velocity_mv_ms=velocity,
        phase=phase_values,
        memory=state_values,
        instantaneous_period_ms=periods,
    )
