"""Timing-model ladder for data-derived phase-template spike generators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from .phase_template_model import (
    PhaseCycleSet,
    PhaseTemplateModel,
    PhaseTemplateSimulation,
)


@dataclass(frozen=True)
class PhaseTimingConfig:
    """Configuration for fitting nested current/adaptation clocks."""

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
    period_lower_factor: float = 0.5
    period_upper_factor: float = 2.0
    family_complexity_penalty: float = 0.01
    voltage_drive_bound: float = 5.0
    spike_jump_bound: float = 2.0
    kernel_tau_basis_ms: tuple[float, ...] = (
        25.0,
        100.0,
        400.0,
        1600.0,
    )
    kernel_state_ridge: float = 0.05
    kernel_amplitude_bound: float = 2.0
    kernel_summary_lags_ms: tuple[float, ...] = (
        50.0,
        200.0,
        800.0,
    )


@dataclass(frozen=True)
class PhaseTimingModel:
    """Current-period spline with an optional scalar recovery state."""

    family: str
    current_levels: tuple[float, ...]
    baseline_log_period: tuple[float, ...]
    tau_ms: float | None
    voltage_drive: float
    spike_jump: float
    voltage_reference_mv: float
    voltage_scale_mv: float
    minimum_period_ms: float
    maximum_period_ms: float

    @property
    def has_state(self) -> bool:
        return self.tau_ms is not None

    @property
    def is_voltage_driven(self) -> bool:
        return self.family == "izhikevich_recovery"

    def baseline(self, current_value: float) -> float:
        levels = np.asarray(self.current_levels, dtype=float)
        values = np.asarray(self.baseline_log_period, dtype=float)
        current = float(np.clip(current_value, levels[0], levels[-1]))
        return float(np.interp(current, levels, values))

    def period(self, current_value: float, state: float = 0.0) -> float:
        log_period = self.baseline(current_value) + float(state)
        lower = math.log(self.minimum_period_ms)
        upper = math.log(self.maximum_period_ms)
        return float(math.exp(float(np.clip(log_period, lower, upper))))


@dataclass(frozen=True)
class PhaseTimingLadderResult:
    current_spline: PhaseTimingModel
    spike_exponential: PhaseTimingModel
    izhikevich_recovery: PhaseTimingModel
    selected: PhaseTimingModel
    candidate_table: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SpikeExponentialTimingResult:
    """Best spike-memory clock across candidate decay constants."""

    model: PhaseTimingModel
    candidate_table: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class PhaseKernelTimingModel:
    """Current-period spline plus a fixed bank of spike-history filters."""

    family: str
    current_levels: tuple[float, ...]
    baseline_log_period: tuple[float, ...]
    tau_basis_ms: tuple[float, ...]
    amplitudes: tuple[float, ...]
    minimum_period_ms: float
    maximum_period_ms: float

    def baseline(self, current_value: float) -> float:
        levels = np.asarray(self.current_levels, dtype=float)
        values = np.asarray(self.baseline_log_period, dtype=float)
        current = float(np.clip(current_value, levels[0], levels[-1]))
        return float(np.interp(current, levels, values))

    def period(
        self,
        current_value: float,
        states: Sequence[float] | np.ndarray,
    ) -> float:
        state_values = np.asarray(states, dtype=float)
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        if state_values.shape != amplitudes.shape:
            raise ValueError("Kernel state count does not match amplitudes")
        log_period = self.baseline(current_value) + float(
            state_values @ amplitudes
        )
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

    def kernel(self, lag_ms: float) -> float:
        taus = np.asarray(self.tau_basis_ms, dtype=float)
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        return float(amplitudes @ np.exp(-float(lag_ms) / taus))


@dataclass(frozen=True)
class PhaseKernelTimingResult:
    model: PhaseKernelTimingModel
    late_period_nrmse: float


@dataclass(frozen=True)
class _PeriodRows:
    currents: np.ndarray
    periods_ms: np.ndarray
    validation: np.ndarray
    voltage_state: np.ndarray
    spike_state: np.ndarray


@dataclass(frozen=True)
class _KernelRows:
    currents: np.ndarray
    periods_ms: np.ndarray
    validation: np.ndarray
    states: np.ndarray


def _advance_voltage_state(
    state: float,
    voltage_cycle_mv: np.ndarray,
    period_ms: float,
    downstroke_ms: float,
    upstroke_ms: float,
    tau_ms: float,
    voltage_reference_mv: float,
    voltage_scale_mv: float,
) -> float:
    """Advance a unit-gain Izhikevich-like voltage filter over one cycle."""
    voltage = np.asarray(voltage_cycle_mv, dtype=float)
    down_count = max(1, len(voltage) // 4)
    up_count = max(1, len(voltage) // 4)
    recovery_count = len(voltage) - down_count - up_count
    recovery_ms = max(0.1, period_ms - downstroke_ms - upstroke_ms)
    durations = np.concatenate(
        (
            np.full(down_count, downstroke_ms / down_count),
            np.full(recovery_count, recovery_ms / recovery_count),
            np.full(up_count, upstroke_ms / up_count),
        )
    )
    targets = (voltage - voltage_reference_mv) / voltage_scale_mv
    decay = np.exp(-durations / tau_ms)
    suffix_decay = np.cumprod(decay[::-1])[::-1]
    decay_after = np.concatenate((suffix_decay[1:], (1.0,)))
    driven = np.sum((1.0 - decay) * targets * decay_after)
    return float(state * suffix_decay[0] + driven)


def _period_rows(
    cycles: Sequence[PhaseCycleSet],
    tau_ms: float | None,
    voltage_reference_mv: float,
    config: PhaseTimingConfig,
    include_voltage_state: bool = True,
) -> _PeriodRows:
    currents = []
    periods = []
    validation = []
    voltage_states = []
    spike_states = []
    for cycle_set in cycles:
        count = len(cycle_set.periods_ms)
        validation_start = max(
            1,
            int(math.floor((1.0 - config.late_validation_fraction) * count)),
        )
        voltage_state = 0.0
        spike_state = 0.0
        for index, period in enumerate(cycle_set.periods_ms):
            currents.append(cycle_set.current_value)
            periods.append(float(period))
            validation.append(index >= validation_start)
            voltage_states.append(voltage_state)
            spike_states.append(spike_state)
            if tau_ms is not None:
                if include_voltage_state:
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
                spike_state = (
                    spike_state * math.exp(-float(period) / tau_ms) + 1.0
                )
    return _PeriodRows(
        currents=np.asarray(currents, dtype=float),
        periods_ms=np.asarray(periods, dtype=float),
        validation=np.asarray(validation, dtype=bool),
        voltage_state=np.asarray(voltage_states, dtype=float),
        spike_state=np.asarray(spike_states, dtype=float),
    )


def _kernel_period_rows(
    cycles: Sequence[PhaseCycleSet],
    tau_basis_ms: Sequence[float],
    config: PhaseTimingConfig,
) -> _KernelRows:
    taus = np.asarray(tau_basis_ms, dtype=float)
    if taus.ndim != 1 or not len(taus) or np.any(taus <= 0.0):
        raise ValueError("Kernel decay constants must be positive")
    currents = []
    periods = []
    validation = []
    state_rows = []
    for cycle_set in cycles:
        count = len(cycle_set.periods_ms)
        validation_start = max(
            1,
            int(math.floor((1.0 - config.late_validation_fraction) * count)),
        )
        states = np.zeros(len(taus), dtype=float)
        for index, period in enumerate(cycle_set.periods_ms):
            currents.append(cycle_set.current_value)
            periods.append(float(period))
            validation.append(index >= validation_start)
            state_rows.append(states.copy())
            states = states * np.exp(-float(period) / taus) + 1.0
    return _KernelRows(
        currents=np.asarray(currents, dtype=float),
        periods_ms=np.asarray(periods, dtype=float),
        validation=np.asarray(validation, dtype=bool),
        states=np.asarray(state_rows, dtype=float),
    )


def _fit_kernel_effects(
    rows: _KernelRows,
    fit_mask: np.ndarray,
    config: PhaseTimingConfig,
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
            math.sqrt(config.kernel_state_ridge)
            * np.eye(normalized.shape[1]),
        )
    )
    augmented_response = np.concatenate(
        (response, np.zeros(normalized.shape[1]))
    )
    bound = float(config.kernel_amplitude_bound) * scales
    normalized_coefficients = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(-bound, bound),
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


def _state_design(rows: _PeriodRows, family: str) -> np.ndarray:
    if family == "current_spline":
        return np.empty((len(rows.periods_ms), 0), dtype=float)
    if family == "spike_exponential":
        return rows.spike_state[:, None]
    if family == "izhikevich_recovery":
        return np.column_stack((rows.voltage_state, rows.spike_state))
    raise ValueError(f"Unknown timing family: {family}")


def _fit_fixed_current_effects(
    rows: _PeriodRows,
    family: str,
    fit_mask: np.ndarray,
    config: PhaseTimingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit current-level intercepts and within-current state effects."""
    levels = np.asarray(sorted(set(rows.currents)), dtype=float)
    target = np.log(np.maximum(rows.periods_ms, 1e-6))
    states = _state_design(rows, family)
    if states.shape[1]:
        centered_states = []
        centered_target = []
        for level in levels:
            group = fit_mask & np.isclose(rows.currents, level)
            if not np.any(group):
                continue
            centered_states.append(states[group] - np.mean(states[group], axis=0))
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
        if family == "spike_exponential":
            bounds = (
                np.asarray((-config.spike_jump_bound,)) * scales,
                np.asarray((config.spike_jump_bound,)) * scales,
            )
        else:
            bounds = (
                np.asarray(
                    (
                        -config.voltage_drive_bound,
                        -config.spike_jump_bound,
                    )
                )
                * scales,
                np.asarray(
                    (
                        config.voltage_drive_bound,
                        config.spike_jump_bound,
                    )
                )
                * scales,
            )
        normalized_coefficients = lsq_linear(
            augmented_design,
            augmented_response,
            bounds=bounds,
        ).x
        coefficients = normalized_coefficients / scales
    else:
        coefficients = np.empty(0, dtype=float)
    baseline = np.asarray(
        [
            np.mean(
                target[fit_mask & np.isclose(rows.currents, level)]
                - states[fit_mask & np.isclose(rows.currents, level)]
                @ coefficients
            )
            for level in levels
        ],
        dtype=float,
    )
    prediction = np.interp(rows.currents, levels, baseline)
    if states.shape[1]:
        prediction = prediction + states @ coefficients
    return levels, baseline, coefficients


def _validation_score(
    rows: _PeriodRows,
    family: str,
    config: PhaseTimingConfig,
) -> float:
    training = ~rows.validation
    levels, baseline, coefficients = _fit_fixed_current_effects(
        rows,
        family,
        training,
        config,
    )
    states = _state_design(rows, family)
    log_prediction = np.interp(rows.currents, levels, baseline)
    if states.shape[1]:
        log_prediction = log_prediction + states @ coefficients
    prediction = np.exp(log_prediction)
    score_mask = rows.validation
    scale = max(1.0, float(np.std(rows.periods_ms[score_mask])))
    return float(
        np.sqrt(
            np.mean(
                (prediction[score_mask] - rows.periods_ms[score_mask]) ** 2
            )
        )
        / scale
    )


def _build_model(
    cycles: Sequence[PhaseCycleSet],
    family: str,
    tau_ms: float | None,
    voltage_reference_mv: float,
    config: PhaseTimingConfig,
    rows: _PeriodRows | None = None,
) -> tuple[PhaseTimingModel, float]:
    if rows is None:
        rows = _period_rows(
            cycles,
            tau_ms,
            voltage_reference_mv,
            config,
        )
    score = _validation_score(rows, family, config)
    levels, baseline, coefficients = _fit_fixed_current_effects(
        rows,
        family,
        np.ones(len(rows.periods_ms), dtype=bool),
        config,
    )
    voltage_drive = (
        float(coefficients[0])
        if family == "izhikevich_recovery"
        else 0.0
    )
    spike_jump = (
        float(coefficients[-1])
        if family != "current_spline"
        else 0.0
    )
    return (
        PhaseTimingModel(
            family=family,
            current_levels=tuple(float(value) for value in levels),
            baseline_log_period=tuple(float(value) for value in baseline),
            tau_ms=tau_ms,
            voltage_drive=voltage_drive,
            spike_jump=spike_jump,
            voltage_reference_mv=float(voltage_reference_mv),
            voltage_scale_mv=config.voltage_scale_mv,
            minimum_period_ms=(
                config.period_lower_factor * float(np.min(rows.periods_ms))
            ),
            maximum_period_ms=(
                config.period_upper_factor * float(np.max(rows.periods_ms))
            ),
        ),
        score,
    )


def fit_phase_timing_ladder(
    cycles: Sequence[PhaseCycleSet],
    voltage_reference_mv: float,
    config: PhaseTimingConfig | None = None,
) -> PhaseTimingLadderResult:
    """Fit current-only, spike-memory, and voltage-driven recovery clocks."""
    config = config or PhaseTimingConfig()
    if not cycles:
        raise ValueError("At least one phase-cycle set is required")
    current_only, current_score = _build_model(
        cycles,
        "current_spline",
        None,
        voltage_reference_mv,
        config,
    )
    rows: list[dict[str, object]] = [
        {
            "family": "current_spline",
            "tau_ms": None,
            "late_period_nrmse": current_score,
            "selection_score": current_score,
        }
    ]

    best_exponential = None
    best_exponential_score = float("inf")
    best_izhikevich = None
    best_izhikevich_score = float("inf")
    for tau_ms in config.tau_candidates_ms:
        tau_rows = _period_rows(
            cycles,
            tau_ms,
            voltage_reference_mv,
            config,
        )
        exponential, exponential_score = _build_model(
            cycles,
            "spike_exponential",
            tau_ms,
            voltage_reference_mv,
            config,
            rows=tau_rows,
        )
        izhikevich, izhikevich_score = _build_model(
            cycles,
            "izhikevich_recovery",
            tau_ms,
            voltage_reference_mv,
            config,
            rows=tau_rows,
        )
        rows.extend(
            (
                {
                    "family": "spike_exponential",
                    "tau_ms": tau_ms,
                    "late_period_nrmse": exponential_score,
                    "selection_score": (
                        exponential_score + config.family_complexity_penalty
                    ),
                },
                {
                    "family": "izhikevich_recovery",
                    "tau_ms": tau_ms,
                    "late_period_nrmse": izhikevich_score,
                    "selection_score": (
                        izhikevich_score
                        + 2.0 * config.family_complexity_penalty
                    ),
                },
            )
        )
        if exponential_score < best_exponential_score:
            best_exponential = exponential
            best_exponential_score = exponential_score
        if izhikevich_score < best_izhikevich_score:
            best_izhikevich = izhikevich
            best_izhikevich_score = izhikevich_score
    if best_exponential is None or best_izhikevich is None:
        raise RuntimeError("Timing-state candidates were not fitted")

    family_models = (current_only, best_exponential, best_izhikevich)
    family_scores = (
        current_score,
        best_exponential_score + config.family_complexity_penalty,
        best_izhikevich_score + 2.0 * config.family_complexity_penalty,
    )
    selected = family_models[int(np.argmin(family_scores))]
    selected_tau = selected.tau_ms
    candidate_rows = tuple(
        {
            **row,
            "tau_selected_within_family": (
                row["family"] == "current_spline"
                or (
                    row["family"] == "spike_exponential"
                    and row["tau_ms"] == best_exponential.tau_ms
                )
                or (
                    row["family"] == "izhikevich_recovery"
                    and row["tau_ms"] == best_izhikevich.tau_ms
                )
            ),
            "family_selected": (
                row["family"] == selected.family
                and (
                    selected_tau is None or row["tau_ms"] == selected_tau
                )
            ),
        }
        for row in rows
    )
    return PhaseTimingLadderResult(
        current_spline=current_only,
        spike_exponential=best_exponential,
        izhikevich_recovery=best_izhikevich,
        selected=selected,
        candidate_table=candidate_rows,
    )


def fit_spike_exponential_timing(
    cycles: Sequence[PhaseCycleSet],
    voltage_reference_mv: float,
    config: PhaseTimingConfig | None = None,
) -> SpikeExponentialTimingResult:
    """Fit only the validated compact spike-memory timing family."""
    config = config or PhaseTimingConfig()
    if not cycles:
        raise ValueError("At least one phase-cycle set is required")
    best_model = None
    best_score = float("inf")
    rows = []
    for tau_ms in config.tau_candidates_ms:
        tau_rows = _period_rows(
            cycles,
            tau_ms,
            voltage_reference_mv,
            config,
            include_voltage_state=False,
        )
        model, score = _build_model(
            cycles,
            "spike_exponential",
            tau_ms,
            voltage_reference_mv,
            config,
            rows=tau_rows,
        )
        rows.append(
            {
                "family": "spike_exponential",
                "tau_ms": tau_ms,
                "late_period_nrmse": score,
            }
        )
        if score < best_score:
            best_model = model
            best_score = score
    if best_model is None:
        raise RuntimeError("Spike-memory timing candidates were not fitted")
    return SpikeExponentialTimingResult(
        model=best_model,
        candidate_table=tuple(
            {
                **row,
                "tau_selected": row["tau_ms"] == best_model.tau_ms,
            }
            for row in rows
        ),
    )


def fit_phase_kernel_timing(
    cycles: Sequence[PhaseCycleSet],
    config: PhaseTimingConfig | None = None,
) -> PhaseKernelTimingResult:
    """Fit a shared fixed-timescale spike-history kernel."""
    config = config or PhaseTimingConfig()
    if not cycles:
        raise ValueError("At least one phase-cycle set is required")
    rows = _kernel_period_rows(
        cycles,
        config.kernel_tau_basis_ms,
        config,
    )
    training = ~rows.validation
    levels, baseline, coefficients = _fit_kernel_effects(
        rows,
        training,
        config,
    )
    log_prediction = (
        np.interp(rows.currents, levels, baseline)
        + rows.states @ coefficients
    )
    prediction = np.exp(log_prediction)
    score_mask = rows.validation
    scale = max(1.0, float(np.std(rows.periods_ms[score_mask])))
    score = float(
        np.sqrt(
            np.mean(
                (prediction[score_mask] - rows.periods_ms[score_mask]) ** 2
            )
        )
        / scale
    )
    levels, baseline, coefficients = _fit_kernel_effects(
        rows,
        np.ones(len(rows.periods_ms), dtype=bool),
        config,
    )
    return PhaseKernelTimingResult(
        model=PhaseKernelTimingModel(
            family="fixed_multiscale_kernel",
            current_levels=tuple(float(value) for value in levels),
            baseline_log_period=tuple(float(value) for value in baseline),
            tau_basis_ms=tuple(
                float(value) for value in config.kernel_tau_basis_ms
            ),
            amplitudes=tuple(float(value) for value in coefficients),
            minimum_period_ms=(
                config.period_lower_factor * float(np.min(rows.periods_ms))
            ),
            maximum_period_ms=(
                config.period_upper_factor * float(np.max(rows.periods_ms))
            ),
        ),
        late_period_nrmse=score,
    )


def _periodic_spline(
    waveform: PhaseTemplateModel,
    current_value: float,
) -> CubicSpline:
    phase = np.asarray(waveform.phase_grid, dtype=float)
    voltage = waveform.voltage_template(current_value)
    return CubicSpline(
        np.concatenate((phase, (1.0,))),
        np.concatenate((voltage, (voltage[0],))),
        bc_type="periodic",
    )


def simulate_phase_timing_model(
    waveform: PhaseTemplateModel,
    timing: PhaseTimingModel,
    time_ms: Sequence[float],
    input_value: Sequence[float],
    initial_voltage_mv: float,
) -> PhaseTemplateSimulation:
    """Simulate a fixed phase template with one of the timing clocks."""
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
    first_cycle = True
    first_recovery_time_scale = 1.0
    current_period = timing.maximum_period_ms
    spline = None
    spline_current = float("nan")
    current_downstroke_ms = float("nan")
    current_upstroke_ms = float("nan")
    for index in range(1, len(time)):
        applied = float(current[index])
        if applied < active_threshold:
            active = False
            first_cycle = True
            if timing.has_state:
                state *= math.exp(-dt / float(timing.tau_ms))
            else:
                state = 0.0
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
            phase_grid = np.asarray(waveform.phase_grid)
            template = waveform.voltage_template(applied)
            derivative = spline(phase_grid, 1)
            recovery_rising = (
                (phase_grid >= 0.25)
                & (phase_grid < 0.75)
                & (derivative > 0.0)
            )
            candidates = np.flatnonzero(recovery_rising)
            if not len(candidates):
                candidates = np.flatnonzero(derivative > 0.0)
            phase_index = int(
                candidates[
                    np.argmin(
                        np.abs(
                            template[candidates] - voltage[index - 1]
                        )
                    )
                ]
            )
            phase = float(phase_grid[phase_index])
            current_period = timing.period(applied, state)
            (
                current_downstroke_ms,
                recovery,
                current_upstroke_ms,
            ) = waveform.segment_durations(
                applied,
                current_period,
            )
            remaining_recovery = max(
                0.1,
                (0.75 - phase) / 0.5 * recovery,
            )
            first_recovery_time_scale = max(
                0.1,
                (
                    waveform.latency(applied)
                    - current_upstroke_ms
                )
                / remaining_recovery,
            )
        elif abs(applied - spline_current) > max(
            1e-9,
            1e-6 * abs(spline_current),
        ):
            spline = _periodic_spline(waveform, applied)
            spline_current = applied
            (
                current_downstroke_ms,
                _,
                current_upstroke_ms,
            ) = waveform.segment_durations(
                applied,
                current_period,
            )
        if timing.has_state:
            decay = math.exp(-dt / float(timing.tau_ms))
            target = (
                timing.voltage_drive
                * (
                    voltage[index - 1] - timing.voltage_reference_mv
                )
                / timing.voltage_scale_mv
                if timing.is_voltage_driven
                else 0.0
            )
            state = target + (state - target) * decay
        recovery = max(
            0.1,
            current_period
            - current_downstroke_ms
            - current_upstroke_ms,
        )
        if phase < 0.25:
            phase_rate = 0.25 / current_downstroke_ms
        elif phase < 0.75:
            recovery_scale = (
                first_recovery_time_scale if first_cycle else 1.0
            )
            phase_rate = 0.5 / (recovery * recovery_scale)
        else:
            phase_rate = 0.25 / current_upstroke_ms
        phase += dt * phase_rate
        if phase >= 1.0:
            wraps = int(math.floor(phase))
            phase -= wraps
            if timing.has_state:
                state += float(wraps) * timing.spike_jump
            first_cycle = False
            first_recovery_time_scale = 1.0
            current_period = timing.period(applied, state)
            phase_rate = 0.25 / current_downstroke_ms
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


def simulate_phase_kernel_timing_model(
    waveform: PhaseTemplateModel,
    timing: PhaseKernelTimingModel,
    time_ms: Sequence[float],
    input_value: Sequence[float],
    initial_voltage_mv: float,
) -> PhaseTemplateSimulation:
    """Simulate the phase template with fixed multi-timescale memory."""
    time = np.asarray(time_ms, dtype=float)
    current = np.asarray(input_value, dtype=float)
    if time.shape != current.shape or len(time) < 2:
        raise ValueError("Time and input must be equal nontrivial arrays")
    dt = float(np.median(np.diff(time)))
    taus = np.asarray(timing.tau_basis_ms, dtype=float)
    amplitudes = np.asarray(timing.amplitudes, dtype=float)
    active_threshold = max(
        1e-12,
        0.1 * float(np.min(np.asarray(waveform.current_levels))),
    )
    voltage = np.empty_like(time)
    velocity = np.empty_like(time)
    phase_values = np.zeros_like(time)
    memory_values = np.zeros_like(time)
    periods = np.full_like(time, np.nan)
    voltage[0] = initial_voltage_mv
    velocity[0] = 0.0
    phase = 0.0
    states = np.zeros(len(taus), dtype=float)
    active = False
    first_cycle = True
    first_recovery_time_scale = 1.0
    current_period = timing.maximum_period_ms
    spline = None
    spline_current = float("nan")
    current_downstroke_ms = float("nan")
    current_upstroke_ms = float("nan")
    for index in range(1, len(time)):
        applied = float(current[index])
        states *= np.exp(-dt / taus)
        if applied < active_threshold:
            active = False
            first_cycle = True
            velocity[index] = -(
                voltage[index - 1] - waveform.resting_voltage_mv
            ) / waveform.rest_relaxation_ms
            voltage[index] = voltage[index - 1] + dt * velocity[index]
            phase_values[index] = phase
            memory_values[index] = float(states @ amplitudes)
            continue
        if not active:
            active = True
            spline = _periodic_spline(waveform, applied)
            spline_current = applied
            phase_grid = np.asarray(waveform.phase_grid)
            template = waveform.voltage_template(applied)
            derivative = spline(phase_grid, 1)
            recovery_rising = (
                (phase_grid >= 0.25)
                & (phase_grid < 0.75)
                & (derivative > 0.0)
            )
            candidates = np.flatnonzero(recovery_rising)
            if not len(candidates):
                candidates = np.flatnonzero(derivative > 0.0)
            phase_index = int(
                candidates[
                    np.argmin(
                        np.abs(template[candidates] - voltage[index - 1])
                    )
                ]
            )
            phase = float(phase_grid[phase_index])
            current_period = timing.period(applied, states)
            (
                current_downstroke_ms,
                recovery,
                current_upstroke_ms,
            ) = waveform.segment_durations(applied, current_period)
            remaining_recovery = max(
                0.1,
                (0.75 - phase) / 0.5 * recovery,
            )
            first_recovery_time_scale = max(
                0.1,
                (
                    waveform.latency(applied)
                    - current_upstroke_ms
                )
                / remaining_recovery,
            )
        elif abs(applied - spline_current) > max(
            1e-9,
            1e-6 * abs(spline_current),
        ):
            spline = _periodic_spline(waveform, applied)
            spline_current = applied
            (
                current_downstroke_ms,
                _,
                current_upstroke_ms,
            ) = waveform.segment_durations(applied, current_period)
        recovery = max(
            0.1,
            current_period
            - current_downstroke_ms
            - current_upstroke_ms,
        )
        if phase < 0.25:
            phase_rate = 0.25 / current_downstroke_ms
        elif phase < 0.75:
            recovery_scale = (
                first_recovery_time_scale if first_cycle else 1.0
            )
            phase_rate = 0.5 / (recovery * recovery_scale)
        else:
            phase_rate = 0.25 / current_upstroke_ms
        phase += dt * phase_rate
        if phase >= 1.0:
            wraps = int(math.floor(phase))
            phase -= wraps
            states += float(wraps)
            first_cycle = False
            first_recovery_time_scale = 1.0
            current_period = timing.period(applied, states)
            phase_rate = 0.25 / current_downstroke_ms
        voltage[index] = float(spline(phase))
        velocity[index] = float(spline(phase, 1) * phase_rate)
        phase_values[index] = phase
        memory_values[index] = float(states @ amplitudes)
        periods[index] = current_period
    return PhaseTemplateSimulation(
        time_ms=time,
        voltage_mv=voltage,
        velocity_mv_ms=velocity,
        phase=phase_values,
        memory=memory_values,
        instantaneous_period_ms=periods,
    )


def timing_parameter_row(
    model: PhaseTimingModel,
    grid_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float | str | None]:
    """Export fixed-width clock parameters for population analyses."""
    lower = float(model.current_levels[0])
    upper = float(model.current_levels[-1])
    row: dict[str, float | str | None] = {
        "family": model.family,
        "tau_ms": model.tau_ms,
        "voltage_drive": model.voltage_drive,
        "spike_jump": model.spike_jump,
        "training_current_min": lower,
        "training_current_max": upper,
    }
    for fraction in grid_fractions:
        current = lower + float(fraction) * (upper - lower)
        row[f"baseline_period_q{int(round(100 * fraction)):03d}_ms"] = (
            model.period(current, 0.0)
        )
    return row


def kernel_timing_parameter_row(
    model: PhaseKernelTimingModel,
    grid_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    summary_lags_ms: Sequence[float] = (50.0, 200.0, 800.0),
) -> dict[str, float | str]:
    """Export identifiable fixed-lag summaries of an adaptation kernel."""
    lower = float(model.current_levels[0])
    upper = float(model.current_levels[-1])
    row: dict[str, float | str] = {
        "family": model.family,
        "training_current_min": lower,
        "training_current_max": upper,
    }
    for index, (tau, amplitude) in enumerate(
        zip(model.tau_basis_ms, model.amplitudes),
        start=1,
    ):
        row[f"kernel_tau_{index:02d}_ms"] = float(tau)
        row[f"kernel_amplitude_{index:02d}"] = float(amplitude)
    for lag in summary_lags_ms:
        row[f"kernel_value_{int(round(float(lag))):04d}ms"] = (
            model.kernel(float(lag))
        )
    amplitudes = np.asarray(model.amplitudes, dtype=float)
    taus = np.asarray(model.tau_basis_ms, dtype=float)
    absolute = np.abs(amplitudes)
    row["kernel_value_0000ms"] = float(np.sum(amplitudes))
    row["kernel_absolute_mass"] = float(np.sum(absolute))
    row["kernel_log_tau_centroid_ms"] = (
        float(
            np.exp(
                np.sum(absolute * np.log(taus))
                / np.sum(absolute)
            )
        )
        if np.sum(absolute) > 1e-12
        else 0.0
    )
    for fraction in grid_fractions:
        current = lower + float(fraction) * (upper - lower)
        zero_state = np.zeros(len(model.tau_basis_ms), dtype=float)
        row[f"baseline_period_q{int(round(100 * fraction)):03d}_ms"] = (
            model.period(current, zero_state)
        )
    return row
