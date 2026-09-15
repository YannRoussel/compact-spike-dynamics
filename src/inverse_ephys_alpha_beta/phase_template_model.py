"""Data-derived phase oscillator with an optional slow adaptation state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from .abstract_phase_model import AbstractObservedTrace


@dataclass(frozen=True)
class PhaseTemplateConfig:
    phase_points: int = 256
    downstroke_phase_points: int = 64
    recovery_phase_points: int = 128
    upstroke_phase_points: int = 64
    upstroke_onset_velocity_mv_ms: float = 20.0
    minimum_peak_height_mv: float = 0.0
    minimum_peak_distance_ms: float = 1.0
    minimum_complete_cycles: int = 2
    slow_tau_candidates_ms: tuple[float, ...] = (
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
    )
    slow_complexity_penalty: float = 0.02
    period_ridge: float = 1e-3
    late_validation_fraction: float = 0.3
    rest_relaxation_ms: float = 10.0
    period_lower_factor: float = 0.5
    period_upper_factor: float = 2.0
    fourier_harmonics: int = 20


@dataclass(frozen=True)
class PhaseCycleSet:
    trace_name: str
    current_value: float
    latency_ms: float
    periods_ms: np.ndarray
    downstroke_durations_ms: np.ndarray
    upstroke_durations_ms: np.ndarray
    voltage_cycles_mv: np.ndarray


@dataclass(frozen=True)
class PhaseTemplateSimulation:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    velocity_mv_ms: np.ndarray
    phase: np.ndarray
    memory: np.ndarray
    instantaneous_period_ms: np.ndarray


@dataclass(frozen=True)
class PhaseTemplateModel:
    phase_grid: tuple[float, ...]
    current_levels: tuple[float, ...]
    voltage_templates_mv: tuple[tuple[float, ...], ...]
    latency_ms: tuple[float, ...]
    downstroke_duration_ms: tuple[float, ...]
    upstroke_duration_ms: tuple[float, ...]
    resting_voltage_mv: float
    period_coefficients: tuple[float, ...]
    input_center: float
    input_scale: float
    memory_scale: float
    slow_tau_ms: float | None
    rheobase_current: float
    input_transform_offset: float
    minimum_period_ms: float
    maximum_period_ms: float
    rest_relaxation_ms: float

    @property
    def has_slow_state(self) -> bool:
        return self.slow_tau_ms is not None

    def _interpolate_current_values(
        self,
        values: np.ndarray,
        current_value: float,
    ) -> np.ndarray:
        levels = np.asarray(self.current_levels, dtype=float)
        current = float(np.clip(current_value, levels[0], levels[-1]))
        if len(levels) == 1:
            return np.asarray(values[0], dtype=float)
        upper = int(np.searchsorted(levels, current, side="right"))
        upper = min(max(1, upper), len(levels) - 1)
        lower = upper - 1
        fraction = (current - levels[lower]) / (
            levels[upper] - levels[lower]
        )
        return (
            (1.0 - fraction) * np.asarray(values[lower], dtype=float)
            + fraction * np.asarray(values[upper], dtype=float)
        )

    def voltage_template(self, current_value: float) -> np.ndarray:
        return self._interpolate_current_values(
            np.asarray(self.voltage_templates_mv, dtype=float),
            current_value,
        )

    def latency(self, current_value: float) -> float:
        return float(
            self._interpolate_current_values(
                np.asarray(self.latency_ms, dtype=float)[:, None],
                current_value,
            )[0]
        )

    def period(self, current_value: float, memory: float) -> float:
        transformed_input = 1.0 / max(
            self.input_transform_offset,
            float(current_value)
            - self.rheobase_current
            + self.input_transform_offset,
        )
        normalized_input = (
            transformed_input - self.input_center
        ) / self.input_scale
        features = [1.0, normalized_input]
        if self.has_slow_state:
            normalized_memory = float(memory) / self.memory_scale
            features.append(normalized_memory)
        value = float(
            np.asarray(features)
            @ np.asarray(self.period_coefficients)
        )
        return float(
            np.clip(
                value,
                self.minimum_period_ms,
                self.maximum_period_ms,
            )
        )

    def segment_durations(
        self,
        current_value: float,
        period_ms: float,
    ) -> tuple[float, float, float]:
        downstroke = float(
            self._interpolate_current_values(
                np.asarray(self.downstroke_duration_ms)[:, None],
                current_value,
            )[0]
        )
        upstroke = float(
            self._interpolate_current_values(
                np.asarray(self.upstroke_duration_ms)[:, None],
                current_value,
            )[0]
        )
        recovery = max(
            0.1,
            float(period_ms) - downstroke - upstroke,
        )
        return downstroke, recovery, upstroke


@dataclass(frozen=True)
class PhaseTemplateLadderResult:
    no_slow_model: PhaseTemplateModel
    slow_models: tuple[PhaseTemplateModel, ...]
    selected: PhaseTemplateModel
    candidate_table: tuple[dict[str, object], ...]
    cycles: tuple[PhaseCycleSet, ...]


def extract_phase_cycles(
    trace: AbstractObservedTrace,
    config: PhaseTemplateConfig | None = None,
) -> PhaseCycleSet | None:
    """Extract peak-to-peak voltage cycles from one repetitive trace."""
    config = config or PhaseTemplateConfig()
    dt_ms = float(np.median(np.diff(trace.time_ms)))
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    indices = np.flatnonzero(during)
    if len(indices) < 3:
        return None
    minimum_distance = max(
        1,
        int(round(config.minimum_peak_distance_ms / dt_ms)),
    )
    local_peaks, _ = find_peaks(
        trace.voltage_mv[indices],
        height=config.minimum_peak_height_mv,
        distance=minimum_distance,
    )
    peaks = indices[local_peaks]
    if len(peaks) < config.minimum_complete_cycles + 1:
        return None
    if (
        config.downstroke_phase_points
        + config.recovery_phase_points
        + config.upstroke_phase_points
        != config.phase_points
    ):
        raise ValueError("Phase segment points must sum to phase_points")
    down_phase = np.linspace(
        0.0,
        0.25,
        config.downstroke_phase_points,
        endpoint=False,
    )
    recovery_phase = np.linspace(
        0.25,
        0.75,
        config.recovery_phase_points,
        endpoint=False,
    )
    up_phase = np.linspace(
        0.75,
        1.0,
        config.upstroke_phase_points,
        endpoint=False,
    )
    periods = np.diff(trace.time_ms[peaks])
    cycles = []
    downstroke_durations = []
    upstroke_durations = []
    for start, stop in zip(peaks[:-1], peaks[1:]):
        between = np.arange(start, stop + 1)
        trough = int(
            between[np.argmin(trace.voltage_mv[between])]
        )
        after_trough = np.arange(trough, stop + 1)
        onset_candidates = after_trough[
            trace.velocity_mv_ms[after_trough]
            >= config.upstroke_onset_velocity_mv_ms
        ]
        onset = (
            int(onset_candidates[0])
            if len(onset_candidates)
            else max(trough + 1, stop - 1)
        )

        def resample(
            segment_start: int,
            segment_stop: int,
            target_phase: np.ndarray,
        ) -> np.ndarray:
            segment = np.arange(segment_start, segment_stop + 1)
            local_phase = np.linspace(
                target_phase[0],
                (
                    target_phase[-1]
                    + (
                        target_phase[1] - target_phase[0]
                        if len(target_phase) > 1
                        else 0.0
                    )
                ),
                len(segment),
            )
            return np.interp(
                target_phase,
                local_phase,
                trace.voltage_mv[segment],
            )

        cycles.append(
            np.concatenate(
                (
                    resample(start, trough, down_phase),
                    resample(trough, onset, recovery_phase),
                    resample(onset, stop, up_phase),
                )
            )
        )
        downstroke_durations.append(
            trace.time_ms[trough] - trace.time_ms[start]
        )
        upstroke_durations.append(
            trace.time_ms[stop] - trace.time_ms[onset]
        )
    current = float(np.median(trace.input_value[during]))
    return PhaseCycleSet(
        trace_name=trace.name,
        current_value=current,
        latency_ms=float(
            trace.time_ms[peaks[0]] - trace.stimulus_start_ms
        ),
        periods_ms=np.asarray(periods, dtype=float),
        downstroke_durations_ms=np.asarray(
            downstroke_durations,
            dtype=float,
        ),
        upstroke_durations_ms=np.asarray(
            upstroke_durations,
            dtype=float,
        ),
        voltage_cycles_mv=np.asarray(cycles, dtype=float),
    )


def _memory_sequence(
    periods_ms: np.ndarray,
    tau_ms: float,
) -> np.ndarray:
    state = 0.0
    values = np.empty(len(periods_ms), dtype=float)
    for index, period in enumerate(periods_ms):
        values[index] = state
        state = state * math.exp(-float(period) / tau_ms) + 1.0
    return values


def _period_rows(
    cycles: Sequence[PhaseCycleSet],
    tau_ms: float | None,
    input_center: float,
    input_scale: float,
    rheobase_current: float,
    input_transform_offset: float,
    late_validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = []
    targets = []
    validation = []
    raw_memory = []
    for cycle_set in cycles:
        count = len(cycle_set.periods_ms)
        validation_start = max(
            1,
            int(
                math.floor(
                    (1.0 - late_validation_fraction) * count
                )
            ),
        )
        memory = (
            _memory_sequence(cycle_set.periods_ms, tau_ms)
            if tau_ms is not None
            else np.zeros(count)
        )
        transformed_input = 1.0 / max(
            input_transform_offset,
            cycle_set.current_value
            - rheobase_current
            + input_transform_offset,
        )
        normalized_input = (
            transformed_input - input_center
        ) / input_scale
        for index, (period, state) in enumerate(
            zip(cycle_set.periods_ms, memory)
        ):
            rows.append((normalized_input, state))
            targets.append(period)
            validation.append(index >= validation_start)
            raw_memory.append(state)
    memory_scale = (
        max(1.0, float(np.max(raw_memory)))
        if tau_ms is not None
        else 1.0
    )
    design = []
    for normalized_input, state in rows:
        features = [1.0, normalized_input]
        if tau_ms is not None:
            normalized_memory = state / memory_scale
            features.append(normalized_memory)
        design.append(features)
    return (
        np.asarray(design, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(validation, dtype=bool),
        memory_scale,
    )


def _fit_period_model(
    cycles: Sequence[PhaseCycleSet],
    tau_ms: float | None,
    input_center: float,
    input_scale: float,
    rheobase_current: float,
    input_transform_offset: float,
    config: PhaseTemplateConfig,
    validation_score: bool,
) -> tuple[np.ndarray, float, float]:
    design, target, validation, memory_scale = _period_rows(
        cycles,
        tau_ms,
        input_center,
        input_scale,
        rheobase_current,
        input_transform_offset,
        config.late_validation_fraction,
    )
    training = ~validation if validation_score else np.ones(
        len(target),
        dtype=bool,
    )
    normal = design[training].T @ design[training]
    normal += config.period_ridge * np.eye(normal.shape[0])
    right = design[training].T @ target[training]
    coefficients = np.linalg.solve(normal, right)
    score_mask = validation if validation_score else training
    residual = design[score_mask] @ coefficients - target[score_mask]
    scale = max(1.0, float(np.std(target[score_mask])))
    nrmse = float(np.sqrt(np.mean(residual**2)) / scale)
    return coefficients, memory_scale, nrmse


def _template_by_current(
    cycles: Sequence[PhaseCycleSet],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    levels = np.asarray(
        sorted({cycle.current_value for cycle in cycles}),
        dtype=float,
    )
    templates = []
    latencies = []
    downstroke_durations = []
    upstroke_durations = []
    for level in levels:
        matching = [
            cycle
            for cycle in cycles
            if np.isclose(cycle.current_value, level)
        ]
        trace_templates = [
            np.median(cycle.voltage_cycles_mv, axis=0)
            for cycle in matching
        ]
        templates.append(np.mean(trace_templates, axis=0))
        latencies.append(np.median([cycle.latency_ms for cycle in matching]))
        downstroke_durations.append(
            np.median(
                np.concatenate(
                    [
                        cycle.downstroke_durations_ms
                        for cycle in matching
                    ]
                )
            )
        )
        upstroke_durations.append(
            np.median(
                np.concatenate(
                    [
                        cycle.upstroke_durations_ms
                        for cycle in matching
                    ]
                )
            )
        )
    return (
        levels,
        np.asarray(templates),
        np.asarray(latencies),
        np.asarray(downstroke_durations),
        np.asarray(upstroke_durations),
    )


def fit_phase_template_ladder(
    traces: Sequence[AbstractObservedTrace],
    resting_voltage_mv: float | None = None,
    config: PhaseTemplateConfig | None = None,
) -> PhaseTemplateLadderResult:
    """Fit phase templates and select slow memory from late-cycle prediction."""
    config = config or PhaseTemplateConfig()
    extracted = tuple(
        cycle
        for trace in traces
        for cycle in (extract_phase_cycles(trace, config),)
        if cycle is not None
    )
    if not extracted:
        raise ValueError("No trace contains enough complete spike cycles")
    (
        levels,
        templates,
        latencies,
        downstroke_durations,
        upstroke_durations,
    ) = _template_by_current(extracted)
    spiking_currents = []
    for trace in traces:
        during = (
            (trace.time_ms >= trace.stimulus_start_ms)
            & (trace.time_ms < trace.stimulus_end_ms)
        )
        above = trace.voltage_mv >= config.minimum_peak_height_mv
        if np.any(above[1:] & ~above[:-1] & during[1:]):
            spiking_currents.append(
                float(np.median(trace.input_value[during]))
            )
    rheobase_current = float(np.min(spiking_currents))
    input_transform_offset = max(
        1e-6,
        0.01
        * (
            float(np.max(levels))
            - rheobase_current
        ),
    )
    transformed_levels = 1.0 / (
        levels - rheobase_current + input_transform_offset
    )
    input_center = float(np.mean(transformed_levels))
    input_scale = max(
        1e-8,
        float(np.ptp(transformed_levels)),
    )
    all_periods = np.concatenate(
        [cycle.periods_ms for cycle in extracted]
    )
    minimum_period = (
        config.period_lower_factor * float(np.min(all_periods))
    )
    maximum_period = (
        config.period_upper_factor * float(np.max(all_periods))
    )
    rest = (
        float(resting_voltage_mv)
        if resting_voltage_mv is not None
        else float(
            np.median(
                [
                    np.median(
                        trace.voltage_mv[
                            trace.time_ms < trace.stimulus_start_ms
                        ]
                    )
                    for trace in traces
                ]
            )
        )
    )

    def build(tau_ms: float | None) -> tuple[PhaseTemplateModel, float]:
        _, _, score = _fit_period_model(
            extracted,
            tau_ms,
            input_center,
            input_scale,
            rheobase_current,
            input_transform_offset,
            config,
            validation_score=True,
        )
        coefficients, memory_scale, _ = _fit_period_model(
            extracted,
            tau_ms,
            input_center,
            input_scale,
            rheobase_current,
            input_transform_offset,
            config,
            validation_score=False,
        )
        model = PhaseTemplateModel(
            phase_grid=tuple(
                np.linspace(
                    0.0,
                    1.0,
                    config.phase_points,
                    endpoint=False,
                )
            ),
            current_levels=tuple(levels),
            voltage_templates_mv=tuple(
                tuple(row) for row in templates
            ),
            latency_ms=tuple(latencies),
            downstroke_duration_ms=tuple(downstroke_durations),
            upstroke_duration_ms=tuple(upstroke_durations),
            resting_voltage_mv=rest,
            period_coefficients=tuple(coefficients),
            input_center=input_center,
            input_scale=input_scale,
            memory_scale=memory_scale,
            slow_tau_ms=tau_ms,
            rheobase_current=rheobase_current,
            input_transform_offset=input_transform_offset,
            minimum_period_ms=minimum_period,
            maximum_period_ms=maximum_period,
            rest_relaxation_ms=config.rest_relaxation_ms,
        )
        return model, score

    no_slow, no_slow_score = build(None)
    slow_pairs = tuple(
        build(tau) for tau in config.slow_tau_candidates_ms
    )
    rows = [
        {
            "label": "phase_only",
            "slow_tau_ms": None,
            "late_period_nrmse": no_slow_score,
            "selection_score": no_slow_score,
        }
    ]
    selected = no_slow
    best_score = no_slow_score
    for (model, score), tau in zip(
        slow_pairs,
        config.slow_tau_candidates_ms,
    ):
        selection_score = score + config.slow_complexity_penalty
        rows.append(
            {
                "label": f"phase_slow_{tau:g}_ms",
                "slow_tau_ms": tau,
                "late_period_nrmse": score,
                "selection_score": selection_score,
            }
        )
        if selection_score < best_score:
            selected = model
            best_score = selection_score
    return PhaseTemplateLadderResult(
        no_slow_model=no_slow,
        slow_models=tuple(pair[0] for pair in slow_pairs),
        selected=selected,
        candidate_table=tuple(rows),
        cycles=extracted,
    )


def _periodic_spline(
    model: PhaseTemplateModel,
    current_value: float,
) -> CubicSpline:
    phase = np.asarray(model.phase_grid)
    voltage = model.voltage_template(current_value)
    return CubicSpline(
        np.concatenate((phase, (1.0,))),
        np.concatenate((voltage, (voltage[0],))),
        bc_type="periodic",
    )


def simulate_phase_template_model(
    model: PhaseTemplateModel,
    time_ms: Sequence[float],
    input_value: Sequence[float],
    initial_voltage_mv: float,
) -> PhaseTemplateSimulation:
    """Generate a continuous spike train from phase and slow-memory dynamics."""
    time = np.asarray(time_ms, dtype=float)
    current = np.asarray(input_value, dtype=float)
    if time.shape != current.shape or len(time) < 2:
        raise ValueError("Time and input must be equal nontrivial arrays")
    dt = float(np.median(np.diff(time)))
    levels = np.asarray(model.current_levels)
    active_threshold = max(1e-12, 0.1 * float(np.min(levels)))
    voltage = np.empty_like(time)
    velocity = np.empty_like(time)
    phase_values = np.zeros_like(time)
    memory_values = np.zeros_like(time)
    periods = np.full_like(time, np.nan)
    voltage[0] = initial_voltage_mv
    velocity[0] = 0.0
    phase = 0.0
    memory = 0.0
    active = False
    first_cycle = True
    first_recovery_time_scale = 1.0
    current_period = model.maximum_period_ms
    spline = None
    spline_current = float("nan")
    for index in range(1, len(time)):
        applied = float(current[index])
        if applied < active_threshold:
            active = False
            first_cycle = True
            memory *= (
                math.exp(-dt / float(model.slow_tau_ms))
                if model.has_slow_state
                else 0.0
            )
            velocity[index] = -(
                voltage[index - 1] - model.resting_voltage_mv
            ) / model.rest_relaxation_ms
            voltage[index] = voltage[index - 1] + dt * velocity[index]
            phase_values[index] = phase
            memory_values[index] = memory
            continue
        if not active:
            active = True
            spline = _periodic_spline(model, applied)
            spline_current = applied
            phase_grid = np.asarray(model.phase_grid)
            template = model.voltage_template(applied)
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
                            template[candidates]
                            - voltage[index - 1]
                        )
                    )
                ]
            )
            phase = float(phase_grid[phase_index])
            current_period = model.period(applied, memory)
            downstroke, recovery, upstroke = model.segment_durations(
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
                    model.latency(applied) - upstroke
                )
                / remaining_recovery,
            )
        if model.has_slow_state:
            memory *= math.exp(-dt / float(model.slow_tau_ms))
        else:
            memory = 0.0
        if not first_cycle:
            current_period = model.period(applied, memory)
        downstroke, recovery, upstroke = model.segment_durations(
            applied,
            current_period,
        )
        if phase < 0.25:
            phase_rate = 0.25 / downstroke
        elif phase < 0.75:
            recovery_scale = (
                first_recovery_time_scale
                if first_cycle
                else 1.0
            )
            phase_rate = 0.5 / (recovery * recovery_scale)
        else:
            phase_rate = 0.25 / upstroke
        phase += dt * phase_rate
        if phase >= 1.0:
            wraps = int(math.floor(phase))
            phase -= wraps
            if model.has_slow_state:
                memory += float(wraps)
            first_cycle = False
            first_recovery_time_scale = 1.0
            current_period = model.period(applied, memory)
            downstroke, _, _ = model.segment_durations(
                applied,
                current_period,
            )
            phase_rate = 0.25 / downstroke
        if not np.isclose(applied, spline_current):
            spline = _periodic_spline(model, applied)
            spline_current = applied
        voltage[index] = float(spline(phase))
        velocity[index] = float(spline(phase, 1) * phase_rate)
        phase_values[index] = phase
        memory_values[index] = memory
        periods[index] = current_period
    return PhaseTemplateSimulation(
        time_ms=time,
        voltage_mv=voltage,
        velocity_mv_ms=velocity,
        phase=phase_values,
        memory=memory_values,
        instantaneous_period_ms=periods,
    )


def phase_plane_chamfer_distance(
    trace: AbstractObservedTrace,
    simulation: PhaseTemplateSimulation,
    maximum_points: int = 3000,
    stable_cycles_only: bool = False,
) -> float:
    """Symmetric nearest-neighbor distance in robustly scaled phase space."""
    def comparison_mask(voltage_mv: np.ndarray) -> np.ndarray:
        during = (
            (trace.time_ms >= trace.stimulus_start_ms)
            & (trace.time_ms < trace.stimulus_end_ms)
        )
        if not stable_cycles_only:
            return during
        crossings = np.flatnonzero(
            (voltage_mv[1:] >= 0.0)
            & (voltage_mv[:-1] < 0.0)
            & during[1:]
        ) + 1
        if len(crossings) < 3:
            return during
        return (
            during
            & (trace.time_ms >= trace.time_ms[crossings[1]])
            & (trace.time_ms <= trace.time_ms[crossings[-1]])
        )

    observed_mask = comparison_mask(trace.voltage_mv)
    predicted_mask = comparison_mask(simulation.voltage_mv)
    observed = np.column_stack(
        (
            trace.voltage_mv[observed_mask],
            trace.velocity_mv_ms[observed_mask],
        )
    )
    predicted = np.column_stack(
        (
            simulation.voltage_mv[predicted_mask],
            simulation.velocity_mv_ms[predicted_mask],
        )
    )
    if len(observed) > maximum_points:
        indices = np.linspace(
            0,
            len(observed) - 1,
            maximum_points,
        ).astype(int)
        observed = observed[indices]
    if len(predicted) > maximum_points:
        indices = np.linspace(
            0,
            len(predicted) - 1,
            maximum_points,
        ).astype(int)
        predicted = predicted[indices]
    scale = np.maximum(
        np.asarray(
            (
                np.ptp(observed[:, 0]),
                np.ptp(observed[:, 1]),
            )
        ),
        1.0,
    )
    observed = observed / scale
    predicted = predicted / scale
    observed_tree = cKDTree(observed)
    predicted_tree = cKDTree(predicted)
    return float(
        0.5
        * (
            np.mean(observed_tree.query(predicted)[0])
            + np.mean(predicted_tree.query(observed)[0])
        )
    )


def template_fourier_coefficients(
    model: PhaseTemplateModel,
    harmonics: int = 20,
) -> list[dict[str, float]]:
    """Return a compact parameter table for later population analyses."""
    rows = []
    for current, template in zip(
        model.current_levels,
        model.voltage_templates_mv,
    ):
        coefficients = np.fft.rfft(
            np.asarray(template, dtype=float)
        ) / len(template)
        for harmonic, value in enumerate(
            coefficients[: harmonics + 1]
        ):
            rows.append(
                {
                    "current_value": float(current),
                    "harmonic": float(harmonic),
                    "cosine_coefficient_mv": float(
                        value.real * (1.0 if harmonic == 0 else 2.0)
                    ),
                    "sine_coefficient_mv": float(
                        -2.0 * value.imag
                    ),
                }
            )
    return rows
