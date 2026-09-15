"""Electrophysiological and phase-plane feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .hh_model import Stimulus, Trace


@dataclass(frozen=True)
class FeatureConfig:
    min_peak_voltage_mv: float = 0.0
    min_prominence_mv: float = 20.0
    min_spike_distance_ms: float = 1.0
    threshold_dvdt_mv_ms: float = 10.0
    threshold_fraction_of_upstroke: float = 0.05
    onset_rapidness_target_mv_ms: float = 20.0
    onset_rapidness_fraction_of_upstroke: float = 0.10
    onset_rapidness_window_mv_ms: float = 10.0
    max_cycle_duration_ms: float = 12.0
    voltage_filter_window_ms: float = 0.15
    voltage_filter_polynomial_order: int = 3
    min_spikes: int = 2


SPIKE_CYCLE_WAVEFORM_FEATURES = (
    "first_threshold_voltage_mv",
    "first_peak_voltage_mv",
    "first_ap_amplitude_mv",
    "first_half_width_ms",
    "first_fast_trough_voltage_mv",
    "first_upstroke_mv_ms",
    "first_downstroke_mv_ms",
    "first_upstroke_downstroke_ratio",
)

SPIKE_CYCLE_TIMING_ONSET_FEATURES = (
    "first_ap_upstroke_time_ms",
    "first_ap_repolarization_time_ms",
    "first_ap_duration_ms",
    "first_onset_voltage_mv",
    "first_onset_dvdt_mv_ms",
    "first_onset_rapidness_per_ms",
)

SPIKE_CYCLE_CURVATURE_FEATURES = (
    "first_max_acceleration_mv_ms2",
    "first_max_acceleration_voltage_mv",
    "first_min_acceleration_mv_ms2",
    "first_min_acceleration_voltage_mv",
    "upstroke_inflection_voltage_mv",
    "upstroke_inflection_dvdt_mv_ms",
    "downstroke_inflection_voltage_mv",
    "downstroke_inflection_dvdt_mv_ms",
    "upstroke_inflection_relative_to_threshold_mv",
    "downstroke_inflection_relative_to_peak_mv",
)

SPIKE_CYCLE_GEOMETRY_FEATURES = (
    "first_ap_phase_area_v2_ms",
    "first_ap_phase_area_normalized",
    "first_spike_phase_area_v2_ms",
    "first_cycle_phase_area_normalized",
    "first_cycle_phase_path_length_normalized",
)

SPIKE_CYCLE_FEATURE_BLOCKS = {
    "waveform": SPIKE_CYCLE_WAVEFORM_FEATURES,
    "timing_onset": SPIKE_CYCLE_TIMING_ONSET_FEATURES,
    "curvature": SPIKE_CYCLE_CURVATURE_FEATURES,
    "geometry": SPIKE_CYCLE_GEOMETRY_FEATURES,
}

SPIKE_CYCLE_OBSERVABLE_FEATURES = tuple(
    feature_name
    for feature_block in SPIKE_CYCLE_FEATURE_BLOCKS.values()
    for feature_name in feature_block
)

SPIKE_CYCLE_MODEL_DIAGNOSTICS = (
    "upstroke_inflection_neg_iion_ua_cm2",
    "downstroke_inflection_neg_iion_ua_cm2",
)

SPIKE_CYCLE_OPTIONAL_FEATURES = (
    "first_onset_voltage_at_20_mv_ms",
    "first_onset_rapidness_at_20_mv_ms_per_ms",
)


@dataclass(frozen=True)
class SpikeDetection:
    peak_indices: np.ndarray
    threshold_indices: np.ndarray


@dataclass(frozen=True)
class VoltageFeatureTrace:
    """Voltage-derived trace containing only quantities observable in a sweep."""

    time_ms: np.ndarray
    voltage_mv: np.ndarray
    dvdt_mv_ms: np.ndarray
    ionic_current_ua_cm2: np.ndarray


FeatureTrace = Union[Trace, VoltageFeatureTrace]


def _savgol_window_samples(
    sample_interval_ms: float,
    filter_window_ms: float,
    polynomial_order: int,
    n_samples: int,
) -> int:
    if filter_window_ms <= 0.0:
        return 0
    window_samples = max(
        polynomial_order + 2,
        int(round(filter_window_ms / sample_interval_ms)),
    )
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples > n_samples:
        window_samples = n_samples if n_samples % 2 == 1 else n_samples - 1
    return window_samples if window_samples > polynomial_order else 0


def voltage_feature_trace(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    filter_window_ms: float = 0.15,
    polynomial_order: int = 3,
) -> VoltageFeatureTrace:
    """Create a consistently filtered, voltage-only trace for feature extraction."""
    time = np.asarray(time_ms, dtype=float)
    voltage = np.asarray(voltage_mv, dtype=float)
    if time.ndim != 1 or voltage.ndim != 1 or len(time) != len(voltage):
        raise ValueError("time_ms and voltage_mv must be one-dimensional and equal length")
    if len(time) < 5 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(voltage)):
        raise ValueError("Voltage trace must contain at least five finite samples")
    intervals = np.diff(time)
    if np.any(intervals <= 0.0):
        raise ValueError("time_ms must be strictly increasing")
    sample_interval = float(np.median(intervals))
    if np.max(np.abs(intervals - sample_interval)) > 0.01 * sample_interval:
        raise ValueError("time_ms must be regularly sampled within one percent")
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be positive")

    window_samples = _savgol_window_samples(
        sample_interval,
        filter_window_ms,
        polynomial_order,
        len(voltage),
    )
    filtered_voltage = (
        savgol_filter(
            voltage,
            window_length=window_samples,
            polyorder=polynomial_order,
            mode="interp",
        )
        if window_samples
        else voltage.copy()
    )
    dvdt = np.gradient(filtered_voltage, time)
    return VoltageFeatureTrace(
        time_ms=time,
        voltage_mv=filtered_voltage,
        dvdt_mv_ms=dvdt,
        ionic_current_ua_cm2=np.full_like(time, np.nan),
    )


def extract_voltage_features(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    stimulus: Stimulus,
    config: FeatureConfig | None = None,
    filter_window_ms: float | None = None,
    polynomial_order: int | None = None,
) -> dict[str, float]:
    """Extract the same feature set from an experimental voltage sweep."""
    config = config or FeatureConfig()
    trace = voltage_feature_trace(
        time_ms,
        voltage_mv,
        filter_window_ms=(
            config.voltage_filter_window_ms
            if filter_window_ms is None
            else filter_window_ms
        ),
        polynomial_order=(
            config.voltage_filter_polynomial_order
            if polynomial_order is None
            else polynomial_order
        ),
    )
    return extract_features(trace, stimulus, config)


def extract_observed_features(
    trace: Trace,
    stimulus: Stimulus,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract measurement-matched features plus model-only current diagnostics."""
    config = config or FeatureConfig()
    exact_features = extract_features(trace, stimulus, config)
    observed_features = extract_voltage_features(
        trace.time_ms,
        trace.voltage_mv,
        stimulus,
        config,
    )
    for feature_name in SPIKE_CYCLE_MODEL_DIAGNOSTICS:
        observed_features[feature_name] = exact_features.get(
            feature_name,
            float("nan"),
        )
    return observed_features


def detect_spikes(
    trace: FeatureTrace,
    stimulus: Stimulus,
    config: FeatureConfig,
) -> SpikeDetection:
    dt = float(np.median(np.diff(trace.time_ms)))
    in_stimulus = (trace.time_ms >= stimulus.start_ms) & (trace.time_ms < stimulus.end_ms)
    candidate_indices = np.flatnonzero(in_stimulus)
    if not len(candidate_indices):
        return SpikeDetection(np.asarray([], dtype=int), np.asarray([], dtype=int))

    start, stop = candidate_indices[0], candidate_indices[-1] + 1
    local_peaks, _ = find_peaks(
        trace.voltage_mv[start:stop],
        height=config.min_peak_voltage_mv,
        prominence=config.min_prominence_mv,
        distance=max(1, int(round(config.min_spike_distance_ms / dt))),
    )
    peak_indices = local_peaks + start

    threshold_indices = []
    lookback = max(2, int(round(5.0 / dt)))
    for peak_index in peak_indices:
        left = max(start, peak_index - lookback)
        segment = trace.dvdt_mv_ms[left : peak_index + 1]
        threshold_target = (
            config.threshold_fraction_of_upstroke * float(np.max(segment))
            if config.threshold_fraction_of_upstroke > 0.0
            else config.threshold_dvdt_mv_ms
        )
        crossings = np.flatnonzero(
            (segment[:-1] < threshold_target)
            & (segment[1:] >= threshold_target)
        )
        threshold_indices.append(
            left + int(crossings[-1] + 1) if len(crossings) else left
        )

    return SpikeDetection(peak_indices, np.asarray(threshold_indices, dtype=int))


def _interpolate_crossing(
    trace: FeatureTrace, derivative: np.ndarray, left_index: int
) -> dict[str, float]:
    right_index = left_index + 1
    left_value, right_value = derivative[left_index : right_index + 1]
    fraction = (
        0.0
        if right_value == left_value
        else float(np.clip(-left_value / (right_value - left_value), 0.0, 1.0))
    )

    def interpolate(values: np.ndarray) -> float:
        return float(values[left_index] + fraction * (values[right_index] - values[left_index]))

    return {
        "time_ms": interpolate(trace.time_ms),
        "voltage_mv": interpolate(trace.voltage_mv),
        "dvdt_mv_ms": interpolate(trace.dvdt_mv_ms),
        "neg_iion_ua_cm2": -interpolate(trace.ionic_current_ua_cm2),
    }


def _inflection_near_extremum(
    trace: FeatureTrace,
    second_derivative: np.ndarray,
    start_index: int,
    stop_index: int,
    mode: str,
) -> dict[str, float]:
    if stop_index - start_index < 3:
        return {}

    segment = trace.dvdt_mv_ms[start_index : stop_index + 1]
    extremum = start_index + int(np.argmax(segment) if mode == "max" else np.argmin(segment))
    crossing_indices = np.flatnonzero(
        second_derivative[start_index:stop_index]
        * second_derivative[start_index + 1 : stop_index + 1]
        <= 0.0
    ) + start_index
    if mode == "max":
        preferred = [
            index
            for index in crossing_indices
            if second_derivative[index] >= 0.0
            and second_derivative[index + 1] <= 0.0
        ]
    else:
        preferred = [
            index
            for index in crossing_indices
            if second_derivative[index] <= 0.0
            and second_derivative[index + 1] >= 0.0
        ]
    candidates = preferred or crossing_indices.tolist()
    if not candidates:
        return {}
    crossing = min(candidates, key=lambda index: abs(index - extremum))
    return _interpolate_crossing(trace, second_derivative, crossing)


def _crossing_time(
    time_ms: np.ndarray,
    values: np.ndarray,
    level: float,
    start: int,
    stop: int,
    rising: bool,
) -> float:
    segment = values[start : stop + 1]
    if rising:
        local = np.flatnonzero((segment[:-1] < level) & (segment[1:] >= level))
    else:
        local = np.flatnonzero((segment[:-1] >= level) & (segment[1:] < level))
    if not len(local):
        return float("nan")
    left = start + int(local[-1] if rising else local[0])
    denominator = values[left + 1] - values[left]
    fraction = 0.0 if denominator == 0.0 else (level - values[left]) / denominator
    return float(time_ms[left] + fraction * (time_ms[left + 1] - time_ms[left]))


def _phase_area(voltage: np.ndarray, dvdt: np.ndarray) -> float:
    if len(voltage) < 3:
        return float("nan")
    return float(
        0.5
        * abs(
            np.dot(voltage, np.roll(dvdt, 1))
            - np.dot(dvdt, np.roll(voltage, 1))
        )
    )


def _normalized_phase_path_length(
    voltage: np.ndarray,
    dvdt: np.ndarray,
    voltage_scale: float,
    dvdt_scale: float,
) -> float:
    if (
        len(voltage) < 2
        or voltage_scale <= 0.0
        or dvdt_scale <= 0.0
    ):
        return float("nan")
    normalized_voltage = (voltage - voltage[0]) / voltage_scale
    normalized_dvdt = dvdt / dvdt_scale
    return float(
        np.sum(
            np.hypot(
                np.diff(normalized_voltage),
                np.diff(normalized_dvdt),
            )
        )
    )


def _downward_voltage_crossing(
    trace: FeatureTrace,
    level_mv: float,
    start_index: int,
    stop_index: int,
) -> tuple[int, float, float] | None:
    segment = trace.voltage_mv[start_index : stop_index + 1]
    crossings = np.flatnonzero(
        (segment[:-1] >= level_mv) & (segment[1:] < level_mv)
    )
    if not len(crossings):
        return None
    left_index = start_index + int(crossings[0])
    denominator = trace.voltage_mv[left_index + 1] - trace.voltage_mv[left_index]
    fraction = (
        0.0
        if denominator == 0.0
        else float(
            np.clip(
                (level_mv - trace.voltage_mv[left_index]) / denominator,
                0.0,
                1.0,
            )
        )
    )
    crossing_time = float(
        trace.time_ms[left_index]
        + fraction * (trace.time_ms[left_index + 1] - trace.time_ms[left_index])
    )
    crossing_dvdt = float(
        trace.dvdt_mv_ms[left_index]
        + fraction * (trace.dvdt_mv_ms[left_index + 1] - trace.dvdt_mv_ms[left_index])
    )
    return left_index + 1, crossing_time, crossing_dvdt


def _onset_rapidness(
    trace: FeatureTrace,
    start_index: int,
    stop_index: int,
    target_mv_ms: float,
    window_mv_ms: float,
) -> tuple[float, float]:
    if stop_index - start_index < 3 or target_mv_ms <= 0.0 or window_mv_ms <= 0.0:
        return float("nan"), float("nan")
    dvdt = trace.dvdt_mv_ms[start_index : stop_index + 1]
    crossings = np.flatnonzero(
        (dvdt[:-1] < target_mv_ms) & (dvdt[1:] >= target_mv_ms)
    )
    if not len(crossings):
        return float("nan"), float("nan")
    left_index = start_index + int(crossings[0])
    denominator = trace.dvdt_mv_ms[left_index + 1] - trace.dvdt_mv_ms[left_index]
    fraction = (
        0.0
        if denominator == 0.0
        else float(
            np.clip(
                (target_mv_ms - trace.dvdt_mv_ms[left_index]) / denominator,
                0.0,
                1.0,
            )
        )
    )
    target_voltage = float(
        trace.voltage_mv[left_index]
        + fraction * (trace.voltage_mv[left_index + 1] - trace.voltage_mv[left_index])
    )

    half_window = min(window_mv_ms / 2.0, target_mv_ms / 2.0)
    lower_target = target_mv_ms - half_window
    upper_target = target_mv_ms + half_window
    local_start = left_index
    while (
        local_start > start_index
        and trace.dvdt_mv_ms[local_start - 1] >= lower_target
    ):
        local_start -= 1
    local_stop = left_index + 1
    while (
        local_stop < stop_index
        and trace.dvdt_mv_ms[local_stop + 1] <= upper_target
    ):
        local_stop += 1
    if local_stop - local_start + 1 < 3:
        local_start = max(start_index, left_index - 1)
        local_stop = min(stop_index, left_index + 2)
    fit_indices = np.arange(local_start, local_stop + 1)
    voltage = trace.voltage_mv[fit_indices]
    local_dvdt = trace.dvdt_mv_ms[fit_indices]
    if len(np.unique(voltage)) < 2:
        return target_voltage, float("nan")
    rapidness = float(np.polyfit(voltage, local_dvdt, 1)[0])
    if rapidness <= 0.0:
        rapidness = float("nan")
    return target_voltage, rapidness


def extract_features(
    trace: FeatureTrace,
    stimulus: Stimulus,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract fixed-width e-features, including phase-plane inflection coordinates."""
    config = config or FeatureConfig()
    detection = detect_spikes(trace, stimulus, config)
    peaks = detection.peak_indices
    thresholds = detection.threshold_indices
    features: dict[str, float] = {
        "spike_count": float(len(peaks)),
        "is_spiking": float(len(peaks) >= config.min_spikes),
    }

    baseline_mask = trace.time_ms < stimulus.start_ms
    stimulus_mask = (trace.time_ms >= stimulus.start_ms) & (trace.time_ms < stimulus.end_ms)
    features["baseline_voltage_mv"] = float(np.mean(trace.voltage_mv[baseline_mask]))
    features["stimulus_voltage_mean_mv"] = float(np.mean(trace.voltage_mv[stimulus_mask]))
    features["stimulus_voltage_std_mv"] = float(np.std(trace.voltage_mv[stimulus_mask]))

    duration_s = max((stimulus.end_ms - stimulus.start_ms) / 1000.0, np.finfo(float).eps)
    features["firing_rate_hz"] = float(len(peaks) / duration_s)
    if not len(peaks):
        return features

    peak_times = trace.time_ms[peaks]
    features["first_spike_latency_ms"] = float(
        trace.time_ms[thresholds[0]] - stimulus.start_ms
    )
    features["terminal_silence_ms"] = float(
        stimulus.end_ms - peak_times[-1]
    )
    midpoint = stimulus.start_ms + 0.5 * (
        stimulus.end_ms - stimulus.start_ms
    )
    features["late_spike_fraction"] = float(
        np.count_nonzero(peak_times >= midpoint) / len(peak_times)
    )
    features["mean_peak_voltage_mv"] = float(np.mean(trace.voltage_mv[peaks]))
    if len(peaks) > 1:
        isi = np.diff(peak_times)
        features["mean_isi_ms"] = float(np.mean(isi))
        features["isi_cv"] = float(np.std(isi) / np.mean(isi))
        features["adaptation_index"] = float(isi[-1] / isi[0])

    dt = float(np.median(np.diff(trace.time_ms)))
    second_derivative = np.gradient(trace.dvdt_mv_ms, dt)
    threshold_voltages = []
    amplitudes = []
    half_widths = []
    ahps = []
    max_dvdt = []
    min_dvdt = []
    first_spike_features: dict[str, float] = {}

    for spike_number, (peak, threshold) in enumerate(zip(peaks, thresholds)):
        next_boundary = (
            thresholds[spike_number + 1]
            if spike_number + 1 < len(thresholds)
            else min(
                len(trace.time_ms) - 1,
                peak + int(round(config.max_cycle_duration_ms / dt)),
            )
        )
        threshold_voltage = trace.voltage_mv[threshold]
        peak_voltage = trace.voltage_mv[peak]
        half_level = threshold_voltage + (peak_voltage - threshold_voltage) / 2.0
        rise_time = _crossing_time(
            trace.time_ms, trace.voltage_mv, half_level, threshold, peak, rising=True
        )
        fall_time = _crossing_time(
            trace.time_ms, trace.voltage_mv, half_level, peak, next_boundary, rising=False
        )

        threshold_voltages.append(threshold_voltage)
        amplitudes.append(peak_voltage - threshold_voltage)
        if np.isfinite(rise_time) and np.isfinite(fall_time):
            half_widths.append(fall_time - rise_time)
        ahps.append(float(np.min(trace.voltage_mv[peak : next_boundary + 1])))
        max_dvdt.append(float(np.max(trace.dvdt_mv_ms[threshold : peak + 1])))
        min_dvdt.append(float(np.min(trace.dvdt_mv_ms[peak : next_boundary + 1])))
        if spike_number == 0:
            first_spike_features = {
                "first_threshold_voltage_mv": float(threshold_voltage),
                "first_peak_voltage_mv": float(peak_voltage),
                "first_ap_amplitude_mv": float(peak_voltage - threshold_voltage),
                "first_half_width_ms": (
                    float(fall_time - rise_time)
                    if np.isfinite(rise_time) and np.isfinite(fall_time)
                    else float("nan")
                ),
                "first_fast_trough_voltage_mv": float(
                    np.min(trace.voltage_mv[peak : next_boundary + 1])
                ),
                "first_upstroke_mv_ms": float(
                    np.max(trace.dvdt_mv_ms[threshold : peak + 1])
                ),
                "first_downstroke_mv_ms": float(
                    np.min(trace.dvdt_mv_ms[peak : next_boundary + 1])
                ),
            }
            first_spike_features["first_upstroke_downstroke_ratio"] = (
                first_spike_features["first_upstroke_mv_ms"]
                / abs(first_spike_features["first_downstroke_mv_ms"])
            )

    features.update(
        {
            "mean_threshold_voltage_mv": float(np.mean(threshold_voltages)),
            "mean_ap_amplitude_mv": float(np.mean(amplitudes)),
            "mean_half_width_ms": float(np.mean(half_widths)) if half_widths else float("nan"),
            "mean_ahp_voltage_mv": float(np.mean(ahps)),
            "mean_max_dvdt_mv_ms": float(np.mean(max_dvdt)),
            "mean_min_dvdt_mv_ms": float(np.mean(min_dvdt)),
        }
    )
    features.update(first_spike_features)

    first_peak = int(peaks[0])
    first_threshold = int(thresholds[0])
    cycle_stop = (
        int(thresholds[1])
        if len(thresholds) > 1
        else min(
            len(trace.time_ms) - 1,
            first_peak + int(round(config.max_cycle_duration_ms / dt)),
        )
    )
    upstroke = _inflection_near_extremum(
        trace, second_derivative, first_threshold, first_peak, "max"
    )
    downstroke = _inflection_near_extremum(
        trace, second_derivative, first_peak, cycle_stop, "min"
    )
    for prefix, point in (("upstroke_inflection", upstroke), ("downstroke_inflection", downstroke)):
        for name, value in point.items():
            features[f"{prefix}_{name}"] = value

    cycle_voltage = trace.voltage_mv[first_threshold : cycle_stop + 1]
    cycle_dvdt = trace.dvdt_mv_ms[first_threshold : cycle_stop + 1]
    cycle_area = _phase_area(cycle_voltage, cycle_dvdt)
    features["first_spike_phase_area_v2_ms"] = cycle_area

    threshold_voltage = float(trace.voltage_mv[first_threshold])
    peak_voltage = float(trace.voltage_mv[first_peak])
    ap_amplitude = peak_voltage - threshold_voltage
    upstroke_index = first_threshold + int(
        np.argmax(trace.dvdt_mv_ms[first_threshold : first_peak + 1])
    )
    downstroke_index = first_peak + int(
        np.argmin(trace.dvdt_mv_ms[first_peak : cycle_stop + 1])
    )
    dvdt_span = float(
        trace.dvdt_mv_ms[upstroke_index] - trace.dvdt_mv_ms[downstroke_index]
    )

    features["first_ap_upstroke_time_ms"] = float(
        trace.time_ms[first_peak] - trace.time_ms[first_threshold]
    )
    threshold_return = _downward_voltage_crossing(
        trace,
        threshold_voltage,
        first_peak,
        cycle_stop,
    )
    if threshold_return is not None:
        ap_stop, return_time, return_dvdt = threshold_return
        ap_voltage = trace.voltage_mv[first_threshold : ap_stop + 1].copy()
        ap_dvdt = trace.dvdt_mv_ms[first_threshold : ap_stop + 1].copy()
        ap_voltage[-1] = threshold_voltage
        ap_dvdt[-1] = return_dvdt
        ap_area = _phase_area(ap_voltage, ap_dvdt)
        features["first_ap_duration_ms"] = float(
            return_time - trace.time_ms[first_threshold]
        )
        features["first_ap_repolarization_time_ms"] = float(
            return_time - trace.time_ms[first_peak]
        )
        features["first_ap_phase_area_v2_ms"] = ap_area
        features["first_ap_phase_area_normalized"] = (
            float(ap_area / (ap_amplitude * dvdt_span))
            if ap_amplitude > 0.0 and dvdt_span > 0.0
            else float("nan")
        )
    else:
        for name in (
            "first_ap_duration_ms",
            "first_ap_repolarization_time_ms",
            "first_ap_phase_area_v2_ms",
            "first_ap_phase_area_normalized",
        ):
            features[name] = float("nan")

    features["first_cycle_phase_area_normalized"] = (
        float(cycle_area / (ap_amplitude * dvdt_span))
        if ap_amplitude > 0.0 and dvdt_span > 0.0
        else float("nan")
    )
    features["first_cycle_phase_path_length_normalized"] = (
        _normalized_phase_path_length(
            cycle_voltage,
            cycle_dvdt,
            ap_amplitude,
            dvdt_span,
        )
    )

    adaptive_onset_target = float(
        config.onset_rapidness_fraction_of_upstroke
        * trace.dvdt_mv_ms[upstroke_index]
    )
    onset_start = max(0, first_threshold - int(round(0.5 / dt)))
    onset_voltage, onset_rapidness = _onset_rapidness(
        trace,
        onset_start,
        upstroke_index,
        adaptive_onset_target,
        config.onset_rapidness_window_mv_ms,
    )
    features["first_onset_voltage_mv"] = onset_voltage
    features["first_onset_dvdt_mv_ms"] = adaptive_onset_target
    features["first_onset_rapidness_per_ms"] = onset_rapidness
    fixed_onset_voltage, fixed_onset_rapidness = _onset_rapidness(
        trace,
        onset_start,
        upstroke_index,
        config.onset_rapidness_target_mv_ms,
        config.onset_rapidness_window_mv_ms,
    )
    features["first_onset_voltage_at_20_mv_ms"] = fixed_onset_voltage
    features["first_onset_rapidness_at_20_mv_ms_per_ms"] = (
        fixed_onset_rapidness
    )

    acceleration_upstroke = second_derivative[first_threshold : upstroke_index + 1]
    acceleration_downstroke = second_derivative[upstroke_index : downstroke_index + 1]
    if len(acceleration_upstroke):
        max_acceleration_index = first_threshold + int(np.argmax(acceleration_upstroke))
        features["first_max_acceleration_mv_ms2"] = float(
            second_derivative[max_acceleration_index]
        )
        features["first_max_acceleration_voltage_mv"] = float(
            trace.voltage_mv[max_acceleration_index]
        )
    if len(acceleration_downstroke):
        min_acceleration_index = upstroke_index + int(np.argmin(acceleration_downstroke))
        features["first_min_acceleration_mv_ms2"] = float(
            second_derivative[min_acceleration_index]
        )
        features["first_min_acceleration_voltage_mv"] = float(
            trace.voltage_mv[min_acceleration_index]
        )

    if upstroke and "voltage_mv" in upstroke:
        features["upstroke_inflection_relative_to_threshold_mv"] = float(
            upstroke["voltage_mv"] - threshold_voltage
        )
    if downstroke and "voltage_mv" in downstroke:
        features["downstroke_inflection_relative_to_peak_mv"] = float(
            downstroke["voltage_mv"] - peak_voltage
        )
    return features


DEFAULT_EFEL_FEATURES = (
    "AP_amplitude",
    "AP_width",
    "AHP_depth",
    "Spikecount",
    "mean_frequency",
    "time_to_first_spike",
)


def extract_efel_features(
    trace: Trace,
    stimulus: Stimulus,
    feature_names: Iterable[str] = DEFAULT_EFEL_FEATURES,
) -> dict[str, float]:
    """Extract optional eFEL features, reducing vector features to their mean."""
    try:
        import efel
    except ImportError as exc:
        raise RuntimeError(
            "eFEL is not installed. Install the optional dependency with "
            "`python -m pip install -e '.[efel]'`."
        ) from exc

    efel_trace = {
        "T": trace.time_ms,
        "V": trace.voltage_mv,
        "stim_start": [stimulus.start_ms],
        "stim_end": [stimulus.end_ms],
    }
    extractor = getattr(efel, "get_feature_values", None) or getattr(
        efel, "getFeatureValues"
    )
    result = extractor([efel_trace], list(feature_names))[0]
    flattened = {}
    for name, values in result.items():
        if values is None:
            flattened[f"efel_{name}"] = float("nan")
            continue
        array = np.asarray(values, dtype=float)
        flattened[f"efel_{name}"] = float(np.mean(array)) if array.size else float("nan")
    return flattened
