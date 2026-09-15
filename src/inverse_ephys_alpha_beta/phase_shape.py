"""Smoothed, branch-wise comparison of action-potential phase planes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from .features import FeatureConfig, detect_spikes, voltage_feature_trace
from .hh_model import Stimulus


@dataclass(frozen=True)
class PhaseShapeConfig:
    grid_points: int = 64
    voltage_filter_window_ms: float = 0.15
    voltage_filter_polynomial_order: int = 3
    max_cycle_duration_ms: float = 25.0
    spline_smoothing: float = 0.002
    curvature_floor_fraction: float = 0.08
    onset_curvature_weight: float = 2.5
    late_upstroke_curvature_weight: float = 1.0
    high_voltage_downstroke_curvature_weight: float = 1.0
    low_voltage_downstroke_curvature_weight: float = 2.5
    downstroke_high_voltage_end: float = 0.45
    downstroke_low_voltage_start: float = 0.65
    physical_curve_scale_fraction: float = 0.12
    physical_curve_scale_floor_mv_ms: float = 20.0
    normalized_curve_scale: float = 0.08
    normalized_slope_scale: float = 0.50
    max_dvdt_scale_mv_ms: float = 35.0
    min_dvdt_scale_mv_ms: float = 25.0
    dvdt_span_scale_mv_ms: float = 45.0
    amplitude_scale_mv: float = 8.0
    extremum_voltage_scale_mv: float = 5.0
    velocity_constraint_fraction: float = 0.10
    max_dvdt_constraint_floor_mv_ms: float = 20.0
    min_dvdt_constraint_floor_mv_ms: float = 15.0
    voltage_extremum_tolerance_mv: float = 4.0


@dataclass(frozen=True)
class PhaseCycle:
    """First-spike phase cycle on fixed upstroke and repolarization grids."""

    grid: np.ndarray
    up_dvdt_mv_ms: np.ndarray
    down_dvdt_mv_ms: np.ndarray
    up_normalized: np.ndarray
    down_normalized: np.ndarray
    up_slope: np.ndarray
    down_slope: np.ndarray
    up_concavity: np.ndarray
    down_concavity: np.ndarray
    threshold_voltage_mv: float
    peak_voltage_mv: float
    down_end_voltage_mv: float
    amplitude_mv: float
    repolarization_amplitude_mv: float
    max_dvdt_mv_ms: float
    min_dvdt_mv_ms: float
    dvdt_span_mv_ms: float
    up_max_relative_voltage: float
    down_min_relative_voltage: float
    threshold_return_voltage_mv: float = float("nan")


@dataclass(frozen=True)
class PhaseShapeScores:
    physical_curve: float
    normalized_shape: float
    slope_shape: float
    concavity: float
    velocity_extent: float
    landmarks: float
    total: float
    onset_concavity_sign_agreement: float
    max_dvdt_residual_mv_ms: float
    min_dvdt_residual_mv_ms: float
    dvdt_span_residual_mv_ms: float
    upstroke_onset_concavity: float
    upstroke_late_concavity: float
    downstroke_high_voltage_concavity: float
    downstroke_low_voltage_concavity: float
    upstroke_late_sign_agreement: float
    downstroke_high_voltage_sign_agreement: float
    downstroke_low_voltage_sign_agreement: float
    peak_voltage_residual_mv: float
    minimum_voltage_residual_mv: float
    physical_constraint_loss: float

    def to_mapping(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


def _huber_array(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    absolute = np.abs(residuals)
    return np.where(
        absolute <= 1.0,
        0.5 * residuals**2,
        absolute - 0.5,
    )


def _strict_branch(
    coordinate: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort one branch and average samples with duplicate coordinates."""
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(values)
    coordinate = np.clip(coordinate[finite], 0.0, 1.0)
    values = values[finite]
    if len(coordinate) < 4:
        raise ValueError("Phase branch contains fewer than four finite samples")
    order = np.argsort(coordinate, kind="stable")
    coordinate = coordinate[order]
    values = values[order]
    unique, inverse = np.unique(coordinate, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    averaged = sums / counts
    if len(unique) < 4 or unique[-1] - unique[0] < 0.8:
        raise ValueError("Phase branch does not span the action potential")
    if unique[0] > 0.0:
        unique = np.insert(unique, 0, 0.0)
        averaged = np.insert(averaged, 0, averaged[0])
    if unique[-1] < 1.0:
        unique = np.append(unique, 1.0)
        averaged = np.append(averaged, averaged[-1])
    return unique, averaged


def _smooth_branch(
    coordinate: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    span: float,
    config: PhaseShapeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate, values = _strict_branch(coordinate, values)
    interpolated = PchipInterpolator(coordinate, values)(grid)
    normalized = interpolated / span
    spline = UnivariateSpline(
        grid,
        normalized,
        k=3,
        s=config.spline_smoothing * len(grid),
    )
    smoothed_normalized = np.asarray(spline(grid), dtype=float)
    return (
        smoothed_normalized * span,
        smoothed_normalized,
        np.asarray(spline.derivative(1)(grid), dtype=float),
        np.asarray(spline.derivative(2)(grid), dtype=float),
    )


def _downward_threshold_index(
    voltage_mv: np.ndarray,
    peak_index: int,
    stop_index: int,
    threshold_voltage_mv: float,
) -> tuple[int, bool]:
    segment = voltage_mv[peak_index : stop_index + 1]
    crossing = np.flatnonzero(segment <= threshold_voltage_mv)
    if len(crossing):
        return peak_index + int(crossing[0]), True
    if len(segment) < 4:
        raise ValueError("First spike has no measurable repolarization branch")
    trough_offset = int(np.argmin(segment[1:])) + 1
    return peak_index + trough_offset, False


def extract_phase_cycle(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    stimulus: Stimulus,
    config: PhaseShapeConfig | None = None,
) -> PhaseCycle:
    """Extract a smoothed first-spike phase cycle from a voltage trace."""
    config = config or PhaseShapeConfig()
    if config.grid_points < 16:
        raise ValueError("Phase grid requires at least 16 points")
    trace = voltage_feature_trace(
        time_ms,
        voltage_mv,
        filter_window_ms=config.voltage_filter_window_ms,
        polynomial_order=config.voltage_filter_polynomial_order,
    )
    feature_config = FeatureConfig(
        min_spikes=1,
        max_cycle_duration_ms=config.max_cycle_duration_ms,
        voltage_filter_window_ms=config.voltage_filter_window_ms,
        voltage_filter_polynomial_order=config.voltage_filter_polynomial_order,
    )
    detection = detect_spikes(trace, stimulus, feature_config)
    if not len(detection.peak_indices):
        raise ValueError("Trace contains no detected action potential")
    threshold_index = int(detection.threshold_indices[0])
    peak_index = int(detection.peak_indices[0])
    dt_ms = float(np.median(np.diff(trace.time_ms)))
    stop_index = (
        int(detection.threshold_indices[1])
        if len(detection.threshold_indices) > 1
        else min(
            len(trace.time_ms) - 1,
            peak_index
            + int(round(feature_config.max_cycle_duration_ms / dt_ms)),
        )
    )
    threshold_voltage = float(trace.voltage_mv[threshold_index])
    peak_voltage = float(trace.voltage_mv[peak_index])
    amplitude = peak_voltage - threshold_voltage
    if not np.isfinite(amplitude) or amplitude <= 10.0:
        raise ValueError("First spike has an invalid voltage amplitude")
    threshold_return_index, crossed_threshold = _downward_threshold_index(
        trace.voltage_mv,
        peak_index,
        stop_index,
        threshold_voltage,
    )
    trough_segment = trace.voltage_mv[peak_index : stop_index + 1]
    if len(trough_segment) < 4:
        raise ValueError("First spike has no measurable post-spike trough")
    trough_offset = int(np.argmin(trough_segment[1:])) + 1
    trough_index = peak_index + trough_offset

    up_voltage = trace.voltage_mv[threshold_index : peak_index + 1]
    up_dvdt = trace.dvdt_mv_ms[threshold_index : peak_index + 1]
    down_voltage = trace.voltage_mv[peak_index : trough_index + 1]
    down_dvdt = trace.dvdt_mv_ms[peak_index : trough_index + 1]
    down_end_voltage = float(trace.voltage_mv[trough_index])
    threshold_return_voltage = (
        float(trace.voltage_mv[threshold_return_index])
        if crossed_threshold
        else float("nan")
    )
    repolarization_amplitude = peak_voltage - down_end_voltage
    if repolarization_amplitude <= 10.0:
        raise ValueError("First spike has an invalid repolarization amplitude")
    maximum = float(np.max(up_dvdt))
    minimum = float(np.min(down_dvdt))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 20.0:
        raise ValueError("First spike has an invalid dV/dt span")

    up_coordinate = (up_voltage - threshold_voltage) / amplitude
    down_coordinate = (
        peak_voltage - down_voltage
    ) / repolarization_amplitude
    grid = np.linspace(0.0, 1.0, config.grid_points)
    up = _smooth_branch(
        up_coordinate,
        up_dvdt,
        grid,
        span,
        config,
    )
    down = _smooth_branch(
        down_coordinate,
        down_dvdt,
        grid,
        span,
        config,
    )
    up_max_index = int(np.argmax(up[0]))
    down_min_index = int(np.argmin(down[0]))
    return PhaseCycle(
        grid=grid,
        up_dvdt_mv_ms=up[0],
        down_dvdt_mv_ms=down[0],
        up_normalized=up[1],
        down_normalized=down[1],
        up_slope=up[2],
        down_slope=down[2],
        up_concavity=up[3],
        down_concavity=down[3],
        threshold_voltage_mv=threshold_voltage,
        peak_voltage_mv=peak_voltage,
        down_end_voltage_mv=down_end_voltage,
        amplitude_mv=float(amplitude),
        repolarization_amplitude_mv=float(repolarization_amplitude),
        max_dvdt_mv_ms=maximum,
        min_dvdt_mv_ms=minimum,
        dvdt_span_mv_ms=float(span),
        up_max_relative_voltage=float(grid[up_max_index]),
        down_min_relative_voltage=float(grid[down_min_index]),
        threshold_return_voltage_mv=threshold_return_voltage,
    )


def _curve_loss(
    model: np.ndarray,
    biological: np.ndarray,
    scale: float,
) -> float:
    return float(np.mean(_huber_array((model - biological) / scale)))


def _regional_concavity_loss(
    model_curvature: np.ndarray,
    biological_curvature: np.ndarray,
    region: np.ndarray,
    config: PhaseShapeConfig,
) -> tuple[float, float]:
    floor = config.curvature_floor_fraction * max(
        1.0,
        float(np.max(np.abs(biological_curvature[region]))),
    )
    informative = region & (np.abs(biological_curvature) >= floor)
    if not np.any(informative):
        informative = region
    scale = max(
        1.0,
        float(np.median(np.abs(biological_curvature[informative]))),
    )
    magnitude = _curve_loss(
        model_curvature[informative],
        biological_curvature[informative],
        scale,
    )
    agreement = float(
        np.mean(
            np.sign(model_curvature[informative])
            == np.sign(biological_curvature[informative])
        )
    )
    return float(magnitude + 1.5 * (1.0 - agreement)), agreement


def _concavity_loss(
    model: PhaseCycle,
    biological: PhaseCycle,
    config: PhaseShapeConfig,
) -> tuple[float, dict[str, float], dict[str, float]]:
    onset = biological.grid <= biological.up_max_relative_voltage
    if np.all(onset):
        onset = biological.grid <= 0.65
    late_upstroke = ~onset
    downstroke_high = (
        biological.grid <= config.downstroke_high_voltage_end
    )
    downstroke_low = (
        biological.grid >= config.downstroke_low_voltage_start
    )
    regions = {
        "upstroke_onset": _regional_concavity_loss(
            model.up_concavity,
            biological.up_concavity,
            onset,
            config,
        ),
        "upstroke_late": _regional_concavity_loss(
            model.up_concavity,
            biological.up_concavity,
            late_upstroke,
            config,
        ),
        "downstroke_high_voltage": _regional_concavity_loss(
            model.down_concavity,
            biological.down_concavity,
            downstroke_high,
            config,
        ),
        "downstroke_low_voltage": _regional_concavity_loss(
            model.down_concavity,
            biological.down_concavity,
            downstroke_low,
            config,
        ),
    }
    weights = {
        "upstroke_onset": config.onset_curvature_weight,
        "upstroke_late": config.late_upstroke_curvature_weight,
        "downstroke_high_voltage": (
            config.high_voltage_downstroke_curvature_weight
        ),
        "downstroke_low_voltage": (
            config.low_voltage_downstroke_curvature_weight
        ),
    }
    losses = {name: result[0] for name, result in regions.items()}
    agreements = {name: result[1] for name, result in regions.items()}
    loss = sum(
        weights[name] * losses[name]
        for name in regions
    ) / sum(weights.values())
    return float(loss), losses, agreements


def _constraint_excess(
    residual: float,
    tolerance: float,
) -> float:
    return max(0.0, abs(residual) - tolerance) / max(tolerance, 1e-12)


def _physical_constraint_loss(
    model: PhaseCycle,
    biological: PhaseCycle,
    config: PhaseShapeConfig,
) -> tuple[float, float, float]:
    peak_residual = model.peak_voltage_mv - biological.peak_voltage_mv
    minimum_residual = (
        model.down_end_voltage_mv - biological.down_end_voltage_mv
    )
    max_tolerance = max(
        config.max_dvdt_constraint_floor_mv_ms,
        config.velocity_constraint_fraction
        * abs(biological.max_dvdt_mv_ms),
    )
    min_tolerance = max(
        config.min_dvdt_constraint_floor_mv_ms,
        config.velocity_constraint_fraction
        * abs(biological.min_dvdt_mv_ms),
    )
    excess = np.asarray(
        (
            _constraint_excess(
                model.max_dvdt_mv_ms - biological.max_dvdt_mv_ms,
                max_tolerance,
            ),
            _constraint_excess(
                model.min_dvdt_mv_ms - biological.min_dvdt_mv_ms,
                min_tolerance,
            ),
            _constraint_excess(
                peak_residual,
                config.voltage_extremum_tolerance_mv,
            ),
            _constraint_excess(
                minimum_residual,
                config.voltage_extremum_tolerance_mv,
            ),
        ),
        dtype=float,
    )
    return (
        float(np.mean(excess**2)),
        float(peak_residual),
        float(minimum_residual),
    )


def compare_phase_cycles(
    model: PhaseCycle,
    biological: PhaseCycle,
    config: PhaseShapeConfig | None = None,
) -> PhaseShapeScores:
    """Compare two phase cycles while separating shape and physical scale."""
    config = config or PhaseShapeConfig()
    physical_scale = max(
        config.physical_curve_scale_floor_mv_ms,
        config.physical_curve_scale_fraction
        * biological.dvdt_span_mv_ms,
    )
    physical = float(
        np.mean(
            (
                _curve_loss(
                    model.up_dvdt_mv_ms,
                    biological.up_dvdt_mv_ms,
                    physical_scale,
                ),
                _curve_loss(
                    model.down_dvdt_mv_ms,
                    biological.down_dvdt_mv_ms,
                    physical_scale,
                ),
            )
        )
    )
    normalized = float(
        np.mean(
            (
                _curve_loss(
                    model.up_normalized,
                    biological.up_normalized,
                    config.normalized_curve_scale,
                ),
                _curve_loss(
                    model.down_normalized,
                    biological.down_normalized,
                    config.normalized_curve_scale,
                ),
            )
        )
    )
    slope = float(
        np.mean(
            (
                _curve_loss(
                    model.up_slope,
                    biological.up_slope,
                    config.normalized_slope_scale,
                ),
                _curve_loss(
                    model.down_slope,
                    biological.down_slope,
                    config.normalized_slope_scale,
                ),
            )
        )
    )
    concavity, regional_concavity, agreements = _concavity_loss(
        model,
        biological,
        config,
    )
    max_residual = model.max_dvdt_mv_ms - biological.max_dvdt_mv_ms
    min_residual = model.min_dvdt_mv_ms - biological.min_dvdt_mv_ms
    span_residual = model.dvdt_span_mv_ms - biological.dvdt_span_mv_ms
    velocity = float(
        np.mean(
            (
                _huber_array(
                    np.asarray(max_residual / config.max_dvdt_scale_mv_ms)
                ),
                _huber_array(
                    np.asarray(min_residual / config.min_dvdt_scale_mv_ms)
                ),
                _huber_array(
                    np.asarray(span_residual / config.dvdt_span_scale_mv_ms)
                ),
            )
        )
    )
    landmark_residuals = np.asarray(
        (
            (
                model.amplitude_mv - biological.amplitude_mv
            ) / config.amplitude_scale_mv,
            (
                model.repolarization_amplitude_mv
                - biological.repolarization_amplitude_mv
            ) / config.amplitude_scale_mv,
            (
                model.up_max_relative_voltage
                - biological.up_max_relative_voltage
            )
            / (
                config.extremum_voltage_scale_mv
                / max(biological.amplitude_mv, 1.0)
            ),
            (
                model.down_min_relative_voltage
                - biological.down_min_relative_voltage
            )
            / (
                config.extremum_voltage_scale_mv
                / max(biological.amplitude_mv, 1.0)
            ),
        )
    )
    landmarks = float(np.mean(_huber_array(landmark_residuals)))
    constraint_loss, peak_residual, minimum_voltage_residual = (
        _physical_constraint_loss(model, biological, config)
    )
    total = float(
        1.5 * physical
        + 3.0 * normalized
        + 1.0 * slope
        + 2.5 * concavity
        + 2.0 * velocity
        + 0.75 * landmarks
        + 4.0 * constraint_loss
    )
    return PhaseShapeScores(
        physical_curve=physical,
        normalized_shape=normalized,
        slope_shape=slope,
        concavity=concavity,
        velocity_extent=velocity,
        landmarks=landmarks,
        total=total,
        onset_concavity_sign_agreement=agreements["upstroke_onset"],
        max_dvdt_residual_mv_ms=float(max_residual),
        min_dvdt_residual_mv_ms=float(min_residual),
        dvdt_span_residual_mv_ms=float(span_residual),
        upstroke_onset_concavity=regional_concavity["upstroke_onset"],
        upstroke_late_concavity=regional_concavity["upstroke_late"],
        downstroke_high_voltage_concavity=regional_concavity[
            "downstroke_high_voltage"
        ],
        downstroke_low_voltage_concavity=regional_concavity[
            "downstroke_low_voltage"
        ],
        upstroke_late_sign_agreement=agreements["upstroke_late"],
        downstroke_high_voltage_sign_agreement=agreements[
            "downstroke_high_voltage"
        ],
        downstroke_low_voltage_sign_agreement=agreements[
            "downstroke_low_voltage"
        ],
        peak_voltage_residual_mv=peak_residual,
        minimum_voltage_residual_mv=minimum_voltage_residual,
        physical_constraint_loss=constraint_loss,
    )


def mean_phase_scores(
    scores: Mapping[str, PhaseShapeScores],
) -> PhaseShapeScores:
    """Average phase scores across protocols."""
    if not scores:
        raise ValueError("At least one phase score is required")
    mappings = [score.to_mapping() for score in scores.values()]
    values = {
        key: float(np.mean([mapping[key] for mapping in mappings]))
        for key in mappings[0]
    }
    return PhaseShapeScores(**values)
