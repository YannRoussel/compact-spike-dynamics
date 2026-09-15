"""Reference-atlas helpers for transcriptomic spike-waveform prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .reduced_rank import (
    RRRFactors,
    residualize_class_fold,
    ridge_rrr,
    sparse_penalty_max,
    sparse_rrr,
)
from .transcriptomics import broad_transcriptomic_class


WAVEFORM_LABEL = "q050"
WAVEFORM_HARMONICS = 10
WAVEFORM_TARGET_NAMES = (
    "resting_voltage_mv",
    "template_mean_relative_mv_q050",
    *tuple(
        name
        for harmonic in range(1, WAVEFORM_HARMONICS + 1)
        for name in (
            f"template_cos_h{harmonic:02d}_mv_q050",
            f"template_sin_h{harmonic:02d}_mv_q050",
        )
    ),
    "downstroke_duration_ms_q050",
    "upstroke_duration_ms_q050",
    "recovery_baseline_period_q050_ms",
)
LOG_WAVEFORM_TARGETS = {
    "downstroke_duration_ms_q050",
    "upstroke_duration_ms_q050",
    "recovery_baseline_period_q050_ms",
}
AP_FEATURE_NAMES = (
    "ap_peak_relative_mv",
    "ap_trough_relative_mv",
    "ap_amplitude_mv",
    "ap_half_width_ms",
    "max_upstroke_mv_ms",
    "max_downstroke_mv_ms",
    "dvdt_span_mv_ms",
    "voltage_at_max_upstroke_mv",
    "voltage_at_max_downstroke_mv",
    "upstroke_downstroke_ratio",
    "phase_loop_area_mv2_ms",
    "normalized_phase_loop_area",
    "peak_to_trough_ms",
    "downstroke_duration_ms",
    "upstroke_duration_ms",
)


@dataclass(frozen=True)
class SpikeCycle:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    velocity_mv_ms: np.ndarray
    period_ms: float
    resting_voltage_mv: float


@dataclass(frozen=True)
class FixedClassRRR:
    factors: RRRFactors
    selected_predictors: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray


@dataclass(frozen=True)
class TranscriptomicDistanceModel:
    gene_mean: np.ndarray
    gene_scale: np.ndarray
    pca: PCA
    training_scores: np.ndarray
    training_types: np.ndarray
    training_classes: np.ndarray
    reference_distance: float


def waveform_target_matrix(parameters: pd.DataFrame) -> np.ndarray:
    """Return the compact q50 spike-cycle target in stable coordinates."""
    missing = set(WAVEFORM_TARGET_NAMES).difference(parameters.columns)
    if missing:
        raise ValueError(f"Missing waveform columns: {sorted(missing)}")
    columns = []
    for name in WAVEFORM_TARGET_NAMES:
        values = pd.to_numeric(parameters[name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite waveform values in {name}")
        columns.append(
            np.log1p(np.maximum(values, 0.0))
            if name in LOG_WAVEFORM_TARGETS
            else values
        )
    return np.column_stack(columns)


def inverse_waveform_target_matrix(values: np.ndarray) -> np.ndarray:
    """Undo positive-scale transforms in a waveform target matrix."""
    result = np.asarray(values, dtype=float).copy()
    for index, name in enumerate(WAVEFORM_TARGET_NAMES):
        if name in LOG_WAVEFORM_TARGETS:
            result[:, index] = np.expm1(result[:, index])
    return result


def waveform_parameter_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(values, dtype=float),
        columns=WAVEFORM_TARGET_NAMES,
    )


def clip_waveform_parameters(
    values: np.ndarray,
    training_values: np.ndarray,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain generated waveforms to the empirical training envelope."""
    values = np.asarray(values, dtype=float)
    training_values = np.asarray(training_values, dtype=float)
    lower = np.quantile(training_values, lower_quantile, axis=0)
    upper = np.quantile(training_values, upper_quantile, axis=0)
    clipped = np.clip(values, lower, upper)
    return clipped, np.any(~np.isclose(values, clipped), axis=1)


def _periodic_derivative(
    values: np.ndarray,
    time_ms: np.ndarray,
    period_ms: float,
) -> np.ndarray:
    previous_values = np.roll(values, 1)
    next_values = np.roll(values, -1)
    previous_time = np.roll(time_ms, 1)
    next_time = np.roll(time_ms, -1)
    previous_time[0] -= period_ms
    next_time[-1] += period_ms
    return (next_values - previous_values) / (next_time - previous_time)


def spike_cycle_from_row(
    row: Mapping[str, object],
    phase_points: int = 256,
) -> SpikeCycle:
    """Reconstruct one periodic q50 spike cycle from Fourier parameters."""
    if phase_points % 4:
        raise ValueError("phase_points must be divisible by four")
    phase = np.linspace(0.0, 1.0, phase_points, endpoint=False)
    relative = np.full(
        phase_points,
        float(row["template_mean_relative_mv_q050"]),
        dtype=float,
    )
    for harmonic in range(1, WAVEFORM_HARMONICS + 1):
        angle = 2.0 * np.pi * harmonic * phase
        relative += (
            float(row[f"template_cos_h{harmonic:02d}_mv_q050"])
            * np.cos(angle)
            + float(row[f"template_sin_h{harmonic:02d}_mv_q050"])
            * np.sin(angle)
        )
    resting = float(row["resting_voltage_mv"])
    downstroke = max(0.02, float(row["downstroke_duration_ms_q050"]))
    upstroke = max(0.02, float(row["upstroke_duration_ms_q050"]))
    period = max(
        downstroke + upstroke + 0.1,
        float(row["recovery_baseline_period_q050_ms"]),
    )
    recovery = period - downstroke - upstroke
    quarter = phase_points // 4
    half = phase_points // 2
    time_ms = np.concatenate(
        (
            np.linspace(0.0, downstroke, quarter, endpoint=False),
            np.linspace(
                downstroke,
                downstroke + recovery,
                half,
                endpoint=False,
            ),
            np.linspace(
                downstroke + recovery,
                period,
                quarter,
                endpoint=False,
            ),
        )
    )
    voltage = resting + relative
    velocity = _periodic_derivative(voltage, time_ms, period)
    return SpikeCycle(time_ms, voltage, velocity, period, resting)


def _cyclic_peak_coordinates(cycle: SpikeCycle) -> tuple[np.ndarray, np.ndarray]:
    peak = int(np.argmax(cycle.voltage_mv))
    voltage = np.roll(cycle.voltage_mv, -peak)
    intervals = np.diff(np.r_[cycle.time_ms, cycle.period_ms])
    intervals = np.roll(intervals, -peak)
    time = np.r_[0.0, np.cumsum(intervals[:-1])]
    return time, voltage


def _crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    level: float,
    start: int,
    stop: int,
    rising: bool,
) -> float:
    for index in range(max(1, start), min(stop, len(values))):
        before = values[index - 1] - level
        after = values[index] - level
        crossed = before <= 0.0 < after if rising else before >= 0.0 > after
        if crossed:
            fraction = abs(before) / max(abs(before) + abs(after), 1e-12)
            return float(
                time[index - 1]
                + fraction * (time[index] - time[index - 1])
            )
    return float("nan")


def spike_cycle_features(cycle: SpikeCycle) -> dict[str, float]:
    """Calculate AP waveform, inflection, and phase-loop features."""
    voltage = cycle.voltage_mv
    velocity = cycle.velocity_mv_ms
    peak = float(np.max(voltage))
    trough = float(np.min(voltage))
    amplitude = peak - trough
    time_from_peak, voltage_from_peak = _cyclic_peak_coordinates(cycle)
    trough_index = int(np.argmin(voltage_from_peak))
    half_level = trough + 0.5 * amplitude
    descending = _crossing_time(
        time_from_peak,
        voltage_from_peak,
        half_level,
        1,
        trough_index + 1,
        rising=False,
    )
    ascending = _crossing_time(
        time_from_peak,
        voltage_from_peak,
        half_level,
        trough_index + 1,
        len(voltage_from_peak),
        rising=True,
    )
    half_width = (
        descending + cycle.period_ms - ascending
        if np.isfinite(descending) and np.isfinite(ascending)
        else float("nan")
    )
    maximum_index = int(np.argmax(velocity))
    minimum_index = int(np.argmin(velocity))
    maximum_velocity = float(velocity[maximum_index])
    minimum_velocity = float(velocity[minimum_index])
    closed_voltage = np.r_[voltage, voltage[0]]
    closed_velocity = np.r_[velocity, velocity[0]]
    loop_area = 0.5 * abs(
        float(
            np.sum(
                closed_voltage[:-1] * closed_velocity[1:]
                - closed_voltage[1:] * closed_velocity[:-1]
            )
        )
    )
    velocity_span = maximum_velocity - minimum_velocity
    normalized_area = loop_area / max(amplitude * velocity_span, 1e-12)
    return {
        "ap_peak_relative_mv": peak - cycle.resting_voltage_mv,
        "ap_trough_relative_mv": trough - cycle.resting_voltage_mv,
        "ap_amplitude_mv": amplitude,
        "ap_half_width_ms": half_width,
        "max_upstroke_mv_ms": maximum_velocity,
        "max_downstroke_mv_ms": minimum_velocity,
        "dvdt_span_mv_ms": velocity_span,
        "voltage_at_max_upstroke_mv": float(voltage[maximum_index]),
        "voltage_at_max_downstroke_mv": float(voltage[minimum_index]),
        "upstroke_downstroke_ratio": maximum_velocity
        / max(abs(minimum_velocity), 1e-12),
        "phase_loop_area_mv2_ms": loop_area,
        "normalized_phase_loop_area": normalized_area,
        "peak_to_trough_ms": float(time_from_peak[trough_index]),
        "downstroke_duration_ms": float(
            cycle.time_ms[len(cycle.time_ms) // 4]
            if len(cycle.time_ms) > 4
            else np.nan
        ),
        "upstroke_duration_ms": float(
            cycle.period_ms - cycle.time_ms[3 * len(cycle.time_ms) // 4]
        ),
    }


def waveform_features(parameters: pd.DataFrame) -> pd.DataFrame:
    rows = [
        spike_cycle_features(spike_cycle_from_row(row))
        for row in parameters.to_dict(orient="records")
    ]
    return pd.DataFrame(rows, columns=AP_FEATURE_NAMES)


def ap_feature_target_matrix(features: pd.DataFrame) -> np.ndarray:
    """Compress AP feature dynamic ranges while retaining their signs."""
    values = features.loc[:, AP_FEATURE_NAMES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("AP feature target contains nonfinite values")
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_ap_feature_target_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.expm1(np.abs(values))


def clip_ap_features(
    values: np.ndarray,
    training_values: np.ndarray,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    lower = np.quantile(training_values, lower_quantile, axis=0)
    upper = np.quantile(training_values, upper_quantile, axis=0)
    clipped = np.clip(values, lower, upper)
    return clipped, np.any(~np.isclose(values, clipped), axis=1)


def atlas_broad_class(label: object) -> str:
    """Map a CTX-HPF atlas leaf alias to the analysis subclasses."""
    text = re.sub(r"^\d+_", "", str(label).strip())
    if text == "CR" or text.startswith("Lamp5") or text.startswith("Pax6"):
        return "Lamp5"
    if re.search(r"\b(?:PT|ET)(?:_| |$)", text):
        return "Glut_ET"
    if re.search(r"\bIT(?:_| |$)", text):
        return "Glut_IT"
    mapped = broad_transcriptomic_class(text)
    if mapped != "Other":
        return mapped
    if re.match(r"^L(?:2|3|4|5|6)(?:/\d)?\s", text) and " PT " not in text:
        return "Glut_IT"
    if text.startswith("IT ") or text.startswith("Car3"):
        return "Glut_IT"
    return "Other"


def load_allen_reference_atlas(
    expression_path: str | Path,
    genes: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load neural SMART-seq cluster means and parsed leaf metadata."""
    requested = set(str(gene) for gene in genes) if genes is not None else None
    frame = pd.read_csv(expression_path, index_col=0)
    frame.index = frame.index.astype(str)
    if requested is not None:
        frame = frame.loc[frame.index.intersection(requested)]
    aliases = frame.columns.astype(str)
    non_neural = aliases.str.contains(
        r"_(?:Oligo|Astro|Endo|SMC-Peri|VLMC|Micro-PVM)$",
        regex=True,
    )
    aliases = aliases[~non_neural]
    expression = frame.loc[:, aliases].T
    expression.index.name = "atlas_type"
    metadata = pd.DataFrame(
        {
            "atlas_type": aliases,
            "cluster_id": [int(alias.split("_", 1)[0]) for alias in aliases],
            "leaf_label": [alias.split("_", 1)[1] for alias in aliases],
            "broad_class": [atlas_broad_class(alias) for alias in aliases],
        }
    ).sort_values("cluster_id")
    expression = expression.loc[metadata["atlas_type"]]
    return expression, metadata.reset_index(drop=True)


def harmonize_atlas_to_patchseq(
    atlas_expression: pd.DataFrame,
    patch_expression: np.ndarray,
    patch_types: Sequence[object],
    genes: Sequence[str],
) -> np.ndarray:
    """Affine-align reference cluster means to Patch-seq type centroids."""
    genes = np.asarray(genes, dtype=str)
    atlas = atlas_expression.loc[:, genes].to_numpy(dtype=float)
    patch = np.asarray(patch_expression, dtype=float)
    labels = np.asarray(patch_types, dtype=object)
    centroids = np.vstack(
        [np.mean(patch[labels == label], axis=0) for label in np.unique(labels)]
    )
    atlas_mean = np.mean(atlas, axis=0)
    atlas_scale = np.std(atlas, axis=0)
    patch_mean = np.mean(centroids, axis=0)
    patch_scale = np.std(centroids, axis=0)
    standardized = (atlas - atlas_mean) / np.where(
        atlas_scale > 1e-8,
        atlas_scale,
        1.0,
    )
    harmonized = standardized * patch_scale + patch_mean
    lower = np.quantile(patch, 0.001, axis=0)
    upper = np.quantile(patch, 0.999, axis=0)
    return np.clip(harmonized, lower, upper)


def _scaling(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    return mean, np.where(scale > 1e-8, scale, 1.0)


def fit_fixed_class_rrr(
    x: np.ndarray,
    y: np.ndarray,
    classes: Sequence[object],
    rank: int,
    ridge_penalty: float,
    sparse_ratio: float,
    l1_ratio: float = 0.5,
) -> FixedClassRRR:
    """Fit a fixed-hyperparameter class-residualized relaxed sparse RRR."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(classes, dtype=object)
    x_residual, _, _ = residualize_class_fold(x, x, labels, labels)
    y_residual, _, _ = residualize_class_fold(y, y, labels, labels)
    x_mean, x_scale = _scaling(x_residual)
    y_mean, y_scale = _scaling(y_residual)
    x_scaled = np.clip((x_residual - x_mean) / x_scale, -8.0, 8.0)
    y_scaled = np.clip((y_residual - y_mean) / y_scale, -8.0, 8.0)
    maximum = sparse_penalty_max(x_scaled, y_scaled, rank, l1_ratio)
    sparse = sparse_rrr(
        x_scaled,
        y_scaled,
        rank,
        alpha=float(sparse_ratio) * maximum,
        l1_ratio=l1_ratio,
        maximum_iterations=15,
    )
    selected = np.linalg.norm(sparse.predictor_weights, axis=1) > 1e-10
    if int(np.sum(selected)) < rank:
        selected = np.ones(x.shape[1], dtype=bool)
    relaxed = ridge_rrr(
        x_scaled[:, selected],
        y_scaled,
        rank,
        ridge_penalty,
    )
    predictor = np.zeros((x.shape[1], rank), dtype=float)
    predictor[selected] = relaxed.predictor_weights
    return FixedClassRRR(
        RRRFactors(predictor, relaxed.response_weights),
        selected,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    )


def predict_fixed_class_rrr(
    model: FixedClassRRR,
    x_training: np.ndarray,
    y_training: np.ndarray,
    training_classes: Sequence[object],
    x_new: np.ndarray,
    new_classes: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(training_classes, dtype=object)
    requested = np.asarray(new_classes, dtype=object)
    _, x_residual, _ = residualize_class_fold(
        x_training,
        x_new,
        labels,
        requested,
    )
    _, _, y_baseline = residualize_class_fold(
        y_training,
        np.zeros((len(x_new), y_training.shape[1]), dtype=float),
        labels,
        requested,
    )
    x_scaled = np.clip((x_residual - model.x_mean) / model.x_scale, -8.0, 8.0)
    residual = (
        x_scaled
        @ model.factors.predictor_weights
        @ model.factors.response_weights.T
    )
    return residual * model.y_scale + model.y_mean + y_baseline, y_baseline


def fit_distance_model(
    x: np.ndarray,
    types: Sequence[object],
    classes: Sequence[object],
    maximum_components: int = 20,
) -> TranscriptomicDistanceModel:
    """Fit a whitened type-centroid PCA used only for support diagnostics."""
    x = np.asarray(x, dtype=float)
    labels = np.asarray(types, dtype=object)
    class_values = np.asarray(classes, dtype=object)
    unique = np.unique(labels)
    centroids = np.vstack([np.mean(x[labels == label], axis=0) for label in unique])
    type_classes = np.asarray(
        [class_values[np.flatnonzero(labels == label)[0]] for label in unique],
        dtype=object,
    )
    mean, scale = _scaling(centroids)
    standardized = np.clip((centroids - mean) / scale, -8.0, 8.0)
    components = min(maximum_components, len(unique) - 1, x.shape[1])
    pca = PCA(n_components=max(1, components), whiten=True, random_state=0)
    scores = pca.fit_transform(standardized)
    nearest = []
    for index, broad_class in enumerate(type_classes):
        candidates = np.flatnonzero(type_classes == broad_class)
        candidates = candidates[candidates != index]
        if not len(candidates):
            candidates = np.delete(np.arange(len(unique)), index)
        nearest.append(
            float(np.min(np.linalg.norm(scores[candidates] - scores[index], axis=1)))
        )
    reference = max(float(np.quantile(nearest, 0.95)), 1e-8)
    return TranscriptomicDistanceModel(
        mean,
        scale,
        pca,
        scores,
        unique.astype(object),
        type_classes,
        reference,
    )


def transcriptomic_support(
    model: TranscriptomicDistanceModel,
    values: np.ndarray,
    classes: Sequence[object],
) -> pd.DataFrame:
    standardized = np.clip(
        (np.asarray(values, dtype=float) - model.gene_mean) / model.gene_scale,
        -8.0,
        8.0,
    )
    scores = model.pca.transform(standardized)
    rows = []
    for score, broad_class in zip(scores, classes):
        candidates = np.flatnonzero(model.training_classes == broad_class)
        class_supported = bool(len(candidates)) and str(broad_class) != "Other"
        if not class_supported:
            candidates = np.arange(len(model.training_types))
        distances = np.linalg.norm(model.training_scores[candidates] - score, axis=1)
        nearest_local = int(np.argmin(distances))
        nearest = int(candidates[nearest_local])
        distance = float(distances[nearest_local])
        ratio = distance / model.reference_distance
        rows.append(
            {
                "nearest_patchseq_type": str(model.training_types[nearest]),
                "transcriptomic_distance": distance,
                "distance_ratio_to_loto_p95": ratio,
                "class_supported": class_supported,
                "support_tier": (
                    "unsupported_class"
                    if not class_supported
                    else "near"
                    if ratio <= 0.5
                    else "moderate"
                    if ratio <= 1.0
                    else "far"
                ),
            }
        )
    return pd.DataFrame(rows)
