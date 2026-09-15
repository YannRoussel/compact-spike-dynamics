"""Quantify how well sampled HH models cover biological e-feature space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class CoverageResult:
    biological_coordinates: pd.DataFrame
    model_coordinates: pd.DataFrame
    pca_feature_loadings: pd.DataFrame
    biological_distances: pd.DataFrame
    model_distances: pd.DataFrame
    summary: pd.DataFrame
    feature_ranges: pd.DataFrame
    explained_variance_ratio: np.ndarray
    features: tuple[str, ...]


@dataclass(frozen=True)
class DistributionComparison:
    values: pd.DataFrame
    summary: pd.DataFrame
    features: tuple[str, ...]


def compare_feature_distributions(
    biological: pd.DataFrame,
    models: pd.DataFrame,
    features: Sequence[str],
    display_tail_fraction: float = 0.005,
) -> DistributionComparison:
    """Normalize each feature to its biological median/IQR and compare cohorts."""
    if not 0.0 <= display_tail_fraction < 0.5:
        raise ValueError("display_tail_fraction must be in [0, 0.5)")

    value_frames = []
    summary_rows = []
    retained_features = []
    for feature in features:
        biological_values = pd.to_numeric(
            biological[feature], errors="coerce"
        ).to_numpy(dtype=float)
        model_values = pd.to_numeric(models[feature], errors="coerce").to_numpy(
            dtype=float
        )
        biological_values = biological_values[np.isfinite(biological_values)]
        model_values = model_values[np.isfinite(model_values)]
        if biological_values.size < 2 or model_values.size < 2:
            continue

        (
            biological_q01,
            biological_q25,
            biological_median,
            biological_q75,
            biological_q99,
        ) = np.quantile(
            biological_values,
            (0.01, 0.25, 0.5, 0.75, 0.99),
        )
        biological_iqr = biological_q75 - biological_q25
        if biological_iqr <= np.finfo(float).eps:
            biological_iqr = float(np.std(biological_values))
        if biological_iqr <= np.finfo(float).eps:
            continue

        biological_normalized = (
            biological_values - biological_median
        ) / biological_iqr
        model_normalized = (model_values - biological_median) / biological_iqr
        normalized_groups = {
            "Biological": biological_normalized,
            "HH models": model_normalized,
        }
        for cohort, normalized in normalized_groups.items():
            if display_tail_fraction:
                display_low, display_high = np.quantile(
                    normalized,
                    (display_tail_fraction, 1.0 - display_tail_fraction),
                )
                display_values = np.clip(normalized, display_low, display_high)
            else:
                display_values = normalized.copy()
            raw_values = (
                biological_values if cohort == "Biological" else model_values
            )
            value_frames.append(
                pd.DataFrame(
                    {
                        "feature": feature,
                        "cohort": cohort,
                        "raw_value": raw_values,
                        "normalized_value": normalized,
                        "display_normalized_value": display_values,
                        "display_value_was_trimmed": display_values != normalized,
                    }
                )
            )

        model_q01, model_q25, model_median, model_q75, model_q99 = np.quantile(
            model_values,
            (0.01, 0.25, 0.5, 0.75, 0.99),
        )
        biological_interval_width = biological_q99 - biological_q01
        model_interval_width = model_q99 - model_q01
        interval_overlap = max(
            0.0,
            min(biological_q99, model_q99)
            - max(biological_q01, model_q01),
        )
        summary_rows.append(
            {
                "feature": feature,
                "n_biological": biological_values.size,
                "n_models": model_values.size,
                "biological_median": biological_median,
                "biological_iqr": biological_iqr,
                "biological_q01": biological_q01,
                "biological_q99": biological_q99,
                "model_median": model_median,
                "model_iqr": model_q75 - model_q25,
                "model_q01": model_q01,
                "model_q99": model_q99,
                "model_median_shift_biological_iqr": (
                    model_median - biological_median
                )
                / biological_iqr,
                "wasserstein_biological_iqr": wasserstein_distance(
                    biological_normalized,
                    model_normalized,
                ),
                "ks_statistic": ks_2samp(
                    biological_values,
                    model_values,
                ).statistic,
                "biological_q01_q99_coverage": (
                    interval_overlap / biological_interval_width
                    if biological_interval_width > 0.0
                    else np.nan
                ),
                "model_q01_q99_coverage": (
                    interval_overlap / model_interval_width
                    if model_interval_width > 0.0
                    else np.nan
                ),
                "model_fraction_inside_biological_q01_q99": np.mean(
                    (model_values >= biological_q01)
                    & (model_values <= biological_q99)
                ),
                "display_tail_fraction": display_tail_fraction,
            }
        )
        retained_features.append(feature)

    if not value_frames:
        raise ValueError("No features contain enough finite values in both cohorts")
    return DistributionComparison(
        values=pd.concat(value_frames, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        features=tuple(retained_features),
    )


def analyze_coverage(
    biological: pd.DataFrame,
    models: pd.DataFrame,
    features: Sequence[str],
    local_neighbor_rank: int = 5,
) -> CoverageResult:
    """Use biological local density to define whether each real cell is covered."""
    features = tuple(features)
    biological_complete = biological.dropna(subset=list(features)).copy()
    model_complete = models.dropna(subset=list(features)).copy()
    if len(biological_complete) <= local_neighbor_rank:
        raise ValueError("Not enough complete biological cells for local coverage")
    if model_complete.empty:
        raise ValueError("No models have a complete feature vector")

    biological_values = biological_complete.loc[:, features].to_numpy(dtype=float)
    model_values = model_complete.loc[:, features].to_numpy(dtype=float)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(biological_values)
    biological_scaled = scaler.transform(biological_values)
    model_scaled = scaler.transform(model_values)

    pca = PCA(n_components=2).fit(biological_scaled)
    biological_pca = pca.transform(biological_scaled)
    model_pca = pca.transform(model_scaled)

    biological_neighbors = NearestNeighbors(
        n_neighbors=local_neighbor_rank + 1
    ).fit(biological_scaled)
    local_distances, _ = biological_neighbors.kneighbors(biological_scaled)
    local_radius = local_distances[:, local_neighbor_rank]

    model_neighbors = NearestNeighbors(n_neighbors=1).fit(model_scaled)
    distance_to_model, nearest_model = model_neighbors.kneighbors(biological_scaled)
    distance_to_model = distance_to_model[:, 0]
    covered = distance_to_model <= local_radius

    real_neighbors = NearestNeighbors(n_neighbors=1).fit(biological_scaled)
    distance_to_real, nearest_real = real_neighbors.kneighbors(model_scaled)
    distance_to_real = distance_to_real[:, 0]
    plausibility_radius = float(np.quantile(local_radius, 0.95))
    plausible = distance_to_real <= plausibility_radius

    biological_coordinates = biological_complete[
        ["dataset", "cell_id"]
    ].reset_index(drop=True)
    biological_coordinates["pc1"] = biological_pca[:, 0]
    biological_coordinates["pc2"] = biological_pca[:, 1]
    model_coordinates = model_complete[["dataset", "cell_id"]].reset_index(drop=True)
    model_coordinates["pc1"] = model_pca[:, 0]
    model_coordinates["pc2"] = model_pca[:, 1]

    pca_feature_loadings = pd.DataFrame({"feature": features})
    for component_index, component_name in enumerate(("pc1", "pc2")):
        pca_feature_loadings[f"{component_name}_weight"] = pca.components_[
            component_index
        ]
        pca_feature_loadings[f"{component_name}_correlation"] = [
            float(
                np.corrcoef(
                    biological_values[:, feature_index],
                    biological_pca[:, component_index],
                )[0, 1]
            )
            for feature_index in range(len(features))
        ]

    biological_distance_frame = biological_complete[
        ["dataset", "cell_id"]
    ].reset_index(drop=True)
    biological_distance_frame["nearest_model_distance"] = distance_to_model
    biological_distance_frame["local_biological_radius"] = local_radius
    biological_distance_frame["distance_ratio"] = distance_to_model / local_radius
    biological_distance_frame["covered"] = covered
    biological_distance_frame["nearest_model_row"] = nearest_model[:, 0]

    model_distance_frame = model_complete[["dataset", "cell_id"]].reset_index(drop=True)
    model_distance_frame["nearest_biological_distance"] = distance_to_real
    model_distance_frame["plausibility_radius"] = plausibility_radius
    model_distance_frame["biologically_plausible"] = plausible
    model_distance_frame["nearest_biological_row"] = nearest_real[:, 0]

    summary_rows = []
    for dataset_name, group in biological_distance_frame.groupby("dataset"):
        summary_rows.append(
            {
                "dataset": dataset_name,
                "n_biological_cells": len(group),
                "coverage_fraction": float(group["covered"].mean()),
                "median_nearest_model_distance": float(
                    group["nearest_model_distance"].median()
                ),
                "median_distance_ratio": float(group["distance_ratio"].median()),
            }
        )
    summary_rows.append(
        {
            "dataset": "all_biological",
            "n_biological_cells": len(biological_distance_frame),
            "coverage_fraction": float(biological_distance_frame["covered"].mean()),
            "median_nearest_model_distance": float(
                biological_distance_frame["nearest_model_distance"].median()
            ),
            "median_distance_ratio": float(
                biological_distance_frame["distance_ratio"].median()
            ),
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary["n_models"] = len(model_complete)
    summary["model_plausibility_fraction"] = float(plausible.mean())

    feature_rows = []
    for feature_index, feature_name in enumerate(features):
        real_low, real_high = np.quantile(
            biological_values[:, feature_index], (0.01, 0.99)
        )
        model_low, model_high = np.quantile(model_values[:, feature_index], (0.01, 0.99))
        overlap = max(0.0, min(real_high, model_high) - max(real_low, model_low))
        feature_rows.append(
            {
                "feature": feature_name,
                "biological_q01": real_low,
                "biological_q99": real_high,
                "model_q01": model_low,
                "model_q99": model_high,
                "biological_interval_coverage": overlap / (real_high - real_low),
            }
        )
    feature_ranges = pd.DataFrame(feature_rows)

    return CoverageResult(
        biological_coordinates=biological_coordinates,
        model_coordinates=model_coordinates,
        pca_feature_loadings=pca_feature_loadings,
        biological_distances=biological_distance_frame,
        model_distances=model_distance_frame,
        summary=summary,
        feature_ranges=feature_ranges,
        explained_variance_ratio=pca.explained_variance_ratio_,
        features=features,
    )


def save_coverage_tables(result: CoverageResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.biological_coordinates.to_csv(
        output / "biological_pca_coordinates.csv", index=False
    )
    result.model_coordinates.to_csv(output / "model_pca_coordinates.csv", index=False)
    result.pca_feature_loadings.to_csv(
        output / "pca_feature_loadings.csv", index=False
    )
    result.biological_distances.to_csv(
        output / "biological_coverage.csv", index=False
    )
    result.model_distances.to_csv(output / "model_plausibility.csv", index=False)
    result.summary.to_csv(output / "coverage_summary.csv", index=False)
    result.feature_ranges.to_csv(output / "feature_range_coverage.csv", index=False)


def save_distribution_tables(
    result: DistributionComparison,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.values.to_csv(output / "feature_distribution_values.csv", index=False)
    result.summary.to_csv(output / "feature_distribution_summary.csv", index=False)
