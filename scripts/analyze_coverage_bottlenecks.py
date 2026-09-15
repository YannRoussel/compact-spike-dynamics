"""Audit why HH populations fail to cover biological spike-cycle space."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from inverse_ephys_alpha_beta.biological_data import (
    SPIKE_CYCLE_COMMON_FEATURES,
    WAVEFORM_CORE_FEATURES,
)
from inverse_ephys_alpha_beta.coverage import analyze_coverage


FEATURE_FAMILIES = {
    "waveform core": WAVEFORM_CORE_FEATURES,
    "velocity": (
        "upstroke_mv_ms",
        "downstroke_mv_ms",
    ),
    "timing": (
        "ap_upstroke_time_ms",
        "ap_repolarization_time_ms",
        "ap_duration_ms",
    ),
    "onset": (
        "onset_voltage_mv",
        "onset_dvdt_mv_ms",
        "onset_rapidness_per_ms",
    ),
    "acceleration": (
        "max_acceleration_mv_ms2",
        "max_acceleration_voltage_mv",
        "min_acceleration_mv_ms2",
        "min_acceleration_voltage_mv",
    ),
    "inflection": (
        "upstroke_inflection_voltage_mv",
        "upstroke_inflection_dvdt_mv_ms",
        "downstroke_inflection_voltage_mv",
        "downstroke_inflection_dvdt_mv_ms",
        "upstroke_inflection_relative_to_threshold_mv",
        "downstroke_inflection_relative_to_peak_mv",
    ),
    "phase geometry": (
        "ap_phase_area_v2_ms",
        "ap_phase_area_normalized",
        "cycle_phase_area_v2_ms",
        "cycle_phase_area_normalized",
        "cycle_phase_path_length_normalized",
    ),
}

NONREDUNDANT_SPIKE_CYCLE_FEATURES = (
    "ap_threshold_mv",
    "ap_peak_mv",
    "ap_width_ms",
    "fast_trough_mv",
    "upstroke_mv_ms",
    "downstroke_mv_ms",
    "ap_upstroke_time_ms",
    "ap_repolarization_time_ms",
    "onset_voltage_mv",
    "onset_rapidness_per_ms",
    "max_acceleration_mv_ms2",
    "max_acceleration_voltage_mv",
    "min_acceleration_mv_ms2",
    "min_acceleration_voltage_mv",
    "upstroke_inflection_voltage_mv",
    "upstroke_inflection_dvdt_mv_ms",
    "downstroke_inflection_voltage_mv",
    "downstroke_inflection_dvdt_mv_ms",
    "ap_phase_area_normalized",
    "cycle_phase_area_normalized",
    "cycle_phase_path_length_normalized",
)


def _coverage_row(
    biological: pd.DataFrame,
    models: pd.DataFrame,
    features: tuple[str, ...],
    analysis: str,
) -> dict[str, float | int | str]:
    result = analyze_coverage(biological, models, features)
    total = result.summary.loc[
        result.summary["dataset"].eq("all_biological")
    ].iloc[0]
    return {
        "analysis": analysis,
        "n_features": len(features),
        "n_biological": int(total["n_biological_cells"]),
        "n_models": int(total["n_models"]),
        "n_covered": int(round(total["coverage_fraction"] * total["n_biological_cells"])),
        "coverage_fraction": float(total["coverage_fraction"]),
        "median_nearest_model_distance": float(
            total["median_nearest_model_distance"]
        ),
        "median_distance_ratio": float(total["median_distance_ratio"]),
    }


def family_ablation(
    biological: pd.DataFrame,
    models: pd.DataFrame,
) -> pd.DataFrame:
    all_features = tuple(SPIKE_CYCLE_COMMON_FEATURES)
    spike_shape = (
        tuple(FEATURE_FAMILIES["waveform core"])
        + tuple(FEATURE_FAMILIES["velocity"])
        + tuple(FEATURE_FAMILIES["timing"])
    )
    rows = [
        _coverage_row(
            biological,
            models,
            tuple(FEATURE_FAMILIES["waveform core"]),
            "waveform core only",
        ),
        _coverage_row(
            biological,
            models,
            spike_shape,
            "waveform + velocity + timing",
        ),
        _coverage_row(
            biological,
            models,
            NONREDUNDANT_SPIKE_CYCLE_FEATURES,
            "nonredundant spike-cycle features",
        ),
        _coverage_row(biological, models, all_features, "all spike-cycle features"),
    ]
    for family, family_features in FEATURE_FAMILIES.items():
        if family == "waveform core":
            continue
        retained = tuple(
            feature
            for feature in all_features
            if feature not in set(family_features)
        )
        rows.append(
            _coverage_row(
                biological,
                models,
                retained,
                f"all except {family}",
            )
        )
    return pd.DataFrame(rows)


def pca_manifold_coverage(
    biological: pd.DataFrame,
    models: pd.DataFrame,
    explained_variance: float = 0.95,
    local_neighbor_rank: int = 5,
) -> dict[str, float | int | str]:
    features = NONREDUNDANT_SPIKE_CYCLE_FEATURES
    biological_complete = biological.dropna(subset=list(features))
    model_complete = models.dropna(subset=list(features))
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
        biological_complete.loc[:, features]
    )
    biological_scaled = scaler.transform(biological_complete.loc[:, features])
    model_scaled = scaler.transform(model_complete.loc[:, features])
    pca = PCA().fit(biological_scaled)
    n_components = int(
        np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_),
            explained_variance,
        )
        + 1
    )
    biological_reduced = pca.transform(biological_scaled)[:, :n_components]
    model_reduced = pca.transform(model_scaled)[:, :n_components]

    biological_neighbors = NearestNeighbors(
        n_neighbors=local_neighbor_rank + 1
    ).fit(biological_reduced)
    local_distances, _ = biological_neighbors.kneighbors(biological_reduced)
    local_radius = local_distances[:, local_neighbor_rank]
    model_neighbors = NearestNeighbors(n_neighbors=1).fit(model_reduced)
    distance_to_model, _ = model_neighbors.kneighbors(biological_reduced)
    distance_to_model = distance_to_model[:, 0]
    covered = distance_to_model <= local_radius
    return {
        "analysis": f"nonredundant PCA {explained_variance:.0%}",
        "n_features": n_components,
        "n_biological": len(biological_complete),
        "n_models": len(model_complete),
        "n_covered": int(covered.sum()),
        "coverage_fraction": float(covered.mean()),
        "median_nearest_model_distance": float(np.median(distance_to_model)),
        "median_distance_ratio": float(np.median(distance_to_model / local_radius)),
    }


def nearest_model_contributions(
    biological: pd.DataFrame,
    models: pd.DataFrame,
) -> pd.DataFrame:
    features = tuple(SPIKE_CYCLE_COMMON_FEATURES)
    biological_complete = biological.dropna(subset=list(features))
    model_complete = models.dropna(subset=list(features))
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
        biological_complete.loc[:, features]
    )
    biological_scaled = scaler.transform(biological_complete.loc[:, features])
    model_scaled = scaler.transform(model_complete.loc[:, features])
    nearest = NearestNeighbors(n_neighbors=1).fit(model_scaled)
    _, indices = nearest.kneighbors(biological_scaled)
    differences = biological_scaled - model_scaled[indices[:, 0]]
    squared = differences**2
    total_squared = squared.sum(axis=1)
    safe_total = np.where(total_squared > 0.0, total_squared, np.nan)
    return pd.DataFrame(
        {
            "feature": features,
            "mean_squared_distance_share": squared.sum(axis=0) / squared.sum(),
            "median_cell_distance_share": np.nanmedian(
                squared / safe_total[:, None],
                axis=0,
            ),
            "median_absolute_scaled_offset": np.median(
                np.abs(differences),
                axis=0,
            ),
        }
    ).sort_values("mean_squared_distance_share", ascending=False)


def same_distribution_benchmark(
    biological: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    features = tuple(SPIKE_CYCLE_COMMON_FEATURES)
    complete = biological.dropna(subset=list(features)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        order = rng.permutation(len(complete))
        midpoint = len(order) // 2
        evaluated = complete.iloc[order[:midpoint]].copy()
        pseudo_models = complete.iloc[order[midpoint:]].copy()
        pseudo_models["dataset"] = "biological_pseudo_model"
        result = analyze_coverage(evaluated, pseudo_models, features)
        total = result.summary.loc[
            result.summary["dataset"].eq("all_biological")
        ].iloc[0]
        rows.append(
            {
                "replicate": replicate,
                "n_evaluated": int(total["n_biological_cells"]),
                "n_pseudo_models": int(total["n_models"]),
                "coverage_fraction": float(total["coverage_fraction"]),
                "median_distance_ratio": float(total["median_distance_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def save_audit_plot(
    ablation: pd.DataFrame,
    contributions: pd.DataFrame,
    benchmark: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ordered = ablation.sort_values("median_distance_ratio")
    axes[0].barh(
        ordered["analysis"],
        ordered["median_distance_ratio"],
        color="#0072B2",
    )
    axes[0].axvline(1.0, color="#D55E00", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="Median nearest-model / local-radius ratio",
        title="Feature-family ablation",
    )

    top = contributions.head(12).sort_values("mean_squared_distance_share")
    axes[1].barh(
        top["feature"].str.replace("_", " "),
        100.0 * top["mean_squared_distance_share"],
        color="#CC79A7",
    )
    axes[1].set(
        xlabel="Share of squared nearest-model distance (%)",
        title="Largest distance contributors",
    )

    actual = ablation.loc[
        ablation["analysis"].eq("all spike-cycle features"),
        "coverage_fraction",
    ].iloc[0]
    benchmark_mean = benchmark["coverage_fraction"].mean()
    benchmark_low, benchmark_high = benchmark["coverage_fraction"].quantile(
        (0.05, 0.95)
    )
    axes[2].bar(
        ["HH models", "Biological\n50:50 split"],
        [actual, benchmark_mean],
        color=["#D55E00", "#009E73"],
    )
    axes[2].errorbar(
        1,
        benchmark_mean,
        yerr=[
            [benchmark_mean - benchmark_low],
            [benchmark_high - benchmark_mean],
        ],
        color="#222222",
        capsize=4,
        linewidth=1.0,
    )
    axes[2].set(
        ylabel="Coverage fraction",
        title="Same-distribution control",
    )
    axes[2].set_ylim(bottom=0.0)
    axes[2].text(
        0,
        actual,
        f"{actual:.1%}",
        ha="center",
        va="bottom",
    )
    axes[2].text(
        1,
        benchmark_mean,
        f"{benchmark_mean:.1%}",
        ha="center",
        va="bottom",
    )
    figure.suptitle(title, y=0.985)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("biological")
    parser.add_argument("models")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    biological = pd.read_csv(args.biological)
    models = pd.read_csv(args.models)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ablation = family_ablation(biological, models)
    ablation = pd.concat(
        [
            ablation,
            pd.DataFrame([pca_manifold_coverage(biological, models)]),
        ],
        ignore_index=True,
    )
    contributions = nearest_model_contributions(biological, models)
    benchmark = same_distribution_benchmark(
        biological,
        replicates=args.replicates,
        seed=args.seed,
    )
    ablation.to_csv(output / "feature_family_ablation.csv", index=False)
    contributions.to_csv(
        output / "nearest_model_feature_contributions.csv",
        index=False,
    )
    benchmark.to_csv(output / "same_distribution_benchmark.csv", index=False)
    save_audit_plot(
        ablation,
        contributions,
        benchmark,
        output / "coverage_bottleneck_audit.png",
        title=args.title,
    )


if __name__ == "__main__":
    main()
