"""Summarize direct-fit residuals and target-local biological coverage."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.direct_optimization import (
    DIRECT_FIT_FEATURE_GROUPS,
    DIRECT_VALIDATION_FEATURE_GROUPS,
    biological_normalization,
    fit_features,
)


def _accepted(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["accepted"]
    if values.dtype == bool:
        return frame.loc[values].copy()
    return frame.loc[values.astype(str).str.lower().eq("true")].copy()


def summarize(
    biological: pd.DataFrame,
    targets: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    features = fit_features()
    complete = biological.dropna(subset=list(features)).copy()
    normalization = biological_normalization(complete, features)
    biological_scaled = (
        complete.loc[:, features] - normalization.center
    ) / normalization.scale
    accepted = _accepted(validation)
    rows = []
    plot_values = {}
    for _, target in targets.iterrows():
        target_id = str(target["target_id"])
        candidates = accepted.loc[accepted["target_id"].eq(target_id)]
        initial = candidates.loc[
            candidates["candidate_origin"].eq("initial")
        ].sort_values("objective_score").iloc[0]
        optimized = candidates.loc[
            candidates["candidate_origin"].eq("optimized")
        ].sort_values("objective_score").iloc[0]
        target_values = target.loc[list(features)].to_numpy(dtype=float)
        initial_values = initial.loc[
            [f"semantic__{feature}" for feature in features]
        ].to_numpy(dtype=float)
        optimized_values = optimized.loc[
            [f"semantic__{feature}" for feature in features]
        ].to_numpy(dtype=float)
        scales = normalization.scale.loc[list(features)].to_numpy(dtype=float)
        initial_residual = (initial_values - target_values) / scales
        optimized_residual = (optimized_values - target_values) / scales

        target_scaled = (
            target.loc[list(features)].to_numpy(dtype=float)
            - normalization.center.loc[list(features)].to_numpy(dtype=float)
        ) / scales
        biological_distances = np.linalg.norm(
            biological_scaled.to_numpy(dtype=float) - target_scaled,
            axis=1,
        )
        local_radius = float(np.partition(biological_distances, 5)[5])
        initial_distance = float(np.linalg.norm(initial_residual))
        optimized_distance = float(np.linalg.norm(optimized_residual))
        rows.append(
            {
                "target_id": target_id,
                "cell_id": target["cell_id"],
                "cluster_size": int(target["cluster_size"]),
                "initial_objective_score": initial["objective_score"],
                "optimized_objective_score": optimized["objective_score"],
                "objective_improvement_fraction": (
                    initial["objective_score"] - optimized["objective_score"]
                )
                / initial["objective_score"],
                "target_local_biological_radius": local_radius,
                "initial_target_distance": initial_distance,
                "optimized_target_distance": optimized_distance,
                "initial_local_radius_ratio": initial_distance / local_radius,
                "optimized_local_radius_ratio": optimized_distance / local_radius,
                "initial_onset_validation": initial["validation__onset"],
                "optimized_onset_validation": optimized["validation__onset"],
                "initial_acceleration_validation": initial[
                    "validation__acceleration"
                ],
                "optimized_acceleration_validation": optimized[
                    "validation__acceleration"
                ],
            }
        )
        family_names = list(DIRECT_FIT_FEATURE_GROUPS) + list(
            DIRECT_VALIDATION_FEATURE_GROUPS
        )
        initial_family = np.asarray(
            [
                initial.get(f"score__{family}", initial[f"validation__{family}"])
                if family in DIRECT_VALIDATION_FEATURE_GROUPS
                else initial[f"score__{family}"]
                for family in family_names
            ],
            dtype=float,
        )
        optimized_family = np.asarray(
            [
                optimized.get(f"score__{family}", optimized[f"validation__{family}"])
                if family in DIRECT_VALIDATION_FEATURE_GROUPS
                else optimized[f"score__{family}"]
                for family in family_names
            ],
            dtype=float,
        )
        plot_values[target_id] = {
            "initial_residual": initial_residual,
            "optimized_residual": optimized_residual,
            "family_names": np.asarray(family_names),
            "initial_family": initial_family,
            "optimized_family": optimized_family,
        }
    return pd.DataFrame(rows), plot_values


def save_plot(
    summary: pd.DataFrame,
    plot_values: dict[str, dict[str, np.ndarray]],
    output: Path,
    title: str,
) -> None:
    features = fit_features()
    n_targets = len(summary)
    figure, axes = plt.subplots(
        n_targets,
        2,
        figsize=(16, 4.3 * n_targets),
        squeeze=False,
    )
    for row_index, row in summary.iterrows():
        target_id = str(row["target_id"])
        values = plot_values[target_id]
        residual_axis = axes[row_index, 0]
        positions = np.arange(len(features))
        residual_axis.axhline(0.0, color="#777777", linewidth=0.8)
        residual_axis.plot(
            positions,
            values["initial_residual"],
            color="#777777",
            marker="o",
            linewidth=1.0,
            label="Initial nearest",
        )
        residual_axis.plot(
            positions,
            values["optimized_residual"],
            color="#D55E00",
            marker="o",
            linewidth=1.2,
            label="Direct optimized",
        )
        residual_axis.set_xticks(positions)
        residual_axis.set_xticklabels(
            [feature.replace("_", " ") for feature in features],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        residual_axis.set_ylabel("(model - target) / biological p10-p90")
        residual_axis.set_title(
            (
                f"{target_id}: normalized feature residuals "
                f"(radius ratio {row['initial_local_radius_ratio']:.2f} "
                f"to {row['optimized_local_radius_ratio']:.2f})"
            )
        )
        residual_axis.legend(frameon=False)
        residual_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

        family_axis = axes[row_index, 1]
        family_positions = np.arange(len(values["family_names"]))
        width = 0.38
        family_axis.bar(
            family_positions - width / 2.0,
            values["initial_family"],
            width=width,
            color="#777777",
            label="Initial nearest",
        )
        family_axis.bar(
            family_positions + width / 2.0,
            values["optimized_family"],
            width=width,
            color="#D55E00",
            label="Direct optimized",
        )
        family_axis.set_xticks(family_positions)
        family_axis.set_xticklabels(
            [name.replace("_", " ") for name in values["family_names"]],
            rotation=35,
            ha="right",
        )
        family_axis.set_ylabel("Grouped Huber loss")
        family_axis.set_title(f"{target_id}: fit and held-out families")
        family_axis.legend(frameon=False)
        family_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("biological")
    parser.add_argument("optimization_dir")
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    optimization_dir = Path(args.optimization_dir)
    biological = pd.read_csv(args.biological)
    targets = pd.read_csv(optimization_dir / "biological_targets.csv")
    validation = pd.read_csv(optimization_dir / "long_validation_models.csv")
    summary, plot_values = summarize(biological, targets, validation)
    summary.to_csv(optimization_dir / "direct_fit_diagnostics.csv", index=False)
    save_plot(
        summary,
        plot_values,
        optimization_dir / "direct_fit_diagnostics.png",
        args.title,
    )


if __name__ == "__main__":
    main()
