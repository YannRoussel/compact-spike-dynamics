"""Diagnostic plots for simulation and parameter recovery."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .coverage import CoverageResult, DistributionComparison
from .hh_model import Trace


def save_trace_plot(trace: Trace, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    axes[0].plot(trace.time_ms, trace.voltage_mv, color="#222222", linewidth=1.2)
    axes[0].set(xlabel="Time (ms)", ylabel="V (mV)", title="Membrane voltage")

    axes[1].plot(
        trace.voltage_mv,
        trace.dvdt_mv_ms,
        color="#0072B2",
        linewidth=1.0,
        label="dV/dt",
    )
    axes[1].set(xlabel="V (mV)", ylabel="dV/dt (mV/ms)", title="Phase trajectory")

    axes[2].plot(trace.time_ms, trace.m, label="m", color="#D55E00")
    axes[2].plot(trace.time_ms, trace.h, label="h", color="#009E73")
    axes[2].plot(trace.time_ms, trace.n, label="n", color="#CC79A7")
    axes[2].set(xlabel="Time (ms)", ylabel="Open probability", title="Gating variables")
    axes[2].legend(frameon=False, ncol=3)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_recovery_plot(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    parameters = metrics["parameter"].tolist()
    n_columns = 3
    n_rows = ceil(len(parameters) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(10, 3.1 * n_rows),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    for axis, parameter in zip(axes_array, parameters):
        observed = predictions[f"observed__{parameter}"]
        predicted = predictions[f"predicted__{parameter}"]
        low = min(observed.min(), predicted.min())
        high = max(observed.max(), predicted.max())
        score = metrics.loc[metrics["parameter"] == parameter, "r2"].iloc[0]
        axis.scatter(observed, predicted, s=20, alpha=0.75, color="#0072B2")
        axis.plot([low, high], [low, high], color="#D55E00", linewidth=1.0)
        axis.set(
            xlabel="Observed",
            ylabel="Predicted",
            title=f"{parameter.removeprefix('param__')}\nR2={score:.2f}",
        )
    for axis in axes_array[len(parameters) :]:
        axis.set_visible(False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_coverage_plot(result: CoverageResult, output_path: str | Path) -> None:
    """Plot the biological manifold, HH models, and coverage diagnostics."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    dataset_styles = {
        "gouwens_visp": ("#0072B2", "Gouwens VISp"),
        "scala_room_temperature": ("#009E73", "Scala room temperature"),
        "scala_physiological_temperature": ("#E69F00", "Scala physiological temperature"),
    }
    for dataset_name, group in result.biological_coordinates.groupby("dataset"):
        color, label = dataset_styles.get(dataset_name, ("#777777", dataset_name))
        axes[0].scatter(
            group["pc1"],
            group["pc2"],
            s=8,
            alpha=0.28,
            color=color,
            linewidths=0,
            label=label,
        )
    axes[0].scatter(
        result.model_coordinates["pc1"],
        result.model_coordinates["pc2"],
        s=42,
        marker="x",
        color="#D62728",
        linewidths=1.4,
        label="HH models",
    )
    variance = 100.0 * result.explained_variance_ratio
    axes[0].set(
        xlabel=f"Biological PC1 ({variance[0]:.1f}%)",
        ylabel=f"Biological PC2 ({variance[1]:.1f}%)",
        title="Biological feature space",
    )
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")

    ranges = result.feature_ranges.sort_values("biological_interval_coverage")
    axes[1].barh(
        ranges["feature"].str.replace("_", " "),
        ranges["biological_interval_coverage"],
        color="#CC79A7",
    )
    axes[1].axvline(1.0, color="#333333", linewidth=0.8)
    axes[1].set(
        xlim=(0.0, 1.05),
        xlabel="Fraction of biological 1-99% interval",
        title="Univariate range coverage",
    )

    grouped_distances = list(result.biological_distances.groupby("dataset"))
    distance_groups = [
        group["distance_ratio"].to_numpy()
        for _, group in grouped_distances
    ]
    distance_labels = [
        (
            f"{dataset_styles.get(name, ('', name))[1]}\n"
            f"{int(group['covered'].sum())}/{len(group)} covered"
        )
        for name, group in grouped_distances
    ]
    axes[2].boxplot(
        distance_groups,
        labels=distance_labels,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#56B4E9", "alpha": 0.7},
        medianprops={"color": "#222222"},
    )
    axes[2].axhline(1.0, color="#D62728", linestyle="--", linewidth=1.0)
    for group_index, (_, group) in enumerate(grouped_distances, start=1):
        covered_ratios = group.loc[group["covered"], "distance_ratio"].to_numpy()
        if not len(covered_ratios):
            continue
        offsets = np.linspace(-0.08, 0.08, len(covered_ratios))
        axes[2].scatter(
            group_index + offsets,
            covered_ratios,
            s=24,
            color="#009E73",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].set(
        ylabel="Nearest-model distance / local biological radius",
        title="Local coverage (green points at or below 1)",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _feature_label(feature: str) -> str:
    replacements = {
        "ap": "AP",
        "dvdt": "dV/dt",
        "mv": "mV",
        "ms": "ms",
        "ms2": "ms^2",
        "mohm": "MOhm",
        "pa": "pA",
        "v2": "V^2",
    }
    words = [replacements.get(word, word) for word in feature.split("_")]
    return " ".join(words).replace(" iqr", " IQR")


def _draw_violin_half(
    axis: plt.Axes,
    values: np.ndarray,
    side: str,
    color: str,
) -> None:
    if values.size < 2 or np.ptp(values) <= np.finfo(float).eps:
        x_position = -0.12 if side == "left" else 0.12
        axis.scatter(x_position, values[0], color=color, s=18, zorder=3)
        return
    violin = axis.violinplot(
        values,
        positions=[0.0],
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=80,
    )
    for body in violin["bodies"]:
        vertices = body.get_paths()[0].vertices
        if side == "left":
            vertices[:, 0] = np.minimum(vertices[:, 0], 0.0)
        else:
            vertices[:, 0] = np.maximum(vertices[:, 0], 0.0)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.72)
        body.set_linewidth(0.7)

    q10, median, q90 = np.quantile(values, (0.1, 0.5, 0.9))
    x_position = -0.13 if side == "left" else 0.13
    axis.plot(
        [x_position, x_position],
        [q10, q90],
        color="#222222",
        linewidth=1.0,
        zorder=3,
    )
    axis.scatter(
        x_position,
        median,
        color="#222222",
        edgecolor="white",
        linewidth=0.4,
        s=16,
        zorder=4,
    )


def save_feature_distribution_violin_plot(
    result: DistributionComparison,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Render split biological/model violins, ordered by distribution mismatch."""
    ordered = result.summary.sort_values(
        "wasserstein_biological_iqr",
        ascending=False,
    )
    n_features = len(ordered)
    n_columns = 3 if n_features <= 15 else 4
    n_rows = ceil(n_features / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.15 * n_columns, 3.15 * n_rows),
        constrained_layout=False,
    )
    axes_array = np.asarray(axes).reshape(-1)
    biological_color = "#0072B2"
    model_color = "#D55E00"

    for axis, row in zip(axes_array, ordered.itertuples(index=False)):
        feature_values = result.values.loc[result.values["feature"].eq(row.feature)]
        biological_values = feature_values.loc[
            feature_values["cohort"].eq("Biological"),
            "display_normalized_value",
        ].to_numpy(dtype=float)
        model_values = feature_values.loc[
            feature_values["cohort"].eq("HH models"),
            "display_normalized_value",
        ].to_numpy(dtype=float)
        _draw_violin_half(axis, biological_values, "left", biological_color)
        _draw_violin_half(axis, model_values, "right", model_color)
        axis.axvline(0.0, color="#777777", linewidth=0.6)
        axis.axhline(0.0, color="#999999", linewidth=0.6, linestyle=":")
        axis.set_xlim(-0.45, 0.45)
        axis.set_xticks([])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.set_title(
            (
                f"{fill(_feature_label(row.feature), width=28)}\n"
                f"W1/IQR={row.wasserstein_biological_iqr:.2f}"
            ),
            fontsize=9,
        )
        for spine in ("top", "right", "bottom"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_color("#AAAAAA")

    for axis in axes_array[n_features:]:
        axis.set_visible(False)
    for row_index in range(n_rows):
        axes_array[row_index * n_columns].set_ylabel(
            "Biological-IQR units",
            fontsize=8,
        )

    figure.legend(
        handles=[
            Patch(facecolor=biological_color, alpha=0.72, label="Biological"),
            Patch(facecolor=model_color, alpha=0.72, label="HH models"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        frameon=False,
        ncol=2,
    )
    if title:
        figure.suptitle(title, y=0.999, fontsize=13)
    figure.text(
        0.5,
        0.006,
        (
            "Normalization: (value - biological median) / biological IQR. "
            "Dots and bars show median and 10-90%; density display trims 0.5% tails. "
            "Panels are ordered by normalized Wasserstein distance."
        ),
        ha="center",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.91 if title else 0.95,
        bottom=0.035,
        hspace=0.72,
        wspace=0.32,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
