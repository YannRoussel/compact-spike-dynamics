#!/usr/bin/env python3
"""Compare nested phase-template timing clocks on a fixed cell cohort."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from inverse_ephys_alpha_beta.phase_timing_population import (
    phase_timing_population_task,
)


DATASETS = ("gouwens_visp", "scala_room_temperature")
MODELS = (
    "legacy_inverse_memory",
    "current_spline",
    "spike_exponential",
    "izhikevich_recovery",
)
MODEL_LABELS = {
    "legacy_inverse_memory": "legacy",
    "current_spline": "current\nspline",
    "spike_exponential": "spike\nmemory",
    "izhikevich_recovery": "Izh\nrecovery",
}
COLORS = {
    "gouwens_visp": "#0072b2",
    "scala_room_temperature": "#d55e00",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint(
    output_root: Path,
    metrics: list[dict],
    candidates: list[dict],
    parameters: list[dict],
    failures: list[dict],
) -> None:
    _write_csv(
        pd.DataFrame(metrics),
        output_root / "timing_model_metrics.csv",
    )
    _write_csv(
        pd.DataFrame(candidates),
        output_root / "timing_candidates.csv",
    )
    _write_csv(
        pd.DataFrame(parameters),
        output_root / "timing_parameters.csv",
    )
    _write_csv(
        pd.DataFrame(failures),
        output_root / "failures.csv",
    )


def _summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), frame in metrics.groupby(
        ["dataset", "model"],
        sort=False,
    ):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "cell_count": frame["cell_id"].nunique(),
                "median_phase_chamfer": frame["phase_chamfer"].median(),
                "median_absolute_spike_count_error": frame[
                    "absolute_spike_count_error"
                ].median(),
                "median_relative_spike_count_error": frame[
                    "relative_spike_count_error"
                ].median(),
                "spike_count_within_one_fraction": (
                    frame["absolute_spike_count_error"] <= 1
                ).mean(),
                "median_absolute_latency_error_ms": frame[
                    "absolute_latency_ms_error"
                ].median(),
                "median_absolute_first_isi_error_ms": frame[
                    "absolute_first_isi_ms_error"
                ].median(),
                "median_absolute_late_isi_error_ms": frame[
                    "absolute_late_isi_ms_error"
                ].median(),
                "median_absolute_adaptation_ratio_error": frame[
                    "absolute_adaptation_ratio_error"
                ].median(),
                "median_spike_time_rmse_ms": frame[
                    "spike_time_rmse_ms"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def _selection_table(parameters: pd.DataFrame) -> pd.DataFrame:
    selected = parameters.loc[parameters["selected_by_training"]].copy()
    return (
        selected.groupby(["dataset", "family"])
        .size()
        .rename("cell_count")
        .reset_index()
    )


def _paired_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("legacy_inverse_memory", "current_spline"),
        ("current_spline", "spike_exponential"),
        ("current_spline", "izhikevich_recovery"),
        ("spike_exponential", "izhikevich_recovery"),
    )
    columns = (
        "absolute_spike_count_error",
        "absolute_first_isi_ms_error",
        "absolute_late_isi_ms_error",
        "absolute_adaptation_ratio_error",
        "spike_time_rmse_ms",
    )
    rows = []
    for dataset, frame in metrics.groupby("dataset"):
        for column in columns:
            paired = frame.pivot(
                index="cell_id",
                columns="model",
                values=column,
            )
            for first, second in comparisons:
                values = paired[[first, second]].dropna()
                difference = values[second] - values[first]
                try:
                    p_value = float(
                        wilcoxon(values[second], values[first]).pvalue
                    )
                except ValueError:
                    p_value = float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": column,
                        "first_model": first,
                        "second_model": second,
                        "cell_count": len(values),
                        "median_second_minus_first": difference.median(),
                        "second_better_fraction": (
                            difference < 0.0
                        ).mean(),
                        "equal_fraction": np.isclose(
                            difference,
                            0.0,
                        ).mean(),
                        "wilcoxon_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def _grouped_boxplot(
    axis,
    metrics: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
) -> None:
    positions = np.arange(len(MODELS), dtype=float)
    width = 0.28
    handles = []
    for dataset_index, dataset in enumerate(DATASETS):
        offset = (dataset_index - 0.5) * width
        values = [
            metrics.loc[
                metrics["dataset"].eq(dataset)
                & metrics["model"].eq(model),
                column,
            ]
            .dropna()
            .to_numpy()
            for model in MODELS
        ]
        box = axis.boxplot(
            values,
            positions=positions + offset,
            widths=width * 0.82,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[dataset])
            patch.set_alpha(0.7)
        handles.append(box["boxes"][0])
    axis.set_xticks(
        positions,
        [MODEL_LABELS[model] for model in MODELS],
        fontsize=8,
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.legend(
        handles,
        [dataset.replace("_", " ") for dataset in DATASETS],
        frameon=False,
        fontsize=8,
    )


def _population_plot(
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 8.2),
        constrained_layout=True,
    )
    specifications = (
        (
            "absolute_spike_count_error",
            "Held-out spike count",
            "Absolute count error",
        ),
        (
            "absolute_first_isi_ms_error",
            "First interspike interval",
            "Absolute error (ms)",
        ),
        (
            "absolute_late_isi_ms_error",
            "Late interspike interval",
            "Absolute error (ms)",
        ),
        (
            "absolute_adaptation_ratio_error",
            "Adaptation ratio",
            "Absolute ratio error",
        ),
        (
            "spike_time_rmse_ms",
            "Accumulated spike timing",
            "Matched-spike RMSE (ms)",
        ),
    )
    for axis, (column, title, ylabel) in zip(
        axes.flat[:5],
        specifications,
    ):
        _grouped_boxplot(axis, metrics, column, title, ylabel)

    selection = _selection_table(parameters)
    axis = axes[1, 2]
    families = (
        "current_spline",
        "spike_exponential",
        "izhikevich_recovery",
    )
    positions = np.arange(len(families))
    width = 0.36
    for dataset_index, dataset in enumerate(DATASETS):
        counts = [
            int(
                selection.loc[
                    selection["dataset"].eq(dataset)
                    & selection["family"].eq(family),
                    "cell_count",
                ].sum()
            )
            for family in families
        ]
        axis.bar(
            positions + (dataset_index - 0.5) * width,
            counts,
            width,
            color=COLORS[dataset],
            label=dataset.replace("_", " "),
        )
    axis.set_xticks(
        positions,
        [MODEL_LABELS[family] for family in families],
        fontsize=8,
    )
    axis.set_ylabel("Cells selected")
    axis.set_title("Training-only family selection", loc="left")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        (
            "Phase-template timing ladder: "
            f"{metrics['cell_id'].nunique()}-cell held-out-current test"
        ),
        fontsize=14,
    )
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--cohort-metrics",
        default=(
            "outputs/phase_template_population_100_stable_cycles/"
            "cell_model_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/phase_timing_population_100",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(
        args.cohort_metrics,
        dtype={"cell_id": "string"},
    )
    cohort = (
        cohort.loc[
            cohort["model"].eq("selected"),
            ["dataset", "cell_id", "nwb_path"],
        ]
        .drop_duplicates(["dataset", "cell_id"])
        .reset_index(drop=True)
    )
    if args.limit_per_dataset is not None:
        cohort = (
            cohort.groupby("dataset", sort=False)
            .head(args.limit_per_dataset)
            .reset_index(drop=True)
        )

    metrics: list[dict] = []
    candidates: list[dict] = []
    parameters: list[dict] = []
    failures: list[dict] = []
    if args.resume and (output_root / "timing_model_metrics.csv").exists():
        metrics = pd.read_csv(
            output_root / "timing_model_metrics.csv",
            dtype={"cell_id": "string"},
        ).to_dict("records")
        candidates = pd.read_csv(
            output_root / "timing_candidates.csv",
            dtype={"cell_id": "string"},
        ).to_dict("records")
        parameters = pd.read_csv(
            output_root / "timing_parameters.csv",
            dtype={"cell_id": "string"},
        ).to_dict("records")
        failure_path = output_root / "failures.csv"
        if failure_path.exists() and failure_path.stat().st_size > 1:
            failures = pd.read_csv(
                failure_path,
                dtype={"cell_id": "string"},
            ).to_dict("records")
    attempted = {
        (str(row["dataset"]), str(row["cell_id"]))
        for row in (*metrics, *failures)
    }
    tasks = [
        {
            "dataset": str(row.dataset),
            "cell_id": str(row.cell_id),
            "nwb_path": str(row.nwb_path),
        }
        for row in cohort.itertuples(index=False)
        if (str(row.dataset), str(row.cell_id)) not in attempted
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(tasks), args.batch_size):
            results = list(
                executor.map(
                    phase_timing_population_task,
                    tasks[start : start + args.batch_size],
                )
            )
            for result in results:
                if result["status"] != "ok":
                    failures.append(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "status"
                        }
                    )
                    continue
                metrics.extend(result["metrics"])
                candidates.extend(result["candidates"])
                parameters.extend(result["parameters"])
            _checkpoint(
                output_root,
                metrics,
                candidates,
                parameters,
                failures,
            )

    metric_frame = pd.DataFrame(metrics)
    parameter_frame = pd.DataFrame(parameters)
    summary = _summary_table(metric_frame)
    selection = _selection_table(parameter_frame)
    paired = _paired_comparison_table(metric_frame)
    _write_csv(summary, output_root / "timing_summary.csv")
    _write_csv(selection, output_root / "family_selection.csv")
    _write_csv(paired, output_root / "paired_comparisons.csv")
    _population_plot(
        metric_frame,
        parameter_frame,
        output_root / "timing_ladder_generalization.png",
    )
    metadata = {
        "cohort_metrics": args.cohort_metrics,
        "requested_cells": int(len(cohort)),
        "completed_cells": int(metric_frame["cell_id"].nunique()),
        "failed_cells": len(failures),
        "workers": args.workers,
        "models": list(MODELS),
        "selection": (
            "late-cycle validation on training currents with 0.01 NRMSE "
            "penalty per recovery-state input"
        ),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
