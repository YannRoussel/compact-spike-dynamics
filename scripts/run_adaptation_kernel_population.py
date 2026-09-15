#!/usr/bin/env python3
"""Compare single-tau and fixed-kernel timing across Patch-seq cells."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.adaptation_kernel_population import (
    adaptation_kernel_population_task,
)


DATASET_COLORS = {
    "gouwens_visp": "#0072b2",
    "scala_room_temperature": "#d55e00",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint(
    output_root: Path,
    parameters: list[dict[str, object]],
    metrics: list[dict[str, object]],
    reliability: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    _write_csv(
        pd.DataFrame(parameters),
        output_root / "kernel_model_parameters.csv",
    )
    _write_csv(
        pd.DataFrame(metrics),
        output_root / "validation_metrics.csv",
    )
    _write_csv(
        pd.DataFrame(reliability),
        output_root / "current_reliability.csv",
    )
    _write_csv(
        pd.DataFrame(failures),
        output_root / "failures.csv",
    )


def _paired_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    wide = metrics.pivot_table(
        index=("dataset", "cell_id"),
        columns="model",
        values=(
            "absolute_spike_count_error",
            "spike_time_rmse_ms",
            "absolute_first_isi_ms_error",
            "absolute_late_isi_ms_error",
            "absolute_adaptation_ratio_error",
        ),
        aggfunc="first",
    )
    rows = []
    for dataset in metrics["dataset"].drop_duplicates():
        subset = wide.loc[dataset]
        for metric in wide.columns.levels[0]:
            single = subset[(metric, "spike_exponential")]
            kernel = subset[(metric, "fixed_multiscale_kernel")]
            finite = np.isfinite(single) & np.isfinite(kernel)
            difference = kernel[finite] - single[finite]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "paired_cells": int(np.sum(finite)),
                    "single_median": float(np.median(single[finite])),
                    "kernel_median": float(np.median(kernel[finite])),
                    "median_kernel_minus_single": float(
                        np.median(difference)
                    ),
                    "kernel_better_fraction": float(
                        np.mean(difference < 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reliability_summary(
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, frame in reliability.groupby("dataset"):
        for lag in (50, 200, 800):
            full = frame[f"full_kernel_value_{lag:04d}ms"].to_numpy(
                dtype=float
            )
            current = frame[
                f"current_kernel_value_{lag:04d}ms"
            ].to_numpy(dtype=float)
            finite = np.isfinite(full) & np.isfinite(current)
            rows.append(
                {
                    "dataset": dataset,
                    "measure": f"kernel_{lag}ms",
                    "comparison_count": int(np.sum(finite)),
                    "correlation": float(
                        np.corrcoef(full[finite], current[finite])[0, 1]
                    ),
                    "median_absolute_error": float(
                        np.median(np.abs(full[finite] - current[finite]))
                    ),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "measure": "single_tau_grid",
                "comparison_count": len(frame),
                "correlation": float(
                    np.mean(frame["single_tau_agrees"].astype(float))
                ),
                "median_absolute_error": float(
                    np.median(
                        np.abs(
                            np.log(
                                frame["full_single_tau_ms"].astype(float)
                            )
                            - np.log(
                                frame[
                                    "current_single_tau_ms"
                                ].astype(float)
                            )
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_plot(
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    reliability: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.5),
        constrained_layout=True,
    )
    plot_metrics = (
        "absolute_spike_count_error",
        "spike_time_rmse_ms",
        "absolute_late_isi_ms_error",
        "absolute_adaptation_ratio_error",
    )
    wide = metrics.pivot_table(
        index=("dataset", "cell_id"),
        columns="model",
        values=plot_metrics,
        aggfunc="first",
    )
    for axis, metric in zip(axes.flat, plot_metrics):
        for dataset in parameters["dataset"].drop_duplicates():
            subset = wide.loc[dataset]
            single = subset[(metric, "spike_exponential")]
            kernel = subset[(metric, "fixed_multiscale_kernel")]
            finite = np.isfinite(single) & np.isfinite(kernel)
            axis.scatter(
                single[finite],
                kernel[finite],
                s=9,
                alpha=0.35,
                color=DATASET_COLORS.get(dataset, "#666666"),
                edgecolors="none",
                label=dataset.replace("_", " "),
            )
        visible = np.concatenate(
            [
                wide[(metric, model)]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
                for model in (
                    "spike_exponential",
                    "fixed_multiscale_kernel",
                )
            ]
        )
        upper = float(np.quantile(visible, 0.98)) if len(visible) else 1.0
        axis.plot((0.0, upper), (0.0, upper), color="#999999", linewidth=1)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_xlabel("Single exponential")
        axis.set_ylabel("Fixed kernel")
        title = metric.replace("_", " ")
        axis.set_title(title, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Held-out timing: points below the diagonal favor the fixed kernel"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
        constrained_layout=True,
    )
    for dataset, frame in parameters.groupby("dataset"):
        axes[0].hist(
            np.log10(frame["single_tau_ms"].astype(float)),
            bins=np.linspace(np.log10(20), np.log10(1000), 13),
            histtype="step",
            linewidth=2,
            color=DATASET_COLORS.get(dataset, "#666666"),
            label=dataset.replace("_", " "),
        )
        axes[1].hist(
            frame["single_tau_relative_score_margin"]
            .clip(upper=2.0),
            bins=np.linspace(0.0, 2.0, 25),
            histtype="step",
            linewidth=2,
            color=DATASET_COLORS.get(dataset, "#666666"),
        )
        subset = reliability.loc[reliability["dataset"].eq(dataset)]
        axes[2].bar(
            dataset.replace("_", "\n"),
            subset.loc[
                subset["measure"].eq("single_tau_grid"),
                "correlation",
            ].iloc[0],
            color=DATASET_COLORS.get(dataset, "#666666"),
        )
    axes[0].set_xlabel("log10 selected tau (ms)")
    axes[0].set_ylabel("Cells")
    axes[0].set_title("Grid-point distribution", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Relative score margin")
    axes[1].set_ylabel("Cells")
    axes[1].set_title("Tau identifiability", loc="left")
    axes[2].set_ylabel("Per-current agreement fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Tau stability across currents", loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(
        path.with_name("adaptation_identifiability.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="outputs/local_patchseq_nwb_inventory.csv",
    )
    parser.add_argument(
        "--eligible-parameters",
        default=(
            "outputs/compact_population_all/"
            "compact_model_parameters.csv"
        ),
        help=(
            "Optional compact-model table used to retain cells that already "
            "passed the repetitive-spiking protocol filter"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/adaptation_kernel_population",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(
        args.inventory,
        dtype={"cell_id": "string"},
    ).drop_duplicates(["dataset", "cell_id"])
    if args.eligible_parameters:
        eligible = pd.read_csv(
            args.eligible_parameters,
            dtype={"cell_id": "string"},
            usecols=("dataset", "cell_id"),
        ).drop_duplicates()
        inventory = inventory.merge(
            eligible,
            on=("dataset", "cell_id"),
            how="inner",
        )
    if args.limit_per_dataset is not None:
        inventory = (
            inventory.groupby("dataset", sort=False)
            .head(args.limit_per_dataset)
            .reset_index(drop=True)
        )
    parameters: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    paths = {
        "parameters": output_root / "kernel_model_parameters.csv",
        "metrics": output_root / "validation_metrics.csv",
        "reliability": output_root / "current_reliability.csv",
        "failures": output_root / "failures.csv",
    }
    if args.resume and paths["parameters"].exists():
        parameters = pd.read_csv(
            paths["parameters"],
            dtype={"cell_id": "string"},
        ).to_dict("records")
        for name, target in (
            ("metrics", metrics),
            ("reliability", reliability),
            ("failures", failures),
        ):
            if paths[name].exists() and paths[name].stat().st_size > 1:
                target.extend(
                    pd.read_csv(
                        paths[name],
                        dtype={"cell_id": "string"},
                    ).to_dict("records")
                )
    attempted = {
        (str(row["dataset"]), str(row["cell_id"]))
        for row in (*parameters, *failures)
    }
    tasks = [
        {
            "dataset": str(row.dataset),
            "cell_id": str(row.cell_id),
            "nwb_path": str(row.nwb_path),
        }
        for row in inventory.itertuples(index=False)
        if (str(row.dataset), str(row.cell_id)) not in attempted
    ]
    executor_type = (
        ProcessPoolExecutor
        if args.executor == "process"
        else ThreadPoolExecutor
    )
    with executor_type(max_workers=args.workers) as executor:
        for start in range(0, len(tasks), args.batch_size):
            results = list(
                executor.map(
                    adaptation_kernel_population_task,
                    tasks[start : start + args.batch_size],
                )
            )
            for result in results:
                if result["status"] == "ok":
                    parameters.append(result["parameters"])
                    metrics.extend(result["metrics"])
                    reliability.extend(result["reliability"])
                else:
                    failures.append(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "status"
                        }
                    )
            _checkpoint(
                output_root,
                parameters,
                metrics,
                reliability,
                failures,
            )
            print(
                f"Attempted {min(start + args.batch_size, len(tasks))}/"
                f"{len(tasks)} new cells; {len(parameters)} fitted, "
                f"{len(failures)} failed",
                flush=True,
            )
    parameter_frame = pd.DataFrame(parameters)
    metric_frame = pd.DataFrame(metrics)
    reliability_frame = pd.DataFrame(reliability)
    paired = _paired_summary(metric_frame)
    reliability_summary = _reliability_summary(reliability_frame)
    _write_csv(paired, output_root / "paired_model_summary.csv")
    _write_csv(
        reliability_summary,
        output_root / "reliability_summary.csv",
    )
    _summary_plot(
        parameter_frame,
        metric_frame,
        paired,
        reliability_summary,
        output_root / "adaptation_kernel_comparison.png",
    )
    (output_root / "metadata.json").write_text(
        json.dumps(
            {
                "inventory": args.inventory,
                "eligible_parameters": args.eligible_parameters,
                "requested_cells": len(inventory),
                "fitted_cells": len(parameter_frame),
                "failed_cells": len(failures),
                "kernel_tau_basis_ms": [25, 100, 400, 1600],
                "kernel_summary_lags_ms": [50, 200, 800],
                "comparison": (
                    "single fitted tau grid versus fixed multiscale "
                    "spike-history kernel"
                ),
            },
            indent=2,
        ),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
