#!/usr/bin/env python3
"""Fit the final compact phase model to every eligible local Patch-seq cell."""

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

from inverse_ephys_alpha_beta.compact_population import (
    compact_population_task,
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
    candidates: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    _write_csv(
        pd.DataFrame(parameters),
        output_root / "compact_model_parameters.csv",
    )
    _write_csv(
        pd.DataFrame(candidates),
        output_root / "timing_candidates.csv",
    )
    _write_csv(
        pd.DataFrame(failures),
        output_root / "failures.csv",
    )


def _summary(parameters: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parameter_datasets = (
        set(parameters["dataset"]) if "dataset" in parameters else set()
    )
    failure_datasets = (
        set(failures["dataset"]) if "dataset" in failures else set()
    )
    for dataset in sorted(
        parameter_datasets | failure_datasets
    ):
        fitted = (
            parameters.loc[parameters["dataset"].eq(dataset)]
            if "dataset" in parameters
            else parameters
        )
        failed = (
            failures.loc[failures["dataset"].eq(dataset)]
            if "dataset" in failures
            else failures
        )
        rows.append(
            {
                "dataset": dataset,
                "fitted_cells": fitted["cell_id"].nunique(),
                "failed_cells": (
                    failed["cell_id"].nunique()
                    if "cell_id" in failed
                    else 0
                ),
                "median_validation_phase_chamfer": fitted[
                    "validation_phase_chamfer"
                ].median(),
                "median_validation_absolute_spike_count_error": fitted[
                    "validation_absolute_spike_count_error"
                ].median(),
                "median_validation_relative_spike_count_error": fitted[
                    "validation_relative_spike_count_error"
                ].median(),
                "median_validation_spike_time_rmse_ms": fitted[
                    "validation_spike_time_rmse_ms"
                ].median(),
                "median_timing_late_period_nrmse": fitted[
                    "timing_late_period_nrmse"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def _qc_plot(parameters: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.5),
        constrained_layout=True,
    )
    specifications = (
        (
            "validation_phase_chamfer",
            "Held-out phase-loop distance",
            "Normalized Chamfer distance",
        ),
        (
            "validation_absolute_spike_count_error",
            "Held-out spike count",
            "Absolute count error",
        ),
        (
            "validation_spike_time_rmse_ms",
            "Held-out accumulated timing",
            "Matched-spike RMSE (ms)",
        ),
        (
            "timing_late_period_nrmse",
            "Within-current late periods",
            "Normalized RMSE",
        ),
    )
    for axis, (column, title, ylabel) in zip(
        axes.flat,
        specifications,
    ):
        datasets = list(parameters["dataset"].drop_duplicates())
        values = [
            parameters.loc[
                parameters["dataset"].eq(dataset),
                column,
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
            for dataset in datasets
        ]
        violin = axis.violinplot(
            values,
            positions=np.arange(len(datasets)),
            showmedians=True,
            showextrema=False,
        )
        for body, dataset in zip(violin["bodies"], datasets):
            body.set_facecolor(
                DATASET_COLORS.get(dataset, "#666666")
            )
            body.set_edgecolor("none")
            body.set_alpha(0.75)
        axis.set_xticks(
            np.arange(len(datasets)),
            [dataset.replace("_", " ") for dataset in datasets],
            rotation=12,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Compact-model QC across {parameters['cell_id'].nunique()} cells"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="outputs/local_patchseq_nwb_inventory.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/compact_population_all",
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
    if args.limit_per_dataset is not None:
        inventory = (
            inventory.groupby("dataset", sort=False)
            .head(args.limit_per_dataset)
            .reset_index(drop=True)
        )

    parameters: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    parameter_path = output_root / "compact_model_parameters.csv"
    candidate_path = output_root / "timing_candidates.csv"
    failure_path = output_root / "failures.csv"
    if args.resume and parameter_path.exists():
        parameters = pd.read_csv(
            parameter_path,
            dtype={"cell_id": "string"},
        ).to_dict("records")
        if candidate_path.exists() and candidate_path.stat().st_size > 1:
            candidates = pd.read_csv(
                candidate_path,
                dtype={"cell_id": "string"},
            ).to_dict("records")
        if failure_path.exists() and failure_path.stat().st_size > 1:
            failures = pd.read_csv(
                failure_path,
                dtype={"cell_id": "string"},
            ).to_dict("records")
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
                    compact_population_task,
                    tasks[start : start + args.batch_size],
                )
            )
            for result in results:
                if result["status"] == "ok":
                    parameters.append(result["parameters"])
                    candidates.extend(result["candidates"])
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
                candidates,
                failures,
            )
            print(
                f"Attempted {min(start + args.batch_size, len(tasks))}/"
                f"{len(tasks)} new cells; {len(parameters)} fitted, "
                f"{len(failures)} failed",
                flush=True,
            )

    parameter_frame = pd.DataFrame(parameters)
    failure_frame = pd.DataFrame(failures)
    summary = _summary(parameter_frame, failure_frame)
    _write_csv(summary, output_root / "fit_summary.csv")
    if not parameter_frame.empty:
        _qc_plot(
            parameter_frame,
            output_root / "compact_model_qc.png",
        )
    metadata = {
        "inventory": args.inventory,
        "requested_cells": int(len(inventory)),
        "fitted_cells": int(len(parameter_frame)),
        "failed_cells": int(len(failure_frame)),
        "workers": args.workers,
        "executor": args.executor,
        "model": "phase template + current spline + spike exponential memory",
        "fit_strategy": (
            "held-out interior current for QC, followed by refit on all "
            "usable spiking current levels"
        ),
        "fourier_harmonics": 10,
        "current_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
