#!/usr/bin/env python3
"""Run a balanced held-out-current phase-template population screen."""

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

from inverse_ephys_alpha_beta.phase_population import (
    phase_population_task,
)


DATASETS = ("gouwens_visp", "scala_room_temperature")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint(
    output_root: Path,
    metrics: list[dict],
    candidates: list[dict],
    fourier: list[dict],
    failures: list[dict],
) -> None:
    _write_csv(
        pd.DataFrame(metrics),
        output_root / "cell_model_metrics.csv",
    )
    _write_csv(
        pd.DataFrame(candidates),
        output_root / "slow_state_candidates.csv",
    )
    _write_csv(
        pd.DataFrame(fourier),
        output_root / "fourier_parameters.csv",
    )
    _write_csv(
        pd.DataFrame(failures),
        output_root / "eligibility_failures.csv",
    )


def _summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["model"].eq("selected")].copy()
    phase_only = metrics.loc[metrics["model"].eq("phase_only")].copy()
    rows = []
    for dataset, frame in selected.groupby("dataset", sort=False):
        paired = phase_only.loc[
            phase_only["dataset"].eq(dataset)
        ].merge(
            frame,
            on=("dataset", "cell_id"),
            suffixes=("_phase", "_selected"),
        )
        rows.append(
            {
                "dataset": dataset,
                "cell_count": len(frame),
                "slow_state_fraction": float(
                    frame["slow_selected"].mean()
                ),
                "median_phase_chamfer": float(
                    frame["phase_chamfer"].median()
                ),
                "phase_chamfer_q90": float(
                    frame["phase_chamfer"].quantile(0.9)
                ),
                "median_phase_only_phase_chamfer": float(
                    paired["phase_chamfer_phase"].median()
                ),
                "median_absolute_spike_count_error": float(
                    frame["absolute_spike_count_error"].median()
                ),
                "median_phase_only_absolute_spike_count_error": float(
                    paired[
                        "absolute_spike_count_error_phase"
                    ].median()
                ),
                "median_relative_spike_count_error": float(
                    frame["relative_spike_count_error"].median()
                ),
                "spike_count_within_one_fraction": float(
                    (
                        frame["absolute_spike_count_error"] <= 1
                    ).mean()
                ),
                "selected_count_error_improvement_fraction": float(
                    (
                        paired[
                            "absolute_spike_count_error_selected"
                        ]
                        < paired[
                            "absolute_spike_count_error_phase"
                        ]
                    ).mean()
                ),
                "median_max_dvdt_ratio": float(
                    np.median(
                        frame["predicted_max_dvdt_mv_ms"]
                        / frame["observed_max_dvdt_mv_ms"]
                    )
                ),
                "median_min_dvdt_ratio": float(
                    np.median(
                        frame["predicted_min_dvdt_mv_ms"]
                        / frame["observed_min_dvdt_mv_ms"]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _population_plot(metrics: pd.DataFrame, path: Path) -> None:
    selected = metrics.loc[metrics["model"].eq("selected")].copy()
    no_slow = metrics.loc[metrics["model"].eq("phase_only")].copy()
    datasets = [
        dataset
        for dataset in DATASETS
        if dataset in set(selected["dataset"])
    ]
    colors = {
        "gouwens_visp": "#0072b2",
        "scala_room_temperature": "#d55e00",
    }
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.0),
        constrained_layout=True,
    )

    values = [
        selected.loc[
            selected["dataset"].eq(dataset),
            "phase_chamfer",
        ].to_numpy()
        for dataset in datasets
    ]
    parts = axes[0, 0].violinplot(
        values,
        showmedians=True,
        showextrema=False,
    )
    for body, dataset in zip(parts["bodies"], datasets):
        body.set_facecolor(colors[dataset])
        body.set_alpha(0.65)
    axes[0, 0].set_xticks(
        np.arange(1, len(datasets) + 1),
        [dataset.replace("_", "\n") for dataset in datasets],
        fontsize=8,
    )
    axes[0, 0].set_ylabel("Normalized phase-plane Chamfer")
    axes[0, 0].set_title("Held-out cycle geometry", loc="left")

    for dataset in datasets:
        frame = selected.loc[selected["dataset"].eq(dataset)]
        axes[0, 1].scatter(
            frame["observed_spike_count"],
            frame["predicted_spike_count"],
            s=22,
            alpha=0.7,
            color=colors[dataset],
            label=dataset.replace("_", " "),
        )
    limit = float(
        max(
            selected["observed_spike_count"].max(),
            selected["predicted_spike_count"].max(),
        )
    )
    axes[0, 1].plot((0, limit), (0, limit), "--", color="#555555")
    axes[0, 1].set_xlabel("Observed spike count")
    axes[0, 1].set_ylabel("Predicted spike count")
    axes[0, 1].set_title("Held-out firing count", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8)

    paired = no_slow.merge(
        selected,
        on=("dataset", "cell_id"),
        suffixes=("_phase", "_selected"),
    )
    axes[0, 2].scatter(
        paired["absolute_spike_count_error_phase"],
        paired["absolute_spike_count_error_selected"],
        c=[
            colors[dataset] for dataset in paired["dataset"]
        ],
        s=22,
        alpha=0.7,
    )
    error_limit = float(
        max(
            1.0,
            paired["absolute_spike_count_error_phase"].max(),
            paired["absolute_spike_count_error_selected"].max(),
        )
    )
    axes[0, 2].plot(
        (0, error_limit),
        (0, error_limit),
        "--",
        color="#555555",
    )
    axes[0, 2].set_xlabel("Phase-only absolute count error")
    axes[0, 2].set_ylabel("Selected-model absolute count error")
    axes[0, 2].set_title("Does slow memory help?", loc="left")

    tau_table = (
        selected.assign(
            tau_label=selected["slow_tau_ms"]
            .fillna(0.0)
            .map(lambda value: "none" if value == 0 else f"{value:g}")
        )
        .groupby(["dataset", "tau_label"])
        .size()
        .unstack(fill_value=0)
    )
    tau_labels = [
        label
        for label in ("none", "20", "50", "100", "200", "500")
        if label in tau_table.columns
    ]
    x = np.arange(len(tau_labels))
    width = 0.8 / max(1, len(datasets))
    for index, dataset in enumerate(datasets):
        counts = (
            tau_table.reindex(index=[dataset], fill_value=0)
            .reindex(columns=tau_labels, fill_value=0)
            .iloc[0]
            .to_numpy()
        )
        axes[1, 0].bar(
            x + (index - (len(datasets) - 1) / 2) * width,
            counts,
            width,
            color=colors[dataset],
            label=dataset.replace("_", " "),
        )
    axes[1, 0].set_xticks(x, tau_labels)
    axes[1, 0].set_xlabel("Selected tau (ms; none = phase only)")
    axes[1, 0].set_ylabel("Cell count")
    axes[1, 0].set_title("Slow-state selection", loc="left")

    for axis, observed, predicted, title in (
        (
            axes[1, 1],
            "observed_max_dvdt_mv_ms",
            "predicted_max_dvdt_mv_ms",
            "Maximum dV/dt",
        ),
        (
            axes[1, 2],
            "observed_min_dvdt_mv_ms",
            "predicted_min_dvdt_mv_ms",
            "Minimum dV/dt",
        ),
    ):
        for dataset in datasets:
            frame = selected.loc[selected["dataset"].eq(dataset)]
            axis.scatter(
                frame[observed],
                frame[predicted],
                s=22,
                alpha=0.7,
                color=colors[dataset],
            )
        values = np.concatenate(
            (
                selected[observed].to_numpy(),
                selected[predicted].to_numpy(),
            )
        )
        lower, upper = np.nanquantile(values, (0.01, 0.99))
        padding = max(1.0, 0.04 * (upper - lower))
        axis.plot(
            (lower, upper),
            (lower, upper),
            "--",
            color="#555555",
        )
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Observed (mV/ms)")
        axis.set_ylabel("Predicted (mV/ms)")
        axis.set_title(title, loc="left")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Landmark phase-template population generalization",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-spike-cycles",
        default="outputs/local_patchseq_spike_cycles.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/phase_template_population_100",
    )
    parser.add_argument("--cells-per-dataset", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        args.raw_spike_cycles,
        dtype={"cell_id": "string"},
    )
    raw = raw.loc[raw["raw_status"].eq("ok")].copy()

    metrics: list[dict] = []
    candidates: list[dict] = []
    fourier: list[dict] = []
    failures: list[dict] = []
    if args.resume and (output_root / "cell_model_metrics.csv").exists():
        metrics = pd.read_csv(
            output_root / "cell_model_metrics.csv"
        ).to_dict("records")
        candidates = pd.read_csv(
            output_root / "slow_state_candidates.csv"
        ).to_dict("records")
        fourier = pd.read_csv(
            output_root / "fourier_parameters.csv"
        ).to_dict("records")
        failures_path = output_root / "eligibility_failures.csv"
        if failures_path.exists() and failures_path.stat().st_size:
            failures = pd.read_csv(failures_path).to_dict("records")
    attempted = {
        (str(row["dataset"]), str(row["cell_id"]))
        for row in (*metrics, *failures)
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for dataset_index, dataset in enumerate(DATASETS):
            accepted = {
                str(row["cell_id"])
                for row in metrics
                if row["dataset"] == dataset
            }
            frame = (
                raw.loc[raw["dataset"].eq(dataset)]
                .sample(
                    frac=1.0,
                    random_state=args.seed + dataset_index,
                )
                .reset_index(drop=True)
            )
            cursor = 0
            while (
                len(accepted) < args.cells_per_dataset
                and cursor < len(frame)
            ):
                tasks = []
                while (
                    len(tasks) < args.batch_size
                    and cursor < len(frame)
                ):
                    row = frame.iloc[cursor]
                    cursor += 1
                    key = (dataset, str(row["cell_id"]))
                    if key in attempted:
                        continue
                    tasks.append(
                        {
                            "dataset": dataset,
                            "cell_id": str(row["cell_id"]),
                            "nwb_path": str(row["nwb_path"]),
                        }
                    )
                    attempted.add(key)
                if not tasks:
                    continue
                results = list(
                    executor.map(phase_population_task, tasks)
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
                    cell_id = str(result["cell_id"])
                    if len(accepted) >= args.cells_per_dataset:
                        failures.append(
                            {
                                "dataset": dataset,
                                "cell_id": cell_id,
                                "nwb_path": result["nwb_path"],
                                "reason": (
                                    "eligible_after_dataset_quota"
                                ),
                            }
                        )
                        continue
                    accepted.add(cell_id)
                    metrics.extend(result["metrics"])
                    candidates.extend(result["candidates"])
                    fourier.extend(result["fourier"])
                _checkpoint(
                    output_root,
                    metrics,
                    candidates,
                    fourier,
                    failures,
                )

    metric_frame = pd.DataFrame(metrics)
    summary = _summary_table(metric_frame)
    _write_csv(summary, output_root / "population_summary.csv")
    _population_plot(
        metric_frame,
        output_root / "population_generalization.png",
    )
    metadata = {
        "requested_cells_per_dataset": args.cells_per_dataset,
        "workers": args.workers,
        "seed": args.seed,
        "accepted_cells": {
            dataset: int(
                metric_frame.loc[
                    metric_frame["dataset"].eq(dataset),
                    "cell_id",
                ].nunique()
            )
            for dataset in DATASETS
        },
        "attempted_cells": len(attempted),
        "selection_rule": (
            "first eligible cells in deterministic shuffled order; "
            "eligibility requires a held-out repetitive current bracketed "
            "by repetitive training currents"
        ),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
