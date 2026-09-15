#!/usr/bin/env python3
"""Run the recovery-plus-onset model on a family-stratified cohort."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from inverse_ephys_alpha_beta.family_overlays import (
    major_families,
    merge_parameters_and_metadata,
)
from inverse_ephys_alpha_beta.recovery_onset_population import (
    MODEL_NAMES,
    recovery_onset_population_task,
)


DATASETS = ("gouwens_visp", "scala_room_temperature")
MODEL_LABELS = {
    "spike_exponential": "spike memory",
    "recovery_timing": "recovery",
    "recovery_onset": "recovery + onset",
}
MODEL_COLORS = {
    "spike_exponential": "#777777",
    "recovery_timing": "#0072B2",
    "recovery_onset": "#D13C55",
}
METADATA_PATHS = {
    "gouwens_visp": Path(
        "outputs/class_aware_rrr/gouwens_visp/all_eligible/"
        "cell_metadata.csv"
    ),
    "scala_room_temperature": Path(
        "outputs/class_aware_rrr/scala_room_temperature/all_eligible/"
        "cell_metadata.csv"
    ),
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _build_cohort(
    parameters: pd.DataFrame,
    per_family: int,
    seed: int,
) -> pd.DataFrame:
    selected = []
    for dataset in DATASETS:
        metadata = pd.read_csv(
            METADATA_PATHS[dataset],
            dtype={"cell_id": "string"},
        )
        merged = merge_parameters_and_metadata(
            parameters,
            metadata,
            dataset,
        )
        for family in major_families(merged, minimum_cells=10):
            family_frame = merged.loc[
                merged["broad_class"].eq(family)
            ].sort_values("cell_id")
            selected.append(
                family_frame.sample(
                    n=min(per_family, len(family_frame)),
                    random_state=seed,
                )
            )
    return pd.concat(selected, ignore_index=True)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
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
                "median_onset_30ms_voltage_rmse_mv": frame[
                    "onset_30ms_voltage_rmse_mv"
                ].median(),
                "median_first_three_isi_rmse_ms": frame[
                    "first_three_isi_rmse_ms"
                ].median(),
                "median_absolute_spike_count_error": frame[
                    "absolute_spike_count_error"
                ].median(),
                "median_absolute_first_isi_error_ms": frame[
                    "absolute_first_isi_ms_error"
                ].median(),
                "median_absolute_late_isi_error_ms": frame[
                    "absolute_late_isi_ms_error"
                ].median(),
                "median_phase_chamfer": frame["phase_chamfer"].median(),
            }
        )
    return pd.DataFrame(rows)


def _paired(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "onset_30ms_voltage_rmse_mv",
        "first_three_isi_rmse_ms",
        "absolute_spike_count_error",
        "absolute_first_isi_ms_error",
        "absolute_late_isi_ms_error",
        "phase_chamfer",
    )
    rows = []
    for dataset, frame in metrics.groupby("dataset"):
        for column in columns:
            pivot = frame.pivot(
                index="cell_id",
                columns="model",
                values=column,
            )[
                ["spike_exponential", "recovery_onset"]
            ].dropna()
            difference = (
                pivot["recovery_onset"]
                - pivot["spike_exponential"]
            )
            try:
                p_value = float(
                    wilcoxon(
                        pivot["recovery_onset"],
                        pivot["spike_exponential"],
                    ).pvalue
                )
            except ValueError:
                p_value = float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "metric": column,
                    "cell_count": len(pivot),
                    "median_new_minus_baseline": difference.median(),
                    "new_better_fraction": (difference < 0.0).mean(),
                    "wilcoxon_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _plot(metrics: pd.DataFrame, destination: Path) -> None:
    specifications = (
        (
            "onset_30ms_voltage_rmse_mv",
            "First 30 ms voltage",
            "RMSE (mV)",
        ),
        (
            "first_three_isi_rmse_ms",
            "Initial burst: first three ISIs",
            "RMSE (ms)",
        ),
        (
            "absolute_spike_count_error",
            "Held-out spike count",
            "Absolute error",
        ),
        (
            "absolute_first_isi_ms_error",
            "First ISI",
            "Absolute error (ms)",
        ),
        (
            "absolute_late_isi_ms_error",
            "Late ISI",
            "Absolute error (ms)",
        ),
        (
            "phase_chamfer",
            "Stable phase loop",
            "Normalized Chamfer",
        ),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.0),
        constrained_layout=True,
    )
    dataset_offsets = {
        "gouwens_visp": -0.17,
        "scala_room_temperature": 0.17,
    }
    for axis, (column, title, ylabel) in zip(
        axes.flat,
        specifications,
    ):
        handles = []
        labels = []
        for model_index, model in enumerate(MODEL_NAMES):
            for dataset in DATASETS:
                values = (
                    metrics.loc[
                        metrics["model"].eq(model)
                        & metrics["dataset"].eq(dataset),
                        column,
                    ]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy()
                )
                box = axis.boxplot(
                    (values,),
                    positions=(
                        model_index + dataset_offsets[dataset],
                    ),
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={
                        "color": "#202020",
                        "linewidth": 1.1,
                    },
                )
                patch = box["boxes"][0]
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(
                    0.9 if dataset == "gouwens_visp" else 0.45
                )
                if model_index == 0:
                    handles.append(patch)
                    labels.append(
                        dataset.replace("_", " ")
                    )
        axis.set_xticks(
            np.arange(len(MODEL_NAMES)),
            [MODEL_LABELS[name] for name in MODEL_NAMES],
            fontsize=8,
        )
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(handles, labels, frameon=False, fontsize=7)
    figure.suptitle(
        "Continuous recovery and onset branch: held-out-current comparison",
        fontsize=14,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(
            "outputs/compact_population_all/compact_model_parameters.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recovery_onset_population"),
    )
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()
    output_root = args["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    parameters = pd.read_csv(
        args["parameters"],
        dtype={"cell_id": "string"},
    )
    cohort = _build_cohort(
        parameters,
        per_family=args["per_family"],
        seed=args["seed"],
    )
    _write_csv(cohort, output_root / "cohort.csv")
    tasks = cohort[
        ["dataset", "cell_id", "nwb_path"]
    ].to_dict("records")
    metrics = []
    fitted_parameters = []
    candidates = []
    failures = []
    metric_path = output_root / "model_metrics.csv"
    parameter_path = output_root / "model_parameters.csv"
    candidate_path = output_root / "timing_candidates.csv"
    failure_path = output_root / "failures.csv"
    if args["resume"] and metric_path.exists():
        metrics = pd.read_csv(
            metric_path,
            dtype={"cell_id": "string"},
        ).to_dict("records")
        fitted_parameters = pd.read_csv(
            parameter_path,
            dtype={"cell_id": "string"},
        ).to_dict("records")
        candidates = pd.read_csv(
            candidate_path,
            dtype={"cell_id": "string"},
        ).to_dict("records")
        try:
            failures = pd.read_csv(
                failure_path,
                dtype={"cell_id": "string"},
            ).to_dict("records")
        except pd.errors.EmptyDataError:
            failures = []
        completed = {
            (str(row["dataset"]), str(row["cell_id"]))
            for row in metrics
        }
        completed.update(
            (str(row["dataset"]), str(row["cell_id"]))
            for row in failures
        )
        tasks = [
            task
            for task in tasks
            if (str(task["dataset"]), str(task["cell_id"]))
            not in completed
        ]

    def checkpoint() -> None:
        _write_csv(pd.DataFrame(metrics), metric_path)
        _write_csv(pd.DataFrame(fitted_parameters), parameter_path)
        _write_csv(pd.DataFrame(candidates), candidate_path)
        _write_csv(pd.DataFrame(failures), failure_path)

    with ThreadPoolExecutor(max_workers=args["workers"]) as executor:
        for result_index, result in enumerate(
            executor.map(
            recovery_onset_population_task,
            tasks,
            ),
            start=1,
        ):
            if result["status"] == "ok":
                metrics.extend(result["metrics"])
                fitted_parameters.append(result["parameters"])
                candidates.extend(result["candidates"])
            else:
                failures.append(result)
            if result_index % args["checkpoint_every"] == 0:
                checkpoint()
    checkpoint()
    metric_frame = pd.DataFrame(metrics)
    _write_csv(
        _summary(metric_frame),
        output_root / "population_summary.csv",
    )
    _write_csv(
        _paired(metric_frame),
        output_root / "paired_comparisons.csv",
    )
    _plot(
        metric_frame,
        output_root / "population_comparison.png",
    )


if __name__ == "__main__":
    main()
