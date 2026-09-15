#!/usr/bin/env python3
"""Combine completed class-aware cohort outputs into one comparison."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs/class_aware_rrr",
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    summaries = []
    for path in root.glob("*/*/summary.json"):
        summaries.append(json.loads(path.read_text(encoding="ascii")))
    if not summaries:
        raise ValueError("No class-aware summary files found")
    summary = pd.DataFrame(summaries).sort_values(
        ["dataset", "cohort"]
    )
    _write_csv(summary, root / "class_aware_summary.csv")

    datasets = ("gouwens_visp", "scala_room_temperature")
    cohorts = ("all_eligible", "high_qc")
    colors = {
        "gouwens_visp": "#0072b2",
        "scala_room_temperature": "#d55e00",
    }
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.5),
        constrained_layout=True,
    )
    labels = []
    class_only = []
    combined = []
    bar_colors = []
    for dataset in datasets:
        for cohort in cohorts:
            row = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["cohort"].eq(cohort)
            ].iloc[0]
            labels.append(
                f"{dataset.split('_')[0]}\n"
                f"{cohort.replace('_', ' ')}"
            )
            class_only.append(row["class_only_nested_oof_r2"])
            combined.append(row["class_plus_rna_nested_oof_r2"])
            bar_colors.append(colors[dataset])
    positions = np.arange(len(labels))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        class_only,
        width,
        color="#9aa4aa",
    )
    axes[0].bar(
        positions + width / 2,
        combined,
        width,
        color=bar_colors,
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Nested donor-held-out R2")
    axes[0].set_title("Class-aware prediction", loc="left")
    axes[0].legend(
        handles=(
            Patch(color="#9aa4aa", label="class only"),
            Patch(
                color=colors["gouwens_visp"],
                label="Gouwens class + RNA",
            ),
            Patch(
                color=colors["scala_room_temperature"],
                label="Scala class + RNA",
            ),
        ),
        frameon=False,
        fontsize=8,
    )

    delta = summary.set_index(["dataset", "cohort"])[
        "delta_nested_oof_r2"
    ]
    for dataset in datasets:
        values = [delta.loc[(dataset, cohort)] for cohort in cohorts]
        offset = -0.18 if dataset == datasets[0] else 0.18
        axes[1].bar(
            np.arange(len(cohorts)) + offset,
            values,
            0.34,
            color=colors[dataset],
            label=dataset.replace("_", " "),
        )
    axes[1].set_xticks(
        np.arange(len(cohorts)),
        [cohort.replace("_", " ") for cohort in cohorts],
    )
    axes[1].set_ylabel("Delta nested R2 from RNA")
    axes[1].set_title("Signal remaining within class", loc="left")
    axes[1].legend(frameon=False, fontsize=8)

    parameters = []
    for dataset in datasets:
        path = (
            root
            / dataset
            / "all_eligible"
            / "parameter_nested_oof_performance.csv"
        )
        frame = pd.read_csv(path).set_index("parameter")
        for parameter in (
            "rheobase_current_pa",
            "timing_spike_jump",
            "timing_tau_ms",
        ):
            parameters.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "delta": frame.loc[parameter, "delta_oof_r2"],
                }
            )
    parameter_frame = pd.DataFrame(parameters)
    parameter_order = (
        "rheobase_current_pa",
        "timing_spike_jump",
        "timing_tau_ms",
    )
    for dataset in datasets:
        frame = parameter_frame.loc[
            parameter_frame["dataset"].eq(dataset)
        ].set_index("parameter")
        offset = -0.18 if dataset == datasets[0] else 0.18
        axes[2].bar(
            np.arange(len(parameter_order)) + offset,
            [frame.loc[name, "delta"] for name in parameter_order],
            0.34,
            color=colors[dataset],
            label=dataset.replace("_", " "),
        )
    axes[2].axhline(0.0, color="#999999", linewidth=0.8)
    axes[2].set_xticks(
        np.arange(len(parameter_order)),
        ("rheobase", "spike jump", "tau grid"),
    )
    axes[2].set_ylabel("Delta parameter R2 from RNA")
    axes[2].set_title("Timing-target specificity", loc="left")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Compact-model transcriptomics after class adjustment")
    figure.savefig(
        root / "class_aware_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
