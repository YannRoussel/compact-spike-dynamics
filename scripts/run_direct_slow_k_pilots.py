#!/usr/bin/env python3
"""Fit one direct slow-K state to the four guarded fast-channel pilots."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_targets import (
    build_cell_optimization_target,
    load_raw_spike_cycle_table,
)
from inverse_ephys_alpha_beta.direct_slow_k import (
    DirectSlowKFitConfig,
    direct_slow_k_summary,
    load_direct_parent,
    optimize_direct_slow_k,
    save_direct_slow_k_result,
)
from inverse_ephys_alpha_beta.phase_shape import PhaseShapeConfig
from inverse_ephys_alpha_beta.protocols import biological_screen_config


PILOTS = (
    (
        "gouwens_704047023",
        "gouwens_visp",
        "704047023",
        34.0,
        "gouwens",
    ),
    (
        "gouwens_674495385",
        "gouwens_visp",
        "674495385",
        34.0,
        "gouwens",
    ),
    (
        "scala_20180920_sample_1",
        "scala_room_temperature",
        "20180920_sample_1",
        22.0,
        "scala",
    ),
    (
        "scala_20190425_sample_4",
        "scala_room_temperature",
        "20190425_sample_4",
        22.0,
        "scala",
    ),
)
SHORT_NAMES = {
    "gouwens_704047023": "G704",
    "gouwens_674495385": "G674",
    "scala_20180920_sample_1": "S20180920",
    "scala_20190425_sample_4": "S20190425",
}


def _cross_cell_plot(summary: pd.DataFrame, path: Path) -> None:
    metrics = (
        "shape_loss",
        "physical_constraint_loss",
        "firing_pattern_loss",
        "rheobase_absolute_error_pa",
    )
    labels = (
        "Phase shape",
        "Physical constraints",
        "Spike train",
        "Rheobase error (pA)",
    )
    stages = ("frozen_fast_parent", "direct_slow_k")
    colors = ("#e76f51", "#457b9d")
    pilots = summary["pilot"].drop_duplicates().tolist()
    short_names = [SHORT_NAMES[pilot] for pilot in pilots]
    x = np.arange(len(pilots))
    width = 0.34
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.5),
        constrained_layout=True,
    )
    for axis, metric, label in zip(axes.flat, metrics, labels):
        for index, (stage, color) in enumerate(zip(stages, colors)):
            values = (
                summary.loc[
                    summary["stage"].eq(stage),
                    ["pilot", metric],
                ]
                .set_index("pilot")
                .reindex(pilots)[metric]
                .to_numpy(dtype=float)
            )
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                color=color,
                label=stage.replace("_", " ").title(),
            )
        axis.set_xticks(x, short_names)
        axis.set_ylabel("Loss" if "error" not in metric else "pA")
        axis.set_title(label, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Frozen direct fast channels versus one direct slow-K state",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _spike_count_plot(features: pd.DataFrame, path: Path) -> None:
    counts = features.loc[features["feature"].eq("spike_count")].copy()
    protocols = ("rheobase", "suprathreshold", "held_out")
    pilots = counts["pilot"].drop_duplicates().tolist()
    short_names = [SHORT_NAMES[pilot] for pilot in pilots]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.2),
        constrained_layout=True,
    )
    x = np.arange(len(pilots))
    width = 0.24
    series = (
        ("biological", "#147d7e"),
        ("frozen_fast_parent", "#e76f51"),
        ("direct_slow_k", "#457b9d"),
    )
    for axis, protocol in zip(axes, protocols):
        frame = (
            counts.loc[counts["protocol"].eq(protocol)]
            .set_index("pilot")
            .reindex(pilots)
        )
        for index, (column, color) in enumerate(series):
            axis.bar(
                x + (index - 1) * width,
                frame[column].to_numpy(dtype=float),
                width,
                color=color,
                label=column.replace("_", " ").title(),
            )
        axis.set_xticks(x, short_names, rotation=20)
        axis.set_title(protocol.replace("_", " ").title(), loc="left")
        axis.set_ylabel("Spikes in fitted window")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Does the added slow-K state correct spike counts?",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-spike-cycles",
        default="outputs/local_patchseq_spike_cycles.csv",
    )
    parser.add_argument(
        "--direct-root",
        default="outputs/direct_phase_pilots_rheobase_guarded",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/direct_slow_k_pilots",
    )
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration-ms", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = load_raw_spike_cycle_table(args.raw_spike_cycles)
    direct_root = Path(args.direct_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(
        direct_root / "guarded_selected_key_metrics.csv"
    ).set_index("pilot")
    summaries = []
    feature_tables = []
    pilots_to_run = PILOTS[
        : args.limit if args.limit is not None else len(PILOTS)
    ]
    for index, (
        pilot,
        dataset,
        cell_id,
        temperature,
        preset,
    ) in enumerate(pilots_to_run):
        output = output_root / pilot
        summary_path = output / "stage_summary.csv"
        feature_path = output / "firing_pattern_comparison.csv"
        if (
            args.resume
            and summary_path.exists()
            and feature_path.exists()
        ):
            summary = pd.read_csv(summary_path)
            features = pd.read_csv(feature_path)
        else:
            target = build_cell_optimization_target(
                raw,
                dataset=dataset,
                cell_id=cell_id,
                temperature_c=temperature,
                local_passive=True,
            )
            parent_stage = str(selected.loc[pilot, "stage"])
            base_model = load_direct_parent(
                direct_root / pilot / "stage_parameters.csv",
                parent_stage,
                target,
            )
            screen_config = replace(
                biological_screen_config(
                    preset,
                    temperature_c=temperature,
                    dt_ms=args.dt,
                ),
                rheobase_tolerance_pa=2.0,
            )
            fit_config = DirectSlowKFitConfig(
                population_size=args.population_size,
                generations=args.generations,
                workers=args.workers,
                seed=args.seed + 1000 * index,
                train_duration_ms=args.duration_ms,
            )
            result = optimize_direct_slow_k(
                target,
                base_model,
                screen_config,
                fit_config,
                PhaseShapeConfig(),
            )
            save_direct_slow_k_result(
                result,
                screen_config,
                output,
            )
            summary = direct_slow_k_summary(result)
            features = pd.read_csv(feature_path)
            summary.insert(0, "parent_stage", parent_stage)
            summary.to_csv(summary_path, index=False)
        summary.insert(0, "pilot", pilot)
        features.insert(0, "pilot", pilot)
        summaries.append(summary)
        feature_tables.append(features)
        slow = summary.loc[
            summary["stage"].eq("direct_slow_k")
        ].iloc[0]
        print(
            f"{pilot}: total={slow['objective_total']:.3f}, "
            f"phase={slow['shape_loss']:.3f}, "
            f"physical={slow['physical_constraint_loss']:.3f}, "
            f"train={slow['firing_pattern_loss']:.3f}, "
            f"count={slow['spike_count_loss']:.3f}, "
            f"rheobase={slow['model_rheobase_pa']:.1f} pA"
        )

    combined = pd.concat(summaries, ignore_index=True)
    combined["rheobase_absolute_error_pa"] = (
        combined["model_rheobase_pa"]
        - combined["biological_rheobase_pa"]
    ).abs()
    combined.to_csv(
        output_root / "combined_stage_summary.csv",
        index=False,
    )
    combined.loc[
        :,
        [
            "pilot",
            "parent_stage",
            "stage",
            "objective_total",
            "shape_loss",
            "physical_constraint_loss",
            "firing_pattern_loss",
            "spike_count_loss",
            "model_rheobase_pa",
            "biological_rheobase_pa",
            "rheobase_absolute_error_pa",
            "rheobase_constraint_loss",
            "extension__gslow_ms_cm2",
            "max_dvdt_residual_mv_ms",
            "min_dvdt_residual_mv_ms",
            "peak_voltage_residual_mv",
            "minimum_voltage_residual_mv",
        ],
    ].to_csv(
        output_root / "slow_k_key_metrics.csv",
        index=False,
    )
    mechanism_rows = []
    for pilot, group in combined.groupby("pilot", sort=False):
        parent = group.loc[
            group["stage"].eq("frozen_fast_parent")
        ].iloc[0]
        slow = group.loc[
            group["stage"].eq("direct_slow_k")
        ].iloc[0]
        train_change = (
            slow["firing_pattern_loss"]
            / max(parent["firing_pattern_loss"], 1e-12)
            - 1.0
        )
        count_change = (
            slow["spike_count_loss"]
            / max(parent["spike_count_loss"], 1e-12)
            - 1.0
        )
        meaningful_train_improvement = min(
            train_change,
            count_change,
        ) <= -0.05
        accept_slow_k = bool(
            slow["objective_total"] < parent["objective_total"]
            and meaningful_train_improvement
        )
        selected_row = slow.copy() if accept_slow_k else parent.copy()
        selected_row["pilot"] = pilot
        selected_row["selected_mechanism"] = (
            "direct_slow_k" if accept_slow_k else "frozen_fast_parent"
        )
        selected_row["slow_k_train_relative_change"] = train_change
        selected_row["slow_k_count_relative_change"] = count_change
        mechanism_rows.append(selected_row)
    pd.DataFrame(mechanism_rows).to_csv(
        output_root / "mechanism_selected_summary.csv",
        index=False,
    )
    features = pd.concat(feature_tables, ignore_index=True)
    features.to_csv(
        output_root / "combined_firing_pattern_comparison.csv",
        index=False,
    )
    _cross_cell_plot(
        combined,
        output_root / "cross_cell_slow_k_comparison.png",
    )
    _spike_count_plot(
        features,
        output_root / "cross_cell_spike_counts.png",
    )


if __name__ == "__main__":
    main()
