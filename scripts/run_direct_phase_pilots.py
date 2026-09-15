#!/usr/bin/env python3
"""Run the staged direct-kinetics experiment on the four pilot neurons."""

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
from inverse_ephys_alpha_beta.direct_phase_fit import (
    DirectPhaseFitConfig,
    direct_phase_summary,
    optimize_direct_phase_fit,
    save_direct_phase_fit_result,
)
from inverse_ephys_alpha_beta.phase_shape import PhaseShapeConfig
from inverse_ephys_alpha_beta.protocols import (
    biological_screen_config,
)


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


def _save_cross_cell_plot(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    metrics = (
        "shape_loss",
        "downstroke_low_voltage_concavity",
        "physical_constraint_loss",
        "held_out__total",
    )
    labels = (
        "Shape loss",
        "Low-V downstroke\nconcavity",
        "Physical\nconstraint loss",
        "Held-out phase\ntotal",
    )
    stages = ("initial_direct", "shape", "scale", "joint")
    colors = ("#9c6644", "#e76f51", "#457b9d", "#7b2cbf")
    pilots = summary["pilot"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 8.0),
        constrained_layout=True,
    )
    x = np.arange(len(pilots))
    width = 0.19
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
            values = np.maximum(values, 1e-3)
            axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=color,
                label=stage.replace("_", " ").title(),
            )
        axis.set_xticks(
            x,
            [pilot.replace("_", "\n", 1) for pilot in pilots],
            fontsize=8,
        )
        axis.set_ylabel("Normalized loss")
        axis.set_yscale("log")
        axis.set_title(label, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(
        "Four-cell direct x_inf/tau fitting experiment",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_selected_residual_plot(
    selected: pd.DataFrame,
    path: Path,
) -> None:
    metrics = (
        "max_dvdt_residual_mv_ms",
        "min_dvdt_residual_mv_ms",
        "peak_voltage_residual_mv",
        "minimum_voltage_residual_mv",
    )
    labels = (
        "Maximum dV/dt residual (mV/ms)",
        "Minimum dV/dt residual (mV/ms)",
        "Peak voltage residual (mV)",
        "Trough voltage residual (mV)",
    )
    short_names = (
        "Gouwens\n704047023",
        "Gouwens\n674495385",
        "Scala\n20180920-1",
        "Scala\n20190425-4",
    )
    colors = ("#2a9d8f", "#2a9d8f", "#e76f51", "#e76f51")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.5),
        constrained_layout=True,
    )
    for axis, metric, label in zip(axes.flat, metrics, labels):
        values = selected[metric].to_numpy(dtype=float)
        axis.axhline(0.0, color="#1f2933", linewidth=1.0)
        axis.bar(
            np.arange(len(selected)),
            values,
            color=colors,
            width=0.68,
        )
        axis.set_xticks(np.arange(len(selected)), short_names, fontsize=8)
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        for index, value in enumerate(values):
            offset = 3.0 if value >= 0 else -3.0
            axis.annotate(
                f"{value:+.1f}",
                (index, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    stage_text = ", ".join(
        f"{name}: {stage}"
        for name, stage in zip(
            ("G704", "G674", "S20180920", "S20190425"),
            selected["stage"],
        )
    )
    figure.suptitle(
        "Selected-stage physical residuals\n" + stage_text,
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
        "--baseline-root",
        default="outputs/phase_shape_pilots",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/direct_phase_pilots",
    )
    parser.add_argument("--shape-population-size", type=int, default=10)
    parser.add_argument("--shape-generations", type=int, default=3)
    parser.add_argument("--shape-elites", type=int, default=2)
    parser.add_argument("--scale-population-size", type=int, default=8)
    parser.add_argument("--scale-generations", type=int, default=2)
    parser.add_argument("--final-population-size", type=int, default=10)
    parser.add_argument("--final-generations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = load_raw_spike_cycle_table(args.raw_spike_cycles)
    baseline_root = Path(args.baseline_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    for pilot_index, (
        pilot,
        dataset,
        cell_id,
        temperature,
        preset,
    ) in enumerate(PILOTS):
        output = output_root / pilot
        summary_path = output / "stage_summary.csv"
        if args.resume and summary_path.exists():
            summary = pd.read_csv(summary_path)
        else:
            target = build_cell_optimization_target(
                raw,
                dataset=dataset,
                cell_id=cell_id,
                temperature_c=temperature,
                local_passive=True,
            )
            baseline = pd.read_csv(
                baseline_root / pilot / "soma_ais_best_model.csv"
            )
            screen_config = replace(
                biological_screen_config(
                    preset,
                    temperature_c=temperature,
                    dt_ms=args.dt,
                ),
                rheobase_tolerance_pa=2.0,
            )
            fit_config = DirectPhaseFitConfig(
                shape_population_size=args.shape_population_size,
                shape_generations=args.shape_generations,
                shape_elites=args.shape_elites,
                scale_population_size=args.scale_population_size,
                scale_generations=args.scale_generations,
                final_population_size=args.final_population_size,
                final_generations=args.final_generations,
                workers=args.workers,
                seed=args.seed + 1000 * pilot_index,
            )
            shape_config = PhaseShapeConfig()
            result = optimize_direct_phase_fit(
                target,
                baseline,
                screen_config,
                fit_config,
                shape_config,
            )
            save_direct_phase_fit_result(
                result,
                screen_config,
                output,
            )
            summary = direct_phase_summary(result)
        summary.insert(0, "pilot", pilot)
        summaries.append(summary)
        joint = summary.loc[summary["stage"].eq("joint")].iloc[0]
        print(
            f"{pilot}: total={joint['objective_total']:.3f}, "
            f"shape={joint['shape_loss']:.3f}, "
            f"constraint={joint['physical_constraint_loss']:.3f}, "
            f"rheobase_constraint="
            f"{joint['rheobase_constraint_loss']:.3f}, "
            f"held_out={joint['held_out__total']:.3f}"
        )

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_root / "combined_stage_summary.csv", index=False)
    selected_rows = []
    for pilot, group in combined.groupby("pilot", sort=False):
        candidates = group.loc[
            group["stage"].isin(("shape", "scale", "joint"))
        ]
        feasible = candidates.loc[
            candidates["rheobase_constraint_loss"].le(1e-12)
        ]
        selection_pool = feasible if not feasible.empty else candidates
        selected = selection_pool.nsmallest(
            1,
            "objective_total",
        ).copy()
        selected["rheobase_feasible_selection"] = not feasible.empty
        selected_rows.append(selected)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(
        output_root / "guarded_selected_summary.csv",
        index=False,
    )
    selected.loc[
        :,
        [
            "pilot",
            "stage",
            "objective_total",
            "shape_loss",
            "downstroke_low_voltage_concavity",
            "downstroke_low_voltage_sign_agreement",
            "physical_constraint_loss",
            "constraints_satisfied",
            "model_rheobase_pa",
            "biological_rheobase_pa",
            "rheobase_constraint_loss",
            "rheobase_feasible_selection",
            "max_dvdt_residual_mv_ms",
            "min_dvdt_residual_mv_ms",
            "peak_voltage_residual_mv",
            "minimum_voltage_residual_mv",
            "held_out__total",
        ],
    ].to_csv(
        output_root / "guarded_selected_key_metrics.csv",
        index=False,
    )
    _save_cross_cell_plot(
        combined,
        output_root / "cross_cell_stage_comparison.png",
    )
    _save_selected_residual_plot(
        selected,
        output_root / "selected_physical_residuals.png",
    )


if __name__ == "__main__":
    main()
