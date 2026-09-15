#!/usr/bin/env python3
"""Fit independent generic soma and AIS currents on four pilot cells."""

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
from inverse_ephys_alpha_beta.effective_components import EffectiveFitConfig
from inverse_ephys_alpha_beta.phase_shape import PhaseShapeConfig
from inverse_ephys_alpha_beta.protocols import biological_screen_config
from inverse_ephys_alpha_beta.two_compartment_effective import (
    fit_two_compartment_ladder,
    load_selected_effective_parent,
    save_two_compartment_fit_result,
    two_compartment_candidate_summary,
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


def _pilot_label(pilot: str) -> str:
    if pilot.startswith("gouwens_"):
        return "Gouwens\n" + pilot.removeprefix("gouwens_")
    if pilot.startswith("scala_"):
        return (
            "Scala\n"
            + pilot.removeprefix("scala_").replace("_sample_", "-")
        )
    return pilot


def _cross_cell_plot(summary: pd.DataFrame, path: Path) -> None:
    parent = summary.loc[summary["label"].eq("shared_ais_parent")]
    selected = summary.loc[summary["selected"]]
    pilots = summary["pilot"].drop_duplicates().tolist()
    metrics = (
        "shape_loss",
        "phase__physical_constraint_loss",
        "firing_pattern_loss",
        "spike_count_loss",
        "validation_loss",
        "selection_score",
    )
    labels = (
        "Phase shape",
        "Physical",
        "Spike train",
        "Spike count",
        "Validation",
        "Selection",
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(14.0, 7.7), constrained_layout=True
    )
    x = np.arange(len(pilots))
    width = 0.34
    for axis, metric, label in zip(axes.flat, metrics, labels):
        for index, (frame, name, color) in enumerate(
            (
                (parent, "Shared AIS parent", "#e76f51"),
                (selected, "Selected soma-AIS", "#457b9d"),
            )
        ):
            values = (
                frame.set_index("pilot")
                .reindex(pilots)[metric]
                .to_numpy(dtype=float)
            )
            axis.bar(
                x + (index - 0.5) * width,
                np.maximum(values, 1e-3),
                width,
                color=color,
                label=name,
            )
        axis.set_xticks(
            x, [_pilot_label(pilot) for pilot in pilots], fontsize=7
        )
        axis.set_yscale("log")
        axis.set_ylabel("Normalized loss")
        axis.set_title(label, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Independent soma-AIS generic-current pilot", fontsize=13
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
        "--effective-root",
        default="outputs/effective_component_pilots",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/two_compartment_effective_pilots",
    )
    parser.add_argument("--fast-parent-count", type=int, default=2)
    parser.add_argument("--mechanism-parent-count", type=int, default=1)
    parser.add_argument("--joint-seed-count", type=int, default=1)
    parser.add_argument("--fast-population-size", type=int, default=8)
    parser.add_argument("--fast-generations", type=int, default=2)
    parser.add_argument(
        "--mechanism-population-size", type=int, default=8
    )
    parser.add_argument("--mechanism-generations", type=int, default=2)
    parser.add_argument("--joint-population-size", type=int, default=10)
    parser.add_argument("--joint-generations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration-ms", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pilot", choices=[pilot[0] for pilot in PILOTS]
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = load_raw_spike_cycle_table(args.raw_spike_cycles)
    effective_root = Path(args.effective_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pilots = (
        tuple(pilot for pilot in PILOTS if pilot[0] == args.pilot)
        if args.pilot is not None
        else PILOTS[
            : args.limit if args.limit is not None else len(PILOTS)
        ]
    )
    for pilot_index, (
        pilot,
        dataset,
        cell_id,
        temperature,
        preset,
    ) in enumerate(pilots):
        output = output_root / pilot
        summary_path = output / "candidate_summary.csv"
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
            parent_values, parent_topologies = (
                load_selected_effective_parent(
                    effective_root / pilot / "selected_parameters.csv",
                    target,
                )
            )
            screen_config = replace(
                biological_screen_config(
                    preset,
                    temperature_c=temperature,
                    dt_ms=args.dt,
                ),
                rheobase_tolerance_pa=2.0,
            )
            fit_config = EffectiveFitConfig(
                fast_parent_count=args.fast_parent_count,
                mechanism_parent_count=args.mechanism_parent_count,
                joint_seed_count=args.joint_seed_count,
                fast_population_size=args.fast_population_size,
                fast_generations=args.fast_generations,
                mechanism_population_size=(
                    args.mechanism_population_size
                ),
                mechanism_generations=args.mechanism_generations,
                joint_population_size=args.joint_population_size,
                joint_generations=args.joint_generations,
                workers=args.workers,
                seed=args.seed + 10_000 * pilot_index,
                train_duration_ms=args.duration_ms,
            )
            result = fit_two_compartment_ladder(
                target,
                parent_values,
                parent_topologies,
                screen_config,
                fit_config,
                PhaseShapeConfig(),
            )
            save_two_compartment_fit_result(
                result, screen_config, output
            )
            summary = two_compartment_candidate_summary(result)
            summary.to_csv(summary_path, index=False)
        selected = summary.loc[summary["selected"]].iloc[0]
        print(
            f"{pilot}: {selected['label']} / "
            f"{selected['topology']}, "
            f"states={selected['extension__active_state_count']:.0f}, "
            f"objective={selected['objective_total']:.3f}, "
            f"validation={selected['validation_loss']:.3f}, "
            f"selection={selected['selection_score']:.3f}",
            flush=True,
        )

    summaries = []
    for pilot, *_ in PILOTS:
        path = output_root / pilot / "candidate_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "pilot", pilot)
            summaries.append(frame)
    if not summaries:
        return
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(
        output_root / "combined_candidate_summary.csv", index=False
    )
    selected = combined.loc[combined["selected"]]
    selected.to_csv(
        output_root / "selected_key_metrics.csv", index=False
    )
    delta_rows = []
    for pilot in combined["pilot"].drop_duplicates():
        pilot_rows = combined.loc[combined["pilot"].eq(pilot)]
        parent = pilot_rows.loc[
            pilot_rows["label"].eq("shared_ais_parent")
        ].iloc[0]
        child = pilot_rows.loc[pilot_rows["selected"]].iloc[0]
        row = {
            "pilot": pilot,
            "selected_label": child["label"],
            "selected_topology": child["topology"],
            "selected_state_count": child[
                "extension__active_state_count"
            ],
            "ais_fast_gna_at_upper_bound": (
                child["extension__ais_gna_fast_ms_cm2"] >= 499.0
            ),
        }
        for metric in (
            "objective_total",
            "shape_loss",
            "phase__physical_constraint_loss",
            "firing_pattern_loss",
            "spike_count_loss",
            "validation_loss",
            "selection_score",
            "model_rheobase_pa",
        ):
            row[f"parent__{metric}"] = parent[metric]
            row[f"selected__{metric}"] = child[metric]
            row[f"delta__{metric}"] = child[metric] - parent[metric]
        delta_rows.append(row)
    pd.DataFrame(delta_rows).to_csv(
        output_root / "parent_selected_deltas.csv", index=False
    )
    feature_tables = []
    for pilot, *_ in PILOTS:
        path = output_root / pilot / "firing_pattern_comparison.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "pilot", pilot)
            feature_tables.append(frame)
    if feature_tables:
        combined_features = pd.concat(
            feature_tables, ignore_index=True
        )
        combined_features.to_csv(
            output_root / "combined_firing_pattern_comparison.csv",
            index=False,
        )
        combined_features.loc[
            combined_features["feature"].eq("spike_count")
        ].to_csv(
            output_root / "selected_spike_counts.csv", index=False
        )
    _cross_cell_plot(
        combined,
        output_root / "cross_cell_soma_ais_comparison.png",
    )


if __name__ == "__main__":
    main()
