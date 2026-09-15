#!/usr/bin/env python3
"""Run the discrete-topology effective-current ladder on four pilot cells."""

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
from inverse_ephys_alpha_beta.effective_components import (
    EffectiveFitConfig,
    effective_candidate_summary,
    fit_effective_component_ladder,
    load_effective_parent_vector,
    save_effective_fit_result,
)
from inverse_ephys_alpha_beta.direct_slow_k import (
    DIRECT_SLOW_K_ALL_PARAMETER_NAMES,
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


def _pilot_label(pilot: str) -> str:
    if pilot.startswith("gouwens_"):
        return "Gouwens\n" + pilot.removeprefix("gouwens_")
    if pilot.startswith("scala_"):
        return (
            "Scala\n"
            + pilot.removeprefix("scala_").replace("_sample_", "-")
        )
    return pilot


def _selected_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[summary["selected"]].copy()


def _cross_cell_plot(summary: pd.DataFrame, path: Path) -> None:
    parent = summary.loc[summary["label"].eq("canonical_parent")]
    selected = _selected_rows(summary)
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
        "Selection score",
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.0, 7.7),
        constrained_layout=True,
    )
    x = np.arange(len(pilots))
    width = 0.34
    for axis, metric, label in zip(axes.flat, metrics, labels):
        for index, (frame, name, color) in enumerate(
            (
                (parent, "Canonical parent", "#e76f51"),
                (selected, "Selected", "#457b9d"),
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
            x,
            [_pilot_label(pilot) for pilot in pilots],
            fontsize=7,
        )
        axis.set_yscale("log")
        axis.set_ylabel("Normalized loss")
        axis.set_title(label, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Discrete topology and conditional memory-state pilot",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _state_selection_plot(summary: pd.DataFrame, path: Path) -> None:
    selected = _selected_rows(summary).set_index("pilot")
    pilots = summary["pilot"].drop_duplicates().tolist()
    figure, axis = plt.subplots(
        figsize=(10.5, 4.6),
        constrained_layout=True,
    )
    states = selected.reindex(pilots)[
        "extension__active_state_count"
    ].to_numpy(dtype=float)
    bars = axis.bar(
        np.arange(len(pilots)),
        states,
        color="#2a9d8f",
    )
    axis.set_xticks(
        np.arange(len(pilots)),
        [_pilot_label(pilot) for pilot in pilots],
        fontsize=8,
    )
    axis.set_ylim(0.0, max(6.0, float(np.nanmax(states)) + 2.2))
    axis.set_ylabel("Independent gate states")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        "Selected memory count and integer gate topology",
        loc="left",
    )
    for bar, pilot in zip(bars, pilots):
        topology = str(selected.loc[pilot, "topology"])
        tokens = topology.split("_")
        topology_label = "\n".join(
            " ".join(tokens[index : index + 2])
            for index in range(0, len(tokens), 2)
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.12,
            topology_label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _spike_count_plot(features: pd.DataFrame, path: Path) -> None:
    counts = features.loc[features["feature"].eq("spike_count")]
    protocols = ("rheobase", "suprathreshold", "held_out")
    protocol_labels = {
        "rheobase": "Sampled rheobase",
        "suprathreshold": "Suprathreshold",
        "held_out": "Validation current",
    }
    pilots = features["pilot"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.8),
        constrained_layout=True,
    )
    x = np.arange(len(pilots))
    width = 0.25
    for axis, protocol in zip(axes, protocols):
        frame = (
            counts.loc[counts["protocol"].eq(protocol)]
            .set_index("pilot")
            .reindex(pilots)
        )
        for index, (column, label, color) in enumerate(
            (
                ("biological", "Biological", "#147d7e"),
                ("canonical_parent", "Canonical parent", "#e76f51"),
                ("selected", "Selected", "#457b9d"),
            )
        ):
            axis.bar(
                x + (index - 1) * width,
                frame[column].to_numpy(dtype=float),
                width,
                color=color,
                label=label,
            )
        axis.set_xticks(
            x,
            [_pilot_label(pilot) for pilot in pilots],
            fontsize=7,
        )
        axis.set_title(
            protocol_labels[protocol],
            loc="left",
        )
        axis.set_ylabel("Spikes in 500 ms")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Absolute-current spike-count comparison",
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
        default="outputs/effective_component_pilots",
    )
    parser.add_argument(
        "--slow-k-root",
        default="outputs/direct_slow_k_absolute_pilots",
    )
    parser.add_argument("--fast-parent-count", type=int, default=3)
    parser.add_argument("--mechanism-parent-count", type=int, default=2)
    parser.add_argument("--joint-seed-count", type=int, default=2)
    parser.add_argument("--fast-population-size", type=int, default=6)
    parser.add_argument("--fast-generations", type=int, default=2)
    parser.add_argument(
        "--mechanism-population-size",
        type=int,
        default=8,
    )
    parser.add_argument("--mechanism-generations", type=int, default=2)
    parser.add_argument("--joint-population-size", type=int, default=10)
    parser.add_argument("--joint-generations", type=int, default=2)
    parser.add_argument(
        "--exhaustive-fast-topologies",
        action="store_true",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration-ms", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pilot",
        choices=[pilot[0] for pilot in PILOTS],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = load_raw_spike_cycle_table(args.raw_spike_cycles)
    direct_root = Path(args.direct_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_direct = pd.read_csv(
        direct_root / "guarded_selected_key_metrics.csv"
    ).set_index("pilot")
    summaries = []
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
            direct_stage = str(
                selected_direct.loc[pilot, "stage"]
            )
            parent_parameters = load_effective_parent_vector(
                direct_root / pilot / "stage_parameters.csv",
                direct_stage,
            )
            slow_k_seed = None
            slow_k_path = (
                Path(args.slow_k_root)
                / pilot
                / "best_slow_k_parameters.csv"
            )
            if slow_k_path.exists():
                slow_k_seed = (
                    pd.read_csv(slow_k_path)
                    .iloc[0]
                    .loc[list(DIRECT_SLOW_K_ALL_PARAMETER_NAMES)]
                    .to_numpy(dtype=float)
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
                exhaustive_fast_topologies=(
                    args.exhaustive_fast_topologies
                ),
            )
            result = fit_effective_component_ladder(
                target,
                parent_parameters,
                screen_config,
                fit_config,
                PhaseShapeConfig(),
                slow_k_seed_parameters=slow_k_seed,
            )
            save_effective_fit_result(
                result,
                screen_config,
                output,
            )
            summary = effective_candidate_summary(result)
            summary.insert(0, "direct_parent_stage", direct_stage)
            summary.to_csv(summary_path, index=False)
        summary.insert(0, "pilot", pilot)
        summaries.append(summary)
        selected = summary.loc[summary["selected"]].iloc[0]
        print(
            f"{pilot}: {selected['label']} / "
            f"{selected['topology']}, "
            f"states={selected['extension__active_state_count']:.0f}, "
            f"objective={selected['objective_total']:.3f}, "
            f"validation={selected['validation_loss']:.3f}, "
            f"selection={selected['selection_score']:.3f}"
        )

    summaries = []
    for pilot, *_ in PILOTS:
        summary_path = output_root / pilot / "candidate_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        summary.insert(0, "pilot", pilot)
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(
        output_root / "combined_candidate_summary.csv",
        index=False,
    )
    selected = _selected_rows(combined)
    selected.loc[
        :,
        [
            "pilot",
            "direct_parent_stage",
            "label",
            "stage",
            "parent_label",
            "topology",
            "objective_total",
            "selection_score",
            "shape_loss",
            "phase__physical_constraint_loss",
            "firing_pattern_loss",
            "spike_count_loss",
            "validation_loss",
            "complexity_penalty",
            "model_rheobase_pa",
            "biological_rheobase_pa",
            "extension__active_state_count",
            "extension__active_component_count",
            "extension__gslow_na_ms_cm2",
            "extension__gslow_k_ms_cm2",
        ],
    ].to_csv(
        output_root / "selected_key_metrics.csv",
        index=False,
    )
    parents = combined.loc[
        combined["label"].eq("canonical_parent")
    ].set_index("pilot")
    selected_indexed = selected.set_index("pilot")
    delta_rows = []
    for pilot in selected_indexed.index:
        parent = parents.loc[pilot]
        child = selected_indexed.loc[pilot]
        delta_rows.append(
            {
                "pilot": pilot,
                "selected_label": child["label"],
                "selected_topology": child["topology"],
                **{
                    f"parent__{metric}": parent[metric]
                    for metric in (
                        "objective_total",
                        "shape_loss",
                        "phase__physical_constraint_loss",
                        "firing_pattern_loss",
                        "spike_count_loss",
                        "validation_loss",
                        "selection_score",
                    )
                },
                **{
                    f"selected__{metric}": child[metric]
                    for metric in (
                        "objective_total",
                        "shape_loss",
                        "phase__physical_constraint_loss",
                        "firing_pattern_loss",
                        "spike_count_loss",
                        "validation_loss",
                        "selection_score",
                    )
                },
            }
        )
    pd.DataFrame(delta_rows).to_csv(
        output_root / "selected_vs_parent_metrics.csv",
        index=False,
    )
    feature_tables = []
    for pilot, *_ in PILOTS:
        feature_path = (
            output_root / pilot / "firing_pattern_comparison.csv"
        )
        if not feature_path.exists():
            continue
        features = pd.read_csv(feature_path)
        features.insert(0, "pilot", pilot)
        feature_tables.append(features)
    combined_features = pd.concat(feature_tables, ignore_index=True)
    combined_features.to_csv(
        output_root / "combined_firing_pattern_comparison.csv",
        index=False,
    )
    combined_features.loc[
        combined_features["feature"].eq("spike_count")
    ].to_csv(
        output_root / "selected_spike_counts.csv",
        index=False,
    )
    _cross_cell_plot(
        combined,
        output_root / "cross_cell_effective_comparison.png",
    )
    _state_selection_plot(
        combined,
        output_root / "selected_state_topologies.png",
    )
    _spike_count_plot(
        combined_features,
        output_root / "cross_cell_spike_counts.png",
    )


if __name__ == "__main__":
    main()
