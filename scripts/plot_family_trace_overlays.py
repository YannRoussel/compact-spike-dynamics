#!/usr/bin/env python3
"""Plot held-out biological and compact-model traces by broad cell family."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.family_overlays import (
    fit_held_out_overlay,
    major_families,
    merge_parameters_and_metadata,
    rank_representative_candidates,
    stable_cycle_slices,
    typical_cycle_slice,
)


BIOLOGICAL_COLOR = "#202020"
MODEL_COLOR = "#D13C55"
BIOLOGICAL_LIGHT = "#8C8C8C"
MODEL_LIGHT = "#EEA0AC"
STIMULUS_COLOR = "#DCE9F0"
DATASET_LABELS = {
    "gouwens_visp": "Gouwens VISp Patch-seq",
    "scala_room_temperature": "Scala M1 Patch-seq, room temperature",
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


def _decimated(indices: np.ndarray, maximum: int = 20_000) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    step = int(np.ceil(len(indices) / maximum))
    return indices[::step]


def _plot_trace(axis: plt.Axes, overlay) -> None:
    observed = overlay.observed
    simulation = overlay.simulation
    relative_time = observed.time_ms - observed.stimulus_start_ms
    duration = observed.stimulus_end_ms - observed.stimulus_start_ms
    visible = np.flatnonzero(
        (relative_time >= -20.0) & (relative_time <= duration + 20.0)
    )
    visible = _decimated(visible)
    axis.axvspan(
        0.0,
        duration,
        color=STIMULUS_COLOR,
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    axis.plot(
        relative_time[visible],
        observed.voltage_mv[visible],
        color=BIOLOGICAL_COLOR,
        linewidth=0.8,
        label="Biological",
        zorder=2,
    )
    axis.plot(
        relative_time[visible],
        simulation.voltage_mv[visible],
        color=MODEL_COLOR,
        linewidth=1.0,
        alpha=0.92,
        label="Compact model",
        zorder=3,
    )
    axis.set_xlim(-20.0, duration + 20.0)
    axis.set_ylabel("V (mV)")
    axis.set_xlabel("Time from stimulus onset (ms)")


def _plot_phase_plane(axis: plt.Axes, overlay) -> None:
    observed = overlay.observed
    simulation = overlay.simulation
    for voltage, velocity, color in (
        (
            observed.voltage_mv,
            observed.velocity_mv_ms,
            BIOLOGICAL_LIGHT,
        ),
        (
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
            MODEL_LIGHT,
        ),
    ):
        for cycle in stable_cycle_slices(
            observed.time_ms,
            voltage,
            observed.stimulus_start_ms,
            observed.stimulus_end_ms,
        ):
            axis.plot(
                voltage[cycle],
                velocity[cycle],
                color=color,
                linewidth=0.55,
                alpha=0.28,
                zorder=1,
            )
    for voltage, velocity, color, label, zorder in (
        (
            observed.voltage_mv,
            observed.velocity_mv_ms,
            BIOLOGICAL_COLOR,
            "Biological",
            2,
        ),
        (
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
            MODEL_COLOR,
            "Compact model",
            3,
        ),
    ):
        cycle = typical_cycle_slice(
            observed.time_ms,
            voltage,
            observed.stimulus_start_ms,
            observed.stimulus_end_ms,
        )
        axis.plot(
            voltage[cycle],
            velocity[cycle],
            color=color,
            linewidth=1.45,
            label=label,
            zorder=zorder,
        )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.7, zorder=0)
    axis.set_xlabel("V (mV)")
    axis.set_ylabel("dV/dt (mV/ms)")


def _fit_representatives(
    merged: pd.DataFrame,
    families: tuple[str, ...],
    maximum_candidates: int,
) -> list[tuple[pd.Series, object]]:
    selected = []
    failures = []
    for family in families:
        family_frame = merged.loc[merged["broad_class"].eq(family)]
        candidates = rank_representative_candidates(family_frame)
        for _, candidate in candidates.head(maximum_candidates).iterrows():
            try:
                selected.append((candidate, fit_held_out_overlay(candidate)))
                break
            except Exception as error:
                failures.append(
                    f"{family}/{candidate['cell_id']}:"
                    f"{type(error).__name__}:{error}"
                )
        else:
            details = "; ".join(failures[-maximum_candidates:])
            raise RuntimeError(
                f"No representative could be replayed for {family}: {details}"
            )
    return selected


def _summary_row(
    candidate: pd.Series,
    overlay,
    family_count: int,
) -> dict[str, object]:
    return {
        "dataset": candidate["dataset"],
        "broad_class": candidate["broad_class"],
        "family_fitted_cell_count": family_count,
        "cell_id": candidate["cell_id"],
        "donor_id": candidate["donor_id"],
        "transcriptomic_type": candidate["transcriptomic_type"],
        "nwb_path": candidate["nwb_path"],
        "validation_sweep_number": (
            overlay.protocols.validation_sweep_number
        ),
        "validation_current_pa": overlay.protocols.validation_current,
        "training_currents_pa": "|".join(
            f"{value:g}" for value in overlay.protocols.training_currents
        ),
        "selection_better_half": candidate["selection_better_half"],
        "selection_quality_rank": candidate["selection_quality_rank"],
        "selection_typicality": candidate["selection_typicality"],
        "population_high_qc": candidate["high_qc"],
        **overlay.metrics,
    }


def plot_dataset(
    dataset: str,
    parameters: pd.DataFrame,
    metadata: pd.DataFrame,
    output_root: Path,
    minimum_family_cells: int,
    maximum_candidates: int,
) -> pd.DataFrame:
    merged = merge_parameters_and_metadata(parameters, metadata, dataset)
    families = major_families(
        merged,
        minimum_cells=minimum_family_cells,
    )
    representatives = _fit_representatives(
        merged,
        families,
        maximum_candidates=maximum_candidates,
    )
    figure, axes = plt.subplots(
        len(representatives),
        2,
        figsize=(15.0, 2.6 * len(representatives)),
        gridspec_kw={"width_ratios": (1.75, 1.0)},
        squeeze=False,
        constrained_layout=True,
    )
    summary_rows = []
    class_counts = merged["broad_class"].value_counts()
    for row_index, (candidate, overlay) in enumerate(representatives):
        metrics = overlay.metrics
        family = str(candidate["broad_class"])
        _plot_trace(axes[row_index, 0], overlay)
        _plot_phase_plane(axes[row_index, 1], overlay)
        axes[row_index, 0].set_title(
            f"{family} (n={int(class_counts[family])}) | "
            f"{candidate['transcriptomic_type']}\n"
            f"cell {candidate['cell_id']} | held-out "
            f"{overlay.protocols.validation_current:g} pA | "
            f"spikes {metrics['observed_spike_count']}/"
            f"{metrics['predicted_spike_count']} (bio/model)",
            loc="left",
            fontsize=9.5,
        )
        axes[row_index, 1].set_title(
            "Stable phase loop | normalized Chamfer "
            f"{float(metrics['phase_chamfer']):.3f}",
            loc="left",
            fontsize=9.5,
        )
        for axis in axes[row_index]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        if row_index == 0:
            axes[row_index, 0].legend(
                loc="upper right",
                frameon=False,
                fontsize=8,
                ncol=2,
            )
        summary_rows.append(
            _summary_row(
                candidate,
                overlay,
                family_count=int(class_counts[family]),
            )
        )
    figure.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)}: held-out compact-model "
        "replay by major family\n"
        "Representative = parameter-typical cell within the better half "
        "of validation fits; faint lines show other stable cycles",
        fontsize=14,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{dataset}_family_overlays.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(summary_rows)


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
        default=Path("outputs/family_trace_phase_overlays"),
    )
    parser.add_argument("--minimum-family-cells", type=int, default=10)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(METADATA_PATHS),
    )
    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()
    parameters = pd.read_csv(
        args["parameters"],
        dtype={"cell_id": "string"},
    )
    datasets = args["dataset"] or list(METADATA_PATHS)
    summaries = []
    for dataset in datasets:
        metadata = pd.read_csv(
            METADATA_PATHS[dataset],
            dtype={"cell_id": "string"},
        )
        summaries.append(
            plot_dataset(
                dataset,
                parameters,
                metadata,
                output_root=args["output_root"],
                minimum_family_cells=args["minimum_family_cells"],
                maximum_candidates=args["maximum_candidates"],
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    args["output_root"].mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        args["output_root"] / "representative_cells.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
