#!/usr/bin/env python3
"""Plot held-out biological, ephys-fit, and RNA-predicted model replays."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.family_overlays import (
    stable_cycle_slices,
    typical_cycle_slice,
)
from inverse_ephys_alpha_beta.phase_population import (
    select_phase_population_protocols,
)
from inverse_ephys_alpha_beta.phase_timing_population import _metric_row
from inverse_ephys_alpha_beta.recovery_onset import (
    simulate_recovery_onset_model,
)
from inverse_ephys_alpha_beta.recovery_onset_compact import (
    recovery_onset_model_from_row,
)


BIOLOGICAL_COLOR = "#202020"
EPHYS_COLOR = "#D13C55"
RNA_COLOR = "#0072B2"
STIMULUS_COLOR = "#DCE9F0"
DATASET_LABELS = {
    "gouwens_visp": "Gouwens VISp Patch-seq",
    "scala_room_temperature": "Scala M1 Patch-seq, room temperature",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _simulate(row: pd.Series, observed):
    waveform, timing, onset = recovery_onset_model_from_row(row)
    return simulate_recovery_onset_model(
        waveform,
        timing,
        onset,
        observed.time_ms,
        observed.input_value,
        float(observed.voltage_mv[0]),
        use_onset_branch=True,
    )


def _full_trace(axis, observed, ephys, rna) -> None:
    relative = observed.time_ms - observed.stimulus_start_ms
    duration = observed.stimulus_end_ms - observed.stimulus_start_ms
    axis.axvspan(
        0.0,
        duration,
        color=STIMULUS_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    axis.plot(
        relative,
        observed.voltage_mv,
        color=BIOLOGICAL_COLOR,
        linewidth=0.7,
        label="Biological",
    )
    axis.plot(
        relative,
        ephys.voltage_mv,
        color=EPHYS_COLOR,
        linewidth=0.9,
        label="Ephys-fit model",
    )
    axis.plot(
        relative,
        rna.voltage_mv,
        color=RNA_COLOR,
        linewidth=0.9,
        label="RNA-predicted model",
    )
    axis.set_xlim(-20.0, duration + 20.0)
    axis.set_xlabel("Time from onset (ms)")
    axis.set_ylabel("V (mV)")


def _early_trace(axis, observed, ephys, rna) -> None:
    relative = observed.time_ms - observed.stimulus_start_ms
    duration = observed.stimulus_end_ms - observed.stimulus_start_ms
    stop = min(150.0, max(60.0, 0.25 * duration))
    axis.axvspan(
        0.0,
        stop,
        color=STIMULUS_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    for voltage, color, width in (
        (observed.voltage_mv, BIOLOGICAL_COLOR, 1.0),
        (ephys.voltage_mv, EPHYS_COLOR, 1.15),
        (rna.voltage_mv, RNA_COLOR, 1.15),
    ):
        axis.plot(relative, voltage, color=color, linewidth=width)
    axis.set_xlim(-5.0, stop)
    axis.set_xlabel("Early response (ms)")
    axis.set_ylabel("V (mV)")


def _phase_plane(axis, observed, ephys, rna) -> None:
    highlighted = []
    series = (
        (
            observed.voltage_mv,
            observed.velocity_mv_ms,
            BIOLOGICAL_COLOR,
        ),
        (ephys.voltage_mv, ephys.velocity_mv_ms, EPHYS_COLOR),
        (rna.voltage_mv, rna.velocity_mv_ms, RNA_COLOR),
    )
    for voltage, velocity, color in series:
        cycles = stable_cycle_slices(
            observed.time_ms,
            voltage,
            observed.stimulus_start_ms,
            observed.stimulus_end_ms,
        )
        for cycle in cycles:
            axis.plot(
                voltage[cycle],
                velocity[cycle],
                color=color,
                linewidth=0.45,
                alpha=0.12,
            )
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
            linewidth=1.35,
        )
        highlighted.append(velocity[cycle])
    combined = np.concatenate(highlighted)
    finite = combined[np.isfinite(combined)]
    if len(finite):
        lower, upper = np.percentile(finite, (0.5, 99.5))
        margin = max(5.0, 0.1 * float(upper - lower))
        axis.set_ylim(float(lower - margin), float(upper + margin))
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.7)
    axis.set_xlabel("V (mV)")
    axis.set_ylabel("dV/dt (mV/ms)")


def _run_dataset(
    dataset: str,
    representatives: pd.DataFrame,
    srrr_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    ephys_parameters = pd.read_csv(
        srrr_root / dataset / "heldout_ephys_parameters.csv",
        dtype={"cell_id": "string"},
    )
    rna_parameters = pd.read_csv(
        srrr_root / dataset / "heldout_rna_parameters.csv",
        dtype={"cell_id": "string"},
    )
    metadata = representatives.loc[
        representatives["dataset"].eq(dataset)
    ].copy()
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata[
        ["cell_id", "broad_class", "transcriptomic_type"]
    ]
    rows = (
        metadata.merge(
            ephys_parameters,
            on="cell_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            rna_parameters,
            on="cell_id",
            suffixes=("_ephys", "_rna"),
            how="inner",
            validate="one_to_one",
        )
    )
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(17.0, 2.7 * len(rows)),
        gridspec_kw={"width_ratios": (1.65, 1.0, 1.0)},
        constrained_layout=True,
        squeeze=False,
    )
    metric_rows = []
    for index, row in rows.iterrows():
        ephys_row = row.filter(regex="_ephys$")
        ephys_row.index = ephys_row.index.str.removesuffix("_ephys")
        rna_row = row.filter(regex="_rna$")
        rna_row.index = rna_row.index.str.removesuffix("_rna")
        protocols = select_phase_population_protocols(
            str(ephys_row["nwb_path"])
        )
        observed = protocols.validation
        ephys = _simulate(ephys_row, observed)
        rna = _simulate(rna_row, observed)
        _full_trace(axes[index, 0], observed, ephys, rna)
        _early_trace(axes[index, 1], observed, ephys, rna)
        _phase_plane(axes[index, 2], observed, ephys, rna)
        common = {
            "validation_current_pa": protocols.validation_current,
            "validation_sweep_number": protocols.validation_sweep_number,
        }
        ephys_metrics = _metric_row(
            dataset,
            str(row["cell_id"]),
            str(ephys_row["nwb_path"]),
            "ephys_fit",
            True,
            observed,
            ephys,
            common,
        )
        rna_metrics = _metric_row(
            dataset,
            str(row["cell_id"]),
            str(ephys_row["nwb_path"]),
            "rna_prediction",
            False,
            observed,
            rna,
            common,
        )
        metric_rows.extend(
            (
                {
                    "broad_class": row["broad_class"],
                    "transcriptomic_type": row["transcriptomic_type"],
                    **ephys_metrics,
                },
                {
                    "broad_class": row["broad_class"],
                    "transcriptomic_type": row["transcriptomic_type"],
                    **rna_metrics,
                },
            )
        )
        axes[index, 0].set_title(
            f"{row['broad_class']} | {row['transcriptomic_type']}\n"
            f"cell {row['cell_id']} | {protocols.validation_current:g} pA | "
            f"spikes bio/ephys/RNA "
            f"{ephys_metrics['observed_spike_count']}/"
            f"{ephys_metrics['predicted_spike_count']}/"
            f"{rna_metrics['predicted_spike_count']}",
            loc="left",
            fontsize=9,
        )
        axes[index, 1].set_title(
            "Onset and initial spike train",
            loc="left",
            fontsize=9,
        )
        axes[index, 2].set_title(
            "Stable cycle | Chamfer ephys/RNA "
            f"{ephys_metrics['phase_chamfer']:.3f}/"
            f"{rna_metrics['phase_chamfer']:.3f}",
            loc="left",
            fontsize=9,
        )
        if index == 0:
            axes[index, 0].legend(
                frameon=False,
                fontsize=8,
                ncol=3,
                loc="upper right",
            )
        for axis in axes[index]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
    figure.suptitle(
        f"{DATASET_LABELS[dataset]}: held-out transcriptome to voltage trace\n"
        "Ephys-fit parameters are targets only; the RNA model was predicted "
        "without the held-out cell in sRRR training",
        fontsize=14,
    )
    destination = output_root / f"{dataset}_heldout_rna_replays.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(metric_rows)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--srrr-root",
        type=Path,
        default=Path("outputs/recovery_onset_heldout_srrr"),
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        default=Path(
            "outputs/recovery_onset_family_overlays/"
            "representative_cells.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recovery_onset_rna_replays"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    representatives = pd.read_csv(
        args.representatives,
        dtype={"cell_id": "string"},
    )
    metrics = [
        _run_dataset(
            dataset,
            representatives,
            args.srrr_root,
            args.output_root,
        )
        for dataset in ("gouwens_visp", "scala_room_temperature")
    ]
    metrics = pd.concat(metrics, ignore_index=True)
    _write_csv(metrics, args.output_root / "replay_metrics.csv")
    summary = (
        metrics.groupby(["dataset", "model"], as_index=False)
        .agg(
            cell_count=("cell_id", "nunique"),
            median_phase_chamfer=("phase_chamfer", "median"),
            median_absolute_spike_count_error=(
                "absolute_spike_count_error",
                "median",
            ),
            median_spike_time_rmse_ms=("spike_time_rmse_ms", "median"),
            median_voltage_rmse_mv=("voltage_rmse_mv", "median"),
            median_absolute_first_isi_error_ms=(
                "absolute_first_isi_ms_error",
                "median",
            ),
            median_absolute_late_isi_error_ms=(
                "absolute_late_isi_ms_error",
                "median",
            ),
        )
    )
    _write_csv(summary, args.output_root / "replay_summary.csv")


if __name__ == "__main__":
    main()
