#!/usr/bin/env python3
"""Plot trace-QC-filtered family examples for the recovery-onset model."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.family_overlays import (
    major_families,
    rank_representative_candidates,
    recording_qc_metrics,
    stable_cycle_slices,
    typical_cycle_slice,
)
from inverse_ephys_alpha_beta.phase_population import (
    select_phase_population_protocols,
)
from inverse_ephys_alpha_beta.phase_template_model import (
    fit_phase_template_ladder,
)
from inverse_ephys_alpha_beta.phase_timing_population import _metric_row
from inverse_ephys_alpha_beta.recovery_onset import (
    fit_recovery_onset_model,
    simulate_recovery_onset_model,
)


BIOLOGICAL_COLOR = "#202020"
MODEL_COLOR = "#D13C55"
STIMULUS_COLOR = "#DCE9F0"
DATASET_LABELS = {
    "gouwens_visp": "Gouwens VISp Patch-seq",
    "scala_room_temperature": "Scala M1 Patch-seq, room temperature",
}
EXCLUDED_CELL_IDS = {
    "20190122_sample_16",
    "20190417_sample_5",
}


@dataclass(frozen=True)
class Overlay:
    observed: object
    simulation: object
    protocols: object
    metrics: dict[str, object]
    recording_qc: dict[str, float | bool]


def _fit_overlay(candidate: pd.Series) -> Overlay:
    protocols = select_phase_population_protocols(candidate["nwb_path"])
    first = protocols.training[0]
    baseline = first.time_ms < first.stimulus_start_ms
    resting_voltage = float(np.median(first.voltage_mv[baseline]))
    waveform = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
    )
    extended = fit_recovery_onset_model(
        protocols.training,
        waveform.cycles,
        voltage_reference_mv=resting_voltage,
    )
    observed = protocols.validation
    simulation = simulate_recovery_onset_model(
        waveform.no_slow_model,
        extended.timing,
        extended.onset,
        observed.time_ms,
        observed.input_value,
        float(observed.voltage_mv[0]),
        use_onset_branch=True,
    )
    metrics = _metric_row(
        candidate["dataset"],
        str(candidate["cell_id"]),
        candidate["nwb_path"],
        "recovery_onset",
        True,
        observed,
        simulation,
        {
            "validation_current_pa": protocols.validation_current,
            "validation_sweep_number": protocols.validation_sweep_number,
        },
    )
    return Overlay(
        observed=observed,
        simulation=simulation,
        protocols=protocols,
        metrics=metrics,
        recording_qc=recording_qc_metrics(observed),
    )


def _new_quality_rank(frame: pd.DataFrame) -> np.ndarray:
    columns = (
        "phase_chamfer",
        "relative_spike_count_error",
        "first_three_isi_rmse_ms",
        "onset_30ms_voltage_rmse_mv",
    )
    ranks = [
        pd.to_numeric(frame[column], errors="coerce")
        .rank(pct=True, ascending=True, na_option="bottom")
        .to_numpy(dtype=float)
        for column in columns
    ]
    return np.mean(np.column_stack(ranks), axis=1)


def _candidate_order(family: pd.DataFrame) -> pd.DataFrame:
    ranked = rank_representative_candidates(family)
    ranked["new_quality_rank"] = _new_quality_rank(ranked)
    quality_limit = ranked["new_quality_rank"].median()
    ranked["new_better_half"] = (
        ranked["new_quality_rank"] <= quality_limit
    )
    ranked["new_priority"] = (~ranked["new_better_half"]).astype(int)
    return ranked.sort_values(
        [
            "new_priority",
            "selection_typicality",
            "new_quality_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _choose_representatives(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    dataset: str,
) -> list[tuple[pd.Series, Overlay]]:
    new_metrics = metrics.loc[
        metrics["dataset"].eq(dataset)
        & metrics["model"].eq("recovery_onset")
    ].copy()
    merged = cohort.loc[cohort["dataset"].eq(dataset)].merge(
        new_metrics[
            [
                "cell_id",
                "phase_chamfer",
                "relative_spike_count_error",
                "first_three_isi_rmse_ms",
                "onset_30ms_voltage_rmse_mv",
            ]
        ],
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    representatives = []
    for family in major_families(merged, minimum_cells=10):
        candidates = _candidate_order(
            merged.loc[merged["broad_class"].eq(family)]
        )
        attempts = []
        for _, candidate in candidates.iterrows():
            if str(candidate["cell_id"]) in EXCLUDED_CELL_IDS:
                continue
            overlay = _fit_overlay(candidate)
            attempts.append((candidate, overlay))
            if bool(overlay.recording_qc["recording_qc_pass"]):
                representatives.append((candidate, overlay))
                break
        else:
            if not attempts:
                raise RuntimeError(f"No replayable candidate for {family}")
            representatives.append(
                min(
                    attempts,
                    key=lambda item: float(
                        item[1].recording_qc["recording_qc_score"]
                    ),
                )
            )
    return representatives


def _plot_full_trace(axis: plt.Axes, overlay: Overlay) -> None:
    observed = overlay.observed
    simulation = overlay.simulation
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
        linewidth=0.75,
        label="Biological",
    )
    axis.plot(
        relative,
        simulation.voltage_mv,
        color=MODEL_COLOR,
        linewidth=1.0,
        label="Recovery + onset",
    )
    axis.set_xlim(-20.0, duration + 20.0)
    axis.set_xlabel("Time from onset (ms)")
    axis.set_ylabel("V (mV)")


def _plot_early_trace(axis: plt.Axes, overlay: Overlay) -> None:
    observed = overlay.observed
    simulation = overlay.simulation
    relative = observed.time_ms - observed.stimulus_start_ms
    duration = observed.stimulus_end_ms - observed.stimulus_start_ms
    stop = min(120.0, max(50.0, 0.25 * duration))
    axis.axvspan(
        0.0,
        stop,
        color=STIMULUS_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    axis.plot(
        relative,
        observed.voltage_mv,
        color=BIOLOGICAL_COLOR,
        linewidth=1.0,
    )
    axis.plot(
        relative,
        simulation.voltage_mv,
        color=MODEL_COLOR,
        linewidth=1.2,
    )
    axis.set_xlim(-5.0, stop)
    axis.set_xlabel("Early response (ms)")
    axis.set_ylabel("V (mV)")


def _plot_phase(axis: plt.Axes, overlay: Overlay) -> None:
    observed = overlay.observed
    simulation = overlay.simulation
    highlighted_values = []
    for voltage, velocity, color in (
        (
            observed.voltage_mv,
            observed.velocity_mv_ms,
            "#A0A0A0",
        ),
        (
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
            "#E9A2AE",
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
                linewidth=0.5,
                alpha=0.22,
            )
    for voltage, velocity, color in (
        (
            observed.voltage_mv,
            observed.velocity_mv_ms,
            BIOLOGICAL_COLOR,
        ),
        (
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
            MODEL_COLOR,
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
            linewidth=1.4,
        )
        highlighted_values.append(velocity[cycle])
    combined_velocity = np.concatenate(highlighted_values)
    lower, upper = np.percentile(combined_velocity, (0.5, 99.5))
    margin = max(5.0, 0.1 * float(upper - lower))
    axis.set_ylim(float(lower - margin), float(upper + margin))
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.7)
    axis.set_xlabel("V (mV)")
    axis.set_ylabel("dV/dt (mV/ms)")


def _summary_row(
    candidate: pd.Series,
    overlay: Overlay,
) -> dict[str, object]:
    return {
        "dataset": candidate["dataset"],
        "broad_class": candidate["broad_class"],
        "cell_id": candidate["cell_id"],
        "transcriptomic_type": candidate["transcriptomic_type"],
        "validation_current_pa": overlay.protocols.validation_current,
        "validation_sweep_number": (
            overlay.protocols.validation_sweep_number
        ),
        **overlay.recording_qc,
        **overlay.metrics,
    }


def _plot_dataset(
    dataset: str,
    representatives: list[tuple[pd.Series, Overlay]],
    output_root: Path,
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        len(representatives),
        3,
        figsize=(17.0, 2.65 * len(representatives)),
        gridspec_kw={"width_ratios": (1.65, 1.0, 1.0)},
        constrained_layout=True,
        squeeze=False,
    )
    rows = []
    for index, (candidate, overlay) in enumerate(representatives):
        _plot_full_trace(axes[index, 0], overlay)
        _plot_early_trace(axes[index, 1], overlay)
        _plot_phase(axes[index, 2], overlay)
        metrics = overlay.metrics
        qc = overlay.recording_qc
        axes[index, 0].set_title(
            f"{candidate['broad_class']} | "
            f"{candidate['transcriptomic_type']}\n"
            f"cell {candidate['cell_id']} | held-out "
            f"{overlay.protocols.validation_current:g} pA | "
            f"spikes {metrics['observed_spike_count']}/"
            f"{metrics['predicted_spike_count']} (bio/model)",
            loc="left",
            fontsize=9,
        )
        axes[index, 1].set_title(
            "Onset and initial burst | amplitude retention "
            f"{float(qc['spike_amplitude_retention']):.2f}",
            loc="left",
            fontsize=9,
        )
        axes[index, 2].set_title(
            "Stable phase loop | Chamfer "
            f"{float(metrics['phase_chamfer']):.3f}",
            loc="left",
            fontsize=9,
        )
        if index == 0:
            axes[index, 0].legend(
                frameon=False,
                fontsize=8,
                ncol=2,
                loc="upper right",
            )
        for axis in axes[index]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        rows.append(_summary_row(candidate, overlay))
    figure.suptitle(
        f"{DATASET_LABELS[dataset]}: recovery + onset held-out replay\n"
        "Representatives pass baseline stability, AP retention, "
        "and sustained-spiking display QC",
        fontsize=14,
    )
    destination = (
        output_root / f"{dataset}_recovery_onset_family_overlays.png"
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(rows)


def parse_args() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population-root",
        type=Path,
        default=Path("outputs/recovery_onset_population_168"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recovery_onset_family_overlays"),
    )
    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()
    population_root = args["population_root"]
    output_root = args["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(
        population_root / "cohort.csv",
        dtype={"cell_id": "string"},
    )
    metrics = pd.read_csv(
        population_root / "model_metrics.csv",
        dtype={"cell_id": "string"},
    )
    summaries = []
    for dataset in DATASET_LABELS:
        representatives = _choose_representatives(
            cohort,
            metrics,
            dataset,
        )
        summaries.append(
            _plot_dataset(dataset, representatives, output_root)
        )
    pd.concat(summaries, ignore_index=True).to_csv(
        output_root / "representative_cells.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
