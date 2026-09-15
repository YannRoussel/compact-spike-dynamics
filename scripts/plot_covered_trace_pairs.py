#!/usr/bin/env python3
"""Compare covered biological spike traces with their nearest HH models."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.biological_data import (
    SPIKE_CYCLE_COMMON_FEATURES,
)
from inverse_ephys_alpha_beta.features import (
    FeatureConfig,
    VoltageFeatureTrace,
    detect_spikes,
    voltage_feature_trace,
)
from inverse_ephys_alpha_beta.hh_model import Stimulus
from inverse_ephys_alpha_beta.kinetics import KineticParameters
from inverse_ephys_alpha_beta.protocols import (
    biological_screen_config,
    run_biological_screen,
)
from inverse_ephys_alpha_beta.raw_patchseq import (
    CurrentClampSweep,
    read_current_clamp_sweeps,
)
from inverse_ephys_alpha_beta.static_parameters import StaticParameterTransforms

BIOLOGICAL_COLOR = "#262626"
MODEL_COLOR = "#D1495B"
STIMULUS_COLOR = "#DDE8F0"


@dataclass(frozen=True)
class SpikeWindow:
    threshold_index: int
    peak_index: int
    stop_index: int


def _normalized_id(value: object) -> str:
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _covered_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


def select_trace_pairs(
    coverage: pd.DataFrame,
    harmonized_models: pd.DataFrame,
    raw_features: pd.DataFrame,
    max_covered: int,
    near_misses: int,
) -> pd.DataFrame:
    """Resolve coverage rows to model IDs and raw NWB sweep provenance."""
    complete_models = harmonized_models.dropna(
        subset=list(SPIKE_CYCLE_COMMON_FEATURES)
    ).reset_index(drop=True)
    ordered = coverage.sort_values("distance_ratio").copy()
    covered = ordered.loc[_covered_mask(ordered["covered"])].head(max_covered)
    uncovered = ordered.loc[~_covered_mask(ordered["covered"])].head(near_misses)
    selected = pd.concat((covered, uncovered), ignore_index=True)
    if selected.empty:
        raise ValueError("No covered cells or near-threshold cells were selected")

    nearest_ids = []
    for model_row in selected["nearest_model_row"].astype(int):
        if not 0 <= model_row < len(complete_models):
            raise IndexError(f"Nearest model row is outside the model table: {model_row}")
        nearest_ids.append(
            _normalized_id(complete_models.iloc[model_row]["cell_id"])
        )
    selected["nearest_model_id"] = nearest_ids

    provenance_columns = [
        "dataset",
        "cell_id",
        "nwb_path",
        "sweep_number",
        "sampling_rate_hz",
        "sampled_rheobase_pa",
        "stimulus_start_ms",
        "stimulus_end_ms",
    ]
    provenance = raw_features.loc[:, provenance_columns].copy()
    provenance["cell_id"] = provenance["cell_id"].astype(str)
    selected["cell_id"] = selected["cell_id"].astype(str)
    selected = selected.merge(
        provenance,
        on=["dataset", "cell_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["nwb_path"].isna().any():
        missing = selected.loc[selected["nwb_path"].isna(), "cell_id"].tolist()
        raise ValueError(f"Raw NWB provenance is missing for cells: {missing}")
    return selected


def _load_sweep(pair: pd.Series) -> CurrentClampSweep:
    sweep_number = int(pair["sweep_number"])
    sweeps = read_current_clamp_sweeps(pair["nwb_path"])
    matches = [sweep for sweep in sweeps if sweep.sweep_number == sweep_number]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sweep {sweep_number} for {pair['cell_id']}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _first_spike_window(
    trace: VoltageFeatureTrace,
    stimulus: Stimulus,
    feature_config: FeatureConfig,
) -> SpikeWindow:
    detection = detect_spikes(trace, stimulus, feature_config)
    if not len(detection.peak_indices):
        raise ValueError("Selected trace contains no detected spikes")
    threshold = int(detection.threshold_indices[0])
    peak = int(detection.peak_indices[0])
    if len(detection.threshold_indices) > 1:
        stop = int(detection.threshold_indices[1])
    else:
        cycle_end = trace.time_ms[threshold] + feature_config.max_cycle_duration_ms
        stop = int(np.searchsorted(trace.time_ms, cycle_end, side="right") - 1)
    stop = min(max(stop, peak + 1), len(trace.time_ms) - 1)
    return SpikeWindow(threshold, peak, stop)


def _baseline_voltage(trace: VoltageFeatureTrace, stimulus: Stimulus) -> float:
    before = trace.voltage_mv[trace.time_ms < stimulus.start_ms]
    return float(np.median(before)) if len(before) else float(trace.voltage_mv[0])


def _plot_full_trace(
    axis: plt.Axes,
    biological: VoltageFeatureTrace,
    biological_stimulus: Stimulus,
    model: VoltageFeatureTrace,
    model_stimulus: Stimulus,
    biological_current_pa: float,
    model_current_pa: float,
) -> None:
    biological_time = biological.time_ms - biological_stimulus.start_ms
    model_time = model.time_ms - model_stimulus.start_ms
    duration = max(
        biological_stimulus.end_ms - biological_stimulus.start_ms,
        model_stimulus.end_ms - model_stimulus.start_ms,
    )
    axis.axvspan(0.0, duration, color=STIMULUS_COLOR, alpha=0.55, linewidth=0)
    axis.plot(
        biological_time,
        biological.voltage_mv,
        color=BIOLOGICAL_COLOR,
        linewidth=0.8,
        label=f"Biological ({biological_current_pa:g} pA)",
    )
    axis.plot(
        model_time,
        model.voltage_mv,
        color=MODEL_COLOR,
        linewidth=1.0,
        alpha=0.9,
        label=f"Nearest HH ({model_current_pa:g} pA)",
    )
    axis.set_xlim(-20.0, duration)
    axis.set_ylabel("V (mV)")
    axis.set_xlabel("Time from stimulus onset (ms)")
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def _plot_aligned_spike(
    axis: plt.Axes,
    biological: VoltageFeatureTrace,
    biological_window: SpikeWindow,
    model: VoltageFeatureTrace,
    model_window: SpikeWindow,
) -> None:
    for trace, window, color, label in (
        (
            biological,
            biological_window,
            BIOLOGICAL_COLOR,
            "Biological",
        ),
        (model, model_window, MODEL_COLOR, "Nearest HH"),
    ):
        aligned_time = trace.time_ms - trace.time_ms[window.threshold_index]
        mask = (aligned_time >= -2.5) & (aligned_time <= 8.0)
        axis.plot(
            aligned_time[mask],
            trace.voltage_mv[mask],
            color=color,
            linewidth=1.5,
            label=label,
        )
        axis.scatter(
            0.0,
            trace.voltage_mv[window.threshold_index],
            color=color,
            s=22,
            zorder=3,
        )
    axis.axvline(0.0, color="#A0A0A0", linewidth=0.8, linestyle=":")
    axis.set_xlim(-2.5, 8.0)
    axis.set_xlabel("Time from AP threshold (ms)")
    axis.set_ylabel("V (mV)")


def _plot_phase_loop(
    axis: plt.Axes,
    biological: VoltageFeatureTrace,
    biological_window: SpikeWindow,
    model: VoltageFeatureTrace,
    model_window: SpikeWindow,
) -> None:
    for trace, window, color, label in (
        (
            biological,
            biological_window,
            BIOLOGICAL_COLOR,
            "Biological",
        ),
        (model, model_window, MODEL_COLOR, "Nearest HH"),
    ):
        cycle = slice(window.threshold_index, window.stop_index + 1)
        axis.plot(
            trace.voltage_mv[cycle],
            trace.dvdt_mv_ms[cycle],
            color=color,
            linewidth=1.4,
            label=label,
        )
        axis.scatter(
            trace.voltage_mv[window.threshold_index],
            trace.dvdt_mv_ms[window.threshold_index],
            color=color,
            s=22,
            zorder=3,
        )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axis.set_xlabel("V (mV)")
    axis.set_ylabel("dV/dt (mV/ms)")


def plot_trace_pairs(
    pair_table: pd.DataFrame,
    model_population: pd.DataFrame,
    preset: str,
    temperature_c: float,
    output_path: str | Path,
    title: str,
) -> pd.DataFrame:
    """Simulate nearest models and plot full, aligned, and phase-plane traces."""
    model_rows = model_population.copy()
    model_rows["_model_id"] = model_rows["sample_id"].map(_normalized_id)
    model_rows = model_rows.set_index("_model_id", drop=False)
    feature_config = FeatureConfig(min_spikes=1)
    screen_config = biological_screen_config(
        preset,
        temperature_c=temperature_c,
    )
    figure, axes = plt.subplots(
        len(pair_table),
        3,
        figsize=(15.5, 3.45 * len(pair_table)),
        squeeze=False,
    )
    summary_rows = []

    for row_index, pair in pair_table.reset_index(drop=True).iterrows():
        model_id = _normalized_id(pair["nearest_model_id"])
        if model_id not in model_rows.index:
            raise ValueError(f"Model {model_id} is missing from the population")
        model_row = model_rows.loc[model_id]
        if isinstance(model_row, pd.DataFrame):
            model_row = model_row.iloc[0]
        stored_waveform_ms = float(
            model_row.get("feature__waveform_step_ms", np.nan)
        )
        if (
            np.isfinite(stored_waveform_ms)
            and not np.isclose(
                stored_waveform_ms,
                screen_config.waveform_step_ms,
            )
        ):
            raise ValueError(
                f"Model {model_id} stored a {stored_waveform_ms:g} ms waveform, "
                f"but preset {preset} uses {screen_config.waveform_step_ms:g} ms"
            )

        sweep = _load_sweep(pair)
        if (
            sweep.stimulus_start_ms is None
            or sweep.stimulus_end_ms is None
            or sweep.stimulus_amplitude_pa is None
        ):
            raise ValueError(f"Sweep stimulus metadata is incomplete for {pair['cell_id']}")
        biological_stimulus = Stimulus(
            amplitude_pa=float(sweep.stimulus_amplitude_pa),
            amplitude_ua_cm2=None,
            start_ms=float(sweep.stimulus_start_ms),
            end_ms=float(sweep.stimulus_end_ms),
        )
        biological_trace = voltage_feature_trace(
            sweep.time_ms,
            sweep.voltage_mv,
            filter_window_ms=feature_config.voltage_filter_window_ms,
            polynomial_order=feature_config.voltage_filter_polynomial_order,
        )
        biological_window = _first_spike_window(
            biological_trace,
            biological_stimulus,
            feature_config,
        )

        kinetics = KineticParameters.from_mapping(model_row)
        biophysics = StaticParameterTransforms.from_mapping(
            model_row
        ).to_biophysics()
        model_result = run_biological_screen(
            kinetics,
            biophysics,
            screen_config,
            feature_config,
        )
        raw_model_trace = model_result.waveform_trace
        model_stimulus = Stimulus(
            amplitude_pa=float(model_result.features["waveform_current_pa"]),
            amplitude_ua_cm2=None,
            start_ms=screen_config.baseline_ms,
            end_ms=screen_config.baseline_ms + screen_config.waveform_step_ms,
        )
        model_trace = voltage_feature_trace(
            raw_model_trace.time_ms,
            raw_model_trace.voltage_mv,
            filter_window_ms=feature_config.voltage_filter_window_ms,
            polynomial_order=feature_config.voltage_filter_polynomial_order,
        )
        model_window = _first_spike_window(
            model_trace,
            model_stimulus,
            feature_config,
        )

        biological_current = float(sweep.stimulus_amplitude_pa)
        model_current = float(model_result.features["waveform_current_pa"])
        _plot_full_trace(
            axes[row_index, 0],
            biological_trace,
            biological_stimulus,
            model_trace,
            model_stimulus,
            biological_current,
            model_current,
        )
        _plot_aligned_spike(
            axes[row_index, 1],
            biological_trace,
            biological_window,
            model_trace,
            model_window,
        )
        _plot_phase_loop(
            axes[row_index, 2],
            biological_trace,
            biological_window,
            model_trace,
            model_window,
        )

        covered = bool(_covered_mask(pd.Series([pair["covered"]])).iloc[0])
        status = "covered" if covered else "first outside threshold"
        axes[row_index, 0].set_title(
            f"{pair['cell_id']} | {status} | radius ratio "
            f"{float(pair['distance_ratio']):.3f}\nnearest model {model_id}",
            loc="left",
            fontsize=10,
        )
        axes[row_index, 1].set_title("First AP aligned at threshold", fontsize=10)
        axes[row_index, 2].set_title("First-cycle phase plane", fontsize=10)

        biological_detection = detect_spikes(
            biological_trace,
            biological_stimulus,
            feature_config,
        )
        model_detection = detect_spikes(
            model_trace,
            model_stimulus,
            feature_config,
        )
        summary_rows.append(
            {
                "dataset": pair["dataset"],
                "cell_id": pair["cell_id"],
                "covered": covered,
                "distance_ratio": float(pair["distance_ratio"]),
                "nearest_model_id": model_id,
                "model_protocol": screen_config.protocol_name,
                "model_waveform_step_ms": screen_config.waveform_step_ms,
                "nwb_path": pair["nwb_path"],
                "sweep_number": int(pair["sweep_number"]),
                "biological_current_pa": biological_current,
                "model_rheobase_pa": model_current,
                "rheobase_ratio_model_to_biological": (
                    model_current / biological_current
                    if biological_current
                    else np.nan
                ),
                "current_fold_mismatch": (
                    max(model_current, biological_current)
                    / min(model_current, biological_current)
                    if model_current > 0.0 and biological_current > 0.0
                    else np.nan
                ),
                "biological_spike_count": len(
                    biological_detection.peak_indices
                ),
                "model_spike_count": len(model_detection.peak_indices),
                "spike_count_abs_difference": abs(
                    len(biological_detection.peak_indices)
                    - len(model_detection.peak_indices)
                ),
                "biological_baseline_mv": _baseline_voltage(
                    biological_trace,
                    biological_stimulus,
                ),
                "model_baseline_mv": _baseline_voltage(
                    model_trace,
                    model_stimulus,
                ),
                "baseline_abs_difference_mv": abs(
                    _baseline_voltage(
                        biological_trace,
                        biological_stimulus,
                    )
                    - _baseline_voltage(
                        model_trace,
                        model_stimulus,
                    )
                ),
                "biological_threshold_mv": biological_trace.voltage_mv[
                    biological_window.threshold_index
                ],
                "model_threshold_mv": model_trace.voltage_mv[
                    model_window.threshold_index
                ],
                "biological_peak_mv": biological_trace.voltage_mv[
                    biological_window.peak_index
                ],
                "model_peak_mv": model_trace.voltage_mv[
                    model_window.peak_index
                ],
            }
        )

    figure.suptitle(title, fontsize=16, y=0.998)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    return summary


def parse_args() -> dict[str, object]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("coverage_dir", type=Path)
    parser.add_argument("model_population", type=Path)
    parser.add_argument("raw_spike_cycles", type=Path)
    parser.add_argument(
        "--preset",
        choices=("fast", "scala", "gouwens"),
        required=True,
        help="Protocol that generated the model features used for coverage",
    )
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-covered", type=int, default=4)
    parser.add_argument("--near-misses", type=int, default=1)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return vars(parser.parse_args())


def main(
    coverage_dir: Path,
    model_population: Path,
    raw_spike_cycles: Path,
    preset: str,
    temperature: float,
    max_covered: int,
    near_misses: int,
    title: str,
    output: Path,
) -> None:
    coverage = pd.read_csv(
        coverage_dir / "biological_coverage.csv",
        dtype={"cell_id": "string"},
    )
    harmonized_models = pd.read_csv(
        coverage_dir / "harmonized_model_features.csv",
        dtype={"cell_id": "string"},
    )
    raw_features = pd.read_csv(
        raw_spike_cycles,
        dtype={"cell_id": "string"},
    )
    pairs = select_trace_pairs(
        coverage,
        harmonized_models,
        raw_features,
        max_covered=max_covered,
        near_misses=near_misses,
    )
    model_frame = pd.read_csv(model_population)
    summary = plot_trace_pairs(
        pairs,
        model_frame,
        preset=preset,
        temperature_c=temperature,
        output_path=output,
        title=title,
    )
    print(summary.to_string(index=False))
    print(f"Trace comparison: {output}")


if __name__ == "__main__":
    main(**parse_args())
