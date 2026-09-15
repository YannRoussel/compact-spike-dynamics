"""Representative-cell selection and held-out compact-model replay helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .abstract_phase_model import AbstractPhaseConfig, AbstractObservedTrace
from .model_transcriptomics import high_qc_mask
from .phase_population import (
    PhasePopulationConfig,
    PhasePopulationProtocols,
    select_phase_population_protocols,
)
from .phase_template_model import (
    PhaseTemplateConfig,
    PhaseTemplateSimulation,
    fit_phase_template_ladder,
)
from .phase_timing import (
    PhaseTimingConfig,
    fit_spike_exponential_timing,
    simulate_phase_timing_model,
)
from .phase_timing_population import _metric_row, _spike_times_ms


FAMILY_ORDER = (
    "Pvalb",
    "Sst",
    "Vip",
    "Lamp5",
    "Sncg",
    "Glut_IT",
    "Glut_ET",
    "Glut_CT",
    "Glut_L6b",
    "Glut_NP",
)

QUALITY_COLUMNS = (
    "validation_phase_chamfer",
    "validation_relative_spike_count_error",
    "validation_spike_time_rmse_ms",
    "timing_late_period_nrmse",
)

TYPICALITY_COLUMNS = (
    "resting_voltage_mv",
    "rheobase_current_pa",
    "baseline_period_q025_ms",
    "baseline_period_q050_ms",
    "baseline_period_q075_ms",
    "timing_tau_ms",
    "timing_spike_jump",
    "template_min_relative_mv_q050",
    "template_max_relative_mv_q050",
    "downstroke_duration_ms_q050",
    "upstroke_duration_ms_q050",
    "template_cos_h01_mv_q050",
    "template_sin_h01_mv_q050",
    "template_cos_h02_mv_q050",
    "template_sin_h02_mv_q050",
)


@dataclass(frozen=True)
class HeldOutOverlay:
    """Observed held-out trace, its simulation, and validation metrics."""

    protocols: PhasePopulationProtocols
    observed: AbstractObservedTrace
    simulation: PhaseTemplateSimulation
    metrics: dict[str, object]


def normalized_cell_id(value: object) -> str:
    """Normalize numeric cell identifiers without altering textual IDs."""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def major_families(
    metadata: pd.DataFrame,
    minimum_cells: int = 10,
) -> tuple[str, ...]:
    """Return ordered broad families meeting a predeclared size threshold."""
    counts = metadata["broad_class"].value_counts()
    eligible = {
        str(name)
        for name, count in counts.items()
        if int(count) >= minimum_cells and str(name) != "Other"
    }
    ordered = [name for name in FAMILY_ORDER if name in eligible]
    ordered.extend(sorted(eligible.difference(ordered)))
    return tuple(ordered)


def merge_parameters_and_metadata(
    parameters: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Join compact fits to paired transcriptomic class labels."""
    fit = parameters.loc[parameters["dataset"].eq(dataset)].copy()
    labels = metadata.copy()
    fit["cell_id"] = fit["cell_id"].map(normalized_cell_id)
    labels["cell_id"] = labels["cell_id"].map(normalized_cell_id)
    merged = fit.merge(
        labels[
            [
                "cell_id",
                "donor_id",
                "transcriptomic_type",
                "broad_class",
            ]
        ],
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    merged["high_qc"] = high_qc_mask(merged)
    return merged


def _rank_quality(frame: pd.DataFrame) -> np.ndarray:
    ranks = []
    for name in QUALITY_COLUMNS:
        values = pd.to_numeric(frame[name], errors="coerce")
        ranks.append(
            values.rank(
                method="average",
                pct=True,
                ascending=True,
                na_option="bottom",
            ).to_numpy(dtype=float)
        )
    return np.mean(np.column_stack(ranks), axis=1)


def _robust_typicality(frame: pd.DataFrame) -> np.ndarray:
    distances = []
    for name in TYPICALITY_COLUMNS:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(values)
        if np.sum(finite) < 3:
            continue
        median = float(np.nanmedian(values))
        q25, q75 = np.nanpercentile(values, (25.0, 75.0))
        scale = float(q75 - q25)
        if scale <= 1e-12:
            scale = float(np.nanmedian(np.abs(values - median)) * 1.4826)
        if scale <= 1e-12:
            continue
        standardized = (values - median) / scale
        standardized[~finite] = 3.0
        distances.append(np.square(np.clip(standardized, -5.0, 5.0)))
    if not distances:
        return np.zeros(len(frame), dtype=float)
    return np.mean(np.column_stack(distances), axis=1)


def rank_representative_candidates(
    family_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Rank typical cells within the better half of held-out fits.

    Quality only defines the candidate half. Within that half, distance to the
    class median in compact-parameter space chooses the representative.
    """
    ranked = family_frame.copy().reset_index(drop=True)
    ranked["selection_quality_rank"] = _rank_quality(ranked)
    ranked["selection_typicality"] = _robust_typicality(ranked)
    candidate_count = max(1, int(math.ceil(len(ranked) / 2.0)))
    quality_order = np.argsort(
        ranked["selection_quality_rank"].to_numpy(dtype=float),
        kind="stable",
    )
    preferred = set(int(index) for index in quality_order[:candidate_count])
    ranked["selection_better_half"] = [
        index in preferred for index in range(len(ranked))
    ]
    ranked["selection_priority"] = (
        ~ranked["selection_better_half"]
    ).astype(int)
    return ranked.sort_values(
        [
            "selection_priority",
            "selection_typicality",
            "selection_quality_rank",
            "cell_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def fit_held_out_overlay(
    row: pd.Series,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    timing_config: PhaseTimingConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> HeldOutOverlay:
    """Refit a compact model without the interior validation current."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    phase_config = phase_config or PhaseTemplateConfig()
    timing_config = timing_config or PhaseTimingConfig()
    protocols = select_phase_population_protocols(
        str(row["nwb_path"]),
        derivative_config,
        population_config,
    )
    first_training = protocols.training[0]
    baseline = first_training.time_ms < first_training.stimulus_start_ms
    resting_voltage = float(np.median(first_training.voltage_mv[baseline]))
    waveform = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    timing = fit_spike_exponential_timing(
        waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    observed = protocols.validation
    simulation = simulate_phase_timing_model(
        waveform.no_slow_model,
        timing.model,
        observed.time_ms,
        observed.input_value,
        float(observed.voltage_mv[0]),
    )
    metrics = _metric_row(
        str(row["dataset"]),
        str(row["cell_id"]),
        str(row["nwb_path"]),
        "spike_exponential",
        True,
        observed,
        simulation,
        {
            "validation_current_pa": protocols.validation_current,
            "validation_sweep_number": protocols.validation_sweep_number,
        },
    )
    return HeldOutOverlay(
        protocols=protocols,
        observed=observed,
        simulation=simulation,
        metrics=metrics,
    )


def stable_cycle_slices(
    time_ms: Iterable[float],
    voltage_mv: Iterable[float],
    stimulus_start_ms: float,
    stimulus_end_ms: float,
) -> tuple[slice, ...]:
    """Find complete stable cycles between consecutive upward 0-mV crossings."""
    time = np.asarray(time_ms, dtype=float)
    voltage = np.asarray(voltage_mv, dtype=float)
    during = (time >= stimulus_start_ms) & (time < stimulus_end_ms)
    crossings = np.flatnonzero(
        (voltage[1:] >= 0.0)
        & (voltage[:-1] < 0.0)
        & during[1:]
    ) + 1
    pairs = list(zip(crossings[1:-1], crossings[2:]))
    return tuple(
        slice(int(start), int(stop) + 1)
        for start, stop in pairs
        if stop > start
    )


def typical_cycle_slice(
    time_ms: Iterable[float],
    voltage_mv: Iterable[float],
    stimulus_start_ms: float,
    stimulus_end_ms: float,
) -> slice:
    """Select the stable cycle whose period is closest to the median."""
    time = np.asarray(time_ms, dtype=float)
    cycles = stable_cycle_slices(
        time,
        voltage_mv,
        stimulus_start_ms,
        stimulus_end_ms,
    )
    if not cycles:
        during = np.flatnonzero(
            (time >= stimulus_start_ms) & (time < stimulus_end_ms)
        )
        if len(during) < 2:
            raise ValueError("No samples in stimulus epoch")
        return slice(int(during[0]), int(during[-1]) + 1)
    periods = np.asarray(
        [time[cycle.stop - 1] - time[cycle.start] for cycle in cycles],
        dtype=float,
    )
    median = float(np.median(periods))
    return cycles[int(np.argmin(np.abs(periods - median)))]


def recording_qc_metrics(
    trace: AbstractObservedTrace,
) -> dict[str, float | bool]:
    """Measure baseline stability and spike-train health for display choices."""
    before_indices = np.flatnonzero(
        trace.time_ms < trace.stimulus_start_ms
    )
    if len(before_indices) < 4:
        raise ValueError("Insufficient baseline samples for recording QC")
    before = trace.voltage_mv[before_indices]
    midpoint = max(1, len(before) // 2)
    baseline = float(np.median(before))
    baseline_drift = float(
        abs(
            np.median(before[midpoint:])
            - np.median(before[:midpoint])
        )
    )
    baseline_noise = float(
        np.median(np.abs(before - np.median(before))) * 1.4826
    )
    spikes = _spike_times_ms(trace, trace.voltage_mv)
    spike_indices = np.searchsorted(trace.time_ms, spikes)
    dt_ms = float(np.median(np.diff(trace.time_ms)))
    peak_width = max(1, int(round(3.0 / dt_ms)))
    peaks = np.asarray(
        [
            np.max(
                trace.voltage_mv[
                    index:min(len(trace.voltage_mv), index + peak_width)
                ]
            )
            for index in spike_indices
        ],
        dtype=float,
    )
    amplitudes = peaks - baseline
    width = min(3, len(amplitudes))
    amplitude_retention = (
        float(
            np.median(amplitudes[-width:])
            / max(1e-6, np.median(amplitudes[:width]))
        )
        if width
        else float("nan")
    )
    duration = trace.stimulus_end_ms - trace.stimulus_start_ms
    last_spike_fraction = (
        float((spikes[-1] - trace.stimulus_start_ms) / duration)
        if len(spikes) and duration > 0.0
        else 0.0
    )
    after = trace.voltage_mv[
        trace.time_ms >= trace.stimulus_end_ms + 10.0
    ]
    post_step_error = (
        float(abs(np.median(after) - baseline))
        if len(after)
        else float("nan")
    )
    first_amplitude = (
        float(amplitudes[0]) if len(amplitudes) else 0.0
    )
    passed = bool(
        -90.0 <= baseline <= -45.0
        and baseline_noise <= 1.5
        and baseline_drift <= 2.5
        and first_amplitude >= 35.0
        and (
            not np.isfinite(amplitude_retention)
            or amplitude_retention >= 0.7
        )
        and last_spike_fraction >= 0.5
        and (
            not np.isfinite(post_step_error)
            or post_step_error <= 15.0
        )
    )
    score = (
        baseline_noise / 1.5
        + baseline_drift / 2.5
        + max(0.0, 0.7 - amplitude_retention) * 5.0
        + max(0.0, 0.5 - last_spike_fraction) * 5.0
        + max(0.0, 35.0 - first_amplitude) / 10.0
        + (
            max(0.0, post_step_error - 15.0) / 10.0
            if np.isfinite(post_step_error)
            else 0.0
        )
    )
    return {
        "recording_qc_pass": passed,
        "baseline_voltage_mv": baseline,
        "baseline_noise_mad_mv": baseline_noise,
        "baseline_drift_mv": baseline_drift,
        "first_spike_amplitude_mv": first_amplitude,
        "spike_amplitude_retention": amplitude_retention,
        "last_spike_fraction": last_spike_fraction,
        "post_step_recovery_error_mv": post_step_error,
        "recording_qc_score": float(score),
    }
