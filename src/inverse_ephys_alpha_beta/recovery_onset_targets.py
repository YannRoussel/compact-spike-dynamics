"""Population fitting of recovery-onset dynamics on existing compact loops."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .phase_population import select_phase_population_protocols
from .phase_template_model import extract_phase_cycles
from .recovery_onset import (
    OnsetModel,
    fit_onset_model,
    fit_recovery_timing,
)
from .recovery_onset_compact import recovery_onset_parameter_row


def _trace_current(trace) -> float:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    return float(np.median(trace.input_value[during]))


def fit_recovery_onset_target(
    compact_row: Mapping[str, object],
) -> dict[str, object]:
    """Append all-current recovery/onset parameters to one compact fit."""
    protocols = select_phase_population_protocols(str(compact_row["nwb_path"]))
    traces = tuple(
        sorted(
            (*protocols.training, protocols.validation),
            key=_trace_current,
        )
    )
    cycles = tuple(
        cycle
        for trace in traces
        if (cycle := extract_phase_cycles(trace)) is not None
    )
    if not cycles:
        raise ValueError("No complete phase cycles")
    resting = float(compact_row["resting_voltage_mv"])
    timing, candidates = fit_recovery_timing(
        cycles,
        voltage_reference_mv=resting,
    )
    try:
        onset = fit_onset_model(traces)
        onset_status = "fitted"
    except ValueError:
        levels = tuple(float(cycle.current_value) for cycle in cycles)
        onset = OnsetModel(
            current_levels=levels,
            tau_ms=tuple(1.0 for _ in levels),
            fit_rmse_mv=tuple(float("nan") for _ in levels),
        )
        onset_status = "default_1ms"
    selected = next(row for row in candidates if row["tau_selected"])
    row = recovery_onset_parameter_row(compact_row, timing, onset)
    row.update(
        {
            "onset_fit_status": onset_status,
            "recovery_late_period_nrmse": selected["late_period_nrmse"],
            "recovery_cycle_current_count": len(cycles),
        }
    )
    return row


def recovery_onset_target_task(
    compact_row: Mapping[str, object],
) -> dict[str, object]:
    """Process-safe wrapper returning failures as rows."""
    try:
        return {
            "status": "ok",
            "row": fit_recovery_onset_target(compact_row),
        }
    except Exception as error:
        return {
            "status": "failed",
            "dataset": str(compact_row["dataset"]),
            "cell_id": str(compact_row["cell_id"]),
            "nwb_path": str(compact_row["nwb_path"]),
            "failure_reason": f"{type(error).__name__}: {error}",
        }
