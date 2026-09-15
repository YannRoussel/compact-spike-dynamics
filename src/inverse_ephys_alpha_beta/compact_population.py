"""All-cell fitting for the compact phase-template plus spike-memory model."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .abstract_phase_model import AbstractPhaseConfig
from .compact_model import compact_model_parameter_row
from .phase_population import (
    PhasePopulationConfig,
    select_phase_population_protocols,
)
from .phase_template_model import (
    PhaseTemplateConfig,
    fit_phase_template_ladder,
)
from .phase_timing import (
    PhaseTimingConfig,
    fit_spike_exponential_timing,
    simulate_phase_timing_model,
)
from .phase_timing_population import _metric_row


def _trace_current(trace) -> float:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    return float(np.median(trace.input_value[during]))


def fit_compact_population_cell(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    timing_config: PhaseTimingConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Cross-validate one cell, then refit the compact model on all currents."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    phase_config = phase_config or PhaseTemplateConfig()
    timing_config = timing_config or PhaseTimingConfig()
    protocols = select_phase_population_protocols(
        nwb_path,
        derivative_config,
        population_config,
    )
    baseline = (
        protocols.training[0].time_ms
        < protocols.training[0].stimulus_start_ms
    )
    resting_voltage = float(
        np.median(protocols.training[0].voltage_mv[baseline])
    )

    training_waveform = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    training_timing = fit_spike_exponential_timing(
        training_waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    validation = protocols.validation
    simulation = simulate_phase_timing_model(
        training_waveform.no_slow_model,
        training_timing.model,
        validation.time_ms,
        validation.input_value,
        float(validation.voltage_mv[0]),
    )
    protocol_metadata = {
        "validation_current_pa": protocols.validation_current,
        "validation_sweep_number": protocols.validation_sweep_number,
        "training_currents_pa": "|".join(
            f"{value:g}" for value in protocols.training_currents
        ),
        "training_sweep_numbers": "|".join(
            str(value) for value in protocols.training_sweep_numbers
        ),
        "repetitive_current_count": len(protocols.repetitive_currents),
    }
    validation_metrics = _metric_row(
        dataset,
        cell_id,
        nwb_path,
        "spike_exponential",
        True,
        validation,
        simulation,
        protocol_metadata,
    )

    all_traces = tuple(
        sorted(
            (*protocols.training, protocols.validation),
            key=_trace_current,
        )
    )
    final_waveform = fit_phase_template_ladder(
        all_traces,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    final_timing = fit_spike_exponential_timing(
        final_waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    selected_timing = next(
        row for row in final_timing.candidate_table if row["tau_selected"]
    )
    row: dict[str, object] = {
        "dataset": dataset,
        "cell_id": str(cell_id),
        "nwb_path": nwb_path,
        "training_current_count": len(final_waveform.cycles),
        "repetitive_current_count": len(protocols.repetitive_currents),
        "validation_current_pa": protocols.validation_current,
        "validation_sweep_number": protocols.validation_sweep_number,
        "validation_phase_chamfer": validation_metrics["phase_chamfer"],
        "validation_relative_spike_count_error": validation_metrics[
            "relative_spike_count_error"
        ],
        "validation_absolute_spike_count_error": validation_metrics[
            "absolute_spike_count_error"
        ],
        "validation_spike_time_rmse_ms": validation_metrics[
            "spike_time_rmse_ms"
        ],
        "validation_voltage_rmse_mv": validation_metrics[
            "voltage_rmse_mv"
        ],
        "timing_late_period_nrmse": selected_timing[
            "late_period_nrmse"
        ],
        **compact_model_parameter_row(
            final_waveform.no_slow_model,
            final_timing.model,
        ),
    }
    candidates = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            **candidate,
        }
        for candidate in final_timing.candidate_table
    ]
    return row, candidates


def compact_population_task(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Thread-friendly compact-model fitting wrapper."""
    dataset = str(task["dataset"])
    cell_id = str(task["cell_id"])
    nwb_path = str(task["nwb_path"])
    try:
        parameters, candidates = fit_compact_population_cell(
            dataset,
            cell_id,
            nwb_path,
        )
        return {
            "status": "ok",
            "parameters": parameters,
            "candidates": candidates,
        }
    except Exception as error:
        return {
            "status": "error",
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "reason": f"{type(error).__name__}:{error}",
        }
