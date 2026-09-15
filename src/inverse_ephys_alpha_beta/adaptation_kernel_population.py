"""Population comparison of single-tau and fixed-kernel adaptation clocks."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .abstract_phase_model import AbstractPhaseConfig
from .compact_model import compact_kernel_model_parameter_row
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
    fit_phase_kernel_timing,
    fit_spike_exponential_timing,
    simulate_phase_kernel_timing_model,
    simulate_phase_timing_model,
)
from .phase_timing_population import _metric_row


def _trace_current(trace) -> float:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    return float(np.median(trace.input_value[during]))


def _single_tau_margin(candidate_table) -> float:
    scores = sorted(
        float(row["late_period_nrmse"])
        for row in candidate_table
    )
    if len(scores) < 2:
        return float("nan")
    return float((scores[1] - scores[0]) / max(scores[0], 1e-8))


def fit_adaptation_kernel_population_cell(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    timing_config: PhaseTimingConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Cross-validate both clocks and refit fixed-kernel parameters."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    phase_config = phase_config or PhaseTemplateConfig()
    timing_config = timing_config or PhaseTimingConfig()
    protocols = select_phase_population_protocols(
        nwb_path,
        derivative_config,
        population_config,
    )
    baseline_mask = (
        protocols.training[0].time_ms
        < protocols.training[0].stimulus_start_ms
    )
    resting_voltage = float(
        np.median(protocols.training[0].voltage_mv[baseline_mask])
    )

    training_waveform = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    training_single = fit_spike_exponential_timing(
        training_waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    training_kernel = fit_phase_kernel_timing(
        training_waveform.cycles,
        config=timing_config,
    )
    validation = protocols.validation
    initial_voltage = float(validation.voltage_mv[0])
    single_simulation = simulate_phase_timing_model(
        training_waveform.no_slow_model,
        training_single.model,
        validation.time_ms,
        validation.input_value,
        initial_voltage,
    )
    kernel_simulation = simulate_phase_kernel_timing_model(
        training_waveform.no_slow_model,
        training_kernel.model,
        validation.time_ms,
        validation.input_value,
        initial_voltage,
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
    metrics = [
        _metric_row(
            dataset,
            cell_id,
            nwb_path,
            "spike_exponential",
            False,
            validation,
            single_simulation,
            protocol_metadata,
        ),
        _metric_row(
            dataset,
            cell_id,
            nwb_path,
            "fixed_multiscale_kernel",
            False,
            validation,
            kernel_simulation,
            protocol_metadata,
        ),
    ]

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
    final_single = fit_spike_exponential_timing(
        final_waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    final_kernel = fit_phase_kernel_timing(
        final_waveform.cycles,
        config=timing_config,
    )
    selected_single = next(
        row for row in final_single.candidate_table if row["tau_selected"]
    )
    kernel_row: dict[str, object] = {
        "dataset": dataset,
        "cell_id": str(cell_id),
        "nwb_path": nwb_path,
        "training_current_count": len(final_waveform.cycles),
        "repetitive_current_count": len(protocols.repetitive_currents),
        "validation_current_pa": protocols.validation_current,
        "validation_sweep_number": protocols.validation_sweep_number,
        "single_tau_ms": final_single.model.tau_ms,
        "single_spike_jump": final_single.model.spike_jump,
        "single_late_period_nrmse": selected_single[
            "late_period_nrmse"
        ],
        "single_tau_relative_score_margin": _single_tau_margin(
            final_single.candidate_table
        ),
        "kernel_late_period_nrmse": final_kernel.late_period_nrmse,
        **compact_kernel_model_parameter_row(
            final_waveform.no_slow_model,
            final_kernel.model,
        ),
    }
    for metric in metrics:
        prefix = (
            "single_validation"
            if metric["model"] == "spike_exponential"
            else "kernel_validation"
        )
        for name in (
            "phase_chamfer",
            "relative_spike_count_error",
            "absolute_spike_count_error",
            "spike_time_rmse_ms",
            "voltage_rmse_mv",
            "absolute_first_isi_ms_error",
            "absolute_late_isi_ms_error",
            "absolute_adaptation_ratio_error",
        ):
            kernel_row[f"{prefix}_{name}"] = metric[name]

    reliability_rows = []
    for cycle_set in final_waveform.cycles:
        if len(cycle_set.periods_ms) < 4:
            continue
        current_single = fit_spike_exponential_timing(
            [cycle_set],
            voltage_reference_mv=resting_voltage,
            config=timing_config,
        )
        current_kernel = fit_phase_kernel_timing(
            [cycle_set],
            config=timing_config,
        )
        row: dict[str, object] = {
            "dataset": dataset,
            "cell_id": str(cell_id),
            "current_pa": cycle_set.current_value,
            "period_count": len(cycle_set.periods_ms),
            "full_single_tau_ms": final_single.model.tau_ms,
            "current_single_tau_ms": current_single.model.tau_ms,
            "single_tau_agrees": (
                current_single.model.tau_ms == final_single.model.tau_ms
            ),
            "current_single_tau_relative_score_margin": (
                _single_tau_margin(current_single.candidate_table)
            ),
        }
        for lag in timing_config.kernel_summary_lags_ms:
            label = int(round(lag))
            row[f"full_kernel_value_{label:04d}ms"] = (
                final_kernel.model.kernel(lag)
            )
            row[f"current_kernel_value_{label:04d}ms"] = (
                current_kernel.model.kernel(lag)
            )
        reliability_rows.append(row)
    return kernel_row, metrics, reliability_rows


def adaptation_kernel_population_task(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Thread-friendly wrapper that records per-cell failures."""
    dataset = str(task["dataset"])
    cell_id = str(task["cell_id"])
    nwb_path = str(task["nwb_path"])
    try:
        parameters, metrics, reliability = (
            fit_adaptation_kernel_population_cell(
                dataset,
                cell_id,
                nwb_path,
            )
        )
        return {
            "status": "ok",
            "parameters": parameters,
            "metrics": metrics,
            "reliability": reliability,
        }
    except Exception as error:
        return {
            "status": "error",
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "reason": f"{type(error).__name__}:{error}",
        }
