"""Population fitting helpers for the phase-template timing ladder."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .abstract_phase_model import AbstractPhaseConfig
from .phase_population import (
    PhasePopulationConfig,
    _typical_cycle_extrema,
    select_phase_population_protocols,
)
from .phase_template_model import (
    PhaseTemplateConfig,
    PhaseTemplateSimulation,
    fit_phase_template_ladder,
    phase_plane_chamfer_distance,
    simulate_phase_template_model,
)
from .phase_timing import (
    PhaseTimingConfig,
    fit_phase_timing_ladder,
    simulate_phase_timing_model,
    timing_parameter_row,
)


def _spike_times_ms(trace, voltage_mv: np.ndarray) -> np.ndarray:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    crossings = np.flatnonzero(
        (voltage_mv[1:] >= 0.0)
        & (voltage_mv[:-1] < 0.0)
        & during[1:]
    ) + 1
    return np.asarray(trace.time_ms[crossings], dtype=float)


def _train_features(
    trace,
    voltage_mv: np.ndarray,
) -> dict[str, float]:
    spikes = _spike_times_ms(trace, voltage_mv)
    intervals = np.diff(spikes)
    width = min(3, len(intervals))
    early = (
        float(np.mean(intervals[:width]))
        if width
        else float("nan")
    )
    late = (
        float(np.mean(intervals[-width:]))
        if width
        else float("nan")
    )
    return {
        "spike_count": float(len(spikes)),
        "latency_ms": (
            float(spikes[0] - trace.stimulus_start_ms)
            if len(spikes)
            else float("nan")
        ),
        "first_isi_ms": (
            float(intervals[0]) if len(intervals) else float("nan")
        ),
        "early_isi_ms": early,
        "late_isi_ms": late,
        "adaptation_ratio": (
            late / early
            if np.isfinite(early) and early > 0.0
            else float("nan")
        ),
    }


def _absolute_difference(first: float, second: float) -> float:
    return (
        abs(float(first) - float(second))
        if np.isfinite(first) and np.isfinite(second)
        else float("nan")
    )


def _metric_row(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    model_name: str,
    selected_by_training: bool,
    validation,
    simulation: PhaseTemplateSimulation,
    protocol_metadata: Mapping[str, object],
) -> dict[str, object]:
    during = (
        (validation.time_ms >= validation.stimulus_start_ms)
        & (validation.time_ms < validation.stimulus_end_ms)
    )
    observed = _train_features(validation, validation.voltage_mv)
    predicted = _train_features(validation, simulation.voltage_mv)
    observed_spikes = _spike_times_ms(validation, validation.voltage_mv)
    predicted_spikes = _spike_times_ms(validation, simulation.voltage_mv)
    matched_count = min(len(observed_spikes), len(predicted_spikes))
    spike_time_rmse = (
        float(
            np.sqrt(
                np.mean(
                    (
                        observed_spikes[:matched_count]
                        - predicted_spikes[:matched_count]
                    )
                    ** 2
                )
            )
        )
        if matched_count
        else float("nan")
    )
    observed_extrema = _typical_cycle_extrema(
        validation,
        validation.voltage_mv,
        validation.velocity_mv_ms,
    )
    predicted_extrema = _typical_cycle_extrema(
        validation,
        simulation.voltage_mv,
        simulation.velocity_mv_ms,
    )
    count_error = int(predicted["spike_count"] - observed["spike_count"])
    row: dict[str, object] = {
        "dataset": dataset,
        "cell_id": str(cell_id),
        "nwb_path": nwb_path,
        "model": model_name,
        "selected_by_training": selected_by_training,
        **protocol_metadata,
        "observed_spike_count": int(observed["spike_count"]),
        "predicted_spike_count": int(predicted["spike_count"]),
        "spike_count_error": count_error,
        "absolute_spike_count_error": abs(count_error),
        "relative_spike_count_error": (
            abs(count_error) / max(1, int(observed["spike_count"]))
        ),
        "spike_time_rmse_ms": spike_time_rmse,
        "voltage_rmse_mv": float(
            np.sqrt(
                np.mean(
                    (
                        simulation.voltage_mv[during]
                        - validation.voltage_mv[during]
                    )
                    ** 2
                )
            )
        ),
        "phase_chamfer": phase_plane_chamfer_distance(
            validation,
            simulation,
            stable_cycles_only=True,
        ),
        "observed_max_dvdt_mv_ms": observed_extrema[0],
        "predicted_max_dvdt_mv_ms": predicted_extrema[0],
        "observed_min_dvdt_mv_ms": observed_extrema[1],
        "predicted_min_dvdt_mv_ms": predicted_extrema[1],
    }
    for name in (
        "latency_ms",
        "first_isi_ms",
        "early_isi_ms",
        "late_isi_ms",
        "adaptation_ratio",
    ):
        row[f"observed_{name}"] = observed[name]
        row[f"predicted_{name}"] = predicted[name]
        row[f"absolute_{name}_error"] = _absolute_difference(
            observed[name],
            predicted[name],
        )
    return row


def fit_phase_timing_population_cell(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    timing_config: PhaseTimingConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Fit all timing families to one cell and score its held-out current."""
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
    waveform_result = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    timing_result = fit_phase_timing_ladder(
        waveform_result.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    validation = protocols.validation
    initial_voltage = float(validation.voltage_mv[0])
    simulations = {
        "legacy_inverse_memory": simulate_phase_template_model(
            waveform_result.selected,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
        ),
        "current_spline": simulate_phase_timing_model(
            waveform_result.no_slow_model,
            timing_result.current_spline,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
        ),
        "spike_exponential": simulate_phase_timing_model(
            waveform_result.no_slow_model,
            timing_result.spike_exponential,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
        ),
        "izhikevich_recovery": simulate_phase_timing_model(
            waveform_result.no_slow_model,
            timing_result.izhikevich_recovery,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
        ),
    }
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
    metric_rows = [
        _metric_row(
            dataset,
            cell_id,
            nwb_path,
            model_name,
            model_name == timing_result.selected.family,
            validation,
            simulation,
            protocol_metadata,
        )
        for model_name, simulation in simulations.items()
    ]
    candidate_rows = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            **row,
        }
        for row in timing_result.candidate_table
    ]
    family_scores = {
        family: min(
            float(row["late_period_nrmse"])
            for row in timing_result.candidate_table
            if row["family"] == family
        )
        for family in (
            "current_spline",
            "spike_exponential",
            "izhikevich_recovery",
        )
    }
    parameter_rows = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            "late_period_nrmse": family_scores[model.family],
            "selected_by_training": (
                model.family == timing_result.selected.family
            ),
            **timing_parameter_row(model),
        }
        for model in (
            timing_result.current_spline,
            timing_result.spike_exponential,
            timing_result.izhikevich_recovery,
        )
    ]
    return metric_rows, candidate_rows, parameter_rows


def phase_timing_population_task(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Thread-friendly wrapper that records per-cell failures."""
    dataset = str(task["dataset"])
    cell_id = str(task["cell_id"])
    nwb_path = str(task["nwb_path"])
    try:
        metrics, candidates, parameters = fit_phase_timing_population_cell(
            dataset,
            cell_id,
            nwb_path,
        )
        return {
            "status": "ok",
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "metrics": metrics,
            "candidates": candidates,
            "parameters": parameters,
        }
    except Exception as error:
        return {
            "status": "error",
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "reason": f"{type(error).__name__}:{error}",
        }
