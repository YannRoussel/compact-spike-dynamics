"""Held-out population comparison for recovery and onset extensions."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .abstract_phase_model import AbstractPhaseConfig
from .phase_population import (
    PhasePopulationConfig,
    select_phase_population_protocols,
)
from .phase_template_model import PhaseTemplateConfig, fit_phase_template_ladder
from .phase_timing import (
    PhaseTimingConfig,
    fit_spike_exponential_timing,
    simulate_phase_timing_model,
)
from .phase_timing_population import _metric_row, _spike_times_ms
from .recovery_onset import (
    RecoveryOnsetConfig,
    fit_recovery_onset_model,
    simulate_recovery_onset_model,
)


MODEL_NAMES = (
    "spike_exponential",
    "recovery_timing",
    "recovery_onset",
)


def _onset_rmse(
    observed,
    predicted_voltage_mv: np.ndarray,
    window_ms: float = 30.0,
) -> float:
    stop = min(
        observed.stimulus_start_ms + window_ms,
        observed.stimulus_end_ms,
    )
    mask = (
        (observed.time_ms >= observed.stimulus_start_ms)
        & (observed.time_ms <= stop)
    )
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    predicted_voltage_mv[mask]
                    - observed.voltage_mv[mask]
                )
            )
        )
    )


def _first_three_isi_rmse(
    observed,
    predicted_voltage_mv: np.ndarray,
) -> float:
    observed_spikes = _spike_times_ms(observed, observed.voltage_mv)
    predicted_spikes = _spike_times_ms(observed, predicted_voltage_mv)
    observed_isi = np.diff(observed_spikes)[:3]
    predicted_isi = np.diff(predicted_spikes)[:3]
    count = min(len(observed_isi), len(predicted_isi))
    if not count:
        return float("nan")
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    predicted_isi[:count] - observed_isi[:count]
                )
            )
        )
    )


def fit_recovery_onset_population_cell(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    timing_config: PhaseTimingConfig | None = None,
    recovery_config: RecoveryOnsetConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    """Fit baseline and extended models while holding out one current."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    phase_config = phase_config or PhaseTemplateConfig()
    timing_config = timing_config or PhaseTimingConfig()
    recovery_config = recovery_config or RecoveryOnsetConfig()
    protocols = select_phase_population_protocols(
        nwb_path,
        derivative_config,
        population_config,
    )
    first_training = protocols.training[0]
    baseline = first_training.time_ms < first_training.stimulus_start_ms
    resting_voltage = float(
        np.median(first_training.voltage_mv[baseline])
    )
    waveform = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    compact = fit_spike_exponential_timing(
        waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=timing_config,
    )
    extended = fit_recovery_onset_model(
        protocols.training,
        waveform.cycles,
        voltage_reference_mv=resting_voltage,
        config=recovery_config,
    )
    validation = protocols.validation
    initial_voltage = float(validation.voltage_mv[0])
    simulations = {
        "spike_exponential": simulate_phase_timing_model(
            waveform.no_slow_model,
            compact.model,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
        ),
        "recovery_timing": simulate_recovery_onset_model(
            waveform.no_slow_model,
            extended.timing,
            extended.onset,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
            use_onset_branch=False,
        ),
        "recovery_onset": simulate_recovery_onset_model(
            waveform.no_slow_model,
            extended.timing,
            extended.onset,
            validation.time_ms,
            validation.input_value,
            initial_voltage,
            use_onset_branch=True,
        ),
    }
    protocol_metadata = {
        "validation_current_pa": protocols.validation_current,
        "validation_sweep_number": protocols.validation_sweep_number,
        "training_currents_pa": "|".join(
            f"{value:g}" for value in protocols.training_currents
        ),
    }
    metric_rows = []
    for model_name, simulation in simulations.items():
        row = _metric_row(
            dataset,
            cell_id,
            nwb_path,
            model_name,
            model_name == "recovery_onset",
            validation,
            simulation,
            protocol_metadata,
        )
        row["onset_30ms_voltage_rmse_mv"] = _onset_rmse(
            validation,
            simulation.voltage_mv,
        )
        row["first_three_isi_rmse_ms"] = _first_three_isi_rmse(
            validation,
            simulation.voltage_mv,
        )
        metric_rows.append(row)
    selected_candidate = next(
        row
        for row in extended.candidate_table
        if row["tau_selected"]
    )
    parameter_row = {
        "dataset": dataset,
        "cell_id": str(cell_id),
        "nwb_path": nwb_path,
        "recovery_tau_ms": extended.timing.tau_ms,
        "recovery_voltage_drive": extended.timing.voltage_drive,
        "recovery_current_drive": extended.timing.current_drive,
        "recovery_spike_jump": extended.timing.spike_jump,
        "recovery_late_period_nrmse": selected_candidate[
            "late_period_nrmse"
        ],
        "onset_tau_median_ms": float(np.median(extended.onset.tau_ms)),
        "onset_fit_rmse_median_mv": float(
            np.median(extended.onset.fit_rmse_mv)
        ),
    }
    candidate_rows = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            **candidate,
        }
        for candidate in extended.candidate_table
    ]
    return metric_rows, parameter_row, candidate_rows


def recovery_onset_population_task(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Thread-friendly population fitting wrapper."""
    dataset = str(task["dataset"])
    cell_id = str(task["cell_id"])
    nwb_path = str(task["nwb_path"])
    try:
        metrics, parameters, candidates = (
            fit_recovery_onset_population_cell(
                dataset,
                cell_id,
                nwb_path,
            )
        )
        return {
            "status": "ok",
            "metrics": metrics,
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
