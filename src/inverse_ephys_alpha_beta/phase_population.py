"""Population-scale held-out-current tests for phase-template models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .abstract_phase_model import (
    AbstractObservedTrace,
    AbstractPhaseConfig,
    prepare_abstract_trace,
)
from .phase_template_model import (
    PhaseTemplateConfig,
    fit_phase_template_ladder,
    phase_plane_chamfer_distance,
    simulate_phase_template_model,
    template_fourier_coefficients,
)
from .raw_patchseq import (
    CurrentClampSweep,
    count_sweep_spikes,
    read_current_clamp_sweeps,
)


@dataclass(frozen=True)
class PhasePopulationConfig:
    minimum_repetitive_spikes: int = 4
    trace_margin_ms: float = 20.0
    current_round_decimals: int = 4


@dataclass(frozen=True)
class PhasePopulationProtocols:
    training: tuple[AbstractObservedTrace, ...]
    validation: AbstractObservedTrace
    training_sweep_numbers: tuple[int, ...]
    validation_sweep_number: int
    training_currents: tuple[float, ...]
    validation_current: float
    repetitive_currents: tuple[float, ...]


def _prepare_sweep(
    sweep: CurrentClampSweep,
    name: str,
    role: str,
    margin_ms: float,
    derivative_config: AbstractPhaseConfig,
) -> AbstractObservedTrace:
    if sweep.stimulus_start_ms is None or sweep.stimulus_end_ms is None:
        raise ValueError("missing_stimulus_epoch")
    start = float(sweep.stimulus_start_ms)
    end = float(sweep.stimulus_end_ms)
    keep = (sweep.time_ms >= start - margin_ms) & (
        sweep.time_ms <= end + margin_ms
    )
    if np.sum(keep) < 9:
        raise ValueError("trace_too_short")
    return prepare_abstract_trace(
        name=name,
        role=role,
        time_ms=sweep.time_ms[keep] - (start - margin_ms),
        voltage_mv=sweep.voltage_mv[keep],
        input_value=sweep.current_pa[keep],
        stimulus_start_ms=margin_ms,
        stimulus_end_ms=margin_ms + end - start,
        config=derivative_config,
    )


def select_phase_population_protocols(
    nwb_path: str | Path,
    derivative_config: AbstractPhaseConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> PhasePopulationProtocols:
    """Select an interior repetitive current for held-out interpolation."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    population_config = population_config or PhasePopulationConfig()
    sweeps = read_current_clamp_sweeps(
        nwb_path,
        long_square_only=True,
    )
    positive = [
        sweep
        for sweep in sweeps
        if sweep.stimulus_amplitude_pa is not None
        and sweep.stimulus_amplitude_pa > 0.0
    ]
    if not positive:
        raise ValueError("no_positive_long_square")

    by_current: dict[float, tuple[CurrentClampSweep, int]] = {}
    for sweep in positive:
        spike_count = count_sweep_spikes(sweep)
        if spike_count <= 0:
            continue
        current = round(
            float(sweep.stimulus_amplitude_pa),
            population_config.current_round_decimals,
        )
        previous = by_current.get(current)
        if previous is None or spike_count > previous[1]:
            by_current[current] = (sweep, spike_count)
    if len(by_current) < 3:
        raise ValueError("fewer_than_three_spiking_current_levels")

    repetitive = sorted(
        current
        for current, (_, spike_count) in by_current.items()
        if spike_count >= population_config.minimum_repetitive_spikes
    )
    if len(repetitive) < 3:
        raise ValueError("fewer_than_three_repetitive_current_levels")
    interior = repetitive[1:-1]
    center = float(np.median(repetitive))
    validation_current = min(
        interior,
        key=lambda current: (abs(current - center), current),
    )
    validation_sweep = by_current[validation_current][0]
    training_pairs = [
        (current, sweep, spike_count)
        for current, (sweep, spike_count) in sorted(by_current.items())
        if current != validation_current
    ]
    repetitive_training = [
        current
        for current, _, spike_count in training_pairs
        if spike_count >= population_config.minimum_repetitive_spikes
    ]
    if not (
        any(current < validation_current for current in repetitive_training)
        and any(current > validation_current for current in repetitive_training)
    ):
        raise ValueError("held_out_current_is_not_bracketed")

    training = tuple(
        _prepare_sweep(
            sweep,
            name=f"train_{current:g}_pa_s{sweep.sweep_number}",
            role="training",
            margin_ms=population_config.trace_margin_ms,
            derivative_config=derivative_config,
        )
        for current, sweep, _ in training_pairs
    )
    validation = _prepare_sweep(
        validation_sweep,
        name=f"held_out_{validation_current:g}_pa",
        role="validation",
        margin_ms=population_config.trace_margin_ms,
        derivative_config=derivative_config,
    )
    return PhasePopulationProtocols(
        training=training,
        validation=validation,
        training_sweep_numbers=tuple(
            sweep.sweep_number for _, sweep, _ in training_pairs
        ),
        validation_sweep_number=validation_sweep.sweep_number,
        training_currents=tuple(
            float(current) for current, _, _ in training_pairs
        ),
        validation_current=float(validation_current),
        repetitive_currents=tuple(float(value) for value in repetitive),
    )


def _spike_count(trace: AbstractObservedTrace, voltage: np.ndarray) -> int:
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    above = voltage >= 0.0
    return int(np.sum(above[1:] & ~above[:-1] & during[1:]))


def _typical_cycle_extrema(
    trace: AbstractObservedTrace,
    voltage_mv: np.ndarray,
    velocity_mv_ms: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return median extrema across complete cycles after the first cycle."""
    during = (
        (trace.time_ms >= trace.stimulus_start_ms)
        & (trace.time_ms < trace.stimulus_end_ms)
    )
    crossings = np.flatnonzero(
        (voltage_mv[1:] >= 0.0)
        & (voltage_mv[:-1] < 0.0)
        & during[1:]
    ) + 1
    cycle_pairs = list(zip(crossings[1:-1], crossings[2:]))
    if not cycle_pairs:
        indices = np.flatnonzero(during)
        if len(indices) == 0:
            raise ValueError("empty_stimulus_epoch")
        cycle_pairs = [(int(indices[0]), int(indices[-1]) + 1)]
    extrema = np.asarray(
        [
            (
                np.max(velocity_mv_ms[start:stop]),
                np.min(velocity_mv_ms[start:stop]),
                np.min(voltage_mv[start:stop]),
                np.max(voltage_mv[start:stop]),
            )
            for start, stop in cycle_pairs
            if stop > start
        ],
        dtype=float,
    )
    return tuple(float(value) for value in np.median(extrema, axis=0))


def fit_phase_population_cell(
    dataset: str,
    cell_id: str,
    nwb_path: str,
    derivative_config: AbstractPhaseConfig | None = None,
    phase_config: PhaseTemplateConfig | None = None,
    population_config: PhasePopulationConfig | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Fit one cell and return metrics, candidate, and Fourier rows."""
    derivative_config = derivative_config or AbstractPhaseConfig()
    phase_config = phase_config or PhaseTemplateConfig()
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
    result = fit_phase_template_ladder(
        protocols.training,
        resting_voltage_mv=resting_voltage,
        config=phase_config,
    )
    validation = protocols.validation
    during = (
        (validation.time_ms >= validation.stimulus_start_ms)
        & (validation.time_ms < validation.stimulus_end_ms)
    )
    observed_spikes = _spike_count(
        validation,
        validation.voltage_mv,
    )
    models = (
        ("phase_only", result.no_slow_model),
        ("selected", result.selected),
    )
    observed_extrema = _typical_cycle_extrema(
        validation,
        validation.voltage_mv,
        validation.velocity_mv_ms,
    )
    metric_rows = []
    phase_only_simulation = None
    for label, model in models:
        if label == "selected" and model is result.no_slow_model:
            simulation = phase_only_simulation
        else:
            simulation = simulate_phase_template_model(
                model,
                validation.time_ms,
                validation.input_value,
                float(validation.voltage_mv[0]),
            )
        if simulation is None:
            raise RuntimeError("phase_template_simulation_failed")
        if label == "phase_only":
            phase_only_simulation = simulation
        predicted_spikes = _spike_count(
            validation,
            simulation.voltage_mv,
        )
        predicted_extrema = _typical_cycle_extrema(
            validation,
            simulation.voltage_mv,
            simulation.velocity_mv_ms,
        )
        metric_rows.append(
            {
                "dataset": dataset,
                "cell_id": str(cell_id),
                "nwb_path": nwb_path,
                "model": label,
                "model_family": (
                    "phase_only"
                    if not model.has_slow_state
                    else f"phase_slow_{model.slow_tau_ms:g}_ms"
                ),
                "slow_tau_ms": model.slow_tau_ms,
                "slow_selected": bool(
                    result.selected.has_slow_state
                ),
                "validation_current_pa": (
                    protocols.validation_current
                ),
                "validation_sweep_number": (
                    protocols.validation_sweep_number
                ),
                "training_currents_pa": "|".join(
                    f"{value:g}"
                    for value in protocols.training_currents
                ),
                "training_sweep_numbers": "|".join(
                    str(value)
                    for value in protocols.training_sweep_numbers
                ),
                "repetitive_current_count": len(
                    protocols.repetitive_currents
                ),
                "observed_spike_count": observed_spikes,
                "predicted_spike_count": predicted_spikes,
                "spike_count_error": predicted_spikes - observed_spikes,
                "absolute_spike_count_error": abs(
                    predicted_spikes - observed_spikes
                ),
                "relative_spike_count_error": abs(
                    predicted_spikes - observed_spikes
                )
                / max(1, observed_spikes),
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
                "observed_voltage_min_mv": observed_extrema[2],
                "predicted_voltage_min_mv": predicted_extrema[2],
                "observed_voltage_max_mv": observed_extrema[3],
                "predicted_voltage_max_mv": predicted_extrema[3],
            }
        )
    candidate_rows = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            **row,
            "selected": row["label"]
            == (
                "phase_only"
                if not result.selected.has_slow_state
                else (
                    f"phase_slow_{result.selected.slow_tau_ms:g}_ms"
                )
            ),
        }
        for row in result.candidate_table
    ]
    fourier_rows = [
        {
            "dataset": dataset,
            "cell_id": str(cell_id),
            "model_family": (
                "phase_only"
                if not result.selected.has_slow_state
                else f"phase_slow_{result.selected.slow_tau_ms:g}_ms"
            ),
            "slow_tau_ms": result.selected.slow_tau_ms,
            **row,
        }
        for row in template_fourier_coefficients(
            result.selected,
            harmonics=phase_config.fourier_harmonics,
        )
    ]
    return metric_rows, candidate_rows, fourier_rows


def phase_population_task(
    task: Mapping[str, object],
) -> dict[str, object]:
    """Pickle/thread-friendly wrapper that records eligibility failures."""
    dataset = str(task["dataset"])
    cell_id = str(task["cell_id"])
    nwb_path = str(task["nwb_path"])
    try:
        metrics, candidates, fourier = fit_phase_population_cell(
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
            "fourier": fourier,
        }
    except Exception as error:
        return {
            "status": "ineligible_or_error",
            "dataset": dataset,
            "cell_id": cell_id,
            "nwb_path": nwb_path,
            "reason": f"{type(error).__name__}:{error}",
        }
