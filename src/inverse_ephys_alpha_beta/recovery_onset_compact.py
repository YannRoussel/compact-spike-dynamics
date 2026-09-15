"""Reversible fixed-width parameters for the recovery-onset phase model."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .compact_model import (
    DEFAULT_CURRENT_FRACTIONS,
    DEFAULT_FOURIER_HARMONICS,
)
from .phase_template_model import PhaseTemplateConfig, PhaseTemplateModel
from .recovery_onset import OnsetModel, RecoveryTimingModel


def _fraction_label(fraction: float) -> str:
    return f"q{int(round(100.0 * float(fraction))):03d}"


def recovery_onset_parameter_row(
    compact_row: Mapping[str, object],
    timing: RecoveryTimingModel,
    onset: OnsetModel,
    current_fractions: Sequence[float] = DEFAULT_CURRENT_FRACTIONS,
) -> dict[str, float | str]:
    """Replace the old spike-memory timing terms with recovery-onset terms."""
    fractions = tuple(float(value) for value in current_fractions)
    row = {
        str(name): value
        for name, value in compact_row.items()
        if not str(name).startswith("baseline_period_")
        and str(name) not in {"timing_tau_ms", "timing_spike_jump"}
    }
    lower = float(row["repetitive_current_min_pa"])
    upper = lower + float(row["repetitive_current_span_pa"])
    for fraction in fractions:
        label = _fraction_label(fraction)
        current = lower + fraction * (upper - lower)
        row[f"recovery_baseline_period_{label}_ms"] = timing.period(
            current,
            0.0,
        )
        row[f"onset_tau_ms_{label}"] = onset.tau(current)
    row.update(
        {
            "recovery_tau_ms": timing.tau_ms,
            "recovery_voltage_drive": timing.voltage_drive,
            "recovery_current_drive": timing.current_drive,
            "recovery_spike_jump": timing.spike_jump,
            "recovery_voltage_reference_mv": timing.voltage_reference_mv,
            "recovery_voltage_scale_mv": timing.voltage_scale_mv,
            "recovery_current_center_pa": timing.current_center,
            "recovery_current_scale_pa": timing.current_scale,
            "recovery_minimum_period_ms": timing.minimum_period_ms,
            "recovery_maximum_period_ms": timing.maximum_period_ms,
            "onset_fit_rmse_median_mv": float(
                np.median(np.asarray(onset.fit_rmse_mv, dtype=float))
            ),
        }
    )
    return row


def _positive(value: object, minimum: float = 1e-6) -> float:
    return max(minimum, float(value))


def recovery_onset_model_from_row(
    row: Mapping[str, object],
    current_fractions: Sequence[float] = DEFAULT_CURRENT_FRACTIONS,
    harmonics: int = DEFAULT_FOURIER_HARMONICS,
    phase_points: int = 256,
) -> tuple[PhaseTemplateModel, RecoveryTimingModel, OnsetModel]:
    """Reconstruct a simulatable model from measured or RNA-predicted values."""
    fractions = tuple(float(value) for value in current_fractions)
    lower = float(row["repetitive_current_min_pa"])
    span = _positive(row["repetitive_current_span_pa"])
    levels = np.asarray(
        [lower + fraction * span for fraction in fractions],
        dtype=float,
    )
    phase = np.linspace(0.0, 1.0, phase_points, endpoint=False)
    resting = float(row["resting_voltage_mv"])
    templates = []
    latencies = []
    downstrokes = []
    upstrokes = []
    baseline_log_period = []
    onset_tau = []
    for fraction, current in zip(fractions, levels):
        label = _fraction_label(fraction)
        relative = np.full(
            phase_points,
            float(row[f"template_mean_relative_mv_{label}"]),
            dtype=float,
        )
        for harmonic in range(1, harmonics + 1):
            angle = 2.0 * np.pi * harmonic * phase
            relative += (
                float(row[f"template_cos_h{harmonic:02d}_mv_{label}"])
                * np.cos(angle)
                + float(row[f"template_sin_h{harmonic:02d}_mv_{label}"])
                * np.sin(angle)
            )
        templates.append(resting + relative)
        latencies.append(_positive(row[f"latency_ms_{label}"], 0.1))
        downstrokes.append(
            _positive(row[f"downstroke_duration_ms_{label}"], 0.05)
        )
        upstrokes.append(
            _positive(row[f"upstroke_duration_ms_{label}"], 0.05)
        )
        baseline_log_period.append(
            np.log(
                _positive(
                    row[f"recovery_baseline_period_{label}_ms"],
                    0.1,
                )
            )
        )
        onset_tau.append(_positive(row[f"onset_tau_ms_{label}"], 0.01))

    minimum_period = _positive(row["recovery_minimum_period_ms"], 0.1)
    maximum_period = max(
        minimum_period + 0.1,
        float(row["recovery_maximum_period_ms"]),
    )
    timing = RecoveryTimingModel(
        current_levels=tuple(float(value) for value in levels),
        baseline_log_period=tuple(float(value) for value in baseline_log_period),
        tau_ms=_positive(row["recovery_tau_ms"], 0.1),
        voltage_drive=float(row["recovery_voltage_drive"]),
        current_drive=float(row["recovery_current_drive"]),
        spike_jump=float(row["recovery_spike_jump"]),
        voltage_reference_mv=float(
            row.get("recovery_voltage_reference_mv", resting)
        ),
        voltage_scale_mv=_positive(
            row.get("recovery_voltage_scale_mv", 100.0),
            1.0,
        ),
        current_center=float(row["recovery_current_center_pa"]),
        current_scale=_positive(row["recovery_current_scale_pa"], 1.0),
        minimum_period_ms=minimum_period,
        maximum_period_ms=maximum_period,
    )
    waveform = PhaseTemplateModel(
        phase_grid=tuple(float(value) for value in phase),
        current_levels=tuple(float(value) for value in levels),
        voltage_templates_mv=tuple(
            tuple(float(value) for value in template)
            for template in templates
        ),
        latency_ms=tuple(float(value) for value in latencies),
        downstroke_duration_ms=tuple(float(value) for value in downstrokes),
        upstroke_duration_ms=tuple(float(value) for value in upstrokes),
        resting_voltage_mv=resting,
        period_coefficients=(minimum_period, 0.0),
        input_center=0.0,
        input_scale=1.0,
        memory_scale=1.0,
        slow_tau_ms=None,
        rheobase_current=float(row["rheobase_current_pa"]),
        input_transform_offset=1.0,
        minimum_period_ms=minimum_period,
        maximum_period_ms=maximum_period,
        rest_relaxation_ms=PhaseTemplateConfig().rest_relaxation_ms,
    )
    onset = OnsetModel(
        current_levels=tuple(float(value) for value in levels),
        tau_ms=tuple(float(value) for value in onset_tau),
        fit_rmse_mv=tuple(0.0 for _ in levels),
    )
    return waveform, timing, onset
