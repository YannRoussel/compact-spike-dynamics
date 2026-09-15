"""Fixed-width parameters for the compact phase-template neuron model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .phase_template_model import PhaseTemplateModel
from .phase_timing import (
    PhaseKernelTimingModel,
    PhaseTimingModel,
    kernel_timing_parameter_row,
    timing_parameter_row,
)


DEFAULT_CURRENT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_FOURIER_HARMONICS = 10


def _fraction_label(fraction: float) -> str:
    return f"q{int(round(100.0 * float(fraction))):03d}"


def compact_model_parameter_row(
    waveform: PhaseTemplateModel,
    timing: PhaseTimingModel,
    current_fractions: Sequence[float] = DEFAULT_CURRENT_FRACTIONS,
    harmonics: int = DEFAULT_FOURIER_HARMONICS,
) -> dict[str, float | str | None]:
    """Export a current-normalized, fixed-dimensional model vector."""
    fractions = tuple(float(value) for value in current_fractions)
    if not fractions:
        raise ValueError("At least one current fraction is required")
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("Current fractions must lie between zero and one")
    if harmonics < 1:
        raise ValueError("At least one Fourier harmonic is required")

    lower = float(waveform.current_levels[0])
    upper = float(waveform.current_levels[-1])
    row: dict[str, float | str | None] = {
        "resting_voltage_mv": waveform.resting_voltage_mv,
        "rheobase_current_pa": waveform.rheobase_current,
        "repetitive_current_min_pa": lower,
        "repetitive_current_max_pa": upper,
        "repetitive_current_span_pa": upper - lower,
        **timing_parameter_row(timing, fractions),
    }
    row.pop("family")
    row.pop("training_current_min")
    row.pop("training_current_max")
    row["timing_tau_ms"] = row.pop("tau_ms")
    row["timing_spike_jump"] = row.pop("spike_jump")
    row.pop("voltage_drive")

    _add_waveform_parameters(
        row,
        waveform,
        fractions,
        harmonics,
        lambda current: timing.period(current, 0.0),
    )
    return row


def compact_kernel_model_parameter_row(
    waveform: PhaseTemplateModel,
    timing: PhaseKernelTimingModel,
    current_fractions: Sequence[float] = DEFAULT_CURRENT_FRACTIONS,
    harmonics: int = DEFAULT_FOURIER_HARMONICS,
) -> dict[str, float | str]:
    """Export compact waveform and fixed-lag adaptation-kernel summaries."""
    fractions = tuple(float(value) for value in current_fractions)
    if not fractions:
        raise ValueError("At least one current fraction is required")
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("Current fractions must lie between zero and one")
    if harmonics < 1:
        raise ValueError("At least one Fourier harmonic is required")
    lower = float(waveform.current_levels[0])
    upper = float(waveform.current_levels[-1])
    timing_row = kernel_timing_parameter_row(timing, fractions)
    timing_row.pop("family")
    timing_row.pop("training_current_min")
    timing_row.pop("training_current_max")
    row: dict[str, float | str] = {
        "resting_voltage_mv": waveform.resting_voltage_mv,
        "rheobase_current_pa": waveform.rheobase_current,
        "repetitive_current_min_pa": lower,
        "repetitive_current_max_pa": upper,
        "repetitive_current_span_pa": upper - lower,
        **timing_row,
    }
    zero_state = np.zeros(len(timing.tau_basis_ms), dtype=float)
    _add_waveform_parameters(
        row,
        waveform,
        fractions,
        harmonics,
        lambda current: timing.period(current, zero_state),
    )
    return row


def _add_waveform_parameters(
    row: dict[str, float | str | None],
    waveform: PhaseTemplateModel,
    fractions: Sequence[float],
    harmonics: int,
    period_at_current,
) -> None:
    lower = float(waveform.current_levels[0])
    upper = float(waveform.current_levels[-1])
    for fraction in fractions:
        label = _fraction_label(fraction)
        current = lower + fraction * (upper - lower)
        template = waveform.voltage_template(current)
        relative_template = template - waveform.resting_voltage_mv
        coefficients = np.fft.rfft(relative_template) / len(relative_template)
        row[f"template_mean_relative_mv_{label}"] = float(
            coefficients[0].real
        )
        for harmonic, value in enumerate(
            coefficients[1 : harmonics + 1],
            start=1,
        ):
            row[f"template_cos_h{harmonic:02d}_mv_{label}"] = float(
                2.0 * value.real
            )
            row[f"template_sin_h{harmonic:02d}_mv_{label}"] = float(
                -2.0 * value.imag
            )
        row[f"latency_ms_{label}"] = waveform.latency(current)
        downstroke, _, upstroke = waveform.segment_durations(
            current,
            period_at_current(current),
        )
        row[f"downstroke_duration_ms_{label}"] = downstroke
        row[f"upstroke_duration_ms_{label}"] = upstroke
        row[f"template_min_relative_mv_{label}"] = float(
            np.min(relative_template)
        )
        row[f"template_max_relative_mv_{label}"] = float(
            np.max(relative_template)
        )


def compact_parameter_columns(
    frame_columns: Sequence[str],
) -> list[str]:
    """Identify model parameters while excluding IDs and fit diagnostics."""
    excluded = {
        "dataset",
        "cell_id",
        "nwb_path",
        "status",
        "failure_reason",
        "validation_current_pa",
        "validation_sweep_number",
        "validation_phase_chamfer",
        "validation_relative_spike_count_error",
        "validation_absolute_spike_count_error",
        "validation_spike_time_rmse_ms",
        "validation_voltage_rmse_mv",
        "timing_late_period_nrmse",
        "recovery_late_period_nrmse",
        "recovery_cycle_current_count",
        "recovery_voltage_reference_mv",
        "recovery_voltage_scale_mv",
        "onset_fit_rmse_median_mv",
        "onset_fit_status",
        "repetitive_current_count",
        "training_current_count",
        "repetitive_current_max_pa",
    }
    return [
        str(column)
        for column in frame_columns
        if str(column) not in excluded
        and not str(column).startswith("template_min_relative_mv_")
        and not str(column).startswith("template_max_relative_mv_")
    ]
