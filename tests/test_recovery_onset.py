"""Tests for the continuous recovery and onset extension."""

from __future__ import annotations

import unittest

import numpy as np

from inverse_ephys_alpha_beta.abstract_phase_model import (
    AbstractObservedTrace,
)
from inverse_ephys_alpha_beta.recovery_onset import (
    OnsetModel,
    RecoveryOnsetConfig,
    RecoveryTimingModel,
    _fit_trace_onset,
    _normalized_exponential,
)
from inverse_ephys_alpha_beta.recovery_onset_compact import (
    recovery_onset_model_from_row,
    recovery_onset_parameter_row,
)


class RecoveryOnsetTests(unittest.TestCase):
    def test_compact_parameters_reconstruct_simulatable_model(self) -> None:
        compact = {
            "resting_voltage_mv": -68.0,
            "rheobase_current_pa": 80.0,
            "repetitive_current_min_pa": 100.0,
            "repetitive_current_max_pa": 300.0,
            "repetitive_current_span_pa": 200.0,
        }
        phase = np.linspace(0.0, 1.0, 256, endpoint=False)
        for fraction in (0, 25, 50, 75, 100):
            label = f"q{fraction:03d}"
            compact[f"template_mean_relative_mv_{label}"] = 5.0
            for harmonic in range(1, 11):
                compact[
                    f"template_cos_h{harmonic:02d}_mv_{label}"
                ] = 30.0 if harmonic == 1 else 0.0
                compact[
                    f"template_sin_h{harmonic:02d}_mv_{label}"
                ] = -20.0 if harmonic == 1 else 0.0
            compact[f"latency_ms_{label}"] = 15.0
            compact[f"downstroke_duration_ms_{label}"] = 1.5
            compact[f"upstroke_duration_ms_{label}"] = 1.0
            compact[f"template_min_relative_mv_{label}"] = -31.0
            compact[f"template_max_relative_mv_{label}"] = 41.0
        timing = RecoveryTimingModel(
            current_levels=(100.0, 300.0),
            baseline_log_period=(np.log(40.0), np.log(12.0)),
            tau_ms=100.0,
            voltage_drive=0.2,
            current_drive=-0.1,
            spike_jump=0.05,
            voltage_reference_mv=-68.0,
            voltage_scale_mv=100.0,
            current_center=200.0,
            current_scale=100.0,
            minimum_period_ms=5.0,
            maximum_period_ms=100.0,
        )
        onset = OnsetModel(
            current_levels=(100.0, 300.0),
            tau_ms=(4.0, 2.0),
            fit_rmse_mv=(1.0, 2.0),
        )
        row = recovery_onset_parameter_row(compact, timing, onset)
        waveform, recovered_timing, recovered_onset = (
            recovery_onset_model_from_row(row)
        )
        expected = -68.0 + 5.0 + 30.0 * np.cos(
            2.0 * np.pi * phase
        ) - 20.0 * np.sin(2.0 * np.pi * phase)
        np.testing.assert_allclose(
            waveform.voltage_template(200.0),
            expected,
        )
        self.assertAlmostEqual(recovered_timing.tau_ms, 100.0)
        self.assertAlmostEqual(recovered_timing.period(100.0, 0.0), 40.0)
        self.assertAlmostEqual(recovered_onset.tau(200.0), 3.0)

    def test_normalized_exponential_reaches_both_endpoints(self) -> None:
        values = _normalized_exponential(
            np.asarray((0.0, 5.0, 10.0)),
            duration_ms=10.0,
            tau_ms=4.0,
        )
        self.assertAlmostEqual(float(values[0]), 0.0)
        self.assertAlmostEqual(float(values[-1]), 1.0)
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_onset_fit_recovers_synthetic_time_constant(self) -> None:
        time = np.linspace(0.0, 40.0, 4001)
        start = 5.0
        duration = 12.0
        tau = 3.5
        voltage = np.full_like(time, -70.0)
        during_onset = (time >= start) & (time <= start + duration)
        elapsed = time[during_onset] - start
        voltage[during_onset] = (
            -70.0
            + 45.0
            * _normalized_exponential(elapsed, duration, tau)
        )
        after = time > start + duration
        voltage[after] = -25.0
        peak_index = int(np.searchsorted(time, start + duration + 0.5))
        voltage[peak_index] = 25.0
        velocity = np.gradient(voltage, time)
        trace = AbstractObservedTrace(
            name="synthetic",
            role="training",
            time_ms=time,
            voltage_mv=voltage,
            velocity_mv_ms=velocity,
            acceleration_mv_ms2=np.gradient(velocity, time),
            input_value=np.where(
                (time >= start) & (time < 35.0),
                100.0,
                0.0,
            ),
            fit_mask=np.ones_like(time, dtype=bool),
            stimulus_start_ms=start,
            stimulus_end_ms=35.0,
        )
        result = _fit_trace_onset(
            trace,
            RecoveryOnsetConfig(onset_velocity_mv_ms=20.0),
        )
        self.assertIsNotNone(result)
        _, fitted_tau, _ = result
        self.assertAlmostEqual(fitted_tau, tau, delta=0.8)


if __name__ == "__main__":
    unittest.main()
