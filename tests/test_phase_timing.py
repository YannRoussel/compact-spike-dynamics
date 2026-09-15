from __future__ import annotations

import unittest

import numpy as np

from inverse_ephys_alpha_beta.abstract_phase_model import (
    AbstractObservedTrace,
)
from inverse_ephys_alpha_beta.phase_template_model import (
    fit_phase_template_ladder,
)
from inverse_ephys_alpha_beta.phase_timing import (
    PhaseTimingConfig,
    _advance_voltage_state,
    fit_phase_kernel_timing,
    fit_phase_timing_ladder,
    fit_spike_exponential_timing,
    kernel_timing_parameter_row,
    simulate_phase_kernel_timing_model,
    simulate_phase_timing_model,
)
from inverse_ephys_alpha_beta.compact_model import (
    compact_kernel_model_parameter_row,
    compact_model_parameter_row,
)


def _adapting_trace() -> AbstractObservedTrace:
    time_ms = np.arange(0.0, 500.05, 0.05)
    voltage_mv = np.full_like(time_ms, -65.0)
    spike_times = [45.0]
    for period in (15.0, 18.0, 22.0, 26.0, 29.0, 31.0, 32.0, 33.0):
        spike_times.append(spike_times[-1] + period)
    for spike_time in spike_times:
        relative = time_ms - spike_time
        voltage_mv += 102.0 * np.exp(-(relative / 0.38) ** 2)
        voltage_mv -= 12.0 * np.exp(
            -((relative - 1.8) / 0.9) ** 2
        )
    velocity = np.gradient(voltage_mv, time_ms)
    acceleration = np.gradient(velocity, time_ms)
    current = np.zeros_like(time_ms)
    current[(time_ms >= 20.0) & (time_ms < 460.0)] = 100.0
    return AbstractObservedTrace(
        name="adapting",
        role="training",
        time_ms=time_ms,
        voltage_mv=voltage_mv,
        velocity_mv_ms=velocity,
        acceleration_mv_ms2=acceleration,
        input_value=current,
        fit_mask=np.ones_like(time_ms, dtype=bool),
        stimulus_start_ms=20.0,
        stimulus_end_ms=460.0,
    )


class PhaseTimingTests(unittest.TestCase):
    def test_vectorized_voltage_state_matches_scalar_recurrence(self):
        voltage = np.linspace(-70.0, 30.0, 256)
        period = 24.0
        downstroke = 2.0
        upstroke = 1.0
        tau = 50.0
        expected = 0.3
        durations = np.concatenate(
            (
                np.full(64, downstroke / 64),
                np.full(128, (period - downstroke - upstroke) / 128),
                np.full(64, upstroke / 64),
            )
        )
        for value, duration in zip(voltage, durations):
            target = (value + 65.0) / 100.0
            decay = np.exp(-duration / tau)
            expected = target + (expected - target) * decay

        observed = _advance_voltage_state(
            0.3,
            voltage,
            period,
            downstroke,
            upstroke,
            tau,
            -65.0,
            100.0,
        )
        self.assertAlmostEqual(observed, expected, places=12)

    def test_nested_timing_ladder_and_izhikevich_simulation(self):
        trace = _adapting_trace()
        waveform = fit_phase_template_ladder(
            [trace],
            resting_voltage_mv=-65.0,
        )
        config = PhaseTimingConfig(
            tau_candidates_ms=(20.0, 100.0, 500.0),
        )
        result = fit_phase_timing_ladder(
            waveform.cycles,
            voltage_reference_mv=-65.0,
            config=config,
        )

        self.assertFalse(result.current_spline.has_state)
        self.assertTrue(result.spike_exponential.has_state)
        self.assertEqual(
            result.izhikevich_recovery.family,
            "izhikevich_recovery",
        )
        self.assertLessEqual(
            abs(result.izhikevich_recovery.voltage_drive),
            config.voltage_drive_bound + 1e-8,
        )
        self.assertLessEqual(
            abs(result.izhikevich_recovery.spike_jump),
            config.spike_jump_bound + 1e-8,
        )

        simulation = simulate_phase_timing_model(
            waveform.no_slow_model,
            result.izhikevich_recovery,
            trace.time_ms,
            trace.input_value,
            initial_voltage_mv=-65.0,
        )
        self.assertTrue(np.isfinite(simulation.voltage_mv).all())
        self.assertTrue(np.isfinite(simulation.memory).all())
        self.assertGreater(np.ptp(simulation.memory), 0.0)
        self.assertGreater(np.max(simulation.voltage_mv), 0.0)

    def test_period_is_positive_and_clipped(self):
        trace = _adapting_trace()
        waveform = fit_phase_template_ladder([trace])
        result = fit_phase_timing_ladder(
            waveform.cycles,
            voltage_reference_mv=-65.0,
        )
        model = result.izhikevich_recovery

        self.assertAlmostEqual(
            model.period(100.0, -1e6),
            model.minimum_period_ms,
        )
        self.assertAlmostEqual(
            model.period(100.0, 1e6),
            model.maximum_period_ms,
        )

    def test_compact_spike_memory_and_fixed_parameter_export(self):
        trace = _adapting_trace()
        waveform = fit_phase_template_ladder(
            [trace],
            resting_voltage_mv=-65.0,
        )
        timing = fit_spike_exponential_timing(
            waveform.cycles,
            voltage_reference_mv=-65.0,
            config=PhaseTimingConfig(
                tau_candidates_ms=(20.0, 100.0),
            ),
        )
        self.assertIn(timing.model.tau_ms, (20.0, 100.0))
        self.assertEqual(
            sum(row["tau_selected"] for row in timing.candidate_table),
            1,
        )

        parameters = compact_model_parameter_row(
            waveform.no_slow_model,
            timing.model,
            harmonics=3,
        )
        self.assertIn("template_cos_h03_mv_q100", parameters)
        self.assertIn("baseline_period_q050_ms", parameters)
        self.assertIn("timing_spike_jump", parameters)
        self.assertNotIn("family", parameters)
        self.assertTrue(
            all(
                np.isfinite(value)
                for value in parameters.values()
                if isinstance(value, (float, int))
            )
        )

    def test_fixed_multiscale_kernel_fit_simulation_and_export(self):
        trace = _adapting_trace()
        waveform = fit_phase_template_ladder(
            [trace],
            resting_voltage_mv=-65.0,
        )
        result = fit_phase_kernel_timing(
            waveform.cycles,
            config=PhaseTimingConfig(
                kernel_tau_basis_ms=(25.0, 100.0, 400.0),
            ),
        )

        self.assertEqual(
            result.model.family,
            "fixed_multiscale_kernel",
        )
        self.assertEqual(len(result.model.amplitudes), 3)
        self.assertTrue(np.isfinite(result.late_period_nrmse))
        exported = kernel_timing_parameter_row(result.model)
        self.assertIn("kernel_value_0200ms", exported)
        compact = compact_kernel_model_parameter_row(
            waveform.no_slow_model,
            result.model,
            harmonics=3,
        )
        self.assertIn("kernel_log_tau_centroid_ms", compact)
        self.assertIn("template_cos_h03_mv_q100", compact)

        simulation = simulate_phase_kernel_timing_model(
            waveform.no_slow_model,
            result.model,
            trace.time_ms,
            trace.input_value,
            initial_voltage_mv=-65.0,
        )
        self.assertTrue(np.isfinite(simulation.voltage_mv).all())
        self.assertTrue(np.isfinite(simulation.memory).all())
        self.assertGreater(np.max(simulation.voltage_mv), 0.0)


if __name__ == "__main__":
    unittest.main()
