from __future__ import annotations

import unittest

import numpy as np

from inverse_ephys_alpha_beta.abstract_phase_model import (
    AbstractObservedTrace,
)
from inverse_ephys_alpha_beta.phase_template_model import (
    PhaseTemplateConfig,
    extract_phase_cycles,
    fit_phase_template_ladder,
    phase_plane_chamfer_distance,
    simulate_phase_template_model,
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


class PhaseTemplateModelTests(unittest.TestCase):
    def test_landmark_cycles_and_slow_period_model(self):
        trace = _adapting_trace()
        config = PhaseTemplateConfig(
            slow_complexity_penalty=0.0,
        )
        cycles = extract_phase_cycles(trace, config)
        self.assertIsNotNone(cycles)
        self.assertEqual(cycles.voltage_cycles_mv.shape[1], 256)
        self.assertGreater(len(cycles.periods_ms), 5)

        result = fit_phase_template_ladder([trace], config=config)
        self.assertTrue(result.selected.has_slow_state)
        simulation = simulate_phase_template_model(
            result.selected,
            trace.time_ms,
            trace.input_value,
            initial_voltage_mv=-65.0,
        )
        self.assertTrue(np.isfinite(simulation.voltage_mv).all())
        self.assertLess(
            phase_plane_chamfer_distance(trace, simulation),
            0.08,
        )


if __name__ == "__main__":
    unittest.main()
