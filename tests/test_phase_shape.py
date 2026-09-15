import unittest

import numpy as np

from inverse_ephys_alpha_beta.hh_model import Stimulus
from inverse_ephys_alpha_beta.phase_shape import (
    PhaseCycle,
    PhaseShapeConfig,
    compare_phase_cycles,
    extract_phase_cycle,
)


def synthetic_cycle(concavity_sign: float = -1.0) -> PhaseCycle:
    grid = np.linspace(0.0, 1.0, 64)
    up = 0.05 + 0.9 * (
        grid + concavity_sign * 0.35 * grid * (1.0 - grid)
    )
    down = -0.75 * np.sin(np.pi * grid)
    up_slope = np.gradient(up, grid)
    down_slope = np.gradient(down, grid)
    up_concavity = np.gradient(up_slope, grid)
    down_concavity = np.gradient(down_slope, grid)
    return PhaseCycle(
        grid=grid,
        up_dvdt_mv_ms=up * 300.0,
        down_dvdt_mv_ms=down * 300.0,
        up_normalized=up,
        down_normalized=down,
        up_slope=up_slope,
        down_slope=down_slope,
        up_concavity=up_concavity,
        down_concavity=down_concavity,
        threshold_voltage_mv=-50.0,
        peak_voltage_mv=30.0,
        down_end_voltage_mv=-50.0,
        amplitude_mv=80.0,
        repolarization_amplitude_mv=80.0,
        max_dvdt_mv_ms=300.0,
        min_dvdt_mv_ms=-225.0,
        dvdt_span_mv_ms=525.0,
        up_max_relative_voltage=1.0,
        down_min_relative_voltage=0.5,
    )


class PhaseShapeTests(unittest.TestCase):
    def test_identical_cycles_have_zero_loss(self):
        cycle = synthetic_cycle()
        scores = compare_phase_cycles(cycle, cycle)
        self.assertAlmostEqual(scores.total, 0.0, places=12)
        self.assertAlmostEqual(
            scores.onset_concavity_sign_agreement,
            1.0,
        )

    def test_opposite_concavity_is_penalized(self):
        biological = synthetic_cycle(-1.0)
        opposite = synthetic_cycle(1.0)
        scores = compare_phase_cycles(opposite, biological)
        self.assertGreater(scores.concavity, 0.5)
        self.assertLess(scores.onset_concavity_sign_agreement, 0.25)

    def test_extracts_a_smoothed_spike_cycle(self):
        dt = 0.01
        time = np.arange(0.0, 30.0 + dt, dt)
        voltage = np.full_like(time, -65.0)
        spike = 100.0 * np.exp(-((time - 12.0) / 0.45) ** 2)
        voltage += spike
        stimulus = Stimulus(
            amplitude_ua_cm2=None,
            amplitude_pa=100.0,
            start_ms=10.0,
            end_ms=20.0,
        )
        cycle = extract_phase_cycle(
            time,
            voltage,
            stimulus,
            PhaseShapeConfig(spline_smoothing=0.0001),
        )
        self.assertEqual(cycle.grid.shape, (64,))
        self.assertGreater(cycle.max_dvdt_mv_ms, 50.0)
        self.assertLess(cycle.min_dvdt_mv_ms, -50.0)
        self.assertTrue(np.all(np.isfinite(cycle.up_concavity)))


if __name__ == "__main__":
    unittest.main()
