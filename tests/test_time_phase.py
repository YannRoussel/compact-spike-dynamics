import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.abstract_phase_model import AbstractObservedTrace
from inverse_ephys_alpha_beta.phase_template_model import PhaseTemplateModel
from inverse_ephys_alpha_beta.recovery_onset import OnsetModel
from inverse_ephys_alpha_beta.time_phase import (
    SEQUENCE_FEATURES, SequenceClock, clock_from_row, clock_parameter_row,
    extract_time_phase_features, fit_sequence_clock, temporal_basis, train_metrics,
    simulate_sequence,
)


class TimePhaseTests(unittest.TestCase):
    def test_sinusoidal_loop_area_and_time_translation(self):
        t = np.arange(0, 100, 0.01)
        omega = 2 * np.pi / 10
        v, u = 20 * np.cos(omega * t), -20 * omega * np.sin(omega * t)
        trace = AbstractObservedTrace("test", "training", t, v, u,
            -20 * omega**2 * np.cos(omega * t), np.ones_like(t),
            np.ones_like(t, dtype=bool), 0, 100)
        table = extract_time_phase_features(trace)
        np.testing.assert_allclose(table.isi_ms, 10, atol=0.02)
        np.testing.assert_allclose(table.loop_area_mv2_ms, np.pi * 400 * omega, rtol=1e-4)
        shifted = extract_time_phase_features(replace(trace, time_ms=t + 1000,
            stimulus_start_ms=1000, stimulus_end_ms=1100))
        np.testing.assert_allclose(table.elapsed_ms, shifted.elapsed_ms)
        self.assertTrue((table.acceleration_root_count >= 1).all())
        self.assertAlmostEqual(train_metrics(trace, v)["adaptation_ratio_error"], 0)
        self.assertTrue(np.isnan(train_metrics(trace, np.full_like(v, -65))["isi_rmse_ms"]))

    def test_adapting_sequence_fit_and_parameter_roundtrip(self):
        coef = np.zeros((2, 4, 5))
        coef[:, 0] = np.log([10, 100, 70, 200, 100])
        coef[:, 2] = [0.5, 0.1, 0.1, -0.2, -0.1]
        truth = SequenceClock(np.array([100, 200]), coef)
        tables = []
        for current in truth.currents:
            elapsed, rows = 0.0, []
            for i in range(30):
                values = truth.predict(current, [elapsed])[0]
                rows.append(dict(zip(SEQUENCE_FEATURES, values), elapsed_ms=elapsed,
                                 current_pa=current))
                elapsed += values[0]
            tables.append(pd.DataFrame(rows))
        model, _ = fit_sequence_clock(tables)
        np.testing.assert_allclose(model.predict(150, [0, 100, 300])[:, 0],
            truth.predict(150, [0, 100, 300])[:, 0], rtol=0.08)
        row = clock_parameter_row(model, 100, 200)
        row.update(repetitive_current_min_pa=100, repetitive_current_span_pa=100)
        recovered = clock_from_row(row)
        np.testing.assert_allclose(recovered.predict(150, [0, 300]), model.predict(150, [0, 300]))
        joint, _ = fit_sequence_clock(tables, joint=True)
        self.assertTrue(np.isfinite(joint.predict(150, [0, 300])).all())

    def test_negative_elapsed_is_rejected(self):
        with self.assertRaises(ValueError):
            temporal_basis([-1])

    def test_simulation_ignores_observed_voltage_and_events(self):
        phase = np.linspace(0, 1, 256, endpoint=False)
        wave = PhaseTemplateModel(tuple(phase), (100.0,),
            (tuple(-20 + 50 * np.cos(2 * np.pi * phase)),), (5.0,), (2.0,), (1.0,),
            -65.0, (10.0, 0.0), 0.0, 1.0, 1.0, None, 50.0, 1.0, 3.0, 100.0, 10.0)
        onset = OnsetModel((100.0,), (2.0,), (0.0,))
        clock = SequenceClock(np.array([100.0]), np.array([[[np.log(10)], [0], [0], [0]]]))
        t = np.arange(0, 120, 0.05)
        zero = np.zeros_like(t)
        trace = AbstractObservedTrace("test", "validation", t, zero, zero, zero,
            np.where((t >= 10) & (t < 110), 100.0, 0.0), np.ones_like(t, bool), 10, 110)
        a = simulate_sequence(wave, onset, clock, trace)
        b = simulate_sequence(wave, onset, clock,
            replace(trace, voltage_mv=100 * np.sin(t), velocity_mv_ms=100 * np.cos(t)))
        np.testing.assert_array_equal(a.voltage_mv, b.voltage_mv)
        self.assertTrue(np.isfinite(a.voltage_mv).all())
        self.assertGreater(np.max(a.voltage_mv), 0)


if __name__ == "__main__":
    unittest.main()
