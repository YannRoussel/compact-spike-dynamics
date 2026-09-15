"""Tests for broad-family representative overlay helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.family_overlays import (
    QUALITY_COLUMNS,
    TYPICALITY_COLUMNS,
    major_families,
    rank_representative_candidates,
    recording_qc_metrics,
    stable_cycle_slices,
    typical_cycle_slice,
)
from inverse_ephys_alpha_beta.abstract_phase_model import AbstractObservedTrace


class FamilyOverlayTests(unittest.TestCase):
    def test_major_families_follow_biological_order_and_threshold(self) -> None:
        metadata = pd.DataFrame(
            {
                "broad_class": (
                    ["Vip"] * 12
                    + ["Pvalb"] * 14
                    + ["Glut_IT"] * 11
                    + ["Glut_NP"] * 3
                    + ["Other"] * 20
                )
            }
        )
        self.assertEqual(
            major_families(metadata, minimum_cells=10),
            ("Pvalb", "Vip", "Glut_IT"),
        )

    def test_representative_is_typical_within_better_quality_half(self) -> None:
        rows = []
        for index in range(6):
            row = {
                "cell_id": str(index),
                **{
                    name: float(index)
                    for name in QUALITY_COLUMNS
                },
                **{
                    name: float(index)
                    for name in TYPICALITY_COLUMNS
                },
            }
            rows.append(row)
        ranked = rank_representative_candidates(pd.DataFrame(rows))
        self.assertTrue(bool(ranked.iloc[0]["selection_better_half"]))
        self.assertEqual(ranked.iloc[0]["cell_id"], "2")

    def test_typical_cycle_uses_stable_crossing_pairs(self) -> None:
        time = np.linspace(0.0, 100.0, 1001)
        voltage = -10.0 + 20.0 * np.sin(2.0 * np.pi * time / 10.0)
        cycles = stable_cycle_slices(time, voltage, 5.0, 95.0)
        self.assertGreaterEqual(len(cycles), 5)
        selected = typical_cycle_slice(time, voltage, 5.0, 95.0)
        period = time[selected.stop - 1] - time[selected.start]
        self.assertAlmostEqual(period, 10.0, places=1)

    def test_recording_qc_accepts_a_stable_repetitive_trace(self) -> None:
        time = np.linspace(0.0, 120.0, 1201)
        voltage = np.full_like(time, -70.0)
        for spike_time in np.arange(25.0, 105.0, 10.0):
            center = int(np.argmin(np.abs(time - spike_time)))
            voltage[center:center + 3] = (5.0, 30.0, 5.0)
        velocity = np.gradient(voltage, time)
        trace = AbstractObservedTrace(
            name="healthy",
            role="validation",
            time_ms=time,
            voltage_mv=voltage,
            velocity_mv_ms=velocity,
            acceleration_mv_ms2=np.gradient(velocity, time),
            input_value=np.where(
                (time >= 20.0) & (time < 110.0),
                100.0,
                0.0,
            ),
            fit_mask=np.ones_like(time, dtype=bool),
            stimulus_start_ms=20.0,
            stimulus_end_ms=110.0,
        )
        qc = recording_qc_metrics(trace)
        self.assertTrue(bool(qc["recording_qc_pass"]))
        self.assertGreater(float(qc["spike_amplitude_retention"]), 0.95)


if __name__ == "__main__":
    unittest.main()
