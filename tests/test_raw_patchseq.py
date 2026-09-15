import unittest

import numpy as np

from inverse_ephys_alpha_beta.raw_patchseq import (
    _as_voltage_mv,
    _longest_stimulus_epoch,
    infer_scala_cell_id,
)


class _ArrayDataset:
    def __init__(self, values, **attrs):
        self._values = np.asarray(values, dtype=float)
        self.attrs = attrs

    def __getitem__(self, item):
        return self._values[item]


class RawPatchSeqTests(unittest.TestCase):
    def test_longest_stimulus_epoch_ignores_short_test_pulse(self):
        current = np.zeros(1000)
        current[50:60] = -20.0
        current[200:800] = 140.0

        start, end, amplitude, plateau_mad = _longest_stimulus_epoch(current, 1000.0)

        self.assertAlmostEqual(start, 200.0)
        self.assertAlmostEqual(end, 800.0)
        self.assertAlmostEqual(amplitude, 140.0)
        self.assertAlmostEqual(plateau_mad, 0.0)

    def test_scala_filename_maps_to_release_cell_id(self):
        path = (
            "sub-mouse-ZLYQM_ses-20190703-sample-12_"
            "slice-20190703-slice-12_cell-20190703-sample-12_icephys.nwb"
        )
        self.assertEqual(infer_scala_cell_id(path), "20190703_sample_12")

    def test_voltage_conversion_handles_mixed_legacy_scala_scaling(self):
        already_mv = _ArrayDataset(
            [-65.0, 30.0],
            unit="mV",
            conversion=0.001,
        )
        microvolt_like = _ArrayDataset(
            [-65000.0, 30000.0],
            unit="mV",
            conversion=1.0,
        )
        volts = _ArrayDataset(
            [-0.065, 0.030],
            unit="V",
            conversion=1.0,
        )
        near_zero_median_mv = _ArrayDataset(
            [-63.0, *([0.0] * 20), 72.0],
            unit="V",
            conversion=1.0,
        )

        expected = np.asarray([-65.0, 30.0])
        np.testing.assert_allclose(_as_voltage_mv(already_mv), expected)
        np.testing.assert_allclose(_as_voltage_mv(microvolt_like), expected)
        np.testing.assert_allclose(_as_voltage_mv(volts), expected)
        np.testing.assert_allclose(
            _as_voltage_mv(near_zero_median_mv),
            near_zero_median_mv._values,
        )


if __name__ == "__main__":
    unittest.main()
