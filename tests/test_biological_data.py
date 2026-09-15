import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.biological_data import (
    _scala_sag_ratio_to_fraction,
    harmonize_raw_spike_cycle_features,
)


class BiologicalDataTests(unittest.TestCase):
    def test_scala_sag_ratio_conversion(self):
        ratios = pd.Series([1.0, 1.25, 2.0])
        fractions = _scala_sag_ratio_to_fraction(ratios)
        np.testing.assert_allclose(fractions, [0.0, 0.2, 0.5])

    def test_raw_harmonization_keeps_only_successful_identifiers(self):
        raw = pd.DataFrame(
            {
                "dataset": ["scala_room_temperature", "gouwens_visp"],
                "cell_id": ["cell-a", "cell-b"],
                "raw_status": ["ok", "error:OSError"],
                "feature__first_threshold_voltage_mv": [-40.0, -35.0],
            }
        )

        harmonized = harmonize_raw_spike_cycle_features(raw)

        self.assertEqual(len(harmonized), 1)
        self.assertEqual(harmonized.loc[0, "dataset"], "scala_room_temperature")
        self.assertEqual(harmonized.loc[0, "cell_id"], "cell-a")
        self.assertEqual(harmonized.loc[0, "ap_threshold_mv"], -40.0)


if __name__ == "__main__":
    unittest.main()
