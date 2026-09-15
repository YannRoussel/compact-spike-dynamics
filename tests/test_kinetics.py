import unittest

import numpy as np

from inverse_ephys_alpha_beta.kinetics import (
    KineticParameters,
    PARAMETER_NAMES,
    default_parameter_bounds,
)


class KineticsTests(unittest.TestCase):
    def test_canonical_rates_are_positive_and_finite(self):
        rates = KineticParameters.canonical().rates(np.linspace(-120.0, 60.0, 361))
        for values in rates.values():
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(values > 0.0))

    def test_vector_round_trip(self):
        lower, upper = default_parameter_bounds()
        vector = (lower + upper) / 2.0
        restored = KineticParameters.from_vector(vector).to_vector()
        np.testing.assert_allclose(restored, vector)
        self.assertEqual(len(PARAMETER_NAMES), 18)


if __name__ == "__main__":
    unittest.main()
