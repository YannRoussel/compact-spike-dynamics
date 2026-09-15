import unittest

import numpy as np

from inverse_ephys_alpha_beta.activation_kinetics import (
    FLEXIBLE_M_VOLTAGE_KNOTS_MV,
    FlexibleActivationKinetics,
    ShiftedActivationKinetics,
    flexible_m_initial,
)
from inverse_ephys_alpha_beta.kinetics import KineticParameters


class ActivationKineticsTests(unittest.TestCase):
    def test_flexible_initial_matches_parent_at_knots(self):
        parent = KineticParameters.canonical()
        parameters = flexible_m_initial(parent)
        flexible = FlexibleActivationKinetics.from_parameters(
            parent,
            parameters,
        )
        voltage = np.asarray(FLEXIBLE_M_VOLTAGE_KNOTS_MV)
        parent_rates = parent.rates(voltage)
        parent_inf = parent_rates["alpha_m"] / (
            parent_rates["alpha_m"] + parent_rates["beta_m"]
        )
        flexible_inf, flexible_tau = flexible.activation_curves(voltage)
        self.assertTrue(np.allclose(parent_inf, flexible_inf, atol=1e-7))
        self.assertTrue(np.all(flexible_tau > 0.0))

    def test_flexible_activation_is_monotone_and_rates_are_positive(self):
        parent = KineticParameters.canonical()
        flexible = FlexibleActivationKinetics.from_parameters(
            parent,
            flexible_m_initial(parent),
        )
        voltage = np.linspace(-120.0, 80.0, 501)
        steady, _ = flexible.activation_curves(voltage)
        rates = flexible.rates(voltage)
        self.assertTrue(np.all(np.diff(steady) >= -1e-12))
        self.assertTrue(np.all(rates["alpha_m"] > 0.0))
        self.assertTrue(np.all(rates["beta_m"] > 0.0))

    def test_activation_shift_does_not_change_h_or_n(self):
        base = KineticParameters.canonical()
        shifted = ShiftedActivationKinetics(base, -8.0)
        original_rates = base.rates_scalar(-50.0)
        shifted_rates = shifted.rates_scalar(-50.0)
        self.assertNotAlmostEqual(original_rates[0], shifted_rates[0])
        self.assertEqual(original_rates[2:], shifted_rates[2:])


if __name__ == "__main__":
    unittest.main()
