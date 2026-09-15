import unittest

import numpy as np

from inverse_ephys_alpha_beta.direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
    DirectKinetics,
    direct_kinetic_initial,
    direct_kinetic_parameter_bounds,
)
from inverse_ephys_alpha_beta.kinetics import KineticParameters


class DirectKineticsTests(unittest.TestCase):
    def test_initialization_preserves_gate_curves_at_knots(self):
        canonical = KineticParameters.canonical()
        parameters = direct_kinetic_initial(canonical)
        direct = DirectKinetics.from_parameters(parameters)
        voltage = np.asarray(direct.voltage_knots_mv)
        canonical_rates = canonical.rates(voltage)
        direct_rates = direct.rates(voltage)
        for gate in ("m", "h", "n"):
            canonical_total = (
                canonical_rates[f"alpha_{gate}"]
                + canonical_rates[f"beta_{gate}"]
            )
            direct_total = (
                direct_rates[f"alpha_{gate}"]
                + direct_rates[f"beta_{gate}"]
            )
            canonical_steady = (
                canonical_rates[f"alpha_{gate}"] / canonical_total
            )
            direct_steady = direct_rates[f"alpha_{gate}"] / direct_total
            np.testing.assert_allclose(direct_steady, canonical_steady)
            self.assertTrue(np.all(np.isfinite(direct_total)))
            self.assertTrue(np.all(direct_total > 0.0))

    def test_bounds_and_monotonicity(self):
        lower, upper = direct_kinetic_parameter_bounds()
        initial = direct_kinetic_initial(KineticParameters.canonical())
        self.assertEqual(
            initial.shape,
            (len(DIRECT_KINETIC_PARAMETER_NAMES),),
        )
        self.assertTrue(np.all(initial > lower))
        self.assertTrue(np.all(initial < upper))
        direct = DirectKinetics.from_parameters(initial)
        voltage = np.linspace(-100.0, 60.0, 401)
        m_inf, _ = direct.gate_curves("m", voltage)
        h_inf, _ = direct.gate_curves("h", voltage)
        n_inf, _ = direct.gate_curves("n", voltage)
        self.assertTrue(np.all(np.diff(m_inf) >= 0.0))
        self.assertTrue(np.all(np.diff(h_inf) <= 0.0))
        self.assertTrue(np.all(np.diff(n_inf) >= 0.0))

    def test_scalar_and_vector_rates_agree(self):
        direct = DirectKinetics.from_parameters(
            direct_kinetic_initial(KineticParameters.canonical())
        )
        vector_rates = direct.rates(np.asarray((-65.0,)))
        scalar_rates = direct.rates_scalar(-65.0)
        expected = tuple(
            float(vector_rates[name][0])
            for name in (
                "alpha_m",
                "beta_m",
                "alpha_h",
                "beta_h",
                "alpha_n",
                "beta_n",
            )
        )
        np.testing.assert_allclose(scalar_rates, expected)


if __name__ == "__main__":
    unittest.main()
