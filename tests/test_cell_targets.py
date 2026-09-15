import unittest

import numpy as np

from inverse_ephys_alpha_beta.cell_targets import (
    admittance_anchored_biophysics,
    anchored_biophysics,
    derive_passive_anchor,
    resting_state_is_stable,
    steady_current_density,
)
from inverse_ephys_alpha_beta.kinetics import KineticParameters


class CellTargetTests(unittest.TestCase):
    def _anchor(self):
        return derive_passive_anchor(
            resting_voltage_mv=-65.0,
            input_resistance_mohm=100.0,
            membrane_tau_ms=10.0,
            passive_current_pa=-40.0,
            passive_duration_ms=600.0,
            passive_sweep_number=7,
        )

    def test_passive_anchor_converts_resistance_and_tau_to_area_and_leak(self):
        anchor = self._anchor()

        self.assertTrue(np.isclose(anchor.total_capacitance_pf, 100.0))
        self.assertTrue(np.isclose(anchor.membrane_area_um2, 10_000.0))
        self.assertTrue(np.isclose(anchor.gleak_ms_cm2, 0.1))

    def test_anchored_biophysics_has_zero_current_at_measured_rest(self):
        anchor = self._anchor()
        kinetics = KineticParameters.canonical()
        biophysics = anchored_biophysics(anchor, kinetics)

        self.assertLess(
            abs(steady_current_density(-65.0, kinetics, biophysics)),
            1e-10,
        )
        self.assertTrue(resting_state_is_stable(anchor, kinetics, biophysics))

    def test_admittance_anchor_matches_rest_and_passive_endpoint(self):
        anchor = derive_passive_anchor(
            resting_voltage_mv=-65.0,
            input_resistance_mohm=100.0,
            membrane_tau_ms=0.5,
            passive_current_pa=-20.0,
            passive_duration_ms=200.0,
            passive_sweep_number=1,
        )
        kinetics = KineticParameters.canonical()
        biophysics = admittance_anchored_biophysics(anchor, kinetics)
        passive_voltage = (
            anchor.resting_voltage_mv
            + anchor.passive_current_pa
            * anchor.input_resistance_mohm
            / 1000.0
        )
        applied_density = biophysics.current_pa_to_density(
            anchor.passive_current_pa
        )

        self.assertLess(
            abs(steady_current_density(-65.0, kinetics, biophysics)),
            1e-10,
        )
        self.assertAlmostEqual(
            steady_current_density(
                passive_voltage,
                kinetics,
                biophysics,
            ),
            float(applied_density),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
