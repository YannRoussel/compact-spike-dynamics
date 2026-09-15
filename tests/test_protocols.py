import unittest
from unittest.mock import patch

import numpy as np

from inverse_ephys_alpha_beta.hh_model import (
    SimulationConfig,
    Stimulus,
    elicits_spike,
    equilibrium_state,
)
from inverse_ephys_alpha_beta.kinetics import KineticParameters
from inverse_ephys_alpha_beta.protocols import (
    _find_rheobase,
    biological_screen_config,
    run_biological_screen,
)
from inverse_ephys_alpha_beta.static_parameters import BiophysicalParameters


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.kinetics = KineticParameters.canonical()
        self.biophysics = BiophysicalParameters()
        self.resting_state = equilibrium_state(self.kinetics, self.biophysics)

    def _step_config(self, current_pa):
        return SimulationConfig(
            duration_ms=120.0,
            stimulus=Stimulus(
                amplitude_ua_cm2=None,
                amplitude_pa=current_pa,
                start_ms=20.0,
                end_ms=120.0,
            ),
            conductances=self.biophysics,
            temperature_c=22.0,
        )

    def test_adaptive_spike_test_brackets_canonical_model(self):
        self.assertFalse(
            elicits_spike(
                self.kinetics,
                self._step_config(600.0),
                self.resting_state,
            )
        )
        self.assertTrue(
            elicits_spike(
                self.kinetics,
                self._step_config(800.0),
                self.resting_state,
            )
        )

    def test_bisection_returns_narrow_spiking_upper_bound(self):
        config = biological_screen_config("fast")
        result = run_biological_screen(
            self.kinetics,
            self.biophysics,
            config,
        )
        self.assertLess(result.rheobase_lower_pa, result.rheobase_upper_pa)
        self.assertLessEqual(
            result.rheobase_upper_pa - result.rheobase_lower_pa,
            config.rheobase_tolerance_pa,
        )
        self.assertGreater(result.bisection_iterations, 0)
        self.assertAlmostEqual(
            result.features["waveform_current_pa"],
            result.rheobase_upper_pa,
        )
        self.assertEqual(result.features["rheobase_step_ms"], 100.0)

    def test_low_current_guard_preserves_depolarization_block_model(self):
        config = biological_screen_config("fast")

        def band_limited_spike(_kinetics, simulation, _state, **_kwargs):
            current = float(simulation.stimulus.amplitude_pa)
            return 7.0 <= current <= 80.0

        with patch(
            "inverse_ephys_alpha_beta.protocols.elicits_spike",
            side_effect=band_limited_spike,
        ):
            lower, upper, _ = _find_rheobase(
                self.kinetics,
                self.biophysics,
                np.asarray(self.resting_state),
                config,
            )

        self.assertLess(lower, 7.0)
        self.assertGreaterEqual(upper, 7.0)
        self.assertLessEqual(upper - lower, config.rheobase_tolerance_pa)

    def test_named_presets_have_expected_durations(self):
        self.assertEqual(
            biological_screen_config("scala").rheobase_step_ms,
            600.0,
        )
        self.assertEqual(
            biological_screen_config("gouwens").rheobase_step_ms,
            1000.0,
        )
        self.assertEqual(
            biological_screen_config("scala-phys").temperature_c,
            34.0,
        )
        self.assertEqual(
            biological_screen_config("gouwens").temperature_c,
            34.0,
        )


if __name__ == "__main__":
    unittest.main()
