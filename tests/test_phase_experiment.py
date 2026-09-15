import unittest

import numpy as np

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.kinetics import KineticParameters
from inverse_ephys_alpha_beta.phase_experiment import (
    AIS_PARAMETER_NAMES,
    KineticDiagnostics,
    kinetic_diagnostics,
)


class PhaseExperimentTests(unittest.TestCase):
    def test_canonical_kinetics_are_not_pathological(self):
        diagnostics = kinetic_diagnostics(
            KineticParameters.canonical(),
            (1.0, 1.0, 1.0),
        )
        self.assertIsInstance(diagnostics, KineticDiagnostics)
        self.assertGreaterEqual(diagnostics.monotonic_fraction, 0.98)
        self.assertEqual(diagnostics.pathology_score, 0.0)
        self.assertGreater(diagnostics.minimum_tau_ms, 0.0)

    def test_soma_ais_adds_only_three_structural_parameters(self):
        self.assertEqual(len(AIS_PARAMETER_NAMES), 3)
        self.assertEqual(
            len(CELL_PARAMETER_NAMES) + len(AIS_PARAMETER_NAMES),
            23,
        )


if __name__ == "__main__":
    unittest.main()
