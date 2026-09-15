import math
import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.staged_ladder import (
    StagedLadderConfig,
    _diverse_fast_elites,
    _trust_bounds,
)


class StagedLadderTests(unittest.TestCase):
    def test_trust_bounds_use_additive_voltage_and_multiplicative_logs(self):
        names = (
            "param__alpha_m__log_rate_scale",
            "param__alpha_m__voltage_shift_mv",
        )
        center = np.asarray((0.2, -3.0))
        lower, upper = _trust_bounds(
            names,
            center,
            np.asarray((-2.0, -20.0)),
            np.asarray((2.0, 20.0)),
            StagedLadderConfig(
                trust_fraction=0.2,
                voltage_shift_margin_mv=4.0,
            ),
        )

        self.assertAlmostEqual(lower[0], 0.2 - math.log(1.2))
        self.assertAlmostEqual(upper[0], 0.2 + math.log(1.2))
        self.assertAlmostEqual(lower[1], -7.0)
        self.assertAlmostEqual(upper[1], 1.0)

    def test_fast_elites_are_ranked_and_parameter_diverse(self):
        rows = []
        for index, offset in enumerate((0.0, 0.01, 0.8)):
            row = {
                name: offset
                for name in CELL_PARAMETER_NAMES
            }
            row.update(
                {
                    "valid": True,
                    "passive": 0.0,
                    "excitability": 0.0,
                    "spike_shape": float(index),
                    "spike_dynamics": 0.0,
                    "phase_geometry": 0.0,
                    "conductance_prior": 0.0,
                }
            )
            rows.append(row)
        history = pd.DataFrame(rows)
        elites = _diverse_fast_elites(
            history,
            np.full(len(CELL_PARAMETER_NAMES), -1.0),
            np.full(len(CELL_PARAMETER_NAMES), 1.0),
            count=2,
            minimum_distance=0.1,
        )

        self.assertEqual(len(elites), 2)
        self.assertAlmostEqual(elites.iloc[0][CELL_PARAMETER_NAMES[0]], 0.0)
        self.assertAlmostEqual(elites.iloc[1][CELL_PARAMETER_NAMES[0]], 0.8)


if __name__ == "__main__":
    unittest.main()
