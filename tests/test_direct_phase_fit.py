import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
)
from inverse_ephys_alpha_beta.direct_kinetics import (
    DIRECT_KINETIC_PARAMETER_NAMES,
)
from inverse_ephys_alpha_beta.direct_phase_fit import (
    DIRECT_ALL_PARAMETER_NAMES,
    DIRECT_STATIC_PARAMETER_NAMES,
    DirectPhaseFitConfig,
    _baseline_centers,
    _joint_trust_bounds,
    _static_bounds,
    decode_direct_model,
)
from inverse_ephys_alpha_beta.model_ladder import (
    AIS_PARAMETER_NAMES,
    _steady_currents,
)


class DirectPhaseFitTests(unittest.TestCase):
    def setUp(self):
        passive = PassiveAnchor(
            resting_voltage_mv=-65.0,
            input_resistance_mohm=10.0,
            membrane_tau_ms=1.0,
            total_capacitance_pf=100.0,
            membrane_area_um2=10_000.0,
            capacitance_uf_cm2=1.0,
            gleak_ms_cm2=1.0,
            passive_current_pa=-40.0,
            passive_duration_ms=600.0,
            passive_sweep_number=1,
        )
        self.target = CellOptimizationTarget(
            dataset="test",
            cell_id="cell",
            nwb_path="cell.nwb",
            sampled_rheobase_pa=100.0,
            temperature_c=22.0,
            passive=passive,
            training_protocols=(),
            validation_protocols=(),
        )
        self.baseline = pd.DataFrame(
            [
                {
                    name: 0.0
                    for name in (
                        *CELL_PARAMETER_NAMES,
                        *AIS_PARAMETER_NAMES,
                    )
                }
            ]
        )

    def test_direct_model_preserves_soma_and_ais_rest(self):
        kinetic, static = _baseline_centers(
            self.baseline,
            DirectPhaseFitConfig(),
        )
        model = decode_direct_model(kinetic, static, self.target)
        self.assertAlmostEqual(
            sum(_steady_currents(-65.0, model, "soma")),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            sum(_steady_currents(-65.0, model, "ais")),
            0.0,
            places=8,
        )

    def test_joint_trust_region_has_expected_parameter_count(self):
        config = DirectPhaseFitConfig()
        kinetic, static = _baseline_centers(self.baseline, config)
        static_lower, static_upper = _static_bounds(config.gna_gk_factor)
        kinetic_lower = np.full(len(DIRECT_KINETIC_PARAMETER_NAMES), -20.0)
        kinetic_upper = np.full(len(DIRECT_KINETIC_PARAMETER_NAMES), 20.0)
        center = np.concatenate((kinetic, static))
        lower, upper = _joint_trust_bounds(
            center,
            np.concatenate((kinetic_lower, static_lower)),
            np.concatenate((kinetic_upper, static_upper)),
            config,
        )
        self.assertEqual(len(DIRECT_ALL_PARAMETER_NAMES), len(center))
        self.assertEqual(
            len(DIRECT_STATIC_PARAMETER_NAMES),
            len(static),
        )
        self.assertTrue(np.all(lower < upper))
        self.assertTrue(np.all(center >= lower))
        self.assertTrue(np.all(center <= upper))


if __name__ == "__main__":
    unittest.main()
