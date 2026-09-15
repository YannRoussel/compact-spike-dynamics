import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import CELL_PARAMETER_NAMES
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
)
from inverse_ephys_alpha_beta.kinetics import (
    PARAMETER_NAMES,
    default_parameter_bounds,
)
from inverse_ephys_alpha_beta.model_ladder import (
    AIS_PARAMETER_NAMES,
    _steady_currents,
)
from inverse_ephys_alpha_beta.phase_pareto import (
    NESTED_PHASE_VARIANTS,
    decode_nested_model,
    nested_parameter_spec,
)


class PhaseParetoTests(unittest.TestCase):
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
        baseline_values = {
            name: 0.0
            for name in (*CELL_PARAMETER_NAMES, *AIS_PARAMETER_NAMES)
        }
        self.baseline = pd.DataFrame([baseline_values])
        lower, upper = default_parameter_bounds()
        bounds = {
            name: [float(low), float(high)]
            for name, low, high in zip(PARAMETER_NAMES, lower, upper)
        }
        temporary = tempfile.NamedTemporaryFile(
            suffix=".json",
            mode="w",
            delete=False,
        )
        json.dump({"bounds": bounds}, temporary)
        temporary.close()
        self.bounds_path = Path(temporary.name)

    def tearDown(self):
        self.bounds_path.unlink(missing_ok=True)

    def test_nested_variants_add_only_the_intended_parameters(self):
        counts = {}
        for variant in NESTED_PHASE_VARIANTS:
            names, lower, upper, center = nested_parameter_spec(
                variant,
                self.baseline,
                self.bounds_path,
                1.5,
            )
            counts[variant] = len(names)
            self.assertEqual(center.shape, lower.shape)
            self.assertEqual(center.shape, upper.shape)
            self.assertTrue(np.all(center >= lower))
            self.assertTrue(np.all(center <= upper))
        self.assertEqual(counts["shared-alpha-beta"], 23)
        self.assertEqual(counts["ais-m-shift"], 24)
        self.assertEqual(counts["flexible-shared-m"], 29)

    def test_all_nested_models_preserve_soma_and_ais_rest(self):
        for variant in NESTED_PHASE_VARIANTS:
            with self.subTest(variant=variant):
                _, _, _, center = nested_parameter_spec(
                    variant,
                    self.baseline,
                    self.bounds_path,
                    1.5,
                )
                model = decode_nested_model(
                    variant,
                    center,
                    self.target,
                    1.5,
                )
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


if __name__ == "__main__":
    unittest.main()
