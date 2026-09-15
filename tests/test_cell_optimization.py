import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.cell_optimization import (
    CELL_PARAMETER_NAMES,
    CellOptimizationConfig,
    feature_scale,
    load_cell_parameter_bounds,
    seed_vectors_from_population,
)
from inverse_ephys_alpha_beta.cell_targets import (
    CellOptimizationTarget,
    PassiveAnchor,
    ProtocolFeatureTarget,
)
from inverse_ephys_alpha_beta.kinetics import PARAMETER_NAMES


def _target():
    passive = PassiveAnchor(
        resting_voltage_mv=-65.0,
        input_resistance_mohm=100.0,
        membrane_tau_ms=10.0,
        total_capacitance_pf=100.0,
        membrane_area_um2=10_000.0,
        capacitance_uf_cm2=1.0,
        gleak_ms_cm2=0.1,
        passive_current_pa=-40.0,
        passive_duration_ms=600.0,
        passive_sweep_number=1,
    )
    rheobase = ProtocolFeatureTarget(
        name="rheobase",
        role="training",
        sweep_number=2,
        current_pa=100.0,
        rheobase_factor=1.0,
        duration_ms=600.0,
        features={
            "spike_count": 10.0,
            "first_threshold_voltage_mv": -40.0,
        },
    )
    return CellOptimizationTarget(
        dataset="test",
        cell_id="cell",
        nwb_path="cell.nwb",
        sampled_rheobase_pa=100.0,
        temperature_c=22.0,
        passive=passive,
        training_protocols=(rheobase,),
        validation_protocols=(),
    )


class CellOptimizationTests(unittest.TestCase):
    def test_load_cell_bounds_adds_symmetric_conductance_nuisance(self):
        bounds = {name: [-1.0, 1.0] for name in PARAMETER_NAMES}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounds.json"
            path.write_text(json.dumps({"bounds": bounds}))
            lower, upper = load_cell_parameter_bounds(
                path,
                gna_gk_factor=1.5,
            )

        self.assertEqual(len(lower), len(CELL_PARAMETER_NAMES))
        np.testing.assert_allclose(lower[-2:], -np.log(1.5))
        np.testing.assert_allclose(upper[-2:], np.log(1.5))

    def test_seed_population_is_ranked_and_clipped(self):
        row = {name: 0.0 for name in PARAMETER_NAMES}
        row.update(
            {
                "feature__rheobase_pa": 100.0,
                "feature__spike_count": 10.0,
                "feature__first_threshold_voltage_mv": -40.0,
                "param__static__log_gna_scale": 2.0,
                "param__static__log_gk_scale": -2.0,
                "param__static__q10_m": 4.0,
            }
        )
        lower = np.full(len(CELL_PARAMETER_NAMES), -0.5)
        upper = np.full(len(CELL_PARAMETER_NAMES), 0.5)

        source = pd.DataFrame([row])
        original = source.copy(deep=True)
        with pd.option_context("mode.copy_on_write", True):
            seeds = seed_vectors_from_population(
                source,
                _target(),
                lower,
                upper,
                maximum_seeds=1,
                missing_penalty=CellOptimizationConfig().missing_feature_penalty,
            )

        self.assertEqual(len(seeds), 1)
        np.testing.assert_allclose(seeds[0][-2:], (0.5, -0.5))
        pd.testing.assert_frame_equal(source, original)
        expected_shift = np.clip(
            ((_target().temperature_c - 6.3) / 10.0) * np.log(4.0 / 3.0),
            -0.5,
            0.5,
        )
        for rate in ("alpha", "beta"):
            index = PARAMETER_NAMES.index(f"param__{rate}_m__log_rate_scale")
            self.assertAlmostEqual(seeds[0][index], expected_shift)

    def test_dynamic_feature_scales_are_nonzero(self):
        self.assertEqual(feature_scale("spike_count", 1.0), 1.0)
        self.assertEqual(
            feature_scale("first_spike_latency_ms", 100.0),
            15.0,
        )


if __name__ == "__main__":
    unittest.main()
