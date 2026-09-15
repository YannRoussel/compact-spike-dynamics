import unittest

from inverse_ephys_alpha_beta.dataset import DatasetConfig, generate_dataset
from inverse_ephys_alpha_beta.features import FeatureConfig
from inverse_ephys_alpha_beta.hh_model import SimulationConfig
from inverse_ephys_alpha_beta.kinetics import PARAMETER_NAMES
from inverse_ephys_alpha_beta.protocols import biological_screen_config
from inverse_ephys_alpha_beta.static_parameters import STATIC_PARAMETER_NAMES


class DatasetTests(unittest.TestCase):
    def test_canonical_model_is_retained(self):
        accepted, rejected, metadata = generate_dataset(
            DatasetConfig(n_samples=1, include_canonical=True),
            SimulationConfig(),
            FeatureConfig(min_spikes=2),
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(metadata["acceptance_fraction"], 1.0)
        self.assertTrue(set(PARAMETER_NAMES).issubset(accepted.columns))
        self.assertTrue(set(STATIC_PARAMETER_NAMES).issubset(accepted.columns))
        self.assertIn("physical__total_capacitance_pf", accepted.columns)
        self.assertIn("feature__upstroke_inflection_voltage_mv", accepted.columns)
        self.assertIn("feature__downstroke_inflection_neg_iion_ua_cm2", accepted.columns)
        self.assertIn("feature__first_onset_rapidness_per_ms", accepted.columns)
        self.assertIn("feature__first_ap_phase_area_normalized", accepted.columns)
        self.assertIn(
            "feature__first_cycle_phase_path_length_normalized",
            accepted.columns,
        )

    def test_biological_screen_metadata_records_single_spike_acceptance(self):
        accepted, rejected, metadata = generate_dataset(
            DatasetConfig(
                n_samples=1,
                include_canonical=True,
                protocol="biological-screen",
            ),
            feature_config=FeatureConfig(min_spikes=2),
            screen_config=biological_screen_config("fast"),
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(metadata["feature_config"]["min_spikes"], 1)


if __name__ == "__main__":
    unittest.main()
