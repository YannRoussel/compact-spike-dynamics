import unittest

import numpy as np

from inverse_ephys_alpha_beta.features import (
    SPIKE_CYCLE_OBSERVABLE_FEATURES,
    FeatureConfig,
    extract_features,
    extract_voltage_features,
)
from inverse_ephys_alpha_beta.hh_model import SimulationConfig, simulate


class HodgkinHuxleyTests(unittest.TestCase):
    def test_canonical_model_spikes(self):
        config = SimulationConfig()
        trace = simulate(config=config)
        features = extract_features(trace, config.stimulus, FeatureConfig(min_spikes=2))
        self.assertGreaterEqual(features["spike_count"], 2)
        self.assertEqual(features["is_spiking"], 1.0)
        self.assertTrue(np.isfinite(features["upstroke_inflection_voltage_mv"]))
        self.assertTrue(np.isfinite(features["downstroke_inflection_voltage_mv"]))
        self.assertTrue(np.isfinite(features["first_upstroke_downstroke_ratio"]))
        self.assertTrue(
            np.isfinite(
                [features[name] for name in SPIKE_CYCLE_OBSERVABLE_FEATURES]
            ).all()
        )
        self.assertGreater(features["first_ap_phase_area_v2_ms"], 0.0)
        self.assertGreater(features["first_cycle_phase_path_length_normalized"], 0.0)

    def test_voltage_only_observation_pipeline_matches_clean_model_trace(self):
        config = SimulationConfig()
        trace = simulate(config=config)
        feature_config = FeatureConfig(min_spikes=1)
        exact = extract_features(trace, config.stimulus, feature_config)
        observed = extract_voltage_features(
            trace.time_ms,
            trace.voltage_mv,
            config.stimulus,
            feature_config,
        )
        self.assertTrue(
            np.isfinite(
                [observed[name] for name in SPIKE_CYCLE_OBSERVABLE_FEATURES]
            ).all()
        )
        self.assertAlmostEqual(
            observed["first_threshold_voltage_mv"],
            exact["first_threshold_voltage_mv"],
            delta=0.5,
        )
        self.assertAlmostEqual(
            observed["first_half_width_ms"],
            exact["first_half_width_ms"],
            delta=0.05,
        )
        self.assertTrue(
            np.isnan(observed["upstroke_inflection_neg_iion_ua_cm2"])
        )


if __name__ == "__main__":
    unittest.main()
