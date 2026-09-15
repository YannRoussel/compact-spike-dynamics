import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.direct_optimization import (
    DirectOptimizationConfig,
    _select_multifidelity_elites,
    biological_normalization,
    fit_features,
    propose_evolution_batch,
    score_models,
    select_biological_medoids,
)
from inverse_ephys_alpha_beta.sampling import ALL_PARAMETER_NAMES


class DirectOptimizationTests(unittest.TestCase):
    def test_target_scores_zero_against_itself(self):
        features = fit_features()
        biological = pd.DataFrame(
            {
                feature: np.linspace(index, index + 4.0, 8)
                for index, feature in enumerate(features)
            }
        )
        normalization = biological_normalization(biological, features)
        target = biological.iloc[3]

        scores = score_models(
            biological.iloc[[3]],
            target,
            normalization,
        )

        self.assertAlmostEqual(float(scores["objective_score"].iloc[0]), 0.0)

    def test_medoids_are_real_biological_rows(self):
        features = fit_features()
        biological = pd.DataFrame(
            {
                feature: np.concatenate(
                    (
                        np.linspace(-2.0, -1.0, 6),
                        np.linspace(1.0, 2.0, 6),
                    )
                )
                for feature in features
            }
        )
        biological["dataset"] = "biological"
        biological["cell_id"] = [f"cell_{index}" for index in range(12)]
        normalization = biological_normalization(biological, features)

        targets = select_biological_medoids(
            biological,
            n_targets=2,
            normalization=normalization,
            seed=4,
        )

        self.assertEqual(len(targets), 2)
        self.assertTrue(set(targets["cell_id"]).issubset(biological["cell_id"]))
        self.assertEqual(int(targets["cluster_size"].sum()), len(biological))

    def test_evolution_candidates_respect_bounds(self):
        lower = np.full(len(ALL_PARAMETER_NAMES), -1.0)
        upper = np.full(len(ALL_PARAMETER_NAMES), 1.0)
        elites = pd.DataFrame(
            np.linspace(-0.5, 0.5, 5 * len(ALL_PARAMETER_NAMES)).reshape(
                5,
                len(ALL_PARAMETER_NAMES),
            ),
            columns=ALL_PARAMETER_NAMES,
        )
        candidates = propose_evolution_batch(
            elites,
            lower,
            upper,
            batch_size=20,
            rng=np.random.default_rng(9),
            round_index=0,
            config=DirectOptimizationConfig(),
        )

        self.assertEqual(candidates.shape, (20, len(ALL_PARAMETER_NAMES)))
        self.assertTrue((candidates.to_numpy() >= lower).all())
        self.assertTrue((candidates.to_numpy() <= upper).all())

    def test_long_scores_select_multifidelity_elites(self):
        fast_pool = pd.DataFrame(
            {
                "model": ["fast_apparent_best"],
                "objective_score": [0.01],
            }
        )
        long_pool = pd.DataFrame(
            {
                "model": ["long_best", "long_second"],
                "objective_score": [0.20, 0.30],
            }
        )

        elites = _select_multifidelity_elites(
            long_pool,
            fast_pool,
            elite_count=1,
        )

        self.assertEqual(elites["model"].tolist(), ["long_best"])


if __name__ == "__main__":
    unittest.main()
