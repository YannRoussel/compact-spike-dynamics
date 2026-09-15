import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.coverage import (
    analyze_coverage,
    compare_feature_distributions,
)


class CoverageTests(unittest.TestCase):
    def test_identical_models_cover_local_biological_space(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(30, 3))
        biological = pd.DataFrame(values, columns=["a", "b", "c"])
        biological.insert(0, "cell_id", [f"bio_{i}" for i in range(len(values))])
        biological.insert(0, "dataset", "biological")
        models = biological.iloc[::3].copy()
        models["dataset"] = "hh_model"
        models["cell_id"] = [f"model_{i}" for i in range(len(models))]

        result = analyze_coverage(
            biological,
            models,
            features=("a", "b", "c"),
            local_neighbor_rank=3,
        )
        self.assertGreater(result.summary.iloc[-1]["coverage_fraction"], 0.0)
        self.assertTrue(result.model_distances["biologically_plausible"].all())
        self.assertEqual(result.biological_coordinates.shape[1], 4)
        self.assertEqual(
            set(result.pca_feature_loadings["feature"]),
            {"a", "b", "c"},
        )
        self.assertTrue(
            {
                "pc1_weight",
                "pc1_correlation",
                "pc2_weight",
                "pc2_correlation",
            }.issubset(result.pca_feature_loadings.columns)
        )

    def test_distribution_comparison_uses_biological_median_and_iqr(self):
        biological = pd.DataFrame(
            {
                "feature_a": np.arange(9, dtype=float),
                "feature_b": np.ones(9),
            }
        )
        models = pd.DataFrame(
            {
                "feature_a": np.arange(9, dtype=float) + 4.0,
                "feature_b": np.ones(9),
            }
        )

        result = compare_feature_distributions(
            biological,
            models,
            features=("feature_a", "feature_b"),
        )

        self.assertEqual(result.features, ("feature_a",))
        summary = result.summary.set_index("feature").loc["feature_a"]
        self.assertEqual(summary["biological_median"], 4.0)
        self.assertEqual(summary["biological_iqr"], 4.0)
        self.assertEqual(summary["model_median_shift_biological_iqr"], 1.0)
        self.assertGreater(summary["wasserstein_biological_iqr"], 0.0)
        biological_normalized = result.values.loc[
            result.values["cohort"].eq("Biological"),
            "normalized_value",
        ]
        self.assertAlmostEqual(float(biological_normalized.median()), 0.0)


if __name__ == "__main__":
    unittest.main()
