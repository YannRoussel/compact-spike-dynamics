from __future__ import annotations

import unittest

import numpy as np

from inverse_ephys_alpha_beta.reduced_rank import (
    fit_class_residualized_rrr_analysis,
    fit_rrr_analysis,
    nested_class_residualized_rrr_evaluation,
    predict_class_residualized_rrr,
    residualize_class_fold,
)


class ReducedRankTests(unittest.TestCase):
    def test_recovers_sparse_rank_two_signal_with_grouped_cv(self):
        random = np.random.default_rng(12)
        cell_count = 100
        predictor_count = 20
        response_count = 8
        x = random.normal(
            size=(cell_count, predictor_count)
        )
        predictor_weights = np.zeros((predictor_count, 2))
        predictor_weights[:4] = random.normal(size=(4, 2))
        response_weights = random.normal(
            size=(response_count, 2)
        )
        y = (
            x @ predictor_weights @ response_weights.T
            + 0.25 * random.normal(size=(cell_count, response_count))
        )
        groups = np.repeat(np.arange(20), 5)

        result = fit_rrr_analysis(
            x,
            y,
            groups,
            ranks=(1, 2, 3),
            penalties=(0.01, 0.1, 1.0),
            sparse_ratios=(0.02, 0.1, 0.4),
            fold_count=4,
        )

        self.assertEqual(result.rank, 2)
        self.assertGreater(result.oof_r2, 0.9)
        self.assertGreaterEqual(
            np.sum(result.selected_predictors[:4]),
            3,
        )
        self.assertTrue(np.isfinite(result.oof_prediction).all())

    def test_class_residualization_uses_training_means(self):
        training = np.asarray([[1.0], [3.0], [10.0], [14.0]])
        test = np.asarray([[5.0], [18.0], [99.0]])
        train_classes = np.asarray(["a", "a", "b", "b"])
        test_classes = np.asarray(["a", "b", "unseen"])

        train_residual, test_residual, baseline = residualize_class_fold(
            training,
            test,
            train_classes,
            test_classes,
        )

        np.testing.assert_allclose(
            train_residual[:, 0],
            (-1.0, 1.0, -2.0, 2.0),
        )
        np.testing.assert_allclose(baseline[:, 0], (2.0, 12.0, 7.0))
        np.testing.assert_allclose(test_residual[:, 0], (3.0, 6.0, 92.0))

    def test_class_residualized_rrr_detects_within_class_signal(self):
        random = np.random.default_rng(19)
        class_count = 4
        cells_per_class = 30
        cell_count = class_count * cells_per_class
        classes = np.repeat(
            [f"class_{index}" for index in range(class_count)],
            cells_per_class,
        )
        groups = np.tile(np.repeat(np.arange(10), 3), class_count)
        x = random.normal(size=(cell_count, 16))
        class_offsets = random.normal(size=(class_count, 6)) * 4.0
        coefficient = np.zeros((16, 6))
        coefficient[:3] = random.normal(size=(3, 6))
        y = (
            class_offsets[
                np.repeat(np.arange(class_count), cells_per_class)
            ]
            + x @ coefficient
            + 0.15 * random.normal(size=(cell_count, 6))
        )

        result = fit_class_residualized_rrr_analysis(
            x,
            y,
            groups,
            classes,
            ranks=(1, 2, 3, 5),
            penalties=(0.01, 0.1, 1.0),
            sparse_ratios=(0.02, 0.1, 0.4),
            fold_count=5,
        )

        self.assertGreater(result.class_only_oof_r2, 0.5)
        self.assertGreater(result.incremental_residual_r2, 0.8)
        self.assertGreater(result.combined_oof_r2, result.class_only_oof_r2)
        self.assertTrue(np.isfinite(result.oof_prediction).all())

        prediction, class_only = predict_class_residualized_rrr(
            result,
            x[:-4],
            y[:-4],
            classes[:-4],
            x[-4:],
            classes[-4:],
        )
        self.assertEqual(prediction.shape, (4, 6))
        self.assertEqual(class_only.shape, (4, 6))
        self.assertTrue(np.isfinite(prediction).all())

        nested = nested_class_residualized_rrr_evaluation(
            x,
            y,
            groups,
            classes,
            ranks=(1, 2, 3),
            penalties=(0.1, 1.0),
            sparse_ratios=(0.02, 0.1),
            outer_fold_count=3,
            inner_fold_count=3,
        )
        self.assertGreater(nested.incremental_residual_r2, 0.7)
        self.assertEqual(len(nested.fold_hyperparameters), 3)
        self.assertTrue(np.isfinite(nested.oof_prediction).all())


if __name__ == "__main__":
    unittest.main()
