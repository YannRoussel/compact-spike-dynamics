"""Donor-aware ridge and sparse reduced-rank regression."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class RRRFactors:
    predictor_weights: np.ndarray
    response_weights: np.ndarray


@dataclass(frozen=True)
class RRRAnalysis:
    factors: RRRFactors
    selected_predictors: np.ndarray
    rank: int
    ridge_penalty: float
    sparse_ratio: float
    l1_ratio: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    ridge_cv: pd.DataFrame
    sparse_cv: pd.DataFrame
    oof_prediction: np.ndarray
    oof_r2: float


@dataclass(frozen=True)
class ClassResidualizedRRRAnalysis:
    """RRR fit after removing training-fold transcriptomic class means."""

    factors: RRRFactors
    selected_predictors: np.ndarray
    rank: int
    ridge_penalty: float
    sparse_ratio: float
    l1_ratio: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    ridge_cv: pd.DataFrame
    sparse_cv: pd.DataFrame
    oof_prediction: np.ndarray
    class_only_oof_prediction: np.ndarray
    combined_oof_r2: float
    class_only_oof_r2: float
    delta_oof_r2: float
    incremental_residual_r2: float


@dataclass(frozen=True)
class NestedClassResidualizedEvaluation:
    """Outer-fold predictions with all RRR choices made in inner folds."""

    oof_prediction: np.ndarray
    class_only_oof_prediction: np.ndarray
    fold_hyperparameters: pd.DataFrame
    combined_oof_r2: float
    class_only_oof_r2: float
    delta_oof_r2: float
    incremental_residual_r2: float


def _orient_factors(
    predictor_weights: np.ndarray,
    response_weights: np.ndarray,
) -> RRRFactors:
    predictor = np.asarray(predictor_weights, dtype=float).copy()
    response = np.asarray(response_weights, dtype=float).copy()
    for component in range(response.shape[1]):
        pivot = int(np.argmax(np.abs(response[:, component])))
        sign = np.sign(response[pivot, component])
        if sign == 0.0:
            sign = 1.0
        predictor[:, component] *= sign
        response[:, component] *= sign
    return RRRFactors(predictor, response)


def ridge_rrr(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    penalty: float,
) -> RRRFactors:
    """Fit ridge RRR to centered/scaled matrices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    maximum_rank = min(x.shape[0], x.shape[1], y.shape[1])
    if rank < 1 or rank > maximum_rank:
        raise ValueError("Invalid reduced rank")
    u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    return _ridge_rrr_from_svd(
        x,
        y,
        rank,
        penalty,
        u,
        singular_values,
        vt,
    )


def _ridge_rrr_from_svd(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    penalty: float,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> RRRFactors:
    shrinkage = singular_values / (
        singular_values**2 + float(penalty) * x.shape[0]
    )
    full_coefficient = (
        vt.T
        @ (shrinkage[:, None] * (u.T @ y))
    )
    _, _, response_vt = np.linalg.svd(
        x @ full_coefficient,
        full_matrices=False,
    )
    response_weights = response_vt[:rank].T
    predictor_weights = full_coefficient @ response_weights
    return _orient_factors(predictor_weights, response_weights)


def _initial_response_weights(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
) -> np.ndarray:
    _, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return vt[:rank].T


def sparse_penalty_max(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    l1_ratio: float,
) -> float:
    """Return the smallest sklearn alpha yielding the all-zero model."""
    response_weights = _initial_response_weights(x, y, rank)
    projected = y @ response_weights
    row_norm = np.linalg.norm(x.T @ projected, axis=1)
    return float(
        np.max(row_norm)
        / (x.shape[0] * max(float(l1_ratio), 1e-8))
    )


def sparse_rrr(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    alpha: float,
    l1_ratio: float = 0.5,
    maximum_iterations: int = 30,
    tolerance: float = 1e-5,
) -> RRRFactors:
    """Fit row-sparse elastic-net RRR by alternating minimization."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    response_weights = _initial_response_weights(x, y, rank)
    previous_loss = float("inf")
    predictor_weights = np.zeros((x.shape[1], rank), dtype=float)
    for _ in range(maximum_iterations):
        regression = MultiTaskElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=False,
            max_iter=5000,
            tol=1e-4,
            selection="cyclic",
        )
        regression.fit(x, y @ response_weights)
        predictor_weights = regression.coef_.T
        if not np.any(predictor_weights):
            return RRRFactors(
                predictor_weights,
                np.zeros_like(response_weights),
            )
        left, _, right = np.linalg.svd(
            y.T @ x @ predictor_weights,
            full_matrices=False,
        )
        response_weights = left @ right
        residual = y - x @ predictor_weights @ response_weights.T
        loss = float(np.sum(residual**2) / np.sum(y**2))
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss
    return _orient_factors(predictor_weights, response_weights)


def _scaling(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def _scale_with_training(
    training: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = _scaling(training)
    return (
        np.clip((training - mean) / scale, -8.0, 8.0),
        np.clip((test - mean) / scale, -8.0, 8.0),
        mean,
        scale,
    )


def _class_baseline(
    training: np.ndarray,
    training_classes: Sequence[object],
    requested_classes: Sequence[object],
) -> np.ndarray:
    """Predict class means, falling back to the global training mean."""
    values = np.asarray(training, dtype=float)
    train_labels = np.asarray(training_classes, dtype=object)
    requested = np.asarray(requested_classes, dtype=object)
    global_mean = np.mean(values, axis=0)
    means = {
        label: np.mean(values[train_labels == label], axis=0)
        for label in np.unique(train_labels)
    }
    return np.asarray(
        [means.get(label, global_mean) for label in requested],
        dtype=float,
    )


def residualize_class_fold(
    training: np.ndarray,
    test: np.ndarray,
    training_classes: Sequence[object],
    test_classes: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove class means estimated exclusively from the training fold."""
    training_baseline = _class_baseline(
        training,
        training_classes,
        training_classes,
    )
    test_baseline = _class_baseline(
        training,
        training_classes,
        test_classes,
    )
    return (
        np.asarray(training, dtype=float) - training_baseline,
        np.asarray(test, dtype=float) - test_baseline,
        test_baseline,
    )


def _folds(
    groups: Sequence[object],
    fold_count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=object)
    groups = np.asarray(
        [
            f"missing_{index}" if pd.isna(value) else str(value)
            for index, value in enumerate(groups)
        ]
    )
    splits = min(int(fold_count), len(np.unique(groups)))
    if splits < 2:
        raise ValueError("At least two donor groups are required")
    return list(
        GroupKFold(n_splits=splits).split(
            np.zeros(len(groups)),
            groups=groups,
        )
    )


def _r2(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
    denominator = float(np.sum(y_true**2))
    if denominator <= 0.0:
        return float("nan")
    return float(
        1.0 - np.sum((y_true - y_prediction) ** 2) / denominator
    )


def cross_validate_ridge_rrr(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    ranks: Sequence[int] = (1, 2, 3, 5, 8),
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    fold_count: int = 5,
) -> pd.DataFrame:
    """Evaluate ridge RRR hyperparameters with donor-held-out folds."""
    folds = _folds(groups, fold_count)
    rows = []
    maximum_rank = min(
        x.shape[1],
        y.shape[1],
        min(len(training) for training, _ in folds),
    )
    valid_ranks = [rank for rank in ranks if rank <= maximum_rank]
    for fold_index, (training, test) in enumerate(folds):
        x_train, x_test, _, _ = _scale_with_training(
            x[training],
            x[test],
        )
        y_train, y_test, _, _ = _scale_with_training(
            y[training],
            y[test],
        )
        u, singular_values, vt = np.linalg.svd(
            x_train,
            full_matrices=False,
        )
        for penalty in penalties:
            for rank in valid_ranks:
                factors = _ridge_rrr_from_svd(
                    x_train,
                    y_train,
                    rank,
                    penalty,
                    u,
                    singular_values,
                    vt,
                )
                prediction = (
                    x_test
                    @ factors.predictor_weights
                    @ factors.response_weights.T
                )
                rows.append(
                    {
                        "fold": fold_index,
                        "rank": rank,
                        "ridge_penalty": penalty,
                        "test_r2": _r2(y_test, prediction),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["rank", "ridge_penalty"], as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            sd_test_r2=("test_r2", "std"),
            fold_count=("test_r2", "size"),
        )
    )
    best = summary["mean_test_r2"].idxmax()
    summary["selected"] = False
    summary.loc[best, "selected"] = True
    return summary


def cross_validate_class_residualized_ridge_rrr(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    classes: Sequence[object],
    ranks: Sequence[int] = (1, 2, 3, 5, 8),
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    fold_count: int = 5,
) -> pd.DataFrame:
    """Tune ridge RRR using class residuals learned within each fold."""
    folds = _folds(groups, fold_count)
    labels = np.asarray(classes, dtype=object)
    rows = []
    maximum_rank = min(
        x.shape[1],
        y.shape[1],
        min(len(training) for training, _ in folds),
    )
    valid_ranks = [rank for rank in ranks if rank <= maximum_rank]
    for fold_index, (training, test) in enumerate(folds):
        x_train_raw, x_test_raw, _ = residualize_class_fold(
            x[training],
            x[test],
            labels[training],
            labels[test],
        )
        y_train_raw, y_test_raw, _ = residualize_class_fold(
            y[training],
            y[test],
            labels[training],
            labels[test],
        )
        x_train, x_test, _, _ = _scale_with_training(
            x_train_raw,
            x_test_raw,
        )
        y_train, y_test, _, _ = _scale_with_training(
            y_train_raw,
            y_test_raw,
        )
        u, singular_values, vt = np.linalg.svd(
            x_train,
            full_matrices=False,
        )
        for penalty in penalties:
            for rank in valid_ranks:
                factors = _ridge_rrr_from_svd(
                    x_train,
                    y_train,
                    rank,
                    penalty,
                    u,
                    singular_values,
                    vt,
                )
                prediction = (
                    x_test
                    @ factors.predictor_weights
                    @ factors.response_weights.T
                )
                rows.append(
                    {
                        "fold": fold_index,
                        "rank": rank,
                        "ridge_penalty": penalty,
                        "test_r2": _r2(y_test, prediction),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["rank", "ridge_penalty"], as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            sd_test_r2=("test_r2", "std"),
            fold_count=("test_r2", "size"),
        )
    )
    best = summary["mean_test_r2"].idxmax()
    summary["selected"] = False
    summary.loc[best, "selected"] = True
    return summary


def _relaxed_sparse_factors(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    ridge_penalty: float,
    sparse_ratio: float,
    l1_ratio: float,
) -> tuple[RRRFactors, np.ndarray]:
    maximum = sparse_penalty_max(x, y, rank, l1_ratio)
    sparse = sparse_rrr(
        x,
        y,
        rank,
        alpha=float(sparse_ratio) * maximum,
        l1_ratio=l1_ratio,
    )
    selected = (
        np.linalg.norm(sparse.predictor_weights, axis=1) > 1e-10
    )
    if np.sum(selected) < rank:
        empty = RRRFactors(
            np.zeros((x.shape[1], rank), dtype=float),
            np.zeros((y.shape[1], rank), dtype=float),
        )
        return empty, selected
    relaxed_subset = ridge_rrr(
        x[:, selected],
        y,
        rank,
        ridge_penalty,
    )
    predictor_weights = np.zeros(
        (x.shape[1], rank),
        dtype=float,
    )
    predictor_weights[selected] = relaxed_subset.predictor_weights
    return (
        RRRFactors(
            predictor_weights,
            relaxed_subset.response_weights,
        ),
        selected,
    )


def cross_validate_sparse_rrr(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    rank: int,
    ridge_penalty: float,
    sparse_ratios: Sequence[float] = (
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
    ),
    l1_ratio: float = 0.5,
    fold_count: int = 5,
) -> pd.DataFrame:
    """Tune sparse gene selection and score its relaxed RRR refit."""
    rows = []
    for fold_index, (training, test) in enumerate(
        _folds(groups, fold_count)
    ):
        x_train, x_test, _, _ = _scale_with_training(
            x[training],
            x[test],
        )
        y_train, y_test, _, _ = _scale_with_training(
            y[training],
            y[test],
        )
        for ratio in sparse_ratios:
            factors, selected = _relaxed_sparse_factors(
                x_train,
                y_train,
                rank,
                ridge_penalty,
                ratio,
                l1_ratio,
            )
            prediction = (
                x_test
                @ factors.predictor_weights
                @ factors.response_weights.T
            )
            rows.append(
                {
                    "fold": fold_index,
                    "sparse_ratio": ratio,
                    "selected_gene_count": int(np.sum(selected)),
                    "test_r2": _r2(y_test, prediction),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("sparse_ratio", as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            sd_test_r2=("test_r2", "std"),
            mean_selected_gene_count=("selected_gene_count", "mean"),
            fold_count=("test_r2", "size"),
        )
        .sort_values("sparse_ratio")
        .reset_index(drop=True)
    )
    best_index = summary["mean_test_r2"].idxmax()
    best = summary.loc[best_index]
    standard_error = (
        float(best["sd_test_r2"])
        / math.sqrt(float(best["fold_count"]))
        if np.isfinite(best["sd_test_r2"])
        else 0.0
    )
    threshold = float(best["mean_test_r2"]) - standard_error
    eligible = summary.loc[summary["mean_test_r2"] >= threshold]
    selected_index = eligible["sparse_ratio"].idxmax()
    summary["best_mean"] = False
    summary.loc[best_index, "best_mean"] = True
    summary["selected_one_se"] = False
    summary.loc[selected_index, "selected_one_se"] = True
    return summary


def cross_validate_class_residualized_sparse_rrr(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    classes: Sequence[object],
    rank: int,
    ridge_penalty: float,
    sparse_ratios: Sequence[float] = (
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
    ),
    l1_ratio: float = 0.5,
    fold_count: int = 5,
) -> pd.DataFrame:
    """Tune sparse RRR on leakage-safe within-class residuals."""
    rows = []
    labels = np.asarray(classes, dtype=object)
    for fold_index, (training, test) in enumerate(
        _folds(groups, fold_count)
    ):
        x_train_raw, x_test_raw, _ = residualize_class_fold(
            x[training],
            x[test],
            labels[training],
            labels[test],
        )
        y_train_raw, y_test_raw, _ = residualize_class_fold(
            y[training],
            y[test],
            labels[training],
            labels[test],
        )
        x_train, x_test, _, _ = _scale_with_training(
            x_train_raw,
            x_test_raw,
        )
        y_train, y_test, _, _ = _scale_with_training(
            y_train_raw,
            y_test_raw,
        )
        for ratio in sparse_ratios:
            factors, selected = _relaxed_sparse_factors(
                x_train,
                y_train,
                rank,
                ridge_penalty,
                ratio,
                l1_ratio,
            )
            prediction = (
                x_test
                @ factors.predictor_weights
                @ factors.response_weights.T
            )
            rows.append(
                {
                    "fold": fold_index,
                    "sparse_ratio": ratio,
                    "selected_gene_count": int(np.sum(selected)),
                    "test_r2": _r2(y_test, prediction),
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("sparse_ratio", as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            sd_test_r2=("test_r2", "std"),
            mean_selected_gene_count=("selected_gene_count", "mean"),
            fold_count=("test_r2", "size"),
        )
        .sort_values("sparse_ratio")
        .reset_index(drop=True)
    )
    best_index = summary["mean_test_r2"].idxmax()
    best = summary.loc[best_index]
    standard_error = (
        float(best["sd_test_r2"])
        / math.sqrt(float(best["fold_count"]))
        if np.isfinite(best["sd_test_r2"])
        else 0.0
    )
    threshold = float(best["mean_test_r2"]) - standard_error
    eligible = summary.loc[summary["mean_test_r2"] >= threshold]
    selected_index = eligible["sparse_ratio"].idxmax()
    summary["best_mean"] = False
    summary.loc[best_index, "best_mean"] = True
    summary["selected_one_se"] = False
    summary.loc[selected_index, "selected_one_se"] = True
    return summary


def fit_rrr_analysis(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    ranks: Sequence[int] = (1, 2, 3, 5, 8),
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    sparse_ratios: Sequence[float] = (
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
    ),
    l1_ratio: float = 0.5,
    fold_count: int = 5,
) -> RRRAnalysis:
    """Tune and fit a relaxed sparse RRR analysis."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ridge_cv = cross_validate_ridge_rrr(
        x,
        y,
        groups,
        ranks=ranks,
        penalties=penalties,
        fold_count=fold_count,
    )
    ridge_selected = ridge_cv.loc[ridge_cv["selected"]].iloc[0]
    rank = int(ridge_selected["rank"])
    ridge_penalty = float(ridge_selected["ridge_penalty"])
    sparse_cv = cross_validate_sparse_rrr(
        x,
        y,
        groups,
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratios=sparse_ratios,
        l1_ratio=l1_ratio,
        fold_count=fold_count,
    )
    sparse_ratio = float(
        sparse_cv.loc[sparse_cv["selected_one_se"], "sparse_ratio"].iloc[0]
    )

    x_mean, x_scale = _scaling(x)
    y_mean, y_scale = _scaling(y)
    x_scaled = np.clip((x - x_mean) / x_scale, -8.0, 8.0)
    y_scaled = np.clip((y - y_mean) / y_scale, -8.0, 8.0)
    factors, selected = _relaxed_sparse_factors(
        x_scaled,
        y_scaled,
        rank,
        ridge_penalty,
        sparse_ratio,
        l1_ratio,
    )

    oof_prediction = np.full_like(y, np.nan, dtype=float)
    fold_scores = []
    for training, test in _folds(groups, fold_count):
        x_train, x_test, _, _ = _scale_with_training(
            x[training],
            x[test],
        )
        y_train, y_test, y_fold_mean, y_fold_scale = (
            _scale_with_training(y[training], y[test])
        )
        fold_factors, _ = _relaxed_sparse_factors(
            x_train,
            y_train,
            rank,
            ridge_penalty,
            sparse_ratio,
            l1_ratio,
        )
        prediction_scaled = (
            x_test
            @ fold_factors.predictor_weights
            @ fold_factors.response_weights.T
        )
        fold_scores.append(_r2(y_test, prediction_scaled))
        oof_prediction[test] = (
            prediction_scaled * y_fold_scale + y_fold_mean
        )
    return RRRAnalysis(
        factors=factors,
        selected_predictors=selected,
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratio=sparse_ratio,
        l1_ratio=l1_ratio,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        ridge_cv=ridge_cv,
        sparse_cv=sparse_cv,
        oof_prediction=oof_prediction,
        oof_r2=float(np.mean(fold_scores)),
    )


def _standardized_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    mean, scale = _scaling(actual)
    return _r2(
        (actual - mean) / scale,
        (predicted - mean) / scale,
    )


def fit_class_residualized_rrr_analysis(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    classes: Sequence[object],
    ranks: Sequence[int] = (1, 2, 3, 5, 8),
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    sparse_ratios: Sequence[float] = (
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
    ),
    l1_ratio: float = 0.5,
    fold_count: int = 5,
) -> ClassResidualizedRRRAnalysis:
    """Fit RRR to variation remaining within broad transcriptomic classes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(classes, dtype=object)
    if len(labels) != len(x):
        raise ValueError("Class labels must match the number of cells")
    ridge_cv = cross_validate_class_residualized_ridge_rrr(
        x,
        y,
        groups,
        labels,
        ranks=ranks,
        penalties=penalties,
        fold_count=fold_count,
    )
    ridge_selected = ridge_cv.loc[ridge_cv["selected"]].iloc[0]
    rank = int(ridge_selected["rank"])
    ridge_penalty = float(ridge_selected["ridge_penalty"])
    sparse_cv = cross_validate_class_residualized_sparse_rrr(
        x,
        y,
        groups,
        labels,
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratios=sparse_ratios,
        l1_ratio=l1_ratio,
        fold_count=fold_count,
    )
    sparse_ratio = float(
        sparse_cv.loc[sparse_cv["selected_one_se"], "sparse_ratio"].iloc[0]
    )

    x_residual, _, _ = residualize_class_fold(
        x,
        x,
        labels,
        labels,
    )
    y_residual, _, _ = residualize_class_fold(
        y,
        y,
        labels,
        labels,
    )
    x_mean, x_scale = _scaling(x_residual)
    y_mean, y_scale = _scaling(y_residual)
    x_scaled = np.clip(
        (x_residual - x_mean) / x_scale,
        -8.0,
        8.0,
    )
    y_scaled = np.clip(
        (y_residual - y_mean) / y_scale,
        -8.0,
        8.0,
    )
    factors, selected = _relaxed_sparse_factors(
        x_scaled,
        y_scaled,
        rank,
        ridge_penalty,
        sparse_ratio,
        l1_ratio,
    )

    oof_prediction = np.full_like(y, np.nan, dtype=float)
    class_only = np.full_like(y, np.nan, dtype=float)
    for training, test in _folds(groups, fold_count):
        x_train_raw, x_test_raw, _ = residualize_class_fold(
            x[training],
            x[test],
            labels[training],
            labels[test],
        )
        y_train_raw, _, y_test_baseline = residualize_class_fold(
            y[training],
            y[test],
            labels[training],
            labels[test],
        )
        x_train, x_test, _, _ = _scale_with_training(
            x_train_raw,
            x_test_raw,
        )
        y_train, _, y_fold_mean, y_fold_scale = _scale_with_training(
            y_train_raw,
            y_train_raw,
        )
        fold_factors, _ = _relaxed_sparse_factors(
            x_train,
            y_train,
            rank,
            ridge_penalty,
            sparse_ratio,
            l1_ratio,
        )
        residual_prediction = (
            x_test
            @ fold_factors.predictor_weights
            @ fold_factors.response_weights.T
        )
        oof_prediction[test] = (
            residual_prediction * y_fold_scale
            + y_fold_mean
            + y_test_baseline
        )
        class_only[test] = y_test_baseline

    combined_r2 = _standardized_r2(y, oof_prediction)
    class_only_r2 = _standardized_r2(y, class_only)
    _, total_scale = _scaling(y)
    combined_error = np.sum(((y - oof_prediction) / total_scale) ** 2)
    class_error = np.sum(((y - class_only) / total_scale) ** 2)
    incremental = (
        float(1.0 - combined_error / class_error)
        if class_error > 0.0
        else float("nan")
    )
    return ClassResidualizedRRRAnalysis(
        factors=factors,
        selected_predictors=selected,
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratio=sparse_ratio,
        l1_ratio=l1_ratio,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        ridge_cv=ridge_cv,
        sparse_cv=sparse_cv,
        oof_prediction=oof_prediction,
        class_only_oof_prediction=class_only,
        combined_oof_r2=combined_r2,
        class_only_oof_r2=class_only_r2,
        delta_oof_r2=combined_r2 - class_only_r2,
        incremental_residual_r2=incremental,
    )


def predict_class_residualized_rrr(
    analysis: ClassResidualizedRRRAnalysis,
    x_training: np.ndarray,
    y_training: np.ndarray,
    training_classes: Sequence[object],
    x_new: np.ndarray,
    new_classes: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict held-out responses using training-only class baselines."""
    x_training = np.asarray(x_training, dtype=float)
    y_training = np.asarray(y_training, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    labels = np.asarray(training_classes, dtype=object)
    requested = np.asarray(new_classes, dtype=object)
    _, x_residual, _ = residualize_class_fold(
        x_training,
        x_new,
        labels,
        requested,
    )
    _, _, class_baseline = residualize_class_fold(
        y_training,
        np.zeros((len(x_new), y_training.shape[1]), dtype=float),
        labels,
        requested,
    )
    x_scaled = np.clip(
        (x_residual - analysis.x_mean) / analysis.x_scale,
        -8.0,
        8.0,
    )
    residual_prediction = (
        x_scaled
        @ analysis.factors.predictor_weights
        @ analysis.factors.response_weights.T
    )
    combined = (
        residual_prediction * analysis.y_scale
        + analysis.y_mean
        + class_baseline
    )
    return combined, class_baseline


def nested_class_residualized_rrr_evaluation(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    classes: Sequence[object],
    ranks: Sequence[int] = (1, 2, 3, 5, 8),
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    sparse_ratios: Sequence[float] = (
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
    ),
    l1_ratio: float = 0.5,
    outer_fold_count: int = 5,
    inner_fold_count: int = 4,
) -> NestedClassResidualizedEvaluation:
    """Estimate class-plus-RNA performance with donor-grouped nested CV."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_values = np.asarray(groups, dtype=object)
    labels = np.asarray(classes, dtype=object)
    oof_prediction = np.full_like(y, np.nan, dtype=float)
    class_only = np.full_like(y, np.nan, dtype=float)
    hyperparameter_rows = []
    for outer_fold, (training, test) in enumerate(
        _folds(group_values, outer_fold_count)
    ):
        inner_groups = group_values[training]
        inner_folds = min(
            int(inner_fold_count),
            len(
                {
                    str(value)
                    for value in inner_groups
                    if not pd.isna(value)
                }
            ),
        )
        if inner_folds < 2:
            raise ValueError("Nested CV requires two inner donor groups")
        ridge_cv = cross_validate_class_residualized_ridge_rrr(
            x[training],
            y[training],
            inner_groups,
            labels[training],
            ranks=ranks,
            penalties=penalties,
            fold_count=inner_folds,
        )
        ridge_selected = ridge_cv.loc[ridge_cv["selected"]].iloc[0]
        rank = int(ridge_selected["rank"])
        ridge_penalty = float(ridge_selected["ridge_penalty"])
        if sparse_ratios:
            sparse_cv = cross_validate_class_residualized_sparse_rrr(
                x[training],
                y[training],
                inner_groups,
                labels[training],
                rank=rank,
                ridge_penalty=ridge_penalty,
                sparse_ratios=sparse_ratios,
                l1_ratio=l1_ratio,
                fold_count=inner_folds,
            )
            sparse_selected = sparse_cv.loc[
                sparse_cv["selected_one_se"]
            ].iloc[0]
            sparse_ratio = float(sparse_selected["sparse_ratio"])
            inner_residual_r2 = float(
                sparse_selected["mean_test_r2"]
            )
        else:
            sparse_ratio = float("nan")
            inner_residual_r2 = float(
                ridge_selected["mean_test_r2"]
            )

        x_train_raw, x_test_raw, _ = residualize_class_fold(
            x[training],
            x[test],
            labels[training],
            labels[test],
        )
        y_train_raw, _, y_test_baseline = residualize_class_fold(
            y[training],
            y[test],
            labels[training],
            labels[test],
        )
        x_train, x_test, _, _ = _scale_with_training(
            x_train_raw,
            x_test_raw,
        )
        y_train, _, y_fold_mean, y_fold_scale = _scale_with_training(
            y_train_raw,
            y_train_raw,
        )
        if sparse_ratios:
            factors, selected = _relaxed_sparse_factors(
                x_train,
                y_train,
                rank,
                ridge_penalty,
                sparse_ratio,
                l1_ratio,
            )
        else:
            factors = ridge_rrr(
                x_train,
                y_train,
                rank,
                ridge_penalty,
            )
            selected = np.ones(x.shape[1], dtype=bool)
        residual_prediction = (
            x_test
            @ factors.predictor_weights
            @ factors.response_weights.T
        )
        oof_prediction[test] = (
            residual_prediction * y_fold_scale
            + y_fold_mean
            + y_test_baseline
        )
        class_only[test] = y_test_baseline
        hyperparameter_rows.append(
            {
                "outer_fold": outer_fold,
                "training_cells": len(training),
                "test_cells": len(test),
                "rank": rank,
                "ridge_penalty": ridge_penalty,
                "sparse_ratio": sparse_ratio,
                "selected_gene_count": int(np.sum(selected)),
                "inner_residual_r2": inner_residual_r2,
            }
        )

    combined_r2 = _standardized_r2(y, oof_prediction)
    class_only_r2 = _standardized_r2(y, class_only)
    _, total_scale = _scaling(y)
    combined_error = np.sum(((y - oof_prediction) / total_scale) ** 2)
    class_error = np.sum(((y - class_only) / total_scale) ** 2)
    incremental = (
        float(1.0 - combined_error / class_error)
        if class_error > 0.0
        else float("nan")
    )
    return NestedClassResidualizedEvaluation(
        oof_prediction=oof_prediction,
        class_only_oof_prediction=class_only,
        fold_hyperparameters=pd.DataFrame(hyperparameter_rows),
        combined_oof_r2=combined_r2,
        class_only_oof_r2=class_only_r2,
        delta_oof_r2=combined_r2 - class_only_r2,
        incremental_residual_r2=incremental,
    )


def correlation_loadings(
    values: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Correlate each variable with each latent component."""
    values = np.asarray(values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    values = (values - np.mean(values, axis=0)) / np.maximum(
        np.std(values, axis=0),
        1e-8,
    )
    scores = (scores - np.mean(scores, axis=0)) / np.maximum(
        np.std(scores, axis=0),
        1e-8,
    )
    return values.T @ scores / values.shape[0]
