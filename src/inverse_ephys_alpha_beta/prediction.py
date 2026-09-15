"""Baseline multi-output prediction of kinetic parameters from e-features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def train_predictor(
    dataset: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.25,
    n_estimators: int = 400,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit an ExtraTrees baseline and return its bundle, metrics, and predictions."""
    if len(dataset) < 12:
        raise ValueError("At least 12 accepted spiking models are needed for a train/test split")

    feature_columns = [
        column
        for column in dataset.columns
        if column.startswith("feature__") and not dataset[column].isna().all()
    ]
    target_columns = [
        column for column in dataset.columns if column.startswith("param__")
    ]
    if not feature_columns or not target_columns:
        raise ValueError("Dataset does not contain the expected feature and parameter columns")

    x_data = dataset[feature_columns]
    y_data = dataset[target_columns]
    train_index, test_index = train_test_split(
        np.arange(len(dataset)),
        test_size=test_fraction,
        random_state=seed,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_data.iloc[train_index], y_data.iloc[train_index])
    predicted = model.predict(x_data.iloc[test_index])
    observed = y_data.iloc[test_index].to_numpy()

    metric_rows = []
    for index, parameter_name in enumerate(target_columns):
        metric_rows.append(
            {
                "parameter": parameter_name,
                "r2": r2_score(observed[:, index], predicted[:, index]),
                "mae": mean_absolute_error(observed[:, index], predicted[:, index]),
                "target_std": float(np.std(observed[:, index])),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    predictions = pd.DataFrame(
        {
            "sample_id": dataset.iloc[test_index]["sample_id"].to_numpy(),
            **{
                f"observed__{name}": observed[:, index]
                for index, name in enumerate(target_columns)
            },
            **{
                f"predicted__{name}": predicted[:, index]
                for index, name in enumerate(target_columns)
            },
        }
    )
    regressor = model.named_steps["regressor"]
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "feature_importance": feature_importance,
        "seed": seed,
        "test_fraction": test_fraction,
    }
    return bundle, metrics, predictions


def save_predictor(bundle: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
