"""Transform compact-model tables into stable multivariate RRR targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compact_model import compact_parameter_columns


@dataclass(frozen=True)
class ModelTargetMatrix:
    values: np.ndarray
    names: tuple[str, ...]
    log1p_transformed: tuple[bool, ...]


def _use_log1p(name: str, values: np.ndarray) -> bool:
    if np.min(values) < 0.0:
        return False
    return (
        name.endswith("_pa")
        or name == "timing_tau_ms"
        or name == "recovery_tau_ms"
        or name.startswith("recovery_baseline_period_")
        or name.startswith("recovery_minimum_period_")
        or name.startswith("recovery_maximum_period_")
        or name.startswith("onset_tau_ms_")
        or name.startswith("baseline_period_")
        or name.startswith("latency_ms_")
        or name.startswith("downstroke_duration_ms_")
        or name.startswith("upstroke_duration_ms_")
    )


def build_model_target_matrix(
    parameters: pd.DataFrame,
) -> ModelTargetMatrix:
    """Select nonredundant finite parameters and stabilize positive scales."""
    names = []
    columns = []
    transformed = []
    for name in compact_parameter_columns(parameters.columns):
        values = pd.to_numeric(parameters[name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            continue
        if np.std(values) <= 1e-10:
            continue
        use_log = _use_log1p(name, values)
        columns.append(np.log1p(values) if use_log else values)
        names.append(name)
        transformed.append(use_log)
    if not columns:
        raise ValueError("No finite varying compact-model parameters")
    return ModelTargetMatrix(
        values=np.column_stack(columns),
        names=tuple(names),
        log1p_transformed=tuple(transformed),
    )


def inverse_model_target_matrix(
    values: np.ndarray,
    target: ModelTargetMatrix,
) -> np.ndarray:
    """Undo target-wise log transforms."""
    result = np.asarray(values, dtype=float).copy()
    for index, transformed in enumerate(target.log1p_transformed):
        if transformed:
            result[:, index] = np.expm1(result[:, index])
    return result


def transform_model_target_matrix(
    parameters: pd.DataFrame,
    target: ModelTargetMatrix,
) -> np.ndarray:
    """Apply training-defined target columns and transforms to new cells."""
    columns = []
    for name, transformed in zip(
        target.names,
        target.log1p_transformed,
    ):
        values = pd.to_numeric(parameters[name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite held-out target values in {name}")
        columns.append(np.log1p(values) if transformed else values)
    return np.column_stack(columns)


def high_qc_mask(parameters: pd.DataFrame) -> np.ndarray:
    """Predeclared sensitivity cohort for interpolation and timing fidelity."""
    return np.asarray(
        parameters["validation_phase_chamfer"].le(0.01)
        & parameters["validation_relative_spike_count_error"].le(0.5)
        & parameters["timing_late_period_nrmse"].le(2.0),
        dtype=bool,
    )
