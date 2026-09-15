"""Build a conservative next-stage prior from the closest biological matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kinetics import PARAMETER_NAMES, default_parameter_bounds
from .static_parameters import STATIC_PARAMETER_NAMES, default_static_parameter_bounds


@dataclass(frozen=True)
class CalibrationResult:
    parameter_summary: pd.DataFrame
    selected_models: pd.DataFrame
    bounds: dict[str, list[float]]


def _default_bounds() -> dict[str, tuple[float, float]]:
    kinetic_lower, kinetic_upper = default_parameter_bounds()
    static_lower, static_upper = default_static_parameter_bounds()
    names = PARAMETER_NAMES + STATIC_PARAMETER_NAMES
    lower = np.concatenate((kinetic_lower, static_lower))
    upper = np.concatenate((kinetic_upper, static_upper))
    return {
        name: (float(bound_lower), float(bound_upper))
        for name, bound_lower, bound_upper in zip(names, lower, upper)
    }


def expanded_kinetic_bounds(expansion_factor: float) -> dict[str, list[float]]:
    """Expand kinetic transformed-coordinate bounds around the canonical origin."""
    if expansion_factor < 1.0:
        raise ValueError("Kinetic expansion factor must be at least 1")
    lower, upper = default_parameter_bounds()
    return {
        parameter_name: [
            float(bound_lower * expansion_factor),
            float(bound_upper * expansion_factor),
        ]
        for parameter_name, bound_lower, bound_upper in zip(
            PARAMETER_NAMES, lower, upper
        )
    }


def calibrate_parameter_bounds(
    model_dataset: pd.DataFrame,
    model_plausibility: pd.DataFrame,
    parameter_group: str = "static",
    minimum_models: int = 5,
    minimum_prior_fraction: float = 0.35,
    padding_fraction: float = 0.5,
) -> CalibrationResult:
    """Propose padded bounds from plausible or nearest retained HH models."""
    if parameter_group == "static":
        candidate_names = STATIC_PARAMETER_NAMES
    elif parameter_group == "kinetic":
        candidate_names = PARAMETER_NAMES
    elif parameter_group == "all":
        candidate_names = PARAMETER_NAMES + STATIC_PARAMETER_NAMES
    else:
        raise ValueError(f"Unknown parameter group: {parameter_group}")

    distances = model_plausibility.copy()
    distances["sample_id"] = pd.to_numeric(distances["cell_id"], errors="coerce")
    distances = distances.dropna(subset=["sample_id"]).copy()
    distances["sample_id"] = distances["sample_id"].astype(int)
    ranked = distances.sort_values("nearest_biological_distance")
    plausible = ranked.loc[ranked["biologically_plausible"].astype(bool)]
    selected_ids = plausible["sample_id"].tolist()
    if len(selected_ids) < minimum_models:
        selected_ids = ranked.head(min(minimum_models, len(ranked)))["sample_id"].tolist()

    selected = model_dataset.loc[
        model_dataset["sample_id"].isin(selected_ids)
    ].copy()
    if selected.empty:
        raise ValueError("No calibration models could be matched by sample_id")

    defaults = _default_bounds()
    rows = []
    calibrated_bounds: dict[str, list[float]] = {}
    for parameter_name in candidate_names:
        if parameter_name not in selected:
            continue
        values = selected[parameter_name].dropna().to_numpy(dtype=float)
        if not len(values) or np.ptp(values) < 1e-12:
            continue
        default_lower, default_upper = defaults[parameter_name]
        default_width = default_upper - default_lower
        observed_lower, observed_upper = np.quantile(values, (0.1, 0.9))
        observed_width = float(observed_upper - observed_lower)
        proposed_width = max(
            observed_width * (1.0 + 2.0 * padding_fraction),
            default_width * minimum_prior_fraction,
        )
        center = float(np.median(values))
        proposed_lower = max(default_lower, center - proposed_width / 2.0)
        proposed_upper = min(default_upper, center + proposed_width / 2.0)
        if proposed_upper - proposed_lower < default_width * minimum_prior_fraction:
            if proposed_lower == default_lower:
                proposed_upper = min(
                    default_upper,
                    default_lower + default_width * minimum_prior_fraction,
                )
            elif proposed_upper == default_upper:
                proposed_lower = max(
                    default_lower,
                    default_upper - default_width * minimum_prior_fraction,
                )
        calibrated_bounds[parameter_name] = [
            float(proposed_lower),
            float(proposed_upper),
        ]
        rows.append(
            {
                "parameter": parameter_name,
                "selected_q10": float(observed_lower),
                "selected_median": center,
                "selected_q90": float(observed_upper),
                "default_lower": default_lower,
                "default_upper": default_upper,
                "proposed_lower": float(proposed_lower),
                "proposed_upper": float(proposed_upper),
                "n_selected": len(values),
            }
        )

    selected = selected.merge(
        ranked[
            [
                "sample_id",
                "nearest_biological_distance",
                "biologically_plausible",
            ]
        ],
        on="sample_id",
        how="left",
    ).sort_values("nearest_biological_distance")
    return CalibrationResult(
        parameter_summary=pd.DataFrame(rows),
        selected_models=selected,
        bounds=calibrated_bounds,
    )
