"""Sampling utilities for the transformed alpha/beta parameter space."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .kinetics import PARAMETER_NAMES, default_parameter_bounds
from .static_parameters import (
    STATIC_PARAMETER_NAMES,
    StaticParameterTransforms,
    default_static_parameter_bounds,
)

ALL_PARAMETER_NAMES = PARAMETER_NAMES + STATIC_PARAMETER_NAMES
KINETIC_SAMPLING_MODES = ("independent", "gate-correlated", "mixed")


def _correlate_gate_transforms(
    samples: np.ndarray,
    sampled_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    differential_fraction: float = 0.25,
) -> np.ndarray:
    """Turn paired alpha/beta coordinates into common plus differential moves."""
    if not 0.0 <= differential_fraction <= 1.0:
        raise ValueError("differential_fraction must be between zero and one")
    correlated = samples.copy()
    name_to_index = {name: index for index, name in enumerate(sampled_names)}
    fields = ("log_rate_scale", "voltage_shift_mv", "log_slope_scale")
    for gate_name in ("m", "h", "n"):
        for field_name in fields:
            alpha_name = f"param__alpha_{gate_name}__{field_name}"
            beta_name = f"param__beta_{gate_name}__{field_name}"
            if alpha_name not in name_to_index or beta_name not in name_to_index:
                continue
            alpha_index = name_to_index[alpha_name]
            beta_index = name_to_index[beta_name]
            common_lower = max(lower[alpha_index], lower[beta_index])
            common_upper = min(upper[alpha_index], upper[beta_index])
            if common_lower >= common_upper:
                continue

            alpha_unit = (
                samples[:, alpha_index] - lower[alpha_index]
            ) / (upper[alpha_index] - lower[alpha_index])
            beta_unit = (
                samples[:, beta_index] - lower[beta_index]
            ) / (upper[beta_index] - lower[beta_index])
            common = common_lower + alpha_unit * (common_upper - common_lower)
            requested_difference = (
                (2.0 * beta_unit - 1.0)
                * differential_fraction
                * (common_upper - common_lower)
            )
            positive_limit = np.minimum(
                upper[alpha_index] - common,
                common - lower[beta_index],
            )
            negative_limit = np.minimum(
                common - lower[alpha_index],
                upper[beta_index] - common,
            )
            half_difference = requested_difference / 2.0
            half_difference = np.minimum(half_difference, positive_limit)
            half_difference = np.maximum(half_difference, -negative_limit)
            correlated[:, alpha_index] = common + half_difference
            correlated[:, beta_index] = common - half_difference
    return correlated


def latin_hypercube_parameters(
    n_samples: int,
    seed: int = 42,
    include_canonical: bool = True,
    include_static: bool = True,
    sample_kinetics: bool = True,
    parameter_bounds: Mapping[str, Sequence[float]] | None = None,
    kinetic_sampling_mode: str = "independent",
) -> pd.DataFrame:
    """Sample kinetic and optional static priors with Latin hypercube sampling."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least one")
    if not include_static and not sample_kinetics:
        raise ValueError("At least one parameter group must be sampled")
    if kinetic_sampling_mode not in KINETIC_SAMPLING_MODES:
        raise ValueError(f"Unknown kinetic sampling mode: {kinetic_sampling_mode}")

    kinetic_lower, kinetic_upper = default_parameter_bounds()
    if include_static:
        static_lower, static_upper = default_static_parameter_bounds()
        parameter_names = ALL_PARAMETER_NAMES
        canonical = np.concatenate(
            (
                np.zeros_like(kinetic_lower),
                StaticParameterTransforms.canonical().to_vector(),
            )
        )
        if sample_kinetics:
            lower = np.concatenate((kinetic_lower, static_lower))
            upper = np.concatenate((kinetic_upper, static_upper))
            sampled_names = ALL_PARAMETER_NAMES
        else:
            lower, upper = static_lower, static_upper
            sampled_names = STATIC_PARAMETER_NAMES
    else:
        lower, upper = kinetic_lower, kinetic_upper
        parameter_names = PARAMETER_NAMES
        canonical = np.zeros_like(kinetic_lower)
        sampled_names = PARAMETER_NAMES

    if parameter_bounds:
        lower = lower.copy()
        upper = upper.copy()
        sampled_name_to_index = {
            parameter_name: index
            for index, parameter_name in enumerate(sampled_names)
        }
        unknown = set(parameter_bounds).difference(sampled_name_to_index)
        if unknown:
            raise ValueError(f"Bounds contain unsampled parameters: {sorted(unknown)}")
        for parameter_name, bounds in parameter_bounds.items():
            if len(bounds) != 2:
                raise ValueError(f"Bounds for {parameter_name} must have two values")
            bound_lower, bound_upper = (float(value) for value in bounds)
            if not bound_lower < bound_upper:
                raise ValueError(f"Invalid bounds for {parameter_name}: {bounds}")
            parameter_index = sampled_name_to_index[parameter_name]
            lower[parameter_index] = bound_lower
            upper[parameter_index] = bound_upper
    random_count = n_samples - int(include_canonical)
    rows = []
    if include_canonical:
        rows.append(canonical)
    if random_count:
        unit_samples = qmc.LatinHypercube(d=len(lower), seed=seed).random(random_count)
        random_samples = qmc.scale(unit_samples, lower, upper)
        if sample_kinetics:
            if kinetic_sampling_mode == "gate-correlated":
                random_samples = _correlate_gate_transforms(
                    random_samples,
                    sampled_names,
                    lower,
                    upper,
                )
            elif kinetic_sampling_mode == "mixed":
                correlated_start = (random_count + 1) // 2
                random_samples[correlated_start:] = _correlate_gate_transforms(
                    random_samples[correlated_start:],
                    sampled_names,
                    lower,
                    upper,
                )
        if include_static and not sample_kinetics:
            random_samples = np.column_stack(
                (
                    np.zeros((random_count, len(kinetic_lower))),
                    random_samples,
                )
            )
        rows.extend(random_samples)

    frame = pd.DataFrame(np.asarray(rows), columns=parameter_names)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=int))
    return frame
