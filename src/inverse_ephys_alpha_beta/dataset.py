"""Dataset generation from sampled HH kinetics."""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import pandas as pd

from .features import (
    FeatureConfig,
    extract_efel_features,
    extract_observed_features,
)
from .hh_model import SimulationConfig, SimulationError, simulate
from .kinetics import KineticParameters, PARAMETER_NAMES
from .protocols import BiologicalScreenConfig, run_biological_screen
from .sampling import ALL_PARAMETER_NAMES, latin_hypercube_parameters
from .static_parameters import StaticParameterTransforms


@dataclass(frozen=True)
class DatasetConfig:
    n_samples: int = 256
    seed: int = 42
    workers: int = 1
    include_canonical: bool = True
    include_efel: bool = False
    include_static: bool = True
    sample_kinetics: bool = True
    kinetic_sampling_mode: str = "independent"
    protocol: str = "fixed"
    parameter_bounds: dict[str, Sequence[float]] | None = None


def _evaluate_one(
    payload: tuple[
        dict[str, Any],
        SimulationConfig,
        FeatureConfig,
        bool,
        str,
        BiologicalScreenConfig | None,
    ]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    (
        row,
        simulation_config,
        feature_config,
        include_efel,
        protocol,
        screen_config,
    ) = payload
    base = {"sample_id": int(row["sample_id"])}
    parameter_names = [name for name in ALL_PARAMETER_NAMES if name in row]
    base.update({name: float(row[name]) for name in parameter_names})
    try:
        kinetics = KineticParameters.from_mapping(row)
        if all(name in row for name in ALL_PARAMETER_NAMES[len(PARAMETER_NAMES) :]):
            static = StaticParameterTransforms.from_mapping(row)
        else:
            static = StaticParameterTransforms.canonical()
        biophysics = static.to_biophysics()
        if protocol == "biological-screen":
            screen_result = run_biological_screen(
                kinetics,
                biophysics,
                screen_config,
                feature_config,
            )
            trace = screen_result.waveform_trace
            feature_values = screen_result.features
        elif protocol == "fixed":
            model_config = replace(simulation_config, conductances=biophysics)
            trace = simulate(kinetics, model_config)
            feature_values = extract_observed_features(
                trace, model_config.stimulus, feature_config
            )
            if include_efel:
                feature_values.update(
                    extract_efel_features(trace, model_config.stimulus)
                )
        else:
            raise ValueError(f"Unknown simulation protocol: {protocol}")
        if not bool(feature_values["is_spiking"]):
            return None, {**base, "rejection_reason": "insufficient_spikes"}
        accepted = {
            **base,
            **biophysics.to_physical_mapping(),
            **{f"feature__{name}": value for name, value in feature_values.items()},
        }
        return accepted, None
    except (FloatingPointError, SimulationError, ValueError) as exc:
        return None, {
            **base,
            "rejection_reason": type(exc).__name__,
            "rejection_detail": str(exc),
        }


def generate_dataset(
    dataset_config: DatasetConfig | None = None,
    simulation_config: SimulationConfig | None = None,
    feature_config: FeatureConfig | None = None,
    screen_config: BiologicalScreenConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate accepted spiking models, rejected models, and run metadata."""
    dataset_config = dataset_config or DatasetConfig()
    parameters = latin_hypercube_parameters(
        dataset_config.n_samples,
        dataset_config.seed,
        dataset_config.include_canonical,
        dataset_config.include_static,
        dataset_config.sample_kinetics,
        dataset_config.parameter_bounds,
        dataset_config.kinetic_sampling_mode,
    )
    return evaluate_parameter_frame(
        parameters,
        dataset_config,
        simulation_config,
        feature_config,
        screen_config,
    )


def evaluate_parameter_frame(
    parameters: pd.DataFrame,
    dataset_config: DatasetConfig | None = None,
    simulation_config: SimulationConfig | None = None,
    feature_config: FeatureConfig | None = None,
    screen_config: BiologicalScreenConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate an existing parameter table with a fixed screening protocol."""
    if parameters.empty:
        raise ValueError("Parameter table is empty")
    if "sample_id" not in parameters:
        raise ValueError("Parameter table must contain sample_id")
    dataset_config = dataset_config or DatasetConfig(n_samples=len(parameters))
    simulation_config = simulation_config or SimulationConfig()
    feature_config = feature_config or FeatureConfig()
    if dataset_config.protocol == "biological-screen":
        feature_config = replace(feature_config, min_spikes=1)
    payloads = [
        (
            row._asdict(),
            simulation_config,
            feature_config,
            dataset_config.include_efel,
            dataset_config.protocol,
            screen_config,
        )
        for row in parameters.itertuples(index=False)
    ]

    executor = None
    if dataset_config.workers == 1:
        results = map(_evaluate_one, payloads)
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=dataset_config.workers)
            results = executor.map(_evaluate_one, payloads)
        except (OSError, PermissionError) as exc:
            warnings.warn(
                f"Process workers are unavailable ({exc}); continuing sequentially.",
                RuntimeWarning,
                stacklevel=2,
            )
            results = map(_evaluate_one, payloads)

    accepted_rows = []
    rejected_rows = []
    try:
        for accepted, rejected in results:
            if accepted is not None:
                accepted_rows.append(accepted)
            if rejected is not None:
                rejected_rows.append(rejected)
    finally:
        if executor is not None:
            executor.shutdown()

    accepted_frame = pd.DataFrame(accepted_rows)
    rejected_frame = pd.DataFrame(rejected_rows)
    metadata = {
        "dataset_config": asdict(dataset_config),
        "simulation_config": asdict(simulation_config),
        "feature_config": asdict(feature_config),
        "screen_config": asdict(screen_config) if screen_config is not None else None,
        "n_accepted": len(accepted_frame),
        "n_rejected": len(rejected_frame),
        "acceptance_fraction": len(accepted_frame) / len(parameters),
        "n_parameter_rows": len(parameters),
    }
    return accepted_frame, rejected_frame, metadata
