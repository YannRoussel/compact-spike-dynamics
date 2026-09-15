"""Targeted direct optimization of HH parameters against biological medoids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .biological_data import harmonize_model_features
from .dataset import DatasetConfig, evaluate_parameter_frame
from .features import FeatureConfig
from .protocols import biological_screen_config
from .sampling import ALL_PARAMETER_NAMES


DIRECT_FIT_FEATURE_GROUPS = {
    "waveform": (
        "ap_threshold_mv",
        "ap_peak_mv",
        "ap_width_ms",
        "fast_trough_mv",
    ),
    "velocity": (
        "upstroke_mv_ms",
        "downstroke_mv_ms",
    ),
    "timing": (
        "ap_upstroke_time_ms",
        "ap_repolarization_time_ms",
    ),
    "inflection": (
        "upstroke_inflection_voltage_mv",
        "upstroke_inflection_dvdt_mv_ms",
        "downstroke_inflection_voltage_mv",
        "downstroke_inflection_dvdt_mv_ms",
    ),
    "phase_geometry": (
        "ap_phase_area_normalized",
        "cycle_phase_area_normalized",
        "cycle_phase_path_length_normalized",
    ),
}

DIRECT_VALIDATION_FEATURE_GROUPS = {
    "onset": (
        "onset_voltage_mv",
        "onset_rapidness_per_ms",
    ),
    "acceleration": (
        "max_acceleration_mv_ms2",
        "max_acceleration_voltage_mv",
        "min_acceleration_mv_ms2",
        "min_acceleration_voltage_mv",
    ),
}


@dataclass(frozen=True)
class DirectOptimizationConfig:
    n_targets: int = 2
    rounds: int = 3
    batch_size: int = 32
    elite_count: int = 12
    validation_count: int = 3
    long_promotions_per_round: int = 4
    workers: int = 1
    seed: int = 42
    differential_weight: float = 0.45
    mutation_fraction: float = 0.35
    initial_mutation_scale: float = 0.08
    mutation_decay: float = 0.70


@dataclass(frozen=True)
class FeatureNormalization:
    center: pd.Series
    scale: pd.Series


@dataclass(frozen=True)
class DirectOptimizationResult:
    targets: pd.DataFrame
    history: pd.DataFrame
    round_summary: pd.DataFrame
    best_fast_models: pd.DataFrame
    long_validation_models: pd.DataFrame
    target_summary: pd.DataFrame
    metadata: dict


def fit_features() -> tuple[str, ...]:
    return tuple(
        feature
        for group in DIRECT_FIT_FEATURE_GROUPS.values()
        for feature in group
    )


def validation_features() -> tuple[str, ...]:
    return tuple(
        feature
        for group in DIRECT_VALIDATION_FEATURE_GROUPS.values()
        for feature in group
    )


def biological_normalization(
    biological: pd.DataFrame,
    features: Sequence[str],
) -> FeatureNormalization:
    values = biological.loc[:, features].apply(pd.to_numeric, errors="coerce")
    center = values.median()
    lower = values.quantile(0.10)
    upper = values.quantile(0.90)
    scale = upper - lower
    fallback = values.quantile(0.75) - values.quantile(0.25)
    scale = scale.where(scale > np.finfo(float).eps, fallback)
    scale = scale.where(scale > np.finfo(float).eps, values.std())
    if scale.isna().any() or (scale <= np.finfo(float).eps).any():
        invalid = scale.index[scale.isna() | (scale <= np.finfo(float).eps)]
        raise ValueError(f"Features have no biological variation: {invalid.tolist()}")
    return FeatureNormalization(center=center, scale=scale)


def select_biological_medoids(
    biological: pd.DataFrame,
    n_targets: int,
    normalization: FeatureNormalization,
    seed: int,
    clustering_clip: float = 3.0,
) -> pd.DataFrame:
    features = tuple(normalization.center.index)
    complete = biological.dropna(subset=list(features)).copy()
    if len(complete) < n_targets:
        raise ValueError("Not enough complete biological cells for target selection")
    scaled = (
        complete.loc[:, features] - normalization.center
    ) / normalization.scale
    clustering_values = np.clip(
        scaled.to_numpy(dtype=float),
        -clustering_clip,
        clustering_clip,
    )
    kmeans = KMeans(
        n_clusters=n_targets,
        n_init=20,
        random_state=seed,
    ).fit(clustering_values)
    target_rows = []
    for cluster_index in range(n_targets):
        member_indices = np.flatnonzero(kmeans.labels_ == cluster_index)
        distances = np.linalg.norm(
            clustering_values[member_indices]
            - kmeans.cluster_centers_[cluster_index],
            axis=1,
        )
        row_position = member_indices[int(np.argmin(distances))]
        target = complete.iloc[row_position].copy()
        target["target_id"] = f"target_{cluster_index:02d}"
        target["cluster_size"] = len(member_indices)
        target["cluster_fraction"] = len(member_indices) / len(complete)
        target["distance_to_cluster_center"] = float(np.min(distances))
        target_rows.append(target)
    return pd.DataFrame(target_rows).sort_values(
        "cluster_size",
        ascending=False,
    ).reset_index(drop=True)


def _huber(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= 1.0, 0.5 * values**2, absolute - 0.5)


def score_models(
    semantic_models: pd.DataFrame,
    target: pd.Series,
    normalization: FeatureNormalization,
    feature_groups: Mapping[str, Sequence[str]] = DIRECT_FIT_FEATURE_GROUPS,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=semantic_models.index)
    group_columns = []
    for group_name, features in feature_groups.items():
        feature_list = list(features)
        differences = (
            semantic_models.loc[:, feature_list]
            - target.loc[feature_list].to_numpy(dtype=float)
        ) / normalization.scale.loc[feature_list].to_numpy(dtype=float)
        group_score = np.nanmean(_huber(differences.to_numpy(dtype=float)), axis=1)
        group_column = f"score__{group_name}"
        scores[group_column] = group_score
        group_columns.append(group_column)
    scores["objective_score"] = scores.loc[:, group_columns].mean(axis=1)
    invalid = semantic_models.loc[
        :,
        [feature for group in feature_groups.values() for feature in group],
    ].isna().any(axis=1)
    scores.loc[invalid, group_columns + ["objective_score"]] = np.inf
    return scores


def load_parameter_bounds(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = document.get("bounds", document)
    missing = set(ALL_PARAMETER_NAMES).difference(bounds)
    if missing:
        raise ValueError(f"Parameter bounds are missing: {sorted(missing)}")
    lower = np.asarray([float(bounds[name][0]) for name in ALL_PARAMETER_NAMES])
    upper = np.asarray([float(bounds[name][1]) for name in ALL_PARAMETER_NAMES])
    if np.any(lower >= upper):
        raise ValueError("All parameter lower bounds must be below upper bounds")
    return lower, upper


def propose_evolution_batch(
    elites: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    round_index: int,
    config: DirectOptimizationConfig,
) -> pd.DataFrame:
    if elites.empty:
        raise ValueError("Cannot propose candidates without elite models")
    elite_values = elites.loc[:, ALL_PARAMETER_NAMES].to_numpy(dtype=float)
    parameter_width = upper - lower
    mutation_scale = (
        config.initial_mutation_scale
        * config.mutation_decay**round_index
        * parameter_width
    )
    candidates = []
    for _ in range(batch_size):
        anchor = elite_values[rng.integers(len(elite_values))]
        first = elite_values[rng.integers(len(elite_values))]
        second = elite_values[rng.integers(len(elite_values))]
        proposal = (
            anchor
            + config.differential_weight * (first - second)
            + rng.normal(0.0, mutation_scale)
        )
        mutation_mask = rng.random(len(ALL_PARAMETER_NAMES)) < config.mutation_fraction
        if not mutation_mask.any():
            mutation_mask[rng.integers(len(ALL_PARAMETER_NAMES))] = True
        candidate = anchor.copy()
        candidate[mutation_mask] = proposal[mutation_mask]
        candidates.append(np.clip(candidate, lower, upper))
    return pd.DataFrame(candidates, columns=ALL_PARAMETER_NAMES)


def _select_multifidelity_elites(
    long_pool: pd.DataFrame,
    fast_pool: pd.DataFrame,
    elite_count: int,
) -> pd.DataFrame:
    source = long_pool if not long_pool.empty else fast_pool
    return source.sort_values("objective_score").head(elite_count)


def _evaluate_candidates(
    candidates: pd.DataFrame,
    workers: int,
    screen_preset: str,
    temperature_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_frame = candidates.copy()
    candidate_frame.insert(0, "sample_id", np.arange(len(candidate_frame), dtype=int))
    dataset_config = DatasetConfig(
        n_samples=len(candidate_frame),
        workers=workers,
        include_canonical=False,
        protocol="biological-screen",
    )
    accepted, rejected, _ = evaluate_parameter_frame(
        candidate_frame,
        dataset_config=dataset_config,
        feature_config=FeatureConfig(min_spikes=1),
        screen_config=biological_screen_config(
            screen_preset,
            temperature_c=temperature_c,
        ),
    )
    return accepted, rejected


def _score_evaluated_models(
    evaluated: pd.DataFrame,
    target: pd.Series,
    normalization: FeatureNormalization,
) -> pd.DataFrame:
    if evaluated.empty:
        return evaluated.copy()
    semantic = harmonize_model_features(evaluated)
    scores = score_models(semantic, target, normalization)
    validation = score_models(
        semantic,
        target,
        normalization,
        feature_groups=DIRECT_VALIDATION_FEATURE_GROUPS,
    ).rename(
        columns={
            column: f"validation__{column.removeprefix('score__')}"
            for column in (
                *(
                    f"score__{name}"
                    for name in DIRECT_VALIDATION_FEATURE_GROUPS
                ),
                "objective_score",
            )
        }
    )
    return pd.concat(
        [
            evaluated.reset_index(drop=True),
            semantic.drop(columns=["dataset", "cell_id"]).add_prefix(
                "semantic__"
            ),
            scores.reset_index(drop=True),
            validation.reset_index(drop=True),
        ],
        axis=1,
    )


def _target_summary_row(
    target_id: str,
    fast_models: pd.DataFrame,
    validation_models: pd.DataFrame,
) -> dict[str, float | int | str]:
    initial_fast = fast_models.loc[fast_models["candidate_origin"].eq("initial")]
    optimized_fast = fast_models.loc[
        fast_models["candidate_origin"].eq("optimized")
    ]
    initial_long = validation_models.loc[
        validation_models["candidate_origin"].eq("initial")
    ]
    optimized_long = validation_models.loc[
        validation_models["candidate_origin"].eq("optimized")
    ]

    def minimum(frame: pd.DataFrame) -> float:
        return (
            float(frame["objective_score"].min())
            if not frame.empty
            else float("nan")
        )

    initial_fast_score = minimum(initial_fast)
    optimized_fast_score = minimum(optimized_fast)
    initial_long_score = minimum(initial_long)
    optimized_long_score = minimum(optimized_long)
    return {
        "target_id": target_id,
        "initial_fast_score": initial_fast_score,
        "optimized_fast_score": optimized_fast_score,
        "fast_improvement_fraction": (
            (initial_fast_score - optimized_fast_score) / initial_fast_score
            if np.isfinite(initial_fast_score) and initial_fast_score > 0.0
            else np.nan
        ),
        "initial_long_score": initial_long_score,
        "optimized_long_score": optimized_long_score,
        "long_improvement_fraction": (
            (initial_long_score - optimized_long_score) / initial_long_score
            if np.isfinite(initial_long_score) and initial_long_score > 0.0
            else np.nan
        ),
        "n_fast_optimized_models": len(optimized_fast),
        "n_long_optimized_models": len(optimized_long),
    }


def _attach_candidate_metadata(
    evaluated: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                *evaluated.columns,
                *candidate_metadata.columns,
            ]
        )
    return evaluated.merge(
        candidate_metadata,
        left_on="sample_id",
        right_index=True,
        how="left",
    )


def _promote_with_long_protocol(
    candidates: pd.DataFrame,
    target: pd.Series,
    normalization: FeatureNormalization,
    workers: int,
    screen_preset: str,
    temperature_c: float,
    target_id: str,
    promotion_round: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.reset_index(drop=True)
    accepted, rejected = _evaluate_candidates(
        candidates.loc[:, ALL_PARAMETER_NAMES],
        workers,
        screen_preset,
        temperature_c,
    )
    metadata = candidates[["candidate_origin", "objective_score"]].rename(
        columns={"objective_score": "fast_objective_score"}
    )
    metadata["promotion_round"] = promotion_round
    accepted = _attach_candidate_metadata(accepted, metadata)
    if accepted.empty:
        accepted_scored = pd.DataFrame(
            columns=[
                "candidate_origin",
                "target_id",
                "round",
                "fidelity",
                "accepted",
                "objective_score",
                "fast_objective_score",
            ]
        )
    else:
        accepted_scored = _score_evaluated_models(
            accepted.drop(
                columns=[
                    "candidate_origin",
                    "fast_objective_score",
                    "promotion_round",
                ],
                errors="ignore",
            ),
            target,
            normalization,
        )
        accepted_scored["candidate_origin"] = accepted[
            "candidate_origin"
        ].to_numpy()
        accepted_scored["fast_objective_score"] = accepted[
            "fast_objective_score"
        ].to_numpy()
        accepted_scored["promotion_round"] = promotion_round
    accepted_scored["target_id"] = target_id
    accepted_scored["round"] = promotion_round
    accepted_scored["fidelity"] = "long"
    accepted_scored["accepted"] = True

    rejected = _attach_candidate_metadata(rejected, metadata)
    rejected["target_id"] = target_id
    rejected["round"] = promotion_round
    rejected["fidelity"] = "long"
    rejected["accepted"] = False
    return accepted_scored, rejected


def optimize_biological_targets(
    biological: pd.DataFrame,
    model_population: pd.DataFrame,
    bounds_path: str | Path,
    temperature_c: float,
    long_screen_preset: str,
    config: DirectOptimizationConfig | None = None,
) -> DirectOptimizationResult:
    config = config or DirectOptimizationConfig()
    if config.long_promotions_per_round < 1:
        raise ValueError("long_promotions_per_round must be positive")
    if config.long_promotions_per_round > config.batch_size:
        raise ValueError("long promotions cannot exceed the fast batch size")
    features = fit_features()
    all_objective_features = features + validation_features()
    biological_complete = biological.dropna(subset=list(features)).copy()
    model_semantic = harmonize_model_features(model_population)
    model_complete_mask = ~model_semantic.loc[:, features].isna().any(axis=1)
    model_population = model_population.loc[model_complete_mask].reset_index(drop=True)
    model_semantic = model_semantic.loc[model_complete_mask].reset_index(drop=True)
    if model_population.empty:
        raise ValueError("No existing models have complete direct-fit features")

    normalization = biological_normalization(
        biological_complete,
        all_objective_features,
    )
    fit_normalization = FeatureNormalization(
        center=normalization.center.loc[list(features)],
        scale=normalization.scale.loc[list(features)],
    )
    targets = select_biological_medoids(
        biological_complete,
        config.n_targets,
        fit_normalization,
        config.seed,
    )
    lower, upper = load_parameter_bounds(bounds_path)
    rng = np.random.default_rng(config.seed)

    history_frames = []
    round_rows = []
    best_fast_frames = []
    validation_frames = []
    target_summary_rows = []
    for _, target in targets.iterrows():
        target_id = str(target["target_id"])
        initial_scores = score_models(model_semantic, target, normalization)
        initial_pool = pd.concat(
            [
                model_population.reset_index(drop=True),
                initial_scores.reset_index(drop=True),
            ],
            axis=1,
        ).sort_values("objective_score")
        initial_elites = initial_pool.head(config.elite_count).copy()
        initial_elites["candidate_origin"] = "initial"
        initial_elites["target_id"] = target_id
        initial_elites["round"] = -1
        initial_elites["fidelity"] = "fast"
        initial_elites["accepted"] = True
        history_frames.append(initial_elites)
        fast_pool = initial_elites.copy()
        start_score = float(initial_elites["objective_score"].iloc[0])

        initial_promotions = initial_elites.head(
            config.long_promotions_per_round
        ).copy()
        initial_long, initial_long_rejected = _promote_with_long_protocol(
            initial_promotions,
            target,
            normalization,
            config.workers,
            long_screen_preset,
            temperature_c,
            target_id,
            -1,
        )
        history_frames.extend((initial_long, initial_long_rejected))
        validation_frames.extend((initial_long, initial_long_rejected))
        long_pool = initial_long.copy()
        round_rows.append(
            {
                "target_id": target_id,
                "round": -1,
                "attempted": 0,
                "accepted": len(initial_elites),
                "best_score": start_score,
                "promoted_long": len(initial_promotions),
                "long_accepted": len(initial_long),
                "best_long_score": (
                    float(initial_long["objective_score"].min())
                    if not initial_long.empty
                    else np.nan
                ),
            }
        )

        for round_index in range(config.rounds):
            elites = _select_multifidelity_elites(
                long_pool,
                fast_pool,
                config.elite_count,
            )
            candidates = propose_evolution_batch(
                elites,
                lower,
                upper,
                config.batch_size,
                rng,
                round_index,
                config,
            )
            accepted, rejected = _evaluate_candidates(
                candidates,
                config.workers,
                "fast",
                temperature_c,
            )
            accepted_scored = _score_evaluated_models(
                accepted,
                target,
                normalization,
            )
            accepted_scored["candidate_origin"] = "optimized"
            accepted_scored["target_id"] = target_id
            accepted_scored["round"] = round_index
            accepted_scored["fidelity"] = "fast"
            accepted_scored["accepted"] = True
            rejected = rejected.copy()
            rejected["candidate_origin"] = "optimized"
            rejected["target_id"] = target_id
            rejected["round"] = round_index
            rejected["fidelity"] = "fast"
            rejected["accepted"] = False
            history_frames.extend((accepted_scored, rejected))
            if not accepted_scored.empty:
                fast_pool = pd.concat(
                    (fast_pool, accepted_scored),
                    ignore_index=True,
                    sort=False,
                )
                fast_pool = fast_pool.sort_values("objective_score").drop_duplicates(
                    subset=list(ALL_PARAMETER_NAMES)
                )
                promotions = accepted_scored.sort_values(
                    "objective_score"
                ).head(config.long_promotions_per_round)
                promoted_long, promoted_rejected = _promote_with_long_protocol(
                    promotions,
                    target,
                    normalization,
                    config.workers,
                    long_screen_preset,
                    temperature_c,
                    target_id,
                    round_index,
                )
                history_frames.extend((promoted_long, promoted_rejected))
                validation_frames.extend((promoted_long, promoted_rejected))
                if not promoted_long.empty:
                    long_pool = pd.concat(
                        (long_pool, promoted_long),
                        ignore_index=True,
                        sort=False,
                    )
                    long_pool = long_pool.sort_values(
                        "objective_score"
                    ).drop_duplicates(subset=list(ALL_PARAMETER_NAMES))
            else:
                promotions = pd.DataFrame()
                promoted_long = pd.DataFrame()
            round_rows.append(
                {
                    "target_id": target_id,
                    "round": round_index,
                    "attempted": len(candidates),
                    "accepted": len(accepted_scored),
                    "best_score": float(fast_pool["objective_score"].min()),
                    "promoted_long": len(promotions),
                    "long_accepted": len(promoted_long),
                    "best_long_score": (
                        float(long_pool["objective_score"].min())
                        if not long_pool.empty
                        else np.nan
                    ),
                }
            )

        optimized_pool = fast_pool.loc[
            fast_pool["candidate_origin"].eq("optimized")
        ]
        selected_optimized = optimized_pool.sort_values("objective_score").head(
            config.validation_count
        )
        best_fast = pd.concat(
            (
                initial_elites.head(1),
                selected_optimized,
            ),
            ignore_index=True,
            sort=False,
        )
        best_fast_frames.append(best_fast)
        target_summary_rows.append(
            _target_summary_row(
                target_id,
                fast_pool,
                long_pool,
            )
        )

    history = pd.concat(
        [frame for frame in history_frames if not frame.empty],
        ignore_index=True,
        sort=False,
    )
    best_fast_models = pd.concat(
        best_fast_frames,
        ignore_index=True,
        sort=False,
    )
    long_validation_models = pd.concat(
        [frame for frame in validation_frames if not frame.empty],
        ignore_index=True,
        sort=False,
    )
    metadata = {
        "config": asdict(config),
        "bounds_path": str(bounds_path),
        "temperature_c": temperature_c,
        "long_screen_preset": long_screen_preset,
        "fit_feature_groups": DIRECT_FIT_FEATURE_GROUPS,
        "validation_feature_groups": DIRECT_VALIDATION_FEATURE_GROUPS,
        "optimization_mode": "multi_fidelity_long_promotions",
        "n_biological_complete": len(biological_complete),
        "n_initial_models_complete": len(model_population),
    }
    return DirectOptimizationResult(
        targets=targets,
        history=history,
        round_summary=pd.DataFrame(round_rows),
        best_fast_models=best_fast_models,
        long_validation_models=long_validation_models,
        target_summary=pd.DataFrame(target_summary_rows),
        metadata=metadata,
    )


def save_direct_optimization(
    result: DirectOptimizationResult,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.targets.to_csv(output / "biological_targets.csv", index=False)
    result.history.to_csv(output / "optimization_history.csv", index=False)
    result.round_summary.to_csv(output / "round_summary.csv", index=False)
    result.best_fast_models.to_csv(output / "best_fast_models.csv", index=False)
    result.long_validation_models.to_csv(
        output / "long_validation_models.csv",
        index=False,
    )
    result.target_summary.to_csv(output / "target_summary.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )
