#!/usr/bin/env python3
"""Predict reference t-type spike waveforms from Allen atlas expression."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression

from inverse_ephys_alpha_beta.atlas_ephys import (
    AP_FEATURE_NAMES,
    WAVEFORM_TARGET_NAMES,
    ap_feature_target_matrix,
    clip_ap_features,
    clip_waveform_parameters,
    fit_distance_model,
    fit_fixed_class_rrr,
    harmonize_atlas_to_patchseq,
    inverse_ap_feature_target_matrix,
    inverse_waveform_target_matrix,
    load_allen_reference_atlas,
    predict_fixed_class_rrr,
    spike_cycle_features,
    spike_cycle_from_row,
    transcriptomic_support,
    waveform_features,
    waveform_parameter_frame,
    waveform_target_matrix,
)
from inverse_ephys_alpha_beta.reduced_rank import (
    fit_class_residualized_rrr_analysis,
    predict_class_residualized_rrr,
)
from inverse_ephys_alpha_beta.transcriptomics import (
    PatchSeqExpression,
    align_expression_to_parameters,
    broad_transcriptomic_class,
    expression_matrix,
    load_gouwens_expression,
    load_scala_expression,
)


DEFAULT_KOBAK_ROOT = Path(
    "/Users/yannroussel/Documents/BBP_postdoc/projects/"
    "sRRR_Kobak/patch-seq-rrr"
)
DEFAULT_PATCHSEQ_ROOT = Path(
    "/Users/yannroussel/Documents/BBP_postdoc/data/Patch Seq data"
)
CLASS_COLORS = {
    "Lamp5": "#7c3f98",
    "Sncg": "#c94f9d",
    "Vip": "#dc7b25",
    "Sst": "#3b8d74",
    "Pvalb": "#3376a6",
    "Glut_IT": "#bb4545",
    "Glut_ET": "#8f5d28",
    "Glut_CT": "#638c3d",
    "Glut_NP": "#5f699f",
    "Glut_L6b": "#7c6f64",
    "Other": "#777777",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _classes(expression: PatchSeqExpression) -> np.ndarray:
    return np.asarray(
        [broad_transcriptomic_class(label) for label in expression.transcriptomic_types],
        dtype=object,
    )


def _load_expression(
    dataset: str,
    parameters: pd.DataFrame,
    args,
) -> tuple[PatchSeqExpression, pd.DataFrame]:
    requested = set(
        parameters.loc[parameters["dataset"].eq(dataset), "cell_id"].astype(str)
    )
    if dataset == "gouwens_visp":
        expression = load_gouwens_expression(
            args.gouwens_mat,
            args.gouwens_metadata,
            channel_cpm_path=args.gouwens_cpm,
            requested_cell_ids=requested,
            channel_cache_path=args.expression_cache / "gouwens_scn_kcn_expression.npz",
        )
    elif dataset == "scala_room_temperature":
        expression = load_scala_expression(
            args.scala_pickle,
            args.scala_metadata,
            include_channel_genes=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return align_expression_to_parameters(expression, parameters)


def _feature_row(values: np.ndarray) -> dict[str, float]:
    row = dict(zip(WAVEFORM_TARGET_NAMES, np.asarray(values, dtype=float)))
    return spike_cycle_features(spike_cycle_from_row(row))


def _loto_fold(
    heldout_type: str,
    x: np.ndarray,
    y: np.ndarray,
    y_raw: np.ndarray,
    types: np.ndarray,
    classes: np.ndarray,
    rank: int,
    ridge_penalty: float,
    sparse_ratio: float,
) -> dict[str, object]:
    test = types == heldout_type
    training = ~test
    broad_class = str(classes[np.flatnonzero(test)[0]])
    model = fit_fixed_class_rrr(
        x[training],
        y[training],
        classes[training],
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratio=sparse_ratio,
    )
    centroid = np.mean(x[test], axis=0, keepdims=True)
    predicted, class_only = predict_fixed_class_rrr(
        model,
        x[training],
        y[training],
        classes[training],
        centroid,
        [broad_class],
    )
    training_raw = y_raw[training]
    predicted_raw = inverse_waveform_target_matrix(predicted)
    class_raw = inverse_waveform_target_matrix(class_only)
    predicted_raw, predicted_clipped = clip_waveform_parameters(
        predicted_raw,
        training_raw,
    )
    class_raw, class_clipped = clip_waveform_parameters(class_raw, training_raw)
    true_raw = np.mean(y_raw[test], axis=0, keepdims=True)
    return {
        "transcriptomic_type": heldout_type,
        "broad_class": broad_class,
        "cell_count": int(np.sum(test)),
        "selected_gene_count": int(np.sum(model.selected_predictors)),
        "true_parameters": true_raw[0],
        "rna_parameters": predicted_raw[0],
        "class_parameters": class_raw[0],
        "rna_feature_values": _feature_row(predicted_raw[0]),
        "class_feature_values": _feature_row(class_raw[0]),
        "true_feature_values": _feature_row(true_raw[0]),
        "rna_parameter_clipped": bool(predicted_clipped[0]),
        "class_parameter_clipped": bool(class_clipped[0]),
    }


def _loto_feature_fold(
    heldout_type: str,
    x: np.ndarray,
    y: np.ndarray,
    y_raw: np.ndarray,
    types: np.ndarray,
    classes: np.ndarray,
    rank: int,
    ridge_penalty: float,
    sparse_ratio: float,
) -> dict[str, object]:
    test = types == heldout_type
    training = ~test
    broad_class = str(classes[np.flatnonzero(test)[0]])
    model = fit_fixed_class_rrr(
        x[training],
        y[training],
        classes[training],
        rank=rank,
        ridge_penalty=ridge_penalty,
        sparse_ratio=sparse_ratio,
    )
    predicted, class_only = predict_fixed_class_rrr(
        model,
        x[training],
        y[training],
        classes[training],
        np.mean(x[test], axis=0, keepdims=True),
        [broad_class],
    )
    predicted_raw = inverse_ap_feature_target_matrix(predicted)
    class_raw = inverse_ap_feature_target_matrix(class_only)
    predicted_raw, predicted_clipped = clip_ap_features(
        predicted_raw,
        y_raw[training],
    )
    class_raw, class_clipped = clip_ap_features(class_raw, y_raw[training])
    return {
        "transcriptomic_type": heldout_type,
        "rna_direct_features": predicted_raw[0],
        "class_direct_features": class_raw[0],
        "true_direct_features": np.mean(y_raw[test], axis=0),
        "rna_direct_clipped": bool(predicted_clipped[0]),
        "class_direct_clipped": bool(class_clipped[0]),
    }


def _loto_distance_table(distance_model) -> pd.DataFrame:
    rows = []
    for index, (label, broad_class) in enumerate(
        zip(distance_model.training_types, distance_model.training_classes)
    ):
        candidates = np.flatnonzero(distance_model.training_classes == broad_class)
        candidates = candidates[candidates != index]
        if not len(candidates):
            candidates = np.delete(np.arange(len(distance_model.training_types)), index)
        distances = np.linalg.norm(
            distance_model.training_scores[candidates]
            - distance_model.training_scores[index],
            axis=1,
        )
        nearest = int(candidates[int(np.argmin(distances))])
        distance = float(np.min(distances))
        rows.append(
            {
                "transcriptomic_type": str(label),
                "nearest_training_type": str(distance_model.training_types[nearest]),
                "transcriptomic_distance": distance,
                "distance_ratio_to_loto_p95": (
                    distance / distance_model.reference_distance
                ),
            }
        )
    return pd.DataFrame(rows)


def _loto_frame(
    results: Sequence[dict[str, object]],
    direct_results: Sequence[dict[str, object]],
) -> pd.DataFrame:
    direct_lookup = {
        str(result["transcriptomic_type"]): result for result in direct_results
    }
    rows = []
    for result in results:
        direct = direct_lookup[str(result["transcriptomic_type"])]
        row = {
            key: result[key]
            for key in (
                "transcriptomic_type",
                "broad_class",
                "cell_count",
                "selected_gene_count",
                "rna_parameter_clipped",
                "class_parameter_clipped",
            )
        }
        row["rna_direct_feature_clipped"] = direct["rna_direct_clipped"]
        row["class_direct_feature_clipped"] = direct["class_direct_clipped"]
        for prefix, source in (
            ("waveform_true", result["true_feature_values"]),
            ("waveform_rna", result["rna_feature_values"]),
            ("waveform_class", result["class_feature_values"]),
        ):
            for feature in AP_FEATURE_NAMES:
                row[f"{prefix}_{feature}"] = source[feature]
        for prefix, values in (
            ("true", direct["true_direct_features"]),
            ("rna", direct["rna_direct_features"]),
            ("class", direct["class_direct_features"]),
        ):
            for feature, value in zip(AP_FEATURE_NAMES, values):
                row[f"{prefix}_{feature}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    true = frame[[f"true_{name}" for name in AP_FEATURE_NAMES]].to_numpy()
    predicted = frame[[f"rna_{name}" for name in AP_FEATURE_NAMES]].to_numpy()
    class_only = frame[[f"class_{name}" for name in AP_FEATURE_NAMES]].to_numpy()
    scale = np.maximum(np.std(true, axis=0), 1e-8)
    frame["rna_feature_nrmse"] = np.sqrt(
        np.mean(np.square((true - predicted) / scale), axis=1)
    )
    frame["class_feature_nrmse"] = np.sqrt(
        np.mean(np.square((true - class_only) / scale), axis=1)
    )
    waveform_true = frame[
        [f"waveform_true_{name}" for name in AP_FEATURE_NAMES]
    ].to_numpy()
    waveform_rna = frame[
        [f"waveform_rna_{name}" for name in AP_FEATURE_NAMES]
    ].to_numpy()
    waveform_scale = np.maximum(np.std(waveform_true, axis=0), 1e-8)
    frame["waveform_derived_feature_nrmse"] = np.sqrt(
        np.mean(np.square((waveform_true - waveform_rna) / waveform_scale), axis=1)
    )
    return frame


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.sum(np.square(actual - predicted)) / denominator)


def _feature_performance(loto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in AP_FEATURE_NAMES:
        actual = loto[f"true_{feature}"].to_numpy(dtype=float)
        rna = loto[f"rna_{feature}"].to_numpy(dtype=float)
        class_only = loto[f"class_{feature}"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "rna_r2": _r2(actual, rna),
                "class_only_r2": _r2(actual, class_only),
                "rna_pearson_r": float(np.corrcoef(actual, rna)[0, 1]),
                "class_only_pearson_r": float(
                    np.corrcoef(actual, class_only)[0, 1]
                ),
                "rna_median_absolute_error": float(np.median(np.abs(actual - rna))),
                "class_median_absolute_error": float(
                    np.median(np.abs(actual - class_only))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["rna_delta_r2_over_class"] = frame["rna_r2"] - frame["class_only_r2"]
    return frame


def _waveform_feature_performance(loto: pd.DataFrame) -> pd.DataFrame:
    renamed = loto.copy()
    for feature in AP_FEATURE_NAMES:
        renamed[f"true_{feature}"] = renamed[f"waveform_true_{feature}"]
        renamed[f"rna_{feature}"] = renamed[f"waveform_rna_{feature}"]
        renamed[f"class_{feature}"] = renamed[f"waveform_class_{feature}"]
    return _feature_performance(renamed)


def _waveform_qc_mask(features: pd.DataFrame) -> np.ndarray:
    """Keep complete AP cycles and reject plateaus at the q50 current."""
    finite = np.isfinite(features.loc[:, AP_FEATURE_NAMES]).all(axis=1)
    return np.asarray(
        finite
        & features["ap_amplitude_mv"].ge(20.0)
        & features["ap_half_width_ms"].between(0.1, 10.0)
        & features["max_upstroke_mv_ms"].between(10.0, 2000.0)
        & features["max_downstroke_mv_ms"].between(-2000.0, -5.0),
        dtype=bool,
    )


def _direct_atlas_mapping(
    distance_model,
    atlas_values: np.ndarray,
    atlas_metadata: pd.DataFrame,
) -> dict[int, list[str]]:
    standardized = np.clip(
        (atlas_values - distance_model.gene_mean) / distance_model.gene_scale,
        -8.0,
        8.0,
    )
    atlas_scores = distance_model.pca.transform(standardized)
    atlas_classes = atlas_metadata["broad_class"].to_numpy(dtype=object)
    matches: dict[int, list[str]] = {}
    for score, label, broad_class in zip(
        distance_model.training_scores,
        distance_model.training_types,
        distance_model.training_classes,
    ):
        if str(broad_class) == "Other":
            continue
        candidates = np.flatnonzero(atlas_classes == broad_class)
        if not len(candidates):
            candidates = np.arange(len(atlas_metadata))
        nearest = int(
            candidates[
                int(np.argmin(np.linalg.norm(atlas_scores[candidates] - score, axis=1)))
            ]
        )
        matches.setdefault(nearest, []).append(str(label))
    return matches


def _prediction_intervals(
    atlas: pd.DataFrame,
    loto: pd.DataFrame,
) -> pd.DataFrame:
    for feature in AP_FEATURE_NAMES:
        residual = (
            loto[f"true_{feature}"].to_numpy(dtype=float)
            - loto[f"rna_{feature}"].to_numpy(dtype=float)
        )
        low, high = np.quantile(residual, (0.05, 0.95))
        atlas[f"{feature}_p05"] = atlas[feature] + low
        atlas[f"{feature}_p95"] = atlas[feature] + high
    return atlas


def _waveform_long(
    dataset: str,
    metadata: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for meta, row in zip(
        metadata.to_dict(orient="records"),
        parameters.to_dict(orient="records"),
    ):
        cycle = spike_cycle_from_row(row)
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "atlas_type": meta["atlas_type"],
                    "time_ms": cycle.time_ms,
                    "voltage_mv": cycle.voltage_mv,
                    "dvdt_mv_ms": cycle.velocity_mv_ms,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _plot_loto(
    dataset: str,
    loto: pd.DataFrame,
    performance: pd.DataFrame,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    axis = figure.add_subplot(grid[:, 0])
    ordered = performance.sort_values("rna_r2")
    y = np.arange(len(ordered))
    axis.barh(
        y - 0.18,
        ordered["class_only_r2"],
        height=0.34,
        color="#b8b8b8",
        label="Class only",
    )
    axis.barh(
        y + 0.18,
        ordered["rna_r2"],
        height=0.34,
        color="#3178a9",
        label="Class + RNA",
    )
    axis.set_yticks(y, [name.replace("_", " ") for name in ordered["feature"]])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Leave-one-t-type-out R2")
    axis.legend(frameon=False, loc="lower right")
    examples = (
        "ap_amplitude_mv",
        "ap_half_width_ms",
        "dvdt_span_mv_ms",
        "voltage_at_max_upstroke_mv",
    )
    for index, feature in enumerate(examples):
        axis = figure.add_subplot(grid[index // 2, 1 + index % 2])
        for broad_class, subset in loto.groupby("broad_class"):
            axis.scatter(
                subset[f"true_{feature}"],
                subset[f"rna_{feature}"],
                s=28 + 2 * np.sqrt(subset["cell_count"]),
                color=CLASS_COLORS.get(str(broad_class), "#777777"),
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
            )
        values = np.r_[loto[f"true_{feature}"], loto[f"rna_{feature}"]]
        lower, upper = np.nanquantile(values, (0.02, 0.98))
        axis.plot((lower, upper), (lower, upper), color="black", linewidth=0.8)
        r2 = performance.set_index("feature").loc[feature, "rna_r2"]
        axis.set_title(f"{feature.replace('_', ' ')}\nR2 = {r2:.2f}")
        axis.set_xlabel("Observed t-type mean")
        axis.set_ylabel("RNA prediction")
    figure.suptitle(
        f"{dataset}: transcriptomic extrapolation by held-out t-type",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_atlas_heatmap(
    dataset: str,
    atlas: pd.DataFrame,
    path: Path,
) -> None:
    supported = atlas.loc[atlas["class_supported"]].copy()
    class_order = {name: index for index, name in enumerate(CLASS_COLORS)}
    supported["class_order"] = supported["broad_class"].map(class_order).fillna(99)
    supported = supported.sort_values(["class_order", "cluster_id"])
    features = (
        "ap_amplitude_mv",
        "ap_half_width_ms",
        "max_upstroke_mv_ms",
        "max_downstroke_mv_ms",
        "dvdt_span_mv_ms",
        "voltage_at_max_upstroke_mv",
        "voltage_at_max_downstroke_mv",
        "upstroke_downstroke_ratio",
        "normalized_phase_loop_area",
    )
    values = supported.loc[:, features].to_numpy(dtype=float)
    values = (values - np.mean(values, axis=0)) / np.maximum(
        np.std(values, axis=0),
        1e-8,
    )
    values = np.clip(values, -2.5, 2.5)
    figure, (class_axis, axis) = plt.subplots(
        1,
        2,
        figsize=(11, 11),
        gridspec_kw={"width_ratios": (0.25, 10)},
    )
    class_rgba = np.asarray(
        [
            matplotlib.colors.to_rgba(CLASS_COLORS.get(str(value), "#777777"))
            for value in supported["broad_class"]
        ]
    )[:, None, :]
    class_axis.imshow(class_rgba, aspect="auto")
    class_axis.set_xticks([])
    class_axis.set_yticks([])
    image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axis.set_xticks(
        np.arange(len(features)),
        [feature.replace("_", " ") for feature in features],
        rotation=55,
        ha="right",
    )
    axis.set_yticks([])
    axis.set_ylabel(f"{len(supported)} supported neural reference t-types")
    axis.set_title(f"{dataset}: predicted AP feature profiles")
    figure.colorbar(image, ax=axis, label="Prediction z-score", shrink=0.55)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_waveform_gallery(
    dataset: str,
    atlas: pd.DataFrame,
    parameters: pd.DataFrame,
    path: Path,
    maximum_types: int = 8,
) -> None:
    eligible = atlas.loc[
        atlas["class_supported"]
        & ~atlas["directly_represented_by_patchseq_mapping"]
        & atlas["distance_ratio_to_loto_p95"].le(1.0)
    ].copy()
    selected = (
        eligible.sort_values("distance_ratio_to_loto_p95")
        .groupby("broad_class", as_index=False)
        .head(1)
        .head(maximum_types)
    )
    if selected.empty:
        return
    indices = selected.index.to_numpy(dtype=int)
    figure, axes = plt.subplots(2, len(indices), figsize=(3.1 * len(indices), 6.5))
    if len(indices) == 1:
        axes = np.asarray(axes)[:, None]
    for column, index in enumerate(indices):
        cycle = spike_cycle_from_row(parameters.iloc[index])
        color = CLASS_COLORS.get(str(atlas.iloc[index]["broad_class"]), "#777777")
        axes[0, column].plot(cycle.time_ms, cycle.voltage_mv, color=color, linewidth=2)
        axes[1, column].plot(
            cycle.voltage_mv,
            cycle.velocity_mv_ms,
            color=color,
            linewidth=2,
        )
        axes[1, column].axhline(0.0, color="#bbbbbb", linewidth=0.6)
        title = str(atlas.iloc[index]["leaf_label"])
        axes[0, column].set_title(
            f"{title}\n{atlas.iloc[index]['support_tier']} support",
            fontsize=9,
        )
        axes[0, column].set_xlabel("Time (ms)")
        axes[1, column].set_xlabel("V (mV)")
    axes[0, 0].set_ylabel("V (mV)")
    axes[1, 0].set_ylabel("dV/dt (mV/ms)")
    figure.suptitle(
        f"{dataset}: atlas-only reference t-type waveform predictions",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _extreme_table(atlas: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    eligible = atlas.loc[
        atlas["class_supported"]
        & ~atlas["directly_represented_by_patchseq_mapping"]
    ]
    rows = []
    for feature in AP_FEATURE_NAMES:
        ordered = eligible.sort_values(feature)
        for direction, subset in (("low", ordered.head(count)), ("high", ordered.tail(count))):
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "atlas_type": row["atlas_type"],
                        "leaf_label": row["leaf_label"],
                        "broad_class": row["broad_class"],
                        "predicted_value": row[feature],
                        "support_tier": row["support_tier"],
                        "distance_ratio_to_loto_p95": row[
                            "distance_ratio_to_loto_p95"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def _run_dataset(
    dataset: str,
    parameters: pd.DataFrame,
    args,
) -> dict[str, object]:
    output = args.output_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    expression, aligned = _load_expression(dataset, parameters, args)
    x_all = expression_matrix(expression)
    atlas_expression, atlas_metadata = load_allen_reference_atlas(
        args.atlas_expression,
        genes=expression.genes,
    )
    common_genes = np.asarray(
        [gene for gene in expression.genes.astype(str) if gene in atlas_expression.columns],
        dtype=str,
    )
    expression_index = {gene: index for index, gene in enumerate(expression.genes.astype(str))}
    x = x_all[:, [expression_index[gene] for gene in common_genes]]
    atlas_expression = atlas_expression.loc[:, common_genes]
    classes = _classes(expression)
    types = expression.transcriptomic_types.astype(str)
    y = waveform_target_matrix(aligned)
    y_raw = inverse_waveform_target_matrix(y)
    cell_features = waveform_features(waveform_parameter_frame(y_raw))
    eligibility = _waveform_qc_mask(cell_features)
    if int(np.sum(eligibility)) < 20:
        raise ValueError(f"Too few q50 AP-eligible cells for {dataset}")
    x = x[eligibility]
    classes = classes[eligibility]
    types = types[eligibility]
    donors = expression.donor_ids[eligibility]
    y = y[eligibility]
    y_raw = y_raw[eligibility]
    feature_raw = cell_features.loc[eligibility, AP_FEATURE_NAMES].to_numpy(dtype=float)
    feature_y = ap_feature_target_matrix(
        pd.DataFrame(feature_raw, columns=AP_FEATURE_NAMES)
    )

    analysis = fit_class_residualized_rrr_analysis(
        x,
        y,
        donors,
        classes,
        ranks=(1, 2, 3, 5, 8),
        penalties=(0.1, 1.0, 10.0, 100.0),
        sparse_ratios=(0.05, 0.1, 0.2),
        fold_count=args.folds,
    )
    feature_analysis = fit_class_residualized_rrr_analysis(
        x,
        feature_y,
        donors,
        classes,
        ranks=(1, 2, 3, 5, 8),
        penalties=(0.1, 1.0, 10.0, 100.0),
        sparse_ratios=(0.05, 0.1, 0.2),
        fold_count=args.folds,
    )
    type_counts = pd.Series(types).value_counts()
    benchmark_types = sorted(
        type_counts.loc[type_counts >= args.minimum_type_cells].index.astype(str)
    )
    distance_model = fit_distance_model(x, types, classes)
    loto_path = output / "loto_ttype_predictions.csv"
    if args.reuse_loto and loto_path.exists():
        loto = pd.read_csv(loto_path)
        performance = _feature_performance(loto)
        waveform_performance = _waveform_feature_performance(loto)
    else:
        results = Parallel(n_jobs=args.jobs, prefer="threads")(
            delayed(_loto_fold)(
                heldout_type,
                x,
                y,
                y_raw,
                types,
                classes,
                analysis.rank,
                analysis.ridge_penalty,
                analysis.sparse_ratio,
            )
            for heldout_type in benchmark_types
        )
        direct_results = Parallel(n_jobs=args.jobs, prefer="threads")(
            delayed(_loto_feature_fold)(
                heldout_type,
                x,
                feature_y,
                feature_raw,
                types,
                classes,
                feature_analysis.rank,
                feature_analysis.ridge_penalty,
                feature_analysis.sparse_ratio,
            )
            for heldout_type in benchmark_types
        )
        loto = _loto_frame(results, direct_results)
        loto = loto.merge(
            _loto_distance_table(distance_model),
            on="transcriptomic_type",
        )
        performance = _feature_performance(loto)
        waveform_performance = _waveform_feature_performance(loto)

    harmonized_atlas = harmonize_atlas_to_patchseq(
        atlas_expression,
        x,
        types,
        common_genes,
    )
    atlas_classes = atlas_metadata["broad_class"].to_numpy(dtype=object)
    predicted, class_only = predict_class_residualized_rrr(
        analysis,
        x,
        y,
        classes,
        harmonized_atlas,
        atlas_classes,
    )
    predicted_raw = inverse_waveform_target_matrix(predicted)
    class_raw = inverse_waveform_target_matrix(class_only)
    predicted_raw, clipped = clip_waveform_parameters(predicted_raw, y_raw)
    class_raw, class_clipped = clip_waveform_parameters(class_raw, y_raw)
    predicted_parameters = waveform_parameter_frame(predicted_raw)
    class_parameters = waveform_parameter_frame(class_raw)
    waveform_atlas_features = waveform_features(predicted_parameters).add_prefix(
        "waveform_derived_"
    )
    waveform_class_features = waveform_features(class_parameters).add_prefix(
        "waveform_class_only_"
    )
    direct_predicted, direct_class = predict_class_residualized_rrr(
        feature_analysis,
        x,
        feature_y,
        classes,
        harmonized_atlas,
        atlas_classes,
    )
    direct_raw = inverse_ap_feature_target_matrix(direct_predicted)
    direct_class_raw = inverse_ap_feature_target_matrix(direct_class)
    direct_raw, direct_clipped = clip_ap_features(direct_raw, feature_raw)
    direct_class_raw, direct_class_clipped = clip_ap_features(
        direct_class_raw,
        feature_raw,
    )
    atlas_features = pd.DataFrame(direct_raw, columns=AP_FEATURE_NAMES)
    class_features = pd.DataFrame(
        direct_class_raw,
        columns=[f"class_only_{name}" for name in AP_FEATURE_NAMES],
    )
    support = transcriptomic_support(distance_model, harmonized_atlas, atlas_classes)
    atlas = pd.concat(
        (
            atlas_metadata.reset_index(drop=True),
            support,
            atlas_features,
            class_features,
            waveform_atlas_features,
            waveform_class_features,
        ),
        axis=1,
    )
    atlas["waveform_parameter_clipped"] = clipped
    atlas["class_waveform_parameter_clipped"] = class_clipped
    atlas["direct_feature_clipped"] = direct_clipped
    atlas["class_direct_feature_clipped"] = direct_class_clipped
    mappings = _direct_atlas_mapping(distance_model, harmonized_atlas, atlas_metadata)
    atlas["mapped_patchseq_types"] = ["; ".join(mappings.get(i, [])) for i in range(len(atlas))]
    atlas["directly_represented_by_patchseq_mapping"] = atlas[
        "mapped_patchseq_types"
    ].ne("")
    atlas["atlas_only"] = ~atlas["directly_represented_by_patchseq_mapping"]

    if len(loto) >= 4:
        isotonic = IsotonicRegression(out_of_bounds="clip", increasing=True)
        isotonic.fit(
            loto["distance_ratio_to_loto_p95"],
            loto["rna_feature_nrmse"],
        )
        atlas["expected_ap_feature_nrmse"] = isotonic.predict(
            atlas["distance_ratio_to_loto_p95"]
        )
    else:
        atlas["expected_ap_feature_nrmse"] = float("nan")
    atlas = _prediction_intervals(atlas, loto)

    atlas_parameters = pd.concat(
        (atlas_metadata[["atlas_type"]].reset_index(drop=True), predicted_parameters),
        axis=1,
    )
    selected_genes = pd.DataFrame(
        {
            "gene": common_genes,
            "waveform_selected": analysis.selected_predictors,
            "waveform_weight_norm": np.linalg.norm(
                analysis.factors.predictor_weights,
                axis=1,
            ),
            "feature_selected": feature_analysis.selected_predictors,
            "feature_weight_norm": np.linalg.norm(
                feature_analysis.factors.predictor_weights,
                axis=1,
            ),
        }
    ).sort_values(
        ["feature_selected", "feature_weight_norm"],
        ascending=[False, False],
    )
    _write_csv(loto, output / "loto_ttype_predictions.csv")
    _write_csv(performance, output / "loto_feature_performance.csv")
    _write_csv(
        waveform_performance,
        output / "loto_waveform_derived_feature_performance.csv",
    )
    _write_csv(atlas, output / "atlas_neural_ttype_ap_predictions.csv")
    _write_csv(atlas_parameters, output / "atlas_neural_ttype_waveform_parameters.csv")
    _write_csv(
        _waveform_long(dataset, atlas_metadata, predicted_parameters),
        output / "atlas_neural_ttype_waveforms.csv",
    )
    _write_csv(_extreme_table(atlas), output / "atlas_only_feature_extremes.csv")
    _write_csv(selected_genes, output / "selected_genes.csv")
    _write_csv(analysis.ridge_cv, output / "ridge_cv.csv")
    _write_csv(analysis.sparse_cv, output / "sparse_cv.csv")
    _write_csv(feature_analysis.ridge_cv, output / "feature_ridge_cv.csv")
    _write_csv(feature_analysis.sparse_cv, output / "feature_sparse_cv.csv")
    _plot_loto(dataset, loto, performance, output / "loto_extrapolation_benchmark.png")
    _plot_atlas_heatmap(dataset, atlas, output / "atlas_predicted_ap_features.png")
    _plot_waveform_gallery(
        dataset,
        atlas,
        predicted_parameters,
        output / "atlas_only_waveform_gallery.png",
    )

    summary = {
        "dataset": dataset,
        "paired_patchseq_cells": len(aligned),
        "q50_ap_eligible_patchseq_cells": int(np.sum(eligibility)),
        "patchseq_ttypes": int(len(np.unique(types))),
        "loto_benchmark_ttypes": len(loto),
        "common_genes": len(common_genes),
        "selected_genes": int(np.sum(analysis.selected_predictors)),
        "selected_rank": analysis.rank,
        "ridge_penalty": analysis.ridge_penalty,
        "sparse_ratio": analysis.sparse_ratio,
        "feature_selected_genes": int(np.sum(feature_analysis.selected_predictors)),
        "feature_selected_rank": feature_analysis.rank,
        "feature_ridge_penalty": feature_analysis.ridge_penalty,
        "feature_sparse_ratio": feature_analysis.sparse_ratio,
        "donor_oof_class_only_r2": analysis.class_only_oof_r2,
        "donor_oof_class_plus_rna_r2": analysis.combined_oof_r2,
        "feature_donor_oof_class_only_r2": feature_analysis.class_only_oof_r2,
        "feature_donor_oof_class_plus_rna_r2": feature_analysis.combined_oof_r2,
        "atlas_neural_ttypes": len(atlas),
        "atlas_supported_class_ttypes": int(np.sum(atlas["class_supported"])),
        "atlas_directly_represented_mapped_ttypes": int(
            np.sum(atlas["directly_represented_by_patchseq_mapping"])
        ),
        "atlas_only_supported_ttypes": int(
            np.sum(atlas["atlas_only"] & atlas["class_supported"])
        ),
        "atlas_far_or_unsupported_ttypes": int(
            np.sum(atlas["support_tier"].isin(("far", "unsupported_class")))
        ),
        "median_loto_rna_feature_nrmse": float(loto["rna_feature_nrmse"].median()),
        "median_loto_class_feature_nrmse": float(
            loto["class_feature_nrmse"].median()
        ),
        "median_feature_rna_r2": float(performance["rna_r2"].median()),
        "median_feature_class_only_r2": float(
            performance["class_only_r2"].median()
        ),
        "median_feature_rna_delta_r2_over_class": float(
            performance["rna_delta_r2_over_class"].median()
        ),
        "median_waveform_derived_feature_rna_r2": float(
            waveform_performance["rna_r2"].median()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="ascii",
    )
    return summary


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(
            "outputs/recovery_onset_targets_all/recovery_onset_parameters.csv"
        ),
    )
    parser.add_argument(
        "--atlas-expression",
        type=Path,
        default=Path("data/external/allen_ctx_hpf_smartseq_trimmed_means.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/atlas_ttype_ephys_prediction"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gouwens_visp", "scala_room_temperature"),
        default=("gouwens_visp", "scala_room_temperature"),
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--minimum-type-cells", type=int, default=3)
    parser.add_argument(
        "--reuse-loto",
        action="store_true",
        help="Reuse existing leave-one-t-type-out tables while refitting the full model.",
    )
    parser.add_argument(
        "--expression-cache",
        type=Path,
        default=Path("outputs/transcriptomic_rrr/expression_cache"),
    )
    parser.add_argument(
        "--gouwens-mat",
        type=Path,
        default=DEFAULT_KOBAK_ROOT
        / "data/gouwens2020/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat",
    )
    parser.add_argument(
        "--gouwens-metadata",
        type=Path,
        default=DEFAULT_PATCHSEQ_ROOT
        / "Patch-seq AIBS/20200711_patchseq_metadata_mouse.csv",
    )
    parser.add_argument(
        "--gouwens-cpm",
        type=Path,
        default=DEFAULT_PATCHSEQ_ROOT
        / "Patch-seq AIBS/transcriptomes/20200513_Mouse_PatchSeq_Release_cpm.v2/"
        "20200513_Mouse_PatchSeq_Release_cpm.v2.csv",
    )
    parser.add_argument(
        "--scala-pickle",
        type=Path,
        default=DEFAULT_KOBAK_ROOT / "data/scala2020.pickle",
    )
    parser.add_argument(
        "--scala-metadata",
        type=Path,
        default=DEFAULT_PATCHSEQ_ROOT / "Scala_pseq_data/m1_patchseq_meta_data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = pd.read_csv(args.parameters, dtype={"cell_id": "string"})
    summaries = [_run_dataset(dataset, parameters, args) for dataset in args.datasets]
    _write_csv(pd.DataFrame(summaries), args.output_root / "summary.csv")
    metadata = {
        "atlas_expression": str(args.atlas_expression),
        "waveform_current_fraction": 0.5,
        "minimum_patchseq_cells_per_loto_type": args.minimum_type_cells,
        "datasets": list(args.datasets),
        "interpretation": (
            "Atlas-only values are predictions. Extrapolation performance is measured "
            "only by leave-one-Patch-seq-t-type-out validation."
        ),
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
