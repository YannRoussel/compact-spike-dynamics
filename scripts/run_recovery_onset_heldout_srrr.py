#!/usr/bin/env python3
"""Predict held-out recovery-onset models from Patch-seq expression."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.model_transcriptomics import (
    build_model_target_matrix,
    inverse_model_target_matrix,
    transform_model_target_matrix,
)
from inverse_ephys_alpha_beta.recovery_onset_compact import (
    recovery_onset_model_from_row,
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
    is_kcn_gene,
    is_scn_gene,
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_expression(
    dataset: str,
    args,
    requested_cell_ids: set[str],
) -> PatchSeqExpression:
    if dataset == "gouwens_visp":
        return load_gouwens_expression(
            args.gouwens_mat,
            args.gouwens_metadata,
            channel_cpm_path=args.gouwens_cpm,
            requested_cell_ids=requested_cell_ids,
            channel_cache_path=(
                args.expression_cache
                / "gouwens_scn_kcn_expression.npz"
            ),
        )
    return load_scala_expression(
        args.scala_pickle,
        args.scala_metadata,
        include_channel_genes=True,
    )


def _classes(expression: PatchSeqExpression) -> np.ndarray:
    return np.asarray(
        [
            broad_transcriptomic_class(label)
            for label in expression.transcriptomic_types
        ],
        dtype=object,
    )


def _standardized_r2(
    actual: np.ndarray,
    prediction: np.ndarray,
    training: np.ndarray,
) -> float:
    mean = np.mean(training, axis=0)
    scale = np.maximum(np.std(training, axis=0), 1e-8)
    actual_scaled = (actual - mean) / scale
    prediction_scaled = (prediction - mean) / scale
    denominator = float(np.sum(np.square(actual_scaled)))
    if denominator <= 0.0:
        return float("nan")
    return float(
        1.0
        - np.sum(np.square(actual_scaled - prediction_scaled))
        / denominator
    )


def _clip_raw_predictions(
    prediction: np.ndarray,
    training_raw: np.ndarray,
) -> tuple[np.ndarray, int]:
    lower = np.quantile(training_raw, 0.005, axis=0)
    upper = np.quantile(training_raw, 0.995, axis=0)
    clipped = np.clip(prediction, lower, upper)
    return clipped, int(np.sum(~np.isclose(clipped, prediction)))


def _prediction_frame(
    heldout: pd.DataFrame,
    names: tuple[str, ...],
    values: np.ndarray,
) -> pd.DataFrame:
    metadata = heldout[
        [
            "dataset",
            "cell_id",
            "nwb_path",
            "validation_current_pa",
            "validation_sweep_number",
        ]
    ].reset_index(drop=True)
    parameters = pd.DataFrame(values, columns=names)
    result = pd.concat((metadata, parameters), axis=1)
    result["repetitive_current_max_pa"] = (
        result["repetitive_current_min_pa"]
        + result["repetitive_current_span_pa"]
    )
    result["recovery_voltage_reference_mv"] = result[
        "resting_voltage_mv"
    ]
    result["recovery_voltage_scale_mv"] = 100.0
    for _, row in result.iterrows():
        recovery_onset_model_from_row(row)
    return result


def _selected_genes(
    expression: PatchSeqExpression,
    analysis,
) -> pd.DataFrame:
    rows = []
    for index in np.flatnonzero(analysis.selected_predictors):
        gene = str(expression.genes[index])
        rows.append(
            {
                "gene": gene,
                "weight_norm": float(
                    np.linalg.norm(
                        analysis.factors.predictor_weights[index]
                    )
                ),
                "is_scn_family": is_scn_gene(gene),
                "is_kcn_family": is_kcn_gene(gene),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=(
                "gene",
                "weight_norm",
                "is_scn_family",
                "is_kcn_family",
            )
        )
    return pd.DataFrame(rows).sort_values(
        "weight_norm",
        ascending=False,
    )


def _run_dataset(
    dataset: str,
    parameters: pd.DataFrame,
    representatives: pd.DataFrame,
    output_root: Path,
    args,
) -> dict[str, object]:
    requested = set(
        parameters.loc[
            parameters["dataset"].eq(dataset),
            "cell_id",
        ].astype(str)
    )
    expression = _load_expression(dataset, args, requested)
    expression, aligned = align_expression_to_parameters(
        expression,
        parameters,
    )
    x = expression_matrix(expression)
    classes = _classes(expression)
    heldout_ids = set(
        representatives.loc[
            representatives["dataset"].eq(dataset),
            "cell_id",
        ].astype(str)
    )
    holdout_mask = aligned["cell_id"].astype(str).isin(heldout_ids).to_numpy()
    if int(np.sum(holdout_mask)) != len(heldout_ids):
        missing = heldout_ids.difference(
            aligned.loc[holdout_mask, "cell_id"].astype(str)
        )
        raise ValueError(
            f"Missing held-out {dataset} cells: {sorted(missing)}"
        )
    training_mask = ~holdout_mask
    training_parameters = aligned.loc[training_mask].reset_index(drop=True)
    heldout_parameters = aligned.loc[holdout_mask].reset_index(drop=True)
    target = build_model_target_matrix(training_parameters)
    y_training = target.values
    y_heldout = transform_model_target_matrix(
        heldout_parameters,
        target,
    )
    analysis = fit_class_residualized_rrr_analysis(
        x[training_mask],
        y_training,
        expression.donor_ids[training_mask],
        classes[training_mask],
        ranks=(1, 2, 3, 5, 8),
        penalties=(0.01, 0.1, 1.0, 10.0, 100.0),
        sparse_ratios=(0.05, 0.1, 0.2),
        fold_count=args.folds,
    )
    predicted, class_only = predict_class_residualized_rrr(
        analysis,
        x[training_mask],
        y_training,
        classes[training_mask],
        x[holdout_mask],
        classes[holdout_mask],
    )
    training_raw = inverse_model_target_matrix(y_training, target)
    heldout_raw = inverse_model_target_matrix(y_heldout, target)
    predicted_raw = inverse_model_target_matrix(predicted, target)
    class_raw = inverse_model_target_matrix(class_only, target)
    predicted_raw, clipped_count = _clip_raw_predictions(
        predicted_raw,
        training_raw,
    )
    class_raw, class_clipped_count = _clip_raw_predictions(
        class_raw,
        training_raw,
    )
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    ephys = heldout_parameters.copy()
    rna = _prediction_frame(
        heldout_parameters,
        target.names,
        predicted_raw,
    )
    class_prediction = _prediction_frame(
        heldout_parameters,
        target.names,
        class_raw,
    )
    _write_csv(ephys, dataset_root / "heldout_ephys_parameters.csv")
    _write_csv(rna, dataset_root / "heldout_rna_parameters.csv")
    _write_csv(
        class_prediction,
        dataset_root / "heldout_class_only_parameters.csv",
    )
    _write_csv(analysis.ridge_cv, dataset_root / "ridge_cv.csv")
    _write_csv(analysis.sparse_cv, dataset_root / "sparse_cv.csv")
    _write_csv(
        _selected_genes(expression, analysis),
        dataset_root / "selected_genes.csv",
    )
    np.savez_compressed(
        dataset_root / "srrr_model.npz",
        predictor_weights=analysis.factors.predictor_weights,
        response_weights=analysis.factors.response_weights,
        selected_predictors=analysis.selected_predictors,
        x_mean=analysis.x_mean,
        x_scale=analysis.x_scale,
        y_mean=analysis.y_mean,
        y_scale=analysis.y_scale,
        genes=expression.genes.astype(str),
        target_names=np.asarray(target.names, dtype=str),
        target_log1p=np.asarray(target.log1p_transformed, dtype=bool),
    )
    cell_metadata = pd.DataFrame(
        {
            "cell_id": expression.cell_ids,
            "donor_id": expression.donor_ids,
            "transcriptomic_type": expression.transcriptomic_types,
            "broad_class": classes,
            "heldout": holdout_mask,
        }
    )
    _write_csv(cell_metadata, dataset_root / "cell_metadata.csv")
    per_cell = cell_metadata.loc[holdout_mask].reset_index(drop=True)
    scale = np.maximum(np.std(training_raw, axis=0), 1e-8)
    per_cell["rna_parameter_nrmse"] = np.sqrt(
        np.mean(np.square((heldout_raw - predicted_raw) / scale), axis=1)
    )
    per_cell["class_parameter_nrmse"] = np.sqrt(
        np.mean(np.square((heldout_raw - class_raw) / scale), axis=1)
    )
    _write_csv(per_cell, dataset_root / "heldout_parameter_errors.csv")
    summary = {
        "dataset": dataset,
        "paired_cell_count": len(aligned),
        "training_cell_count": int(np.sum(training_mask)),
        "heldout_cell_count": int(np.sum(holdout_mask)),
        "target_parameter_count": len(target.names),
        "gene_count": len(expression.genes),
        "selected_rank": analysis.rank,
        "ridge_penalty": analysis.ridge_penalty,
        "sparse_ratio": analysis.sparse_ratio,
        "selected_gene_count": int(np.sum(analysis.selected_predictors)),
        "training_class_only_oof_r2": analysis.class_only_oof_r2,
        "training_class_plus_rna_oof_r2": analysis.combined_oof_r2,
        "training_delta_oof_r2": analysis.delta_oof_r2,
        "training_incremental_residual_r2": (
            analysis.incremental_residual_r2
        ),
        "heldout_class_only_r2": _standardized_r2(
            y_heldout,
            class_only,
            y_training,
        ),
        "heldout_class_plus_rna_r2": _standardized_r2(
            y_heldout,
            predicted,
            y_training,
        ),
        "heldout_median_rna_parameter_nrmse": float(
            per_cell["rna_parameter_nrmse"].median()
        ),
        "heldout_median_class_parameter_nrmse": float(
            per_cell["class_parameter_nrmse"].median()
        ),
        "clipped_rna_parameter_count": clipped_count,
        "clipped_class_parameter_count": class_clipped_count,
    }
    (dataset_root / "summary.json").write_text(
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
            "outputs/recovery_onset_targets_all/"
            "recovery_onset_parameters.csv"
        ),
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        default=Path(
            "outputs/recovery_onset_family_overlays/"
            "representative_cells.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recovery_onset_heldout_srrr"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--expression-cache",
        type=Path,
        default=Path("outputs/transcriptomic_rrr/expression_cache"),
    )
    parser.add_argument(
        "--gouwens-mat",
        type=Path,
        default=(
            DEFAULT_KOBAK_ROOT
            / "data/gouwens2020/"
            "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
        ),
    )
    parser.add_argument(
        "--gouwens-metadata",
        type=Path,
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Patch-seq AIBS/20200711_patchseq_metadata_mouse.csv"
        ),
    )
    parser.add_argument(
        "--gouwens-cpm",
        type=Path,
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Patch-seq AIBS/transcriptomes/"
            "20200513_Mouse_PatchSeq_Release_cpm.v2/"
            "20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
        ),
    )
    parser.add_argument(
        "--scala-pickle",
        type=Path,
        default=DEFAULT_KOBAK_ROOT / "data/scala2020.pickle",
    )
    parser.add_argument(
        "--scala-metadata",
        type=Path,
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Scala_pseq_data/m1_patchseq_meta_data.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    parameters = pd.read_csv(
        args.parameters,
        dtype={"cell_id": "string"},
    )
    representatives = pd.read_csv(
        args.representatives,
        dtype={"cell_id": "string"},
    )
    summaries = [
        _run_dataset(
            dataset,
            parameters,
            representatives,
            args.output_root,
            args,
        )
        for dataset in ("gouwens_visp", "scala_room_temperature")
    ]
    _write_csv(
        pd.DataFrame(summaries),
        args.output_root / "summary.csv",
    )


if __name__ == "__main__":
    main()
