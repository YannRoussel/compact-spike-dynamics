#!/usr/bin/env python3
"""Class-residualized and within-class RNA prediction of compact parameters."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.model_transcriptomics import (
    build_model_target_matrix,
    high_qc_mask,
    inverse_model_target_matrix,
)
from inverse_ephys_alpha_beta.reduced_rank import (
    fit_class_residualized_rrr_analysis,
    fit_rrr_analysis,
    nested_class_residualized_rrr_evaluation,
    residualize_class_fold,
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


def _subset_expression(
    expression: PatchSeqExpression,
    mask: np.ndarray,
) -> PatchSeqExpression:
    return PatchSeqExpression(
        dataset=expression.dataset,
        cell_ids=expression.cell_ids[mask],
        genes=expression.genes,
        values=expression.values[mask],
        donor_ids=expression.donor_ids[mask],
        transcriptomic_types=expression.transcriptomic_types[mask],
        already_log_normalized=expression.already_log_normalized,
        library_size=(
            expression.library_size[mask]
            if expression.library_size is not None
            else None
        ),
        source_cell_count=expression.source_cell_count,
    )


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
                Path(args.expression_cache)
                / "gouwens_scn_kcn_expression.npz"
            ),
        )
    return load_scala_expression(
        args.scala_pickle,
        args.scala_metadata,
        include_channel_genes=True,
    )


def _parameter_performance(
    target,
    actual: np.ndarray,
    combined: np.ndarray,
    class_only: np.ndarray,
) -> pd.DataFrame:
    actual_raw = inverse_model_target_matrix(actual, target)
    combined_raw = inverse_model_target_matrix(combined, target)
    class_raw = inverse_model_target_matrix(class_only, target)
    rows = []
    for index, name in enumerate(target.names):
        centered = actual_raw[:, index] - np.mean(actual_raw[:, index])
        denominator = float(np.sum(centered**2))
        class_error = float(
            np.sum((actual_raw[:, index] - class_raw[:, index]) ** 2)
        )
        combined_error = float(
            np.sum((actual_raw[:, index] - combined_raw[:, index]) ** 2)
        )
        class_r2 = (
            1.0
            - class_error / denominator
            if denominator > 0.0
            else np.nan
        )
        combined_r2 = (
            1.0
            - combined_error / denominator
            if denominator > 0.0
            else np.nan
        )
        rows.append(
            {
                "parameter": name,
                "class_only_oof_r2": class_r2,
                "class_plus_rna_oof_r2": combined_r2,
                "delta_oof_r2": combined_r2 - class_r2,
                "incremental_residual_r2": (
                    1.0 - combined_error / class_error
                    if class_error > 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "delta_oof_r2",
        ascending=False,
    )


def _selected_gene_table(
    genes: np.ndarray,
    analysis,
) -> pd.DataFrame:
    rows = []
    for index in np.flatnonzero(analysis.selected_predictors):
        gene = str(genes[index])
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


def _within_class_analyses(
    x: np.ndarray,
    y: np.ndarray,
    expression: PatchSeqExpression,
    classes: np.ndarray,
    minimum_cells: int,
    fold_count: int,
    output_root: Path,
) -> pd.DataFrame:
    rows = []
    for class_name in sorted(set(classes)):
        mask = classes == class_name
        donor_count = pd.Series(
            expression.donor_ids[mask]
        ).nunique(dropna=True)
        if np.sum(mask) < minimum_cells or donor_count < 3:
            continue
        analysis = fit_rrr_analysis(
            x[mask],
            y[mask],
            expression.donor_ids[mask],
            fold_count=min(fold_count, int(donor_count)),
        )
        class_root = output_root / "within_class" / class_name
        class_root.mkdir(parents=True, exist_ok=True)
        _write_csv(analysis.ridge_cv, class_root / "ridge_cv.csv")
        _write_csv(analysis.sparse_cv, class_root / "sparse_cv.csv")
        genes = _selected_gene_table(expression.genes, analysis)
        _write_csv(genes, class_root / "selected_genes.csv")
        rows.append(
            {
                "class": class_name,
                "cell_count": int(np.sum(mask)),
                "donor_count": int(donor_count),
                "cv_replay_r2": analysis.oof_r2,
                "rank": analysis.rank,
                "selected_gene_count": int(
                    np.sum(analysis.selected_predictors)
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_plot(
    classes: np.ndarray,
    performance: pd.DataFrame,
    summary: dict[str, object],
    within_class: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.5),
        constrained_layout=True,
    )
    counts = pd.Series(classes).value_counts().sort_values()
    axes[0, 0].barh(counts.index, counts.values, color="#6f7c85")
    axes[0, 0].set_xlabel("Cells")
    axes[0, 0].set_title("Broad transcriptomic classes", loc="left")

    values = (
        summary["class_only_nested_oof_r2"],
        summary["class_plus_rna_nested_oof_r2"],
    )
    axes[0, 1].bar(
        ("class only", "class + RNA"),
        values,
        color=("#8f9ba3", "#0072b2"),
    )
    axes[0, 1].set_ylabel("Nested donor-held-out R2")
    axes[0, 1].set_title(
        f"Delta R2 = {summary['delta_nested_oof_r2']:.3f}",
        loc="left",
    )

    axes[1, 0].scatter(
        performance["class_only_oof_r2"],
        performance["class_plus_rna_oof_r2"],
        s=18,
        alpha=0.65,
        color="#009e73",
        edgecolors="none",
    )
    limits = (-0.5, 0.9)
    axes[1, 0].plot(limits, limits, color="#999999", linewidth=1)
    axes[1, 0].set_xlim(limits)
    axes[1, 0].set_ylim(limits)
    axes[1, 0].set_xlabel("Class-only parameter R2")
    axes[1, 0].set_ylabel("Class + RNA parameter R2")
    axes[1, 0].set_title("Nested per-parameter performance", loc="left")
    for parameter in (
        "timing_tau_ms",
        "timing_spike_jump",
        "rheobase_current_pa",
    ):
        row = performance.loc[performance["parameter"].eq(parameter)]
        if len(row):
            axes[1, 0].annotate(
                parameter.replace("_", " "),
                (
                    row["class_only_oof_r2"].iloc[0],
                    row["class_plus_rna_oof_r2"].iloc[0],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    if len(within_class):
        shown = within_class.sort_values("cv_replay_r2")
        axes[1, 1].barh(
            shown["class"],
            shown["cv_replay_r2"],
            color="#d55e00",
        )
        axes[1, 1].axvline(0.0, color="#999999", linewidth=0.8)
    axes[1, 1].set_xlabel("Donor-held-out CV replay R2")
    axes[1, 1].set_title("Separate within-class RRR", loc="left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{summary['dataset'].replace('_', ' ')} | {summary['cohort']}"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _run_cohort(
    dataset: str,
    cohort: str,
    expression: PatchSeqExpression,
    parameters: pd.DataFrame,
    output_root: Path,
    args,
) -> dict[str, object]:
    cohort_root = output_root / dataset / cohort
    cohort_root.mkdir(parents=True, exist_ok=True)
    target = build_model_target_matrix(parameters)
    x = expression_matrix(expression)
    y = target.values
    classes = np.asarray(
        [
            broad_transcriptomic_class(label)
            for label in expression.transcriptomic_types
        ],
        dtype=object,
    )
    final = fit_class_residualized_rrr_analysis(
        x,
        y,
        expression.donor_ids,
        classes,
        sparse_ratios=(0.1,),
        fold_count=args.folds,
    )
    nested = nested_class_residualized_rrr_evaluation(
        x,
        y,
        expression.donor_ids,
        classes,
        ranks=(2, 5, 8),
        penalties=(0.1, 1.0, 10.0),
        sparse_ratios=(),
        outer_fold_count=args.folds,
        inner_fold_count=args.inner_folds,
    )
    performance = _parameter_performance(
        target,
        y,
        nested.oof_prediction,
        nested.class_only_oof_prediction,
    )
    genes = _selected_gene_table(expression.genes, final)
    within_class = _within_class_analyses(
        x,
        y,
        expression,
        classes,
        args.within_class_min_cells,
        args.folds,
        cohort_root,
    )
    class_counts = (
        pd.DataFrame(
            {
                "class": classes,
                "donor_id": expression.donor_ids,
            }
        )
        .groupby("class", as_index=False)
        .agg(
            cell_count=("class", "size"),
            donor_count=("donor_id", "nunique"),
        )
    )
    _write_csv(class_counts, cohort_root / "class_counts.csv")
    _write_csv(final.ridge_cv, cohort_root / "ridge_cv.csv")
    _write_csv(final.sparse_cv, cohort_root / "sparse_cv.csv")
    _write_csv(
        nested.fold_hyperparameters,
        cohort_root / "nested_fold_hyperparameters.csv",
    )
    _write_csv(
        performance,
        cohort_root / "parameter_nested_oof_performance.csv",
    )
    _write_csv(genes, cohort_root / "selected_genes.csv")
    _write_csv(
        within_class,
        cohort_root / "within_class_summary.csv",
    )
    cell_predictions = pd.DataFrame(
        {
            "cell_id": expression.cell_ids,
            "donor_id": expression.donor_ids,
            "transcriptomic_type": expression.transcriptomic_types,
            "broad_class": classes,
        }
    )
    _write_csv(
        cell_predictions,
        cohort_root / "cell_metadata.csv",
    )
    x_residual, _, _ = residualize_class_fold(
        x,
        x,
        classes,
        classes,
    )
    x_scaled = np.clip(
        (x_residual - final.x_mean) / final.x_scale,
        -8.0,
        8.0,
    )
    components = x_scaled @ final.factors.predictor_weights
    component_frame = cell_predictions.copy()
    for index in range(final.rank):
        component_frame[f"rna_component_{index + 1}"] = components[:, index]
    _write_csv(
        component_frame,
        cohort_root / "within_class_components.csv",
    )
    summary = {
        "dataset": dataset,
        "cohort": cohort,
        "cell_count": len(parameters),
        "donor_count": int(
            pd.Series(expression.donor_ids).nunique(dropna=True)
        ),
        "broad_class_count": int(len(set(classes))),
        "selected_rank_full_fit": final.rank,
        "selected_gene_count_full_fit": int(
            np.sum(final.selected_predictors)
        ),
        "class_only_nested_oof_r2": nested.class_only_oof_r2,
        "class_plus_rna_nested_oof_r2": nested.combined_oof_r2,
        "delta_nested_oof_r2": nested.delta_oof_r2,
        "incremental_residual_nested_r2": (
            nested.incremental_residual_r2
        ),
        "class_residualized_cv_replay_r2": (
            final.incremental_residual_r2
        ),
    }
    (cohort_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="ascii",
    )
    _summary_plot(
        classes,
        performance,
        summary,
        within_class,
        cohort_root / "class_aware_rrr_summary.png",
    )
    return summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--parameters",
        default=(
            "outputs/compact_population_all/"
            "compact_model_parameters.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/class_aware_rrr",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gouwens_visp", "scala_room_temperature"),
        default=("gouwens_visp", "scala_room_temperature"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--within-class-min-cells", type=int, default=60)
    parser.add_argument(
        "--expression-cache",
        default="outputs/transcriptomic_rrr/expression_cache",
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
    args = parser.parse_args()

    parameter_frame = pd.read_csv(
        args.parameters,
        dtype={"cell_id": "string"},
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in args.datasets:
        requested = set(
            parameter_frame.loc[
                parameter_frame["dataset"].eq(dataset),
                "cell_id",
            ].astype(str)
        )
        expression = _load_expression(dataset, args, requested)
        aligned_expression, aligned_parameters = (
            align_expression_to_parameters(expression, parameter_frame)
        )
        cohorts = {
            "all_eligible": np.ones(
                len(aligned_parameters),
                dtype=bool,
            ),
            "high_qc": high_qc_mask(aligned_parameters),
        }
        for cohort, mask in cohorts.items():
            if np.sum(mask) < args.minimum_cells:
                continue
            print(
                f"Fitting {dataset}/{cohort}: {int(np.sum(mask))} cells",
                flush=True,
            )
            summaries.append(
                _run_cohort(
                    dataset,
                    cohort,
                    _subset_expression(aligned_expression, mask),
                    aligned_parameters.loc[mask].reset_index(drop=True),
                    output_root,
                    args,
                )
            )
    summary_frame = pd.DataFrame(summaries)
    _write_csv(summary_frame, output_root / "class_aware_summary.csv")
    (output_root / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "parameters": args.parameters,
                "outer_folds": args.folds,
                "inner_folds": args.inner_folds,
                "residualization": (
                    "RNA and compact parameters residualized by broad "
                    "class using training-fold means only"
                ),
                "primary_metric": (
                    "nested donor-held-out delta R2 over class-only"
                ),
                "nested_prediction_model": (
                    "ridge RRR; sparse RRR is fitted separately for "
                    "descriptive gene selection"
                ),
                "descriptive_sparse_ratio": 0.1,
                "nested_hyperparameter_grid": {
                    "ranks": [2, 5, 8],
                    "ridge_penalties": [0.1, 1.0, 10.0],
                },
                "within_class_note": (
                    "separate class fits are CV replay summaries"
                ),
            },
            indent=2,
        ),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
