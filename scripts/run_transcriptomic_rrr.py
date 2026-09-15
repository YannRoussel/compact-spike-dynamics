#!/usr/bin/env python3
"""Relate compact-model parameters to paired Patch-seq gene expression."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom

from inverse_ephys_alpha_beta.model_transcriptomics import (
    build_model_target_matrix,
    high_qc_mask,
    inverse_model_target_matrix,
)
from inverse_ephys_alpha_beta.reduced_rank import (
    correlation_loadings,
    fit_rrr_analysis,
)
from inverse_ephys_alpha_beta.transcriptomics import (
    align_expression_to_parameters,
    expression_matrix,
    load_gouwens_expression,
    load_scala_expression,
    is_kcn_gene,
    is_scn_gene,
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
    output_root: Path,
):
    if dataset == "gouwens_visp":
        return load_gouwens_expression(
            args.gouwens_mat,
            args.gouwens_metadata,
            channel_cpm_path=(
                None if args.no_force_channel_genes else args.gouwens_cpm
            ),
            requested_cell_ids=requested_cell_ids,
            channel_cache_path=(
                output_root
                / "expression_cache/gouwens_scn_kcn_expression.npz"
            ),
        )
    if dataset == "scala_room_temperature":
        return load_scala_expression(
            args.scala_pickle,
            args.scala_metadata,
            include_channel_genes=not args.no_force_channel_genes,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _gene_table(
    genes: np.ndarray,
    x_scaled: np.ndarray,
    x_scores: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
) -> pd.DataFrame:
    correlations = correlation_loadings(x_scaled, x_scores)
    rows = []
    for index in np.flatnonzero(selected):
        row = {
            "gene": str(genes[index]),
            "weight_norm": float(np.linalg.norm(weights[index])),
            "is_scn_family": is_scn_gene(genes[index]),
            "is_kcn_family": is_kcn_gene(genes[index]),
        }
        for component in range(weights.shape[1]):
            row[f"weight_component_{component + 1}"] = weights[
                index,
                component,
            ]
            row[f"correlation_component_{component + 1}"] = correlations[
                index,
                component,
            ]
        rows.append(row)
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


def _parameter_table(
    names: tuple[str, ...],
    transformed: tuple[bool, ...],
    y_scaled: np.ndarray,
    y_scores: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    correlations = correlation_loadings(y_scaled, y_scores)
    rows = []
    for index, name in enumerate(names):
        row = {
            "parameter": name,
            "log1p_transformed": transformed[index],
            "weight_norm": float(np.linalg.norm(weights[index])),
            "correlation_norm": float(
                np.linalg.norm(correlations[index])
            ),
        }
        for component in range(weights.shape[1]):
            row[f"weight_component_{component + 1}"] = weights[
                index,
                component,
            ]
            row[f"correlation_component_{component + 1}"] = correlations[
                index,
                component,
            ]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "correlation_norm",
        ascending=False,
    )


def _parameter_prediction_table(
    target,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    actual_raw = inverse_model_target_matrix(actual, target)
    predicted_raw = inverse_model_target_matrix(predicted, target)
    rows = []
    for index, name in enumerate(target.names):
        denominator = np.sum(
            (
                actual_raw[:, index]
                - np.mean(actual_raw[:, index])
            )
            ** 2
        )
        rows.append(
            {
                "parameter": name,
                "oof_r2": (
                    1.0
                    - np.sum(
                        (
                            actual_raw[:, index]
                            - predicted_raw[:, index]
                        )
                        ** 2
                    )
                    / denominator
                    if denominator > 0.0
                    else np.nan
                ),
                "oof_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                actual_raw[:, index]
                                - predicted_raw[:, index]
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("oof_r2", ascending=False)


def _component_table(
    cell_ids: np.ndarray,
    donor_ids: np.ndarray,
    transcriptomic_types: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor_id": donor_ids,
            "transcriptomic_type": transcriptomic_types,
        }
    )
    for component in range(x_scores.shape[1]):
        frame[f"rna_component_{component + 1}"] = x_scores[:, component]
        frame[f"model_component_{component + 1}"] = y_scores[:, component]
    return frame


def _summary_plot(
    analysis,
    genes: pd.DataFrame,
    parameters: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.5),
        constrained_layout=True,
    )
    ridge = analysis.ridge_cv.pivot(
        index="rank",
        columns="ridge_penalty",
        values="mean_test_r2",
    )
    image = axes[0, 0].imshow(
        ridge,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max(0.1, abs(np.nanmin(ridge.values))),
        vmax=max(0.1, abs(np.nanmax(ridge.values))),
    )
    axes[0, 0].set_xticks(
        np.arange(len(ridge.columns)),
        [f"{value:g}" for value in ridge.columns],
    )
    axes[0, 0].set_yticks(
        np.arange(len(ridge.index)),
        ridge.index,
    )
    axes[0, 0].set_xlabel("Ridge penalty")
    axes[0, 0].set_ylabel("Rank")
    axes[0, 0].set_title("Donor-held-out ridge RRR", loc="left")
    figure.colorbar(image, ax=axes[0, 0], label="Mean test R2")

    sparse = analysis.sparse_cv
    axes[0, 1].plot(
        sparse["mean_selected_gene_count"],
        sparse["mean_test_r2"],
        "o-",
        color="#0072b2",
    )
    selected = sparse.loc[sparse["selected_one_se"]].iloc[0]
    axes[0, 1].scatter(
        [selected["mean_selected_gene_count"]],
        [selected["mean_test_r2"]],
        s=90,
        color="#d55e00",
        zorder=4,
        label="one-SE choice",
    )
    axes[0, 1].set_xlabel("Mean selected genes")
    axes[0, 1].set_ylabel("Mean donor-held-out R2")
    axes[0, 1].set_title("Sparse relaxed RRR", loc="left")
    axes[0, 1].legend(frameon=False)

    top_genes = genes.head(12).iloc[::-1]
    axes[1, 0].barh(
        top_genes["gene"],
        top_genes["weight_norm"],
        color=[
            "#cc79a7"
            if scn or kcn
            else "#6f7c85"
            for scn, kcn in zip(
                top_genes["is_scn_family"],
                top_genes["is_kcn_family"],
            )
        ],
    )
    axes[1, 0].set_xlabel("Predictor weight norm")
    axes[1, 0].set_title("Selected genes", loc="left")

    top_parameters = parameters.head(12).iloc[::-1]
    axes[1, 1].barh(
        top_parameters["parameter"],
        top_parameters["correlation_norm"],
        color="#009e73",
    )
    axes[1, 1].set_xlabel("Component-correlation norm")
    axes[1, 1].set_title("Compact-model parameters", loc="left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _component_plot(
    components: pd.DataFrame,
    rank: int,
    path: Path,
    title: str,
) -> None:
    shown = min(rank, 5)
    figure, axes = plt.subplots(
        1,
        shown,
        figsize=(4.0 * shown, 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    for component, axis in enumerate(axes[0], start=1):
        x = components[f"rna_component_{component}"]
        y = components[f"model_component_{component}"]
        correlation = np.corrcoef(x, y)[0, 1]
        axis.scatter(
            x,
            y,
            s=12,
            alpha=0.45,
            color="#3d5665",
            edgecolors="none",
        )
        axis.set_xlabel(f"RNA component {component}")
        axis.set_ylabel(f"Model component {component}")
        axis.set_title(f"r = {correlation:.2f}", loc="left")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title)
    figure.text(
        0.5,
        0.01,
        (
            "In-sample component correspondence; predictive performance "
            "is assessed in donor-held-out folds"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _run_cohort(
    dataset: str,
    cohort_name: str,
    expression,
    parameters: pd.DataFrame,
    output_root: Path,
    fold_count: int,
) -> dict[str, object]:
    target = build_model_target_matrix(parameters)
    x = expression_matrix(expression)
    y = target.values
    analysis = fit_rrr_analysis(
        x,
        y,
        expression.donor_ids,
        fold_count=fold_count,
    )
    x_scaled = np.clip(
        (x - analysis.x_mean) / analysis.x_scale,
        -8.0,
        8.0,
    )
    y_scaled = np.clip(
        (y - analysis.y_mean) / analysis.y_scale,
        -8.0,
        8.0,
    )
    x_scores = x_scaled @ analysis.factors.predictor_weights
    y_scores = y_scaled @ analysis.factors.response_weights
    genes = _gene_table(
        expression.genes,
        x_scaled,
        x_scores,
        analysis.factors.predictor_weights,
        analysis.selected_predictors,
    )
    parameter_loadings = _parameter_table(
        target.names,
        target.log1p_transformed,
        y_scaled,
        y_scores,
        analysis.factors.response_weights,
    )
    predictions = _parameter_prediction_table(
        target,
        y,
        analysis.oof_prediction,
    )
    components = _component_table(
        expression.cell_ids,
        expression.donor_ids,
        expression.transcriptomic_types,
        x_scores,
        y_scores,
    )

    cohort_root = output_root / dataset / cohort_name
    cohort_root.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis.ridge_cv, cohort_root / "ridge_cv.csv")
    _write_csv(analysis.sparse_cv, cohort_root / "sparse_cv.csv")
    _write_csv(genes, cohort_root / "selected_genes.csv")
    _write_csv(
        parameter_loadings,
        cohort_root / "parameter_loadings.csv",
    )
    _write_csv(
        predictions,
        cohort_root / "parameter_oof_performance.csv",
    )
    _write_csv(components, cohort_root / "cell_components.csv")
    weights = pd.DataFrame(
        analysis.factors.predictor_weights[
            analysis.selected_predictors
        ],
        index=expression.genes[analysis.selected_predictors],
        columns=[
            f"component_{index + 1}"
            for index in range(analysis.rank)
        ],
    ).reset_index(names="gene")
    _write_csv(weights, cohort_root / "selected_gene_weights.csv")
    response_weights = pd.DataFrame(
        analysis.factors.response_weights,
        index=target.names,
        columns=[
            f"component_{index + 1}"
            for index in range(analysis.rank)
        ],
    ).reset_index(names="parameter")
    _write_csv(
        response_weights,
        cohort_root / "response_weights.csv",
    )
    title = (
        f"{dataset.replace('_', ' ')} | {cohort_name} | "
        f"n={len(parameters)}, rank={analysis.rank}"
    )
    _summary_plot(
        analysis,
        genes,
        parameter_loadings,
        cohort_root / "rrr_summary.png",
        title,
    )
    _component_plot(
        components,
        analysis.rank,
        cohort_root / "component_correspondence.png",
        title,
    )
    summary = {
        "dataset": dataset,
        "cohort": cohort_name,
        "cell_count": len(parameters),
        "donor_count": int(
            pd.Series(expression.donor_ids).nunique(dropna=True)
        ),
        "gene_count": len(expression.genes),
        "response_parameter_count": len(target.names),
        "selected_rank": analysis.rank,
        "selected_ridge_penalty": analysis.ridge_penalty,
        "selected_sparse_ratio": analysis.sparse_ratio,
        "l1_ratio": analysis.l1_ratio,
        "selected_gene_count": int(
            np.sum(analysis.selected_predictors)
        ),
        "cv_replay_r2_after_hyperparameter_selection": analysis.oof_r2,
        "selected_scn_genes": genes.loc[
            genes["is_scn_family"],
            "gene",
        ].tolist(),
        "selected_kcn_genes": genes.loc[
            genes["is_kcn_family"],
            "gene",
        ].tolist(),
    }
    (cohort_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="ascii",
    )
    return summary


def _cross_dataset_comparison(
    output_root: Path,
    gene_sets: dict[str, set[str]],
) -> None:
    overlap_rows = []
    alignment_rows = []
    channel_rows = []
    for cohort in ("all_eligible", "high_qc"):
        roots = {
            dataset: output_root / dataset / cohort
            for dataset in (
                "gouwens_visp",
                "scala_room_temperature",
            )
        }
        if not all(
            (root / "selected_genes.csv").exists()
            for root in roots.values()
        ):
            continue
        shared_universe = (
            gene_sets["gouwens_visp"]
            & gene_sets["scala_room_temperature"]
        )
        selected = {}
        for dataset, root in roots.items():
            genes = pd.read_csv(root / "selected_genes.csv")
            selected[dataset] = (
                set(genes["gene"].astype(str)) & shared_universe
            )
            all_selected = set(genes["gene"].astype(str))
            channels = {
                gene
                for gene in gene_sets[dataset]
                if is_scn_gene(gene) or is_kcn_gene(gene)
            }
            selected_channels = all_selected & channels
            channel_rows.append(
                {
                    "dataset": dataset,
                    "cohort": cohort,
                    "gene_universe": len(gene_sets[dataset]),
                    "channel_genes_in_universe": len(channels),
                    "selected_genes": len(all_selected),
                    "selected_channel_genes": len(selected_channels),
                    "expected_selected_channel_genes": (
                        len(all_selected)
                        * len(channels)
                        / len(gene_sets[dataset])
                    ),
                    "hypergeometric_enrichment_p": hypergeom.sf(
                        len(selected_channels) - 1,
                        len(gene_sets[dataset]),
                        len(channels),
                        len(all_selected),
                    ),
                    "channel_genes": "|".join(
                        sorted(selected_channels)
                    ),
                }
            )
        first = selected["gouwens_visp"]
        second = selected["scala_room_temperature"]
        overlap = first & second
        union = first | second
        universe_size = len(shared_universe)
        overlap_rows.append(
            {
                "cohort": cohort,
                "shared_gene_universe": universe_size,
                "gouwens_selected_in_universe": len(first),
                "scala_selected_in_universe": len(second),
                "overlap_count": len(overlap),
                "jaccard": (
                    len(overlap) / len(union) if union else np.nan
                ),
                "expected_overlap_by_chance": (
                    len(first) * len(second) / universe_size
                    if universe_size
                    else np.nan
                ),
                "hypergeometric_enrichment_p": (
                    hypergeom.sf(
                        len(overlap) - 1,
                        universe_size,
                        len(first),
                        len(second),
                    )
                    if universe_size
                    else np.nan
                ),
                "overlap_genes": "|".join(sorted(overlap)),
                "overlap_scn_kcn": "|".join(
                    sorted(
                        gene
                        for gene in overlap
                        if is_scn_gene(gene) or is_kcn_gene(gene)
                    )
                ),
            }
        )

        response = {
            dataset: pd.read_csv(root / "response_weights.csv").set_index(
                "parameter"
            )
            for dataset, root in roots.items()
        }
        shared_parameters = response["gouwens_visp"].index.intersection(
            response["scala_room_temperature"].index
        )
        first_weights = response["gouwens_visp"].loc[
            shared_parameters
        ].to_numpy(dtype=float)
        second_weights = response["scala_room_temperature"].loc[
            shared_parameters
        ].to_numpy(dtype=float)
        first_weights /= np.maximum(
            np.linalg.norm(first_weights, axis=0),
            1e-12,
        )
        second_weights /= np.maximum(
            np.linalg.norm(second_weights, axis=0),
            1e-12,
        )
        cosine = first_weights.T @ second_weights
        first_components, second_components = linear_sum_assignment(
            -np.abs(cosine)
        )
        for first_component, second_component in zip(
            first_components,
            second_components,
        ):
            alignment_rows.append(
                {
                    "cohort": cohort,
                    "shared_parameter_count": len(shared_parameters),
                    "gouwens_component": first_component + 1,
                    "scala_component": second_component + 1,
                    "response_loading_cosine": cosine[
                        first_component,
                        second_component,
                    ],
                    "absolute_response_loading_cosine": abs(
                        cosine[first_component, second_component]
                    ),
                }
            )
    if overlap_rows:
        _write_csv(
            pd.DataFrame(overlap_rows),
            output_root / "cross_dataset_gene_overlap.csv",
        )
    if alignment_rows:
        _write_csv(
            pd.DataFrame(alignment_rows),
            output_root / "cross_dataset_component_alignment.csv",
        )
    if channel_rows:
        _write_csv(
            pd.DataFrame(channel_rows),
            output_root / "channel_family_enrichment.csv",
        )


def _parameter_family(name: str) -> str:
    if name.startswith(("template_cos_", "template_sin_")):
        return "waveform harmonics"
    if name.startswith("template_mean_"):
        return "waveform mean"
    if name.startswith(("downstroke_", "upstroke_")):
        return "segment durations"
    if name.startswith(("baseline_period_", "latency_")):
        return "spike timing"
    if name.startswith("timing_"):
        return "memory"
    return "rest / current"


def _overview_outputs(
    output_root: Path,
    summaries: list[dict[str, object]],
    joins: list[dict[str, object]],
) -> None:
    datasets = ("gouwens_visp", "scala_room_temperature")
    cohorts = ("all_eligible", "high_qc")
    performance = {}
    for dataset in datasets:
        for cohort in cohorts:
            path = (
                output_root
                / dataset
                / cohort
                / "parameter_oof_performance.csv"
            )
            if path.exists():
                frame = pd.read_csv(path)[["parameter", "oof_r2"]]
                performance[(dataset, cohort)] = frame.rename(
                    columns={
                        "oof_r2": f"{dataset}_{cohort}_oof_r2"
                    }
                )
    if not all(
        (dataset, "all_eligible") in performance
        for dataset in datasets
    ):
        return
    comparison = performance[(datasets[0], "all_eligible")]
    for key, frame in performance.items():
        if key == (datasets[0], "all_eligible"):
            continue
        comparison = comparison.merge(frame, on="parameter", how="outer")
    comparison["parameter_family"] = comparison["parameter"].map(
        _parameter_family
    )
    _write_csv(
        comparison,
        output_root / "cross_dataset_parameter_performance.csv",
    )

    summary_frame = pd.DataFrame(summaries)
    join_frame = pd.DataFrame(joins).set_index("dataset")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.0),
        constrained_layout=True,
    )
    positions = np.arange(len(datasets))
    width = 0.34
    fitted = [join_frame.loc[dataset, "fitted_cells"] for dataset in datasets]
    joined = [join_frame.loc[dataset, "joined_cells"] for dataset in datasets]
    high_qc = [
        int(
            summary_frame.loc[
                summary_frame["dataset"].eq(dataset)
                & summary_frame["cohort"].eq("high_qc"),
                "cell_count",
            ].iloc[0]
        )
        for dataset in datasets
    ]
    axes[0, 0].bar(
        positions - width,
        fitted,
        width,
        label="compact fit",
        color="#8f9ba3",
    )
    axes[0, 0].bar(
        positions,
        joined,
        width,
        label="RNA matched",
        color="#0072b2",
    )
    axes[0, 0].bar(
        positions + width,
        high_qc,
        width,
        label="RNA matched + high QC",
        color="#d55e00",
    )
    axes[0, 0].set_xticks(
        positions,
        [dataset.replace("_", " ") for dataset in datasets],
    )
    axes[0, 0].set_ylabel("Cells")
    axes[0, 0].set_title("Population flow", loc="left")
    axes[0, 0].legend(frameon=False)

    for cohort_index, cohort in enumerate(cohorts):
        subset = summary_frame.loc[
            summary_frame["cohort"].eq(cohort)
        ].set_index("dataset")
        values = [
            subset.loc[
                dataset,
                "cv_replay_r2_after_hyperparameter_selection",
            ]
            for dataset in datasets
        ]
        bars = axes[0, 1].bar(
            positions + (cohort_index - 0.5) * width,
            values,
            width,
            label=cohort.replace("_", " "),
            color=("#0072b2", "#d55e00")[cohort_index],
        )
        for bar, dataset in zip(bars, datasets):
            rank = int(subset.loc[dataset, "selected_rank"])
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"rank {rank}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0, 1].set_xticks(
        positions,
        [dataset.replace("_", " ") for dataset in datasets],
    )
    axes[0, 1].set_ylim(0.0, 0.45)
    axes[0, 1].set_ylabel("CV replay R2")
    axes[0, 1].set_title("RNA prediction of compact parameters", loc="left")
    axes[0, 1].legend(frameon=False)

    g_column = "gouwens_visp_all_eligible_oof_r2"
    s_column = "scala_room_temperature_all_eligible_oof_r2"
    family_colors = {
        "waveform harmonics": "#0072b2",
        "waveform mean": "#56b4e9",
        "segment durations": "#009e73",
        "spike timing": "#e69f00",
        "memory": "#cc79a7",
        "rest / current": "#6f7c85",
    }
    for family, frame in comparison.groupby("parameter_family"):
        axes[1, 0].scatter(
            frame[g_column],
            frame[s_column],
            s=25,
            alpha=0.72,
            color=family_colors[family],
            label=family,
            edgecolors="none",
        )
    limits = (-0.3, 0.72)
    axes[1, 0].plot(limits, limits, color="#999999", linewidth=1)
    axes[1, 0].axhline(0.0, color="#cccccc", linewidth=0.8)
    axes[1, 0].axvline(0.0, color="#cccccc", linewidth=0.8)
    axes[1, 0].set_xlim(limits)
    axes[1, 0].set_ylim(limits)
    axes[1, 0].set_xlabel("Gouwens parameter OOF R2")
    axes[1, 0].set_ylabel("Scala parameter OOF R2")
    axes[1, 0].set_title("Which parameters generalize?", loc="left")
    for name in (
        "timing_tau_ms",
        "timing_spike_jump",
        "rheobase_current_pa",
        "downstroke_duration_ms_q025",
    ):
        row = comparison.loc[comparison["parameter"].eq(name)]
        if len(row):
            axes[1, 0].annotate(
                name.replace("_", " "),
                (row[g_column].iloc[0], row[s_column].iloc[0]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    cohort_labels = (
        ("gouwens_visp", "all_eligible", "G all"),
        ("gouwens_visp", "high_qc", "G QC"),
        ("scala_room_temperature", "all_eligible", "S all"),
        ("scala_room_temperature", "high_qc", "S QC"),
    )
    channel_tables = []
    for dataset, cohort, label in cohort_labels:
        frame = pd.read_csv(
            output_root / dataset / cohort / "selected_genes.csv"
        )
        frame = frame.loc[
            frame["is_scn_family"] | frame["is_kcn_family"],
            ["gene", "weight_norm"],
        ].rename(columns={"weight_norm": label})
        channel_tables.append(frame)
    channel = channel_tables[0]
    for frame in channel_tables[1:]:
        channel = channel.merge(frame, on="gene", how="outer")
    channel = channel.set_index("gene").fillna(0.0)
    channel = channel / np.maximum(channel.max(axis=0), 1e-12)
    channel["selection_count"] = (channel > 0.0).sum(axis=1)
    channel["maximum_weight"] = channel.drop(
        columns="selection_count"
    ).max(axis=1)
    channel = channel.sort_values(
        ["selection_count", "maximum_weight"],
        ascending=False,
    ).head(20)
    heatmap = channel[
        [label for _, _, label in cohort_labels]
    ].to_numpy()
    image = axes[1, 1].imshow(
        heatmap,
        aspect="auto",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_xticks(
        np.arange(len(cohort_labels)),
        [label for _, _, label in cohort_labels],
    )
    axes[1, 1].set_yticks(
        np.arange(len(channel)),
        channel.index,
        fontsize=8,
    )
    axes[1, 1].set_title("Selected Scn/Kcn genes", loc="left")
    figure.colorbar(
        image,
        ax=axes[1, 1],
        label="Weight / cohort maximum",
    )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Compact spike dynamics and Patch-seq expression")
    figure.savefig(
        output_root / "transcriptomic_rrr_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


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
        default="outputs/transcriptomic_rrr",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gouwens_visp", "scala_room_temperature"),
        default=("gouwens_visp", "scala_room_temperature"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gouwens-mat",
        default=(
            DEFAULT_KOBAK_ROOT
            / "data/gouwens2020/"
            "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
        ),
        type=Path,
    )
    parser.add_argument(
        "--gouwens-metadata",
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Patch-seq AIBS/"
            "20200711_patchseq_metadata_mouse.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--gouwens-cpm",
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Patch-seq AIBS/transcriptomes/"
            "20200513_Mouse_PatchSeq_Release_cpm.v2/"
            "20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--scala-pickle",
        default=(
            DEFAULT_KOBAK_ROOT / "data/scala2020.pickle"
        ),
        type=Path,
    )
    parser.add_argument("--no-force-channel-genes", action="store_true")
    parser.add_argument(
        "--scala-metadata",
        default=(
            DEFAULT_PATCHSEQ_ROOT
            / "Scala_pseq_data/m1_patchseq_meta_data.csv"
        ),
        type=Path,
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    parameter_frame = pd.read_csv(
        args.parameters,
        dtype={"cell_id": "string"},
    )
    summaries = []
    join_rows = []
    dataset_gene_sets: dict[str, set[str]] = {}
    for dataset in args.datasets:
        requested_cell_ids = set(
            parameter_frame.loc[
                parameter_frame["dataset"].eq(dataset),
                "cell_id",
            ].astype(str)
        )
        expression = _load_expression(
            dataset,
            args,
            requested_cell_ids,
            output_root,
        )
        dataset_gene_sets[dataset] = set(expression.genes.astype(str))
        aligned_expression, aligned_parameters = (
            align_expression_to_parameters(
                expression,
                parameter_frame,
            )
        )
        join_rows.append(
            {
                "dataset": dataset,
                "fitted_cells": int(
                    parameter_frame["dataset"].eq(dataset).sum()
                ),
                "source_rna_cells": (
                    expression.source_cell_count
                    if expression.source_cell_count is not None
                    else len(expression.cell_ids)
                ),
                "candidate_rna_cells": len(expression.cell_ids),
                "joined_cells": len(aligned_parameters),
                "joined_donors": int(
                    pd.Series(
                        aligned_expression.donor_ids
                    ).nunique(dropna=True)
                ),
            }
        )
        cohorts = {
            "all_eligible": np.ones(
                len(aligned_parameters),
                dtype=bool,
            ),
            "high_qc": high_qc_mask(aligned_parameters),
        }
        for cohort_name, mask in cohorts.items():
            if int(np.sum(mask)) < args.minimum_cells:
                continue
            cohort_root = output_root / dataset / cohort_name
            summary_path = cohort_root / "summary.json"
            if args.resume and summary_path.exists():
                summaries.append(
                    json.loads(summary_path.read_text(encoding="ascii"))
                )
                print(
                    f"Skipping completed {dataset}/{cohort_name}",
                    flush=True,
                )
                continue
            subset_expression = type(aligned_expression)(
                dataset=aligned_expression.dataset,
                cell_ids=aligned_expression.cell_ids[mask],
                genes=aligned_expression.genes,
                values=aligned_expression.values[mask],
                donor_ids=aligned_expression.donor_ids[mask],
                transcriptomic_types=(
                    aligned_expression.transcriptomic_types[mask]
                ),
                already_log_normalized=(
                    aligned_expression.already_log_normalized
                ),
                library_size=(
                    aligned_expression.library_size[mask]
                    if aligned_expression.library_size is not None
                    else None
                ),
                source_cell_count=(
                    aligned_expression.source_cell_count
                ),
            )
            print(
                f"Fitting {dataset}/{cohort_name}: "
                f"{int(np.sum(mask))} cells",
                flush=True,
            )
            summary = _run_cohort(
                dataset,
                cohort_name,
                subset_expression,
                aligned_parameters.loc[mask].reset_index(drop=True),
                output_root,
                args.folds,
            )
            summaries.append(summary)
            print(
                f"Completed {dataset}/{cohort_name}: "
                f"rank {summary['selected_rank']}, "
                f"{summary['selected_gene_count']} genes",
                flush=True,
            )
    _write_csv(pd.DataFrame(join_rows), output_root / "rna_join_summary.csv")
    _write_csv(pd.DataFrame(summaries), output_root / "rrr_summary.csv")
    if {
        "gouwens_visp",
        "scala_room_temperature",
    }.issubset(dataset_gene_sets):
        _cross_dataset_comparison(output_root, dataset_gene_sets)
        _overview_outputs(output_root, summaries, join_rows)
    metadata = {
        "parameter_file": args.parameters,
        "datasets": list(args.datasets),
        "fold_count": args.folds,
        "minimum_cells": args.minimum_cells,
        "gouwens_expression": str(args.gouwens_mat),
        "gouwens_metadata": str(args.gouwens_metadata),
        "gouwens_cpm": str(args.gouwens_cpm),
        "scala_expression": str(args.scala_pickle),
        "scala_metadata": str(args.scala_metadata),
        "forced_channel_gene_inclusion": (
            not args.no_force_channel_genes
        ),
        "high_qc_definition": {
            "validation_phase_chamfer_max": 0.01,
            "validation_relative_spike_count_error_max": 0.5,
            "timing_late_period_nrmse_max": 2.0,
        },
        "rrr": {
            "ranks": [1, 2, 3, 5, 8],
            "ridge_penalties": [0.01, 0.1, 1.0, 10.0, 100.0],
            "elastic_net_l1_ratio": 0.5,
            "sparse_ratios": [0.02, 0.05, 0.1, 0.2, 0.4, 0.7],
            "selection": "one standard error, then relaxed ridge RRR",
        },
    }
    (output_root / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
