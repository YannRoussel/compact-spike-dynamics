#!/usr/bin/env python3
"""Compare RNA prediction of single-tau and fixed-kernel timing targets."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.reduced_rank import (
    fit_class_residualized_rrr_analysis,
    nested_class_residualized_rrr_evaluation,
)
from inverse_ephys_alpha_beta.transcriptomics import (
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
TARGETS = {
    "single_tau": (
        "single_log_tau_ms",
        "single_spike_jump",
    ),
    "fixed_kernel": (
        "kernel_value_0000ms",
        "kernel_value_0050ms",
        "kernel_value_0200ms",
        "kernel_value_0800ms",
    ),
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_expression(
    dataset: str,
    parameters: pd.DataFrame,
    args,
):
    requested = set(
        parameters.loc[
            parameters["dataset"].eq(dataset),
            "cell_id",
        ].astype(str)
    )
    if dataset == "gouwens_visp":
        return load_gouwens_expression(
            args.gouwens_mat,
            args.gouwens_metadata,
            channel_cpm_path=args.gouwens_cpm,
            requested_cell_ids=requested,
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


def _performance(
    names: tuple[str, ...],
    actual: np.ndarray,
    prediction: np.ndarray,
    class_only: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(names):
        centered = actual[:, index] - np.mean(actual[:, index])
        denominator = float(np.sum(centered**2))
        class_error = float(
            np.sum((actual[:, index] - class_only[:, index]) ** 2)
        )
        combined_error = float(
            np.sum((actual[:, index] - prediction[:, index]) ** 2)
        )
        class_r2 = 1.0 - class_error / denominator
        combined_r2 = 1.0 - combined_error / denominator
        rows.append(
            {
                "parameter": name,
                "class_only_oof_r2": class_r2,
                "class_plus_rna_oof_r2": combined_r2,
                "delta_oof_r2": combined_r2 - class_r2,
                "incremental_residual_r2": (
                    1.0 - combined_error / class_error
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_matrix(
    parameters: pd.DataFrame,
    target_family: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    frame = parameters.copy()
    frame["single_log_tau_ms"] = np.log1p(
        frame["single_tau_ms"].astype(float)
    )
    names = TARGETS[target_family]
    return (
        frame.loc[:, names].to_numpy(dtype=float),
        names,
    )


def _plot(
    summaries: pd.DataFrame,
    performance: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.5),
        constrained_layout=True,
    )
    labels = []
    class_values = []
    combined_values = []
    for row in summaries.itertuples(index=False):
        labels.append(
            f"{row.dataset.split('_')[0]}\n"
            f"{row.target_family.replace('_', ' ')}"
        )
        class_values.append(row.class_only_nested_r2)
        combined_values.append(row.class_plus_rna_nested_r2)
    positions = np.arange(len(labels))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        class_values,
        width,
        color="#8f9ba3",
        label="class only",
    )
    axes[0].bar(
        positions + width / 2,
        combined_values,
        width,
        color="#0072b2",
        label="class + RNA",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Nested donor-held-out R2")
    axes[0].set_title("Adaptation target prediction", loc="left")
    axes[0].legend(frameon=False)

    colors = {
        "gouwens_visp": "#0072b2",
        "scala_room_temperature": "#d55e00",
    }
    short_names = {
        "single_log_tau_ms": "tau",
        "single_spike_jump": "jump",
        "kernel_value_0000ms": "K(0)",
        "kernel_value_0050ms": "K(50)",
        "kernel_value_0200ms": "K(200)",
        "kernel_value_0800ms": "K(800)",
    }
    offsets = ((5, 5), (5, -11), (5, 5), (5, -11), (5, 5), (5, -11))
    for axis, dataset in zip(
        axes[1:],
        ("gouwens_visp", "scala_room_temperature"),
    ):
        frame = performance.loc[performance["dataset"].eq(dataset)]
        axis.scatter(
            frame["class_only_oof_r2"],
            frame["class_plus_rna_oof_r2"],
            s=55,
            color=colors[dataset],
        )
        for row, offset in zip(frame.itertuples(index=False), offsets):
            axis.annotate(
                short_names[row.parameter],
                (row.class_only_oof_r2, row.class_plus_rna_oof_r2),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
        limits = (-0.2, 0.6) if dataset == "gouwens_visp" else (-0.2, 0.4)
        axis.plot(limits, limits, color="#999999", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Class-only R2")
        axis.set_ylabel("Class + RNA R2")
        axis.set_title(dataset.replace("_", " "), loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--parameters",
        default=(
            "outputs/adaptation_kernel_eligible_pilot_200/"
            "kernel_model_parameters.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/kernel_transcriptomic_pilot",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
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

    parameters = pd.read_csv(
        args.parameters,
        dtype={"cell_id": "string"},
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    performance_rows = []
    for dataset in ("gouwens_visp", "scala_room_temperature"):
        expression = _load_expression(dataset, parameters, args)
        aligned_expression, aligned_parameters = (
            align_expression_to_parameters(expression, parameters)
        )
        x = expression_matrix(aligned_expression)
        classes = np.asarray(
            [
                broad_transcriptomic_class(label)
                for label in aligned_expression.transcriptomic_types
            ],
            dtype=object,
        )
        for target_family in TARGETS:
            y, names = _target_matrix(
                aligned_parameters,
                target_family,
            )
            ranks = tuple(
                rank for rank in (1, 2, 3, 5) if rank <= y.shape[1]
            )
            nested = nested_class_residualized_rrr_evaluation(
                x,
                y,
                aligned_expression.donor_ids,
                classes,
                ranks=ranks,
                penalties=(0.1, 1.0, 10.0),
                sparse_ratios=(),
                outer_fold_count=args.folds,
                inner_fold_count=args.inner_folds,
            )
            sparse = fit_class_residualized_rrr_analysis(
                x,
                y,
                aligned_expression.donor_ids,
                classes,
                ranks=ranks,
                sparse_ratios=(0.1,),
                fold_count=args.folds,
            )
            family_performance = _performance(
                names,
                y,
                nested.oof_prediction,
                nested.class_only_oof_prediction,
            )
            family_performance.insert(0, "target_family", target_family)
            family_performance.insert(0, "dataset", dataset)
            performance_rows.append(family_performance)
            summaries.append(
                {
                    "dataset": dataset,
                    "target_family": target_family,
                    "cell_count": len(y),
                    "response_count": y.shape[1],
                    "class_only_nested_r2": nested.class_only_oof_r2,
                    "class_plus_rna_nested_r2": nested.combined_oof_r2,
                    "delta_nested_r2": nested.delta_oof_r2,
                    "incremental_residual_nested_r2": (
                        nested.incremental_residual_r2
                    ),
                    "sparse_cv_replay_residual_r2": (
                        sparse.incremental_residual_r2
                    ),
                    "sparse_selected_gene_count": int(
                        np.sum(sparse.selected_predictors)
                    ),
                }
            )
            target_root = output_root / dataset / target_family
            target_root.mkdir(parents=True, exist_ok=True)
            _write_csv(
                nested.fold_hyperparameters,
                target_root / "nested_fold_hyperparameters.csv",
            )
            selected = pd.DataFrame(
                {
                    "gene": aligned_expression.genes[
                        sparse.selected_predictors
                    ],
                    "weight_norm": np.linalg.norm(
                        sparse.factors.predictor_weights[
                            sparse.selected_predictors
                        ],
                        axis=1,
                    ),
                }
            ).sort_values("weight_norm", ascending=False)
            _write_csv(
                selected,
                target_root / "selected_genes.csv",
            )
    summary = pd.DataFrame(summaries)
    performance = pd.concat(performance_rows, ignore_index=True)
    _write_csv(summary, output_root / "summary.csv")
    _write_csv(performance, output_root / "parameter_performance.csv")
    _plot(
        summary,
        performance,
        output_root / "kernel_transcriptomic_comparison.png",
    )
    (output_root / "metadata.json").write_text(
        json.dumps(
            {
                "parameters": args.parameters,
                "comparison": (
                    "single fitted tau/jump versus fixed-lag kernel values"
                ),
                "primary_prediction": (
                    "class-residualized donor-grouped nested ridge RRR"
                ),
                "descriptive_gene_selection": (
                    "class-residualized sparse RRR at ratio 0.1"
                ),
            },
            indent=2,
        ),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
