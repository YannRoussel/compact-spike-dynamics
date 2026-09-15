#!/usr/bin/env python3
"""Combine temperature-specific Allen atlas ephys predictions."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.atlas_ephys import AP_FEATURE_NAMES


INHIBITORY = {"Lamp5", "Sncg", "Vip", "Sst", "Pvalb"}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _add_delta(performance: pd.DataFrame) -> pd.DataFrame:
    result = performance.copy()
    result["rna_delta_r2_over_class"] = (
        result["rna_r2"] - result["class_only_r2"]
    )
    return result


def _summary(root: Path, dataset: str) -> dict[str, object]:
    path = root / dataset / "summary.json"
    summary = json.loads(path.read_text(encoding="ascii"))
    performance_path = root / dataset / "loto_feature_performance.csv"
    performance = _add_delta(pd.read_csv(performance_path))
    _write_csv(performance, performance_path)
    summary["median_feature_rna_delta_r2_over_class"] = float(
        performance["rna_delta_r2_over_class"].median()
    )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    return summary


def _combined_predictions(root: Path) -> pd.DataFrame:
    warm = pd.read_csv(root / "gouwens_visp" / "atlas_neural_ttype_ap_predictions.csv")
    room = pd.read_csv(
        root / "scala_room_temperature" / "atlas_neural_ttype_ap_predictions.csv"
    )
    warm = warm.set_index("atlas_type", drop=False)
    room = room.set_index("atlas_type", drop=False)
    rows = []
    for atlas_type in room.index:
        broad_class = str(room.loc[atlas_type, "broad_class"])
        if broad_class in INHIBITORY:
            source = warm.loc[atlas_type]
            dataset = "gouwens_visp"
            temperature = "34C"
        elif broad_class.startswith("Glut_"):
            source = room.loc[atlas_type]
            dataset = "scala_room_temperature"
            temperature = "room_temperature"
        else:
            source = room.loc[atlas_type]
            dataset = "none"
            temperature = "unsupported"
        row = source.to_dict()
        row["calibration_dataset"] = dataset
        row["calibration_temperature"] = temperature
        supported = bool(source["class_supported"]) and dataset != "none"
        row["prediction_supported"] = supported
        if not supported:
            status = "unsupported_class"
        elif not bool(source["atlas_only"]):
            status = "represented_mapping"
        else:
            status = f"atlas_only_{source['support_tier']}"
        row["prediction_status"] = status
        row["interpret_as_ttype_specific"] = bool(
            supported
            and dataset == "gouwens_visp"
            and float(source["distance_ratio_to_loto_p95"]) <= 1.0
        )
        if not supported:
            for feature in AP_FEATURE_NAMES:
                row[feature] = np.nan
                row[f"{feature}_p05"] = np.nan
                row[f"{feature}_p95"] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    leading = [
        "atlas_type",
        "cluster_id",
        "leaf_label",
        "broad_class",
        "calibration_dataset",
        "calibration_temperature",
        "prediction_supported",
        "prediction_status",
        "interpret_as_ttype_specific",
        "atlas_only",
        "support_tier",
        "distance_ratio_to_loto_p95",
        "expected_ap_feature_nrmse",
        "nearest_patchseq_type",
        "mapped_patchseq_types",
    ]
    remaining = [column for column in result.columns if column not in leading]
    return result.loc[:, leading + remaining].sort_values("cluster_id")


def _cross_calibration(root: Path) -> pd.DataFrame:
    warm = pd.read_csv(root / "gouwens_visp" / "atlas_neural_ttype_ap_predictions.csv")
    room = pd.read_csv(
        root / "scala_room_temperature" / "atlas_neural_ttype_ap_predictions.csv"
    )
    merged = warm.merge(room, on="atlas_type", suffixes=("_warm", "_room"))
    merged = merged.loc[merged["broad_class_warm"].isin(INHIBITORY)]
    rows = []
    for feature in AP_FEATURE_NAMES:
        warm_values = merged[f"{feature}_warm"].to_numpy(dtype=float)
        room_values = merged[f"{feature}_room"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "atlas_type_count": len(merged),
                "warm_room_pearson_r": float(np.corrcoef(warm_values, room_values)[0, 1]),
                "median_warm_minus_room": float(np.median(warm_values - room_values)),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/atlas_ttype_ephys_prediction"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        _summary(args.output_root, dataset)
        for dataset in ("gouwens_visp", "scala_room_temperature")
    ]
    _write_csv(pd.DataFrame(summaries), args.output_root / "summary.csv")
    combined = _combined_predictions(args.output_root)
    _write_csv(
        combined,
        args.output_root / "best_available_neural_ttype_ap_predictions.csv",
    )
    _write_csv(
        _cross_calibration(args.output_root),
        args.output_root / "inhibitory_cross_temperature_agreement.csv",
    )
    metadata = {
        "reference_atlas": "Allen Mouse Whole Cortex and Hippocampus SMART-seq",
        "atlas_neural_ttype_count": int(len(combined)),
        "prediction_supported_count": int(np.sum(combined["prediction_supported"])),
        "atlas_only_supported_count": int(
            np.sum(combined["prediction_supported"] & combined["atlas_only"])
        ),
        "ttype_specific_interpretation_count": int(
            np.sum(combined["interpret_as_ttype_specific"])
        ),
        "calibration_policy": {
            "inhibitory": "Gouwens VISp, 34C",
            "cortical_glutamatergic": "Scala, room temperature",
            "other_or_hippocampal": "unsupported",
        },
        "important_limitation": (
            "True atlas-only t-types have no ephys ground truth. Their recovery is "
            "estimated by leave-one-Patch-seq-t-type-out validation."
        ),
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
