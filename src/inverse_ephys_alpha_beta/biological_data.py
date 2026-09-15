"""Download and harmonize the Scala and Gouwens Patch-seq e-feature releases."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

SCALA_COMMIT = "3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9"
GOUWENS_COMMIT = "6c0be92e7a919321a4b877623354ee48d5f75921"

SOURCE_FILES = {
    "scala/m1_patchseq_ephys_features.csv": (
        "https://raw.githubusercontent.com/berenslab/mini-atlas/"
        f"{SCALA_COMMIT}/data/m1_patchseq_ephys_features.csv"
    ),
    "scala/m1_patchseq_meta_data.csv": (
        "https://raw.githubusercontent.com/berenslab/mini-atlas/"
        f"{SCALA_COMMIT}/data/m1_patchseq_meta_data.csv"
    ),
    "scala/m1_patchseq_phys_temp_ephys_features.csv": (
        "https://raw.githubusercontent.com/berenslab/mini-atlas/"
        f"{SCALA_COMMIT}/data/m1_patchseq_phys_temp_ephys_features.csv"
    ),
    "scala/m1_patchseq_phys_temp_meta_data.csv": (
        "https://raw.githubusercontent.com/berenslab/mini-atlas/"
        f"{SCALA_COMMIT}/data/m1_patchseq_phys_temp_meta_data.csv"
    ),
    "gouwens/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat": (
        "https://raw.githubusercontent.com/AllenInstitute/coupledAE-patchseq/"
        f"{GOUWENS_COMMIT}/data/proc/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
    ),
}

WAVEFORM_CORE_FEATURES = (
    "ap_threshold_mv",
    "ap_peak_mv",
    "ap_width_ms",
    "fast_trough_mv",
    "upstroke_downstroke_ratio",
)

EXPANDED_COMMON_FEATURES = WAVEFORM_CORE_FEATURES + (
    "baseline_voltage_mv",
    "latency_ms",
)

INTRINSIC_COMMON_FEATURES = EXPANDED_COMMON_FEATURES + (
    "input_resistance_mohm",
    "membrane_tau_ms",
    "rheobase_pa",
    "sag_ratio",
)

SPIKE_CYCLE_COMMON_FEATURES = WAVEFORM_CORE_FEATURES + (
    "upstroke_mv_ms",
    "downstroke_mv_ms",
    "ap_upstroke_time_ms",
    "ap_repolarization_time_ms",
    "ap_duration_ms",
    "onset_voltage_mv",
    "onset_dvdt_mv_ms",
    "onset_rapidness_per_ms",
    "max_acceleration_mv_ms2",
    "max_acceleration_voltage_mv",
    "min_acceleration_mv_ms2",
    "min_acceleration_voltage_mv",
    "upstroke_inflection_voltage_mv",
    "upstroke_inflection_dvdt_mv_ms",
    "downstroke_inflection_voltage_mv",
    "downstroke_inflection_dvdt_mv_ms",
    "upstroke_inflection_relative_to_threshold_mv",
    "downstroke_inflection_relative_to_peak_mv",
    "ap_phase_area_v2_ms",
    "ap_phase_area_normalized",
    "cycle_phase_area_v2_ms",
    "cycle_phase_area_normalized",
    "cycle_phase_path_length_normalized",
)


def download_biological_data(
    data_dir: str | Path = "data/external",
    force: bool = False,
) -> list[Path]:
    """Download small, commit-pinned processed files from the authors' repositories."""
    root = Path(data_dir)
    downloaded = []
    for relative_path, url in SOURCE_FILES.items():
        target = root / relative_path
        if target.exists() and not force:
            downloaded.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
        downloaded.append(target)

    manifest = {
        "scala_commit": SCALA_COMMIT,
        "gouwens_commit": GOUWENS_COMMIT,
        "files": SOURCE_FILES,
    }
    (root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return downloaded


def _clean_identifier(values: Iterable) -> pd.Series:
    return pd.Series(values, dtype="string").str.strip()


def _scala_sag_ratio_to_fraction(values: pd.Series) -> pd.Series:
    """Convert peak/steady deflection ratio to recovered sag fraction."""
    return 1.0 - 1.0 / values


def _load_scala_release(
    feature_path: Path,
    metadata_path: Path,
    dataset_name: str,
    temperature_c: float,
) -> pd.DataFrame:
    features = pd.read_csv(feature_path, index_col=0)
    features.index = features.index.astype(str).str.strip()
    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata["Cell"] = _clean_identifier(metadata["Cell"])
    metadata = metadata.drop_duplicates("Cell").set_index("Cell")

    output = pd.DataFrame(index=features.index)
    output["dataset"] = dataset_name
    output["cell_id"] = output.index
    output["temperature_c"] = temperature_c
    output["transcriptomic_type"] = metadata["RNA type"].reindex(output.index).to_numpy()
    output["transcriptomic_family"] = metadata["RNA family"].reindex(output.index).to_numpy()
    output["cre_line"] = metadata["Cre"].reindex(output.index).to_numpy()
    output["ap_threshold_mv"] = features["AP threshold (mV)"]
    output["ap_amplitude_mv"] = features["AP amplitude (mV)"]
    output["ap_peak_mv"] = output["ap_threshold_mv"] + output["ap_amplitude_mv"]
    output["ap_width_ms"] = features["AP width (ms)"]
    output["ahp_from_threshold_mv"] = features["Afterhyperpolarization (mV)"]
    output["fast_trough_mv"] = (
        output["ap_threshold_mv"] + output["ahp_from_threshold_mv"]
    )
    output["upstroke_downstroke_ratio"] = features["Upstroke-to-downstroke ratio"]
    output["baseline_voltage_mv"] = features["Resting membrane potential (mV)"]
    output["latency_ms"] = features["Latency (ms)"]
    output["isi_cv"] = features["ISI coefficient of variation"]
    output["isi_adaptation_index"] = features["ISI adaptation index"]
    output["max_firing_rate_hz"] = features["Max number of APs"] / 0.6
    output["input_resistance_mohm"] = features["Input resistance (MOhm)"]
    output["membrane_tau_ms"] = features["Membrane time constant (ms)"]
    output["rheobase_pa"] = features["Rheobase (pA)"]
    output["sag_ratio"] = _scala_sag_ratio_to_fraction(features["Sag ratio"])
    return output.reset_index(drop=True)


def load_scala_features(
    data_dir: str | Path = "data/external",
    include_physiological_temperature: bool = True,
) -> pd.DataFrame:
    """Load the Scala M1 scalar e-features and selected transcriptomic metadata."""
    root = Path(data_dir) / "scala"
    frames = [
        _load_scala_release(
            root / "m1_patchseq_ephys_features.csv",
            root / "m1_patchseq_meta_data.csv",
            "scala_room_temperature",
            22.0,
        )
    ]
    if include_physiological_temperature:
        frames.append(
            _load_scala_release(
                root / "m1_patchseq_phys_temp_ephys_features.csv",
                root / "m1_patchseq_phys_temp_meta_data.csv",
                "scala_physiological_temperature",
                34.0,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _matlab_strings(values: np.ndarray) -> list[str]:
    strings = []
    for value in np.asarray(values).ravel():
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        strings.append(str(value))
    return strings


def load_gouwens_features(data_dir: str | Path = "data/external") -> pd.DataFrame:
    """Load Gouwens IPFX features and invert the release's z-scoring."""
    path = Path(data_dir) / "gouwens" / "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
    content = loadmat(path)
    feature_names = _matlab_strings(content["feature_name"])
    normalized = np.asarray(content["E_feature"], dtype=float)
    feature_mean = np.asarray(content["feature_mean"], dtype=float).ravel()
    feature_std = np.asarray(content["feature_std"], dtype=float).ravel()
    physical = pd.DataFrame(
        normalized * feature_std + feature_mean,
        columns=feature_names,
    )

    output = pd.DataFrame(index=np.arange(len(physical)))
    output["dataset"] = "gouwens_visp"
    output["cell_id"] = (
        np.asarray(content["E_spec_id_label"]).ravel().astype(np.int64).astype(str)
    )
    output["temperature_c"] = np.nan
    output["transcriptomic_type"] = _matlab_strings(content["cluster"])
    output["transcriptomic_family"] = pd.NA
    output["cre_line"] = pd.NA
    output["ap_threshold_mv"] = physical["ap_1_threshold_v_0_long_square"]
    output["ap_peak_mv"] = physical["ap_1_peak_v_0_long_square"]
    output["ap_amplitude_mv"] = output["ap_peak_mv"] - output["ap_threshold_mv"]
    output["ap_width_ms"] = physical["ap_1_width_0_long_square"] * 1000.0
    output["fast_trough_mv"] = physical["ap_1_fast_trough_v_0_long_square"]
    output["ahp_from_threshold_mv"] = (
        output["fast_trough_mv"] - output["ap_threshold_mv"]
    )
    output["upstroke_mv_ms"] = physical["ap_1_upstroke_0_long_square"]
    output["downstroke_mv_ms"] = physical["ap_1_downstroke_0_long_square"]
    output["upstroke_downstroke_ratio"] = physical[
        "ap_1_upstroke_downstroke_ratio_0_long_square"
    ]
    output["baseline_voltage_mv"] = physical["v_baseline"]
    output["latency_ms"] = physical["latency_0_long_square"] * 1000.0
    output["rheobase_firing_rate_hz"] = physical["avg_rate_0_long_square"]
    output["input_resistance_mohm"] = physical["input_resistance"]
    output["membrane_tau_ms"] = physical["tau"] * 1000.0
    output["rheobase_pa"] = physical["rheobase_i"]
    output["sag_ratio"] = physical["sag_nearest_minus_100"]
    return output


def load_biological_features(
    data_dir: str | Path = "data/external",
    include_scala_physiological_temperature: bool = True,
) -> pd.DataFrame:
    """Load both biological releases into one semantic feature table."""
    scala = load_scala_features(
        data_dir,
        include_physiological_temperature=include_scala_physiological_temperature,
    )
    gouwens = load_gouwens_features(data_dir)
    return pd.concat((scala, gouwens), ignore_index=True, sort=False)


def _model_column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        column = f"feature__{name}"
        if column in frame:
            return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def harmonize_model_features(model_dataset: pd.DataFrame) -> pd.DataFrame:
    """Map simulator feature columns onto the biological semantic schema."""
    output = pd.DataFrame(index=model_dataset.index)
    output["dataset"] = "hh_model"
    output["cell_id"] = model_dataset["sample_id"].astype(str)
    output["ap_threshold_mv"] = _model_column(
        model_dataset, "first_threshold_voltage_mv", "mean_threshold_voltage_mv"
    )
    output["ap_peak_mv"] = _model_column(
        model_dataset, "first_peak_voltage_mv", "mean_peak_voltage_mv"
    )
    output["ap_amplitude_mv"] = _model_column(
        model_dataset, "first_ap_amplitude_mv", "mean_ap_amplitude_mv"
    )
    output["ap_width_ms"] = _model_column(
        model_dataset, "first_half_width_ms", "mean_half_width_ms"
    )
    output["fast_trough_mv"] = _model_column(
        model_dataset, "first_fast_trough_voltage_mv", "mean_ahp_voltage_mv"
    )
    output["ahp_from_threshold_mv"] = (
        output["fast_trough_mv"] - output["ap_threshold_mv"]
    )
    output["upstroke_mv_ms"] = _model_column(
        model_dataset, "first_upstroke_mv_ms", "mean_max_dvdt_mv_ms"
    )
    output["downstroke_mv_ms"] = _model_column(
        model_dataset, "first_downstroke_mv_ms", "mean_min_dvdt_mv_ms"
    )
    output["upstroke_downstroke_ratio"] = _model_column(
        model_dataset, "first_upstroke_downstroke_ratio"
    )
    missing_ratio = output["upstroke_downstroke_ratio"].isna()
    output.loc[missing_ratio, "upstroke_downstroke_ratio"] = (
        output.loc[missing_ratio, "upstroke_mv_ms"]
        / output.loc[missing_ratio, "downstroke_mv_ms"].abs()
    )
    output["baseline_voltage_mv"] = _model_column(
        model_dataset, "resting_voltage_mv", "baseline_voltage_mv"
    )
    output["latency_ms"] = _model_column(model_dataset, "first_spike_latency_ms")
    output["rheobase_firing_rate_hz"] = _model_column(
        model_dataset, "firing_rate_hz"
    )
    output["input_resistance_mohm"] = _model_column(
        model_dataset, "input_resistance_mohm"
    )
    output["membrane_tau_ms"] = _model_column(model_dataset, "membrane_tau_ms")
    output["rheobase_pa"] = _model_column(model_dataset, "rheobase_pa")
    output["sag_ratio"] = _model_column(model_dataset, "sag_ratio")
    spike_cycle_mapping = {
        "ap_upstroke_time_ms": "first_ap_upstroke_time_ms",
        "ap_repolarization_time_ms": "first_ap_repolarization_time_ms",
        "ap_duration_ms": "first_ap_duration_ms",
        "onset_voltage_mv": "first_onset_voltage_mv",
        "onset_dvdt_mv_ms": "first_onset_dvdt_mv_ms",
        "onset_rapidness_per_ms": "first_onset_rapidness_per_ms",
        "max_acceleration_mv_ms2": "first_max_acceleration_mv_ms2",
        "max_acceleration_voltage_mv": "first_max_acceleration_voltage_mv",
        "min_acceleration_mv_ms2": "first_min_acceleration_mv_ms2",
        "min_acceleration_voltage_mv": "first_min_acceleration_voltage_mv",
        "upstroke_inflection_voltage_mv": "upstroke_inflection_voltage_mv",
        "upstroke_inflection_dvdt_mv_ms": "upstroke_inflection_dvdt_mv_ms",
        "downstroke_inflection_voltage_mv": "downstroke_inflection_voltage_mv",
        "downstroke_inflection_dvdt_mv_ms": "downstroke_inflection_dvdt_mv_ms",
        "upstroke_inflection_relative_to_threshold_mv": (
            "upstroke_inflection_relative_to_threshold_mv"
        ),
        "downstroke_inflection_relative_to_peak_mv": (
            "downstroke_inflection_relative_to_peak_mv"
        ),
        "ap_phase_area_v2_ms": "first_ap_phase_area_v2_ms",
        "ap_phase_area_normalized": "first_ap_phase_area_normalized",
        "cycle_phase_area_v2_ms": "first_spike_phase_area_v2_ms",
        "cycle_phase_area_normalized": "first_cycle_phase_area_normalized",
        "cycle_phase_path_length_normalized": (
            "first_cycle_phase_path_length_normalized"
        ),
    }
    for output_name, model_name in spike_cycle_mapping.items():
        output[output_name] = _model_column(model_dataset, model_name)
    invalid_waveform = (
        (output["ap_peak_mv"] <= output["ap_threshold_mv"])
        | (output["ap_width_ms"] <= 0.0)
        | (output["upstroke_mv_ms"] <= 0.0)
        | (output["downstroke_mv_ms"] >= 0.0)
        | (output["upstroke_downstroke_ratio"] <= 0.0)
    )
    output["waveform_valid"] = ~invalid_waveform
    output.loc[invalid_waveform, list(WAVEFORM_CORE_FEATURES)] = np.nan
    return output.reset_index(drop=True)


def harmonize_raw_spike_cycle_features(
    raw_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Map raw NWB extraction output onto the biological semantic schema."""
    required = {"dataset", "cell_id", "raw_status"}
    missing = required.difference(raw_dataset.columns)
    if missing:
        raise ValueError(f"Raw spike-cycle table is missing columns: {sorted(missing)}")

    raw = raw_dataset.loc[raw_dataset["raw_status"].eq("ok")].copy()
    raw["sample_id"] = raw["cell_id"].astype(str)
    output = harmonize_model_features(raw)
    output["dataset"] = raw["dataset"].astype(str).to_numpy()
    output["cell_id"] = raw["cell_id"].astype(str).to_numpy()
    for column in (
        "raw_status",
        "nwb_path",
        "sweep_number",
        "sampling_rate_hz",
        "stimulus_start_ms",
        "stimulus_end_ms",
        "sampled_rheobase_pa",
        "stimulus_plateau_mad_pa",
        "spike_count",
    ):
        if column in raw:
            output[column] = raw[column].to_numpy()
    return output


def load_local_spike_cycle_features(
    raw_spike_cycle_path: str | Path,
    data_dir: str | Path = "data/external",
    include_scala_physiological_temperature: bool = True,
) -> pd.DataFrame:
    """Join raw NWB spike-cycle features to processed Patch-seq metadata."""
    raw = pd.read_csv(raw_spike_cycle_path, dtype={"cell_id": "string"})
    raw_semantic = harmonize_raw_spike_cycle_features(raw)
    biological = load_biological_features(
        data_dir,
        include_scala_physiological_temperature=include_scala_physiological_temperature,
    )
    metadata_columns = (
        "dataset",
        "cell_id",
        "temperature_c",
        "transcriptomic_type",
        "transcriptomic_family",
        "cre_line",
    )
    metadata = biological.loc[:, metadata_columns].drop_duplicates(
        ["dataset", "cell_id"]
    )
    return metadata.merge(
        raw_semantic,
        on=["dataset", "cell_id"],
        how="inner",
        validate="one_to_one",
    )
