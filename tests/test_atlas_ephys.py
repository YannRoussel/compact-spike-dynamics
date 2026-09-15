from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from inverse_ephys_alpha_beta.atlas_ephys import (
    AP_FEATURE_NAMES,
    WAVEFORM_TARGET_NAMES,
    atlas_broad_class,
    ap_feature_target_matrix,
    fit_fixed_class_rrr,
    harmonize_atlas_to_patchseq,
    inverse_waveform_target_matrix,
    inverse_ap_feature_target_matrix,
    load_allen_reference_atlas,
    predict_fixed_class_rrr,
    spike_cycle_features,
    spike_cycle_from_row,
    waveform_parameter_frame,
    waveform_target_matrix,
)


def _waveform_row(scale=1.0):
    row = {name: 0.0 for name in WAVEFORM_TARGET_NAMES}
    row.update(
        {
            "resting_voltage_mv": -65.0,
            "template_mean_relative_mv_q050": 20.0,
            "template_cos_h01_mv_q050": 35.0 * scale,
            "downstroke_duration_ms_q050": 1.0,
            "upstroke_duration_ms_q050": 0.5,
            "recovery_baseline_period_q050_ms": 12.0,
        }
    )
    return row


def test_waveform_target_roundtrip_and_features():
    frame = pd.DataFrame([_waveform_row(), _waveform_row(0.8)])
    target = waveform_target_matrix(frame)
    recovered = inverse_waveform_target_matrix(target)
    assert np.allclose(recovered, frame.loc[:, WAVEFORM_TARGET_NAMES])
    cycle = spike_cycle_from_row(waveform_parameter_frame(recovered).iloc[0])
    features = spike_cycle_features(cycle)
    assert set(features) == set(AP_FEATURE_NAMES)
    assert np.isfinite(list(features.values())).all()
    assert features["ap_amplitude_mv"] > 60.0
    assert features["max_upstroke_mv_ms"] > 0.0
    assert features["max_downstroke_mv_ms"] < 0.0
    feature_frame = pd.DataFrame([features], columns=AP_FEATURE_NAMES)
    transformed = ap_feature_target_matrix(feature_frame)
    assert np.allclose(
        inverse_ap_feature_target_matrix(transformed),
        feature_frame,
    )


def test_atlas_loader_keeps_only_neural_clusters(tmp_path):
    path = tmp_path / "trimmed.csv"
    pd.DataFrame(
        {
            "feature": ["Scn1a", "Kcnc1"],
            "1_CR": [2.0, 1.0],
            "108_Pvalb": [5.0, 7.0],
            "229_L6 IT CTX": [3.0, 4.0],
            "365_Oligo": [0.0, 0.0],
        }
    ).set_index("feature").to_csv(path)
    expression, metadata = load_allen_reference_atlas(path)
    assert expression.shape == (3, 2)
    assert set(metadata["broad_class"]) == {"Lamp5", "Pvalb", "Glut_IT"}
    assert "365_Oligo" not in expression.index


def test_atlas_broad_class_handles_reference_aliases():
    assert atlas_broad_class("67_Sst") == "Sst"
    assert atlas_broad_class("123_Pvalb Vipr2") == "Pvalb"
    assert atlas_broad_class("253_L5 PT CTX") == "Glut_ET"
    assert atlas_broad_class("279_L6 CT CTX") == "Glut_CT"
    assert atlas_broad_class("303_L6b CTX") == "Glut_L6b"


def test_harmonization_and_fixed_class_rrr_are_finite():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(36, 12))
    classes = np.asarray(["Pvalb"] * 18 + ["Sst"] * 18, dtype=object)
    types = np.asarray(
        ["Pvalb A"] * 9
        + ["Pvalb B"] * 9
        + ["Sst A"] * 9
        + ["Sst B"] * 9,
        dtype=object,
    )
    y = x[:, :4] @ rng.normal(size=(4, 3)) + rng.normal(scale=0.1, size=(36, 3))
    atlas = pd.DataFrame(
        rng.normal(size=(5, 12)),
        columns=[f"g{index}" for index in range(12)],
    )
    harmonized = harmonize_atlas_to_patchseq(
        atlas,
        x,
        types,
        atlas.columns,
    )
    model = fit_fixed_class_rrr(
        x,
        y,
        classes,
        rank=2,
        ridge_penalty=1.0,
        sparse_ratio=0.05,
    )
    predicted, baseline = predict_fixed_class_rrr(
        model,
        x,
        y,
        classes,
        harmonized,
        ["Pvalb"] * len(harmonized),
    )
    assert harmonized.shape == (5, 12)
    assert predicted.shape == baseline.shape == (5, 3)
    assert np.isfinite(predicted).all()


def load_tests(loader, tests, pattern):
    """Include function-style atlas checks in the project's unittest suite."""
    def check_loader():
        with TemporaryDirectory() as directory:
            test_atlas_loader_keeps_only_neural_clusters(Path(directory))

    return unittest.TestSuite(
        unittest.FunctionTestCase(check)
        for check in (
            test_waveform_target_roundtrip_and_features,
            check_loader,
            test_atlas_broad_class_handles_reference_aliases,
            test_harmonization_and_fixed_class_rrr_are_finite,
        )
    )
