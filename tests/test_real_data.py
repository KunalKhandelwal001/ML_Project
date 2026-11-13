"""Optional checks against the real Framingham CSV.

The real dataset is not committed, so these tests are skipped on a fresh clone
and run automatically when ``data/framingham.csv`` is present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_data

REAL_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "framingham.csv"

pytestmark = pytest.mark.skipif(
    not REAL_DATA_PATH.exists(),
    reason="data/framingham.csv is not present",
)


def test_real_framingham_schema_and_split():
    bundle = load_data(REAL_DATA_PATH, random_state=42)

    assert bundle.n_features == 15
    assert bundle.n_train + bundle.n_val + bundle.n_test == 4240
    assert bundle.metadata["split_sizes"] == {
        "train": 2968,
        "val": 424,
        "test": 848,
    }
    assert bundle.metadata["target_name"] == "TenYearCHD"
    assert bundle.y_train.nunique() == 2
    assert bundle.y_val.nunique() == 2
    assert bundle.y_test.nunique() == 2
    assert bundle.X_train.isna().sum().sum() == 0
    assert bundle.X_val.isna().sum().sum() == 0
    assert bundle.X_test.isna().sum().sum() == 0


def test_real_framingham_has_expected_class_balance():
    bundle = load_data(REAL_DATA_PATH, random_state=42)

    for split_name, target in (
        ("train", bundle.y_train),
        ("val", bundle.y_val),
        ("test", bundle.y_test),
    ):
        rate = float(target.mean())
        assert 0.14 < rate < 0.17, f"{split_name} positive rate: {rate}"


def test_real_data_is_not_used_as_a_feature():
    bundle = load_data(REAL_DATA_PATH, random_state=42)

    assert "TenYearCHD" not in bundle.feature_names
    assert bundle.feature_names == (
        "male",
        "age",
        "education",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    )


def test_real_data_missing_values_are_recorded():
    bundle = load_data(REAL_DATA_PATH, random_state=42)

    before = bundle.metadata["missing_values_before_imputation"]
    assert sum(before.values()) == 645
    assert all(value == 0 for value in bundle.metadata[
        "missing_values_after_imputation"
    ].values())
    assert pd.concat([bundle.y_train, bundle.y_val, bundle.y_test]).sum() == 644
