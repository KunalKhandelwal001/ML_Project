from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_data, prepare_data, save_data_summary, save_splits


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_split_sizes_and_shapes(synthetic_framingham, synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)

    assert bundle.metadata["split_sizes"] == {
        "train": len(bundle.X_train),
        "val": len(bundle.X_val),
        "test": len(bundle.X_test),
    }
    assert len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test) == len(
        synthetic_framingham
    )
    assert bundle.X_train.shape[1] == bundle.n_features
    assert bundle.X_val.shape[1] == bundle.n_features
    assert bundle.X_test.shape[1] == bundle.n_features
    assert bundle.y_train.nunique() == 2
    assert bundle.y_val.nunique() == 2
    assert bundle.y_test.nunique() == 2


def test_splits_have_no_overlap_and_cover_all_rows(synthetic_framingham, synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)

    train_index = set(bundle.X_train.index)
    val_index = set(bundle.X_val.index)
    test_index = set(bundle.X_test.index)

    assert train_index.isdisjoint(val_index)
    assert train_index.isdisjoint(test_index)
    assert val_index.isdisjoint(test_index)
    assert train_index | val_index | test_index == set(synthetic_framingham.index)


def test_same_seed_is_deterministic(synthetic_csv):
    first = load_data(synthetic_csv, random_state=42)
    second = load_data(synthetic_csv, random_state=42)

    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_frame_equal(first.X_val, second.X_val)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_train, second.y_train)
    pd.testing.assert_series_equal(first.y_val, second.y_val)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_different_seed_changes_the_split(synthetic_csv):
    first = load_data(synthetic_csv, random_state=42)
    second = load_data(synthetic_csv, random_state=7)

    assert not first.X_train.index.equals(second.X_train.index)


def test_scaler_is_fit_on_training_only(synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)

    means = bundle.X_train.mean(axis=0).to_numpy()
    stds = bundle.X_train.std(axis=0, ddof=0).to_numpy()

    assert np.allclose(means, 0.0, atol=1e-8)
    assert np.allclose(stds[stds > 0], 1.0, atol=1e-8)


def test_validation_and_test_values_do_not_affect_preprocessing(
    synthetic_framingham,
    synthetic_csv,
):
    bundle = load_data(synthetic_csv, random_state=42)
    feature_columns = [
        column
        for column in synthetic_framingham.columns
        if column not in {"id", "TenYearCHD"}
    ]

    modified = synthetic_framingham.copy()
    changed_rows = bundle.X_val.index.union(bundle.X_test.index)
    modified.loc[changed_rows, feature_columns] = (
        modified.loc[changed_rows, feature_columns] + 1000.0
    )
    modified_path = synthetic_csv.with_name("modified.csv")
    _write_csv(modified, modified_path)

    changed_bundle = load_data(modified_path, random_state=42)

    np.testing.assert_allclose(
        bundle.scaler.mean_,
        changed_bundle.scaler.mean_,
    )
    np.testing.assert_allclose(
        bundle.scaler.scale_,
        changed_bundle.scaler.scale_,
    )
    assert (
        bundle.metadata["imputation_values"]
        == changed_bundle.metadata["imputation_values"]
    )


def test_missing_values_are_imputed(synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)

    assert bundle.metadata["missing_values_before_imputation"]["train"] > 0
    assert bundle.X_train.isna().sum().sum() == 0
    assert bundle.X_val.isna().sum().sum() == 0
    assert bundle.X_test.isna().sum().sum() == 0
    assert bundle.metadata["missing_values_after_imputation"] == {
        "train": 0,
        "val": 0,
        "test": 0,
    }


def test_rows_with_missing_target_are_dropped(synthetic_framingham, synthetic_csv):
    frame = synthetic_framingham.copy()
    frame.loc[frame.index[:7], "TenYearCHD"] = np.nan
    path = _write_csv(frame, synthetic_csv.with_name("missing_target.csv"))

    bundle = load_data(path, random_state=42)

    assert bundle.metadata["n_rows_dropped_missing_target"] == 7
    assert sum(bundle.metadata["split_sizes"].values()) == len(frame) - 7


def test_id_columns_are_not_features(synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)

    assert "id" not in bundle.feature_names
    assert bundle.metadata["id_columns_dropped"] == ["id"]


def test_stratification_preserves_class_balance(synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42)
    overall = float(
        pd.concat([bundle.y_train, bundle.y_val, bundle.y_test]).mean()
    )

    for split_name, target in (
        ("train", bundle.y_train),
        ("val", bundle.y_val),
        ("test", bundle.y_test),
    ):
        assert abs(float(target.mean()) - overall) < 0.05, split_name


def test_invalid_target_raises(synthetic_framingham, synthetic_csv):
    frame = synthetic_framingham.copy()
    frame["TenYearCHD"] = 1
    path = _write_csv(frame, synthetic_csv.with_name("one_class.csv"))

    with pytest.raises(ValueError, match="binary"):
        load_data(path, random_state=42)


def test_missing_target_column_raises(synthetic_framingham, synthetic_csv):
    frame = synthetic_framingham.rename(columns={"TenYearCHD": "target"})
    path = _write_csv(frame, synthetic_csv.with_name("wrong_target.csv"))

    with pytest.raises(ValueError, match="not found"):
        load_data(path, random_state=42)


def test_target_is_not_dropped_as_an_id_column(synthetic_framingham, tmp_path):
    frame = synthetic_framingham.drop(columns=["id"]).rename(
        columns={"TenYearCHD": "id"}
    )
    path = _write_csv(frame, tmp_path / "id_target.csv")

    bundle = load_data(path, target_col="id", random_state=42)

    assert bundle.target_name == "id"
    assert "id" not in bundle.feature_names


def test_categorical_columns_are_one_hot_encoded(synthetic_categorical, tmp_path):
    path = _write_csv(synthetic_categorical, tmp_path / "categorical.csv")

    bundle = load_data(path, random_state=42)

    assert any(name.startswith("region_") for name in bundle.feature_names)
    assert bundle.X_train.isna().sum().sum() == 0
    assert bundle.X_val.isna().sum().sum() == 0
    assert bundle.X_test.isna().sum().sum() == 0


def test_scale_false_returns_unscaled_features(synthetic_csv):
    bundle = load_data(synthetic_csv, random_state=42, scale=False)

    assert bundle.scaler is None
    assert not np.allclose(bundle.X_train.mean(axis=0).to_numpy(), 0.0, atol=1e-8)
    assert bundle.metadata["scaled"] is False


def test_invalid_split_sizes_raise(synthetic_csv):
    with pytest.raises(ValueError):
        load_data(synthetic_csv, test_size=0.8, val_size=0.3)


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "does_not_exist.csv")


def test_save_data_summary(synthetic_csv, tmp_path):
    bundle = load_data(synthetic_csv, random_state=42)
    path = save_data_summary(bundle, tmp_path / "summary.json")

    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["split_sizes"] == bundle.metadata["split_sizes"]
    assert summary["positive_rate"] == bundle.metadata["positive_rate"]
    assert summary["seed"] == 42
    assert summary["target_name"] == "TenYearCHD"
    assert summary["feature_names"] == list(bundle.feature_names)
    assert summary["n_features"] == bundle.n_features


def test_save_splits(synthetic_csv, tmp_path):
    bundle = load_data(synthetic_csv, random_state=42)
    out_dir = save_splits(bundle, tmp_path / "splits")

    for split_name, X, y in (
        ("train", bundle.X_train, bundle.y_train),
        ("val", bundle.X_val, bundle.y_val),
        ("test", bundle.X_test, bundle.y_test),
    ):
        path = out_dir / f"{split_name}.csv"
        assert path.exists()
        saved = pd.read_csv(path)
        assert len(saved) == len(X)
        assert list(saved.columns) == [*bundle.feature_names, bundle.target_name]
        assert saved[bundle.target_name].tolist() == y.tolist()


def test_prepare_data_accepts_dataframe(synthetic_framingham):
    bundle = prepare_data(synthetic_framingham, random_state=42, source="memory")

    assert bundle.metadata["source"] == "memory"
    assert bundle.n_train + bundle.n_val + bundle.n_test == len(synthetic_framingham)
