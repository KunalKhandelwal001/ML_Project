"""Data loading, cleaning, splitting, and scaling for FedDef.

The public entry point is :func:`load_data` (CSV path) or
:func:`prepare_data` (already-loaded DataFrame, useful for Streamlit).

Key guarantees:
- The target is binary and encoded as 0/1.
- The split is stratified and deterministic for a given seed.
- Imputation, one-hot encoding, and scaling are fitted on the training
  split only.  Validation and test rows never influence preprocessing.
- The returned ``DataBundle`` contains no missing or non-finite values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_TARGET = "TenYearCHD"

# Common identifier columns that must not be used as model features.
ID_COLUMNS: tuple[str, ...] = (
    "id",
    "ID",
    "Id",
    "patient_id",
    "PatientID",
    "Unnamed: 0",
)


@dataclass(frozen=True)
class DataBundle:
    """Prepared train/validation/test data.

    ``X_train``, ``X_val``, and ``X_test`` are already imputed, encoded, and
    (by default) standardised.  ``scaler`` is the fitted scaler, or ``None``
    when ``scale=False``.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: tuple[str, ...]
    scaler: StandardScaler | None
    preprocessor: ColumnTransformer
    target_name: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")

        for split_name, X, y in (
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ):
            if len(X) != len(y):
                raise ValueError(f"{split_name} X and y have different lengths")
            if not X.index.equals(y.index):
                raise ValueError(f"{split_name} X and y indices are not aligned")
            if tuple(X.columns) != self.feature_names:
                raise ValueError(f"{split_name} columns do not match feature_names")
            if X.isna().any().any():
                raise ValueError(f"{split_name} contains missing values after imputation")
            if not np.isfinite(X.to_numpy(dtype=float)).all():
                raise ValueError(f"{split_name} contains non-finite values")

        if self.target_name in self.feature_names:
            raise ValueError("target column must not appear in feature_names")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_data(
    path: str | Path,
    *,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
) -> DataBundle:
    """Load a CSV file and return a prepared :class:`DataBundle`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place framingham.csv in data/ "
            "or pass an explicit --data path. See data/README.md."
        )
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    frame = pd.read_csv(path)
    return prepare_data(
        frame,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        source=str(path),
    )


def prepare_data(
    frame: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    source: str | None = None,
) -> DataBundle:
    """Clean ``frame`` and create aligned train/validation/test splits."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("Input dataframe is empty")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")
    if not isinstance(scale, bool):
        raise TypeError("scale must be a boolean")

    source = str(source) if source is not None else None
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not allowed: {duplicated}")

    target_col = target_col.strip()
    if target_col not in frame.columns:
        raise ValueError(
            f"Target column {target_col!r} not found. "
            f"Available columns: {list(frame.columns)}"
        )

    n_rows_original = len(frame)

    # Identifier columns are not features. Never drop the requested target,
    # even if its name happens to appear in the identifier list.
    id_columns_dropped = [
        column
        for column in ID_COLUMNS
        if column in frame.columns and column != target_col
    ]
    if id_columns_dropped:
        frame = frame.drop(columns=id_columns_dropped)

    # The target must be present; feature rows may be imputed later.
    frame = frame.dropna(subset=[target_col]).copy()
    n_rows_dropped_missing_target = n_rows_original - len(frame)
    if frame.empty:
        raise ValueError("No rows remain after dropping missing target values")

    _validate_binary_target(frame, target_col)

    if len(frame.columns) < 2:
        raise ValueError("Dataset must contain at least one feature besides the target")

    train_df, val_df, test_df = _split_three_way(
        frame,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_val_raw = val_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.drop(columns=[target_col]).copy()

    y_train = train_df[target_col].astype(int).copy()
    y_val = val_df[target_col].astype(int).copy()
    y_test = test_df[target_col].astype(int).copy()

    # A column that is entirely missing in training cannot be imputed from
    # training data, so drop it from every split.
    all_missing_train_columns = [
        column for column in X_train_raw.columns if X_train_raw[column].isna().all()
    ]
    if all_missing_train_columns:
        X_train_raw = X_train_raw.drop(columns=all_missing_train_columns)
        X_val_raw = X_val_raw.drop(columns=all_missing_train_columns)
        X_test_raw = X_test_raw.drop(columns=all_missing_train_columns)

    if X_train_raw.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning")

    numeric_columns, categorical_columns = _classify_columns(X_train_raw)

    # Make numeric columns consistently numeric and categorical columns
    # consistently string-valued before fitting the preprocessing pipeline.
    _coerce_numeric_columns(
        numeric_columns,
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )
    _stringify_categorical_columns(
        categorical_columns,
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    missing_before = {
        "train": int(X_train_raw.isna().sum().sum()),
        "val": int(X_val_raw.isna().sum().sum()),
        "test": int(X_test_raw.isna().sum().sum()),
    }

    imputation_values = _imputation_values(
        X_train_raw,
        numeric_columns,
        categorical_columns,
    )

    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    preprocessor.fit(X_train_raw)
    feature_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Preprocessing produced duplicate feature names. Rename the "
            "colliding numeric/categorical columns before training."
        )

    train_matrix = _transform(preprocessor, X_train_raw, feature_names)
    val_matrix = _transform(preprocessor, X_val_raw, feature_names)
    test_matrix = _transform(preprocessor, X_test_raw, feature_names)

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(train_matrix)
        val_matrix = scaler.transform(val_matrix)
        test_matrix = scaler.transform(test_matrix)

    X_train = _to_frame(train_matrix, feature_names, X_train_raw.index)
    X_val = _to_frame(val_matrix, feature_names, X_val_raw.index)
    X_test = _to_frame(test_matrix, feature_names, X_test_raw.index)

    metadata: dict[str, Any] = {
        "data_summary_version": 1,
        "source": source,
        "target_name": target_col,
        "target_values": [0, 1],
        "seed": random_state,
        "test_size": test_size,
        "val_size": val_size,
        "scaled": scale,
        "n_rows_original": n_rows_original,
        "n_rows_after_target_drop": int(len(frame)),
        "n_rows_dropped_missing_target": int(n_rows_dropped_missing_target),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "id_columns_dropped": id_columns_dropped,
        "all_missing_train_columns_dropped": all_missing_train_columns,
        "dropped_columns": list(
            dict.fromkeys(id_columns_dropped + all_missing_train_columns)
        ),
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "class_counts": {
            "train": _class_counts(y_train),
            "val": _class_counts(y_val),
            "test": _class_counts(y_test),
        },
        "positive_rate": {
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "missing_values_before_imputation": missing_before,
        "missing_values_after_imputation": {
            "train": int(X_train.isna().sum().sum()),
            "val": int(X_val.isna().sum().sum()),
            "test": int(X_test.isna().sum().sum()),
        },
        "imputation_values": imputation_values,
        "preprocessing": (
            "median imputation for numeric columns; most-frequent imputation "
            "and one-hot encoding for categorical columns; all fitted on the "
            "training split only"
        ),
        "split_protocol": "stratified train/validation/test split",
    }

    return DataBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        preprocessor=preprocessor,
        target_name=target_col,
        metadata=metadata,
    )


def save_data_summary(bundle: DataBundle, path: str | Path) -> Path:
    """Write the split metadata to a JSON file."""
    path = Path(path)
    payload = dict(bundle.metadata)
    payload["feature_names"] = list(bundle.feature_names)
    payload["n_features"] = bundle.n_features
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def save_splits(bundle: DataBundle, out_dir: str | Path) -> Path:
    """Save the prepared (scaled) splits as CSV files.

    Each file contains the feature columns plus the target column.  These are
    the exact matrices used by the models; raw data should be kept separately.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, X, y in (
        ("train", bundle.X_train, bundle.y_train),
        ("val", bundle.X_val, bundle.y_val),
        ("test", bundle.X_test, bundle.y_test),
    ):
        prepared = X.copy()
        prepared[bundle.target_name] = y
        prepared.to_csv(out_dir / f"{split_name}.csv", index=False)
    return out_dir


def _validate_binary_target(frame: pd.DataFrame, target_col: str) -> None:
    try:
        target = pd.to_numeric(frame[target_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Target column {target_col!r} must contain numeric 0/1 values"
        ) from exc

    if target.isna().any():
        raise ValueError(
            f"Target column {target_col!r} still contains missing values"
        )
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Target column {target_col!r} contains non-finite values"
        )

    unique = np.sort(target.unique())
    if len(unique) != 2 or not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(
            f"Target column {target_col!r} must be binary with values 0 and 1; "
            f"found {unique.tolist()}"
        )

    frame[target_col] = target.astype(int)


def _split_three_way(
    frame: pd.DataFrame,
    *,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_val_df, test_df = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame[target_col],
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create a stratified train/test split. The dataset may "
            "be too small or too imbalanced for the requested split sizes."
        ) from exc

    val_fraction = val_size / (1.0 - test_size)
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            random_state=random_state + 1,
            stratify=train_val_df[target_col],
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create a stratified train/validation split. The "
            "dataset may be too small or too imbalanced for the requested "
            "split sizes."
        ) from exc

    for split_name, split in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        if split.empty:
            raise ValueError(f"{split_name} split is empty")
        if split[target_col].nunique() != 2:
            raise ValueError(
                f"{split_name} split contains only one class. Use more data "
                "or adjust test_size/val_size."
            )

    return train_df, val_df, test_df


def _classify_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical groups.

    Object columns whose non-missing values are all numeric strings are
    treated as numeric.
    """
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        non_missing = int(series.notna().sum())
        if non_missing == 0:
            categorical_columns.append(column)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        if int(converted.notna().sum()) == non_missing:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def _coerce_numeric_columns(
    numeric_columns: list[str],
    *frames: pd.DataFrame,
) -> None:
    for column in numeric_columns:
        for frame in frames:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.replace([np.inf, -np.inf], np.nan).astype(float)


def _stringify_categorical_columns(
    categorical_columns: list[str],
    *frames: pd.DataFrame,
) -> None:
    for column in categorical_columns:
        for frame in frames:
            frame[column] = frame[column].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )


def _imputation_values(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in numeric_columns:
        median = frame[column].median()
        values[column] = float(median) if pd.notna(median) else None
    for column in categorical_columns:
        mode = frame[column].mode(dropna=True)
        values[column] = str(mode.iloc[0]) if not mode.empty else None
    return values


def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No numeric or categorical feature columns found")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _transform(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    matrix = preprocessor.transform(frame)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Preprocessor did not produce a 2-D matrix")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Preprocessor output width does not match feature_names "
            f"({matrix.shape[1]} != {len(feature_names)})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed features contain NaN or infinite values")
    return matrix


def _to_frame(
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    index: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=list(feature_names), index=index)


def _class_counts(target: pd.Series) -> dict[str, int]:
    return {
        "0": int((target == 0).sum()),
        "1": int((target == 1).sum()),
    }
