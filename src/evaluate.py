"""Evaluation metrics and result saving for FedDef.

The project predicts a binary target (TenYearCHD), so every metric here is
computed for the positive class (label 1).  Accuracy alone is misleading on
this imbalanced dataset, which is why precision, recall, F1, and ROC-AUC are
reported alongside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BINARY_LABELS: tuple[int, int] = (0, 1)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute binary classification metrics.

    ``y_prob`` may be:
    - a 1-D array of positive-class scores/probabilities, or
    - a 2-D array with two columns, where column 1 is the positive class
      (as returned by ``predict_proba``).
    """
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)

    if y_true_array.size == 0:
        raise ValueError("y_true must not be empty")
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"({y_true_array.size} != {y_pred_array.size})"
        )

    _validate_binary_vector(y_true_array, "y_true")
    _validate_binary_vector(y_pred_array, "y_pred")

    labels = np.array(BINARY_LABELS)
    cm = confusion_matrix(y_true_array, y_pred_array, labels=labels)
    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    scores = _positive_scores(y_prob, expected_length=y_true_array.size)
    roc_auc: float | None = None
    if scores is not None and np.unique(y_true_array).size == 2:
        try:
            roc_auc = float(roc_auc_score(y_true_array, scores))
        except ValueError:
            # Undefined when all scores are identical.
            roc_auc = None

    n_samples = int(y_true_array.size)
    n_positive = int((y_true_array == 1).sum())
    n_negative = int((y_true_array == 0).sum())

    return {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": float(n_positive / n_samples),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": report,
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Evaluate a fitted classifier using predictions and probabilities."""
    y_pred = np.asarray(model.predict(X)).reshape(-1)

    y_prob = None
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            y_prob = predict_proba(X)
        except (AttributeError, NotImplementedError):
            y_prob = None

    return evaluate_predictions(y, y_pred, y_prob)


def save_metrics(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Save a metrics dictionary as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return values


def format_metrics(metrics: Mapping[str, Any]) -> str:
    """Return a compact human-readable metrics summary."""
    roc_auc = metrics.get("roc_auc")
    roc_auc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return "\n".join(
        [
            f"samples:         {int(metrics['n_samples'])}",
            f"accuracy:        {float(metrics['accuracy']):.4f}",
            f"precision:       {float(metrics['precision']):.4f}",
            f"recall:          {float(metrics['recall']):.4f}",
            f"f1:              {float(metrics['f1']):.4f}",
            f"roc_auc:         {roc_auc_text}",
            f"confusion_matrix:{metrics['confusion_matrix']}",
        ]
    )


def save_confusion_matrix_plot(
    metrics: Mapping[str, Any],
    path: str | Path,
    *,
    labels: tuple[str, str] = ("No CHD", "CHD"),
    title: str = "Confusion matrix",
) -> Path:
    """Save a labelled confusion-matrix heatmap as a PNG."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    if matrix.shape != (2, 2):
        raise ValueError(
            f"Expected a 2x2 confusion matrix, got shape {matrix.shape}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(4.5, 4.0))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )

    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=list(labels),
        yticklabels=list(labels),
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _positive_scores(
    y_prob: Sequence[float] | np.ndarray | pd.DataFrame | None,
    *,
    expected_length: int,
) -> np.ndarray | None:
    if y_prob is None:
        return None

    scores = np.asarray(y_prob)
    if scores.ndim == 1:
        scores = scores.reshape(-1)
    elif scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.shape[1] == 2:
            scores = scores[:, 1]
        else:
            raise ValueError(
                "y_prob must have one or two columns; "
                f"got shape {scores.shape}"
            )
    else:
        raise ValueError(
            "y_prob must be a 1-D or 2-D array; "
            f"got shape {scores.shape}"
        )

    if scores.size != expected_length:
        raise ValueError(
            f"y_prob length ({scores.size}) does not match y_true "
            f"({expected_length})"
        )
    if not np.isfinite(scores.astype(float)).all():
        raise ValueError("y_prob contains NaN or infinite values")
    return scores.astype(float)


def _validate_binary_vector(values: np.ndarray, name: str) -> None:
    unique = np.unique(values)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(
            f"{name} must contain binary labels 0 and 1; found {unique.tolist()}"
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
