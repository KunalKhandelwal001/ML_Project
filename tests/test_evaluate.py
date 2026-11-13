from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import (
    evaluate_model,
    evaluate_predictions,
    format_metrics,
    load_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)
from src.data import load_data


def test_known_metrics():
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.4, 0.6, 0.9],
    )

    assert metrics["n_samples"] == 4
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_perfect_predictions():
    metrics = evaluate_predictions(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.05, 0.95, 0.20, 0.80],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_roc_auc_is_none_with_one_class():
    metrics = evaluate_predictions(
        y_true=[0, 0, 0],
        y_pred=[0, 0, 0],
        y_prob=[0.1, 0.2, 0.3],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] is None
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_two_dimensional_probabilities_use_positive_column():
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_prob=[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_probability_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        evaluate_predictions(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_prob=[0.1, 0.2, 0.3],
        )


def test_invalid_probability_shape_raises():
    with pytest.raises(ValueError, match="one or two columns"):
        evaluate_predictions(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1],
            y_prob=np.zeros((4, 3)),
        )


def test_invalid_labels_raise():
    with pytest.raises(ValueError, match="binary"):
        evaluate_predictions(y_true=[0, 2], y_pred=[0, 1])


def test_save_and_load_metrics_roundtrip(tmp_path):
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.4, 0.6, 0.9],
    )
    path = save_metrics(metrics, tmp_path / "metrics.json")

    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    loaded = load_metrics(path)

    assert raw["accuracy"] == pytest.approx(0.5)
    assert loaded["confusion_matrix"] == [[1, 1], [1, 1]]
    assert loaded["roc_auc"] == pytest.approx(1.0)


def test_confusion_matrix_plot_is_created(tmp_path):
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.4, 0.6, 0.9],
    )
    path = save_confusion_matrix_plot(metrics, tmp_path / "cm.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_format_metrics_contains_expected_fields():
    metrics = evaluate_predictions(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.9, 0.2, 0.8],
    )
    text = format_metrics(metrics)

    assert "accuracy" in text
    assert "precision" in text
    assert "recall" in text
    assert "f1" in text
    assert "roc_auc" in text
    assert "confusion_matrix" in text


def test_evaluate_model_integration(synthetic_csv):
    from sklearn.linear_model import LogisticRegression

    bundle = load_data(synthetic_csv, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(bundle.X_train, bundle.y_train)

    metrics = evaluate_model(model, bundle.X_val, bundle.y_val)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["confusion_matrix"][0][0] >= 0
