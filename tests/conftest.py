"""Shared test fixtures.

The real Framingham data is intentionally not committed.  Tests use a
synthetic dataset with the same column names and a realistic class balance so
that the data and evaluation code can be verified in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def synthetic_framingham() -> pd.DataFrame:
    """Return a deterministic, Framingham-shaped binary-classification frame."""
    rng = np.random.default_rng(2024)
    n_rows = 900

    age = rng.integers(30, 76, n_rows)
    male = rng.integers(0, 2, n_rows)
    education = rng.integers(1, 5, n_rows).astype(float)
    current_smoker = rng.integers(0, 2, n_rows)
    cigs_per_day = np.where(
        current_smoker == 1,
        rng.integers(1, 41, n_rows),
        0,
    ).astype(float)
    bp_meds = rng.integers(0, 2, n_rows)
    prevalent_stroke = rng.binomial(1, 0.02, n_rows)
    prevalent_hyp = rng.binomial(1, 0.30, n_rows)
    diabetes = rng.binomial(1, 0.08, n_rows)
    tot_chol = np.clip(rng.normal(240, 40, n_rows), 120, 500)
    sys_bp = np.clip(rng.normal(135, 20, n_rows), 90, 220)
    dia_bp = np.clip(rng.normal(85, 12, n_rows), 50, 140)
    bmi = np.clip(rng.normal(27, 4, n_rows), 15, 50)
    heart_rate = np.clip(rng.normal(75, 12, n_rows), 40, 130)
    glucose = np.clip(rng.normal(100, 25, n_rows), 50, 300)

    logit = (
        -2.5
        + 0.055 * (age - 50)
        + 0.85 * male
        + 0.70 * current_smoker
        + 0.012 * (tot_chol - 240)
        + 0.022 * (sys_bp - 135)
        + 0.018 * (glucose - 100)
        + 0.45 * diabetes
        + 0.35 * prevalent_hyp
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "male": male,
            "age": age,
            "education": education,
            "currentSmoker": current_smoker,
            "cigsPerDay": cigs_per_day,
            "BPMeds": bp_meds,
            "prevalentStroke": prevalent_stroke,
            "prevalentHyp": prevalent_hyp,
            "diabetes": diabetes,
            "totChol": tot_chol,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose,
            "TenYearCHD": target,
        }
    )

    # Missing values exercise the imputation path.
    for column, fraction in (
        ("education", 0.05),
        ("cigsPerDay", 0.08),
        ("BMI", 0.02),
        ("glucose", 0.03),
    ):
        mask = rng.random(n_rows) < fraction
        frame.loc[mask, column] = np.nan

    assert frame["TenYearCHD"].nunique() == 2
    assert frame["TenYearCHD"].sum() > 30
    return frame


@pytest.fixture
def synthetic_categorical(synthetic_framingham: pd.DataFrame) -> pd.DataFrame:
    """Framingham-shaped data with one extra categorical feature."""
    rng = np.random.default_rng(7)
    frame = synthetic_framingham.copy()
    region = rng.choice(["north", "south", "east"], size=len(frame)).astype(object)
    region[rng.random(len(frame)) < 0.05] = np.nan
    frame["region"] = region
    return frame


@pytest.fixture
def synthetic_csv(
    tmp_path: Path,
    synthetic_framingham: pd.DataFrame,
) -> Path:
    """Write the synthetic dataset to a temporary CSV."""
    path = tmp_path / "framingham.csv"
    synthetic_framingham.to_csv(path, index=False)
    return path
