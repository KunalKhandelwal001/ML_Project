from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_main_runs_end_to_end_with_synthetic_data(
    synthetic_framingham,
    tmp_path,
):
    data_path = tmp_path / "framingham.csv"
    synthetic_framingham.to_csv(data_path, index=False)
    results_dir = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--data",
            str(data_path),
            "--results-dir",
            str(results_dir),
            "--save-splits",
            "--smoke-train",
            "--evaluate-test",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Task 1 data foundation" in result.stdout
    assert "Task 1 foundation complete" in result.stdout

    assert (results_dir / "run_config.json").exists()
    assert (results_dir / "data_summary.json").exists()
    assert (results_dir / "smoke_val_metrics.json").exists()
    assert (results_dir / "smoke_val_confusion_matrix.png").exists()
    assert (results_dir / "smoke_test_metrics.json").exists()
    assert (results_dir / "smoke_test_confusion_matrix.png").exists()
    assert (results_dir / "splits" / "train.csv").exists()
    assert (results_dir / "splits" / "val.csv").exists()
    assert (results_dir / "splits" / "test.csv").exists()


def test_main_reports_missing_dataset(tmp_path):
    results_dir = tmp_path / "results"
    missing_path = tmp_path / "does_not_exist.csv"

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--data",
            str(missing_path),
            "--results-dir",
            str(results_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "Dataset not found" in result.stderr
