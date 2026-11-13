"""FedDef command-line entry point.

Task 1 (foundation) responsibilities:
- load and validate the Framingham dataset
- create the shared, reproducible train/validation/test split
- save ``results/data_summary.json`` and ``results/run_config.json``
- optionally run a centralized logistic-regression smoke check

FedAvg, attacks, and the defense are implemented in their own modules by
Tasks 2-4.  The ``--smoke-train`` flag is only a data/evaluation sanity check,
not the project's FedAvg baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import DataBundle, load_data, save_data_summary, save_splits
from src.evaluate import (
    evaluate_model,
    format_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FedDef data foundation and sanity check.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to framingham.csv. Defaults to data/framingham.csv, "
            "then framingham.csv in the repository root."
        ),
    )
    parser.add_argument(
        "--target",
        default="TenYearCHD",
        help="Name of the binary target column (default: TenYearCHD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split and smoke model (default: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the test set (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of all rows reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated summaries and metrics (default: results).",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Also save the prepared train/val/test CSV files.",
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help=(
            "Fit a temporary centralized logistic regression and evaluate it "
            "on the validation set. This is a pipeline sanity check, not the "
            "Task 2 FedAvg baseline."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "With --smoke-train, also evaluate on the held-out test set. "
            "Use this only for a final check, not during tuning."
        ),
    )
    return parser


def resolve_data_path(cli_path: Path | None) -> Path:
    """Resolve the dataset path without silently guessing wrong files."""
    if cli_path is not None:
        return cli_path

    for candidate in (Path("data/framingham.csv"), Path("framingham.csv")):
        if candidate.exists():
            return candidate
    return Path("data/framingham.csv")


def print_bundle_summary(bundle: DataBundle) -> None:
    metadata = bundle.metadata
    print("=" * 68)
    print("FedDef - Task 1 data foundation")
    print("=" * 68)
    print(f"Source:          {metadata.get('source') or 'unknown'}")
    print(
        "Rows:            "
        f"{metadata['n_rows_after_target_drop']:,} used / "
        f"{metadata['n_rows_original']:,} original"
    )
    print(f"Target:          {bundle.target_name}")
    print(f"Features:        {bundle.n_features}")
    print(
        "Dropped:         "
        f"{metadata['dropped_columns'] if metadata['dropped_columns'] else 'none'}"
    )
    print(
        "Missing values:  "
        f"{sum(metadata['missing_values_before_imputation'].values()):,} imputed "
        "from training statistics"
    )
    print()
    for split_name in ("train", "val", "test"):
        size = metadata["split_sizes"][split_name]
        positive_rate = metadata["positive_rate"][split_name]
        print(f"{split_name.capitalize():>5}: {size:>5,} rows | {positive_rate:6.2%} positive")
    print()
    print("Feature names:")
    for name in bundle.feature_names:
        print(f"  - {name}")


def run_smoke_check(
    bundle: DataBundle,
    results_dir: Path,
    seed: int,
    evaluate_test: bool,
) -> None:
    """Fit a centralized baseline as a data/evaluation sanity check."""
    print()
    print("-" * 68)
    print("Smoke check: centralized LogisticRegression (validation set)")
    print("-" * 68)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(bundle.X_train, bundle.y_train)

    val_metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
    save_metrics(val_metrics, results_dir / "smoke_val_metrics.json")
    save_confusion_matrix_plot(
        val_metrics,
        results_dir / "smoke_val_confusion_matrix.png",
        title="Validation confusion matrix (smoke check)",
    )
    print(format_metrics(val_metrics))
    print(f"Saved: {results_dir / 'smoke_val_metrics.json'}")

    if evaluate_test:
        test_metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
        save_metrics(test_metrics, results_dir / "smoke_test_metrics.json")
        save_confusion_matrix_plot(
            test_metrics,
            results_dir / "smoke_test_confusion_matrix.png",
            title="Test confusion matrix (final check only)",
        )
        print()
        print("Held-out test metrics (final check):")
        print(format_metrics(test_metrics))
        print(f"Saved: {results_dir / 'smoke_test_metrics.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.evaluate_test and not args.smoke_train:
        parser.error("--evaluate-test requires --smoke-train")

    config = Config(
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        target_col=args.target,
        data_path=resolve_data_path(args.data),
        results_dir=args.results_dir,
    )
    config.ensure_results_dir()
    config.save(config.results_dir / "run_config.json")

    if not config.data_path.exists():
        print(f"Dataset not found: {config.data_path}", file=sys.stderr)
        print(
            "Place framingham.csv in data/framingham.csv "
            "(see data/README.md) or pass --data <path>.",
            file=sys.stderr,
        )
        return 1

    try:
        bundle = load_data(
            config.data_path,
            target_col=config.target_col,
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.seed,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    summary_path = save_data_summary(
        bundle,
        config.results_dir / "data_summary.json",
    )
    print_bundle_summary(bundle)
    print(f"\nSaved data summary: {summary_path}")

    if args.save_splits:
        splits_dir = save_splits(bundle, config.results_dir / "splits")
        print(f"Saved prepared splits: {splits_dir}")

    if args.smoke_train:
        run_smoke_check(
            bundle,
            results_dir=config.results_dir,
            seed=config.seed,
            evaluate_test=args.evaluate_test,
        )

    print()
    print("Task 1 foundation complete. FedAvg baseline is Task 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
