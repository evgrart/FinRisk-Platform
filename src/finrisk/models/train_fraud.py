"""CLI entry point for transaction fraud training."""

from __future__ import annotations

import argparse
from pathlib import Path

from finrisk.models.fraud import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FinRisk fraud model")
    parser.add_argument("--data", type=Path, default=Path("data/raw/transactions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fraud"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_from_csv(args.data, seed=args.seed, output_dir=args.output_dir)
    test_metrics = result.metrics["calibrated"]["test"]
    print(f"selected_model={result.selected_model_name}")
    print(f"threshold={result.threshold:.4f}")
    print(f"test_pr_auc={test_metrics['pr_auc']:.4f}")
    print(f"test_recall={test_metrics['recall']:.4f}")
    print(f"test_expected_cost={test_metrics['expected_cost']:.1f}")


if __name__ == "__main__":
    main()
