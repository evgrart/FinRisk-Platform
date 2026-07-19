"""Lightweight population stability monitoring without external services."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_distribution(values: pd.Series, categories: pd.Index) -> np.ndarray:
    distribution = values.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    return np.maximum(distribution, 1e-6)


def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) <= 2:
        edges = np.array([reference.min(), reference.max() + 1e-9])
    cut_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    reference_buckets = pd.cut(reference, bins=cut_edges, include_lowest=True)
    current_buckets = pd.cut(current, bins=cut_edges, include_lowest=True)
    categories = reference_buckets.cat.categories.union(current_buckets.cat.categories)
    expected = _safe_distribution(reference_buckets, categories)
    actual = _safe_distribution(current_buckets, categories)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.astype("string").fillna("__missing__")
    current = current.astype("string").fillna("__missing__")
    categories = reference.unique().tolist()
    categories.extend(category for category in current.unique() if category not in categories)
    category_index = pd.Index(categories)
    expected = _safe_distribution(reference, category_index)
    actual = _safe_distribution(current, category_index)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """Calculate PSI; values above 0.2 are treated as material drift."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    both_numeric = pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current)
    if both_numeric:
        return _numeric_psi(reference, current, bins)
    return _categorical_psi(reference, current)


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Build a per-feature PSI report for a production monitoring job."""

    columns = columns or sorted(set(reference.columns) & set(current.columns))
    records = []
    for column in columns:
        psi = population_stability_index(reference[column], current[column], bins=bins)
        records.append(
            {
                "feature": column,
                "psi": psi,
                "drift_level": "missing" if pd.isna(psi) else "high" if psi >= 0.2 else "medium" if psi >= 0.1 else "low",
                "reference_missing_rate": float(reference[column].isna().mean()),
                "current_missing_rate": float(current[column].isna().mean()),
            }
        )
    return pd.DataFrame(records).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CSV snapshots with PSI")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = drift_report(pd.read_csv(args.reference), pd.read_csv(args.current))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
