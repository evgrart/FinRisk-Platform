"""Schema and quality checks for FinRisk datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable result of a dataset validation run."""

    dataset: str
    rows: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.passed:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"{self.dataset} failed validation:\n{details}")


def _validate_common(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str],
    id_column: str,
    target_column: str,
    time_column: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        errors.append(f"missing required columns: {', '.join(missing_columns)}")
        return errors, warnings

    if frame.empty:
        errors.append("dataset is empty")
    if frame[id_column].isna().any():
        errors.append(f"{id_column} contains missing values")
    if frame[id_column].duplicated().any():
        errors.append(f"{id_column} must be unique")

    parsed_time = pd.to_datetime(frame[time_column], errors="coerce")
    invalid_time = parsed_time.isna() & frame[time_column].notna()
    if invalid_time.any():
        errors.append(f"{time_column} contains unparseable timestamps")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    invalid_target = target.isna() | ~target.isin([0, 1])
    if invalid_target.any():
        errors.append(f"{target_column} must contain only binary 0/1 values")
    elif target.nunique() < 2 and len(frame) > 1:
        warnings.append(f"{target_column} contains only one class")

    for column in frame.columns:
        missing_rate = float(frame[column].isna().mean())
        if missing_rate > 0.20:
            warnings.append(f"{column} has {missing_rate:.1%} missing values")

    return errors, warnings


def _check_numeric_bounds(
    frame: pd.DataFrame,
    errors: list[str],
    *,
    column: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid_format = frame[column].notna() & values.isna()
    if invalid_format.any():
        errors.append(f"{column} contains non-numeric values")
        return

    invalid = values.notna()
    if lower is not None:
        invalid &= values < lower
    if upper is not None:
        invalid &= values > upper
    if invalid.any():
        limits = []
        if lower is not None:
            limits.append(f">= {lower:g}")
        if upper is not None:
            limits.append(f"<= {upper:g}")
        errors.append(f"{column} must be {' and '.join(limits)}")


def validate_credit_applications(frame: pd.DataFrame) -> ValidationReport:
    """Validate credit application snapshots."""

    required = {
        "application_id", "customer_id", "application_date", "age", "annual_income",
        "employment_years", "requested_amount", "term_months", "debt_to_income",
        "credit_history_months", "active_credit_lines", "delinquencies_12m",
        "hard_inquiries_6m", "home_ownership", "employment_type", "region", "channel",
        "has_prior_default", "default_90d",
    }
    errors, warnings = _validate_common(
        frame, required_columns=required, id_column="application_id",
        target_column="default_90d", time_column="application_date",
    )
    if errors and not required.issubset(frame.columns):
        return ValidationReport("credit_applications", len(frame), tuple(errors), tuple(warnings))

    bounded_columns = {
        "age": (18, 100), "annual_income": (0, None), "employment_years": (0, 80),
        "requested_amount": (0, None), "debt_to_income": (0, 2),
        "credit_history_months": (0, 1_000), "active_credit_lines": (0, 100),
        "delinquencies_12m": (0, 100), "hard_inquiries_6m": (0, 100),
        "has_prior_default": (0, 1),
    }
    for column, (lower, upper) in bounded_columns.items():
        _check_numeric_bounds(frame, errors, column=column, lower=lower, upper=upper)

    allowed_terms = {6, 12, 18, 24, 36, 48, 60}
    observed_terms = set(pd.to_numeric(frame["term_months"], errors="coerce").dropna().unique())
    if not observed_terms.issubset(allowed_terms):
        errors.append("term_months contains unsupported term values")
    return ValidationReport("credit_applications", len(frame), tuple(errors), tuple(warnings))


def validate_transactions(frame: pd.DataFrame) -> ValidationReport:
    """Validate transaction snapshots."""

    required = {
        "transaction_id", "customer_id", "transaction_ts", "amount", "merchant_category",
        "entry_mode", "device_trust_score", "distance_km", "hour", "is_international",
        "is_new_merchant", "transactions_24h", "amount_zscore", "is_fraud",
    }
    errors, warnings = _validate_common(
        frame, required_columns=required, id_column="transaction_id",
        target_column="is_fraud", time_column="transaction_ts",
    )
    if errors and not required.issubset(frame.columns):
        return ValidationReport("transactions", len(frame), tuple(errors), tuple(warnings))

    bounded_columns = {
        "amount": (0, None), "device_trust_score": (0, 1), "distance_km": (0, None),
        "hour": (0, 23), "is_international": (0, 1), "is_new_merchant": (0, 1),
        "transactions_24h": (0, 1_000), "amount_zscore": (-10, 100),
    }
    for column, (lower, upper) in bounded_columns.items():
        _check_numeric_bounds(frame, errors, column=column, lower=lower, upper=upper)
    return ValidationReport("transactions", len(frame), tuple(errors), tuple(warnings))


def _print_report(report: ValidationReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"[{status}] {report.dataset}: {report.rows:,} rows")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for error in report.errors:
        print(f"  error: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FinRisk CSV datasets")
    parser.add_argument("--credit", type=Path, help="path to credit_applications.csv")
    parser.add_argument("--transactions", type=Path, help="path to transactions.csv")
    args = parser.parse_args()
    if args.credit is None and args.transactions is None:
        parser.error("provide --credit and/or --transactions")

    reports = []
    if args.credit is not None:
        reports.append(validate_credit_applications(pd.read_csv(args.credit)))
    if args.transactions is not None:
        reports.append(validate_transactions(pd.read_csv(args.transactions)))
    for report in reports:
        _print_report(report)
    if any(not report.passed for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
