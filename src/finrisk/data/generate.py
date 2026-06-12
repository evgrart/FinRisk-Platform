"""Generate realistic, reproducible synthetic fintech datasets.

The generator intentionally creates only point-in-time features. This makes the
datasets useful for demonstrating temporal validation and leakage prevention
without pretending to contain real customer information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def generate_credit_applications(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate point-in-time credit application snapshots."""

    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 to support train/validation/test splits")

    rng = np.random.default_rng(seed)
    application_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 1095, size=n_rows), unit="D"
    )
    age = rng.integers(21, 71, size=n_rows)
    income = np.clip(rng.lognormal(mean=np.log(95_000), sigma=0.55, size=n_rows), 25_000, 900_000)
    employment_years = np.minimum(
        rng.exponential(scale=6.0, size=n_rows), np.maximum(age - 18, 1)
    ).round(1)
    requested_amount = np.clip(
        income * rng.uniform(0.15, 1.8, size=n_rows) + rng.normal(0, 35_000, size=n_rows),
        20_000,
        1_500_000,
    ).round(-2)
    term_months = rng.choice([6, 12, 18, 24, 36, 48, 60], size=n_rows, p=[0.04, 0.12, 0.08, 0.23, 0.34, 0.12, 0.07])
    debt_to_income = np.clip(
        0.08 + 0.0000003 * requested_amount / np.maximum(income, 1) * 100_000
        + rng.beta(2.2, 7.0, size=n_rows) * 0.72,
        0.02,
        0.98,
    ).round(3)
    credit_history_months = np.clip((age - 18) * 12 + rng.normal(0, 24, size=n_rows), 6, 600).round().astype(int)
    active_credit_lines = np.clip(rng.poisson(2.5, size=n_rows), 0, 15)
    delinquencies_12m = np.clip(rng.poisson(0.22, size=n_rows), 0, 5)
    hard_inquiries_6m = np.clip(rng.poisson(1.4, size=n_rows), 0, 10)
    home_ownership = rng.choice(["rent", "own", "mortgage", "family"], size=n_rows, p=[0.38, 0.18, 0.34, 0.10])
    employment_type = rng.choice(["full_time", "part_time", "self_employed", "student", "retired"], size=n_rows, p=[0.57, 0.08, 0.16, 0.12, 0.07])
    region = rng.choice(["central", "northwest", "volga", "south", "ural", "siberia", "far_east"], size=n_rows, p=[0.28, 0.16, 0.16, 0.12, 0.10, 0.12, 0.06])
    channel = rng.choice(["mobile_app", "website", "branch", "partner"], size=n_rows, p=[0.52, 0.24, 0.14, 0.10])
    has_prior_default = rng.binomial(1, 0.075, size=n_rows)

    # Latent mechanism is deliberately nonlinear enough for model comparison,
    # while every target-driving value is available at application time.
    logit = (
        -2.45
        + 0.030 * (40 - age)
        + 0.010 * (debt_to_income * 100 - 35)
        + 0.34 * delinquencies_12m
        + 0.11 * hard_inquiries_6m
        + 0.92 * has_prior_default
        + 0.0000025 * (requested_amount - income * 0.75)
        - 0.0000030 * (income - 95_000)
        - 0.045 * employment_years
        + 0.008 * (term_months - 24)
        + 0.16 * (employment_type == "self_employed")
        + 0.13 * (home_ownership == "rent")
        + rng.normal(0, 0.42, size=n_rows)
    )
    default_probability = _sigmoid(logit)
    default_90d = rng.binomial(1, default_probability)

    frame = pd.DataFrame(
        {
            "application_id": [f"app_{i:07d}" for i in range(n_rows)],
            "customer_id": [f"cus_{i:06d}" for i in rng.integers(0, max(n_rows // 3, 1), size=n_rows)],
            "application_date": application_dates,
            "age": age,
            "annual_income": income.round(2),
            "employment_years": employment_years,
            "requested_amount": requested_amount,
            "term_months": term_months,
            "debt_to_income": debt_to_income,
            "credit_history_months": credit_history_months,
            "active_credit_lines": active_credit_lines,
            "delinquencies_12m": delinquencies_12m,
            "hard_inquiries_6m": hard_inquiries_6m,
            "home_ownership": home_ownership,
            "employment_type": employment_type,
            "region": region,
            "channel": channel,
            "has_prior_default": has_prior_default,
            "default_90d": default_90d,
        }
    ).sort_values("application_date", kind="stable").reset_index(drop=True)

    # Missingness is injected after target creation: missing values are an
    # observation-quality issue, not a hidden target proxy.
    for column, rate in {"annual_income": 0.018, "employment_years": 0.012, "home_ownership": 0.009}.items():
        missing = rng.random(n_rows) < rate
        frame.loc[missing, column] = np.nan

    return frame


def generate_transactions(n_rows: int, n_customers: int, seed: int) -> pd.DataFrame:
    """Generate point-in-time transaction snapshots with a rare fraud label."""

    if n_rows < 100 or n_customers < 10:
        raise ValueError("n_rows must be at least 100 and n_customers at least 10")

    rng = np.random.default_rng(seed)
    transaction_ts = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60, size=n_rows), unit="m"
    )
    amount = np.clip(rng.lognormal(mean=np.log(2_500), sigma=1.05, size=n_rows), 50, 450_000).round(2)
    merchant_category = rng.choice(
        ["groceries", "electronics", "travel", "gaming", "jewelry", "restaurants", "utilities", "cash_withdrawal"],
        size=n_rows,
        p=[0.22, 0.12, 0.10, 0.07, 0.035, 0.18, 0.20, 0.075],
    )
    entry_mode = rng.choice(["chip", "contactless", "online", "magstripe", "manual"], size=n_rows, p=[0.26, 0.42, 0.24, 0.05, 0.03])
    device_trust_score = np.clip(rng.beta(7, 2, size=n_rows), 0, 1).round(3)
    distance_km = np.clip(rng.lognormal(mean=np.log(3), sigma=1.4, size=n_rows), 0, 2_000).round(2)
    hour = transaction_ts.hour.to_numpy()
    is_international = rng.binomial(1, 0.065, size=n_rows)
    is_new_merchant = rng.binomial(1, 0.18, size=n_rows)
    transactions_24h = np.clip(rng.poisson(2.7, size=n_rows), 0, 60)
    amount_zscore = np.clip(rng.normal(0.0, 1.0, size=n_rows) + np.log1p(amount / 2_500) * 0.25, -4, 12).round(3)
    merchant_risk = pd.Series(merchant_category).map(
        {"gaming": 0.9, "jewelry": 1.1, "cash_withdrawal": 0.7, "electronics": 0.45}.get
    ).fillna(0.0).to_numpy()

    logit = (
        -5.25
        + 1.55 * (entry_mode == "online")
        + 1.10 * (entry_mode == "magstripe")
        + 1.65 * (device_trust_score < 0.35)
        + 0.0009 * distance_km
        + 1.15 * is_international
        + 1.05 * is_new_merchant
        + 0.075 * transactions_24h
        + 0.32 * np.maximum(amount_zscore, 0)
        + merchant_risk
        + 0.75 * ((hour <= 5) | (hour >= 23))
        + rng.normal(0, 0.55, size=n_rows)
    )
    fraud_probability = _sigmoid(logit)
    is_fraud = rng.binomial(1, fraud_probability)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:08d}" for i in range(n_rows)],
            "customer_id": [f"cus_{i:06d}" for i in rng.integers(0, n_customers, size=n_rows)],
            "transaction_ts": transaction_ts,
            "amount": amount,
            "merchant_category": merchant_category,
            "entry_mode": entry_mode,
            "device_trust_score": device_trust_score,
            "distance_km": distance_km,
            "hour": hour,
            "is_international": is_international,
            "is_new_merchant": is_new_merchant,
            "transactions_24h": transactions_24h,
            "amount_zscore": amount_zscore,
            "is_fraud": is_fraud,
        }
    ).sort_values("transaction_ts", kind="stable").reset_index(drop=True)

    missing = rng.random(n_rows) < 0.006
    frame.loc[missing, "distance_km"] = np.nan
    return frame


def generate_datasets(
    output_dir: str | Path = "data/raw",
    *,
    applications: int = 50_000,
    transactions: int = 150_000,
    customers: int = 12_000,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate both datasets and return their paths."""

    if customers <= 0:
        raise ValueError("customers must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    applications_path = destination / "credit_applications.csv"
    transactions_path = destination / "transactions.csv"

    generate_credit_applications(applications, seed).to_csv(applications_path, index=False)
    generate_transactions(transactions, customers, seed + 1).to_csv(transactions_path, index=False)
    return applications_path, transactions_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--applications", type=int, default=50_000)
    parser.add_argument("--transactions", type=int, default=150_000)
    parser.add_argument("--customers", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    applications_path, transactions_path = generate_datasets(
        args.output_dir,
        applications=args.applications,
        transactions=args.transactions,
        customers=args.customers,
        seed=args.seed,
    )
    print(f"Generated: {applications_path}")
    print(f"Generated: {transactions_path}")


if __name__ == "__main__":
    main()
