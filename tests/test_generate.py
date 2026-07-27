from finrisk.data.generate import (
    generate_credit_applications,
    generate_datasets,
    generate_transactions,
)


def test_credit_generator_is_reproducible():
    first = generate_credit_applications(500, seed=7)
    second = generate_credit_applications(500, seed=7)
    assert first.equals(second)


def test_credit_generator_has_expected_schema_and_rare_target():
    frame = generate_credit_applications(2_000, seed=7)
    assert {"application_id", "application_date", "default_90d"}.issubset(frame.columns)
    assert frame["application_id"].is_unique
    assert 0.03 < frame["default_90d"].mean() < 0.45


def test_transaction_generator_has_expected_schema_and_rare_target():
    frame = generate_transactions(2_000, n_customers=100, seed=8)
    assert {"transaction_id", "transaction_ts", "is_fraud"}.issubset(frame.columns)
    assert frame["transaction_id"].is_unique
    assert 0.001 < frame["is_fraud"].mean() < 0.20


def test_generate_datasets_writes_two_csvs(tmp_path):
    applications_path, transactions_path = generate_datasets(
        tmp_path, applications=150, transactions=200, customers=20, seed=11
    )
    assert applications_path.exists()
    assert transactions_path.exists()
