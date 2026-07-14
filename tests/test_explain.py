import pandas as pd

from finrisk.models.explain import credit_reason_codes, fraud_reason_codes


def test_credit_reason_codes_are_human_readable():
    frame = pd.DataFrame(
        [{"has_prior_default": 1, "delinquencies_12m": 2, "debt_to_income": 0.7,
          "annual_income": 100_000, "requested_amount": 150_000, "hard_inquiries_6m": 1}]
    )
    reasons = credit_reason_codes(frame)
    assert len(reasons) <= 4
    assert any("дефолт" in reason for reason in reasons)


def test_fraud_reason_codes_capture_suspicious_signals():
    frame = pd.DataFrame(
        [{"entry_mode": "online", "device_trust_score": 0.1, "distance_km": 500,
          "is_international": 1, "is_new_merchant": 1, "transactions_24h": 20,
          "amount_zscore": 4}]
    )
    reasons = fraud_reason_codes(frame)
    assert len(reasons) == 4
    assert any("устрой" in reason for reason in reasons)
