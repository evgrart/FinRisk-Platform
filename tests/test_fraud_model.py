import numpy as np
import pandas as pd

from finrisk.data.generate import generate_transactions
from finrisk.data.split import temporal_split
from finrisk.models.fraud import build_fraud_features, select_cost_threshold


def test_fraud_feature_engineering_drops_identifiers_and_adds_time_features():
    frame = generate_transactions(300, n_customers=50, seed=41)
    features = build_fraud_features(frame)
    assert "transaction_id" not in features.columns
    assert "customer_id" not in features.columns
    assert "is_fraud" not in features.columns
    assert {"transaction_hour_sin", "transaction_hour_cos", "log_amount"}.issubset(features.columns)


def test_cost_threshold_prefers_reducing_false_negatives_when_they_are_expensive():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.02, 0.15, 0.25, 0.55, 0.80, 0.95])
    threshold = select_cost_threshold(y_true, probabilities, false_negative_cost=100)
    assert threshold <= 0.55


def test_fraud_split_can_feed_feature_engineering():
    frame = generate_transactions(300, n_customers=50, seed=42)
    split = temporal_split(frame, time_column="transaction_ts")
    assert len(build_fraud_features(split.test)) > 0
