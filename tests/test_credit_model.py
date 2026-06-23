import numpy as np
import pandas as pd

from finrisk.data.generate import generate_credit_applications
from finrisk.data.split import temporal_split
from finrisk.models.credit import build_credit_features, select_f1_threshold


def test_credit_feature_engineering_drops_identifiers_and_adds_ratios():
    frame = generate_credit_applications(200, seed=31)
    features = build_credit_features(frame)
    assert "application_id" not in features.columns
    assert "customer_id" not in features.columns
    assert "default_90d" not in features.columns
    assert {"requested_to_income", "income_per_term", "application_weekday"}.issubset(features.columns)


def test_threshold_is_selected_from_validation_probabilities():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.02, 0.15, 0.25, 0.55, 0.80, 0.95])
    threshold = select_f1_threshold(y_true, probabilities)
    assert 0.02 <= threshold <= 0.95


def test_credit_split_can_feed_feature_engineering():
    frame = generate_credit_applications(300, seed=32)
    split = temporal_split(frame, time_column="application_date")
    assert len(build_credit_features(split.train)) > 0
