import pandas as pd
import pytest

from finrisk.data.generate import generate_credit_applications, generate_transactions
from finrisk.data.split import split_summary, temporal_split
from finrisk.data.validate import validate_credit_applications, validate_transactions


def test_generated_credit_data_passes_validation():
    report = validate_credit_applications(generate_credit_applications(1_000, seed=21))
    assert report.passed, report.errors


def test_generated_transactions_pass_validation():
    report = validate_transactions(generate_transactions(1_000, n_customers=100, seed=22))
    assert report.passed, report.errors


def test_validator_catches_duplicate_ids_and_invalid_values():
    frame = generate_credit_applications(200, seed=23)
    frame.loc[1, "application_id"] = frame.loc[0, "application_id"]
    frame.loc[2, "age"] = 5
    report = validate_credit_applications(frame)
    assert not report.passed
    assert any("unique" in error for error in report.errors)
    assert any("age" in error for error in report.errors)


def test_temporal_split_is_ordered_and_disjoint():
    frame = pd.DataFrame(
        {
            "event_id": range(100),
            "event_time": pd.date_range("2025-01-01", periods=100, freq="D"),
            "target": [index % 2 for index in range(100)],
        }
    ).sample(frac=1, random_state=1)
    split = temporal_split(frame, time_column="event_time")

    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)
    assert set(split.train.index).isdisjoint(split.validation.index)
    assert set(split.validation.index).isdisjoint(split.test.index)
    assert split.train["event_time"].max() < split.validation["event_time"].min()
    assert split.validation["event_time"].max() < split.test["event_time"].min()

    summary = split_summary(split, time_column="event_time", target_column="target")
    assert list(summary["split"]) == ["train", "validation", "test"]
    assert summary["rows"].sum() == 100


def test_temporal_split_rejects_invalid_time_column():
    with pytest.raises(KeyError):
        temporal_split(pd.DataFrame({"x": [1, 2, 3]}), time_column="event_time")
