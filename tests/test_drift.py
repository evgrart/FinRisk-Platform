import pandas as pd
import pytest

from finrisk.monitoring.drift import drift_report, population_stability_index


def test_psi_is_low_for_identical_distributions():
    reference = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    assert population_stability_index(reference, reference) == pytest.approx(0.0)


def test_psi_detects_numeric_shift():
    reference = pd.Series(range(1_000), dtype=float)
    current = pd.Series(range(2_000, 3_000), dtype=float)
    assert population_stability_index(reference, current) > 0.2


def test_report_supports_categorical_features_and_missingness():
    reference = pd.DataFrame({"channel": ["app", "app", "branch", "app"], "x": [1, 2, 3, 4]})
    current = pd.DataFrame({"channel": ["branch", "branch", "branch", None], "x": [1, 2, 3, 4]})
    report = drift_report(reference, current)
    assert set(report["feature"]) == {"channel", "x"}
    assert report.loc[report["feature"] == "channel", "psi"].iloc[0] > 0
