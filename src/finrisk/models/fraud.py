"""Cost-sensitive transaction fraud training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finrisk.data.split import TemporalSplit, temporal_split
from finrisk.data.validate import validate_transactions


TARGET = "is_fraud"
ID_COLUMNS = {"transaction_id", "customer_id", "transaction_ts", TARGET}


@dataclass(frozen=True)
class FraudTrainingResult:
    model: CalibratedClassifierCV
    threshold: float
    selected_model_name: str
    metrics: dict[str, Any]
    feature_columns: tuple[str, ...]
    false_positive_cost: float
    false_negative_cost: float


def build_fraud_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build authorization-time features without future transaction data."""

    result = frame.copy()
    transaction_ts = pd.to_datetime(result.pop("transaction_ts"), errors="coerce")
    if transaction_ts.isna().any():
        raise ValueError("transaction_ts contains invalid timestamps")

    hour = transaction_ts.dt.hour + transaction_ts.dt.minute / 60
    result["transaction_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["transaction_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["log_amount"] = np.log1p(result["amount"])
    result["distance_amount_interaction"] = result["distance_km"] * result["amount_zscore"].clip(lower=0)
    result = result.drop(columns=[column for column in ID_COLUMNS if column in result.columns])
    return result


def _make_pipeline(model: Any, feature_frame: pd.DataFrame) -> Pipeline:
    numeric = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _candidate_models(feature_frame: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": _make_pipeline(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1_000, random_state=seed),
            feature_frame,
        ),
        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            feature_frame,
        ),
    }


def _cost_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    actual_fraud = y_true.to_numpy(dtype=bool)
    false_positives = float((predictions & ~actual_fraud).sum())
    false_negatives = float((~predictions & actual_fraud).sum())
    true_negatives = float((~predictions & ~actual_fraud).sum())
    false_positive_rate = false_positives / max(false_positives + true_negatives, 1.0)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "recall": float((predictions & actual_fraud).sum() / max(actual_fraud.sum(), 1)),
        "false_positive_rate": false_positive_rate,
        "review_rate": float(predictions.mean()),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "expected_cost": false_positives * false_positive_cost + false_negatives * false_negative_cost,
    }


def select_cost_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 15.0,
) -> float:
    """Minimize validation business cost; the test set remains untouched."""

    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("error costs must be positive")
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.001, 0.999, 200)))
    costs = []
    actual_fraud = y_true.to_numpy(dtype=bool)
    for threshold in candidates:
        predictions = probabilities >= threshold
        false_positives = (predictions & ~actual_fraud).sum()
        false_negatives = (~predictions & actual_fraud).sum()
        costs.append(false_positives * false_positive_cost + false_negatives * false_negative_cost)
    return float(candidates[int(np.argmin(costs))])


def train_fraud_model(
    split: TemporalSplit,
    *,
    seed: int = 42,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 15.0,
) -> FraudTrainingResult:
    """Compare, calibrate and cost-tune fraud classifiers."""

    train_x, train_y = build_fraud_features(split.train), split.train[TARGET].astype(int)
    validation_x, validation_y = build_fraud_features(split.validation), split.validation[TARGET].astype(int)
    test_x, test_y = build_fraud_features(split.test), split.test[TARGET].astype(int)
    models = _candidate_models(train_x, seed)
    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(validation_x)[:, 1]
        candidate_metrics[name] = _cost_metrics(
            validation_y, probabilities, threshold=0.5,
            false_positive_cost=false_positive_cost, false_negative_cost=false_negative_cost,
        )
        fitted_models[name] = model

    selected_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["pr_auc"])
    calibrated = CalibratedClassifierCV(
        estimator=fitted_models[selected_name], method="sigmoid", cv=3, n_jobs=-1
    )
    calibrated.fit(train_x, train_y)
    calibrated_validation = calibrated.predict_proba(validation_x)[:, 1]
    calibrated_test = calibrated.predict_proba(test_x)[:, 1]
    threshold = select_cost_threshold(
        validation_y, calibrated_validation,
        false_positive_cost=false_positive_cost, false_negative_cost=false_negative_cost,
    )

    metrics = {
        "candidates": candidate_metrics,
        "selected_model": selected_name,
        "costs": {
            "false_positive": false_positive_cost,
            "false_negative": false_negative_cost,
        },
        "calibrated": {
            "validation": _cost_metrics(
                validation_y, calibrated_validation, threshold,
                false_positive_cost=false_positive_cost, false_negative_cost=false_negative_cost,
            ),
            "test": _cost_metrics(
                test_y, calibrated_test, threshold,
                false_positive_cost=false_positive_cost, false_negative_cost=false_negative_cost,
            ),
        },
    }
    return FraudTrainingResult(
        model=calibrated,
        threshold=threshold,
        selected_model_name=selected_name,
        metrics=metrics,
        feature_columns=tuple(train_x.columns),
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )


def save_fraud_artifacts(result: FraudTrainingResult, output_dir: str | Path = "artifacts/fraud") -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "threshold": result.threshold,
            "selected_model_name": result.selected_model_name,
            "feature_columns": result.feature_columns,
            "false_positive_cost": result.false_positive_cost,
            "false_negative_cost": result.false_negative_cost,
        },
        destination / "model.joblib",
    )
    (destination / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train_from_csv(
    path: str | Path = "data/raw/transactions.csv",
    *,
    seed: int = 42,
    output_dir: str | Path = "artifacts/fraud",
) -> FraudTrainingResult:
    frame = pd.read_csv(path)
    report = validate_transactions(frame)
    report.raise_if_invalid()
    split = temporal_split(frame, time_column="transaction_ts")
    result = train_fraud_model(split, seed=seed)
    save_fraud_artifacts(result, output_dir)
    return result
