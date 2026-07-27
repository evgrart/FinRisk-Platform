"""Credit-risk training pipeline.

The module keeps feature construction, model comparison and evaluation in
regular Python functions so the same code can later be reused by the API.
"""

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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finrisk.data.split import TemporalSplit, temporal_split
from finrisk.data.validate import validate_credit_applications

TARGET = "default_90d"
ID_COLUMNS = {"application_id", "customer_id", TARGET, "application_date"}


@dataclass(frozen=True)
class CreditTrainingResult:
    """Fitted model, selected threshold and JSON-compatible metrics."""

    model: Pipeline | CalibratedClassifierCV
    threshold: float
    selected_model_name: str
    metrics: dict[str, Any]
    feature_columns: tuple[str, ...]


def build_credit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build only application-time features for credit scoring."""

    result = frame.copy()
    application_date = pd.to_datetime(result.pop("application_date"), errors="coerce")
    if application_date.isna().any():
        raise ValueError("application_date contains invalid timestamps")

    result["application_month"] = application_date.dt.month
    result["application_weekday"] = application_date.dt.weekday
    result["requested_to_income"] = result["requested_amount"] / result["annual_income"].clip(lower=1)
    result["income_per_term"] = result["annual_income"] / result["term_months"].clip(lower=1)
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
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=1_000, random_state=seed),
            feature_frame,
        ),
        "random_forest": _make_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            feature_frame,
        ),
    }


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_rate": float(predictions.mean()),
    }


def select_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a validation threshold without looking at the test set."""

    candidates = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 99)))
    scores = [f1_score(y_true, probabilities >= candidate, zero_division=0) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def _prepare_split(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_credit_features(partition), partition[TARGET].astype(int)


def train_credit_model(
    split: TemporalSplit,
    *,
    seed: int = 42,
) -> CreditTrainingResult:
    """Compare candidates, calibrate the winner and evaluate on holdout data."""

    train_x, train_y = _prepare_split(split.train)
    validation_x, validation_y = _prepare_split(split.validation)
    test_x, test_y = _prepare_split(split.test)
    models = _candidate_models(train_x, seed)
    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(validation_x)[:, 1]
        candidate_metrics[name] = _classification_metrics(validation_y, probabilities, threshold=0.5)
        fitted_models[name] = model

    selected_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["pr_auc"])
    calibrated = CalibratedClassifierCV(
        estimator=fitted_models[selected_name], method="sigmoid", cv=3, n_jobs=-1
    )
    calibrated.fit(train_x, train_y)
    calibrated_validation = calibrated.predict_proba(validation_x)[:, 1]
    calibrated_test = calibrated.predict_proba(test_x)[:, 1]
    threshold = select_f1_threshold(validation_y, calibrated_validation)

    metrics = {
        "candidates": candidate_metrics,
        "selected_model": selected_name,
        "calibrated": {
            "validation": _classification_metrics(validation_y, calibrated_validation, threshold),
            "test": _classification_metrics(test_y, calibrated_test, threshold),
        },
    }
    return CreditTrainingResult(
        model=calibrated,
        threshold=threshold,
        selected_model_name=selected_name,
        metrics=metrics,
        feature_columns=tuple(train_x.columns),
    )


def save_credit_artifacts(result: CreditTrainingResult, output_dir: str | Path = "artifacts/credit") -> None:
    """Persist the model bundle and metrics for later API loading."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.model,
        "threshold": result.threshold,
        "selected_model_name": result.selected_model_name,
        "feature_columns": result.feature_columns,
    }
    joblib.dump(bundle, destination / "model.joblib")
    (destination / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train_from_csv(
    path: str | Path = "data/raw/credit_applications.csv",
    *,
    seed: int = 42,
    output_dir: str | Path = "artifacts/credit",
) -> CreditTrainingResult:
    """Load, validate, temporally split and train from a CSV file."""

    frame = pd.read_csv(path)
    report = validate_credit_applications(frame)
    report.raise_if_invalid()
    split = temporal_split(frame, time_column="application_date")
    result = train_credit_model(split, seed=seed)
    save_credit_artifacts(result, output_dir)
    return result
