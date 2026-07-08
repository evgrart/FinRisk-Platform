"""FastAPI application for credit and transaction scoring."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from finrisk.models.credit import build_credit_features
from finrisk.models.fraud import build_fraud_features
from finrisk.service.policy import credit_decision, fraud_decision


class CreditApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_date: datetime
    age: int = Field(ge=18, le=100)
    annual_income: float | None = Field(default=None, gt=0)
    employment_years: float | None = Field(default=None, ge=0, le=80)
    requested_amount: float = Field(gt=0)
    term_months: Literal[6, 12, 18, 24, 36, 48, 60]
    debt_to_income: float = Field(ge=0, le=2)
    credit_history_months: int = Field(ge=0, le=1_000)
    active_credit_lines: int = Field(ge=0, le=100)
    delinquencies_12m: int = Field(ge=0, le=100)
    hard_inquiries_6m: int = Field(ge=0, le=100)
    home_ownership: str | None = None
    employment_type: str
    region: str
    channel: str
    has_prior_default: int = Field(ge=0, le=1)


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_ts: datetime
    amount: float = Field(gt=0)
    merchant_category: str
    entry_mode: str
    device_trust_score: float = Field(ge=0, le=1)
    distance_km: float | None = Field(default=None, ge=0)
    is_international: int = Field(ge=0, le=1)
    is_new_merchant: int = Field(ge=0, le=1)
    transactions_24h: int = Field(ge=0, le=1_000)
    amount_zscore: float = Field(ge=-10, le=100)


class ScoreResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    decision: str
    threshold: float = Field(ge=0, le=1)
    model_version: str


class ModelStore:
    """Lazy model loader so the service can expose health before artifacts exist."""

    def __init__(self, credit_path: Path, fraud_path: Path) -> None:
        self.credit_path = credit_path
        self.fraud_path = fraud_path
        self._credit: dict | None = None
        self._fraud: dict | None = None

    @property
    def credit(self) -> dict:
        if self._credit is None:
            self._credit = self._load(self.credit_path)
        return self._credit

    @property
    def fraud(self) -> dict:
        if self._fraud is None:
            self._fraud = self._load(self.fraud_path)
        return self._fraud

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"model artifact is missing: {path}")
        return joblib.load(path)

    def status(self) -> dict[str, bool]:
        return {"credit_model_ready": self.credit_path.exists(), "fraud_model_ready": self.fraud_path.exists()}


def _credit_frame(request: CreditApplicationRequest) -> pd.DataFrame:
    return pd.DataFrame([request.model_dump()])


def _transaction_frame(request: TransactionRequest) -> pd.DataFrame:
    values = request.model_dump()
    values["hour"] = values["transaction_ts"].hour
    return pd.DataFrame([values])


def create_app(
    *,
    credit_model_path: Path | str = "artifacts/credit/model.joblib",
    fraud_model_path: Path | str = "artifacts/fraud/model.joblib",
) -> FastAPI:
    store = ModelStore(Path(credit_model_path), Path(fraud_model_path))
    app = FastAPI(
        title="FinRisk Scoring API",
        version="0.1.0",
        description="Credit-risk and transaction-fraud scoring service",
    )

    @app.get("/health")
    def health() -> dict:
        status = store.status()
        ready = all(status.values())
        return {"status": "ok" if ready else "degraded", **status}

    @app.get("/model-info")
    def model_info() -> dict:
        return store.status()

    @app.post("/score/credit", response_model=ScoreResponse)
    def score_credit(request: CreditApplicationRequest) -> ScoreResponse:
        try:
            bundle = store.credit
            features = build_credit_features(_credit_frame(request))
            probability = float(bundle["model"].predict_proba(features)[:, 1][0])
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        threshold = float(bundle["threshold"])
        return ScoreResponse(
            risk_score=probability,
            decision=credit_decision(probability, threshold),
            threshold=threshold,
            model_version=str(bundle.get("selected_model_name", "unknown")),
        )

    @app.post("/score/transaction", response_model=ScoreResponse)
    def score_transaction(request: TransactionRequest) -> ScoreResponse:
        try:
            bundle = store.fraud
            features = build_fraud_features(_transaction_frame(request))
            probability = float(bundle["model"].predict_proba(features)[:, 1][0])
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        threshold = float(bundle["threshold"])
        return ScoreResponse(
            risk_score=probability,
            decision=fraud_decision(probability, threshold),
            threshold=threshold,
            model_version=str(bundle.get("selected_model_name", "unknown")),
        )

    return app


app = create_app(
    credit_model_path=os.getenv("FINRISK_CREDIT_MODEL", "artifacts/credit/model.joblib"),
    fraud_model_path=os.getenv("FINRISK_FRAUD_MODEL", "artifacts/fraud/model.joblib"),
)
