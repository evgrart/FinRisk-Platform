"""Human-readable reason codes for analyst-facing decisions.

These are deliberately transparent business rules, not a claim that a rule
is the model's exact causal explanation. They make the API useful to a human
reviewer and are easy to audit.
"""

from __future__ import annotations

import pandas as pd


def credit_reason_codes(frame: pd.DataFrame) -> list[str]:
    row = frame.iloc[0]
    reasons: list[str] = []
    annual_income = row.get("annual_income")
    requested_amount = row.get("requested_amount")
    if pd.notna(row.get("has_prior_default")) and row["has_prior_default"] == 1:
        reasons.append("есть предыдущий дефолт")
    if pd.notna(row.get("delinquencies_12m")) and row["delinquencies_12m"] >= 1:
        reasons.append("были просрочки за последние 12 месяцев")
    if pd.notna(row.get("debt_to_income")) and row["debt_to_income"] >= 0.45:
        reasons.append("высокая долговая нагрузка")
    if (
        pd.notna(annual_income)
        and pd.notna(requested_amount)
        and requested_amount / max(float(annual_income), 1.0) >= 1.0
    ):
        reasons.append("запрошенная сумма сопоставима с годовым доходом")
    if pd.notna(row.get("hard_inquiries_6m")) and row["hard_inquiries_6m"] >= 5:
        reasons.append("много кредитных запросов за последние 6 месяцев")
    return reasons[:4] or ["значимых риск-факторов по правилам не обнаружено"]


def fraud_reason_codes(frame: pd.DataFrame) -> list[str]:
    row = frame.iloc[0]
    reasons: list[str] = []
    if row.get("entry_mode") in {"online", "magstripe", "manual"}:
        reasons.append("неприсутствующий или менее доверенный способ оплаты")
    if pd.notna(row.get("device_trust_score")) and row["device_trust_score"] < 0.35:
        reasons.append("низкий уровень доверия к устройству")
    if pd.notna(row.get("distance_km")) and row["distance_km"] >= 300:
        reasons.append("нетипично большое расстояние до предыдущей активности")
    if row.get("is_international") == 1:
        reasons.append("международная транзакция")
    if row.get("is_new_merchant") == 1:
        reasons.append("новый торговец")
    if pd.notna(row.get("transactions_24h")) and row["transactions_24h"] >= 10:
        reasons.append("аномально высокая активность за последние 24 часа")
    if pd.notna(row.get("amount_zscore")) and row["amount_zscore"] >= 3:
        reasons.append("сумма сильно отличается от обычного поведения")
    return reasons[:4] or ["значимых риск-факторов по правилам не обнаружено"]
