"""Decision policies applied after probability scoring."""

from __future__ import annotations


def credit_decision(probability: float, threshold: float) -> str:
    """Map default probability to a conservative credit decision."""

    if probability >= threshold:
        return "reject"
    if probability >= threshold * 0.60:
        return "review"
    return "approve"


def fraud_decision(probability: float, threshold: float) -> str:
    """Map fraud probability to an authorization decision."""

    if probability >= threshold:
        return "block"
    if probability >= threshold * 0.60:
        return "review"
    return "allow"
