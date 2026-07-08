from finrisk.service.policy import credit_decision, fraud_decision


def test_credit_policy_has_three_bands():
    assert credit_decision(0.02, 0.20) == "approve"
    assert credit_decision(0.15, 0.20) == "review"
    assert credit_decision(0.25, 0.20) == "reject"


def test_fraud_policy_is_conservative():
    assert fraud_decision(0.02, 0.20) == "allow"
    assert fraud_decision(0.15, 0.20) == "review"
    assert fraud_decision(0.25, 0.20) == "block"
