from fastapi.testclient import TestClient

from api.main import app
from src.config import MODEL_PATH
from src.features import engineer_features

client = TestClient(app)


def sample_payload() -> dict:
    return {
        "applications": [
            {
                "Month": 1,
                "Age": 32,
                "Occupation": "Engineer",
                "Annual_Income": 85000.0,
                "Monthly_Inhand_Salary": 6500.0,
                "Num_Bank_Accounts": 3,
                "Num_Credit_Card": 2,
                "Interest_Rate": 10.5,
                "Num_of_Loan": 1,
                "Delay_from_due_date": 2.0,
                "Num_of_Delayed_Payment": 1.0,
                "Changed_Credit_Limit": 5.0,
                "Num_Credit_Inquiries": 2.0,
                "Credit_Mix": "Good",
                "Outstanding_Debt": 12000.0,
                "Credit_Utilization_Ratio": 28.5,
                "Credit_History_Age": 180.0,
                "Payment_of_Min_Amount": "No",
                "Total_EMI_per_month": 500.0,
                "Amount_invested_monthly": 800.0,
                "Payment_Behaviour": "High_spent_Medium_value_payments",
                "Monthly_Balance": 1200.0,
            }
        ]
    }


def test_engineer_features_drops_pii_and_adds_ratios():
    df = sample_payload()["applications"]
    import pandas as pd
    out = engineer_features(pd.DataFrame(df))
    assert "SSN" not in out.columns
    assert "Name" not in out.columns
    assert "emi_to_monthly_income" in out.columns
    assert "debt_to_annual_income" in out.columns


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_endpoint():
    if not MODEL_PATH.exists():
        return
    res = client.post("/predict", json=sample_payload())
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert "predicted_credit_score" in body["predictions"][0]