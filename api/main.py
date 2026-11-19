import json
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import CreditRiskPredictor

app = FastAPI(
    title="Paisabazaar Credit Risk Prediction API",
    version="1.0.0",
)

_predictor: CreditRiskPredictor | None = None


def get_predictor() -> CreditRiskPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Trained model artifact not found. Run `python -m src.train` first.",
            )
        _predictor = CreditRiskPredictor()
    return _predictor


class CreditApplicationInput(BaseModel):
    Month: int = 1
    Age: float = Field(..., ge=18, le=100)
    Occupation: str = "Engineer"
    Annual_Income: float = Field(..., ge=0)
    Monthly_Inhand_Salary: float = Field(..., ge=0)
    Num_Bank_Accounts: int = Field(1, ge=0)
    Num_Credit_Card: int = Field(1, ge=0)
    Interest_Rate: float = Field(12.0, ge=0)
    Num_of_Loan: int = Field(0, ge=0)
    Delay_from_due_date: float = Field(0.0, ge=0)
    Num_of_Delayed_Payment: float = Field(0.0, ge=0)
    Changed_Credit_Limit: float = 0.0
    Num_Credit_Inquiries: float = Field(0.0, ge=0)
    Credit_Mix: str = "Standard"
    Outstanding_Debt: float = Field(0.0, ge=0)
    Credit_Utilization_Ratio: float = Field(30.0, ge=0)
    Credit_History_Age: float = Field(120.0, ge=0)
    Payment_of_Min_Amount: str = "Yes"
    Total_EMI_per_month: float = Field(0.0, ge=0)
    Amount_invested_monthly: float = Field(0.0, ge=0)
    Payment_Behaviour: str = "Low_spent_Small_value_payments"
    Monthly_Balance: float = 0.0


class PredictBatchRequest(BaseModel):
    applications: list[CreditApplicationInput] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(request: PredictBatchRequest) -> dict[str, Any]:
    predictor = get_predictor()
    records = [app_item.model_dump() for app_item in request.applications]
    predictions = predictor.predict_records(records)
    return {"count": len(predictions), "predictions": predictions}