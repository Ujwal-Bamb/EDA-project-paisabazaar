from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "credit_risk_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TARGET_COL = "Credit_Score"
GROUP_COL = "Customer_ID"

# Drop identifiers, PII, and high-cardinality raw text for baseline tabular modeling
DROP_COLS = ["ID", "Name", "SSN", "Type_of_Loan"]

CATEGORICAL_COLS = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

NUMERIC_COLS = [
    "Month",
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    # Engineered features
    "emi_to_monthly_income",
    "debt_to_annual_income",
    "investment_to_monthly_income",
    "net_monthly_cashflow_est",
    "debt_per_loan",
]