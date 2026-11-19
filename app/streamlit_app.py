import streamlit as st

from src.config import MODEL_PATH
from src.explain import explain_single_record, get_global_feature_importance
from src.predict import CreditRiskPredictor
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
st.set_page_config(page_title="Paisabazaar Credit Risk Scorer", layout="wide")
st.title("Paisabazaar Credit Risk Prediction & SHAP Explainability")

if not MODEL_PATH.exists():
    st.error("Model artifact not found at `artifacts/credit_risk_model.joblib`. Run `python -m src.train` first.")
    st.stop()

predictor = CreditRiskPredictor(MODEL_PATH)

st.sidebar.header("Applicant Inputs")
age = st.sidebar.slider("Age", min_value=18, max_value=75, value=32)
occupation = st.sidebar.selectbox(
    "Occupation",
    ["Engineer", "Developer", "Scientist", "Accountant", "Manager", "Doctor", "Lawyer", "Teacher", "Other"],
)
annual_income = st.sidebar.number_input("Annual Income", min_value=0.0, value=65000.0, step=1000.0)
monthly_salary = st.sidebar.number_input("Monthly Inhand Salary", min_value=0.0, value=5200.0, step=100.0)
num_bank_accounts = st.sidebar.number_input("Bank Accounts", min_value=0, max_value=15, value=3)
num_credit_cards = st.sidebar.number_input("Credit Cards", min_value=0, max_value=15, value=2)
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.0)
num_loans = st.sidebar.number_input("Number of Loans", min_value=0, max_value=15, value=1)
delay_days = st.sidebar.number_input("Delay From Due Date (days)", min_value=0.0, max_value=90.0, value=3.0)
delayed_payments = st.sidebar.number_input("Delayed Payments Count", min_value=0.0, max_value=30.0, value=2.0)
credit_inquiries = st.sidebar.number_input("Credit Inquiries", min_value=0.0, max_value=25.0, value=2.0)
credit_mix = st.sidebar.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
outstanding_debt = st.sidebar.number_input("Outstanding Debt", min_value=0.0, value=15000.0, step=500.0)
utilization_ratio = st.sidebar.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=32.0)
credit_history_age = st.sidebar.number_input("Credit History Age (months)", min_value=0.0, max_value=500.0, value=180.0)
pay_min = st.sidebar.selectbox("Payment of Min Amount", ["No", "Yes", "NM"])
total_emi = st.sidebar.number_input("Total EMI Per Month", min_value=0.0, value=600.0, step=50.0)
amount_invested = st.sidebar.number_input("Amount Invested Monthly", min_value=0.0, value=500.0, step=50.0)
payment_behaviour = st.sidebar.selectbox(
    "Payment Behaviour",
    [
        "Low_spent_Small_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments",
        "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
    ],
)
monthly_balance = st.sidebar.number_input("Monthly Balance", min_value=0.0, value=800.0, step=50.0)

record = {
    "Month": 1,
    "Age": float(age),
    "Occupation": occupation,
    "Annual_Income": float(annual_income),
    "Monthly_Inhand_Salary": float(monthly_salary),
    "Num_Bank_Accounts": int(num_bank_accounts),
    "Num_Credit_Card": int(num_credit_cards),
    "Interest_Rate": float(interest_rate),
    "Num_of_Loan": int(num_loans),
    "Delay_from_due_date": float(delay_days),
    "Num_of_Delayed_Payment": float(delayed_payments),
    "Changed_Credit_Limit": 0.0,
    "Num_Credit_Inquiries": float(credit_inquiries),
    "Credit_Mix": credit_mix,
    "Outstanding_Debt": float(outstanding_debt),
    "Credit_Utilization_Ratio": float(utilization_ratio),
    "Credit_History_Age": float(credit_history_age),
    "Payment_of_Min_Amount": pay_min,
    "Total_EMI_per_month": float(total_emi),
    "Amount_invested_monthly": float(amount_invested),
    "Payment_Behaviour": payment_behaviour,
    "Monthly_Balance": float(monthly_balance),
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button("Predict Credit Score", type="primary"):
        result = predictor.predict_records([record])[0]
        st.success(f"Predicted Credit Score: **{result['predicted_credit_score']}** ({result['model_name']})")
        if "probabilities" in result:
            st.json(result["probabilities"])

        st.subheader("Top SHAP Feature Drivers for This Applicant")
        exp_df = explain_single_record(record, MODEL_PATH, top_n=8)
        st.dataframe(exp_df, use_container_width=True)

with col2:
    st.subheader("Global Top Features")
    global_df = get_global_feature_importance(MODEL_PATH, top_n=10)
    st.bar_chart(global_df.set_index("feature")["importance"])