import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.config import METRICS_PATH, MODEL_PATH
from src.explain import explain_single_record, get_global_feature_importance
from src.predict import CreditRiskPredictor

st.set_page_config(page_title="Paisabazaar Credit Risk Scorer", layout="wide")
st.title("Paisabazaar Credit Risk Prediction & SHAP Explainability")

if not MODEL_PATH.exists():
    st.error(
        "Model artifact not found at `artifacts/credit_risk_model.joblib`. Run `python -m src.train` first."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_record" not in st.session_state:
    st.session_state["last_record"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

metrics_data = (
    json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if METRICS_PATH.exists()
    else {}
)
best_model_name = metrics_data.get("best_model", "Random Forest")
models_dict = metrics_data.get("models", {})
best_scores = models_dict.get(best_model_name, {})

with st.sidebar:
    st.markdown("### 🏦 Paisabazaar")
    st.caption("Credit Risk Intelligence Platform")
    st.divider()
    nav = st.radio(
        "NAVIGATION",
        ["Overview", "New Assessment", "Prediction History", "Explainability"],
        index=0,
    )
    st.divider()
    st.markdown("### Model Information")
    st.success("Model Loaded")
    st.caption("Credit Risk Prediction Model")
    st.caption("SHAP Explainability Enabled")
    st.divider()
    st.caption("Paisabazaar Credit Intelligence v1.0")

predictor = CreditRiskPredictor(MODEL_PATH)
model_name = getattr(predictor, "best_model_name", best_model_name)
classes = getattr(predictor, "classes", ["Good", "Poor", "Standard"])

if nav == "Overview":
    st.subheader("Overview")
    st.caption(
        "Credit risk classification trained on 100,000 applications (12,500 customers) using Customer_ID group splitting to prevent leakage."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active / Best Model", model_name)
    k2.metric(
        "Accuracy",
        f"{best_scores.get('accuracy', 0.7131):.4f}",
    )
    k3.metric(
        "Macro F1",
        f"{best_scores.get('macro_f1', 0.6922):.4f}",
    )
    k4.metric(
        "Weighted F1",
        f"{best_scores.get('weighted_f1', 0.7146):.4f}",
    )

    if models_dict:
        st.markdown("#### Model Benchmark Comparison")
        bench_df = (
            pd.DataFrame.from_dict(models_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "Model"})
        )
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown("#### Key Highlights")
    st.markdown(
        f"- **Target Classes:** `{', '.join(classes)}`\n"
        "- **Leakage Prevention:** Grouped train/test split on `Customer_ID` across multi-month records.\n"
        "- **Explainability & Serving:** Per-applicant SHAP drivers, FastAPI inference endpoint, and Docker support."
    )

elif nav == "New Assessment":
    st.subheader("Applicant Inputs")

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    with row1_c1:
        age = st.number_input("Age", min_value=18, max_value=75, value=32, step=1)
    with row1_c2:
        occupation = st.selectbox(
            "Occupation",
            [
                "Engineer",
                "Developer",
                "Scientist",
                "Accountant",
                "Manager",
                "Doctor",
                "Lawyer",
                "Teacher",
                "Other",
            ],
            index=0,
        )
    with row1_c3:
        annual_income = st.number_input(
            "Annual Income", min_value=0.0, value=65000.00, step=1000.0, format="%.2f"
        )
    with row1_c4:
        monthly_salary = st.number_input(
            "Monthly Inhand Salary", min_value=0.0, value=5200.00, step=100.0, format="%.2f"
        )

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        num_bank_accounts = st.number_input(
            "Bank Accounts", min_value=0, max_value=15, value=3, step=1
        )
    with row2_c2:
        num_credit_cards = st.number_input(
            "Credit Cards", min_value=0, max_value=15, value=2, step=1
        )
    with row2_c3:
        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=1.00, max_value=40.00, value=12.00, step=0.5, format="%.2f"
        )
    with row2_c4:
        num_loans = st.number_input(
            "Number of Loans", min_value=0, max_value=15, value=1, step=1
        )

    row3_c1, row3_c2, row3_c3, row3_c4 = st.columns(4)
    with row3_c1:
        delay_days = st.number_input(
            "Delay From Due Date (days)", min_value=0.0, max_value=90.0, value=3.00, step=1.0, format="%.2f"
        )
    with row3_c2:
        delayed_payments = st.number_input(
            "Delayed Payments Count", min_value=0.0, max_value=30.0, value=2.00, step=1.0, format="%.2f"
        )
    with row3_c3:
        credit_inquiries = st.number_input(
            "Credit Inquiries", min_value=0.0, max_value=25.0, value=2.00, step=1.0, format="%.2f"
        )
    with row3_c4:
        credit_mix = st.selectbox(
            "Credit Mix",
            ["Good", "Standard", "Bad"],
            index=0,
        )

    row4_c1, row4_c2, row4_c3, row4_c4 = st.columns(4)
    with row4_c1:
        outstanding_debt = st.number_input(
            "Outstanding Debt", min_value=0.0, value=15000.00, step=500.0, format="%.2f"
        )
    with row4_c2:
        utilization_ratio = st.number_input(
            "Credit Utilization Ratio (%)", min_value=0.00, max_value=100.00, value=32.00, step=1.0, format="%.2f"
        )
    with row4_c3:
        credit_history_age = st.number_input(
            "Credit History Age (months)", min_value=0.0, max_value=500.0, value=180.00, step=6.0, format="%.2f"
        )
    with row4_c4:
        pay_min = st.selectbox(
            "Payment of Min Amount",
            ["No", "Yes", "NM"],
            index=0,
        )

    row5_c1, row5_c2, row5_c3, row5_c4 = st.columns(4)
    with row5_c1:
        total_emi = st.number_input(
            "Total EMI Per Month", min_value=0.0, value=600.00, step=50.0, format="%.2f"
        )
    with row5_c2:
        amount_invested = st.number_input(
            "Amount Invested Monthly", min_value=0.0, value=500.00, step=50.0, format="%.2f"
        )
    with row5_c3:
        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ],
            index=0,
        )
    with row5_c4:
        monthly_balance = st.number_input(
            "Monthly Balance", min_value=0.0, value=800.00, step=50.0, format="%.2f"
        )

    predict_clicked = st.button("Predict Credit Score", type="primary")

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

    if predict_clicked:
        st.divider()
        result = predictor.predict_records([record])[0]
        st.session_state["last_record"] = record
        st.session_state["last_result"] = result
        st.session_state["history"].append({
            "Age": record["Age"],
            "Occupation": record["Occupation"],
            "Annual_Income": record["Annual_Income"],
            "Credit_Mix": record["Credit_Mix"],
            "Prediction": result["predicted_credit_score"],
            "Model": result.get("model_name", model_name),
        })

        st.subheader("Prediction")
        st.success(
            f"Predicted Credit Score: **{result['predicted_credit_score']}** ({result.get('model_name', model_name)})"
        )
        if "probabilities" in result:
            st.json(result["probabilities"])

        st.subheader("Top SHAP Feature Drivers for This Applicant")
        exp_df = explain_single_record(record, MODEL_PATH, top_n=8)
        st.dataframe(exp_df, use_container_width=True)

elif nav == "Prediction History":
    st.subheader("Prediction History")
    if st.session_state["history"]:
        st.dataframe(st.session_state["history"], use_container_width=True)
    else:
        st.info("No predictions made yet.")

elif nav == "Explainability":
    st.subheader("Explainability")
    if st.session_state["last_record"] is not None:
        exp_df = explain_single_record(st.session_state["last_record"], MODEL_PATH, top_n=8)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("Run a prediction first.")
    global_df = get_global_feature_importance(MODEL_PATH, top_n=10)
    st.bar_chart(global_df.set_index("feature")["importance"])