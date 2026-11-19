
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

from pathlib import Path
from datetime import datetime

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODEL_PATH
from src.explain import explain_single_record
from src.predict import CreditRiskPredictor


st.set_page_config(
    page_title="Paisabazaar | Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #F5F7FB;
}

.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #14213D;
}

.subtitle {
    font-size: 14px;
    color: #64748B;
    margin-bottom: 25px;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px;
    min-height: 110px;
}

.metric-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #14213D;
    font-size: 26px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    color: #14213D;
    font-size: 19px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 12px;
}

.result-card {
    background-color: #F1F5F9;
    border-radius: 12px;
    padding: 22px;
    border: 1px solid #E2E8F0;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 20px;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "latest_record" not in st.session_state:
    st.session_state.latest_record = None

if "latest_explanation" not in st.session_state:
    st.session_state.latest_explanation = None

if "sample_mode" not in st.session_state:
    st.session_state.sample_mode = False


# ============================================================
# 4. MODEL LOADING
# ============================================================

@st.cache_resource
def load_predictor():
    return CreditRiskPredictor(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Model artifact not found. "
        "Please run python -m src.train first."
    )
    st.stop()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def format_number(value):
    return f"{value:,.2f}"


def create_metric(label, value, description=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div style="color:#94A3B8;font-size:11px;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def validate_inputs(record):

    errors = []

    if record["Annual_Income"] <= 0:
        errors.append("Annual income must be greater than zero.")

    if record["Monthly_Inhand_Salary"] <= 0:
        errors.append("Monthly salary must be greater than zero.")

    if record["Monthly_Inhand_Salary"] > record["Annual_Income"]:
        errors.append(
            "Monthly salary cannot exceed annual income."
        )

    if record["Outstanding_Debt"] < 0:
        errors.append("Outstanding debt cannot be negative.")

    if record["Credit_Utilization_Ratio"] > 100:
        errors.append("Credit utilization cannot exceed 100%.")

    return errors


def add_to_history(record, result):

    history_item = {
        "Assessment ID": len(st.session_state.history) + 1,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Age": record["Age"],
        "Annual Income": record["Annual_Income"],
        "Outstanding Debt": record["Outstanding_Debt"],
        "Credit Score": str(result["predicted_credit_score"]),
        "Model": result["model_name"]
    }

    st.session_state.history.append(history_item)


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("# 🏦 Paisabazaar")

    st.caption("Credit Risk Intelligence Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "New Assessment",
            "Prediction History",
            "Explainability"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### Model Information")

    st.success("Model Loaded")

    st.caption("Credit Risk Prediction Model")

    st.caption("SHAP Explainability Enabled")

    st.divider()

    st.caption("Paisabazaar Credit Intelligence v1.0")


# ============================================================
# 7. OVERVIEW DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">Credit Risk Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor applicant assessments and credit risk predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    total_assessments = len(st.session_state.history)

    if total_assessments:

        scores = [
            item["Credit Score"]
            for item in st.session_state.history
        ]

        latest_score = scores[-1]

    else:
        latest_score = "N/A"

    c1, c2, c3 = st.columns(3)

    with c1:
        create_metric(
            "Total Assessments",
            total_assessments,
            "Current session"
        )

    with c2:
        create_metric(
            "Latest Prediction",
            latest_score,
            "Most recent assessment"
        )

    with c3:
        create_metric(
            "Model Status",
            "Active",
            "Prediction engine loaded"
        )

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("### Recent Assessments")

        if st.session_state.history:

            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                history_df.tail(5).iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No assessments available. "
                "Go to New Assessment to evaluate an applicant."
            )

    with right:

        st.markdown("### Quick Actions")

        if st.button(
            "➕ New Applicant Assessment",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.latest_result = None
            st.session_state.latest_record = None
            st.session_state.latest_explanation = None
            st.rerun()

        if st.button(
            "📊 View Prediction History",
            use_container_width=True
        ):
            st.info(
                "Select Prediction History from the sidebar."
            )

    st.divider()

    st.markdown("### Assessment Activity")

    if st.session_state.history:

        activity_df = pd.DataFrame(
            st.session_state.history
        )

        activity_df["Assessment"] = (
            "Assessment " +
            activity_df["Assessment ID"].astype(str)
        )

        fig = px.bar(
            activity_df,
            x="Assessment",
            y="Age",
            title="Applicant Assessment Activity",
            labels={"Age": "Applicant Age"}
        )

        fig.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Charts will appear after your first applicant assessment."
        )


# ============================================================
# 8. NEW APPLICANT ASSESSMENT
# ============================================================

elif page == "New Assessment":

    st.markdown(
        '<div class="main-title">Applicant Credit Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter applicant financial and credit history details '
        'to generate a credit score prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("applicant_form"):

        # ----------------------------------------------------
        # PERSONAL INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">👤 Personal Information</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=75,
                value=32
            )

        with c2:
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
                    "Other"
                ]
            )

        with c3:
            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=65000.0,
                step=1000.0
            )

        monthly_salary = st.number_input(
            "Monthly Inhand Salary",
            min_value=0.0,
            value=5200.0,
            step=100.0
        )

        st.divider()

        # ----------------------------------------------------
        # BANKING INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🏦 Banking Information</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            num_bank_accounts = st.number_input(
                "Bank Accounts",
                min_value=0,
                max_value=15,
                value=3
            )

        with c2:
            num_credit_cards = st.number_input(
                "Credit Cards",
                min_value=0,
                max_value=15,
                value=2
            )

        with c3:
            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=1.0,
                max_value=40.0,
                value=12.0,
                step=0.5
            )

        with c4:
            num_loans = st.number_input(
                "Number of Loans",
                min_value=0,
                max_value=15,
                value=1
            )

        st.divider()

        # ----------------------------------------------------
        # CREDIT HISTORY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📋 Credit History</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            delay_days = st.number_input(
                "Delay From Due Date (Days)",
                min_value=0.0,
                max_value=90.0,
                value=3.0
            )

        with c2:
            delayed_payments = st.number_input(
                "Delayed Payments Count",
                min_value=0.0,
                max_value=30.0,
                value=2.0
            )

        with c3:
            credit_inquiries = st.number_input(
                "Credit Inquiries",
                min_value=0.0,
                max_value=25.0,
                value=2.0
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            credit_mix = st.selectbox(
                "Credit Mix",
                ["Good", "Standard", "Bad"]
            )

        with c2:
            credit_history_age = st.number_input(
                "Credit History Age (Months)",
                min_value=0.0,
                max_value=500.0,
                value=180.0,
                step=6.0
            )

        with c3:
            pay_min = st.selectbox(
                "Payment of Minimum Amount",
                ["No", "Yes", "NM"]
            )

        st.divider()

        # ----------------------------------------------------
        # FINANCIAL OBLIGATIONS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">💳 Financial Obligations</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            outstanding_debt = st.number_input(
                "Outstanding Debt",
                min_value=0.0,
                value=15000.0,
                step=500.0
            )

        with c2:
            utilization_ratio = st.number_input(
                "Credit Utilization Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=32.0
            )

        with c3:
            total_emi = st.number_input(
                "Total EMI Per Month",
                min_value=0.0,
                value=600.0
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            amount_invested = st.number_input(
                "Amount Invested Monthly",
                min_value=0.0,
                value=500.0
            )

        with c2:
            payment_behaviour = st.selectbox(
                "Payment Behaviour",
                [
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments",
                    "High_spent_Small_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments"
                ]
            )

        with c3:
            monthly_balance = st.number_input(
                "Monthly Balance",
                min_value=0.0,
                value=800.0
            )

        st.divider()

        submit = st.form_submit_button(
            "🔍 Predict Credit Score",
            use_container_width=True,
            type="primary"
        )

    # --------------------------------------------------------
    # BUILD RECORD
    # --------------------------------------------------------

    if submit:

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
            "Monthly_Balance": float(monthly_balance)
        }

        errors = validate_inputs(record)

        if errors:

            for error in errors:
                st.error(error)

        else:

            with st.spinner(
                "Analyzing applicant financial profile..."
            ):

                try:

                    result = predictor.predict_records(
                        [record]
                    )[0]

                    explanation = explain_single_record(
                        record,
                        MODEL_PATH,
                        top_n=8
                    )

                    st.session_state.latest_result = result
                    st.session_state.latest_record = record
                    st.session_state.latest_explanation = explanation

                    add_to_history(record, result)

                    st.success(
                        "Credit score prediction generated successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    if st.session_state.latest_result is not None:

        result = st.session_state.latest_result

        st.divider()

        st.markdown("## Prediction Results")

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown("### Predicted Credit Score")

        st.markdown(
            f"## {result['predicted_credit_score']}"
        )

        st.caption(
            f"Model: {result['model_name']}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if "probabilities" in result:

            st.markdown("### Prediction Probabilities")

            probabilities = result["probabilities"]

            if isinstance(probabilities, dict):

                probability_df = pd.DataFrame(
                    {
                        "Credit Category": list(probabilities.keys()),
                        "Probability": list(probabilities.values())
                    }
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    probability_df,
                    x="Credit Category",
                    y="Probability",
                    text="Probability",
                    title="Predicted Class Probabilities"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.write(probabilities)

        st.info(
            "For detailed feature-level explanations, "
            "open the Explainability section from the sidebar."
        )


# ============================================================
# 9. PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Review previous applicant assessments in this session.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        c1, c2 = st.columns([3, 1])

        with c1:

            st.markdown("### Assessment Records")

        with c2:

            csv = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Export CSV",
                data=csv,
                file_name="credit_risk_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        if st.button(
            "Clear Assessment History",
            type="secondary"
        ):

            st.session_state.history = []

            st.success(
                "Session assessment history cleared."
            )

            st.rerun()

    else:

        st.info(
            "No prediction history available yet."
        )


# ============================================================
# 10. SHAP EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.markdown(
        '<div class="main-title">Model Explainability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand the key features influencing the latest prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    explanation = st.session_state.latest_explanation

    record = st.session_state.latest_record

    result = st.session_state.latest_result

    if explanation is None or record is None:

        st.warning(
            "No applicant prediction available. "
            "Please complete a New Assessment first."
        )

    else:

        st.markdown("### Applicant Prediction")

        c1, c2 = st.columns(2)

        with c1:

            create_metric(
                "Predicted Credit Score",
                result["predicted_credit_score"],
                result["model_name"]
            )

        with c2:

            create_metric(
                "Features Explained",
                len(explanation),
                "Top contributing features"
            )

        st.divider()

        st.markdown("### Top SHAP Feature Drivers")

        st.caption(
            "SHAP values represent how individual features "
            "contribute to the model output. Their interpretation "
            "depends on the model's output scale."
        )

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE CHART
        # ----------------------------------------------------

        if (
            isinstance(explanation, pd.DataFrame)
            and not explanation.empty
        ):

            st.markdown("### Feature Contribution Visualization")

            numeric_columns = explanation.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:

                shap_candidates = [
                    col for col in numeric_columns
                    if "shap" in col.lower()
                ]

                value_column = (
                    shap_candidates[0]
                    if shap_candidates
                    else numeric_columns[-1]
                )

                feature_column = (
                    "feature"
                    if "feature" in explanation.columns
                    else explanation.columns[0]
                )

                chart_df = explanation[
                    [feature_column, value_column]
                ].copy()

                chart_df[feature_column] = (
                    chart_df[feature_column].astype(str)
                )

                chart_df = chart_df.sort_values(
                    value_column
                )

                fig = px.bar(
                    chart_df,
                    x=value_column,
                    y=feature_column,
                    orientation="h",
                    title="SHAP Feature Contributions",
                    labels={
                        value_column: "SHAP Value",
                        feature_column: "Feature"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    height=450,
                    yaxis_title=None
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numeric SHAP contribution column "
                    "was found in the explanation output."
                )

        st.divider()

        st.markdown("### Applicant Financial Profile")

        profile_df = pd.DataFrame(
            list(record.items()),
            columns=["Feature", "Applicant Value"]
        )

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Paisabazaar Credit Risk Intelligence Platform |
        Machine Learning & SHAP Explainability
    </div>
    """,
    unsafe_allow_html=True
)
