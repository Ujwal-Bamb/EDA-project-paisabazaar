import numpy as np
import pandas as pd
from src.config import DROP_COLS, GROUP_COL, TARGET_COL


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])

    # Safe denominators to avoid division-by-zero / inf
    safe_monthly_sal = data["Monthly_Inhand_Salary"].clip(lower=1.0)
    safe_annual_inc = data["Annual_Income"].clip(lower=1.0)
    safe_loans = data["Num_of_Loan"].clip(lower=1.0)

    data["emi_to_monthly_income"] = (data["Total_EMI_per_month"] / safe_monthly_sal).round(4)
    data["debt_to_annual_income"] = (data["Outstanding_Debt"] / safe_annual_inc).round(4)
    data["investment_to_monthly_income"] = (
        data["Amount_invested_monthly"] / safe_monthly_sal
    ).round(4)
    data["net_monthly_cashflow_est"] = (
        data["Monthly_Inhand_Salary"]
        - data["Total_EMI_per_month"]
        - data["Amount_invested_monthly"]
    ).round(2)
    data["debt_per_loan"] = np.where(
        data["Num_of_Loan"] > 0,
        (data["Outstanding_Debt"] / safe_loans).round(2),
        0.0,
    )

    return data


def split_features_target_groups(df: pd.DataFrame):
    engineered = engineer_features(df)
    y = engineered[TARGET_COL].astype(str)
    groups = engineered[GROUP_COL]
    X = engineered.drop(columns=[TARGET_COL, GROUP_COL])
    return X, y, groups