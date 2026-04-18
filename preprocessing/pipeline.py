"""
Preprocessing pipeline extracted from Capstone_Submission_Ready notebook.
Reproduces the exact feature engineering and encoding used during training.

Key changes from original notebook:
- Leakage columns (Satisfaction Score, CLTV, Total Charges, Total Revenue, etc.) removed
- Tenure Group dropped from model (used for viz only)
- State kept for geographic signal
- Service columns use explicit Yes/No → 1/0 mapping
- Only engineered features: Total Services, Charge Per Service
"""

import pandas as pd
import numpy as np


# Leakage columns removed during data cleaning (Cell 14)
LEAKAGE_COLUMNS = [
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Churn Label",
    "Customer Status",
    "Satisfaction Score",
    "Total Charges",
    "Total Revenue",
    "Total Refunds",
    "Total Long Distance Charges",
    "Total Extra Data Charges",
    "CLTV",
]

# Columns dropped before modeling (Cell 32)
COLUMNS_TO_DROP = [
    "Customer ID",
    "Zip Code",
    "Lat Long",
    "City",
    "Country",
    "Latitude",
    "Longitude",
    "Tenure Group",  # dropped from model — tree handles continuous tenure
    "Quarter",
    "Churn",
]

# Service columns for Total Services count (Cell 26)
SERVICE_COLS = [
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce feature engineering from Cell 26."""
    df = df.copy()

    # Tenure Groups (kept in df_clean for viz, dropped before modeling)
    if "Tenure Group" not in df.columns:
        df["Tenure Group"] = pd.cut(
            df["Tenure in Months"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-1 yr", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr", "5-6 yr"],
        )

    # Total Services — explicit Yes/No mapping (Cell 26)
    if "Total Services" not in df.columns:
        total = 0
        for col in SERVICE_COLS:
            if col in df.columns:
                if df[col].dtype == object:
                    total = total + (df[col] == "Yes").astype(int)
                else:
                    total = total + df[col]
        df["Total Services"] = total

    # Charge Per Service (Cell 26)
    if "Charge Per Service" not in df.columns:
        df["Charge Per Service"] = df["Monthly Charge"] / (df["Total Services"] + 1)

    return df


def preprocess_customer(customer_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """
    Take a single customer row from df_clean and produce a 1-row DataFrame
    with columns matching the trained model's feature_names exactly.
    """
    df = pd.DataFrame([customer_row])

    # Feature engineering
    df = add_engineered_features(df)

    # Drop non-modeling columns (leakage + non-predictive)
    all_drops = LEAKAGE_COLUMNS + COLUMNS_TO_DROP
    cols_to_drop = [c for c in all_drops if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # One-hot encode categoricals (Cell 34: get_dummies with drop_first=True)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Reindex to match training feature set exactly
    df = df.reindex(columns=feature_names, fill_value=0)

    # Ensure correct dtypes
    df = df.astype(float)

    return df
