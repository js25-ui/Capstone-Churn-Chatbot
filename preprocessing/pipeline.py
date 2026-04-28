"""
Preprocessing pipeline — matches the clean 41-feature model (no leakage).

Trained from Capstone_Submission_Ready notebook with all post-outcome
columns removed: Satisfaction Score, Total Charges, Total Revenue,
Total Refunds, Total Long Distance Charges, Total Extra Data Charges, CLTV.
"""

import pandas as pd
import numpy as np


# All columns to drop before modeling
COLUMNS_TO_DROP = [
    # Non-predictive identifiers
    "Customer ID", "Zip Code", "Lat Long", "City", "State",
    "Country", "Latitude", "Longitude", "Quarter",
    # Target
    "Churn",
    # Leakage / post-outcome (already removed from df_clean, but safety check)
    "Churn Category", "Churn Reason", "Churn Score", "Churn Label",
    "Customer Status", "Satisfaction Score", "Total Charges", "Total Revenue",
    "Total Refunds", "Total Long Distance Charges", "Total Extra Data Charges", "CLTV",
    # Tenure Group dropped from model (tree handles continuous tenure)
    "Tenure Group",
]

# Service columns for Total Services count
SERVICE_COLS = [
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from notebook Cell 26."""
    df = df.copy()

    if "Tenure Group" not in df.columns:
        df["Tenure Group"] = pd.cut(
            df["Tenure in Months"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-1 yr", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr", "5-6 yr"],
        )

    if "Total Services" not in df.columns:
        total = 0
        for col in SERVICE_COLS:
            if col in df.columns:
                if df[col].dtype == object:
                    total = total + (df[col] == "Yes").astype(int)
                else:
                    total = total + df[col]
        df["Total Services"] = total

    if "Charge Per Service" not in df.columns:
        df["Charge Per Service"] = df["Monthly Charge"] / (df["Total Services"] + 1)

    return df


def preprocess_customer(customer_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """Produce a 1-row DataFrame matching the trained model's 41 features."""
    df = pd.DataFrame([customer_row])
    df = add_engineered_features(df)

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    df = df.reindex(columns=feature_names, fill_value=0)
    df = df.astype(float)
    return df
