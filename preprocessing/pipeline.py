"""
Preprocessing pipeline — matches the trained model in churn_artifacts.pkl.

The pickle contains a model trained with 56 features from the original
Merged_Capstone_Code notebook. This pipeline reproduces that exact
feature engineering and encoding so predictions match.

Note: The Capstone_Submission_Ready notebook removes leakage columns
and produces a different model. When a new pickle is exported from that
notebook, update this pipeline accordingly.
"""

import pandas as pd
import numpy as np


# Columns dropped before modeling (from the notebook that produced the pickle)
COLUMNS_TO_DROP = [
    "Customer ID",
    "Zip Code",
    "Lat Long",
    "City",
    "State",
    "Country",
    "Latitude",
    "Longitude",
    "Quarter",
    "Churn",
    # Leakage columns already dropped during data cleaning:
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Customer Status",
]

# Service columns used for Total Services (Cell 26 of both notebooks)
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
    """
    Reproduce feature engineering from the training notebook.
    Matches the 56-feature model in churn_artifacts.pkl.
    """
    df = df.copy()

    # Tenure Groups
    if "Tenure Group" not in df.columns:
        df["Tenure Group"] = pd.cut(
            df["Tenure in Months"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-1 yr", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr", "5-6 yr"],
        )

    # Total Services — sum of binary service columns
    if "Total Services" not in df.columns:
        df["Total Services"] = df[SERVICE_COLS].sum(axis=1)

    # Revenue Per Month
    if "Revenue Per Month" not in df.columns and "Total Revenue" in df.columns:
        df["Revenue Per Month"] = df["Total Revenue"] / (df["Tenure in Months"] + 1)

    # Charge Per Tenure Month
    if "Charge Per Tenure Month" not in df.columns and "Total Charges" in df.columns:
        df["Charge Per Tenure Month"] = df["Total Charges"] / (df["Tenure in Months"] + 1)

    # Revenue Efficiency
    if "Revenue Efficiency" not in df.columns and "Total Revenue" in df.columns:
        df["Revenue Efficiency"] = df["Total Revenue"] / (df["Tenure in Months"] + 1)

    # Charge Per Service
    if "Charge Per Service" not in df.columns:
        df["Charge Per Service"] = df["Monthly Charge"] / (df["Total Services"] + 1)

    return df


def preprocess_customer(customer_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """
    Take a single customer row from df_clean and produce a 1-row DataFrame
    with columns matching the trained model's feature_names exactly.
    """
    df = pd.DataFrame([customer_row])

    # Feature engineering (if not already present)
    df = add_engineered_features(df)

    # Drop non-modeling columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # One-hot encode categoricals (get_dummies with drop_first=True)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Reindex to match training feature set exactly
    df = df.reindex(columns=feature_names, fill_value=0)

    # Ensure correct dtypes
    df = df.astype(float)

    return df
