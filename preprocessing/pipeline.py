"""
Preprocessing pipeline extracted from Merged_Capstone_Code notebook.
Reproduces the exact feature engineering and encoding used during training.
"""

import pandas as pd
import numpy as np


# Columns dropped before modeling (Cell 31)
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
]

# Service columns used for Total Services (Cell 26)
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

# Categorical columns that get one-hot encoded (Cell 33)
CATEGORICAL_COLS = [
    "Contract",
    "Gender",
    "Internet Type",
    "Offer",
    "Payment Method",
    "Tenure Group",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce feature engineering from Cell 26."""
    df = df.copy()

    # Tenure Groups
    df["Tenure Group"] = pd.cut(
        df["Tenure in Months"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-1 yr", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr", "5-6 yr"],
    )

    # Total Services
    df["Total Services"] = df[SERVICE_COLS].sum(axis=1)

    # Revenue Per Month
    df["Revenue Per Month"] = df["Total Revenue"] / (df["Tenure in Months"] + 1)

    # Charge Per Tenure Month
    df["Charge Per Tenure Month"] = df["Total Charges"] / (df["Tenure in Months"] + 1)

    # Revenue Efficiency
    df["Revenue Efficiency"] = df["Total Revenue"] / (df["Tenure in Months"] + 1)

    # Charge Per Service
    df["Charge Per Service"] = df["Monthly Charge"] / (df["Total Services"] + 1)

    return df


def preprocess_customer(customer_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """
    Take a single customer row from df_clean and produce a 1-row DataFrame
    with columns matching the trained model's feature_names exactly.
    """
    df = pd.DataFrame([customer_row])

    # Feature engineering (if not already present)
    if "Total Services" not in df.columns:
        df = add_engineered_features(df)

    # Drop non-modeling columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # One-hot encode categoricals (Cell 33: get_dummies with drop_first=True)
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)

    # Reindex to match training feature set exactly
    df = df.reindex(columns=feature_names, fill_value=0)

    # Ensure correct dtypes
    df = df.astype(float)

    return df
