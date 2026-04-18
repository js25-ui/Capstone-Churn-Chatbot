"""Risk Agent — runs churn prediction and SHAP analysis on a customer."""

import numpy as np
import pandas as pd
from state import ChurnState
from preprocessing.pipeline import preprocess_customer


def risk_agent(state: ChurnState, df_clean, model, explainer, feature_names) -> dict:
    """Predict churn probability and extract top SHAP drivers."""
    cid = state["customer_id"]
    row = df_clean[df_clean["Customer ID"] == cid].iloc[0]

    # Preprocess to match training features
    X = preprocess_customer(row, feature_names)

    # Predict
    churn_prob = float(model.predict_proba(X)[0][1])

    # Risk tier
    if churn_prob >= 0.60:
        tier = "HIGH"
    elif churn_prob >= 0.30:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # SHAP analysis — use [1] for binary classifiers (positive class)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    feature_impacts = sorted(
        zip(feature_names, shap_vals, X.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_drivers = []
    for name, sv, fv in feature_impacts[:5]:
        top_drivers.append({
            "feature": name,
            "shap_value": float(sv),
            "feature_value": float(fv),
            "direction": "risk" if sv > 0 else "protective",
        })

    risk_data = {
        "churn_probability": churn_prob,
        "risk_tier": tier,
        "top_drivers": top_drivers,
    }

    return {
        "risk_data": risk_data,
        "phase": "diagnosis",
    }
