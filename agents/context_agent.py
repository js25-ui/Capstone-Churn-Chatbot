"""Context Agent — looks up a customer by ID and returns their profile."""

from state import ChurnState


def context_agent(state: ChurnState, df_clean) -> dict:
    """Look up customer in df_clean by Customer ID."""
    cid = state["customer_id"]
    match = df_clean[df_clean["Customer ID"] == cid]

    if match.empty:
        return {
            "customer_profile": None,
            "phase": "identify",
        }

    row = match.iloc[0]

    # Build profile from all available fields in df_clean
    def _get(col, default=None):
        return row[col] if col in row.index else default

    profile = {
        "customer_id": cid,
        "tenure_months": int(row["Tenure in Months"]),
        "contract": _get("Contract", "Unknown"),
        "monthly_charge": float(row["Monthly Charge"]),
        "total_charges": float(_get("Total Charges", 0)),
        "total_revenue": float(_get("Total Revenue", 0)),
        "satisfaction_score": int(_get("Satisfaction Score", 0)) if _get("Satisfaction Score") is not None else None,
        "cltv": float(_get("CLTV", 0)),
        "internet_type": _get("Internet Type", "Unknown"),
        "offer": _get("Offer", "No Offer"),
        "payment_method": _get("Payment Method", "Unknown"),
        "gender": _get("Gender", "Unknown"),
        "age": int(row["Age"]) if "Age" in row.index else None,
        "dependents": int(_get("Dependents", 0)),
        "number_of_referrals": int(_get("Number of Referrals", 0)),
        "total_services": int(_get("Total Services", 0)),
        "state": _get("State", "Unknown"),
    }

    return {
        "customer_profile": profile,
        "phase": "risk",
    }
