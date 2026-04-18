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

    # Build profile from non-leakage fields only
    profile = {
        "customer_id": cid,
        "tenure_months": int(row["Tenure in Months"]),
        "contract": row.get("Contract", "Unknown"),
        "monthly_charge": float(row["Monthly Charge"]),
        "internet_type": row.get("Internet Type", "Unknown"),
        "offer": row.get("Offer", "No Offer"),
        "payment_method": row.get("Payment Method", "Unknown"),
        "gender": row.get("Gender", "Unknown"),
        "age": int(row["Age"]) if "Age" in row.index else None,
        "dependents": int(row.get("Dependents", 0)),
        "number_of_referrals": int(row.get("Number of Referrals", 0)),
        "total_services": int(row["Total Services"]) if "Total Services" in row.index else None,
        "state": row.get("State", "Unknown"),
    }

    return {
        "customer_profile": profile,
        "phase": "risk",
    }
