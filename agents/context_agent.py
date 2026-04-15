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
    profile = {
        "customer_id": cid,
        "tenure_months": int(row["Tenure in Months"]),
        "contract": row["Contract"],
        "satisfaction_score": int(row["Satisfaction Score"]),
        "monthly_charge": float(row["Monthly Charge"]),
        "total_services": int(row["Total Services"]) if "Total Services" in row.index else None,
        "payment_method": row["Payment Method"],
        "cltv": float(row["CLTV"]),
        "internet_type": row["Internet Type"],
        "offer": row["Offer"],
        "total_revenue": float(row["Total Revenue"]),
        "total_charges": float(row["Total Charges"]),
        "gender": row["Gender"],
        "age": int(row["Age"]),
        "number_of_referrals": int(row["Number of Referrals"]),
        "dependents": int(row["Dependents"]),
    }

    return {
        "customer_profile": profile,
        "phase": "risk",
    }
