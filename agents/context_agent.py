"""Context Agent — looks up a customer by ID and returns their profile."""

from state import ChurnState


def context_agent(state: ChurnState, df_clean) -> dict:
    """Look up customer in df_clean by Customer ID."""
    cid = state["customer_id"]
    match = df_clean[df_clean["Customer ID"] == cid]

    if match.empty:
        return {"customer_profile": None, "phase": "identify"}

    row = match.iloc[0]

    def _g(col, default=None):
        return row[col] if col in row.index else default

    # Only pre-outcome features — no leakage columns
    profile = {
        "customer_id": cid,
        "tenure_months": int(row["Tenure in Months"]),
        "contract": _g("Contract", "Unknown"),
        "monthly_charge": float(row["Monthly Charge"]),
        "internet_type": _g("Internet Type", "Unknown"),
        "offer": _g("Offer", "No Offer"),
        "payment_method": _g("Payment Method", "Unknown"),
        "gender": _g("Gender", "Unknown"),
        "age": int(row["Age"]) if "Age" in row.index else None,
        "dependents": int(_g("Dependents", 0)),
        "number_of_referrals": int(_g("Number of Referrals", 0)),
        "total_services": int(_g("Total Services", 0)),
    }

    return {"customer_profile": profile, "phase": "risk"}
