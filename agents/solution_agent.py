"""
Solution Agent — retention strategy + message generation.
Exact code from notebook Cells 82 and 84 (clean model, no leakage).
"""

import numpy as np
from state import ChurnState


def retention_strategy(row: dict) -> str:
    """Exact code from notebook Cell 82 (clean model)."""
    if row["Risk Tier"] == "HIGH":
        contract = row.get("Contract", "Unknown")
        charge = row.get("Monthly Charge", 0)
        if contract == "Month-to-Month":
            return "Offer contract upgrade incentive and immediate retention discount."
        elif charge > 80:
            return "Offer pricing review or loyalty discount."
        else:
            return "Proactive outreach with personalized retention offer."
    elif row["Risk Tier"] == "MEDIUM":
        contract = row.get("Contract", "Unknown")
        if contract == "Month-to-Month":
            return "Send targeted email with contract lock-in discount."
        else:
            return "Offer complimentary service upgrade for 3 months."
    else:
        return "Continue standard engagement. Monitor for changes."


def generate_retention_message(risk_profile: dict, use_llm=False, client=None) -> str:
    """Exact template fallback from notebook Cell 84."""
    prob = risk_profile["churn_probability"]
    tier = risk_profile["risk_tier"]
    drivers = risk_profile["top_risk_drivers"]

    # Template fallback
    top_driver = drivers[0][0] if drivers else "general factors"
    top_magnitude = abs(drivers[0][1]) if drivers else 0

    if tier == "HIGH":
        if 'Contract' in top_driver or 'Month' in top_driver:
            discount = "25%" if top_magnitude > 0.3 else "15%"
            return (f"We would like to offer you an exclusive {discount} discount "
                    f"to switch to an annual plan, plus complimentary Tech Support for 3 months.")
        else:
            return ("We noticed your account could benefit from a plan review. "
                    "We are offering you 20% off for 6 months plus priority support.")
    elif tier == "MEDIUM":
        return ("We have a special offer — switch to an annual plan and save 15% "
                "on your monthly bill. Reply YES to lock it in.")
    else:
        return "Thanks for being with us! Check our app for exclusive deals."


def solution_agent(state: ChurnState) -> dict:
    """Generate a personalized retention offer."""
    profile = state["customer_profile"]
    risk_data = state["risk_data"]

    strategy_row = {
        "Risk Tier": risk_data["risk_tier"],
        "Contract": profile.get("contract", "Unknown"),
        "Monthly Charge": profile.get("monthly_charge", 0),
    }
    baseline = retention_strategy(strategy_row)

    risk_profile = {
        "churn_probability": risk_data["churn_probability"],
        "risk_tier": risk_data["risk_tier"],
        "top_risk_drivers": [
            (d["feature"], d["shap_value"], d["feature_value"])
            for d in risk_data.get("top_drivers", [])
        ],
    }
    message = generate_retention_message(risk_profile)

    return {
        "baseline_strategy": baseline,
        "template_message": message,
        "offers_made": state.get("offers_made", []) + [message],
        "phase": "conversation",
    }
