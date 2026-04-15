"""Escalation Agent — builds a handoff summary for a human agent."""

from state import ChurnState


def escalation_agent(state: ChurnState) -> dict:
    """Build a structured handoff summary for human agent escalation."""
    profile = state.get("customer_profile", {})
    risk_data = state.get("risk_data", {})
    offers = state.get("offers_made", [])
    sentiment = state.get("sentiment_history", [])

    drivers_text = ""
    if risk_data.get("top_drivers"):
        lines = []
        for d in risk_data["top_drivers"][:5]:
            lines.append(f"  - {d['feature']}: {d['direction']} (SHAP: {d['shap_value']:.4f}, value: {d['feature_value']:.2f})")
        drivers_text = "\n".join(lines)

    avg_sentiment = sum(sentiment) / len(sentiment) if sentiment else 0
    trend = "declining" if len(sentiment) >= 2 and sentiment[-1] < sentiment[0] else "stable"

    # Authority recommendation based on risk tier and CLTV
    cltv = profile.get("cltv", 0)
    tier = risk_data.get("risk_tier", "UNKNOWN")
    if tier == "HIGH" and cltv > 5000:
        authority = "Authorize up to 30% discount, free premium upgrade for 6 months, and contract credit up to $200"
    elif tier == "HIGH":
        authority = "Authorize up to 25% discount and contract credit up to $100"
    elif tier == "MEDIUM":
        authority = "Authorize up to 15% discount or service upgrade"
    else:
        authority = "Standard retention offers apply"

    summary = f"""
========================================
  ESCALATION HANDOFF SUMMARY
========================================

CUSTOMER: {profile.get('customer_id', 'Unknown')}
  Age: {profile.get('age', 'N/A')} | Tenure: {profile.get('tenure_months', 'N/A')} months
  Contract: {profile.get('contract', 'N/A')} | Monthly: ${profile.get('monthly_charge', 0):.2f}
  CLTV: ${profile.get('cltv', 0):.0f} | Satisfaction: {profile.get('satisfaction_score', 'N/A')}/5

CHURN RISK: {tier} ({risk_data.get('churn_probability', 0):.1%})

TOP SHAP DRIVERS:
{drivers_text if drivers_text else '  No data available'}

OFFERS ALREADY MADE:
{chr(10).join(f'  - {o[:100]}...' for o in offers) if offers else '  None'}

SENTIMENT: avg={avg_sentiment:.1f}/5, trend={trend}
  History: {sentiment}

RECOMMENDED NEXT ACTION:
  {authority}
========================================
""".strip()

    return {
        "escalation_summary": summary,
        "phase": "escalated",
    }
