"""Diagnosis Agent — generates targeted questions based on top SHAP drivers."""

import anthropic
from state import ChurnState

client = anthropic.Anthropic()


def diagnosis_agent(state: ChurnState) -> dict:
    """Generate 2-3 diagnostic questions based on top SHAP risk drivers."""
    risk_data = state["risk_data"]
    profile = state["customer_profile"]
    top_drivers = risk_data["top_drivers"]

    # Build driver summary for the prompt
    driver_lines = []
    for d in top_drivers[:5]:
        driver_lines.append(
            f"- {d['feature']}: value={d['feature_value']:.2f}, "
            f"SHAP={d['shap_value']:.4f} ({d['direction']})"
        )
    drivers_text = "\n".join(driver_lines)

    prompt = f"""You are a customer retention specialist. A customer has been flagged as {risk_data['risk_tier']} risk for churn (probability: {risk_data['churn_probability']:.1%}).

Customer profile:
- Tenure: {profile['tenure_months']} months
- Contract: {profile['contract']}
- Monthly charge: ${profile['monthly_charge']:.2f}
- Satisfaction score: {profile['satisfaction_score']}/5
- Internet type: {profile['internet_type']}
- Total services: {profile['total_services']}

Top churn drivers (from SHAP analysis):
{drivers_text}

Generate exactly 3 targeted diagnostic questions to ask this customer, based on their top risk drivers:
- If a top driver relates to Satisfaction Score, ask about recent service issues
- If it relates to Contract type, ask about flexibility vs commitment preferences
- If it relates to Monthly Charge or pricing features, ask about budget concerns
- If it relates to Tenure, ask about their onboarding or early experience
- If it relates to services (Internet, Streaming, etc.), ask about service quality

Be empathetic and natural. Do NOT mention SHAP, machine learning, or risk scores.
Return ONLY the 3 questions, one per line, numbered 1-3."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    questions = []
    for line in response.content[0].text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Strip the number prefix
            q = line.lstrip("0123456789.)")
            questions.append(q.strip())

    return {
        "diagnosis_questions": questions[:3],
        "diagnosis_step": 0,
        "phase": "diagnosis",
    }
