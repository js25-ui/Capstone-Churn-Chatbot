"""Solution Agent — rule-based retention strategy + Claude personalization."""

import anthropic
from state import ChurnState

client = anthropic.Anthropic()


def retention_strategy(profile: dict, risk_tier: str) -> str:
    """
    Rule-based retention strategy from notebook Week 7.5 (Cell 89).
    Exactly reproduces the notebook logic.
    """
    if risk_tier == "HIGH":
        if profile["contract"] == "Month-to-Month":
            return "Offer a contract upgrade incentive and immediate retention discount."
        elif profile["satisfaction_score"] <= 2:
            return "Prioritize customer support outreach and provide a service recovery offer."
        elif profile["monthly_charge"] > 80:
            return "Offer a pricing review or loyalty discount to reduce cost sensitivity."
        else:
            return "Provide proactive outreach with a personalized retention offer."

    elif risk_tier == "MEDIUM":
        if profile["satisfaction_score"] <= 3:
            return "Send a satisfaction follow-up and offer a targeted service improvement incentive."
        elif profile["total_services"] <= 3:
            return "Promote bundled service discounts to increase engagement."
        else:
            return "Send a targeted loyalty campaign with moderate retention incentives."

    else:  # LOW
        return "Maintain engagement through loyalty messaging and regular check-ins."


def solution_agent(state: ChurnState) -> dict:
    """Generate a personalized retention offer using rule-based baseline + Claude."""
    profile = state["customer_profile"]
    risk_data = state["risk_data"]
    answers = state.get("diagnosis_answers", [])
    questions = state.get("diagnosis_questions", [])

    # Step 1: Rule-based baseline
    baseline = retention_strategy(profile, risk_data["risk_tier"])

    # Build Q&A context
    qa_text = ""
    if questions and answers:
        qa_lines = []
        for q, a in zip(questions, answers):
            qa_lines.append(f"Q: {q}\nA: {a}")
        qa_text = "\n".join(qa_lines)

    driver_lines = []
    for d in risk_data["top_drivers"][:3]:
        driver_lines.append(f"- {d['feature']}: {d['direction']} factor (value: {d['feature_value']:.2f})")
    drivers_text = "\n".join(driver_lines)

    prompt = f"""You are a customer retention specialist crafting a personalized retention message.

Customer profile:
- Tenure: {profile['tenure_months']} months
- Contract: {profile['contract']}
- Monthly charge: ${profile['monthly_charge']:.2f}
- Satisfaction: {profile['satisfaction_score']}/5
- Internet: {profile['internet_type']}
- Services: {profile['total_services']}
- CLTV: ${profile['cltv']:.0f}

Risk: {risk_data['risk_tier']} ({risk_data['churn_probability']:.1%} churn probability)

Top churn drivers:
{drivers_text}

Baseline retention strategy: {baseline}

Customer's responses during diagnosis:
{qa_text if qa_text else "No diagnosis responses collected."}

Craft a warm, personalized retention message to this customer. Include:
1. Acknowledge their specific concerns (from diagnosis answers if available)
2. A concrete offer aligned with the baseline strategy
3. Specific dollar amounts or percentages where appropriate
4. A clear next step

Keep it conversational and under 150 words. Do NOT mention churn risk, SHAP, or ML models."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    offer_text = response.content[0].text.strip()

    return {
        "offers_made": state.get("offers_made", []) + [offer_text],
        "phase": "conversation",
    }
