"""LangGraph orchestrator — routes between agents and maintains shared state."""

import pickle
import os
import random

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from state import ChurnState
from agents.context_agent import context_agent
from agents.risk_agent import risk_agent
from agents.diagnosis_agent import diagnosis_agent
from agents.solution_agent import solution_agent
from agents.sentiment_agent import sentiment_agent
from agents.escalation_agent import escalation_agent


# ---------------------------------------------------------------------------
# Load artifacts once at module import
# ---------------------------------------------------------------------------
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "churn_artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    _artifacts = pickle.load(f)

MODEL = _artifacts["model"]
EXPLAINER = _artifacts["explainer"]
FEATURE_NAMES = _artifacts["feature_names"]
DF_CLEAN = _artifacts["df_clean"]

# Pre-compute engineered features on df_clean so lookups work
from preprocessing.pipeline import add_engineered_features

if "Total Services" not in DF_CLEAN.columns:
    DF_CLEAN = add_engineered_features(DF_CLEAN)


def get_random_customer_id() -> str:
    return random.choice(DF_CLEAN["Customer ID"].tolist())


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_router(state: ChurnState) -> dict:
    """Passthrough node for conditional routing."""
    return {}


def node_greeting(state: ChurnState) -> dict:
    sample_ids = random.sample(DF_CLEAN["Customer ID"].tolist(), 3)
    msg = (
        "Welcome! I'm your Customer Retention Assistant. "
        "I can analyze any customer's churn risk and help craft a personalized retention strategy.\n\n"
        f"Please enter a **Customer ID** to get started.\n"
        f"Here are some sample IDs you can try: `{sample_ids[0]}`, `{sample_ids[1]}`, `{sample_ids[2]}`\n\n"
        "Or type **random** to analyze a random customer."
    )
    return {"messages": [AIMessage(content=msg)], "phase": "identify"}


def node_context(state: ChurnState) -> dict:
    result = context_agent(state, DF_CLEAN)
    if result["customer_profile"] is None:
        msg = f"Customer ID `{state['customer_id']}` not found. Please try another ID."
        return {"messages": [AIMessage(content=msg)], "phase": "identify", "customer_profile": None}

    p = result["customer_profile"]
    msg = (
        f"**Customer Found: {p['customer_id']}**\n\n"
        f"| Field | Value |\n|-------|-------|\n"
        f"| Age | {p['age']} |\n"
        f"| Tenure | {p['tenure_months']} months |\n"
        f"| Contract | {p['contract']} |\n"
        f"| Monthly Charge | ${p['monthly_charge']:.2f} |\n"
        f"| Satisfaction | {p['satisfaction_score']}/5 |\n"
        f"| Internet Type | {p['internet_type']} |\n"
        f"| Total Services | {p['total_services']} |\n"
        f"| Payment Method | {p['payment_method']} |\n"
        f"| CLTV | ${p['cltv']:.0f} |\n"
        f"| Offer | {p['offer']} |\n\n"
        "Analyzing churn risk..."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "customer_profile": result["customer_profile"],
        "phase": "risk",
    }


def node_risk(state: ChurnState) -> dict:
    result = risk_agent(state, DF_CLEAN, MODEL, EXPLAINER, FEATURE_NAMES)
    rd = result["risk_data"]

    tier_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    emoji = tier_emoji.get(rd["risk_tier"], "")

    driver_lines = []
    for d in rd["top_drivers"]:
        arrow = "↑" if d["direction"] == "risk" else "↓"
        driver_lines.append(
            f"| {d['feature']} | {d['feature_value']:.2f} | {arrow} {d['direction']} | {d['shap_value']:.4f} |"
        )
    drivers_table = "\n".join(driver_lines)

    msg = (
        f"## {emoji} Risk Assessment: **{rd['risk_tier']}**\n"
        f"**Churn Probability: {rd['churn_probability']:.1%}**\n\n"
        f"### Top Churn Drivers\n"
        f"| Feature | Value | Direction | Impact |\n"
        f"|---------|-------|-----------|--------|\n"
        f"{drivers_table}\n\n"
        "Let me ask a few questions to better understand the situation..."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "risk_data": result["risk_data"],
        "phase": "diagnosis",
    }


def node_diagnosis(state: ChurnState) -> dict:
    result = diagnosis_agent(state)
    questions = result["diagnosis_questions"]
    if not questions:
        return {"phase": "solution", "diagnosis_questions": [], "diagnosis_step": 0}

    msg = f"**Question 1 of {len(questions)}:**\n{questions[0]}"
    return {
        "messages": [AIMessage(content=msg)],
        "diagnosis_questions": questions,
        "diagnosis_step": 0,
        "diagnosis_answers": [],
        "phase": "diagnosis",
    }


def node_collect_answer(state: ChurnState) -> dict:
    questions = state.get("diagnosis_questions", [])
    answers = state.get("diagnosis_answers", [])
    step = state.get("diagnosis_step", 0)

    last_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    answers = list(answers) + [last_msg]
    next_step = step + 1

    if next_step < len(questions):
        msg = f"**Question {next_step + 1} of {len(questions)}:**\n{questions[next_step]}"
        return {
            "messages": [AIMessage(content=msg)],
            "diagnosis_answers": answers,
            "diagnosis_step": next_step,
            "phase": "diagnosis",
        }
    else:
        return {
            "diagnosis_answers": answers,
            "diagnosis_step": next_step,
            "phase": "solution",
        }


def node_solution(state: ChurnState) -> dict:
    result = solution_agent(state)
    offers = result["offers_made"]
    latest_offer = offers[-1] if offers else ""

    msg = (
        f"## Personalized Retention Offer\n\n"
        f"{latest_offer}\n\n"
        "---\n"
        "*How does this sound? I can adjust the offer, answer questions, or you can enter a new Customer ID to analyze another customer.*"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "offers_made": offers,
        "phase": "conversation",
    }


def node_escalation(state: ChurnState) -> dict:
    result = escalation_agent(state)
    msg = (
        "I understand your frustration, and I want to make sure you get the help you deserve. "
        "I'm connecting you with a senior retention specialist who has more authority to help.\n\n"
        f"```\n{result['escalation_summary']}\n```\n\n"
        "*A specialist will be with you shortly. Is there anything else I can note for them?*"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "escalation_summary": result["escalation_summary"],
        "escalated": True,
        "phase": "escalated",
    }


def node_conversation(state: ChurnState) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    profile = state.get("customer_profile", {})
    risk_data = state.get("risk_data", {})
    offers = state.get("offers_made", [])

    last_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=300,
        system=(
            "You are a customer retention specialist. You've already made the customer an offer. "
            "Be helpful, empathetic, and flexible. You can adjust offers within reason. "
            "Do NOT mention ML models, SHAP, churn scores, or internal systems. "
            f"Customer: {profile.get('customer_id', 'Unknown')}, "
            f"Risk: {risk_data.get('risk_tier', 'Unknown')}, "
            f"Last offer: {offers[-1][:200] if offers else 'None'}"
        ),
        messages=[{"role": "user", "content": last_msg}],
    )

    return {"messages": [AIMessage(content=response.content[0].text)]}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_from_router(state: ChurnState) -> str:
    if state.get("escalated") and state.get("phase") == "escalated":
        return "escalation"

    phase = state.get("phase", "greeting")

    if phase == "greeting":
        return "greeting"
    elif phase == "identify":
        return "context"
    elif phase == "risk":
        return "risk"
    elif phase == "diagnosis":
        questions = state.get("diagnosis_questions")
        if questions is None:
            return "diagnosis_generate"
        step = state.get("diagnosis_step", 0)
        if step < len(questions):
            return "collect_answer"
        return "solution"
    elif phase == "solution":
        return "solution"
    elif phase == "conversation":
        return "conversation"
    elif phase == "escalated":
        return "escalation"
    return "greeting"


def route_after_context(state: ChurnState) -> str:
    if state.get("customer_profile"):
        return "risk"
    return END


def route_after_collect(state: ChurnState) -> str:
    if state.get("phase") == "solution":
        return "solution"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ChurnState)

    graph.add_node("router", node_router)
    graph.add_node("greeting", node_greeting)
    graph.add_node("context", node_context)
    graph.add_node("risk", node_risk)
    graph.add_node("diagnosis_generate", node_diagnosis)
    graph.add_node("collect_answer", node_collect_answer)
    graph.add_node("solution", node_solution)
    graph.add_node("conversation", node_conversation)
    graph.add_node("escalation", node_escalation)

    # Entry point is always the router
    graph.set_entry_point("router")

    # Router dispatches based on phase
    graph.add_conditional_edges("router", route_from_router)

    # greeting → END (wait for user input)
    graph.add_edge("greeting", END)

    # context → risk (if found) or END (not found)
    graph.add_conditional_edges("context", route_after_context)

    # risk → diagnosis_generate (always)
    graph.add_edge("risk", "diagnosis_generate")

    # diagnosis_generate → END (wait for user to answer)
    graph.add_edge("diagnosis_generate", END)

    # collect_answer → solution (if all answered) or END (wait for next answer)
    graph.add_conditional_edges("collect_answer", route_after_collect)

    # solution → END
    graph.add_edge("solution", END)

    # conversation → END
    graph.add_edge("conversation", END)

    # escalation → END
    graph.add_edge("escalation", END)

    return graph.compile()


# Build once at import
workflow = build_graph()
