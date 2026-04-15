"""Chainlit chat frontend for the Churn Retention Chatbot (customer-facing)."""

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

from orchestrator import workflow, get_random_customer_id, DF_CLEAN


def initial_state() -> dict:
    return {
        "messages": [],
        "customer_id": None,
        "customer_name": None,
        "customer_profile": None,
        "risk_data": None,
        "diagnosis_questions": None,
        "diagnosis_answers": None,
        "diagnosis_step": 0,
        "offers_made": [],
        "sentiment_history": [],
        "escalated": False,
        "escalation_summary": None,
        "phase": "greeting",
    }


@cl.on_chat_start
async def start():
    state = initial_state()
    state["phase"] = "greeting"
    cl.user_session.set("state", state)
    await cl.Message(
        content=(
            "Hi there! Welcome to TelcoCare Support.\n\n"
            "I'm here to make sure you're getting the most out of your plan. "
            "Could I get your **name** so I can pull up your account?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    state = cl.user_session.get("state")
    user_text = message.content.strip()
    state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=user_text)]

    # Route based on phase
    result = workflow.invoke(state, {"recursion_limit": 25})

    prev_count = len(state.get("messages", []))
    new_msgs = result.get("messages", [])[prev_count:]
    for msg in new_msgs:
        if isinstance(msg, AIMessage):
            await cl.Message(content=msg.content).send()

    cl.user_session.set("state", result)
