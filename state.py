"""Shared state definition for the LangGraph orchestrator."""

from typing import TypedDict, Optional
from langgraph.graph import MessagesState


class ChurnState(MessagesState):
    customer_id: Optional[str]
    customer_profile: Optional[dict]
    risk_data: Optional[dict]
    diagnosis_questions: Optional[list[str]]
    diagnosis_answers: Optional[list[str]]
    diagnosis_step: int  # which question we're on
    offers_made: list[str]
    sentiment_history: list[int]
    escalated: bool
    escalation_summary: Optional[str]
    phase: str  # "greeting", "identify", "risk", "diagnosis", "solution", "conversation", "escalated"
