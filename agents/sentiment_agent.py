"""Sentiment Agent — scores customer messages 1-5, flags escalation triggers."""

import anthropic
from state import ChurnState

client = anthropic.Anthropic()

ESCALATION_KEYWORDS = {"manager", "cancel", "unacceptable", "lawyer", "complaint", "sue", "attorney", "supervisor"}


def sentiment_agent(state: ChurnState, user_message: str) -> dict:
    """Score sentiment 1-5 and check for escalation triggers."""
    # Keyword check
    msg_lower = user_message.lower()
    keyword_triggered = any(kw in msg_lower for kw in ESCALATION_KEYWORDS)

    # LLM sentiment scoring
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": f"""Rate the sentiment of this customer message on a scale of 1-5:
1 = Very negative/angry
2 = Negative/frustrated
3 = Neutral
4 = Positive
5 = Very positive/satisfied

Message: "{user_message}"

Reply with ONLY a single number 1-5.""",
            }
        ],
    )

    try:
        score = int(response.content[0].text.strip()[0])
        score = max(1, min(5, score))
    except (ValueError, IndexError):
        score = 3

    history = state.get("sentiment_history", []) + [score]
    should_escalate = keyword_triggered or score < 3

    result = {
        "sentiment_history": history,
    }

    if should_escalate and not state.get("escalated", False):
        result["escalated"] = True
        result["phase"] = "escalated"

    return result
