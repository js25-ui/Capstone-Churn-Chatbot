# Customer Churn Retention Chatbot

Multi-agent chatbot for customer churn prediction and personalized retention, built on the Fordham AI Capstone XGBoost model.

## Architecture

7 specialized agents orchestrated by LangGraph:

1. **Context Agent** — Looks up customer profile from `df_clean`
2. **Risk Agent** — Runs XGBoost prediction + SHAP analysis, assigns risk tier (HIGH/MEDIUM/LOW)
3. **Diagnosis Agent** — Generates targeted questions based on top SHAP drivers (Claude API)
4. **Solution Agent** — Rule-based retention strategy + Claude-personalized offer
5. **Sentiment Agent** — Scores every message 1-5, flags escalation triggers
6. **Escalation Agent** — Builds handoff summary for human agents
7. **Orchestrator** — LangGraph StateGraph routing all agents

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
cp .env.example .env
# Edit .env and add your key: ANTHROPIC_API_KEY=sk-ant-...

# Run the chatbot
chainlit run app.py
```

Open http://localhost:8000 in your browser.

## Usage

1. The chatbot greets you with sample Customer IDs
2. Enter a Customer ID (e.g., `4526-ZJJTM`) or type `random`
3. View the customer profile and churn risk assessment
4. Answer 3 diagnostic questions
5. Receive a personalized retention offer
6. Continue chatting — the bot adapts to your responses
7. If sentiment drops or escalation keywords are used, it triggers human handoff

## Project Structure

```
agents/
  context_agent.py      # Customer lookup
  risk_agent.py         # XGBoost + SHAP
  diagnosis_agent.py    # Targeted questions (Claude)
  solution_agent.py     # Retention offers (rules + Claude)
  sentiment_agent.py    # Sentiment scoring (Claude)
  escalation_agent.py   # Human handoff summary
preprocessing/
  pipeline.py           # Feature engineering (matches notebook exactly)
orchestrator.py         # LangGraph graph definition
state.py                # TypedDict shared state
app.py                  # Chainlit frontend
churn_artifacts.pkl     # Trained model + SHAP explainer + data
```

## Tech Stack

- **LangGraph** — Agent orchestration
- **Anthropic Claude API** (claude-sonnet-4-5) — LLM calls
- **Chainlit** — Chat UI
- **XGBoost** — Churn prediction model
- **SHAP** — Feature importance analysis

## Team

Fordham AI Master's Capstone 2026
