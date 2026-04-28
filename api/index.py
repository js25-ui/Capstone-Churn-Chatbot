"""
Conversational AI Churn Mitigation — 4-Agent Pipeline

Agent 1: Context Extraction (LLM) — sentiment, complaint type, risk signals
Agent 2: Churn Prediction (ML/XGBoost) — churn probability, risk tier, SHAP drivers
Agent 3: Strategy Engine (LLM) — every response is generated from full context
Agent 4: Simulation Agent — counterfactual interventions + A/B validation + contextual re-ranking

NO scripted questions. NO fixed question lists. NO separate "diagnosis mode."
Every turn: extract context → decide intent → generate a human response.
"""

import os
import sys
import json
import random
import pickle
from flask import Flask, request, jsonify

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

_candidates = [
    os.path.join(_root, "customer_data.json"),
    os.path.join(os.getcwd(), "customer_data.json"),
]
_data_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
with open(_data_path) as f:
    CUSTOMERS = json.load(f)
CUSTOMER_LIST = list(CUSTOMERS.values())

# Load model artifacts for Simulation Agent (needs live model for counterfactuals)
_pkl_candidates = [
    os.path.join(_root, "churn_artifacts.pkl"),
    os.path.join(os.getcwd(), "churn_artifacts.pkl"),
]
_pkl_path = next((p for p in _pkl_candidates if os.path.exists(p)), None)

SIM_MODEL = None
SIM_FEATURE_NAMES = None
SIM_X_TEST = None
SIM_AB_CACHE = None  # cached A/B results (computed once)

if _pkl_path:
    try:
        with open(_pkl_path, "rb") as f:
            _artifacts = pickle.load(f)
        SIM_MODEL = _artifacts["model"]
        SIM_FEATURE_NAMES = _artifacts["feature_names"]
        # Reconstruct X_test from df_clean for A/B validation
        from preprocessing.pipeline import preprocess_customer, add_engineered_features
        _df = _artifacts["df_clean"]
        if "Total Services" not in _df.columns:
            _df = add_engineered_features(_df)
        # Use all customers as the test population for A/B
        import pandas as pd
        _rows = []
        for _, row in _df.iterrows():
            try:
                _rows.append(preprocess_customer(row, SIM_FEATURE_NAMES).iloc[0])
            except Exception:
                pass
        if _rows:
            SIM_X_TEST = pd.DataFrame(_rows)
    except Exception:
        pass

app = Flask(__name__)

# ===================================================================
# AGENT 1: CONTEXT EXTRACTION
# ===================================================================
_SENT_1 = {"hate", "despise", "worst", "terrible", "horrible", "awful",
           "disgusting", "furious", "livid", "scam", "fraud", "robbed",
           "stealing", "ripped off", "garbage", "trash", "useless"}
_SENT_2 = {"angry", "frustrated", "annoyed", "upset", "disappointing",
           "unacceptable", "ridiculous", "pathetic", "fed up", "sick of",
           "tired of", "done with", "had enough", "can't stand", "pissed",
           "way too high", "way too much", "way too expensive",
           "ripping me off", "complete waste", "totally unacceptable"}
_SENT_5 = {"love", "amazing", "perfect", "excellent", "fantastic",
           "wonderful", "incredible", "outstanding", "best", "thrilled"}
_SENT_4 = {"good", "great", "nice", "happy", "pleased", "satisfied",
           "fine", "no complaints", "thankful", "appreciate"}

_COMPLAINT_MAP = {
    "billing": ["afford", "expensive", "cost", "price", "bill", "too much",
                "money", "pay", "charge", "overcharg", "fee", "rate",
                "too high", "overpriced"],
    "service_quality": ["slow", "drop", "dropping", "outage", "disconnect",
                        "speed", "buffer", "lag", "down", "unreliable",
                        "spotty", "intermittent", "cuts out", "wifi"],
    "value": ["not worth", "waste", "rip", "better deal", "cheaper",
              "competitor", "switch", "other provider"],
    "support": ["wait", "hold", "rude", "unhelpful", "no one", "nobody",
                "ignored", "runaround", "transferred"],
    "general_dissatisfaction": ["problem", "issue", "bad", "suck", "poor",
                                "terrible", "hate", "unhappy", "miserable"],
}

_RISK_SIGNALS = {
    "competitor_mention": ["other provider", "competitor", "switch to",
                           "looking at", "better deal", "tmobile", "t-mobile",
                           "verizon", "at&t", "att", "xfinity", "comcast"],
    "cancel_intent": ["cancel", "leave", "leaving", "quit", "done", "end my",
                       "terminate", "closing", "get rid of", "thinking of leaving",
                       "going to leave", "want to leave", "switching"],
    "urgency": ["today", "right now", "immediately", "asap", "last chance",
                "final", "running out of patience"],
    "emotional_escalation": ["hate", "worst", "furious", "livid",
                              "unacceptable", "lawyer", "attorney",
                              "complaint", "sue", "bbb"],
}

HARD_ESCALATION_PHRASES = {
    "cancel my account", "cancel my service", "cancel my plan",
    "cancel everything", "i want to cancel",
    "speak to a manager", "talk to a manager", "get me a manager",
    "speak to a human", "talk to a human",
    "speak to a supervisor", "talk to a supervisor",
    "i want a manager", "get me a supervisor",
    "lawyer", "attorney", "legal action",
    "file a complaint", "formal complaint",
}


def agent1_extract(msg, history):
    """Extract sentiment, complaint type, risk signals from a message."""
    try:
        convo = "\n".join(
            f"Customer: {t['customer']}" + (f"\nAgent: {t['agent']}" if t.get("agent") else "")
            for t in history[-5:]
        )
        raw = _call_claude(
            "You are a context extraction system. Analyze the customer message and return "
            "a JSON object: {\"sentiment\": int 1-5, \"complaint_type\": str, "
            "\"risk_signals\": [str], \"summary\": str}.\n"
            "sentiment: 1=hostile/hateful, 2=angry, 3=unhappy, 4=neutral, 5=positive\n"
            "complaint_type: billing|service_quality|value|support|general_dissatisfaction|none\n"
            "risk_signals: competitor_mention|cancel_intent|urgency|emotional_escalation (or empty)\n"
            "Return ONLY valid JSON.",
            f"History:\n{convo}\n\nMessage: \"{msg}\"",
            max_tokens=200,
        )
        clean = raw.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        d = json.loads(clean)
        return {
            "sentiment": max(1, min(5, int(d.get("sentiment", 3)))),
            "complaint_type": d.get("complaint_type", "none"),
            "risk_signals": d.get("risk_signals", []),
            "summary": d.get("summary", ""),
            "source": "llm",
        }
    except Exception:
        return _extract_keywords(msg)


def _extract_keywords(msg):
    low = msg.lower()
    if any(w in low for w in _SENT_1):
        s = 1
    elif any(w in low for w in _SENT_2):
        s = 2
    elif any(w in low for w in _SENT_5):
        s = 5
    elif any(w in low for w in _SENT_4):
        s = 4
    elif any(w in low for w in ["problem", "issue", "concern", "expensive", "slow", "bad",
                                 "suck", "drop", "dropping", "disconnect", "outage", "afford",
                                 "poor", "too high", "too much", "overpriced", "not worth"]):
        s = 3
    else:
        s = 4
    ct = "none"
    for ctype, kws in _COMPLAINT_MAP.items():
        if any(kw in low for kw in kws):
            ct = ctype
            break
    rs = [sig for sig, kws in _RISK_SIGNALS.items() if any(kw in low for kw in kws)]
    return {"sentiment": s, "complaint_type": ct, "risk_signals": rs,
            "summary": msg[:80], "source": "keywords"}


# ===================================================================
# AGENT 3: STRATEGY ENGINE — one function, generates EVERY response
# ===================================================================

def agent3_respond(state, user_msg):
    """
    Generate the bot's next response based on the FULL conversation context.
    No modes, no phases, no scripted questions.
    The LLM decides what to say based on everything it knows.
    Returns: {response: str, action: str, offer_made: bool}
    """
    profile = state.get("customer_profile", {})
    risk_data = state.get("risk_data", {})
    ctx = state.get("last_context", {})
    name = state.get("customer_name", "there")
    name_s = name if name != "there" else ""
    history = state.get("conversation_history", [])
    turns = len(history)
    sentiment = ctx.get("sentiment", 4)
    complaint = ctx.get("complaint_type", "none")
    signals = ctx.get("risk_signals", [])
    offers = state.get("offers_made", [])
    baseline = _retention_strategy(profile, risk_data.get("risk_tier", "MEDIUM"))

    convo = _build_convo(history)
    drivers = ", ".join(f"{d['feature']}({d['direction']})" for d in risk_data.get("top_drivers", [])[:5])

    # Decide: should we make an offer this turn?
    should_offer = False
    offer_reason = ""
    if offers:
        # Already made an offer — this is follow-up
        should_offer = False
    elif sentiment <= 1 and turns >= 1:
        should_offer = True
        offer_reason = f"Very low sentiment ({sentiment}/5) — immediate offer"
    elif sentiment <= 2 and turns >= 2:
        should_offer = True
        offer_reason = f"Low sentiment ({sentiment}/5) after {turns} turns"
    elif sentiment <= 3 and turns >= 3:
        should_offer = True
        offer_reason = f"Sustained unhappiness ({sentiment}/5) over {turns} turns"
    elif turns >= 4:
        should_offer = True
        offer_reason = f"Sufficient context ({turns} turns)"
    elif "cancel_intent" in signals and turns >= 1:
        should_offer = True
        offer_reason = "Cancel intent detected"

    # Intervention hint (set by chat() before calling agent3_respond)
    intervention_hint = state.get("_intervention_hint", "")

    # --- Try Claude ---
    try:
        if offers:
            # Post-offer conversation
            response = _call_claude(
                f"You are a customer service agent talking to {name_s or 'a customer'}. "
                f"You already made an offer. Respond to exactly what they just said.\n"
                f"If they accepted → confirm enthusiastically, explain next steps.\n"
                f"If they rejected → acknowledge, ask what would work better. "
                f"If they rejected twice already, offer to connect with a specialist.\n"
                f"If they're asking a question → answer it.\n"
                f"Do NOT repeat previous offers. Do NOT mention internal systems.\n"
                f"Under 60 words.",
                f"Account: {profile.get('contract')}, ${profile.get('monthly_charge',0):.2f}/mo\n"
                f"Offers made so far: {len(offers)}\n"
                f"Last offer: {offers[-1][:150] if offers else 'none'}\n"
                f"Conversation:\n{convo}\n\nCustomer: \"{user_msg}\"",
                max_tokens=150,
            )
            # Check if they rejected
            is_reject = any(w in user_msg.lower() for w in
                           ["no", "not enough", "nah", "pass", "nope", "won't work",
                            "not interested", "don't want", "too little", "not good"])
            if is_reject and len(offers) < 2:
                return {"response": response, "action": "Offer rejected — follow-up",
                        "offer_made": False, "is_rejection": True}
            return {"response": response, "action": "Post-offer conversation", "offer_made": False}

        elif should_offer:
            response = _call_claude(
                f"You are a customer retention specialist talking to {name_s or 'a customer'}.\n"
                f"Based on the conversation, make a personalized retention offer NOW.\n\n"
                f"RULES:\n"
                f"- First, briefly acknowledge what they just said (1 sentence)\n"
                f"- Then present a specific offer with dollar amounts or percentages\n"
                f"- Reference their ACTUAL complaints from the conversation\n"
                f"- If they complained about price → lead with discount\n"
                f"- If service quality → lead with service fix + credit\n"
                f"- If general unhappiness → lead with biggest available discount\n"
                f"- Do NOT mention churn, SHAP, risk, models, or analytics\n"
                f"- Under 80 words. End with 'Would you like me to apply this?'\n",
                f"Account: {profile.get('tenure_months')}mo, {profile.get('contract')}, "
                f"${profile.get('monthly_charge',0):.2f}/mo, {profile.get('internet_type','N/A')} internet\n"
                f"Risk: {risk_data.get('risk_tier')} ({risk_data.get('churn_probability',0):.1%})\n"
                f"Complaint: {complaint}, Sentiment: {sentiment}/5\n"
                f"Baseline: {baseline}\n"
                f"{intervention_hint}\n"
                f"Conversation:\n{convo}\n\nCustomer: \"{user_msg}\"",
            )
            return {"response": response, "action": f"Retention offer ({offer_reason})",
                    "offer_made": True}

        else:
            # Still learning — have a real conversation
            response = _call_claude(
                f"You are a customer service agent having a REAL conversation with {name_s or 'a customer'}.\n\n"
                f"The customer just said something. You MUST respond to EXACTLY what they said.\n\n"
                f"RULES:\n"
                f"- Read their message carefully. Respond to the SPECIFIC content.\n"
                f"- If they said 'i hate this' → say you're sorry, ask what specifically went wrong\n"
                f"- If they said 'too expensive' → acknowledge the cost concern, ask what budget works\n"
                f"- If they said 'internet drops' → empathize, ask how often and when\n"
                f"- If they said 'yes' to confirm account → ask how things have been going\n"
                f"- NEVER ask a generic question that ignores what they just said\n"
                f"- NEVER repeat a question you already asked\n"
                f"- You are trying to understand their situation so you can help\n"
                f"- 2-3 sentences max. Be warm, specific, human.\n"
                f"- Do NOT mention SHAP, churn, risk, models\n",
                f"HIDDEN (do not reveal): sentiment={sentiment}/5, complaint={complaint}, "
                f"signals={signals}, drivers={drivers}\n"
                f"Account: {profile.get('tenure_months')}mo, {profile.get('contract')}, "
                f"${profile.get('monthly_charge',0):.2f}/mo, {profile.get('internet_type')}\n\n"
                f"FULL CONVERSATION:\n{convo}\n\nCustomer just said: \"{user_msg}\"",
                max_tokens=200,
            )
            return {"response": response, "action": f"Conversation (turn {turns+1}, sentiment {sentiment}/5)",
                    "offer_made": False}

    except Exception:
        # --- Keyword fallback ---
        return _respond_fallback(state, user_msg, should_offer, sentiment, complaint, offers)


def _respond_fallback(state, msg, should_offer, sentiment, complaint, offers):
    """Generate response without Claude — keyword-based but context-aware."""
    profile = state.get("customer_profile", {})
    risk_data = state.get("risk_data", {})
    name = state.get("customer_name", "there")
    name_s = name if name != "there" else ""
    addr = f", {name_s}" if name_s else ""
    low = msg.lower()

    # Post-offer
    if offers:
        if any(w in low for w in ["yes", "sure", "sounds good", "deal", "accept", "go ahead", "ok", "okay", "great"]):
            return {"response": f"Wonderful{addr}! I've applied the changes to your account. "
                    f"You'll see the updated rate on your next bill. Anything else I can help with?",
                    "action": "Offer accepted", "offer_made": False}
        if any(w in low for w in ["no", "not enough", "nah", "pass", "nope", "won't work"]):
            return {"response": f"I understand{addr}. What would work better for you? "
                    f"I want to find something that actually fits.",
                    "action": "Offer rejected", "offer_made": False, "is_rejection": True}
        return {"response": f"Thanks for letting me know{addr}. "
                f"Is there anything specific I can help clarify about the offer?",
                "action": "Post-offer followup", "offer_made": False}

    # Make an offer
    if should_offer:
        discount = 0.80 if risk_data.get("risk_tier") == "HIGH" else 0.85
        new_price = profile.get("monthly_charge", 0) * discount
        pct = int((1 - discount) * 100)

        if sentiment <= 1:
            resp = (f"I hear you{addr}, and I'm truly sorry. Let me make this right: "
                    f"**{pct}% off for 12 months** → **${new_price:.2f}/month**, "
                    f"plus a **${profile.get('monthly_charge',0)*0.5:.0f} credit**. "
                    f"Would you like me to apply this?")
        elif complaint == "billing":
            resp = (f"I understand the cost concern{addr}. Here's what I can do: "
                    f"**{pct}% off** for 6 months → **${new_price:.2f}/month**. "
                    f"That saves **${profile.get('monthly_charge',0)-new_price:.2f}/month**. "
                    f"Would you like me to apply this?")
        elif complaint == "service_quality":
            resp = (f"The service issues are not okay{addr}. I'll flag your line for a "
                    f"**priority technical review** and give you **{pct}% off** for 3 months. "
                    f"Would you like me to apply this?")
        else:
            resp = (f"Based on what you've told me{addr}, I'd like to offer "
                    f"**{pct}% off** for 6 months → **${new_price:.2f}/month**. "
                    f"Would you like me to apply this?")
        return {"response": resp, "action": f"Retention offer ({complaint})", "offer_made": True}

    # Conversational — react to what they said
    if sentiment <= 1:
        resp = (f"I'm really sorry you feel that way{addr}. That's not okay, "
                f"and I take this seriously. Can you tell me what specifically has gone wrong?")
    elif sentiment <= 2:
        resp = (f"I can hear how frustrated you are{addr}. I want to fix this. "
                f"What's been the biggest problem?")
    elif "afford" in low or "expensive" in low or "price" in low or "cost" in low or "bill" in low or "too much" in low or "too high" in low:
        resp = (f"I understand{addr} — nobody should feel like they're overpaying. "
                f"Is it the monthly rate that's the issue, or are there extra charges adding up?")
    elif "slow" in low or "drop" in low or "disconnect" in low or "speed" in low or "internet" in low or "wifi" in low:
        resp = (f"That's really frustrating{addr} — you're paying for a service that should work. "
                f"How often does this happen? Is it all the time or at certain times of day?")
    elif "problem" in low or "issue" in low or "bad" in low or "suck" in low:
        resp = (f"I'm sorry to hear that{addr}. Can you tell me more about "
                f"what's been going on? I want to understand the specifics so I can actually help.")
    elif any(w in low for w in ["yes", "yeah", "correct", "right", "yep", "that's right", "looks right"]):
        resp = (f"Great{addr}! So tell me — how have things been going with your service? "
                f"Anything on your mind?")
    else:
        resp = (f"Thanks for sharing that{addr}. Can you tell me a bit more? "
                f"I want to make sure I understand your situation so I can actually do something about it.")

    return {"response": resp, "action": f"Conversation (sentiment {sentiment}/5)", "offer_made": False}


# ===================================================================
# Helpers
# ===================================================================
def _call_claude(system, user_msg, max_tokens=400):
    """Call Claude API using raw requests to avoid httpx connection issues on Vercel."""
    import requests as req
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    resp = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-4-sonnet-20250514",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _retention_strategy(profile, tier):
    """Exact retention_strategy from notebook Cell 82."""
    if tier == "HIGH":
        contract = profile.get("contract", "Unknown")
        charge = profile.get("monthly_charge", 0)
        if contract == "Month-to-Month":
            return "Offer contract upgrade incentive and immediate retention discount."
        elif charge > 80:
            return "Offer pricing review or loyalty discount."
        else:
            return "Proactive outreach with personalized retention offer."
    elif tier == "MEDIUM":
        contract = profile.get("contract", "Unknown")
        if contract == "Month-to-Month":
            return "Send targeted email with contract lock-in discount."
        else:
            return "Offer complimentary service upgrade for 3 months."
    else:
        return "Continue standard engagement. Monitor for changes."


def _build_convo(history):
    if not history:
        return "(First message)"
    return "\n".join(
        f"Customer: {t['customer']}" + (f"\nAgent: {t['agent']}" if t.get("agent") else "")
        for t in history
    )


def _friendly_tenure(months):
    if months < 12:
        return f"{months} months"
    y, r = divmod(months, 12)
    if r == 0:
        return f"{y} year{'s' if y > 1 else ''}"
    return f"{y} year{'s' if y > 1 else ''} and {r} month{'s' if r > 1 else ''}"


def _load_customer(state, messages, log, customer, first_name):
    """Load customer profile, run Phase 1 simulations, show account summary."""
    pr = customer["profile"]
    rd = customer["risk"]
    state.update({
        "customer_name": first_name,
        "customer_id": pr["customer_id"],
        "customer_profile": pr,
        "risk_data": rd,
        "phase": "conversation",
    })
    log.append({"agent": "Agent 2: Churn Prediction", "output": {
        "churn_probability": rd["churn_probability"],
        "risk_tier": rd["risk_tier"],
        "top_driver": rd["top_drivers"][0]["feature"] if rd["top_drivers"] else "N/A",
    }})

    # --- AGENT 4: Simulation Phase 1 (pre-computed in customer_data.json) ---
    sim_phase1 = customer.get("simulations", [])
    if sim_phase1:
        log.append({"agent": "Agent 4: Simulation", "output": {
            "phase": 1,
            "interventions": len(sim_phase1),
            "top": sim_phase1[0]["name"] if sim_phase1 else "N/A",
            "top_reduction": sim_phase1[0].get("reduction_pct", 0) if sim_phase1 else 0,
        }})

    state["simulation_phase1"] = sim_phase1
    state["simulation_phase2"] = None
    state["best_intervention"] = sim_phase1[0]["name"] if sim_phase1 else None

    t = _friendly_tenure(pr["tenure_months"])
    if first_name and first_name != "there":
        intro = f"Great to meet you, {first_name}! I've pulled up your account."
    else:
        intro = "Thanks! I've pulled up your account."
    messages.append({"role": "assistant", "content": (
        f"{intro}\n\n"
        f"I can see you've been with us for **{t}** "
        f"on a **{pr['contract']}** plan at **${pr['monthly_charge']:.2f}/month** "
        f"with **{pr['internet_type']}** internet"
        f"{' and ' + str(pr['total_services']) + ' services' if pr['total_services'] > 1 else ''}.\n\n"
        f"How can I help you today?"
    )})
    state["pipeline_log"] = log
    return jsonify({"state": state, "messages": messages})


# ===================================================================
# ROUTES
# ===================================================================

@app.route("/api/start", methods=["POST"])
def start():
    state = {
        "customer_id": None, "customer_name": None,
        "customer_profile": None, "risk_data": None,
        "last_context": None, "conversation_history": [],
        "offers_made": [], "solutions_attempted": 0,
        "sentiment_history": [], "escalated": False,
        "escalation_summary": None, "phase": "greeting",
        "simulation_phase1": [], "simulation_phase2": None,
        "best_intervention": None,
        "pipeline_log": [],
    }
    return jsonify({"state": state, "messages": [{"role": "assistant", "content": (
        "Hi there! Welcome to TelcoCare Support. \n\n"
        "I'm here to make sure you're getting the most out of your plan. "
        "Could I get your **name** so I can pull up your account?"
    )}]})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    state = data.get("state", {})
    user_msg = data.get("message", "").strip()
    messages = []
    phase = state.get("phase", "greeting")
    log = list(state.get("pipeline_log", []))

    # === GREETING — have a real conversation first ===
    if phase == "greeting":
        raw = user_msg.strip()
        low = raw.lower()

        # Is it a Customer ID?
        test_id = raw.upper()
        if test_id in CUSTOMERS:
            return _load_customer(state, messages, log, CUSTOMERS[test_id], "there")

        # Is it just a greeting (not a name)?
        greetings = {"hi", "hello", "hey", "yo", "sup", "what's up", "whats up",
                     "good morning", "good afternoon", "good evening", "howdy",
                     "hola", "greetings", "hi there", "hey there", "hello there"}
        is_greeting = low.rstrip("!., ") in greetings or low in {"h", "hii", "hiii", "heyyy", "heyy"}

        if is_greeting:
            # Respond naturally — don't treat it as a name
            messages.append({"role": "assistant", "content": (
                "Hey! Thanks for reaching out. I'd love to help you today.\n\n"
                "Could I get your **name** so I can look up your account?"
            )})
            # Stay in greeting phase — wait for actual name
            state["pipeline_log"] = log
            return jsonify({"state": state, "messages": messages})

        # Is it a question or statement rather than a name?
        non_name_signals = ["i have", "i need", "i want", "my account", "my bill",
                           "help me", "can you", "i'm having", "problem", "issue",
                           "question", "why", "how", "what", "when", "?"]
        is_not_name = any(s in low for s in non_name_signals)

        if is_not_name:
            # They're jumping straight to their issue — that's fine, acknowledge it
            # but we still need a name to "look up" their account
            messages.append({"role": "assistant", "content": (
                "I hear you, and I definitely want to help with that. "
                "Let me just get your **name** first so I can pull up your account, "
                "and then we'll get right into it."
            )})
            state["pipeline_log"] = log
            return jsonify({"state": state, "messages": messages})

        # It's probably a name — extract it
        nm = raw.title()
        for p in ["my name is ", "i'm ", "im ", "i am ", "it's ", "its ",
                   "the name is ", "call me ", "name's ", "names "]:
            if nm.lower().startswith(p):
                nm = nm[len(p):].strip().title()
                break
        first_name = nm.split()[0] if nm.split() else "there"

        # Sanity check — if the "name" is very long or has numbers, it's not a name
        if len(first_name) > 20 or any(c.isdigit() for c in first_name):
            messages.append({"role": "assistant", "content": (
                "I didn't quite catch that — could you tell me your **first name**?"
            )})
            state["pipeline_log"] = log
            return jsonify({"state": state, "messages": messages})

        # Name given but no matching account — we don't have names in the dataset
        state["customer_name"] = first_name
        state["phase"] = "not_found"
        sample_ids = random.sample(list(CUSTOMERS.keys()), 3)
        messages.append({"role": "assistant", "content": (
            f"Thanks, {first_name}! I searched our system but wasn't able to find "
            f"an account under that name.\n\n"
            f"Could you provide your **Customer ID** instead? It's usually in the format "
            f"`XXXX-XXXXX` — you can find it on your bill or in your account settings.\n\n"
            f"Here are some example IDs if you'd like to try a demo: "
            f"`{sample_ids[0]}`, `{sample_ids[1]}`, `{sample_ids[2]}`"
        )})
        state["pipeline_log"] = log
        return jsonify({"state": state, "messages": messages})

    # === NOT FOUND — waiting for a Customer ID ===
    if phase == "not_found":
        test_id = user_msg.strip().upper()
        if test_id in CUSTOMERS:
            name = state.get("customer_name", "there")
            return _load_customer(state, messages, log, CUSTOMERS[test_id], name)
        else:
            sample_ids = random.sample(list(CUSTOMERS.keys()), 3)
            messages.append({"role": "assistant", "content": (
                f"I couldn't find an account with `{user_msg.strip()}`. "
                f"Please double-check the Customer ID — it looks like `XXXX-XXXXX`.\n\n"
                f"You can try one of these: `{sample_ids[0]}`, `{sample_ids[1]}`, `{sample_ids[2]}`"
            )})
            state["pipeline_log"] = log
            return jsonify({"state": state, "messages": messages})

    # === EVERY OTHER MESSAGE: run the 3-agent pipeline ===

    # AGENT 1: Extract context
    ctx = agent1_extract(user_msg, state.get("conversation_history", []))
    state["last_context"] = ctx
    state["sentiment_history"] = state.get("sentiment_history", []) + [ctx["sentiment"]]
    log.append({"agent": "Agent 1: Context Extraction", "output": {
        "sentiment": ctx["sentiment"], "complaint_type": ctx["complaint_type"],
        "risk_signals": ctx["risk_signals"], "source": ctx["source"]}})

    # Check hard escalation
    low = user_msg.lower().strip()
    if not state.get("escalated"):
        for phrase in HARD_ESCALATION_PHRASES:
            if phrase in low:
                state["escalated"] = True
                state["phase"] = "escalated"
                name = state.get("customer_name", "there")
                addr = f", {name}" if name != "there" else ""
                log.append({"agent": "Orchestrator", "output": {"decision": "ESCALATE"}})
                state["pipeline_log"] = log
                messages.append({"role": "assistant", "content": (
                    f"I completely understand{addr}. I'm connecting you with a senior specialist "
                    f"who can do more to help. They'll have full context from our conversation."
                )})
                return jsonify({"state": state, "messages": messages})

    # Record turn
    state["conversation_history"] = state.get("conversation_history", []) + [
        {"customer": user_msg, "agent": None}
    ]

    # AGENT 4: Simulation Phase 2 (contextual re-ranking before offer)
    # Run Phase 2 if we have Phase 1 results and enough conversation context
    if (state.get("simulation_phase1") and not state.get("simulation_phase2")
            and len(state.get("conversation_history", [])) >= 2):
        try:
            from agents.simulation_agent import run_contextual_reranking
            customer_msgs = [t["customer"] for t in state.get("conversation_history", []) if t.get("customer")]
            phase2 = run_contextual_reranking(
                state["simulation_phase1"], customer_msgs, _call_claude
            )
            state["simulation_phase2"] = phase2
            state["best_intervention"] = phase2[0]["name"] if phase2 else state.get("best_intervention")
            log.append({"agent": "Agent 4: Simulation", "output": {
                "phase": 2,
                "best": phase2[0]["name"] if phase2 else "N/A",
                "combined_score": phase2[0].get("combined_score", 0) if phase2 else 0,
            }})
        except Exception as e:
            log.append({"agent": "Agent 4: Simulation", "output": {"phase": 2, "error": str(e)[:100]}})

    # Build intervention hint for Agent 3
    best_int = state.get("best_intervention", "")
    hint = ""
    if best_int:
        sim_data = state.get("simulation_phase2") or state.get("simulation_phase1") or []
        for s in sim_data:
            if s["name"] == best_int and s.get("eligible"):
                hint = (
                    f"\nRECOMMENDED INTERVENTION (from simulation): {s['name']} — {s['description']}. "
                    f"Projected churn reduction: {s.get('reduction_pct', 0):.1f}%. "
                    f"Build your offer around this intervention."
                )
                break
    state["_intervention_hint"] = hint

    # AGENT 3: Generate response (it decides whether to offer or keep talking)
    result = agent3_respond(state, user_msg)
    response = result["response"]
    state["conversation_history"][-1]["agent"] = response

    if result.get("offer_made"):
        state["offers_made"] = state.get("offers_made", []) + [response]
        state["solutions_attempted"] = state.get("solutions_attempted", 0) + 1

    # Handle rejection → use NEXT-BEST intervention from Simulation Agent
    if result.get("is_rejection") and state.get("solutions_attempted", 0) < 2:
        state["solutions_attempted"] = state.get("solutions_attempted", 0) + 1
        pr = state.get("customer_profile", {})
        name_r = state.get("customer_name", "there")
        addr_r = f", {name_r}" if name_r != "there" else ""

        # Find the next eligible intervention (skip the one already offered)
        sim_data = state.get("simulation_phase2") or state.get("simulation_phase1") or []
        first_offered = state.get("best_intervention", "")
        next_intervention = None
        for s in sim_data:
            if s.get("eligible") and s["name"] != first_offered:
                next_intervention = s
                break

        if next_intervention:
            # Build offer from the next-best simulation intervention
            int_name = next_intervention["name"]
            int_desc = next_intervention["description"]
            int_reduction = next_intervention.get("reduction_pct", 0)
            state["best_intervention"] = int_name

            try:
                convo_r = _build_convo(state.get("conversation_history", []))
                response = _call_claude(
                    f"You are a customer service agent. The customer rejected your first offer. "
                    f"Now offer a DIFFERENT solution based on this specific intervention:\n"
                    f"Intervention: {int_name} — {int_desc}\n"
                    f"Projected impact: {int_reduction:.1f}% churn reduction\n\n"
                    f"RULES:\n"
                    f"- Acknowledge they didn't like the first offer\n"
                    f"- Present this intervention as a concrete offer\n"
                    f"- For 'Autopay Migration': offer to switch payment method + small credit\n"
                    f"- For 'Online Security': offer free online security add-on\n"
                    f"- For 'Service Bundling': offer discounted bundle deal\n"
                    f"- For 'Referral Program': offer referral bonus + credit for each referral\n"
                    f"- For 'Tenure Survival': offer loyalty lock-in with guaranteed rate\n"
                    f"- Do NOT mention churn, SHAP, simulations, or analytics\n"
                    f"- Under 80 words\n",
                    f"Customer: {pr.get('contract')}, ${pr.get('monthly_charge',0):.2f}/mo\n"
                    f"Rejected offer: {state.get('offers_made', [''])[- 1][:150]}\n"
                    f"Customer said: \"{user_msg}\"\n"
                    f"Conversation:\n{convo_r}",
                )
            except Exception:
                # Keyword fallback for each intervention type
                int_lower = int_name.lower()
                if "autopay" in int_lower:
                    response = (
                        f"I hear you{addr_r}. How about this instead — if you switch to "
                        f"**credit card autopay**, I can give you a **$10/month discount** "
                        f"plus a **$20 account credit** right now. No contract change needed. "
                        f"Would you like me to set that up?"
                    )
                elif "security" in int_lower:
                    response = (
                        f"Understood{addr_r}. Let me try something different — I can add "
                        f"**Online Security** to your account **free for 6 months** "
                        f"(normally $10/month). It protects your devices and data. "
                        f"Would you like me to add that?"
                    )
                elif "bundling" in int_lower or "service" in int_lower:
                    response = (
                        f"No problem{addr_r}. What if we bundled in an extra service? "
                        f"I can add **2 additional services** at a **discounted bundle rate** "
                        f"— that usually saves customers 15-20% overall. Interested?"
                    )
                elif "referral" in int_lower:
                    response = (
                        f"Fair enough{addr_r}. Here's another option — join our "
                        f"**Referral Rewards program** and earn a **$25 credit** for each "
                        f"friend you refer. Plus I'll add a **$15 credit** to your account "
                        f"right now just for signing up. Sound good?"
                    )
                elif "tenure" in int_lower:
                    response = (
                        f"I understand{addr_r}. How about a **rate lock guarantee**? "
                        f"I'll freeze your current rate at **${pr.get('monthly_charge',0):.2f}/month** "
                        f"for 24 months — no increases, guaranteed. Would that work?"
                    )
                else:
                    response = (
                        f"I understand{addr_r}. Let me offer something completely different — "
                        f"a **$25 account credit** plus a **free service upgrade** for 3 months. "
                        f"Would you like me to apply that?"
                    )

            log.append({"agent": "Agent 4: Simulation", "output": {
                "action": f"Rejected '{first_offered}' → offering '{int_name}' "
                          f"({int_reduction:.1f}% reduction)"}})
        else:
            # No more eligible interventions — generic fallback
            response = (
                f"I understand{addr_r}. Let me check with my team to see what "
                f"other options we have available. Is there a specific type of "
                f"offer that would interest you — a discount, a service upgrade, or something else?"
            )

        state["offers_made"] = state.get("offers_made", []) + [response]
        state["conversation_history"][-1]["agent"] = response
        result = {"response": response, "action": f"Second offer: {next_intervention['name'] if next_intervention else 'generic'}",
                  "offer_made": True}

    log.append({"agent": "Agent 3: Strategy Engine", "output": {
        "action": result.get("action", ""), "offer_made": result.get("offer_made", False)}})

    state["phase"] = "conversation"
    state["pipeline_log"] = log
    messages.append({"role": "assistant", "content": response})
    return jsonify({"state": state, "messages": messages})


@app.route("/api/health", methods=["GET"])
def health():
    # Test Claude connectivity
    claude_status = "unknown"
    claude_error = None
    try:
        result = _call_claude("Reply with just 'ok'.", "test", max_tokens=5)
        claude_status = f"ok: {result}"
    except Exception as e:
        claude_status = "error"
        claude_error = f"{type(e).__name__}: {str(e)[:200]}"
    return jsonify({
        "status": "ok",
        "customers": len(CUSTOMERS),
        "claude": claude_status,
        "claude_error": claude_error,
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    })
