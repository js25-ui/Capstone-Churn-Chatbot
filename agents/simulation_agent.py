"""
Simulation Agent — Counterfactual intervention simulations + A/B validation.

Phase 1: Statistical ranking using exact notebook counterfactual patterns.
Phase 2: Contextual re-ranking using Claude API after diagnosis.

Code extracted directly from Capstone notebook Cell 95 (counterfactual simulations).
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ===================================================================
# INTERVENTION DEFINITIONS
# Each intervention: name, description, apply function, eligibility check
# ===================================================================

def _apply_contract_upgrade(row, feature_names):
    """Convert Month-to-Month → One Year contract."""
    simulated = row.copy()
    if "Contract_One Year" in feature_names:
        simulated["Contract_One Year"] = 1
    if "Contract_Two Year" in feature_names:
        simulated["Contract_Two Year"] = 0
    return simulated


def _eligible_contract_upgrade(row, feature_names):
    """Only MTM customers (neither One Year nor Two Year)."""
    is_one = row.get("Contract_One Year", 0) == 1
    is_two = row.get("Contract_Two Year", 0) == 1
    return not is_one and not is_two


def _apply_tenure_survival(row, feature_names):
    """Extend tenure past the critical 12-month threshold."""
    simulated = row.copy()
    if "Tenure in Months" in feature_names:
        simulated["Tenure in Months"] = 24
    return simulated


def _eligible_tenure_survival(row, feature_names):
    return row.get("Tenure in Months", 99) < 12


def _apply_autopay(row, feature_names):
    """Switch to credit card autopay."""
    simulated = row.copy()
    pay_cols = [c for c in feature_names if "Payment Method_" in c]
    for col in pay_cols:
        simulated[col] = 0
    if "Payment Method_Credit Card" in feature_names:
        simulated["Payment Method_Credit Card"] = 1
    return simulated


def _eligible_autopay(row, feature_names):
    return row.get("Payment Method_Credit Card", 0) != 1


def _apply_online_security(row, feature_names):
    """Add online security service."""
    simulated = row.copy()
    if "Online Security" in feature_names:
        simulated["Online Security"] = 1
    return simulated


def _eligible_online_security(row, feature_names):
    return row.get("Online Security", 1) != 1


def _apply_service_bundling(row, feature_names):
    """Increase services to at least 3."""
    simulated = row.copy()
    if "Total Services" in feature_names:
        simulated["Total Services"] = max(3, row.get("Total Services", 0))
    if "Charge Per Service" in feature_names and "Monthly Charge" in feature_names:
        simulated["Charge Per Service"] = simulated["Monthly Charge"] / (simulated["Total Services"] + 1)
    return simulated


def _eligible_service_bundling(row, feature_names):
    return row.get("Total Services", 99) <= 1


def _apply_referral_program(row, feature_names):
    """Boost referral count."""
    simulated = row.copy()
    if "Number of Referrals" in feature_names:
        simulated["Number of Referrals"] = 5
    if "Referred a Friend" in feature_names:
        simulated["Referred a Friend"] = 1
    return simulated


def _eligible_referral_program(row, feature_names):
    return row.get("Number of Referrals", 99) <= 1


INTERVENTIONS = [
    {
        "name": "Contract Upgrade",
        "description": "Convert Month-to-Month → One Year plan",
        "apply": _apply_contract_upgrade,
        "eligible": _eligible_contract_upgrade,
    },
    {
        "name": "Tenure Survival",
        "description": "Onboarding support to reach 24 months",
        "apply": _apply_tenure_survival,
        "eligible": _eligible_tenure_survival,
    },
    {
        "name": "Autopay Migration",
        "description": "Switch to credit card autopay",
        "apply": _apply_autopay,
        "eligible": _eligible_autopay,
    },
    {
        "name": "Online Security",
        "description": "Add online security service",
        "apply": _apply_online_security,
        "eligible": _eligible_online_security,
    },
    {
        "name": "Service Bundling",
        "description": "Increase to 3+ services",
        "apply": _apply_service_bundling,
        "eligible": _eligible_service_bundling,
    },
    {
        "name": "Referral Program",
        "description": "Incentivize referrals (target: 5+)",
        "apply": _apply_referral_program,
        "eligible": _eligible_referral_program,
    },
]


# ===================================================================
# PHASE 1: Statistical Simulation (per-customer)
# Uses exact pattern from notebook Cell 95
# ===================================================================

def run_statistical_simulations(customer_features, model, feature_names):
    """
    Run all 6 counterfactual simulations for a single customer.
    customer_features: dict or Series with the customer's encoded features.
    Returns list of dicts sorted by reduction %.
    """
    # Build a 1-row DataFrame matching model input
    row = pd.DataFrame([customer_features])[feature_names].iloc[0]
    current_prob = float(model.predict_proba(pd.DataFrame([row]))[0][1])

    results = []
    for intervention in INTERVENTIONS:
        eligible = intervention["eligible"](row, feature_names)

        if not eligible:
            results.append({
                "name": intervention["name"],
                "description": intervention["description"],
                "eligible": False,
                "current_prob": round(float(current_prob), 4),
                "simulated_prob": None,
                "reduction_pct": 0.0,
                "reason": "Already has this feature",
            })
            continue

        simulated_row = intervention["apply"](row.copy(), feature_names)
        simulated_prob = float(model.predict_proba(pd.DataFrame([simulated_row]))[0][1])

        if current_prob > 0:
            reduction = float((current_prob - simulated_prob) / current_prob * 100)
        else:
            reduction = 0.0

        results.append({
            "name": intervention["name"],
            "description": intervention["description"],
            "eligible": True,
            "current_prob": round(float(current_prob), 4),
            "simulated_prob": round(float(simulated_prob), 4),
            "reduction_pct": round(reduction, 1),
        })

    # Sort by reduction (highest first)
    results.sort(key=lambda x: x.get("reduction_pct", 0), reverse=True)
    return results


# ===================================================================
# PHASE 1 A/B Validation (population-level)
# Uses exact pattern from notebook — mannwhitneyu test
# ===================================================================

def run_ab_validation(X_test, model, feature_names):
    """
    Run A/B validation for each intervention across the full test population.
    Returns dict mapping intervention name → {p_value, significant, n_treatment, n_control}.
    """
    results = {}

    for intervention in INTERVENTIONS:
        name = intervention["name"]

        # Find eligible subset
        eligible_mask = X_test.apply(
            lambda row: intervention["eligible"](row, feature_names), axis=1
        )
        subset = X_test[eligible_mask]

        if len(subset) < 10:
            results[name] = {"p_value": 1.0, "significant": False, "n": 0}
            continue

        # Random assignment — exact notebook pattern
        np.random.seed(42)
        assignment = np.random.choice(
            ["treatment", "control"], size=len(subset), p=[0.5, 0.5]
        )
        treatment_idx = subset.index[assignment == "treatment"]
        control_idx = subset.index[assignment == "control"]

        if len(treatment_idx) < 5 or len(control_idx) < 5:
            results[name] = {"p_value": 1.0, "significant": False, "n": len(subset)}
            continue

        # Control: unchanged
        control_probs = model.predict_proba(subset.loc[control_idx])[:, 1]

        # Treatment: apply intervention
        treatment_data = subset.loc[treatment_idx].copy()
        for idx in treatment_data.index:
            row = treatment_data.loc[idx]
            simulated = intervention["apply"](row, feature_names)
            treatment_data.loc[idx] = simulated

        treatment_probs = model.predict_proba(treatment_data)[:, 1]

        # Mann-Whitney U test — exact notebook pattern
        stat, p_value = mannwhitneyu(
            control_probs, treatment_probs, alternative="greater"
        )

        results[name] = {
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
            "n_treatment": int(len(treatment_idx)),
            "n_control": int(len(control_idx)),
        }

    return results


# ===================================================================
# PHASE 2: Contextual Re-ranking (after diagnosis)
# Uses Claude to score relevance of each intervention to the complaint
# ===================================================================

def run_contextual_reranking(phase1_results, customer_messages, call_claude_fn):
    """
    Re-rank interventions based on how relevant they are to the customer's
    specific complaint. Uses Claude API for scoring.

    Returns phase1_results list with added contextual_fit and combined_score fields.
    """
    messages_text = " | ".join(customer_messages[-5:]) if customer_messages else ""

    intervention_list = "\n".join(
        f"{i+1}. {r['name']} — {r['description']}"
        for i, r in enumerate(phase1_results)
    )

    prompt = (
        f"A telecom customer said: '{messages_text}'\n\n"
        f"Rate how relevant each intervention is to their specific complaint (0-10):\n"
        f"{intervention_list}\n\n"
        f"Rules:\n"
        f"- Service quality complaints (speed, reliability, outages): service improvements score high, pricing interventions score low\n"
        f"- Price/cost complaints: discount and contract upgrade score high\n"
        f"- Billing friction complaints: autopay scores high\n"
        f"- General unhappiness with no specific issue: give moderate scores (5-6) across the board\n"
        f"- Score 0 if the intervention would feel tone-deaf to this customer\n\n"
        f"Return ONLY a JSON object mapping intervention number to score, like: "
        f'{{\"1\": 8, \"2\": 3, \"3\": 2, \"4\": 9, \"5\": 7, \"6\": 1}}'
    )

    try:
        import json
        raw = call_claude_fn(
            "You are an intervention scoring system. Return ONLY valid JSON.",
            prompt,
            max_tokens=100,
        )
        clean = raw.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        scores = json.loads(clean)
    except Exception:
        # Fallback: keyword-based scoring
        scores = _contextual_fallback(messages_text, phase1_results)

    # Normalize statistical reductions to 0-10
    max_reduction = max((r.get("reduction_pct", 0) for r in phase1_results), default=1)
    if max_reduction <= 0:
        max_reduction = 1

    # Combine scores
    reranked = []
    for i, r in enumerate(phase1_results):
        stat_norm = (r.get("reduction_pct", 0) / max_reduction) * 10 if r.get("eligible") else 0
        ctx_score = float(scores.get(str(i + 1), 5))

        combined = (0.4 * stat_norm) + (0.6 * ctx_score)

        entry = {**r, "contextual_fit": round(ctx_score, 1), "combined_score": round(combined, 1)}
        reranked.append(entry)

    # Sort by combined score
    reranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    return reranked


def _contextual_fallback(messages_text, phase1_results):
    """Keyword-based contextual scoring when Claude is unavailable."""
    low = messages_text.lower()
    scores = {}

    for i, r in enumerate(phase1_results):
        name = r["name"].lower()
        score = 5  # default moderate

        if any(w in low for w in ["price", "expensive", "cost", "afford", "bill", "too much", "too high"]):
            if "contract" in name:
                score = 9
            elif "bundling" in name:
                score = 7
            elif "autopay" in name:
                score = 6
            elif "security" in name:
                score = 2
            elif "referral" in name:
                score = 3
            elif "tenure" in name:
                score = 4

        elif any(w in low for w in ["slow", "speed", "drop", "outage", "disconnect", "internet", "wifi", "buffer"]):
            if "security" in name:
                score = 8
            elif "bundling" in name:
                score = 7
            elif "contract" in name:
                score = 3
            elif "autopay" in name:
                score = 2
            elif "referral" in name:
                score = 2
            elif "tenure" in name:
                score = 4

        elif any(w in low for w in ["hate", "terrible", "worst", "awful", "angry", "frustrated"]):
            if "contract" in name:
                score = 8
            elif "bundling" in name:
                score = 6
            elif "autopay" in name:
                score = 4
            elif "security" in name:
                score = 5
            elif "referral" in name:
                score = 3
            elif "tenure" in name:
                score = 5

        scores[str(i + 1)] = score

    return scores
