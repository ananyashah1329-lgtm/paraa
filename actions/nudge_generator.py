"""
Nexus Action Generator
Generates context-aware re-engagement message drafts using Claude API.
Falls back to template-based generation when API is unavailable.
"""

import os
import json
import random
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Optional, List
import logging

from nexus.core.models import (
    ContactState, NudgeResult, DecisionType, ScoreTier
)
from nexus.core.config import CONFIG, FALLBACK_TEMPLATES
from nexus.decision.engine import build_decision_context

logger = logging.getLogger(__name__)


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def _build_prompt(decision_type: DecisionType, context: Dict) -> str:
    """Build a structured prompt for Claude API."""
    name = context.get("contact_name", "this person")
    days = int(context.get("days_since_last_message", 0))
    score = context.get("nexus_score", 50)
    tier = context.get("relationship_tier", "acquaintance")
    topics = context.get("last_topics", [])
    milestones = context.get("detected_milestones", [])
    sentiment = context.get("sentiment_mean", 0.0)
    anomalies = context.get("anomalies", [])

    # Context snippets
    topics_str = ", ".join(topics[:3]) if topics else "general topics"
    milestone_str = f"Detected milestones: {', '.join(milestones)}." if milestones else ""
    sentiment_str = "positive" if sentiment > 0.2 else ("negative" if sentiment < -0.2 else "neutral")
    anomaly_str = f"Notable: {anomalies[0]}" if anomalies else ""

    base_context = f"""Contact: {name}
Relationship tier: {tier}
Nexus Score: {score}/100 ({context.get('score_tier', 'unknown')} zone)
Days since last message: {days} days
Conversation sentiment: {sentiment_str}
Common topics: {topics_str}
{milestone_str}
{anomaly_str}
Initiation balance: User initiates {context.get('initiation_ratio', 0.5)*100:.0f}% of conversations"""

    if decision_type == DecisionType.MILESTONE:
        instruction = f"""Generate a warm, genuine congratulatory message for {name}.
The milestone detected is: {', '.join(milestones)}.
Make it personal, enthusiastic, and brief (2-3 sentences). Don't use generic phrases like "Hope you're doing well"."""

    elif decision_type in (DecisionType.HARD_NUDGE, DecisionType.ESCALATE):
        instruction = f"""Generate a warm, genuine re-engagement message for {name}.
It has been {days} days since you last spoke. The relationship is at risk.
Make it personal, reference something specific to your relationship, and invite a response.
Keep it casual, 2-3 sentences. Avoid being needy or dramatic. Don't mention the long gap awkwardly."""

    elif decision_type == DecisionType.SOFT_NUDGE:
        instruction = f"""Generate a light, casual conversation starter for {name}.
You haven't talked in a while and want to reconnect naturally.
Keep it brief (1-2 sentences), warm, and low-pressure. Could reference a shared interest or just check in."""

    else:
        instruction = f"Generate a friendly check-in message for {name}. Keep it brief and warm."

    return f"""{base_context}

Task: {instruction}

Rules:
- Write ONLY the message text, no preamble, no quotes around it
- Be warm and genuine, not formulaic
- Match the relationship tier (close = warmer, professional = slightly more formal)
- Do not start with "Hey" if the tier is professional
- Maximum 3 sentences"""


def _build_rationale(decision_type: DecisionType, context: Dict) -> str:
    """Generate a human-readable rationale for the decision."""
    name = context.get("contact_name", "Contact")
    score = context.get("nexus_score", 50)
    days = int(context.get("days_since_last_message", 0))
    anomalies = context.get("anomalies", [])

    rationale_map = {
        DecisionType.PASSIVE: f"{name}'s relationship score ({score:.0f}) is healthy. No action required.",
        DecisionType.SOFT_NUDGE: f"{name}'s score has drifted to {score:.0f}. A light check-in can re-energize the relationship.",
        DecisionType.HARD_NUDGE: f"{name}'s score is {score:.0f} and you haven't spoken in {days} days. Proactive re-engagement recommended.",
        DecisionType.ESCALATE: f"URGENT: {name} (close relationship) has a critical score of {score:.0f}. Immediate attention needed.",
        DecisionType.MILESTONE: f"A milestone was detected in your conversations with {name}. This is a great opportunity to connect.",
        DecisionType.NO_ACTION: "No action determined.",
    }

    base = rationale_map.get(decision_type, "Action required.")
    if anomalies:
        base += f" Detected: {anomalies[0]}"
    return base


# ─── LLM API Call ─────────────────────────────────────────────────────────────

def _call_claude_api(prompt: str, api_key: str) -> Optional[str]:
    """Call Claude API to generate a message draft."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": CONFIG.llm_model,
        "max_tokens": CONFIG.llm_max_tokens,
        "system": "You are Nexus, an AI relationship intelligence assistant. You help people maintain meaningful relationships by generating warm, contextual re-engagement messages. Always write naturally, as if the user themselves composed the message.",
        "messages": [{"role": "user", "content": prompt}]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("content") and result["content"][0].get("type") == "text":
                return result["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        logger.error(f"Claude API HTTP error: {e.code} - {e.read().decode()}")
    except urllib.error.URLError as e:
        logger.error(f"Claude API connection error: {e}")
    except Exception as e:
        logger.error(f"Claude API unexpected error: {e}")

    return None


# ─── Fallback Template Generation ────────────────────────────────────────────

def _generate_from_template(decision_type: DecisionType, context: Dict) -> str:
    """Generate a message using fallback templates when LLM is unavailable."""
    name = context.get("contact_name", "Hey")
    days = int(context.get("days_since_last_message", 0))
    topics = context.get("last_topics", [])
    milestones = context.get("detected_milestones", [])

    if decision_type == DecisionType.MILESTONE and milestones:
        template = random.choice(FALLBACK_TEMPLATES["milestone"])
        return template

    elif decision_type in (DecisionType.HARD_NUDGE, DecisionType.ESCALATE):
        templates = FALLBACK_TEMPLATES["hard_nudge"]
        template = random.choice(templates)
        return template.format(days=days)

    else:
        template = random.choice(FALLBACK_TEMPLATES["soft_nudge"])
        return template


# ─── Main Action Generator ────────────────────────────────────────────────────

def generate_nudge(
    state: ContactState,
    decision_type: DecisionType,
    api_key: Optional[str] = None,
) -> NudgeResult:
    """
    Generate a nudge result for a contact.

    Args:
        state: Full contact state
        decision_type: Pre-determined decision type
        api_key: Anthropic API key (optional)

    Returns:
        NudgeResult with message draft and metadata
    """
    context = build_decision_context(state)
    rationale = _build_rationale(decision_type, context)

    # No message needed for passive
    if decision_type in (DecisionType.PASSIVE, DecisionType.NO_ACTION):
        return NudgeResult(
            contact_id=state.contact_id,
            contact_name=state.contact_name,
            decision_type=decision_type,
            nexus_score=state.nexus_score,
            score_tier=state.score_breakdown.tier if state.score_breakdown else ScoreTier.GREEN,
            message_draft="",
            rationale=rationale,
            context_used=context,
            llm_used=False,
        )

    # Try LLM first
    message_draft = None
    llm_used = False
    effective_api_key = api_key or CONFIG.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if effective_api_key:
        prompt = _build_prompt(decision_type, context)
        message_draft = _call_claude_api(prompt, effective_api_key)
        if message_draft:
            llm_used = True
            logger.info(f"LLM nudge generated for {state.contact_id}")

    # Fallback to template
    if not message_draft:
        message_draft = _generate_from_template(decision_type, context)
        logger.info(f"Template nudge generated for {state.contact_id}")

    result = NudgeResult(
        contact_id=state.contact_id,
        contact_name=state.contact_name,
        decision_type=decision_type,
        nexus_score=state.nexus_score,
        score_tier=state.score_breakdown.tier if state.score_breakdown else ScoreTier.RED,
        message_draft=message_draft,
        rationale=rationale,
        context_used=context,
        llm_used=llm_used,
    )

    # Update last_action on state
    state.last_action = {
        "type": decision_type.value,
        "timestamp": datetime.utcnow().isoformat(),
        "message_draft": message_draft,
    }

    return result


def generate_batch_nudges(
    decisions: List[Dict],
    states_map: Dict[str, ContactState],
    api_key: Optional[str] = None,
) -> List[NudgeResult]:
    """Generate nudges for all contacts needing action."""
    results = []
    for d in decisions:
        contact_id = d["contact_id"]
        decision_type = d["decision_type"]
        state = states_map.get(contact_id)

        if not state:
            continue

        result = generate_nudge(state, decision_type, api_key=api_key)
        results.append(result)

    return results


def generate_weekly_digest(states: List[ContactState]) -> Dict:
    """Generate a weekly relationship health summary."""
    green = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.GREEN]
    yellow = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.YELLOW]
    red = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.RED]

    # Score changes (if history available)
    improving = []
    declining = []
    for s in states:
        if len(s.score_history) >= 2:
            delta = s.score_history[-1]["score"] - s.score_history[-2]["score"]
            if delta > 5:
                improving.append({"name": s.contact_name, "delta": round(delta, 1)})
            elif delta < -5:
                declining.append({"name": s.contact_name, "delta": round(delta, 1)})

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_contacts": len(states),
            "healthy": len(green),
            "drifting": len(yellow),
            "critical": len(red),
        },
        "improving_relationships": sorted(improving, key=lambda x: -x["delta"])[:3],
        "declining_relationships": sorted(declining, key=lambda x: x["delta"])[:3],
        "top_at_risk": [
            {"name": s.contact_name, "score": round(s.nexus_score, 1)}
            for s in sorted(red, key=lambda s: s.nexus_score)[:3]
        ],
        "wins": [
            {"name": s.contact_name, "score": round(s.nexus_score, 1)}
            for s in sorted(green, key=lambda s: -s.nexus_score)[:3]
        ],
    }
