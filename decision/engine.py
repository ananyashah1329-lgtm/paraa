"""
Nexus Decision Engine
Evaluates each contact's state and determines the appropriate action.
Hybrid rule-based + contextual logic.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

from nexus.core.models import (
    ContactState, ScoreTier, DecisionType, RelationshipTier, NudgeResult
)
from nexus.core.config import CONFIG

logger = logging.getLogger(__name__)


def _get_silence_threshold(tier: RelationshipTier) -> float:
    """Return the max acceptable silence days for a relationship tier."""
    thresholds = {
        RelationshipTier.CLOSE: CONFIG.decision.max_silence_days_close,
        RelationshipTier.PROFESSIONAL: CONFIG.decision.max_silence_days_professional,
        RelationshipTier.ACQUAINTANCE: CONFIG.decision.max_silence_days_acquaintance,
        RelationshipTier.UNKNOWN: CONFIG.decision.max_silence_days_acquaintance,
    }
    return thresholds.get(tier, CONFIG.decision.max_silence_days_acquaintance)


def _nudge_on_cooldown(state: ContactState) -> bool:
    """Check if a nudge was sent recently (within cooldown period)."""
    if not state.last_action:
        return False
    last_action_ts = state.last_action.get("timestamp")
    if not last_action_ts:
        return False
    try:
        last_ts = datetime.fromisoformat(last_action_ts)
        cooldown = timedelta(days=CONFIG.decision.nudge_cooldown_days)
        return (datetime.utcnow() - last_ts) < cooldown
    except (ValueError, TypeError):
        return False


def decide(state: ContactState) -> DecisionType:
    """
    Determine the decision type for a contact.

    Decision priority (highest to lowest):
    1. Milestone detected → MILESTONE
    2. Score < escalate_threshold AND close tier → ESCALATE
    3. Score < hard_nudge_threshold OR silence > max → HARD_NUDGE
    4. Score < green_threshold → SOFT_NUDGE
    5. Otherwise → PASSIVE
    """
    if not state.score_breakdown:
        return DecisionType.NO_ACTION

    score = state.nexus_score
    tier = state.score_breakdown.tier
    days_silent = state.metrics.days_since_last_message if state.metrics else 0.0
    silence_threshold = _get_silence_threshold(state.relationship_tier)

    # Check cooldown
    if _nudge_on_cooldown(state):
        logger.debug(f"{state.contact_id}: nudge on cooldown")
        return DecisionType.PASSIVE

    # 1. Milestone
    if state.metrics and state.metrics.detected_milestones:
        return DecisionType.MILESTONE

    # 2. Escalate (critical + close relationship)
    if (score < CONFIG.decision.escalate_threshold and
            state.relationship_tier == RelationshipTier.CLOSE):
        return DecisionType.ESCALATE

    # 3. Hard nudge: very low score OR extended silence
    if (score < CONFIG.decision.hard_nudge_threshold or
            days_silent > silence_threshold):
        return DecisionType.HARD_NUDGE

    # 4. Soft nudge: drifting
    if tier == ScoreTier.YELLOW:
        return DecisionType.SOFT_NUDGE

    # 5. Healthy
    return DecisionType.PASSIVE


def build_decision_context(state: ContactState) -> Dict:
    """
    Build the context payload used for nudge generation.
    Structured data passed to the action layer.
    """
    metrics = state.metrics
    context = {
        "contact_id": state.contact_id,
        "contact_name": state.contact_name,
        "relationship_tier": state.relationship_tier.value,
        "nexus_score": round(state.nexus_score, 1),
        "score_tier": state.score_breakdown.tier.value if state.score_breakdown else "unknown",
        "days_since_last_message": round(metrics.days_since_last_message, 0) if metrics else 0,
        "initiation_ratio": round(metrics.initiation_ratio, 2) if metrics else 0.5,
        "sentiment_mean": round(metrics.sentiment_mean, 2) if metrics else 0.0,
        "sentiment_trend": round(metrics.sentiment_trend, 3) if metrics else 0.0,
        "last_messages_preview": metrics.last_messages_preview if metrics else [],
        "detected_milestones": metrics.detected_milestones if metrics else [],
        "last_topics": metrics.last_topics if metrics else [],
        "anomalies": [a.description for a in state.anomalies[:3]],
        "total_messages": metrics.total_messages if metrics else 0,
        "score_breakdown": state.score_breakdown.to_dict() if state.score_breakdown else {},
    }
    return context


def batch_decide(states: List[ContactState]) -> List[Dict]:
    """
    Run decision logic on all contact states.

    Returns:
        List of decision records {contact_id, decision_type, context}
    """
    decisions = []
    for state in states:
        decision_type = decide(state)
        context = build_decision_context(state)
        decisions.append({
            "contact_id": state.contact_id,
            "contact_name": state.contact_name,
            "decision_type": decision_type,
            "context": context,
        })
        logger.info(f"Decision [{state.contact_id}]: {decision_type.value} (score={state.nexus_score:.1f})")

    return decisions


def apply_feedback(state: ContactState, feedback: str) -> ContactState:
    """
    Apply user feedback to update contact state.
    Accepted: increase relationship tier confidence
    Dismissed: put on longer cooldown
    Snoozed: extend cooldown by 7 days
    """
    from nexus.core.models import FeedbackType
    try:
        fb = FeedbackType(feedback.lower())
        state.user_feedback = fb

        if fb == FeedbackType.ACCEPTED and state.last_action:
            state.last_action["feedback"] = "accepted"
            # Promote tier if unknown
            if state.relationship_tier == RelationshipTier.UNKNOWN:
                state.relationship_tier = RelationshipTier.ACQUAINTANCE

        elif fb == FeedbackType.DISMISSED and state.last_action:
            state.last_action["feedback"] = "dismissed"
            # Extend cooldown by setting last_action to now
            state.last_action["timestamp"] = datetime.utcnow().isoformat()

        elif fb == FeedbackType.SNOOZED and state.last_action:
            state.last_action["feedback"] = "snoozed"
            # Snooze: set action time to 7 days ago so cooldown expires in 7-3=4 more days
            snooze_ts = datetime.utcnow() - timedelta(days=CONFIG.decision.nudge_cooldown_days - 7)
            state.last_action["timestamp"] = snooze_ts.isoformat()

    except ValueError:
        logger.warning(f"Invalid feedback type: {feedback}")

    return state
