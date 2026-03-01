"""
Nexus Score™ Engine
Computes the composite 0-100 relationship health score for each contact.

Formula:
  score = w1*frequency + w2*latency + w3*balance + w4*sentiment + w5*recency
  Each signal is independently normalized to 0-100 before weighting.
  Final score is clipped to [0, 100].
"""

import math
from typing import Optional
import logging

from nexus.core.models import (
    ContactMetrics, ScoreBreakdown, ScoreTier, ContactState, RelationshipTier
)
from nexus.core.config import CONFIG

logger = logging.getLogger(__name__)


def _normalize_frequency(msgs_per_month: float, healthy_baseline: float) -> float:
    """
    Normalize message frequency to 0-100.
    Uses a logarithmic scale so diminishing returns above healthy_baseline.
    """
    if msgs_per_month <= 0:
        return 0.0
    # Log scale: healthy_baseline → 100
    ratio = msgs_per_month / max(healthy_baseline, 1.0)
    score = 100 * (1 - math.exp(-ratio))
    return min(100.0, max(0.0, score))


def _normalize_latency(mean_reply_hours: float,
                        fast_threshold: float,
                        slow_threshold: float) -> float:
    """
    Normalize reply latency to 0-100.
    Fast replies = high score; slow replies = low score.
    Handles 0 latency gracefully.
    """
    if mean_reply_hours <= 0:
        return 70.0  # Unknown latency → neutral-good

    fast_h = fast_threshold / 60.0  # Convert minutes to hours
    slow_h = slow_threshold

    if mean_reply_hours <= fast_h:
        return 100.0
    elif mean_reply_hours >= slow_h:
        return 0.0
    else:
        # Linear interpolation between fast and slow thresholds
        return 100.0 * (1 - (mean_reply_hours - fast_h) / (slow_h - fast_h))


def _normalize_balance(initiation_ratio: float) -> float:
    """
    Normalize initiation balance to 0-100.
    0.5 (perfectly balanced) → 100
    0.0 or 1.0 (completely one-sided) → 0
    """
    # Distance from ideal 0.5
    distance = abs(initiation_ratio - 0.5)
    # Scale: distance of 0 → 100, distance of 0.5 → 0
    return max(0.0, 100.0 * (1 - 2 * distance))


def _normalize_sentiment(mean_sentiment: float) -> float:
    """
    Normalize sentiment score to 0-100.
    mean_sentiment is in [-1, 1].
    """
    # Map [-1, 1] → [0, 100]
    return max(0.0, min(100.0, (mean_sentiment + 1.0) * 50.0))


def _compute_recency_score(days_since_last: float, decay_lambda: float) -> float:
    """
    Compute recency score using exponential decay.
    score = 100 * e^(-lambda * days_since_last)
    """
    if days_since_last < 0:
        days_since_last = 0
    return 100.0 * math.exp(-decay_lambda * days_since_last)


def _tier_from_score(score: float) -> ScoreTier:
    if score >= CONFIG.scoring.green_threshold:
        return ScoreTier.GREEN
    elif score >= CONFIG.scoring.yellow_threshold:
        return ScoreTier.YELLOW
    return ScoreTier.RED


def compute_nexus_score(
    metrics: ContactMetrics,
    weight_override: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Compute the Nexus Score™ for a contact.

    Args:
        metrics: ContactMetrics computed by temporal analyzer
        weight_override: Optional dict to override default weights

    Returns:
        ScoreBreakdown with per-signal scores and composite score
    """
    cfg = CONFIG.scoring
    weights = cfg.weights

    if weight_override:
        from nexus.core.config import ScoringWeights
        weights = ScoringWeights(**{
            k: weight_override.get(k, getattr(cfg.weights, k))
            for k in ["frequency", "latency", "balance", "sentiment", "recency"]
        })

    # ── Signal 1: Frequency ───────────────────────────────────────────────────
    # Normalize to monthly rate
    msgs_per_month = metrics.message_volume_30d
    frequency_score = _normalize_frequency(msgs_per_month, cfg.healthy_frequency_per_month)

    # ── Signal 2: Latency ─────────────────────────────────────────────────────
    latency_score = _normalize_latency(
        metrics.mean_reply_latency_hours,
        cfg.fast_reply_threshold_minutes,
        cfg.slow_reply_threshold_hours,
    )

    # ── Signal 3: Balance ─────────────────────────────────────────────────────
    balance_score = _normalize_balance(metrics.initiation_ratio)

    # ── Signal 4: Sentiment ───────────────────────────────────────────────────
    sentiment_score = _normalize_sentiment(metrics.sentiment_mean)

    # ── Signal 5: Recency ─────────────────────────────────────────────────────
    recency_score = _compute_recency_score(
        metrics.days_since_last_message,
        cfg.recency_decay_lambda,
    )

    # ── Composite Score ───────────────────────────────────────────────────────
    composite = (
        weights.frequency * frequency_score +
        weights.latency * latency_score +
        weights.balance * balance_score +
        weights.sentiment * sentiment_score +
        weights.recency * recency_score
    )

    # Clip to valid range
    composite = max(0.0, min(100.0, composite))

    breakdown = ScoreBreakdown(
        frequency_score=frequency_score,
        latency_score=latency_score,
        balance_score=balance_score,
        sentiment_score=sentiment_score,
        recency_score=recency_score,
        composite_score=composite,
        tier=_tier_from_score(composite),
    )

    logger.debug(
        f"Score [{metrics.contact_id}]: freq={frequency_score:.1f} lat={latency_score:.1f} "
        f"bal={balance_score:.1f} sent={sentiment_score:.1f} rec={recency_score:.1f} "
        f"→ composite={composite:.1f} [{breakdown.tier.value}]"
    )

    return breakdown


def build_contact_state(
    contact_id: str,
    metrics: ContactMetrics,
    previous_state: Optional[ContactState] = None,
    relationship_tier: RelationshipTier = RelationshipTier.UNKNOWN,
) -> ContactState:
    """
    Build or update a ContactState for a contact.
    Incorporates previous state for history tracking.
    """
    from datetime import datetime

    score_breakdown = compute_nexus_score(metrics)
    composite_score = score_breakdown.composite_score

    # History: carry over previous or start fresh
    history = []
    if previous_state:
        history = previous_state.score_history.copy()
        relationship_tier = previous_state.relationship_tier

    history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "score": round(composite_score, 1),
    })

    # Keep last 30 history points
    history = history[-30:]

    # Infer tier if unknown
    if relationship_tier == RelationshipTier.UNKNOWN:
        if composite_score >= CONFIG.scoring.green_threshold:
            relationship_tier = RelationshipTier.ACQUAINTANCE
        elif composite_score >= CONFIG.scoring.yellow_threshold:
            relationship_tier = RelationshipTier.ACQUAINTANCE
        else:
            relationship_tier = RelationshipTier.UNKNOWN

    state = ContactState(
        contact_id=contact_id,
        contact_name=metrics.contact_name,
        relationship_tier=relationship_tier,
        nexus_score=composite_score,
        score_history=history,
        score_breakdown=score_breakdown,
        metrics=metrics,
        last_updated=datetime.utcnow(),
    )

    return state


def rank_priority_queue(states: list) -> list:
    """
    Rank contacts into priority queue for immediate attention.
    Priority considers: score, tier, relationship_tier, recency.
    """
    def priority_key(state: ContactState) -> float:
        score = state.nexus_score
        # Close relationships get higher priority boost
        tier_boost = {
            RelationshipTier.CLOSE: 20.0,
            RelationshipTier.PROFESSIONAL: 10.0,
            RelationshipTier.ACQUAINTANCE: 0.0,
            RelationshipTier.UNKNOWN: 0.0,
        }.get(state.relationship_tier, 0.0)

        # Lower score = higher urgency (negate for sort)
        return -(100 - score + tier_boost)

    # Filter to yellow and red only
    at_risk = [s for s in states if s.score_breakdown and s.score_breakdown.tier != ScoreTier.GREEN]
    at_risk.sort(key=priority_key)

    return [
        {
            "rank": i + 1,
            "contact_id": s.contact_id,
            "contact_name": s.contact_name,
            "nexus_score": round(s.nexus_score, 1),
            "tier": s.score_breakdown.tier.value if s.score_breakdown else "unknown",
            "relationship_tier": s.relationship_tier.value,
            "days_since_last_message": round(s.metrics.days_since_last_message, 0) if s.metrics else None,
            "top_anomaly": s.anomalies[0].description if s.anomalies else None,
        }
        for i, s in enumerate(at_risk)
    ]
