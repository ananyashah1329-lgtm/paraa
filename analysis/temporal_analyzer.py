"""
Nexus Temporal Pattern Analyzer
Computes interaction frequency, response latency, initiation ratios,
and message volume trends across rolling windows.
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import logging

from nexus.core.models import Message, ContactMetrics, Anomaly, AnomalyType
from nexus.core.config import CONFIG

logger = logging.getLogger(__name__)


def _days_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 86400.0


def _messages_in_window(messages: List[Message], reference: datetime, days: int) -> List[Message]:
    cutoff = reference - timedelta(days=days)
    return [m for m in messages if m.timestamp and m.timestamp >= cutoff]


def compute_contact_metrics(
    contact_id: str,
    messages: List[Message],
    reference_time: Optional[datetime] = None,
) -> ContactMetrics:
    """
    Compute all temporal and interaction metrics for a contact.

    Args:
        contact_id: The contact's identifier
        messages: All messages between user and this contact (sorted by time)
        reference_time: The "now" reference point (defaults to current time)

    Returns:
        ContactMetrics with all computed signals
    """
    if not reference_time:
        reference_time = datetime.utcnow()

    metrics = ContactMetrics(
        contact_id=contact_id,
        contact_name=contact_id,
    )

    if not messages:
        return metrics

    # Basic counts
    metrics.total_messages = len(messages)
    metrics.messages_sent = sum(1 for m in messages if m.is_outgoing)
    metrics.messages_received = metrics.total_messages - metrics.messages_sent

    # Temporal bounds
    timestamped = [m for m in messages if m.timestamp]
    if not timestamped:
        return metrics

    timestamped.sort(key=lambda m: m.timestamp)
    metrics.first_seen = timestamped[0].timestamp
    metrics.last_seen = timestamped[-1].timestamp
    metrics.days_since_last_message = _days_between(metrics.last_seen, reference_time)

    # Compute inter-message gaps (days between consecutive messages)
    gaps = []
    for i in range(1, len(timestamped)):
        gap = _days_between(timestamped[i-1].timestamp, timestamped[i].timestamp)
        if gap > 0:
            gaps.append(gap)

    if gaps:
        metrics.mean_gap_days = statistics.mean(gaps)
        metrics.std_gap_days = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    else:
        metrics.mean_gap_days = 0.0
        metrics.std_gap_days = 0.0

    # Rolling window volumes
    metrics.message_volume_30d = len(_messages_in_window(timestamped, reference_time, 30))
    metrics.message_volume_60d = len(_messages_in_window(timestamped, reference_time, 60))
    metrics.message_volume_90d = len(_messages_in_window(timestamped, reference_time, 90))

    # Initiation ratio
    # A "conversation" starts after >2 hour silence
    CONVERSATION_BREAK_HOURS = 2.0
    conversations_started_by_user = 0
    conversations_started_by_contact = 0

    prev_ts = None
    for m in timestamped:
        if prev_ts is None or _days_between(prev_ts, m.timestamp) * 24 > CONVERSATION_BREAK_HOURS:
            if m.is_outgoing:
                conversations_started_by_user += 1
            else:
                conversations_started_by_contact += 1
        prev_ts = m.timestamp

    total_convs = conversations_started_by_user + conversations_started_by_contact
    if total_convs > 0:
        metrics.initiation_ratio = conversations_started_by_user / total_convs
    else:
        metrics.initiation_ratio = 0.5

    # Reply latency
    latencies = [
        m.reply_latency_ms for m in messages
        if m.reply_latency_ms and 60000 < m.reply_latency_ms < 86400000  # 1min to 24hr
    ]
    if latencies:
        metrics.mean_reply_latency_hours = statistics.mean(latencies) / 3600000.0
    else:
        # Compute from message pairs where direction changes
        computed_latencies = _compute_latencies_from_pairs(timestamped)
        if computed_latencies:
            metrics.mean_reply_latency_hours = statistics.mean(computed_latencies) / 3600000.0

    # Last messages preview (anonymized — no real content stored)
    last_msgs = timestamped[-3:]
    metrics.last_messages_preview = [
        f"[{'Sent' if m.is_outgoing else 'Received'} message at {m.timestamp.strftime('%Y-%m-%d %H:%M') if m.timestamp else 'unknown time'}]"
        for m in last_msgs
    ]

    return metrics


def _compute_latencies_from_pairs(messages: List[Message]) -> List[float]:
    """Compute reply latencies from direction-change pairs."""
    latencies = []
    for i in range(1, len(messages)):
        prev = messages[i-1]
        curr = messages[i]
        if (prev.timestamp and curr.timestamp and
                prev.is_outgoing != curr.is_outgoing):
            gap_ms = (curr.timestamp - prev.timestamp).total_seconds() * 1000
            if 60000 < gap_ms < 86400000:
                latencies.append(gap_ms)
    return latencies


def detect_anomalies(
    metrics: ContactMetrics,
    historical_metrics: Optional[ContactMetrics] = None,
    reference_time: Optional[datetime] = None,
) -> List[Anomaly]:
    """
    Detect behavioral anomalies for a contact.

    Checks:
      - Sudden silence (gap z-score > threshold)
      - One-sided messaging (initiation ratio extreme)
      - Long inactivity (days since last message > threshold)
      - Volume collapse (30d vs 60d drop)
    """
    if not reference_time:
        reference_time = datetime.utcnow()

    anomalies = []
    zscore_threshold = CONFIG.scoring.anomaly_zscore_threshold

    # ── Sudden Silence ────────────────────────────────────────────────────────
    if (metrics.mean_gap_days > 0 and
            metrics.std_gap_days > 0 and
            metrics.days_since_last_message > 0):
        z = (metrics.days_since_last_message - metrics.mean_gap_days) / max(metrics.std_gap_days, 0.1)
        if z > zscore_threshold:
            severity = min(1.0, z / (zscore_threshold * 3))
            anomalies.append(Anomaly(
                type=AnomalyType.SUDDEN_SILENCE,
                severity=severity,
                description=f"Silence of {metrics.days_since_last_message:.0f} days is {z:.1f}σ above historical mean gap of {metrics.mean_gap_days:.1f} days.",
                z_score=z,
            ))

    # ── Long Inactivity ───────────────────────────────────────────────────────
    if metrics.days_since_last_message > 30:
        severity = min(1.0, metrics.days_since_last_message / 90.0)
        anomalies.append(Anomaly(
            type=AnomalyType.LONG_INACTIVITY,
            severity=severity,
            description=f"No messages for {metrics.days_since_last_message:.0f} days.",
        ))

    # ── One-Sided Messaging ───────────────────────────────────────────────────
    if metrics.total_messages >= 5:
        imbalance = abs(metrics.initiation_ratio - 0.5) * 2  # 0 = balanced, 1 = fully one-sided
        if imbalance > 0.7:  # > 85% initiated by one party
            initiator = "You" if metrics.initiation_ratio > 0.85 else metrics.contact_name
            anomalies.append(Anomaly(
                type=AnomalyType.ONE_SIDED_MESSAGING,
                severity=imbalance,
                description=f"{initiator} initiates {metrics.initiation_ratio*100:.0f}% of conversations. Significant imbalance.",
            ))

    # ── Volume Collapse ───────────────────────────────────────────────────────
    if metrics.message_volume_60d > 5:
        expected_30d = metrics.message_volume_60d / 2
        if expected_30d > 0:
            volume_ratio = metrics.message_volume_30d / expected_30d
            if volume_ratio < 0.3:  # 70%+ drop in activity
                severity = 1.0 - volume_ratio
                anomalies.append(Anomaly(
                    type=AnomalyType.VOLUME_COLLAPSE,
                    severity=severity,
                    description=f"Message volume dropped {(1-volume_ratio)*100:.0f}% in last 30 days vs previous 30 days.",
                ))

    return anomalies


def compute_sentiment_trend(sentiment_scores: List[Tuple[datetime, float]]) -> float:
    """
    Compute sentiment trend slope over time.
    Returns positive value for improving sentiment, negative for declining.
    """
    if len(sentiment_scores) < 3:
        return 0.0

    # Sort by time
    sorted_scores = sorted(sentiment_scores, key=lambda x: x[0])
    n = len(sorted_scores)

    # Convert timestamps to numeric (days from start)
    t0 = sorted_scores[0][0]
    xs = [(s[0] - t0).total_seconds() / 86400.0 for s in sorted_scores]
    ys = [s[1] for s in sorted_scores]

    # Simple linear regression slope
    if len(xs) < 2:
        return 0.0

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    return numerator / denominator  # slope in sentiment_units/day


def compute_weekly_activity(messages: List[Message], weeks: int = 12) -> List[Dict]:
    """Compute weekly message counts for trend visualization."""
    if not messages:
        return []

    now = datetime.utcnow()
    weekly_data = []

    for week in range(weeks - 1, -1, -1):
        week_end = now - timedelta(weeks=week)
        week_start = week_end - timedelta(weeks=1)
        count = sum(
            1 for m in messages
            if m.timestamp and week_start <= m.timestamp < week_end
        )
        weekly_data.append({
            "week": (now - timedelta(weeks=week)).strftime("%Y-W%U"),
            "count": count,
            "week_start": week_start.isoformat(),
        })

    return weekly_data
