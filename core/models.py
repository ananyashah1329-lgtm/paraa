"""
Nexus Core Data Models
All shared schemas used across the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ─── Enums ────────────────────────────────────────────────────────────────────

class Channel(str, Enum):
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    CSV = "csv"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class ScoreTier(str, Enum):
    GREEN = "green"       # 70-100: Healthy
    YELLOW = "yellow"     # 40-69: Drifting
    RED = "red"           # 0-39: Critical


class DecisionType(str, Enum):
    PASSIVE = "passive"           # Score healthy, no action
    SOFT_NUDGE = "soft_nudge"     # Score 40-69: light conversation starter
    HARD_NUDGE = "hard_nudge"     # Score < 40: personalized re-engagement
    ESCALATE = "escalate"         # Score < 20 + close tier
    MILESTONE = "milestone"       # Detected birthday/achievement
    NO_ACTION = "no_action"


class RelationshipTier(str, Enum):
    CLOSE = "close"
    ACQUAINTANCE = "acquaintance"
    PROFESSIONAL = "professional"
    UNKNOWN = "unknown"


class FeedbackType(str, Enum):
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    SNOOZED = "snoozed"


class AnomalyType(str, Enum):
    SUDDEN_SILENCE = "sudden_silence"
    ONE_SIDED_MESSAGING = "one_sided_messaging"
    SENTIMENT_DROP = "sentiment_drop"
    VOLUME_COLLAPSE = "volume_collapse"
    LONG_INACTIVITY = "long_inactivity"


# ─── Core Message Schema ──────────────────────────────────────────────────────

@dataclass
class Message:
    """Unified message schema across all channels."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    timestamp: Optional[datetime] = None
    content: str = ""
    content_hash: str = ""        # SHA256 of content for privacy
    channel: Channel = Channel.UNKNOWN
    thread_id: Optional[str] = None
    reply_latency_ms: Optional[float] = None
    mentions: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    is_outgoing: bool = True  # True = sent by the "user"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "content_hash": self.content_hash,
            "channel": self.channel.value,
            "thread_id": self.thread_id,
            "reply_latency_ms": self.reply_latency_ms,
            "mentions": self.mentions,
            "sentiment_score": self.sentiment_score,
            "is_outgoing": self.is_outgoing,
        }


# ─── Contact & Relationship Models ───────────────────────────────────────────

@dataclass
class ContactMetrics:
    """Computed metrics for a single contact relationship."""
    contact_id: str = ""
    contact_name: str = ""
    total_messages: int = 0
    messages_sent: int = 0        # by user
    messages_received: int = 0   # from contact
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    days_since_last_message: float = 0.0
    mean_gap_days: float = 0.0
    std_gap_days: float = 0.0
    initiation_ratio: float = 0.5   # proportion user initiates
    mean_reply_latency_hours: float = 0.0
    message_volume_30d: int = 0
    message_volume_60d: int = 0
    message_volume_90d: int = 0
    sentiment_mean: float = 0.0
    sentiment_trend: float = 0.0    # positive = improving
    detected_milestones: List[str] = field(default_factory=list)
    last_topics: List[str] = field(default_factory=list)
    last_messages_preview: List[str] = field(default_factory=list)  # last 3 messages (anonymized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "total_messages": self.total_messages,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "days_since_last_message": round(self.days_since_last_message, 1),
            "mean_gap_days": round(self.mean_gap_days, 1),
            "std_gap_days": round(self.std_gap_days, 1),
            "initiation_ratio": round(self.initiation_ratio, 3),
            "mean_reply_latency_hours": round(self.mean_reply_latency_hours, 2),
            "message_volume_30d": self.message_volume_30d,
            "message_volume_60d": self.message_volume_60d,
            "message_volume_90d": self.message_volume_90d,
            "sentiment_mean": round(self.sentiment_mean, 3),
            "sentiment_trend": round(self.sentiment_trend, 3),
            "detected_milestones": self.detected_milestones,
            "last_topics": self.last_topics,
            "last_messages_preview": self.last_messages_preview,
        }


@dataclass
class ScoreBreakdown:
    """Per-signal breakdown of the Nexus Score."""
    frequency_score: float = 0.0      # 0-100
    latency_score: float = 0.0        # 0-100
    balance_score: float = 0.0        # 0-100
    sentiment_score: float = 0.0      # 0-100
    recency_score: float = 0.0        # 0-100
    composite_score: float = 0.0      # weighted composite
    tier: ScoreTier = ScoreTier.RED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency_score": round(self.frequency_score, 1),
            "latency_score": round(self.latency_score, 1),
            "balance_score": round(self.balance_score, 1),
            "sentiment_score": round(self.sentiment_score, 1),
            "recency_score": round(self.recency_score, 1),
            "composite_score": round(self.composite_score, 1),
            "tier": self.tier.value,
        }


@dataclass
class Anomaly:
    """A detected behavioral anomaly."""
    type: AnomalyType = AnomalyType.SUDDEN_SILENCE
    severity: float = 0.0    # 0-1
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)
    z_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": round(self.severity, 3),
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "z_score": round(self.z_score, 2) if self.z_score else None,
        }


@dataclass
class ContactState:
    """Full persisted state for a contact."""
    contact_id: str = ""
    contact_name: str = ""
    relationship_tier: RelationshipTier = RelationshipTier.UNKNOWN
    nexus_score: float = 0.0
    score_history: List[Dict] = field(default_factory=list)  # [{timestamp, score}]
    score_breakdown: Optional[ScoreBreakdown] = None
    metrics: Optional[ContactMetrics] = None
    anomalies: List[Anomaly] = field(default_factory=list)
    last_action: Optional[Dict] = None
    user_feedback: Optional[FeedbackType] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "relationship_tier": self.relationship_tier.value,
            "nexus_score": round(self.nexus_score, 1),
            "score_history": self.score_history,
            "score_breakdown": self.score_breakdown.to_dict() if self.score_breakdown else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "last_action": self.last_action,
            "user_feedback": self.user_feedback.value if self.user_feedback else None,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class NudgeResult:
    """Result of a nudge generation for a contact."""
    contact_id: str = ""
    contact_name: str = ""
    decision_type: DecisionType = DecisionType.NO_ACTION
    nexus_score: float = 0.0
    score_tier: ScoreTier = ScoreTier.GREEN
    message_draft: str = ""
    rationale: str = ""
    context_used: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "contact_name": self.contact_name,
            "decision_type": self.decision_type.value,
            "nexus_score": round(self.nexus_score, 1),
            "score_tier": self.score_tier.value,
            "message_draft": self.message_draft,
            "rationale": self.rationale,
            "context_used": self.context_used,
            "generated_at": self.generated_at.isoformat(),
            "llm_used": self.llm_used,
        }


@dataclass
class PipelineResult:
    """Final output of a full pipeline run."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_contacts: int = 0
    total_messages: int = 0
    contacts_green: int = 0
    contacts_yellow: int = 0
    contacts_red: int = 0
    contact_states: List[ContactState] = field(default_factory=list)
    nudge_results: List[NudgeResult] = field(default_factory=list)
    priority_queue: List[Dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)
    run_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_contacts": self.total_contacts,
            "total_messages": self.total_messages,
            "contacts_green": self.contacts_green,
            "contacts_yellow": self.contacts_yellow,
            "contacts_red": self.contacts_red,
            "contact_states": [cs.to_dict() for cs in self.contact_states],
            "nudge_results": [nr.to_dict() for nr in self.nudge_results],
            "priority_queue": self.priority_queue,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "pipeline_metadata": self.pipeline_metadata,
            "run_at": self.run_at.isoformat(),
        }
