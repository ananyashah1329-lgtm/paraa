"""
Nexus State Store
SQLite-based persistence for contact states, session results, and feedback.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict
import logging

from nexus.core.models import ContactState, PipelineResult
from nexus.core.config import CONFIG

logger = logging.getLogger(__name__)


class StateStore:
    """
    Lightweight SQLite state store for Nexus.
    Handles contact state persistence, session logs, and feedback tracking.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or CONFIG.db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS contact_states (
                    contact_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    data TEXT NOT NULL,
                    nexus_score REAL,
                    tier TEXT,
                    last_updated TEXT
                );

                CREATE TABLE IF NOT EXISTS pipeline_sessions (
                    session_id TEXT PRIMARY KEY,
                    run_at TEXT,
                    total_contacts INTEGER,
                    total_messages INTEGER,
                    contacts_green INTEGER,
                    contacts_yellow INTEGER,
                    contacts_red INTEGER,
                    processing_time_ms REAL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS nudge_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    contact_id TEXT,
                    contact_name TEXT,
                    decision_type TEXT,
                    nexus_score REAL,
                    message_draft TEXT,
                    rationale TEXT,
                    feedback TEXT,
                    generated_at TEXT,
                    feedback_at TEXT
                );

                CREATE TABLE IF NOT EXISTS score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact_id TEXT,
                    session_id TEXT,
                    score REAL,
                    tier TEXT,
                    recorded_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_contact_states_score ON contact_states(nexus_score);
                CREATE INDEX IF NOT EXISTS idx_nudge_log_contact ON nudge_log(contact_id);
                CREATE INDEX IF NOT EXISTS idx_score_history_contact ON score_history(contact_id);
            """)
        logger.info(f"StateStore initialized at {self.db_path}")

    # ── Contact State CRUD ────────────────────────────────────────────────────

    def save_contact_state(self, state: ContactState, session_id: str = ""):
        """Upsert a contact state."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO contact_states
                    (contact_id, session_id, data, nexus_score, tier, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                state.contact_id,
                session_id,
                json.dumps(state.to_dict()),
                round(state.nexus_score, 2),
                state.score_breakdown.tier.value if state.score_breakdown else "unknown",
                datetime.utcnow().isoformat(),
            ))

    def get_contact_state(self, contact_id: str) -> Optional[ContactState]:
        """Load a contact state by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM contact_states WHERE contact_id = ?",
                (contact_id,)
            ).fetchone()
        if not row:
            return None
        try:
            return _deserialize_state(json.loads(row["data"]))
        except Exception as e:
            logger.error(f"Failed to deserialize state for {contact_id}: {e}")
            return None

    def get_all_contact_states(self) -> List[ContactState]:
        """Load all contact states."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM contact_states ORDER BY nexus_score ASC"
            ).fetchall()
        states = []
        for row in rows:
            try:
                states.append(_deserialize_state(json.loads(row["data"])))
            except Exception as e:
                logger.warning(f"Skipping malformed state: {e}")
        return states

    def delete_contact_state(self, contact_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM contact_states WHERE contact_id = ?", (contact_id,))

    def clear_all_states(self):
        """Clear all data — useful for resetting between demo sessions."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM contact_states;
                DELETE FROM pipeline_sessions;
                DELETE FROM nudge_log;
                DELETE FROM score_history;
            """)
        logger.info("All states cleared")

    # ── Session Logging ───────────────────────────────────────────────────────

    def save_session(self, result: PipelineResult):
        """Log a pipeline run."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipeline_sessions
                    (session_id, run_at, total_contacts, total_messages,
                     contacts_green, contacts_yellow, contacts_red,
                     processing_time_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.session_id,
                result.run_at.isoformat(),
                result.total_contacts,
                result.total_messages,
                result.contacts_green,
                result.contacts_yellow,
                result.contacts_red,
                result.processing_time_ms,
                json.dumps(result.pipeline_metadata),
            ))

    def get_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pipeline_sessions ORDER BY run_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Nudge Log ─────────────────────────────────────────────────────────────

    def log_nudge(self, nudge_result, session_id: str = ""):
        """Log a nudge result."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO nudge_log
                    (session_id, contact_id, contact_name, decision_type,
                     nexus_score, message_draft, rationale, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                nudge_result.contact_id,
                nudge_result.contact_name,
                nudge_result.decision_type.value,
                round(nudge_result.nexus_score, 2),
                nudge_result.message_draft,
                nudge_result.rationale,
                nudge_result.generated_at.isoformat(),
            ))

    def update_nudge_feedback(self, contact_id: str, feedback: str):
        """Update feedback for most recent nudge for a contact."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE nudge_log SET feedback = ?, feedback_at = ?
                WHERE contact_id = ?
                AND id = (SELECT MAX(id) FROM nudge_log WHERE contact_id = ?)
            """, (feedback, datetime.utcnow().isoformat(), contact_id, contact_id))

    def get_nudge_history(self, contact_id: str, limit: int = 10) -> List[Dict]:
        """Get nudge history for a contact."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM nudge_log WHERE contact_id = ? ORDER BY generated_at DESC LIMIT ?",
                (contact_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Score History ─────────────────────────────────────────────────────────

    def record_score(self, contact_id: str, score: float, tier: str, session_id: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO score_history (contact_id, session_id, score, tier, recorded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (contact_id, session_id, round(score, 2), tier, datetime.utcnow().isoformat()))

    def get_score_history(self, contact_id: str, limit: int = 30) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT score, tier, recorded_at FROM score_history WHERE contact_id = ? ORDER BY recorded_at DESC LIMIT ?",
                (contact_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get database stats."""
        with self._connect() as conn:
            contacts = conn.execute("SELECT COUNT(*) as n FROM contact_states").fetchone()["n"]
            sessions = conn.execute("SELECT COUNT(*) as n FROM pipeline_sessions").fetchone()["n"]
            nudges = conn.execute("SELECT COUNT(*) as n FROM nudge_log").fetchone()["n"]
        return {
            "total_contacts": contacts,
            "total_sessions": sessions,
            "total_nudges": nudges,
            "db_path": self.db_path,
            "db_size_kb": round(os.path.getsize(self.db_path) / 1024, 1) if os.path.exists(self.db_path) else 0,
        }


# ─── Deserialization Helper ───────────────────────────────────────────────────

def _deserialize_state(data: Dict) -> ContactState:
    """Rebuild a ContactState from its dict representation."""
    from nexus.core.models import (
        ContactMetrics, ScoreBreakdown, Anomaly,
        ScoreTier, RelationshipTier, FeedbackType, AnomalyType
    )

    state = ContactState()
    state.contact_id = data.get("contact_id", "")
    state.contact_name = data.get("contact_name", "")
    state.nexus_score = data.get("nexus_score", 0.0)
    state.score_history = data.get("score_history", [])
    state.last_action = data.get("last_action")
    state.last_updated = datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.utcnow()

    try:
        state.relationship_tier = RelationshipTier(data.get("relationship_tier", "unknown"))
    except ValueError:
        state.relationship_tier = RelationshipTier.UNKNOWN

    try:
        fb = data.get("user_feedback")
        state.user_feedback = FeedbackType(fb) if fb else None
    except ValueError:
        state.user_feedback = None

    # Rebuild breakdown
    bd = data.get("score_breakdown")
    if bd:
        breakdown = ScoreBreakdown()
        breakdown.frequency_score = bd.get("frequency_score", 0)
        breakdown.latency_score = bd.get("latency_score", 0)
        breakdown.balance_score = bd.get("balance_score", 0)
        breakdown.sentiment_score = bd.get("sentiment_score", 0)
        breakdown.recency_score = bd.get("recency_score", 0)
        breakdown.composite_score = bd.get("composite_score", 0)
        try:
            breakdown.tier = ScoreTier(bd.get("tier", "red"))
        except ValueError:
            breakdown.tier = ScoreTier.RED
        state.score_breakdown = breakdown

    # Rebuild metrics (partial)
    m = data.get("metrics")
    if m:
        metrics = ContactMetrics()
        for key in [
            "contact_id", "contact_name", "total_messages", "messages_sent",
            "messages_received", "days_since_last_message", "mean_gap_days",
            "std_gap_days", "initiation_ratio", "mean_reply_latency_hours",
            "message_volume_30d", "message_volume_60d", "message_volume_90d",
            "sentiment_mean", "sentiment_trend", "detected_milestones",
            "last_topics", "last_messages_preview"
        ]:
            if key in m:
                setattr(metrics, key, m[key])
        state.metrics = metrics

    # Anomalies
    anomalies = []
    for a_data in data.get("anomalies", []):
        try:
            anomaly = Anomaly(
                type=AnomalyType(a_data.get("type", "sudden_silence")),
                severity=a_data.get("severity", 0),
                description=a_data.get("description", ""),
            )
            anomalies.append(anomaly)
        except ValueError:
            pass
    state.anomalies = anomalies

    return state


# Global store instance
store = StateStore()
