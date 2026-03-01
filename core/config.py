"""
Nexus Configuration
All tunable parameters for the scoring, decision, and action layers.
Edit this file to recalibrate without touching pipeline code.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScoringWeights:
    """Weights for each signal in the Nexus Score composite formula."""
    frequency: float = 0.30
    latency: float = 0.20
    balance: float = 0.15
    sentiment: float = 0.20
    recency: float = 0.15

    def validate(self):
        total = self.frequency + self.latency + self.balance + self.sentiment + self.recency
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"


@dataclass
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Recency decay lambda: higher = faster decay
    # score * e^(-lambda * days_since_last_msg)
    recency_decay_lambda: float = 0.05

    # Frequency baseline (messages/month considered "healthy")
    healthy_frequency_per_month: float = 20.0

    # Latency: threshold for "instant" reply (minutes)
    fast_reply_threshold_minutes: float = 30.0
    # Latency: threshold for "slow" reply (hours)
    slow_reply_threshold_hours: float = 48.0

    # Anomaly detection z-score threshold
    anomaly_zscore_threshold: float = 2.0

    # Score tier boundaries
    green_threshold: float = 70.0
    yellow_threshold: float = 40.0


@dataclass
class DecisionConfig:
    # Score thresholds for decision routing
    hard_nudge_threshold: float = 40.0
    escalate_threshold: float = 20.0

    # Days of silence before triggering hard nudge (regardless of score)
    max_silence_days_close: float = 14.0
    max_silence_days_acquaintance: float = 30.0
    max_silence_days_professional: float = 21.0

    # Minimum gap between nudges for same contact (days)
    nudge_cooldown_days: float = 3.0


@dataclass
class NexusConfig:
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # Anthropic API settings
    anthropic_api_key: str = ""  # Set via env var ANTHROPIC_API_KEY
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 500

    # Rolling window sizes (days)
    short_window_days: int = 30
    medium_window_days: int = 60
    long_window_days: int = 90

    # Flask server
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

    # SQLite database path
    db_path: str = "nexus.db"


# Global config instance (override via environment or at startup)
CONFIG = NexusConfig()
CONFIG.scoring.weights.validate()


# Milestone detection keywords
MILESTONE_KEYWORDS = {
    "birthday": ["birthday", "bday", "happy birthday", "born", "turning"],
    "achievement": ["graduated", "promotion", "new job", "hired", "accepted", "got in", "passed", "congrats", "congratulations", "achievement", "award"],
    "life_event": ["engaged", "married", "wedding", "baby", "pregnant", "expecting", "moved", "new house", "apartment"],
    "loss": ["passed away", "funeral", "grief", "lost", "death", "died"],
    "travel": ["trip", "vacation", "traveling", "visiting", "abroad", "flight"],
}

# Conversation topic keywords for context extraction
TOPIC_KEYWORDS = {
    "academics": ["exam", "assignment", "project", "deadline", "study", "university", "college", "class", "lecture", "professor", "grade"],
    "career": ["internship", "job", "interview", "work", "office", "meeting", "project", "salary", "company"],
    "entertainment": ["movie", "series", "game", "music", "concert", "show", "watch", "play", "listen"],
    "food": ["food", "restaurant", "eat", "lunch", "dinner", "coffee", "cook", "recipe", "taste"],
    "travel": ["trip", "travel", "visit", "city", "flight", "hotel", "vacation"],
    "health": ["gym", "workout", "run", "sleep", "sick", "doctor", "health", "fit"],
}

# Re-engagement message templates (fallback when LLM unavailable)
FALLBACK_TEMPLATES = {
    "soft_nudge": [
        "Hey! It's been a while — how have things been going?",
        "Hey, just thinking about you. What's new on your end?",
        "Hope everything's been good! We should catch up sometime.",
    ],
    "hard_nudge": [
        "Hey! It's been {days} days since we last talked. Miss our conversations — how are you?",
        "It's been way too long! Hope things are going well. Would love to hear what you've been up to.",
        "Hey, just realized we haven't spoken in a while. Everything good with you?",
    ],
    "milestone": [
        "Just saw the news — huge congratulations! You deserve it.",
        "Heard the exciting news! So happy for you. Congrats!",
    ],
}
