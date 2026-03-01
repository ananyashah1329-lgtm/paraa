"""
Nexus Sentiment Analyzer
Uses a rule-based lexicon approach (VADER-inspired) for sentiment scoring.
Works without external NLP dependencies.
Falls back gracefully — designed for robustness.
"""

import re
import math
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging

from nexus.core.models import Message
from nexus.core.config import MILESTONE_KEYWORDS, TOPIC_KEYWORDS

logger = logging.getLogger(__name__)

# ─── Sentiment Lexicon ────────────────────────────────────────────────────────
# Polarity scores: -2.0 to +2.0

POSITIVE_WORDS = {
    # Strong positive
    "amazing": 2.0, "fantastic": 2.0, "wonderful": 2.0, "excellent": 1.8,
    "incredible": 1.8, "awesome": 1.7, "brilliant": 1.7, "love": 1.5,
    "perfect": 1.8, "outstanding": 1.8, "exceptional": 1.8,
    # Moderate positive
    "great": 1.3, "good": 1.0, "nice": 1.0, "happy": 1.3, "glad": 1.2,
    "pleased": 1.2, "excited": 1.5, "proud": 1.3, "grateful": 1.5,
    "thankful": 1.3, "blessed": 1.3, "enjoy": 1.1, "fun": 1.2,
    "funny": 1.0, "cool": 1.0, "sweet": 1.1, "cute": 1.0,
    "congratulations": 1.8, "congrats": 1.8, "celebrate": 1.5,
    "miss": 1.0, "smile": 1.2, "laugh": 1.2, "joy": 1.5, "hope": 0.8,
    # Mild positive
    "okay": 0.3, "fine": 0.3, "sure": 0.3, "thanks": 0.8, "thank": 0.8,
    "please": 0.3, "yes": 0.4, "yeah": 0.3, "yep": 0.3,
}

NEGATIVE_WORDS = {
    # Strong negative
    "terrible": -2.0, "horrible": -2.0, "awful": -1.8, "dreadful": -1.8,
    "hate": -1.8, "disgusting": -1.8, "disaster": -1.5, "worst": -1.8,
    "devastating": -1.7, "miserable": -1.7, "depressed": -1.7, "hopeless": -1.7,
    # Moderate negative
    "bad": -1.2, "sad": -1.2, "angry": -1.3, "upset": -1.2, "frustrated": -1.2,
    "disappointed": -1.3, "annoyed": -1.1, "worried": -1.0, "scared": -1.0,
    "anxious": -1.1, "stressed": -1.1, "tired": -0.8, "exhausted": -1.2,
    "sick": -1.0, "pain": -1.2, "hurt": -1.2, "lonely": -1.3,
    "miss you": 0.5,  # Override: "miss you" is positive
    # Mild negative
    "no": -0.4, "nope": -0.4, "nah": -0.3, "meh": -0.4, "ugh": -0.7,
    "boring": -0.8, "difficult": -0.6, "hard": -0.5, "rough": -0.7,
    "sorry": -0.3, "oops": -0.3, "unfortunately": -0.8,
    "never": -0.5, "can't": -0.4, "cannot": -0.4, "won't": -0.4,
}

# Intensifiers and negators
INTENSIFIERS = {
    "very": 1.4, "really": 1.3, "so": 1.2, "super": 1.4, "extremely": 1.6,
    "absolutely": 1.5, "totally": 1.3, "completely": 1.4, "utterly": 1.5,
    "quite": 1.1, "pretty": 1.1, "kinda": 0.8, "sorta": 0.8, "barely": 0.6,
}

NEGATORS = {"not", "no", "never", "neither", "nobody", "nothing", "nor",
            "hardly", "barely", "scarcely", "don't", "doesn't", "didn't",
            "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't"}

# Emoji sentiment
EMOJI_POSITIVE = {"😊", "😄", "😃", "😁", "🥰", "😍", "🤩", "🎉", "🎊", "❤️", "💕", "💖", "💗",
                  "👍", "🙌", "💪", "✨", "🌟", "⭐", "🤗", "😂", "😆", "🥳", "🎂", "🎁", "🏆", "👏"}
EMOJI_NEGATIVE = {"😢", "😭", "😞", "😔", "😟", "😣", "😩", "😫", "😤", "😠", "😡", "💔", "👎",
                  "😰", "😨", "😱", "🤦", "😒", "🙁", "☹️", "😑"}


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer."""
    text = text.lower()
    text = re.sub(r"[^\w\s'!?]", " ", text)
    return text.split()


def score_message(content: str) -> float:
    """
    Compute sentiment score for a single message.
    Returns float in range [-1.0, 1.0].
    """
    if not content or not content.strip():
        return 0.0

    # Emoji scoring
    emoji_score = 0.0
    for emoji in EMOJI_POSITIVE:
        if emoji in content:
            emoji_score += 0.4 * content.count(emoji)
    for emoji in EMOJI_NEGATIVE:
        if emoji in content:
            emoji_score -= 0.4 * content.count(emoji)

    tokens = _tokenize(content)
    if not tokens:
        return max(-1.0, min(1.0, emoji_score))

    word_scores = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        score = None

        # Check multi-word phrases first
        if i < len(tokens) - 1:
            bigram = token + " " + tokens[i+1]
            if bigram in POSITIVE_WORDS:
                score = POSITIVE_WORDS[bigram]
                i += 2
                word_scores.append(score)
                continue
            elif bigram in NEGATIVE_WORDS:
                score = NEGATIVE_WORDS[bigram]
                i += 2
                word_scores.append(score)
                continue

        # Single word
        if token in POSITIVE_WORDS:
            score = POSITIVE_WORDS[token]
        elif token in NEGATIVE_WORDS:
            score = NEGATIVE_WORDS[token]

        if score is not None:
            # Check for negator in previous 2 tokens
            context_start = max(0, i - 2)
            context = tokens[context_start:i]
            if any(t in NEGATORS for t in context):
                score = -score * 0.7

            # Check for intensifier in previous token
            if i > 0 and tokens[i-1] in INTENSIFIERS:
                score *= INTENSIFIERS[tokens[i-1]]

            word_scores.append(score)

        i += 1

    if not word_scores and emoji_score == 0:
        return 0.0

    # Aggregate: mean with emoji boost
    text_score = sum(word_scores) / max(len(word_scores), 1) if word_scores else 0.0

    # Exclamation marks boost positivity
    exclamation_count = content.count("!")
    if exclamation_count > 0 and text_score > 0:
        text_score *= (1 + 0.1 * min(exclamation_count, 3))

    # Combine text + emoji
    combined = text_score * 0.7 + emoji_score * 0.3

    # Normalize to [-1, 1]
    return max(-1.0, min(1.0, combined / 2.0))


def analyze_messages_sentiment(
    messages: List[Message],
) -> Tuple[List[Message], float, float]:
    """
    Analyze sentiment for a list of messages.

    Returns:
        (messages_with_scores, mean_sentiment, sentiment_trend)
    """
    scored = []
    time_sentiment_pairs = []

    for msg in messages:
        score = score_message(msg.content or "")
        msg.sentiment_score = score
        scored.append(msg)
        if msg.timestamp:
            time_sentiment_pairs.append((msg.timestamp, score))

    if not scored:
        return scored, 0.0, 0.0

    # Mean sentiment
    scores = [m.sentiment_score for m in scored if m.sentiment_score is not None]
    mean_sentiment = sum(scores) / len(scores) if scores else 0.0

    # Compute trend (slope of sentiment over time)
    from nexus.analysis.temporal_analyzer import compute_sentiment_trend
    trend = compute_sentiment_trend(time_sentiment_pairs)

    return scored, mean_sentiment, trend


def detect_milestones(messages: List[Message]) -> List[str]:
    """
    Scan messages for milestone keywords.
    Returns list of detected milestone types.
    """
    detected = set()
    for msg in messages:
        content_lower = (msg.content or "").lower()
        for milestone_type, keywords in MILESTONE_KEYWORDS.items():
            for kw in keywords:
                if kw in content_lower:
                    detected.add(milestone_type)
                    break
    return list(detected)


def extract_topics(messages: List[Message]) -> List[str]:
    """
    Extract top conversation topics from messages.
    Returns ordered list of topic labels.
    """
    topic_counts: Dict[str, int] = {}
    for msg in messages:
        content_lower = (msg.content or "").lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in content_lower:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    break

    # Sort by frequency
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_topics[:5]]
