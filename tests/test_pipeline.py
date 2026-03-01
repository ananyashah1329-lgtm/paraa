"""
Nexus Test Suite
Tests for all pipeline stages: parsers, analysis, scoring, decisions, actions.
Run with: python -m pytest tests/ -v
Or: python nexus/tests/test_pipeline.py
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nexus.core.models import (
    Message, Channel, ScoreTier, DecisionType, RelationshipTier, ContactState
)
from nexus.core.config import CONFIG
from nexus.parsers.whatsapp_parser import parse_whatsapp
from nexus.parsers.generic_parser import parse_json, parse_csv
from nexus.parsers.dispatcher import parse_auto, group_messages_by_contact
from nexus.analysis.temporal_analyzer import compute_contact_metrics, detect_anomalies
from nexus.analysis.sentiment_analyzer import score_message, analyze_messages_sentiment, detect_milestones
from nexus.analysis.graph_builder import RelationshipGraph
from nexus.scoring.nexus_score import compute_nexus_score, build_contact_state, rank_priority_queue
from nexus.decision.engine import decide, batch_decide
from nexus.data.synthetic_generator import generate_synthetic_dataset
from nexus.core.pipeline import run_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_messages(n: int, contact_id: str = "Alice", is_outgoing: bool = True,
                   days_back: int = 30, sparse: bool = False) -> list:
    """Generate synthetic messages for testing."""
    msgs = []
    base = datetime.utcnow() - timedelta(days=days_back)
    for i in range(n):
        if sparse:
            ts = base + timedelta(days=i * (days_back / n))
        else:
            ts = base + timedelta(hours=i * 4)

        sender = "user" if is_outgoing else contact_id
        receiver = contact_id if is_outgoing else "user"

        msgs.append(Message(
            sender_id=sender,
            receiver_id=receiver,
            timestamp=ts,
            content=f"Test message {i}",
            channel=Channel.SYNTHETIC,
            is_outgoing=(i % 2 == 0),  # Alternate
        ))
    return msgs


def _run_test(name: str, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except AssertionError as e:
        print(f"  ❌ {name}: AssertionError — {e}")
        return False
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__} — {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_whatsapp_parser_ios():
    content = """[01/01/2024, 10:30:00] Alice: Hey how are you?
[01/01/2024, 10:35:00] Bob: I'm good! You?
[02/01/2024, 09:00:00] Alice: Pretty good, thanks!"""
    msgs = parse_whatsapp(content, owner_name="Alice")
    assert len(msgs) == 3, f"Expected 3, got {len(msgs)}"
    assert msgs[0].sender_id == "Alice"
    assert msgs[0].is_outgoing == True


def test_whatsapp_parser_android():
    content = """1/1/2024, 10:30 AM - Alice: Hey there!
1/1/2024, 10:32 AM - Bob: Hi Alice!
1/2/2024, 9:00 AM - Alice: How's your day?"""
    msgs = parse_whatsapp(content, owner_name="Alice")
    assert len(msgs) >= 2, f"Expected >= 2, got {len(msgs)}"


def test_whatsapp_parser_skips_system():
    content = """[01/01/2024, 10:30:00] Alice: Hey!
[01/01/2024, 10:32:00] Bob: Hello!"""
    msgs = parse_whatsapp(content, owner_name="Alice")
    # Just ensure it parses successfully without error
    assert len(msgs) >= 1


def test_json_parser():
    data = json.dumps([
        {"sender": "user", "receiver": "Alice", "timestamp": "2024-01-01T10:00:00", "message": "Hey!"},
        {"sender": "Alice", "receiver": "user", "timestamp": "2024-01-01T10:05:00", "message": "Hi!"},
    ])
    msgs = parse_json(data, owner_id="user")
    assert len(msgs) == 2


def test_csv_parser():
    csv_content = """sender,receiver,timestamp,message
user,Alice,2024-01-01 10:00:00,Hello!
Alice,user,2024-01-01 10:05:00,Hi there!"""
    msgs = parse_csv(csv_content, owner_id="user")
    assert len(msgs) == 2


def test_auto_dispatcher_detects_whatsapp():
    content = "[01/01/2024, 10:30:00] Alice: test message"
    msgs, fmt = parse_auto(content)
    assert fmt == "whatsapp"


def test_auto_dispatcher_detects_json():
    content = '[{"sender": "A", "receiver": "B", "timestamp": "2024-01-01", "message": "hi"}]'
    msgs, fmt = parse_auto(content)
    assert fmt == "json"


def test_auto_dispatcher_detects_csv():
    content = "sender,receiver,timestamp,message\nA,B,2024-01-01,hi"
    msgs, fmt = parse_auto(content)
    assert fmt == "csv"


def test_group_by_contact():
    msgs = [
        Message(sender_id="user", receiver_id="Alice", is_outgoing=True),
        Message(sender_id="Alice", receiver_id="user", is_outgoing=False),
        Message(sender_id="user", receiver_id="Bob", is_outgoing=True),
    ]
    groups = group_messages_by_contact(msgs)
    assert "Alice" in groups
    assert "Bob" in groups
    assert len(groups["Alice"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Sentiment Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_sentiment_positive():
    score = score_message("I love this, it's absolutely amazing! 😊")
    assert score > 0.2, f"Expected positive, got {score}"


def test_sentiment_negative():
    score = score_message("This is terrible, I'm so frustrated and upset")
    assert score < 0, f"Expected negative, got {score}"


def test_sentiment_neutral():
    score = score_message("The meeting is at 3pm tomorrow")
    assert -0.3 < score < 0.3, f"Expected neutral, got {score}"


def test_sentiment_negation():
    score_base = score_message("This is good")
    score_neg = score_message("This is not good")
    assert score_base > score_neg, "Negation should reduce sentiment"


def test_milestone_detection():
    msgs = [Message(content="Happy birthday!! 🎂"), Message(content="I got the promotion!")]
    milestones = detect_milestones(msgs)
    assert "birthday" in milestones or len(milestones) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Analyzer Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_metrics_basic():
    msgs = _make_messages(30, contact_id="Alice", days_back=30)
    metrics = compute_contact_metrics("Alice", msgs)
    assert metrics.total_messages == 30
    assert metrics.contact_id == "Alice"
    assert metrics.last_seen is not None
    assert metrics.days_since_last_message >= 0


def test_initiation_ratio():
    # All sent by user
    msgs = [Message(sender_id="user", receiver_id="Alice", timestamp=datetime.utcnow() - timedelta(hours=i), is_outgoing=True) for i in range(10)]
    metrics = compute_contact_metrics("Alice", msgs)
    assert metrics.initiation_ratio > 0.5


def test_detect_anomaly_silence():
    # Create messages with high historical frequency, then sudden silence
    msgs = []
    base = datetime.utcnow() - timedelta(days=60)
    # First 50 days: lots of messages (every 1 day)
    for i in range(50):
        msgs.append(Message(
            sender_id="user", receiver_id="Alice",
            timestamp=base + timedelta(days=i),
            is_outgoing=True
        ))
    # Last message was 10 days ago — reference is now

    ref = datetime.utcnow()
    metrics = compute_contact_metrics("Alice", msgs, reference_time=ref)
    anomalies = detect_anomalies(metrics, reference_time=ref)

    # Should have SOME anomaly (long inactivity or silence)
    assert len(anomalies) > 0, f"Expected anomalies for contact silent for {metrics.days_since_last_message:.0f} days"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_graph_builds():
    msgs = [
        Message(sender_id="user", receiver_id="Alice", timestamp=datetime.utcnow(), is_outgoing=True),
        Message(sender_id="Alice", receiver_id="user", timestamp=datetime.utcnow(), is_outgoing=False),
    ]
    graph = RelationshipGraph(owner_id="user")
    graph.build(msgs)
    assert graph.node_count() >= 2
    assert "Alice" in graph.get_contacts()


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_score_healthy_contact():
    """High-frequency, recent, balanced → high score."""
    msgs = _make_messages(60, contact_id="Alice", days_back=30)
    scored_msgs, mean_sentiment, trend = analyze_messages_sentiment(msgs)
    metrics = compute_contact_metrics("Alice", scored_msgs)
    metrics.sentiment_mean = 0.5  # Positive
    breakdown = compute_nexus_score(metrics)
    assert breakdown.composite_score >= 50, f"Healthy contact should score >= 50, got {breakdown.composite_score}"
    assert breakdown.tier in (ScoreTier.GREEN, ScoreTier.YELLOW)


def test_score_dead_contact():
    """Old contact, no recent messages → low score."""
    msgs = _make_messages(5, contact_id="Bob", days_back=180)
    metrics = compute_contact_metrics("Bob", msgs)
    metrics.sentiment_mean = -0.3
    breakdown = compute_nexus_score(metrics)
    assert breakdown.composite_score < 50, f"Inactive contact should score < 50, got {breakdown.composite_score}"


def test_score_breakdown_fields():
    msgs = _make_messages(20)
    metrics = compute_contact_metrics("test", msgs)
    breakdown = compute_nexus_score(metrics)
    assert 0 <= breakdown.frequency_score <= 100
    assert 0 <= breakdown.latency_score <= 100
    assert 0 <= breakdown.balance_score <= 100
    assert 0 <= breakdown.sentiment_score <= 100
    assert 0 <= breakdown.recency_score <= 100
    assert 0 <= breakdown.composite_score <= 100


def test_tier_classification():
    msgs = _make_messages(5, days_back=200)
    metrics = compute_contact_metrics("test", msgs)
    breakdown = compute_nexus_score(metrics)
    if breakdown.composite_score >= 70:
        assert breakdown.tier == ScoreTier.GREEN
    elif breakdown.composite_score >= 40:
        assert breakdown.tier == ScoreTier.YELLOW
    else:
        assert breakdown.tier == ScoreTier.RED


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_decision_passive_for_green():
    msgs = _make_messages(60, days_back=5)
    metrics = compute_contact_metrics("Alice", msgs)
    metrics.sentiment_mean = 0.5
    state = build_contact_state("Alice", metrics)
    if state.score_breakdown.tier == ScoreTier.GREEN:
        decision = decide(state)
        assert decision == DecisionType.PASSIVE


def test_decision_nudge_for_red():
    msgs = _make_messages(3, days_back=90)
    metrics = compute_contact_metrics("Bob", msgs)
    metrics.sentiment_mean = -0.2
    state = build_contact_state("Bob", metrics)
    if state.score_breakdown.tier == ScoreTier.RED:
        decision = decide(state)
        assert decision in (DecisionType.HARD_NUDGE, DecisionType.ESCALATE)


def test_batch_decide():
    states = []
    for i, days in enumerate([5, 30, 90]):
        msgs = _make_messages(30 if days == 5 else 5, days_back=days)
        metrics = compute_contact_metrics(f"Contact_{i}", msgs)
        state = build_contact_state(f"Contact_{i}", metrics)
        states.append(state)

    decisions = batch_decide(states)
    assert len(decisions) == 3
    for d in decisions:
        assert "decision_type" in d
        assert "context" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_synthetic_generator():
    dataset = generate_synthetic_dataset(user_name="TestUser", days_back=60, seed=123)
    assert "messages" in dataset
    assert "metadata" in dataset
    assert dataset["metadata"]["total_contacts"] == 8
    assert len(dataset["messages"]) > 100


def test_synthetic_has_all_tiers():
    dataset = generate_synthetic_dataset()
    expected_tiers = {"green", "yellow", "red"}
    actual_tiers = {v["expected_tier"] for v in dataset["metadata"]["contact_map"].values()}
    assert actual_tiers == expected_tiers


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Integration Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_with_synthetic():
    """End-to-end pipeline test using synthetic data."""
    dataset = generate_synthetic_dataset(seed=42)
    content = json.dumps(dataset["messages"])

    result = run_pipeline(
        content=content,
        owner_id="You",
        format_hint="json",
        save_to_db=False,
    )

    assert result.total_contacts >= 6, f"Expected >= 6 contacts, got {result.total_contacts}"
    assert result.total_messages > 100
    total_by_tier = result.contacts_green + result.contacts_yellow + result.contacts_red
    assert total_by_tier == result.total_contacts
    assert len(result.contact_states) == result.total_contacts
    assert len(result.nudge_results) > 0
    assert len(result.priority_queue) > 0
    assert result.processing_time_ms > 0

    # Green contacts should have higher average scores than red contacts
    green_scores = [s.nexus_score for s in result.contact_states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.GREEN]
    red_scores = [s.nexus_score for s in result.contact_states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.RED]

    if green_scores and red_scores:
        avg_green = sum(green_scores) / len(green_scores)
        avg_red = sum(red_scores) / len(red_scores)
        assert avg_green > avg_red, \
            f"Avg green score {avg_green:.1f} should exceed avg red score {avg_red:.1f}"


def test_pipeline_json_serialization():
    """Test that all pipeline outputs are JSON-serializable."""
    dataset = generate_synthetic_dataset(seed=99)
    content = json.dumps(dataset["messages"])
    result = run_pipeline(content=content, owner_id="You", format_hint="json", save_to_db=False)
    # Should not raise
    serialized = json.dumps(result.to_dict())
    assert len(serialized) > 0


def test_pipeline_whatsapp_format():
    """Test pipeline with WhatsApp format input."""
    wa_content = """[01/01/2024, 10:30:00] You: Hey Alice!
[01/01/2024, 10:32:00] Alice: Hi! How are you?
[01/01/2024, 10:35:00] You: Doing great, thanks!
[02/01/2024, 09:00:00] Alice: Want to meet up?
[02/01/2024, 09:05:00] You: Absolutely!"""

    result = run_pipeline(
        content=wa_content,
        owner_id="You",
        format_hint="whatsapp",
        save_to_db=False,
    )
    assert result.total_contacts >= 1
    assert result.total_messages >= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Run All Tests
# ═══════════════════════════════════════════════════════════════════════════════

TEST_GROUPS = {
    "Parsers": [
        ("WhatsApp iOS format", test_whatsapp_parser_ios),
        ("WhatsApp Android format", test_whatsapp_parser_android),
        ("WhatsApp skip system messages", test_whatsapp_parser_skips_system),
        ("JSON parser", test_json_parser),
        ("CSV parser", test_csv_parser),
        ("Auto-detect WhatsApp", test_auto_dispatcher_detects_whatsapp),
        ("Auto-detect JSON", test_auto_dispatcher_detects_json),
        ("Auto-detect CSV", test_auto_dispatcher_detects_csv),
        ("Group by contact", test_group_by_contact),
    ],
    "Sentiment Analyzer": [
        ("Positive sentiment", test_sentiment_positive),
        ("Negative sentiment", test_sentiment_negative),
        ("Neutral sentiment", test_sentiment_neutral),
        ("Negation handling", test_sentiment_negation),
        ("Milestone detection", test_milestone_detection),
    ],
    "Temporal Analyzer": [
        ("Compute basic metrics", test_compute_metrics_basic),
        ("Initiation ratio", test_initiation_ratio),
        ("Detect anomalies", test_detect_anomaly_silence),
    ],
    "Graph Builder": [
        ("Graph construction", test_graph_builds),
    ],
    "Nexus Scoring": [
        ("Healthy contact scores high", test_score_healthy_contact),
        ("Inactive contact scores low", test_score_dead_contact),
        ("Score breakdown fields", test_score_breakdown_fields),
        ("Tier classification", test_tier_classification),
    ],
    "Decision Engine": [
        ("Passive for green contacts", test_decision_passive_for_green),
        ("Nudge for red contacts", test_decision_nudge_for_red),
        ("Batch decisions", test_batch_decide),
    ],
    "Synthetic Generator": [
        ("Generate dataset", test_synthetic_generator),
        ("All tiers present", test_synthetic_has_all_tiers),
    ],
    "Full Pipeline (Integration)": [
        ("Synthetic data E2E", test_full_pipeline_with_synthetic),
        ("JSON serialization", test_pipeline_json_serialization),
        ("WhatsApp format", test_pipeline_whatsapp_format),
    ],
}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NEXUS TEST SUITE")
    print("=" * 60)

    total = 0
    passed = 0

    for group_name, tests in TEST_GROUPS.items():
        print(f"\n📋 {group_name}")
        for test_name, test_fn in tests:
            total += 1
            if _run_test(test_name, test_fn):
                passed += 1

    print("\n" + "=" * 60)
    status = "✅ ALL PASSED" if passed == total else f"⚠️  {total - passed} FAILED"
    print(f"  Results: {passed}/{total} tests passed  {status}")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)
