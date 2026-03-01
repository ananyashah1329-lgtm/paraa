"""
Nexus Pipeline Orchestrator
The main entry point — wires all 4 pipeline stages together.

Stage 1 — INGEST: Parse and normalize messages
Stage 2 — ANALYZE: Compute metrics, sentiment, build graph
Stage 3 — DECIDE: Score contacts and determine actions
Stage 4 — ACT: Generate nudges and assemble results
"""

import time
from datetime import datetime
from typing import Optional, List, Dict
import logging

from nexus.core.models import (
    PipelineResult, ContactState, RelationshipTier
)
from nexus.parsers.dispatcher import parse_auto, group_messages_by_contact
from nexus.analysis.temporal_analyzer import compute_contact_metrics, detect_anomalies
from nexus.analysis.sentiment_analyzer import (
    analyze_messages_sentiment, detect_milestones, extract_topics
)
from nexus.analysis.graph_builder import RelationshipGraph
from nexus.scoring.nexus_score import build_contact_state, rank_priority_queue
from nexus.decision.engine import batch_decide
from nexus.actions.nudge_generator import generate_batch_nudges, generate_weekly_digest
from nexus.utils.state_store import store

logger = logging.getLogger(__name__)


def run_pipeline(
    content: str,
    filename: Optional[str] = None,
    owner_id: Optional[str] = None,
    format_hint: Optional[str] = None,
    api_key: Optional[str] = None,
    contact_tiers: Optional[Dict[str, str]] = None,
    reference_time: Optional[datetime] = None,
    save_to_db: bool = True,
) -> PipelineResult:
    """
    Run the full Nexus pipeline on communication data.

    Args:
        content: Raw text/JSON/CSV content of chat logs
        filename: Optional filename for format detection
        owner_id: The user's identifier in the logs
        format_hint: Optional format override ('whatsapp'/'json'/'csv'/'email')
        api_key: Anthropic API key for LLM nudge generation
        contact_tiers: Optional dict mapping contact_id -> tier ('close'/'acquaintance'/'professional')
        reference_time: The "now" reference for recency computations
        save_to_db: Whether to persist results to SQLite

    Returns:
        PipelineResult with all states, scores, nudges, and priority queue
    """
    start_time = time.time()
    result = PipelineResult()

    if not reference_time:
        reference_time = datetime.utcnow()

    logger.info("=" * 60)
    logger.info("NEXUS PIPELINE STARTING")
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: INGEST
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 1] Ingesting and parsing messages...")

    messages, detected_format = parse_auto(
        content,
        filename=filename,
        owner_id=owner_id,
        format_hint=format_hint,
    )

    if not messages:
        logger.warning("No messages parsed — check format and content")
        result.pipeline_metadata["error"] = "No messages parsed"
        result.pipeline_metadata["detected_format"] = detected_format
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    logger.info(f"[Stage 1] Parsed {len(messages)} messages (format: {detected_format})")

    # Group by contact
    contact_groups = group_messages_by_contact(messages, user_id=owner_id)
    logger.info(f"[Stage 1] Identified {len(contact_groups)} contacts")

    result.total_messages = len(messages)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: ANALYZE
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 2] Running behavioral analysis...")

    # Build relationship graph
    graph = RelationshipGraph(owner_id=owner_id or "user")
    graph.build(messages, reference_time=reference_time)
    logger.info(f"[Stage 2] Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

    # Per-contact analysis
    all_states: List[ContactState] = []
    states_map: Dict[str, ContactState] = {}

    for contact_id, contact_messages in contact_groups.items():
        logger.debug(f"[Stage 2] Analyzing: {contact_id} ({len(contact_messages)} messages)")

        # Sentiment analysis
        scored_messages, sentiment_mean, sentiment_trend = analyze_messages_sentiment(contact_messages)

        # Temporal metrics
        metrics = compute_contact_metrics(
            contact_id=contact_id,
            messages=scored_messages,
            reference_time=reference_time,
        )
        metrics.sentiment_mean = sentiment_mean
        metrics.sentiment_trend = sentiment_trend
        metrics.contact_name = contact_id

        # Milestones and topics
        metrics.detected_milestones = detect_milestones(scored_messages)
        metrics.last_topics = extract_topics(scored_messages)

        # Anomaly detection
        anomalies = detect_anomalies(metrics, reference_time=reference_time)

        # Load previous state if exists
        previous_state = store.get_contact_state(contact_id) if save_to_db else None

        # Determine relationship tier
        rel_tier = RelationshipTier.UNKNOWN
        if contact_tiers and contact_id in contact_tiers:
            try:
                rel_tier = RelationshipTier(contact_tiers[contact_id])
            except ValueError:
                pass
        elif previous_state:
            rel_tier = previous_state.relationship_tier

        # ────────────────────────────────────────────────────────────────────
        # STAGE 3: SCORE
        # ────────────────────────────────────────────────────────────────────
        state = build_contact_state(
            contact_id=contact_id,
            metrics=metrics,
            previous_state=previous_state,
            relationship_tier=rel_tier,
        )
        state.anomalies = anomalies

        all_states.append(state)
        states_map[contact_id] = state

        logger.info(
            f"[Stage 3] Score [{contact_id}]: {state.nexus_score:.1f} "
            f"[{state.score_breakdown.tier.value}]"
        )

    # Aggregate tier counts
    result.total_contacts = len(all_states)
    result.contacts_green = sum(1 for s in all_states if s.score_breakdown and s.score_breakdown.tier.value == "green")
    result.contacts_yellow = sum(1 for s in all_states if s.score_breakdown and s.score_breakdown.tier.value == "yellow")
    result.contacts_red = sum(1 for s in all_states if s.score_breakdown and s.score_breakdown.tier.value == "red")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: DECIDE + ACT
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 4] Running decision engine and generating nudges...")

    decisions = batch_decide(all_states)
    nudge_results = generate_batch_nudges(decisions, states_map, api_key=api_key)

    logger.info(f"[Stage 4] Generated {len(nudge_results)} nudge results")

    # Priority queue
    priority_queue = rank_priority_queue(all_states)

    # ──────────────────────────────────────────────────────────────────────────
    # ASSEMBLE RESULT
    # ──────────────────────────────────────────────────────────────────────────
    result.session_id = result.session_id
    result.contact_states = all_states
    result.nudge_results = nudge_results
    result.priority_queue = priority_queue
    result.processing_time_ms = (time.time() - start_time) * 1000
    result.pipeline_metadata = {
        "detected_format": detected_format,
        "owner_id": owner_id,
        "reference_time": reference_time.isoformat(),
        "graph_nodes": graph.node_count(),
        "graph_edges": graph.edge_count(),
        "anomalies_detected": sum(len(s.anomalies) for s in all_states),
        "nudges_generated": len([n for n in nudge_results if n.message_draft]),
        "llm_used": any(n.llm_used for n in nudge_results),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # PERSIST
    # ──────────────────────────────────────────────────────────────────────────
    if save_to_db:
        for state in all_states:
            store.save_contact_state(state, session_id=result.session_id)
            if state.score_breakdown:
                store.record_score(
                    state.contact_id,
                    state.nexus_score,
                    state.score_breakdown.tier.value,
                    session_id=result.session_id,
                )

        for nudge in nudge_results:
            if nudge.message_draft:
                store.log_nudge(nudge, session_id=result.session_id)

        store.save_session(result)

    logger.info("=" * 60)
    logger.info(
        f"NEXUS PIPELINE COMPLETE | "
        f"{result.total_contacts} contacts | "
        f"🟢{result.contacts_green} 🟡{result.contacts_yellow} 🔴{result.contacts_red} | "
        f"{result.processing_time_ms:.0f}ms"
    )
    logger.info("=" * 60)

    return result


def run_pipeline_from_file(
    file_path: str,
    owner_id: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """Convenience wrapper to run pipeline from a file path."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    filename = file_path.split("/")[-1]
    return run_pipeline(content, filename=filename, owner_id=owner_id, api_key=api_key, **kwargs)
