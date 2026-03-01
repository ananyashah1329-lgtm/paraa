"""
Nexus Flask REST API
All endpoints for the frontend to consume.

Endpoints:
  POST /api/pipeline/run          - Run full pipeline on uploaded data
  POST /api/pipeline/synthetic    - Run pipeline on generated synthetic data
  GET  /api/contacts              - List all contact states
  GET  /api/contacts/<id>         - Get single contact state
  GET  /api/contacts/<id>/score-history  - Score history for contact
  GET  /api/contacts/<id>/nudge-history  - Nudge history for contact
  POST /api/contacts/<id>/tier    - Update relationship tier
  POST /api/contacts/<id>/feedback - Submit nudge feedback
  POST /api/nudge/generate        - Generate on-demand nudge
  GET  /api/dashboard             - Dashboard summary
  GET  /api/priority-queue        - Get at-risk contacts
  GET  /api/digest/weekly         - Weekly relationship digest
  GET  /api/sessions              - Recent pipeline sessions
  GET  /api/health                - Health check
  POST /api/reset                 - Reset all state (demo use)
"""

import os
import json
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, Response

from nexus.core.pipeline import run_pipeline
from nexus.core.models import RelationshipTier, ScoreTier
from nexus.data.synthetic_generator import generate_synthetic_dataset
from nexus.actions.nudge_generator import generate_nudge, generate_weekly_digest
from nexus.decision.engine import apply_feedback, decide, build_decision_context
from nexus.utils.state_store import store

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexus.api")


# ─── CORS Middleware ──────────────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return Response(status=200)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _success(data: dict, status: int = 200) -> Response:
    return jsonify({"success": True, "data": data}), status


def _error(message: str, status: int = 400) -> Response:
    return jsonify({"success": False, "error": message}), status


def _get_api_key() -> str:
    """Extract API key from request headers or body."""
    return (
        request.headers.get("X-API-Key") or
        request.headers.get("Authorization", "").replace("Bearer ", "") or
        (request.get_json(silent=True) or {}).get("api_key") or
        os.environ.get("ANTHROPIC_API_KEY", "")
    )


# ─── Pipeline Endpoints ───────────────────────────────────────────────────────

@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline_endpoint():
    """
    Run the full Nexus pipeline on uploaded chat data.

    Request: multipart/form-data
      - file: chat log file (txt/json/csv)
      - OR content: raw text content (JSON body)
    
    Optional fields:
      - owner_id: string — user's name/ID in the chat
      - format: string — format hint (whatsapp/json/csv/email)
      - contact_tiers: JSON object mapping contact_id -> tier
      - api_key: Anthropic API key for LLM nudge generation
    """
    try:
        api_key = _get_api_key()
        owner_id = None
        content = None
        filename = None
        format_hint = None
        contact_tiers = {}

        # Handle multipart file upload
        if request.files.get("file"):
            f = request.files["file"]
            filename = f.filename
            content = f.read().decode("utf-8", errors="replace")
            owner_id = request.form.get("owner_id")
            format_hint = request.form.get("format")
            tiers_json = request.form.get("contact_tiers", "{}")
            try:
                contact_tiers = json.loads(tiers_json)
            except json.JSONDecodeError:
                contact_tiers = {}

        # Handle JSON body
        elif request.is_json:
            body = request.get_json()
            content = body.get("content") or body.get("data")
            owner_id = body.get("owner_id")
            format_hint = body.get("format")
            contact_tiers = body.get("contact_tiers", {})
            filename = body.get("filename")

        if not content:
            return _error("No chat data provided. Send file or content field.")

        logger.info(f"Pipeline run: owner={owner_id}, format={format_hint}, size={len(content)} chars")

        result = run_pipeline(
            content=content,
            filename=filename,
            owner_id=owner_id,
            format_hint=format_hint,
            api_key=api_key,
            contact_tiers=contact_tiers,
            save_to_db=True,
        )

        return _success(result.to_dict())

    except Exception as e:
        logger.exception("Pipeline run failed")
        return _error(f"Pipeline error: {str(e)}", 500)


@app.route("/api/pipeline/synthetic", methods=["POST"])
def run_synthetic_pipeline():
    """
    Run pipeline on auto-generated synthetic data.
    Great for demos and testing.

    Optional JSON body:
      - user_name: string (default: "You")
      - days_back: int (default: 90)
      - seed: int (default: 42)
      - api_key: Anthropic API key
    """
    try:
        body = request.get_json(silent=True) or {}
        api_key = _get_api_key()

        user_name = body.get("user_name", "You")
        days_back = int(body.get("days_back", 90))
        seed = int(body.get("seed", 42))

        logger.info(f"Generating synthetic dataset: user={user_name}, days={days_back}, seed={seed}")

        dataset = generate_synthetic_dataset(
            user_name=user_name,
            days_back=days_back,
            seed=seed,
        )

        content = json.dumps(dataset["messages"])

        result = run_pipeline(
            content=content,
            owner_id=user_name,
            format_hint="json",
            api_key=api_key,
            save_to_db=True,
        )

        response_data = result.to_dict()
        response_data["synthetic_metadata"] = dataset["metadata"]
        return _success(response_data)

    except Exception as e:
        logger.exception("Synthetic pipeline failed")
        return _error(f"Synthetic pipeline error: {str(e)}", 500)


# ─── Contact Endpoints ────────────────────────────────────────────────────────

@app.route("/api/contacts", methods=["GET"])
def list_contacts():
    """List all contact states, optionally filtered by tier."""
    try:
        tier_filter = request.args.get("tier")  # green/yellow/red
        states = store.get_all_contact_states()

        if tier_filter:
            states = [
                s for s in states
                if s.score_breakdown and s.score_breakdown.tier.value == tier_filter
            ]

        # Sort by score ascending (most at-risk first)
        states.sort(key=lambda s: s.nexus_score)

        return _success({
            "total": len(states),
            "contacts": [s.to_dict() for s in states],
        })
    except Exception as e:
        logger.exception("List contacts failed")
        return _error(str(e), 500)


@app.route("/api/contacts/<contact_id>", methods=["GET"])
def get_contact(contact_id: str):
    """Get full state for a single contact."""
    state = store.get_contact_state(contact_id)
    if not state:
        return _error(f"Contact '{contact_id}' not found", 404)
    return _success(state.to_dict())


@app.route("/api/contacts/<contact_id>/score-history", methods=["GET"])
def get_score_history(contact_id: str):
    """Get score history for a contact."""
    limit = int(request.args.get("limit", 30))
    history = store.get_score_history(contact_id, limit=limit)
    return _success({"contact_id": contact_id, "history": history})


@app.route("/api/contacts/<contact_id>/nudge-history", methods=["GET"])
def get_nudge_history(contact_id: str):
    """Get nudge history for a contact."""
    limit = int(request.args.get("limit", 10))
    history = store.get_nudge_history(contact_id, limit=limit)
    return _success({"contact_id": contact_id, "history": history})


@app.route("/api/contacts/<contact_id>/tier", methods=["POST"])
def update_tier(contact_id: str):
    """
    Update a contact's relationship tier.
    Body: {"tier": "close"|"acquaintance"|"professional"}
    """
    state = store.get_contact_state(contact_id)
    if not state:
        return _error(f"Contact '{contact_id}' not found", 404)

    body = request.get_json(silent=True) or {}
    tier_str = body.get("tier", "").lower()

    try:
        state.relationship_tier = RelationshipTier(tier_str)
    except ValueError:
        return _error(f"Invalid tier '{tier_str}'. Must be: close, acquaintance, professional")

    store.save_contact_state(state)
    return _success({"contact_id": contact_id, "tier": state.relationship_tier.value})


@app.route("/api/contacts/<contact_id>/feedback", methods=["POST"])
def submit_feedback(contact_id: str):
    """
    Submit feedback for the most recent nudge for a contact.
    Body: {"feedback": "accepted"|"dismissed"|"snoozed"}
    """
    state = store.get_contact_state(contact_id)
    if not state:
        return _error(f"Contact '{contact_id}' not found", 404)

    body = request.get_json(silent=True) or {}
    feedback = body.get("feedback", "").lower()

    if feedback not in ("accepted", "dismissed", "snoozed"):
        return _error("feedback must be 'accepted', 'dismissed', or 'snoozed'")

    # Apply feedback to state
    state = apply_feedback(state, feedback)
    store.save_contact_state(state)
    store.update_nudge_feedback(contact_id, feedback)

    logger.info(f"Feedback [{contact_id}]: {feedback}")
    return _success({
        "contact_id": contact_id,
        "feedback": feedback,
        "updated_score": round(state.nexus_score, 1),
    })


# ─── Nudge Endpoints ──────────────────────────────────────────────────────────

@app.route("/api/nudge/generate", methods=["POST"])
def generate_nudge_endpoint():
    """
    Generate an on-demand nudge for a specific contact.
    
    Body:
      - contact_id: string (required)
      - decision_type: optional override (soft_nudge/hard_nudge/milestone)
      - api_key: optional Anthropic API key
    """
    try:
        body = request.get_json(silent=True) or {}
        contact_id = body.get("contact_id")
        api_key = _get_api_key()

        if not contact_id:
            return _error("contact_id is required")

        state = store.get_contact_state(contact_id)
        if not state:
            return _error(f"Contact '{contact_id}' not found", 404)

        # Determine decision type
        from nexus.core.models import DecisionType
        decision_str = body.get("decision_type")
        if decision_str:
            try:
                decision_type = DecisionType(decision_str)
            except ValueError:
                return _error(f"Invalid decision_type: {decision_str}")
        else:
            decision_type = decide(state)

        nudge = generate_nudge(state, decision_type, api_key=api_key)
        store.save_contact_state(state)
        store.log_nudge(nudge)

        return _success(nudge.to_dict())

    except Exception as e:
        logger.exception("Nudge generation failed")
        return _error(str(e), 500)


# ─── Dashboard & Analytics ────────────────────────────────────────────────────

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    """
    Get dashboard summary: tier distribution, top contacts, recent activity.
    """
    try:
        states = store.get_all_contact_states()

        green = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.GREEN]
        yellow = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.YELLOW]
        red = [s for s in states if s.score_breakdown and s.score_breakdown.tier == ScoreTier.RED]

        avg_score = sum(s.nexus_score for s in states) / len(states) if states else 0

        # Top healthy relationships
        top_healthy = sorted(green, key=lambda s: -s.nexus_score)[:3]

        # Most urgent (red + lowest score)
        most_urgent = sorted(red, key=lambda s: s.nexus_score)[:3]

        # Recent sessions
        sessions = store.get_sessions(limit=5)

        return _success({
            "summary": {
                "total_contacts": len(states),
                "contacts_green": len(green),
                "contacts_yellow": len(yellow),
                "contacts_red": len(red),
                "average_nexus_score": round(avg_score, 1),
                "health_percentage": round(len(green) / len(states) * 100, 0) if states else 0,
            },
            "top_healthy": [
                {"name": s.contact_name, "score": round(s.nexus_score, 1)} for s in top_healthy
            ],
            "most_urgent": [
                {
                    "name": s.contact_name,
                    "score": round(s.nexus_score, 1),
                    "days_silent": round(s.metrics.days_since_last_message, 0) if s.metrics else None,
                }
                for s in most_urgent
            ],
            "score_distribution": _compute_score_distribution(states),
            "recent_sessions": sessions,
            "db_stats": store.get_stats(),
        })

    except Exception as e:
        logger.exception("Dashboard failed")
        return _error(str(e), 500)


def _compute_score_distribution(states) -> list:
    """Compute histogram of scores in 10-point buckets."""
    buckets = {f"{i}-{i+10}": 0 for i in range(0, 100, 10)}
    for s in states:
        bucket_idx = min(int(s.nexus_score // 10) * 10, 90)
        key = f"{bucket_idx}-{bucket_idx+10}"
        buckets[key] = buckets.get(key, 0) + 1
    return [{"range": k, "count": v} for k, v in buckets.items()]


@app.route("/api/priority-queue", methods=["GET"])
def priority_queue():
    """Get ranked priority queue of at-risk contacts."""
    try:
        from nexus.scoring.nexus_score import rank_priority_queue
        states = store.get_all_contact_states()
        queue = rank_priority_queue(states)
        return _success({"priority_queue": queue, "total": len(queue)})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/digest/weekly", methods=["GET"])
def weekly_digest():
    """Generate weekly relationship health digest."""
    try:
        states = store.get_all_contact_states()
        digest = generate_weekly_digest(states)
        return _success(digest)
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List recent pipeline sessions."""
    limit = int(request.args.get("limit", 10))
    sessions = store.get_sessions(limit=limit)
    return _success({"sessions": sessions, "total": len(sessions)})


# ─── Utility Endpoints ────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """API health check."""
    return _success({
        "status": "healthy",
        "version": "1.0.0",
        "service": "Nexus Relationship Intelligence API",
        "timestamp": datetime.utcnow().isoformat(),
        "db_stats": store.get_stats(),
    })


@app.route("/api/reset", methods=["POST"])
def reset_state():
    """
    Reset all state. Use for demo restarts.
    Body: {"confirm": true}
    """
    body = request.get_json(silent=True) or {}
    if not body.get("confirm"):
        return _error("Send {'confirm': true} to reset all state")

    store.clear_all_states()
    logger.warning("All state cleared via API reset")
    return _success({"message": "All state cleared successfully"})


@app.route("/api/config/weights", methods=["GET"])
def get_weights():
    """Get current scoring weights."""
    from nexus.core.config import CONFIG
    w = CONFIG.scoring.weights
    return _success({
        "frequency": w.frequency,
        "latency": w.latency,
        "balance": w.balance,
        "sentiment": w.sentiment,
        "recency": w.recency,
    })


@app.route("/api/config/weights", methods=["POST"])
def update_weights():
    """
    Update scoring weights (affects next pipeline run).
    Body: {"frequency": 0.3, "latency": 0.2, ...}
    All values must sum to 1.0.
    """
    from nexus.core.config import CONFIG, ScoringWeights
    body = request.get_json(silent=True) or {}

    try:
        new_weights = ScoringWeights(
            frequency=float(body.get("frequency", CONFIG.scoring.weights.frequency)),
            latency=float(body.get("latency", CONFIG.scoring.weights.latency)),
            balance=float(body.get("balance", CONFIG.scoring.weights.balance)),
            sentiment=float(body.get("sentiment", CONFIG.scoring.weights.sentiment)),
            recency=float(body.get("recency", CONFIG.scoring.weights.recency)),
        )
        new_weights.validate()
        CONFIG.scoring.weights = new_weights
        return _success({"message": "Weights updated", "weights": body})
    except AssertionError as e:
        return _error(str(e))
    except Exception as e:
        return _error(str(e))


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return _error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _error("Method not allowed", 405)


@app.errorhandler(413)
def too_large(e):
    return _error("File too large (max 10MB)", 413)


@app.errorhandler(500)
def internal_error(e):
    return _error("Internal server error", 500)
