# Nexus — AI Relationship Intelligence Platform
### Backend v1.0.0

> An AI-driven automation pipeline that transforms raw communication logs into relationship health scores, behavioral insights, and personalized re-engagement actions.

---

## Architecture

```
INGEST → ANALYZE → DECIDE → ACT
```

| Stage | Module | Responsibility |
|-------|--------|---------------|
| **Ingest** | `parsers/` | Parse WhatsApp, JSON, CSV, Email logs → unified Message schema |
| **Analyze** | `analysis/` | Temporal metrics, sentiment, contact graph, anomaly detection |
| **Score** | `scoring/` | Nexus Score™ (5-signal composite 0-100) + tier classification |
| **Decide** | `decision/` | Rule-based + contextual decision routing |
| **Act** | `actions/` | LLM-powered nudge generation (Claude API) + fallback templates |
| **API** | `api/` | Flask REST API — all endpoints |
| **State** | `utils/` | SQLite persistence for contact states, scores, feedback |

---

## Quick Start

### 1. Install dependencies
```bash
pip install flask networkx numpy scipy pandas requests
```

### 2. Start the server
```bash
cd nexus/
python app.py
# With Anthropic API key for LLM nudges:
python app.py --api-key sk-ant-...
# Or set environment variable:
ANTHROPIC_API_KEY=sk-ant-... python app.py
```

Server starts at `http://localhost:5000`

### 3. Run tests
```bash
python tests/test_pipeline.py
# Expected: 30/30 tests passed
```

### 4. Try the synthetic demo
```bash
curl -X POST http://localhost:5000/api/pipeline/synthetic \
  -H "Content-Type: application/json" \
  -d '{"user_name": "You", "days_back": 90}'
```

---

## API Reference

### Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pipeline/run` | Run pipeline on uploaded chat data |
| `POST` | `/api/pipeline/synthetic` | Run on auto-generated synthetic data |

**POST /api/pipeline/run** — multipart/form-data:
- `file`: chat log file (.txt/.json/.csv)
- `owner_id`: your name/ID in the chat
- `format`: optional hint (`whatsapp`/`json`/`csv`/`email`)
- `contact_tiers`: JSON object `{"Alice": "close", "Bob": "professional"}`
- Header `X-API-Key`: Anthropic API key

**POST /api/pipeline/run** — JSON body:
```json
{
  "content": "raw chat log text...",
  "owner_id": "You",
  "format": "whatsapp",
  "contact_tiers": {"Alice": "close"},
  "api_key": "sk-ant-..."
}
```

### Contacts

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/contacts` | List all contacts (optionally `?tier=red`) |
| `GET` | `/api/contacts/:id` | Full contact state |
| `GET` | `/api/contacts/:id/score-history` | Score history |
| `GET` | `/api/contacts/:id/nudge-history` | Nudge history |
| `POST` | `/api/contacts/:id/tier` | Update relationship tier |
| `POST` | `/api/contacts/:id/feedback` | Submit nudge feedback |

### Nudges & Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/nudge/generate` | On-demand nudge for a contact |
| `GET` | `/api/dashboard` | Dashboard summary |
| `GET` | `/api/priority-queue` | Ranked at-risk contacts |
| `GET` | `/api/digest/weekly` | Weekly health digest |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/reset` | Reset all state (demo use) |
| `GET` | `/api/config/weights` | Get scoring weights |
| `POST` | `/api/config/weights` | Update scoring weights |

---

## Nexus Score™ Formula

```
Score = 0.30 × frequency + 0.20 × latency + 0.15 × balance + 0.20 × sentiment + 0.15 × recency
```

| Signal | Range | Description |
|--------|-------|-------------|
| **Frequency** | 0-100 | Monthly message volume vs. healthy baseline (log-normalized) |
| **Latency** | 0-100 | Mean reply time — fast = 100, >48h = 0 |
| **Balance** | 0-100 | Initiation ratio — 50/50 = 100, fully one-sided = 0 |
| **Sentiment** | 0-100 | Lexicon-based VADER-inspired sentiment scoring |
| **Recency** | 0-100 | Exponential decay: `100 × e^(-0.05 × days_since_last_msg)` |

**Tiers:** 🟢 Green (70-100) · 🟡 Yellow (40-69) · 🔴 Red (0-39)

---

## Supported Input Formats

| Format | Description | Example |
|--------|-------------|---------|
| WhatsApp iOS | `[DD/MM/YYYY, HH:MM:SS] Name: Message` | Standard iOS export |
| WhatsApp Android | `DD/MM/YY, HH:MM - Name: Message` | Standard Android export |
| JSON | Array of `{sender, receiver, timestamp, message}` objects | Flexible schema |
| CSV | Header row with `sender, receiver, timestamp, message` columns | Flexible headers |
| Email | mbox format or JSON email array | `{from, to, date, body}` |

---

## Project Structure

```
nexus/
├── app.py                    # Entry point
├── requirements.txt
├── core/
│   ├── models.py             # All dataclasses & enums
│   ├── config.py             # Scoring weights, thresholds, templates
│   └── pipeline.py           # Main orchestrator
├── parsers/
│   ├── whatsapp_parser.py    # iOS + Android WhatsApp formats
│   ├── generic_parser.py     # JSON, CSV, Email parsers
│   └── dispatcher.py         # Auto-detect + group by contact
├── analysis/
│   ├── temporal_analyzer.py  # Gaps, frequency, initiation ratio
│   ├── sentiment_analyzer.py # Lexicon-based sentiment + milestones
│   └── graph_builder.py      # NetworkX relationship graph
├── scoring/
│   └── nexus_score.py        # Nexus Score™ engine + priority queue
├── decision/
│   └── engine.py             # Decision routing + feedback application
├── actions/
│   └── nudge_generator.py    # LLM + template nudge generation
├── data/
│   └── synthetic_generator.py # Demo dataset with 8 contacts, 3 tiers
├── utils/
│   └── state_store.py        # SQLite persistence layer
├── api/
│   └── routes.py             # All Flask REST endpoints
└── tests/
    └── test_pipeline.py      # 30 tests, all stages
```
