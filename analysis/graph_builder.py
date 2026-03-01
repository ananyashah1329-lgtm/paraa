"""
Nexus Contact Graph Builder
Constructs a weighted directed graph from all messages.
Edge weights represent interaction strength.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not available, using basic graph implementation")

from nexus.core.models import Message

logger = logging.getLogger(__name__)


class RelationshipGraph:
    """
    Weighted directed graph of communication relationships.
    
    Nodes: contact IDs
    Edges: (user -> contact) and (contact -> user) with interaction weights
    Edge attributes:
      - message_count: total messages in direction
      - last_interaction: most recent message timestamp
      - sentiment_mean: mean sentiment of messages on this edge
      - recency_weight: exponentially decayed weight
    """

    def __init__(self, owner_id: str = "user"):
        self.owner_id = owner_id
        self.graph = nx.DiGraph() if HAS_NETWORKX else None
        self._edges: Dict[Tuple, Dict] = {}  # Fallback if no networkx
        self._nodes: Dict[str, Dict] = {}

    def build(self, messages: List[Message], reference_time: Optional[datetime] = None) -> "RelationshipGraph":
        """Build the graph from a list of messages."""
        if not reference_time:
            reference_time = datetime.utcnow()

        # Aggregate edge data
        edge_data: Dict[Tuple[str, str], Dict] = {}

        for msg in messages:
            if not msg.sender_id or not msg.receiver_id:
                continue

            src = msg.sender_id
            dst = msg.receiver_id
            edge_key = (src, dst)

            if edge_key not in edge_data:
                edge_data[edge_key] = {
                    "message_count": 0,
                    "timestamps": [],
                    "sentiments": [],
                    "latencies": [],
                }

            ed = edge_data[edge_key]
            ed["message_count"] += 1
            if msg.timestamp:
                ed["timestamps"].append(msg.timestamp)
            if msg.sentiment_score is not None:
                ed["sentiments"].append(msg.sentiment_score)
            if msg.reply_latency_ms:
                ed["latencies"].append(msg.reply_latency_ms)

        # Build graph edges with computed weights
        for (src, dst), data in edge_data.items():
            count = data["message_count"]
            timestamps = sorted(data["timestamps"])
            last_ts = timestamps[-1] if timestamps else None

            # Recency weight: exponential decay
            days_since = 0.0
            if last_ts:
                days_since = max(0, (reference_time - last_ts).total_seconds() / 86400.0)
            recency_weight = math.exp(-0.05 * days_since)

            # Sentiment
            sentiments = data["sentiments"]
            sentiment_mean = sum(sentiments) / len(sentiments) if sentiments else 0.0

            # Interaction weight = log(count+1) * recency
            interaction_weight = math.log(count + 1) * recency_weight

            # Latency
            latencies = data["latencies"]
            mean_latency_hr = (sum(latencies) / len(latencies) / 3600000.0) if latencies else 0.0

            edge_attrs = {
                "message_count": count,
                "last_interaction": last_ts,
                "days_since": days_since,
                "sentiment_mean": round(sentiment_mean, 3),
                "recency_weight": round(recency_weight, 3),
                "interaction_weight": round(interaction_weight, 3),
                "mean_latency_hr": round(mean_latency_hr, 2),
                "weight": round(interaction_weight, 3),
            }

            if HAS_NETWORKX:
                if src not in self.graph:
                    self.graph.add_node(src, node_type="contact" if src != self.owner_id else "user")
                if dst not in self.graph:
                    self.graph.add_node(dst, node_type="contact" if dst != self.owner_id else "user")
                self.graph.add_edge(src, dst, **edge_attrs)
            else:
                self._edges[(src, dst)] = edge_attrs
                self._nodes[src] = {"node_type": "contact" if src != self.owner_id else "user"}
                self._nodes[dst] = {"node_type": "contact" if dst != self.owner_id else "user"}

        logger.info(f"Graph built: {self.node_count()} nodes, {self.edge_count()} edges")
        return self

    def get_contacts(self) -> List[str]:
        """Get all contacts (nodes that are not the owner)."""
        if HAS_NETWORKX:
            return [n for n in self.graph.nodes if n != self.owner_id]
        return [n for n in self._nodes if n != self.owner_id]

    def node_count(self) -> int:
        if HAS_NETWORKX:
            return self.graph.number_of_nodes()
        return len(self._nodes)

    def edge_count(self) -> int:
        if HAS_NETWORKX:
            return self.graph.number_of_edges()
        return len(self._edges)

    def get_edge(self, src: str, dst: str) -> Optional[Dict]:
        if HAS_NETWORKX:
            if self.graph.has_edge(src, dst):
                return dict(self.graph[src][dst])
            return None
        return self._edges.get((src, dst))

    def get_bidirectional_weight(self, contact_id: str) -> float:
        """Get combined interaction weight for both directions."""
        out_edge = self.get_edge(self.owner_id, contact_id) or {}
        in_edge = self.get_edge(contact_id, self.owner_id) or {}
        return out_edge.get("interaction_weight", 0) + in_edge.get("interaction_weight", 0)

    def get_contact_summary(self, contact_id: str) -> Dict:
        """Summarize all edge data for a contact."""
        out_edge = self.get_edge(self.owner_id, contact_id) or {}
        in_edge = self.get_edge(contact_id, self.owner_id) or {}

        sent = out_edge.get("message_count", 0)
        received = in_edge.get("message_count", 0)
        total = sent + received

        last_ts = max(
            [t for t in [out_edge.get("last_interaction"), in_edge.get("last_interaction")] if t],
            default=None
        )

        return {
            "contact_id": contact_id,
            "total_messages": total,
            "messages_sent": sent,
            "messages_received": received,
            "last_interaction": last_ts.isoformat() if last_ts else None,
            "bidirectional_weight": round(self.get_bidirectional_weight(contact_id), 3),
        }

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        contacts_summary = [self.get_contact_summary(c) for c in self.get_contacts()]
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "owner_id": self.owner_id,
            "contacts": contacts_summary,
        }

    def get_strongest_connections(self, n: int = 5) -> List[str]:
        """Return top-n contacts by bidirectional interaction weight."""
        contacts = self.get_contacts()
        weights = [(c, self.get_bidirectional_weight(c)) for c in contacts]
        weights.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in weights[:n]]

    def get_at_risk_contacts(self, days_threshold: float = 14.0) -> List[str]:
        """Return contacts with no interaction in the last N days."""
        at_risk = []
        ref = datetime.utcnow()
        for contact in self.get_contacts():
            out_edge = self.get_edge(self.owner_id, contact) or {}
            in_edge = self.get_edge(contact, self.owner_id) or {}

            last_ts = max(
                [t for t in [out_edge.get("last_interaction"), in_edge.get("last_interaction")] if t],
                default=None
            )
            if last_ts:
                days_since = (ref - last_ts).total_seconds() / 86400.0
                if days_since > days_threshold:
                    at_risk.append(contact)
            else:
                at_risk.append(contact)

        return at_risk
