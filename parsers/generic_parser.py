"""
Email, JSON, and CSV Parsers for Nexus
Handles: JSON interaction logs, CSV exports, basic email thread format
"""

import csv
import json
import hashlib
import io
import email as email_lib
from datetime import datetime
from typing import List, Optional
import logging

from nexus.core.models import Message, Channel

logger = logging.getLogger(__name__)


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse ISO 8601 or common datetime strings."""
    if not ts:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    ts = ts.strip().replace("Z", "")
    # Normalize: replace T separator with space for uniform parsing
    ts_normalized = ts.replace("T", " ")
    for fmt in formats:
        # Try both original and normalized forms
        for candidate in [ts, ts_normalized]:
            try:
                return datetime.strptime(candidate[:19], fmt[:19] if len(fmt) > 19 else fmt)
            except ValueError:
                pass
        try:
            return datetime.strptime(ts_normalized.split(".")[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    logger.warning(f"Could not parse datetime: {ts}")
    return None


# ─── JSON Parser ──────────────────────────────────────────────────────────────

def parse_json(content: str, owner_id: Optional[str] = None) -> List[Message]:
    """
    Parse JSON interaction log.

    Expected format (flexible, handles multiple schemas):
    [
      {
        "sender": "Alice",
        "receiver": "Bob",       # OR "contact"
        "timestamp": "2024-01-01T12:00:00",
        "message": "Hey!"        # OR "content" OR "text" OR "body"
      }
    ]
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return []

    if isinstance(data, dict):
        # Try to find the list inside common wrappers
        for key in ["messages", "data", "conversations", "chats", "interactions"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            logger.error("JSON: could not find messages array")
            return []

    if not isinstance(data, list):
        logger.error("JSON: root element must be a list")
        return []

    messages: List[Message] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        # Flexible field extraction
        sender = str(item.get("sender") or item.get("from") or item.get("sender_id") or "")
        receiver = str(item.get("receiver") or item.get("to") or item.get("receiver_id") or item.get("contact") or "")
        ts_raw = str(item.get("timestamp") or item.get("time") or item.get("date") or item.get("created_at") or "")
        content_text = str(item.get("message") or item.get("content") or item.get("text") or item.get("body") or "")
        channel_raw = str(item.get("channel") or item.get("platform") or "unknown").lower()

        if not sender or not content_text:
            continue

        channel = Channel.UNKNOWN
        for ch in Channel:
            if ch.value in channel_raw:
                channel = ch
                break

        dt = _parse_iso(ts_raw)
        is_outgoing = (owner_id and sender.lower() == owner_id.lower()) or (not owner_id and True)

        msg = Message(
            sender_id=sender,
            receiver_id=receiver,
            timestamp=dt,
            content=content_text,
            content_hash=_hash_content(content_text),
            channel=channel if channel != Channel.UNKNOWN else Channel.CSV,
            reply_latency_ms=item.get("reply_latency_ms"),
            is_outgoing=is_outgoing,
        )
        messages.append(msg)

    logger.info(f"JSON parser: parsed {len(messages)} messages")
    return messages


# ─── CSV Parser ───────────────────────────────────────────────────────────────

def parse_csv(content: str, owner_id: Optional[str] = None) -> List[Message]:
    """
    Parse CSV interaction log.

    Required columns (flexible naming):
      - sender / from / sender_id
      - receiver / to / receiver_id / contact
      - timestamp / time / date / created_at
      - message / content / text / body

    Optional:
      - channel / platform
      - reply_latency_ms
    """
    reader = csv.DictReader(io.StringIO(content))
    messages: List[Message] = []

    SENDER_KEYS = ["sender", "from", "sender_id", "author"]
    RECEIVER_KEYS = ["receiver", "to", "receiver_id", "contact", "recipient"]
    TS_KEYS = ["timestamp", "time", "date", "created_at", "datetime"]
    CONTENT_KEYS = ["message", "content", "text", "body", "msg"]

    def find(row, keys):
        for k in keys:
            for col in row:
                if col.lower().strip() == k:
                    return row[col]
        return ""

    for row in reader:
        sender = find(row, SENDER_KEYS)
        receiver = find(row, RECEIVER_KEYS)
        ts_raw = find(row, TS_KEYS)
        content_text = find(row, CONTENT_KEYS)
        channel_raw = (find(row, ["channel", "platform"]) or "").lower()

        if not sender or not content_text:
            continue

        channel = Channel.CSV
        for ch in Channel:
            if ch.value in channel_raw:
                channel = ch
                break

        dt = _parse_iso(ts_raw)
        is_outgoing = (owner_id and sender.lower() == owner_id.lower()) if owner_id else True

        try:
            latency = float(find(row, ["reply_latency_ms", "latency"]) or 0) or None
        except (ValueError, TypeError):
            latency = None

        msg = Message(
            sender_id=str(sender).strip(),
            receiver_id=str(receiver).strip(),
            timestamp=dt,
            content=str(content_text).strip(),
            content_hash=_hash_content(str(content_text)),
            channel=channel,
            reply_latency_ms=latency,
            is_outgoing=is_outgoing,
        )
        messages.append(msg)

    logger.info(f"CSV parser: parsed {len(messages)} messages")
    return messages


# ─── Email Parser ─────────────────────────────────────────────────────────────

def parse_email_thread(content: str, owner_email: Optional[str] = None) -> List[Message]:
    """
    Parse email thread in mbox-like format or simple email JSON list.
    Supports:
      - Raw RFC 2822 email messages separated by "From " lines (mbox)
      - JSON array of email dicts with {from, to, subject, date, body}
    """
    messages: List[Message] = []

    # Try JSON first
    if content.strip().startswith("[") or content.strip().startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                sender = str(item.get("from") or item.get("sender") or "")
                receiver = str(item.get("to") or item.get("receiver") or "")
                body = str(item.get("body") or item.get("content") or item.get("text") or "")
                ts_raw = str(item.get("date") or item.get("timestamp") or "")
                dt = _parse_iso(ts_raw)
                is_outgoing = (owner_email and owner_email.lower() in sender.lower()) if owner_email else True

                # Contact = the other party
                contact = receiver if is_outgoing else sender

                msg = Message(
                    sender_id=sender,
                    receiver_id=receiver,
                    timestamp=dt,
                    content=body,
                    content_hash=_hash_content(body),
                    channel=Channel.EMAIL,
                    is_outgoing=is_outgoing,
                )
                messages.append(msg)
            logger.info(f"Email JSON parser: parsed {len(messages)} emails")
            return messages
        except json.JSONDecodeError:
            pass

    # Try mbox format
    raw_emails = content.split("\nFrom ")
    for raw in raw_emails:
        if not raw.strip():
            continue
        if not raw.startswith("From "):
            raw = "From " + raw
        try:
            msg_obj = email_lib.message_from_string(raw)
            sender = msg_obj.get("From", "")
            receiver = msg_obj.get("To", "")
            date_str = msg_obj.get("Date", "")
            dt = None
            if date_str:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(date_str).replace(tzinfo=None)
                except Exception:
                    pass

            # Extract body
            body = ""
            if msg_obj.is_multipart():
                for part in msg_obj.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                        break
            else:
                body = msg_obj.get_payload(decode=True)
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="replace")
                body = body or ""

            is_outgoing = (owner_email and owner_email.lower() in sender.lower()) if owner_email else True

            msg = Message(
                sender_id=sender,
                receiver_id=receiver,
                timestamp=dt,
                content=body.strip()[:2000],
                content_hash=_hash_content(body[:2000]),
                channel=Channel.EMAIL,
                is_outgoing=is_outgoing,
            )
            messages.append(msg)
        except Exception as e:
            logger.warning(f"Email parse error: {e}")
            continue

    logger.info(f"Email mbox parser: parsed {len(messages)} emails")
    return messages
