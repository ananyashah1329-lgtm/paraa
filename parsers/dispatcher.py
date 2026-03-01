"""
Nexus Parser Dispatcher
Auto-detects chat log format and routes to the correct parser.
"""

import re
from typing import List, Optional, Tuple
import logging

from nexus.core.models import Message, Channel
from nexus.parsers.whatsapp_parser import parse_whatsapp
from nexus.parsers.generic_parser import parse_json, parse_csv, parse_email_thread

logger = logging.getLogger(__name__)


def _detect_format(content: str) -> str:
    """Detect the format of the chat log content."""
    sample = content[:3000].strip()

    # WhatsApp: starts with [dd/mm/yyyy or mm/dd/yy format lines
    whatsapp_ios = re.search(r"^\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}", sample, re.MULTILINE)
    whatsapp_android = re.search(r"^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[-–]", sample, re.MULTILINE)
    if whatsapp_ios or whatsapp_android:
        return "whatsapp"

    # JSON
    if sample.startswith("[") or sample.startswith("{"):
        return "json"

    # CSV: first line looks like a header with commas
    first_line = sample.split("\n")[0]
    if "," in first_line and any(k in first_line.lower() for k in ["sender", "from", "message", "content", "timestamp", "date"]):
        return "csv"

    # Email mbox
    if sample.startswith("From ") or "MIME-Version:" in sample or "Content-Type:" in sample:
        return "email"

    # Fallback: try CSV
    if "," in first_line:
        return "csv"

    return "unknown"


def parse_auto(
    content: str,
    filename: Optional[str] = None,
    owner_id: Optional[str] = None,
    format_hint: Optional[str] = None
) -> Tuple[List[Message], str]:
    """
    Auto-detect and parse chat logs.

    Args:
        content: Raw text content
        filename: Optional filename for format hint
        owner_id: The user's identifier (name/email)
        format_hint: Optional explicit format override

    Returns:
        Tuple of (messages, detected_format)
    """
    # Use explicit hint or filename extension
    fmt = format_hint
    if not fmt and filename:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("txt",):
            fmt = "whatsapp"
        elif ext in ("json",):
            fmt = "json"
        elif ext in ("csv",):
            fmt = "csv"
        elif ext in ("mbox", "eml"):
            fmt = "email"

    # Auto-detect from content
    if not fmt:
        fmt = _detect_format(content)

    logger.info(f"Parser dispatcher: detected format = {fmt}")

    if fmt == "whatsapp":
        messages = parse_whatsapp(content, owner_name=owner_id)
    elif fmt == "json":
        messages = parse_json(content, owner_id=owner_id)
    elif fmt == "csv":
        messages = parse_csv(content, owner_id=owner_id)
    elif fmt == "email":
        messages = parse_email_thread(content, owner_email=owner_id)
    else:
        # Last resort: try all parsers
        logger.warning("Unknown format, trying all parsers...")
        for parser, name in [
            (parse_whatsapp, "whatsapp"),
            (lambda c, owner_id=None: parse_json(c, owner_id=owner_id), "json"),
            (lambda c, owner_id=None: parse_csv(c, owner_id=owner_id), "csv"),
        ]:
            try:
                result = parser(content, owner_id=owner_id) if name != "whatsapp" else parser(content, owner_name=owner_id)
                if result:
                    logger.info(f"Fallback parser succeeded with: {name}")
                    return result, name
            except Exception:
                continue
        messages = []
        fmt = "failed"

    return messages, fmt


def group_messages_by_contact(messages: List[Message], user_id: Optional[str] = None) -> dict:
    """
    Group messages by contact.

    Returns:
        Dict mapping contact_id -> List[Message]
    """
    groups: dict = {}

    for msg in messages:
        # Determine the contact (the other person)
        if msg.is_outgoing:
            contact_id = msg.receiver_id
        else:
            contact_id = msg.sender_id

        # Normalize contact ID
        contact_id = contact_id.strip() if contact_id else "unknown"
        if not contact_id or contact_id.lower() in ("", "unknown", "none"):
            continue

        if contact_id not in groups:
            groups[contact_id] = []
        groups[contact_id].append(msg)

    # Sort each group by timestamp
    for contact_id in groups:
        groups[contact_id].sort(key=lambda m: m.timestamp or datetime(1970, 1, 1))

    return groups


# Import datetime at module level
from datetime import datetime
