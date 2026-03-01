"""
WhatsApp Chat Log Parser
Handles both standard WhatsApp export formats:
  - [DD/MM/YYYY, HH:MM:SS] Name: Message
  - MM/DD/YY, HH:MM - Name: Message
"""

import re
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple
import logging

from nexus.core.models import Message, Channel

logger = logging.getLogger(__name__)

# Pattern 1: [dd/mm/yyyy, hh:mm:ss] Name: message  (iOS)
PATTERN_IOS = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s+(.+?):\s+(.+)$",
    re.IGNORECASE
)

# Pattern 2: mm/dd/yy, hh:mm - Name: message  (Android)
PATTERN_ANDROID = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?:\s*[AP]M)?)\s*-\s*(.+?):\s+(.+)$",
    re.IGNORECASE
)

# System messages to skip
SYSTEM_PATTERNS = [
    re.compile(r"Messages and calls are end-to-end encrypted", re.I),
    re.compile(r"(added|removed|left|created|changed) (the group|this group)", re.I),
    re.compile(r"<Media omitted>", re.I),
    re.compile(r"This message was deleted", re.I),
    re.compile(r"You deleted this message", re.I),
    re.compile(r"null$", re.I),
    re.compile(r"^\s*$"),
]


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Try multiple datetime format combinations."""
    combined = f"{date_str.strip()} {time_str.strip()}"
    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%y %I:%M%p",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %I:%M%p",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    logger.warning(f"Could not parse datetime: {combined}")
    return None


def _is_system_message(content: str) -> bool:
    return any(p.search(content) for p in SYSTEM_PATTERNS)


def _infer_user(messages: List[Tuple]) -> str:
    """Infer the 'user' (owner of the export) as the most frequent sender."""
    from collections import Counter
    counts = Counter(m[0] for m in messages)
    if counts:
        return counts.most_common(1)[0][0]
    return "user"


def parse_whatsapp(content: str, owner_name: Optional[str] = None) -> List[Message]:
    """
    Parse a WhatsApp export text file.

    Args:
        content: Raw text content of the WhatsApp export
        owner_name: Name of the chat owner (the 'user'). If None, inferred.

    Returns:
        List of normalized Message objects
    """
    raw_messages = []
    current_entry = None

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try iOS pattern first, then Android
        match = PATTERN_IOS.match(line) or PATTERN_ANDROID.match(line)

        if match:
            date_str, time_str, sender, msg_content = match.groups()
            dt = _parse_datetime(date_str, time_str)
            if current_entry:
                raw_messages.append(current_entry)
            current_entry = (sender.strip(), msg_content.strip(), dt)
        elif current_entry:
            # Continuation of previous message
            sender, content_so_far, dt = current_entry
            current_entry = (sender, content_so_far + " " + line, dt)

    if current_entry:
        raw_messages.append(current_entry)

    if not raw_messages:
        logger.warning("WhatsApp parser: no messages found")
        return []

    # Infer owner
    if not owner_name:
        owner_name = _infer_user(raw_messages)
    logger.info(f"WhatsApp parser: inferred owner = '{owner_name}'")

    # Build Message objects
    messages: List[Message] = []
    prev_by_contact: dict = {}  # contact -> last message timestamp

    for sender, content_text, dt in raw_messages:
        if _is_system_message(content_text):
            continue

        is_outgoing = (sender.lower() == owner_name.lower())
        contact_id = "user" if not is_outgoing else sender

        # Determine contact (the other person)
        if is_outgoing:
            # In a 1:1 chat, try to find the other person
            other_senders = {s for s, _, _ in raw_messages if s.lower() != owner_name.lower()}
            contact_id = list(other_senders)[0] if len(other_senders) == 1 else "group_chat"
        else:
            contact_id = sender

        sender_id = owner_name if is_outgoing else sender
        receiver_id = contact_id if is_outgoing else owner_name

        # Compute reply latency
        latency_ms = None
        last_key = f"{contact_id}_{not is_outgoing}"
        if dt and last_key in prev_by_contact and prev_by_contact[last_key]:
            gap = (dt - prev_by_contact[last_key]).total_seconds() * 1000
            if 0 < gap < 86400000:  # Only count latency < 24 hours
                latency_ms = gap
        if dt:
            prev_by_contact[f"{contact_id}_{is_outgoing}"] = dt

        msg = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=dt,
            content=content_text,  # Will be used for analysis, then discarded
            content_hash=_hash_content(content_text),
            channel=Channel.WHATSAPP,
            reply_latency_ms=latency_ms,
            is_outgoing=is_outgoing,
        )
        messages.append(msg)

    logger.info(f"WhatsApp parser: parsed {len(messages)} messages")
    return messages
