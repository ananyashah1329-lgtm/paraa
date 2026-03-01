"""
Nexus Synthetic Dataset Generator
Generates realistic chat logs for testing and hackathon demos.
Creates 8 contacts with varying health levels (2 green, 4 yellow, 2 red).
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict
import hashlib


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


# Sample message content pools
MESSAGES_CASUAL = [
    "Hey! How's it going?", "What are you up to today?", "Did you see that movie?",
    "We should catch up soon!", "Haha that's hilarious 😂", "Miss you!",
    "How was your weekend?", "Omg same!", "No way 😱", "That's so cool!",
    "I can't believe it", "When are you free?", "Let's plan something",
    "That sounds amazing!", "I've been so busy lately", "Same honestly",
    "Have you tried that new cafe?", "We should go!", "Definitely!",
    "How's family?", "Things have been wild", "Tell me everything",
    "That's wild", "You're so funny", "Ugh I know right",
    "I'm so tired today", "Same, what a week", "Weekend can't come soon enough",
    "Did you finish the project?", "Almost, how about you?", "Yeah finally done!",
]

MESSAGES_ACADEMIC = [
    "How did the exam go?", "It was rough ngl", "Same, the questions were unexpected",
    "Did you understand that lecture?", "Partially lol", "The assignment is killing me",
    "When's the deadline again?", "Friday I think", "Let's study together",
    "The professor posted the grades!", "How'd you do?", "Better than expected tbh",
    "Can you share your notes?", "Sure, give me a sec", "Thanks so much!",
    "I'm panicking about finals", "We'll get through it", "Group study session?",
]

MESSAGES_CAREER = [
    "I got the internship!", "NO WAY that's amazing!!", "So proud of you!",
    "The interview was intense", "But you aced it!", "Fingers crossed 🤞",
    "First day was good!", "Tell me everything", "The team seems nice",
    "Got a promotion!", "You deserve it!", "Congrats, legend!",
    "Working late again", "The project is due tomorrow", "You got this!",
    "My manager is great", "That's rare honestly", "Lucky!",
]

MESSAGES_MILESTONE = [
    "Happy birthday!! 🎂🎉", "Omg I completely forgot, I'm so sorry!",
    "Congrats on graduating!", "Finally!! 🎓", "So proud of you!",
    "You got engaged??", "YES! Last night!!", "AHHH show me the ring!",
    "Heard you moved to a new place!", "Yeah just settled in", "Love that!",
    "Happy birthday to you too! 🎉", "Thank you!! This means a lot",
]


def _generate_conversation(
    contact_name: str,
    user_name: str,
    start_date: datetime,
    end_date: datetime,
    frequency: str = "medium",  # high / medium / low / silent
    sentiment: str = "positive",  # positive / neutral / negative
    include_milestones: bool = False,
) -> List[Dict]:
    """Generate a realistic conversation between user and contact."""

    messages = []
    current = start_date
    total_days = (end_date - start_date).days

    # Determine message frequency
    freq_map = {
        "high": (1.5, 0.5),      # avg_per_day, std
        "medium": (0.5, 0.3),
        "low": (0.15, 0.1),
        "silent": (0.03, 0.02),   # Very sparse
    }
    avg_per_day, std_per_day = freq_map[frequency]

    pool = MESSAGES_CASUAL + MESSAGES_ACADEMIC + MESSAGES_CAREER
    if include_milestones:
        pool += MESSAGES_MILESTONE

    day = 0
    while current < end_date:
        msgs_today = max(0, int(random.gauss(avg_per_day, std_per_day)))
        for _ in range(msgs_today):
            # Randomly choose sender
            is_outgoing = random.random() < 0.55  # Slight user bias
            sender = user_name if is_outgoing else contact_name
            receiver = contact_name if is_outgoing else user_name

            # Pick message
            content = random.choice(pool)

            # Add some sentiment variation
            if sentiment == "negative" and random.random() < 0.3:
                content = random.choice([
                    "Things have been rough lately", "Not doing great ngl",
                    "I'm stressed", "It's been hard", "I feel so overwhelmed",
                    "Not in a great headspace", "This week has been terrible",
                ])
            elif sentiment == "positive" and random.random() < 0.4:
                content = random.choice([
                    "Life's been amazing lately!", "Everything is going so well",
                    "I'm really happy right now", "Best week ever honestly",
                    "You always make me smile", "So grateful for you!",
                ])

            # Time within day
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            ts = current.replace(hour=hour, minute=minute, second=random.randint(0, 59))

            # Reply latency (ms)
            latency = None
            if not is_outgoing:
                latency = random.gauss(3600000, 1800000)  # ~1 hour avg
                latency = max(60000, min(latency, 86400000))

            messages.append({
                "sender": sender,
                "receiver": receiver,
                "timestamp": ts.isoformat(),
                "message": content,
                "channel": "whatsapp",
                "reply_latency_ms": latency,
            })

        current += timedelta(days=1)
        day += 1

    return sorted(messages, key=lambda m: m["timestamp"])


def generate_synthetic_dataset(
    user_name: str = "You",
    days_back: int = 90,
    seed: int = 42
) -> Dict:
    """
    Generate a synthetic dataset with 8 contacts across 3 health tiers.
    2 Green (healthy), 4 Yellow (drifting), 2 Red (critical).
    """
    random.seed(seed)
    end = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)

    contacts = [
        # GREEN (healthy, active)
        {
            "name": "Priya Sharma",
            "frequency": "high",
            "sentiment": "positive",
            "milestones": True,
            "expected_tier": "green",
            "description": "Best friend - very active, positive conversations",
        },
        {
            "name": "Arjun Mehta",
            "frequency": "high",
            "sentiment": "positive",
            "milestones": False,
            "expected_tier": "green",
            "description": "Roommate - frequent daily chats",
        },
        # YELLOW (drifting)
        {
            "name": "Sneha Patel",
            "frequency": "medium",
            "sentiment": "neutral",
            "milestones": False,
            "expected_tier": "yellow",
            "description": "College friend - was close, drifting a bit",
        },
        {
            "name": "Rahul Kumar",
            "frequency": "low",
            "sentiment": "neutral",
            "milestones": False,
            "expected_tier": "yellow",
            "description": "Study group friend - infrequent contact",
        },
        {
            "name": "Ananya Singh",
            "frequency": "medium",
            "sentiment": "neutral",
            "milestones": True,
            "expected_tier": "yellow",
            "description": "School friend - occasional check-ins",
        },
        {
            "name": "Vikram Nair",
            "frequency": "low",
            "sentiment": "neutral",
            "milestones": False,
            "expected_tier": "yellow",
            "description": "Internship colleague - professional",
        },
        # RED (critical)
        {
            "name": "Kavya Iyer",
            "frequency": "silent",
            "sentiment": "negative",
            "milestones": False,
            "expected_tier": "red",
            "description": "Used to be close - now almost no contact, sentiment dropped",
        },
        {
            "name": "Rohan Das",
            "frequency": "silent",
            "sentiment": "neutral",
            "milestones": False,
            "expected_tier": "red",
            "description": "Old friend - complete silence for weeks",
        },
    ]

    all_messages = []
    contact_map = {}

    for contact in contacts:
        name = contact["name"]

        # RED contacts: active earlier, then silent
        if contact["expected_tier"] == "red":
            mid = start + timedelta(days=days_back // 3)
            # Some early messages
            early = _generate_conversation(name, user_name, start, mid, "low", contact["sentiment"], contact["milestones"])
            # Then silence
            late = _generate_conversation(name, user_name, mid, end, "silent", contact["sentiment"], False)
            msgs = early + late
        # YELLOW: medium activity, recent slowdown
        elif contact["expected_tier"] == "yellow":
            mid = start + timedelta(days=days_back * 2 // 3)
            early = _generate_conversation(name, user_name, start, mid, "medium", contact["sentiment"], contact["milestones"])
            late = _generate_conversation(name, user_name, mid, end, "low", contact["sentiment"], False)
            msgs = early + late
        # GREEN: consistently active
        else:
            msgs = _generate_conversation(name, user_name, start, end, contact["frequency"], contact["sentiment"], contact["milestones"])

        all_messages.extend(msgs)
        contact_map[name] = {
            "expected_tier": contact["expected_tier"],
            "description": contact["description"],
            "message_count": len(msgs),
        }

    all_messages.sort(key=lambda m: m["timestamp"])

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "user_name": user_name,
            "days_back": days_back,
            "total_messages": len(all_messages),
            "total_contacts": len(contacts),
            "contact_map": contact_map,
            "seed": seed,
        },
        "messages": all_messages,
    }


def generate_whatsapp_export(contact_name: str, user_name: str = "You", days: int = 60) -> str:
    """Generate a WhatsApp-format text export for a single contact."""
    end = datetime.now()
    start = end - timedelta(days=days)

    tier = random.choice(["high", "medium", "low"])
    msgs = _generate_conversation(contact_name, user_name, start, end, tier, "positive", True)

    lines = []
    lines.append("Messages and calls are end-to-end encrypted.")
    for m in msgs:
        dt = datetime.fromisoformat(m["timestamp"])
        formatted_date = dt.strftime("%d/%m/%Y")
        formatted_time = dt.strftime("%H:%M:%S")
        lines.append(f"[{formatted_date}, {formatted_time}] {m['sender']}: {m['message']}")

    return "\n".join(lines)


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    print(json.dumps(dataset["metadata"], indent=2))
    print(f"\nTotal messages: {dataset['metadata']['total_messages']}")
    for name, info in dataset["metadata"]["contact_map"].items():
        print(f"  {name}: {info['message_count']} msgs — expected {info['expected_tier']}")
