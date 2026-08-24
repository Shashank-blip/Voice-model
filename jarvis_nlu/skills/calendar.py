"""Calendar skills: add an event, query a day.

Composes the storage layer (Task 2) with the datetime extractor (Task 3).
No code here calls `datetime.now()` bare -- every time-dependent function
takes an injectable `now`, forwarded straight through to `Storage`.

Order of operations in `add` matters, same as reminders: the time
expression is extracted FIRST, and only the *residual* (the input with
that expression already removed) has the lead-in phrase ("add"/"put"/
"schedule") and the trailing "to my calendar" stripped from it.

Unlike reminders, an event with no parseable time is NOT saved -- a
calendar entry with no date is meaningless, so `add` just asks when
instead of writing a dateless row.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from jarvis_nlu.slots import extract_time
from jarvis_nlu.storage import Storage

_ADD_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?(?:add|put|schedule|book|create|stick)\s+", re.I)
_ADD_TRAILER = re.compile(
    r"\s*(?:to|on|in|into)\s+(?:my\s+|the\s+)?calendar\s*$", re.I)


def _pretty(when: datetime) -> str:
    try:
        return when.strftime("%#I:%M %p")
    except ValueError:
        return when.strftime("%-I:%M %p")


def add(storage: Storage, text: str, now: datetime) -> str:
    slot = extract_time(text, now)
    if slot is None:
        return "Sure thing — when's that happenin', sugar?"
    title = _ADD_TRAILER.sub("", _ADD_LEAD_IN.sub("", slot.residual)).strip(" ,.")
    if not title:
        return "What should I call that one?"
    storage.add_event(title, slot.when, now=now)
    return f"Got it — {title} at {_pretty(slot.when)} on {slot.when.strftime('%A')}."


def query(storage: Storage, text: str, now: datetime) -> str:
    """Which day the user means. `extract_time` resolves bare 'tomorrow' to a
    daypart default, which is fine -- we only use the date part here."""
    lowered = text.lower()
    if "tomorrow" in lowered:
        day = (now + timedelta(days=1)).date()
        label = "Tomorrow"
    else:
        slot = extract_time(text, now)
        if slot is not None and "today" not in lowered:
            day, label = slot.when.date(), slot.when.strftime("%A")
        else:
            day, label = now.date(), "Today"

    events = storage.events_on(day)
    if not events:
        return f"{label}'s lookin' clear, sugar."
    listed = "; ".join(f"{e.title} at {_pretty(e.starts_at)}" for e in events)
    return f"{label} you've got: {listed}."
