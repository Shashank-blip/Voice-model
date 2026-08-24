"""Reminder skills: add, list, cancel.

Composes the storage layer (Task 2) with the datetime extractor (Task 3).
No code here calls `datetime.now()` bare -- every time-dependent function
takes an injectable `now`, forwarded straight through to `Storage`.

Order of operations in `add` matters: the time expression is extracted
FIRST, and only the *residual* (the input with that expression already
removed) has the lead-in phrase stripped from it. Stripping the lead-in
before extracting the time would leave the time expression sitting inside
the saved body.

A reminder with no parseable time is still saved -- with `due_at=None` --
rather than rejected; the reply asks the user when. This is deliberate:
refusing to save loses the reminder's content, and the user can always
attach a time in a follow-up.

`cancel` is ambiguity-safe: it never destroys a row unless exactly one
reminder matches the needle. Zero matches gets a "couldn't find" reply;
more than one gets a "which one" reply listing the candidates, and nothing
is cancelled.
"""
from __future__ import annotations

import re
from datetime import datetime

from jarvis_nlu.slots import extract_time
from jarvis_nlu.storage import Storage

# Lead-ins stripped before the reminder body is stored.
_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:remind me(?:\s+(?:to|that|about))?|set a reminder(?:\s+(?:to|for|about))?|"
    r"don'?t let me forget(?:\s+to)?|nudge me(?:\s+(?:to|about))?)\s*", re.I)

_CANCEL_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:cancel|delete|remove|forget|drop|scrap)\s+"
    r"(?:the\s+|my\s+)?", re.I)
_CANCEL_TRAILER = re.compile(r"\s*reminder(?:s)?\s*$", re.I)


def _body(text: str) -> str:
    return _LEAD_IN.sub("", text).strip(" ,.")


def needs_time(text: str, now: datetime) -> bool:
    return extract_time(text, now) is None


def add(storage: Storage, text: str, now: datetime) -> str:
    slot = extract_time(text, now)
    # Strip the time first so the lead-in regex sees a clean body.
    body = _body(slot.residual if slot else text)
    if not body:
        # extract_time can leave an empty residual (e.g. "at 6" alone) --
        # fall back to a placeholder so the saved text and spoken reply
        # both stay sane rather than empty.
        body = "that"
    storage.add_reminder(body, slot.when if slot else None, now=now)
    if slot is None:
        return f"Saved — {body}. When should I remind you, sugar?"
    return f"You got it. I'll remind you to {body} at {_pretty(slot.when)}."


def _pretty(when: datetime) -> str:
    try:
        return when.strftime("%#I:%M %p on %A")
    except ValueError:
        return when.strftime("%-I:%M %p on %A")


def list_pending(storage: Storage, now: datetime) -> str:
    pending = storage.list_reminders()
    if not pending:
        return "You've got no reminders pendin', sugar."
    parts = []
    for reminder in pending[:5]:
        when = f" at {_pretty(reminder.due_at)}" if reminder.due_at else " (no time set)"
        parts.append(f"{reminder.text}{when}")
    return "Here's what you've got: " + "; ".join(parts) + "."


def cancel(storage: Storage, text: str, now: datetime) -> str:
    needle = _CANCEL_TRAILER.sub("", _CANCEL_LEAD_IN.sub("", text)).strip(" ,.")
    matches = storage.find_reminders_by_text(needle) if needle else []
    if not matches:
        return "I couldn't find a reminder matchin' that, sugar."
    if len(matches) > 1:
        listed = "; ".join(m.text for m in matches[:3])
        return f"I've got more than one of those — which did you mean? {listed}"
    storage.cancel_reminder(matches[0].id)
    return f"Done, I've dropped the reminder to {matches[0].text}."
