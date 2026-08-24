"""The v1 intent roster. This module is the single source of truth --
training, evaluation, routing and tests all read the roster from here."""
from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    # Reminders
    ADD_REMINDER = "add_reminder"
    LIST_REMINDERS = "list_reminders"
    CANCEL_REMINDER = "cancel_reminder"
    # Calendar
    ADD_CALENDAR_EVENT = "add_calendar_event"
    QUERY_CALENDAR = "query_calendar"
    # Notes
    ADD_NOTE = "add_note"
    LIST_NOTES = "list_notes"
    SEARCH_NOTES = "search_notes"
    # Clock
    GET_TIME = "get_time"
    GET_DATE = "get_date"
    # Small talk
    GREETING = "greeting"
    HOW_ARE_YOU = "how_are_you"
    USER_MOOD = "user_mood"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    # Meta
    AFFIRM = "affirm"
    DENY = "deny"
    REPEAT_LAST = "repeat_last"
    SLEEP = "sleep"
    # Fallback
    OUT_OF_SCOPE = "out_of_scope"


ALL_INTENTS: tuple[Intent, ...] = tuple(Intent)

CHORE_INTENTS: frozenset[Intent] = frozenset({
    Intent.ADD_REMINDER, Intent.LIST_REMINDERS, Intent.CANCEL_REMINDER,
    Intent.ADD_CALENDAR_EVENT, Intent.QUERY_CALENDAR,
    Intent.ADD_NOTE, Intent.LIST_NOTES, Intent.SEARCH_NOTES,
    Intent.GET_TIME, Intent.GET_DATE,
})

SMALLTALK_INTENTS: frozenset[Intent] = frozenset({
    Intent.GREETING, Intent.HOW_ARE_YOU, Intent.USER_MOOD,
    Intent.THANKS, Intent.GOODBYE,
})

META_INTENTS: frozenset[Intent] = frozenset({
    Intent.AFFIRM, Intent.DENY, Intent.REPEAT_LAST, Intent.SLEEP,
})

# Intents that destroy user data and therefore require confirmation when the
# classifier is merely probable rather than confident.
DESTRUCTIVE_INTENTS: frozenset[Intent] = frozenset({Intent.CANCEL_REMINDER})

# Intents whose meaning depends on a resolved time slot.
TIME_SLOT_INTENTS: frozenset[Intent] = frozenset({
    Intent.ADD_REMINDER, Intent.ADD_CALENDAR_EVENT,
})
