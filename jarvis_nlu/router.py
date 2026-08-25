"""Intent dispatch. Replaces the ordered-regex chain whose earlier patterns
shadowed later ones -- there is exactly one scored decision here, so no branch
can silently swallow another."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from jarvis_nlu.config import Config
from jarvis_nlu.intents import DESTRUCTIVE_INTENTS, Intent
from jarvis_nlu.model import Classifier, Thresholds
from jarvis_nlu.responses import ResponsePool
from jarvis_nlu.skills import calendar, clock, notes, reminders, smalltalk
from jarvis_nlu.storage import Storage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    handled: bool
    reply: str | None
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    needs_confirmation: bool = False
    # Task F (two-strike deferral policy). A re-prompt IS spoken, so it
    # carries handled=True like any other answered turn -- `handled` alone
    # must never be overloaded to also mean "call the LLM". `reprompt=True`
    # is the caller's signal that this reply exists only to ask the user to
    # repeat themselves, not because the turn was genuinely answered.
    reprompt: bool = False


DEFERRED = Result(handled=False, reply=None,
                  intent=Intent.OUT_OF_SCOPE.value, confidence=0.0)


class Assistant:
    def __init__(self, config: Config, storage: Storage, classifier: Classifier,
                 thresholds: Thresholds, pool: ResponsePool | None = None,
                 embedder: Callable[[str], bytes] | None = None):
        self.config = config
        self.storage = storage
        self.classifier = classifier
        self.thresholds = thresholds
        self.pool = pool or ResponsePool()
        self.embedder = embedder
        self.last_reply: str | None = None
        # Held for the duration of a turn so the proactive scheduler never
        # speaks over the user mid-exchange.
        self.turn_lock = threading.Lock()
        self._pending: tuple[str, str] | None = None  # (intent_value, original text)
        # Task F: consecutive below-`defer` turns. Only ever 0 or 1 --
        # reaching the 2-strike threshold fires the LLM signal AND resets
        # it in the same step (see _handle_low_confidence), so it never
        # climbs past 1 and, since it is only ever incremented from 0 or
        # reset to 0, it can never go negative either.
        self._consecutive_unsure = 0

    def handle(self, text: str, now: datetime | None = None) -> Result:
        now = now or datetime.now()
        text = (text or "").strip()
        if not text:
            return DEFERRED

        with self.turn_lock:
            intent_value, confidence = self.classifier.classify(text)

            # Below defer: the model is merely unsure, most often a garbled
            # transcript rather than a genuinely out-of-scope request. The
            # two-strike policy handles this BEFORE the out-of-scope check
            # below, since it applies regardless of predicted intent.
            if confidence < self.thresholds.defer:
                return self._handle_low_confidence(intent_value, confidence)

            # Confidently predicted out-of-scope: this genuinely needs the
            # LLM, immediately -- no re-prompt, no strike bookkeeping.
            if intent_value == Intent.OUT_OF_SCOPE.value:
                return Result(handled=False, reply=None,
                              intent=intent_value, confidence=confidence)

            if self._pending and intent_value in (Intent.AFFIRM.value, Intent.DENY.value):
                return self._resolve_pending(intent_value, confidence, now)

            # A destructive intent we are merely probable about asks first.
            if (intent_value in {i.value for i in DESTRUCTIVE_INTENTS}
                    and confidence < self.thresholds.confirm):
                self._pending = (intent_value, text)
                return self._reply(f"Just to be sure, sugar — you want me to {text}?",
                                   intent_value, confidence, needs_confirmation=True)

            return self._dispatch(intent_value, text, confidence, now)

    def _handle_low_confidence(self, intent_value: str, confidence: float) -> Result:
        """Task F two-strike policy. First consecutive below-`defer` turn:
        free local re-prompt, no LLM call. Second consecutive: genuinely
        escalate to the LLM (handled=False) -- and reset immediately, so a
        run of low-confidence turns cycles re-prompt/escalate/re-prompt/...
        rather than escalating on every turn from the second onward."""
        self._consecutive_unsure += 1
        if self._consecutive_unsure >= 2:
            self._consecutive_unsure = 0
            return Result(handled=False, reply=None,
                          intent=intent_value, confidence=confidence)

        reply = self.pool.pick("reprompt")
        self.last_reply = reply
        return Result(handled=True, reply=reply, intent=intent_value,
                      confidence=confidence, reprompt=True)

    def _resolve_pending(self, answer: str, confidence: float, now: datetime) -> Result:
        pending_intent, pending_text = self._pending
        self._pending = None
        if answer == Intent.DENY.value:
            return self._reply(self.pool.pick("deny"), answer, confidence)
        return self._dispatch(pending_intent, pending_text, confidence, now)

    def _reply(self, text: str, intent_value: str, confidence: float,
               needs_confirmation: bool = False, slots: dict | None = None) -> Result:
        # The single choke point for every genuinely-handled reply
        # (dispatch success/apology, confirmation prompts, deny) -- Task F:
        # any successfully handled turn resets the strike counter to zero.
        # _handle_low_confidence's re-prompt deliberately bypasses this
        # method so the counter is NOT reset by a re-prompt itself.
        self._consecutive_unsure = 0
        self.last_reply = text
        return Result(handled=True, reply=text, intent=intent_value,
                      confidence=confidence, slots=slots or {},
                      needs_confirmation=needs_confirmation)

    def _dispatch(self, intent_value: str, text: str, confidence: float,
                  now: datetime) -> Result:
        try:
            reply = self._run_skill(intent_value, text, now)
        except Exception:
            # A skill fault must never kill the assistant.
            log.exception("skill %s failed on %r", intent_value, text)
            return self._reply(self.pool.pick("unknown_error"), intent_value, confidence)
        return self._reply(reply, intent_value, confidence)

    def _run_skill(self, intent_value: str, text: str, now: datetime) -> str:
        store = self.storage
        if intent_value == Intent.ADD_REMINDER.value:
            return reminders.add(store, text, now)
        if intent_value == Intent.LIST_REMINDERS.value:
            return reminders.list_pending(store, now)
        if intent_value == Intent.CANCEL_REMINDER.value:
            return reminders.cancel(store, text, now)
        if intent_value == Intent.ADD_CALENDAR_EVENT.value:
            return calendar.add(store, text, now)
        if intent_value == Intent.QUERY_CALENDAR.value:
            return calendar.query(store, text, now)
        if intent_value == Intent.ADD_NOTE.value:
            return notes.add(store, text, now, self.embedder)
        if intent_value == Intent.LIST_NOTES.value:
            return notes.list_recent(store, now)
        if intent_value == Intent.SEARCH_NOTES.value:
            return notes.search(store, text, self.embedder)
        if intent_value == Intent.GET_TIME.value:
            return clock.get_time(now)
        if intent_value == Intent.GET_DATE.value:
            return clock.get_date(now)
        if intent_value == Intent.REPEAT_LAST.value:
            return self.last_reply or "I haven't said anything yet, sugar."
        # Small talk, affirm/deny with nothing pending, and sleep.
        return smalltalk.respond(intent_value, text, self.pool)
