"""The learning log.

PRIVACY: this module writes every utterance spoken to the assistant, in
plain text, to `data/learning_log.jsonl`, forever -- that is the whole
point, it is the raw material real training data gets made from. That
file is local-only and gitignored (see `.gitignore`'s `data/` entry); it
never leaves this machine, and this module makes no network calls. Delete
the file at any time to erase the history -- nothing in this package
depends on it persisting.

Design constraint this file exists to protect (see the plan, Task B): you
cannot train on the model's own predictions. Every entry this module
writes carries `predicted_intent` -- what the model guessed -- and never
an `intent` key. Ground truth (`intent`) is only ever added later, by a
human, via the review CLI (a separate task). `training/generate.py` only
merges rows that carry `intent`, so every row this module writes is
inert as training data until a person has looked at it.

Note the module name: `turnlog.py`, not `logging.py` -- the latter would
shadow the stdlib `logging` module for every other file in this package.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jarvis_nlu.intents import Intent

log = logging.getLogger(__name__)

# How soon a following turn has to arrive to count as evidence about the
# turn before it (a repeat, or a negation right after an action). Chosen
# generously -- a user re-stating themselves or saying "no, not that"
# rarely takes longer than this, and a false positive here just means one
# extra row for a human to glance at and skip in review, not a corrupted
# label (nothing here ever writes a training label on its own).
FOLLOWUP_WINDOW_SECONDS = 10.0


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _near_duplicate(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    return bool(na) and bool(nb) and na == nb


@dataclass
class _PriorTurn:
    id: str
    text: str
    timestamp: datetime
    predicted_intent: str
    handled: bool
    needs_confirmation: bool
    reprompt: bool


class TurnLogger:
    """Appends one JSON object per spoken turn to a JSONL file under `data/`.

    Append-only. Earlier lines are never rewritten -- the two signals
    that only become knowable from a LATER turn (`repeat_within_seconds`
    and `user_negation`) are captured by appending a small *correction
    record* that references the prior entry's `id` via `correction_for`,
    rather than seeking back into the file to edit an earlier line. That
    keeps every write a single append-and-flush, which is what makes this
    safe to call from a live, concurrently-running turn loop -- editing a
    line in place while another process/thread is mid-append to the same
    file risks corrupting the log.

    Every public method here is defensive: a disk-full condition, a
    permissions error, an unwritable path -- none of it may ever escape
    to the caller. A spoken turn must complete even if logging cannot.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._prior: _PriorTurn | None = None

    def log(self, *, text: str, predicted_intent: str, confidence: float,
            now: datetime, handled: bool = True, reprompt: bool = False,
            needs_confirmation: bool = False, resolved_answer: str | None = None,
            skill_error: str | None = None, slots: dict | None = None) -> str | None:
        """Append one entry for this turn. Returns its `id`, or `None` if
        the write failed (callers should not depend on the return value --
        it exists for tests and for the odd caller that wants to correlate
        entries, not as a signal to branch on)."""
        entry_id = uuid.uuid4().hex
        entry = {
            "id": entry_id,
            "timestamp": now.isoformat(),
            "transcript": text,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "handled": handled,
            # "did it fall below threshold and reach the LLM" -- handled=False
            # is exactly the cases that fall through to the LLM caller
            # (a confident out_of_scope, or a second-strike escalation).
            "deferred": not handled,
            "reprompt": reprompt,
            "needs_confirmation": needs_confirmation,
        }
        if resolved_answer is not None:
            entry["resolved_answer"] = resolved_answer
        if skill_error is not None:
            entry["skill_error"] = skill_error
        if slots:
            entry["slots"] = slots

        written = self._write(entry)
        self._maybe_flag_followups(text, predicted_intent, now)
        # Track the prior turn even if THIS write failed, so a later
        # near-duplicate can still be detected relative to what was
        # actually said (the correction record referencing a never-
        # written id is itself best-effort and never fatal to anything).
        self._prior = _PriorTurn(id=entry_id, text=text, timestamp=now,
                                 predicted_intent=predicted_intent,
                                 handled=handled,
                                 needs_confirmation=needs_confirmation,
                                 reprompt=reprompt)
        return entry_id if written else None

    def _maybe_flag_followups(self, text: str, predicted_intent: str, now: datetime) -> None:
        prior = self._prior
        if prior is None:
            return
        try:
            elapsed = (now - prior.timestamp).total_seconds()
        except Exception:
            return
        if elapsed < 0 or elapsed > FOLLOWUP_WINDOW_SECONDS:
            return

        if _near_duplicate(prior.text, text):
            self._write({
                "id": uuid.uuid4().hex,
                "timestamp": now.isoformat(),
                "correction_for": prior.id,
                "repeat_within_seconds": round(elapsed, 3),
            })
            return  # a repeat and a negation can't both be this turn

        # The model classifying this turn as a plain "no" right after an
        # action that was NOT itself asking for confirmation means the
        # action was wrong -- free signal, no label from the user
        # required. A deny that resolves an existing confirmation prompt
        # is the flow working correctly, not a miss, so it is explicitly
        # excluded via needs_confirmation (and a re-prompt, which asked
        # nothing that "no" could be answering, is excluded too).
        if (predicted_intent == Intent.DENY.value and prior.handled
                and not prior.needs_confirmation and not prior.reprompt):
            self._write({
                "id": uuid.uuid4().hex,
                "timestamp": now.isoformat(),
                "correction_for": prior.id,
                "user_negation": True,
            })

    def _write(self, obj: dict) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(obj) + "\n")
                handle.flush()
            return True
        except Exception:
            log.warning("turn log write failed for %s", self.path, exc_info=True)
            return False
