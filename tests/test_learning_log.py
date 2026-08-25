"""Task B: the learning log.

Core property under test throughout this file: the log records what the
model predicted (`predicted_intent`) and evidence about whether that
prediction was right, but NEVER a ground-truth `intent` -- that only ever
gets added later, by a human, via the review CLI. See
jarvis_nlu/turnlog.py for the full rationale.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis_nlu.config import Config
from jarvis_nlu.model import FakeClassifier, Thresholds
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage
from jarvis_nlu.turnlog import TurnLogger

REPO_ROOT = Path(__file__).parent.parent
NOW = datetime(2026, 8, 25, 9, 0)
THRESHOLDS = Thresholds(defer=0.6, confirm=0.85)


def build(scripted, tmp_path, logger=None):
    storage = Storage(tmp_path / "t.db")
    return Assistant(config=Config(), storage=storage,
                     classifier=FakeClassifier(scripted), thresholds=THRESHOLDS,
                     logger=logger)


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- one line per turn, and the predicted/ground-truth separation --------

def test_one_turn_appends_exactly_one_line(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"what time is it": ("get_time", 0.98)}, tmp_path, logger=logger)

    a.handle("what time is it", NOW)

    lines = read_lines(log_path)
    assert len(lines) == 1


def test_logged_line_carries_predicted_intent_and_not_intent(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"what time is it": ("get_time", 0.98)}, tmp_path, logger=logger)

    a.handle("what time is it", NOW)

    entry = read_lines(log_path)[0]
    assert entry["predicted_intent"] == "get_time"
    assert "intent" not in entry, (
        "the log must never carry a ground-truth `intent` on its own -- "
        "that would make an unreviewed prediction look like a training label"
    )
    assert entry["transcript"] == "what time is it"
    assert entry["confidence"] == pytest.approx(0.98)


# --- deferred / re-prompted turns are logged too (the most valuable rows) -

def test_first_strike_reprompt_turn_is_logged(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"mumble mumble": ("out_of_scope", 0.2)}, tmp_path, logger=logger)

    result = a.handle("mumble mumble", NOW)

    assert result.reprompt is True
    entry = read_lines(log_path)[0]
    assert entry["reprompt"] is True
    assert entry["handled"] is True
    assert entry["deferred"] is False
    assert "intent" not in entry


def test_second_strike_llm_escalation_turn_is_also_logged(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"mumble mumble": ("out_of_scope", 0.2),
              "still mumbling": ("out_of_scope", 0.2)}, tmp_path, logger=logger)

    a.handle("mumble mumble", NOW)
    second = a.handle("still mumbling", NOW)

    assert second.handled is False
    lines = read_lines(log_path)
    assert len(lines) == 2
    assert lines[1]["handled"] is False
    assert lines[1]["deferred"] is True
    assert "intent" not in lines[1]


def test_needs_confirmation_turn_and_its_resolution_are_both_logged(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"drop the mom thing": ("cancel_reminder", 0.7),
              "yes": ("affirm", 0.9)}, tmp_path, logger=logger)
    a.storage.add_reminder("call mom", NOW + timedelta(hours=1))

    a.handle("drop the mom thing", NOW)
    a.handle("yes", NOW + timedelta(seconds=2))

    lines = read_lines(log_path)
    assert lines[0]["needs_confirmation"] is True
    # The resolving turn's `predicted_intent` reflects the action actually
    # executed (cancel_reminder) -- what the user answered lives separately
    # in `resolved_answer`, since those are two different pieces of signal.
    assert lines[1]["predicted_intent"] == "cancel_reminder"
    assert lines[1]["resolved_answer"] == "affirm"


# --- repeat detection -----------------------------------------------------

def test_repeat_within_window_is_flagged_against_the_previous_entry(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    first_id = logger.log(text="set a reminder for the thing", predicted_intent="out_of_scope",
                          confidence=0.3, now=NOW, handled=True, reprompt=True)
    logger.log(text="set a reminder for the thing", predicted_intent="out_of_scope",
              confidence=0.3, now=NOW + timedelta(seconds=3), handled=True, reprompt=True)

    lines = read_lines(log_path)
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert len(corrections) == 1
    assert corrections[0]["repeat_within_seconds"] == pytest.approx(3.0)
    # A correction record is never a candidate training row.
    assert "intent" not in corrections[0]
    assert "transcript" not in corrections[0]


def test_repeat_outside_the_window_is_not_flagged(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    first_id = logger.log(text="what's the weather", predicted_intent="out_of_scope",
                          confidence=0.3, now=NOW, handled=True, reprompt=True)
    logger.log(text="what's the weather", predicted_intent="out_of_scope",
              confidence=0.3, now=NOW + timedelta(minutes=5), handled=True, reprompt=True)

    lines = read_lines(log_path)
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert corrections == []


def test_different_utterances_are_not_flagged_as_a_repeat(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    first_id = logger.log(text="what time is it", predicted_intent="get_time",
                          confidence=0.95, now=NOW, handled=True)
    logger.log(text="what's on my calendar", predicted_intent="query_calendar",
              confidence=0.95, now=NOW + timedelta(seconds=2), handled=True)

    lines = read_lines(log_path)
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert corrections == []


def test_end_to_end_repeat_via_assistant_handle(tmp_path):
    """The same near-duplicate text, spoken twice in a row through the real
    Assistant.handle path, produces a repeat correction record."""
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"turn off the lights": ("out_of_scope", 0.2)}, tmp_path, logger=logger)

    a.handle("turn off the lights", NOW)
    a.handle("turn off the lights", NOW + timedelta(seconds=2))

    lines = read_lines(log_path)
    first_id = lines[0]["id"]
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert len(corrections) == 1
    assert corrections[0]["repeat_within_seconds"] == pytest.approx(2.0)


# --- user_negation ----------------------------------------------------------

def test_confident_deny_right_after_a_handled_action_flags_negation(tmp_path):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    first_id = logger.log(text="remind me to call mom", predicted_intent="add_reminder",
                          confidence=0.97, now=NOW, handled=True, needs_confirmation=False)
    logger.log(text="no that's wrong", predicted_intent="deny",
              confidence=0.9, now=NOW + timedelta(seconds=2), handled=True)

    lines = read_lines(log_path)
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert len(corrections) == 1
    assert corrections[0]["user_negation"] is True


def test_deny_that_resolves_a_confirmation_prompt_is_not_flagged_as_negation(tmp_path):
    """A deny answering 'you sure you want to cancel that?' is the flow
    working as intended, not a miss -- must not be conflated with an
    unprompted negation."""
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    first_id = logger.log(text="drop the mom thing", predicted_intent="cancel_reminder",
                          confidence=0.7, now=NOW, handled=True, needs_confirmation=True)
    logger.log(text="no", predicted_intent="deny",
              confidence=0.95, now=NOW + timedelta(seconds=2), handled=True)

    lines = read_lines(log_path)
    corrections = [line for line in lines if line.get("correction_for") == first_id]
    assert corrections == []


# --- skill_error ------------------------------------------------------------

def test_skill_error_is_captured_on_the_entry(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)
    a = build({"add a note": ("add_note", 0.95)}, tmp_path, logger=logger)

    def boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(a, "_run_skill", boom)
    a.handle("add a note", NOW)

    entry = read_lines(log_path)[0]
    assert "disk exploded" in entry["skill_error"]


# --- the logging fault must never break a spoken turn ----------------------

def test_a_logging_failure_never_breaks_the_turn(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log.jsonl"
    logger = TurnLogger(log_path)

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(logger, "log", explode)
    a = build({"what time is it": ("get_time", 0.98)}, tmp_path, logger=logger)

    result = a.handle("what time is it", NOW)

    assert result.handled is True
    assert result.intent == "get_time"
    assert "the" in result.reply.lower() or result.reply  # a real, normal reply


def test_a_write_failure_inside_turnlogger_never_raises(tmp_path):
    """Belt-and-braces: even calling TurnLogger.log() directly against an
    unwritable path must not raise -- the caller does not have to know
    that the disk is misbehaving."""
    unwritable_dir = tmp_path / "not_a_directory"
    unwritable_dir.write_text("i am a file, not a directory")
    logger = TurnLogger(unwritable_dir / "learning_log.jsonl")

    entry_id = logger.log(text="hello", predicted_intent="greeting",
                          confidence=0.9, now=NOW, handled=True)

    assert entry_id is None  # write failed, but nothing raised


def test_assistant_with_no_logger_works_exactly_as_before(tmp_path):
    a = build({"what time is it": ("get_time", 0.98)}, tmp_path, logger=None)
    result = a.handle("what time is it", NOW)
    assert result.handled is True
    assert result.intent == "get_time"


# --- location & privacy -----------------------------------------------------

def test_data_directory_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/learning_log.jsonl"],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, "data/learning_log.jsonl must be gitignored"
