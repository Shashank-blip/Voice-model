"""Task D: the review CLI.

This file is deliberately paranoid about the log's safety -- see
training/review.py's module docstring and jarvis_nlu/turnlog.py's. The
log is irreplaceable user data: it may be appended to, live, by a running
assistant while a human is reviewing it, so every write path here is
tested for "does the original survive" before "does labelling work".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jarvis_nlu.intents import ALL_INTENTS
from training.review import (
    Candidate,
    ConcurrentModificationError,
    ReviewSession,
    _build_menu,
    run_review,
)

NOW = "2026-08-25T09:00:00"


def write_log(path: Path, lines: list[dict | str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            if isinstance(line, str):
                handle.write(line + "\n")
            else:
                handle.write(json.dumps(line) + "\n")


def entry(id_, transcript, predicted_intent, confidence=0.9, **extra) -> dict:
    base = {
        "id": id_,
        "timestamp": NOW,
        "transcript": transcript,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "handled": True,
        "deferred": False,
        "reprompt": False,
        "needs_confirmation": False,
    }
    base.update(extra)
    return base


def correction(target_id, **signal) -> dict:
    return {"id": "c-" + target_id, "timestamp": NOW, "correction_for": target_id, **signal}


# --- selection ---------------------------------------------------------

def test_unlabelled_entries_are_listed(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["a"]


def test_labelled_entries_are_excluded(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "what time is it", "get_time"),
        entry("b", "add a note", "add_note", intent="add_note"),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["a"]


def test_correction_records_are_excluded(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "turn off the lights", "out_of_scope"),
        correction("a", repeat_within_seconds=2.0),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["a"]


def test_empty_or_missing_log_yields_no_candidates(tmp_path):
    log = tmp_path / "learning_log.jsonl"

    assert ReviewSession(log).load() == []


# --- ordering: likely-miss first ---------------------------------------

def test_reprompted_entry_sorts_before_confident_entry(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("confident", "what time is it", "get_time", confidence=0.98),
        entry("missed", "mumble mumble", "out_of_scope", confidence=0.3, reprompt=True),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["missed", "confident"]


def test_deferred_entry_sorts_before_confident_entry(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("confident", "what's my next reminder", "list_reminders", confidence=0.95),
        entry("missed", "still mumbling", "out_of_scope", confidence=0.3, deferred=True,
              handled=False),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["missed", "confident"]


def test_entry_with_a_correction_record_sorts_before_confident_entry(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("confident", "what's the date", "get_date", confidence=0.95),
        entry("missed", "remind me to call mom", "add_reminder", confidence=0.9),
        correction("missed", user_negation=True),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["missed", "confident"]


def test_skill_error_entry_sorts_before_confident_entry(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("confident", "what time is it", "get_time", confidence=0.95),
        entry("missed", "add a note", "add_note", confidence=0.9,
              skill_error="RuntimeError: disk exploded"),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["missed", "confident"]


def test_low_confidence_sorts_before_high_confidence_among_plain_entries(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("high", "what time is it", "get_time", confidence=0.98),
        entry("low", "add a note about eggs", "add_note", confidence=0.55),
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["low", "high"]


def test_repeat_signal_outranks_a_bare_low_confidence_entry(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("low_conf", "add a note about eggs", "add_note", confidence=0.55),
        entry("repeated", "turn off the lights", "out_of_scope", confidence=0.9),
        correction("repeated", repeat_within_seconds=1.5),
    ])

    candidates = ReviewSession(log).load()

    assert candidates[0].id == "repeated"


# --- labelling behaviour -------------------------------------------------

def test_confirming_writes_intent_equal_to_predicted_intent(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])

    run_review(log, input_func=lambda _prompt: "", print_func=lambda *_: None)

    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["intent"] == "get_time"


def test_correcting_writes_the_chosen_intent(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what's on my calendar", "get_time")])
    menu = _build_menu()
    correct_number = next(n for n, i in menu if i.value == "query_calendar")

    run_review(log, input_func=lambda _prompt: str(correct_number),
              print_func=lambda *_: None)

    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["intent"] == "query_calendar"


def test_skipping_leaves_the_entry_unchanged(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])
    original = log.read_text(encoding="utf-8")

    run_review(log, input_func=lambda _prompt: "s", print_func=lambda *_: None)

    assert log.read_text(encoding="utf-8") == original
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert "intent" not in lines[0]


def test_quit_saves_labels_entered_so_far(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "what time is it", "get_time"),
        entry("b", "what's the date", "get_date"),
    ])

    # Confirm the first (highest-priority-sorted) entry, then quit before
    # the second is ever shown.
    answers = iter(["", "q"])
    run_review(log, input_func=lambda _prompt: next(answers), print_func=lambda *_: None)

    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    labelled = {l["id"]: l.get("intent") for l in lines}
    assert labelled["a"] == "get_time"
    assert labelled["b"] is None


def test_running_counts_are_reported(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "what time is it", "get_time"),
        entry("b", "what's the date", "get_date"),
    ])
    printed = []

    run_review(log, input_func=lambda _prompt: "", print_func=printed.append)

    joined = "\n".join(printed)
    assert "2" in joined  # total candidates, and/or the final "labelled 2" tally
    assert "summary" in joined.lower() or "confirmed" in joined.lower()


# --- atomic rewrite --------------------------------------------------------

def test_rewrite_is_atomic_original_survives_a_failure_mid_write(tmp_path, monkeypatch):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])
    original = log.read_text(encoding="utf-8")

    session = ReviewSession(log)
    session.load()
    session.record("a", "get_time")

    def exploding_fsync(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", exploding_fsync)

    with pytest.raises(OSError):
        session.save()

    assert log.read_text(encoding="utf-8") == original, (
        "a failure between writing the temp file and the atomic rename must "
        "leave the original file completely untouched"
    )
    # No stray temp file left behind either.
    leftovers = [p for p in tmp_path.iterdir() if p != log]
    assert leftovers == [], f"temp file(s) not cleaned up: {leftovers}"


def test_successful_save_leaves_no_temp_file_behind(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])

    session = ReviewSession(log)
    session.load()
    session.record("a", "get_time")
    session.save()

    leftovers = [p for p in tmp_path.iterdir() if p != log]
    assert leftovers == []


# --- malformed lines survive ------------------------------------------------

def test_malformed_line_survives_a_round_trip(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "what time is it", "get_time"),
        "{not valid json at all",
    ])

    run_review(log, input_func=lambda _prompt: "", print_func=lambda *_: None)

    raw_lines = log.read_text(encoding="utf-8").splitlines()
    assert "{not valid json at all" in raw_lines


def test_malformed_line_is_not_offered_as_a_candidate(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [
        entry("a", "what time is it", "get_time"),
        "{not valid json at all",
    ])

    candidates = ReviewSession(log).load()

    assert [c.id for c in candidates] == ["a"]


# --- concurrent appends during the session ---------------------------------

def test_lines_appended_during_the_session_are_preserved(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])

    session = ReviewSession(log)
    session.load()
    session.record("a", "get_time")

    # Simulate the running assistant appending a brand-new turn while the
    # human is mid-review.
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry("b", "what's the date", "get_date")) + "\n")

    result = session.save()

    assert result.written is True
    assert result.appended_by_others == 1
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    ids = {l["id"] for l in lines}
    assert ids == {"a", "b"}
    labelled = {l["id"]: l.get("intent") for l in lines}
    assert labelled["a"] == "get_time"
    assert labelled["b"] is None  # untouched, appended after load()


def test_non_append_modification_refuses_to_write(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])

    session = ReviewSession(log)
    session.load()
    session.record("a", "get_time")

    # Something else rewrote an existing line underneath us -- not a pure
    # append. This must be detected and refused, not silently clobbered.
    write_log(log, [entry("a", "what time is it", "get_time", confidence=0.42)])

    with pytest.raises(ConcurrentModificationError):
        session.save()

    # Refusing means: the file as some OTHER writer left it, unclobbered by us.
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert "intent" not in lines[0]
    assert lines[0]["confidence"] == pytest.approx(0.42)


def test_run_review_reports_refusal_without_raising(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])
    printed = []

    # Patch ReviewSession.save (via run_review's internal session) is
    # awkward from outside; instead exercise the same underlying scenario
    # through the public run_review entry point using a second write
    # between the prompt answer and save by monkeypatching input_func to
    # mutate the file as a side effect of answering the one prompt.
    def answer_and_corrupt(_prompt):
        write_log(log, [entry("a", "what time is it", "get_time", confidence=0.11)])
        return ""

    summary = run_review(log, input_func=answer_and_corrupt, print_func=printed.append)

    assert summary.saved is False
    assert any("refus" in line.lower() for line in printed)


# --- the intent menu is never hardcoded -------------------------------------

def test_menu_matches_all_intents_exactly():
    menu = _build_menu()

    assert [intent for _n, intent in menu] == list(ALL_INTENTS)
    assert [n for n, _i in menu] == list(range(1, len(ALL_INTENTS) + 1))


def test_menu_entries_shown_to_the_user_include_every_intent_value(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    write_log(log, [entry("a", "what time is it", "get_time")])
    printed = []

    run_review(log, input_func=lambda _prompt: "q", print_func=printed.append)

    joined = "\n".join(printed)
    for intent in ALL_INTENTS:
        assert intent.value in joined
