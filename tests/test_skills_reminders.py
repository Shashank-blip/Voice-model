from datetime import datetime

import pytest

from jarvis_nlu.skills import reminders
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_strips_the_lead_in_and_the_time(store):
    reply = reminders.add(store, "remind me to call mom at 6 pm", NOW)
    saved = store.list_reminders()[0]
    assert saved.text == "call mom"
    assert saved.due_at == datetime(2026, 8, 24, 18, 0)
    assert "call mom" in reply


def test_add_handles_bare_time_without_a_day(store):
    """The bug: this was rejected outright before."""
    reminders.add(store, "remind me to call mom at 6", NOW)
    assert store.list_reminders()[0].due_at == datetime(2026, 8, 24, 18, 0)


def test_add_keeps_apostrophes_in_the_body(store):
    """The greeting-hijack bug phrase must survive as a real reminder."""
    reminders.add(store, "remind me I'm meeting Bob tomorrow at 5 pm", NOW)
    saved = store.list_reminders()[0]
    assert "meeting Bob" in saved.text
    assert saved.due_at == datetime(2026, 8, 25, 17, 0)


def test_needs_time_detects_a_missing_slot():
    assert reminders.needs_time("remind me to buy milk", NOW) is True
    assert reminders.needs_time("remind me to buy milk at 6", NOW) is False


def test_add_without_a_time_saves_undated(store):
    reply = reminders.add(store, "remind me to buy milk", NOW)
    saved = store.list_reminders()[0]
    assert saved.text == "buy milk"
    assert saved.due_at is None
    assert "when" in reply.lower()


def test_list_pending_says_so_when_empty(store):
    assert "no" in reminders.list_pending(store, NOW).lower()


def test_list_pending_includes_saved_items(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reminders.add(store, "remind me to pay rent tomorrow at 9 am", NOW)
    reply = reminders.list_pending(store, NOW)
    assert "call mom" in reply
    assert "pay rent" in reply


def test_cancel_removes_a_matching_reminder(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reply = reminders.cancel(store, "forget the mom reminder", NOW)
    assert store.list_reminders() == []
    assert "call mom" in reply


def test_cancel_reports_when_nothing_matches(store):
    reply = reminders.cancel(store, "cancel the dentist reminder", NOW)
    assert "couldn't find" in reply.lower() or "could not find" in reply.lower()


def test_cancel_asks_when_multiple_match(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reminders.add(store, "remind me to call mom back tomorrow at 9 am", NOW)
    reply = reminders.cancel(store, "cancel the call mom reminder", NOW)
    assert len(store.list_reminders()) == 2  # nothing destroyed on ambiguity
    assert "which" in reply.lower()


@pytest.mark.parametrize("phrase", [
    "scratch that mom reminder",
    "scrap that mom reminder",
    "never mind the mom reminder",
    "nix the mom reminder",
    "kill the mom reminder",
    "clear the mom reminder",
])
def test_cancel_recognises_verbs_the_classifier_now_recognises(store, phrase):
    """The intent classifier now recognises these as cancel_reminder; the
    skill's lead-in regex has to keep up or a correctly-routed turn still
    fails to find the reminder it named (task-12 review round 2,
    Important 2)."""
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reply = reminders.cancel(store, phrase, NOW)
    assert store.list_reminders() == []          # the reminder was actually cancelled
    assert "call mom" in reply


def test_add_reply_reads_naturally_for_a_verb_phrase_body(store):
    reply = reminders.add(store, "remind me to call mom at 6 pm", NOW)
    assert reply == "You got it — call mom, at 6:00 PM on Monday."


def test_add_reply_reads_naturally_for_a_clause_body(store):
    reply = reminders.add(store, "remind me I'm meeting Bob tomorrow at 5 pm", NOW)
    assert reply == "You got it — I'm meeting Bob, at 5:00 PM on Tuesday."


def test_add_reply_drops_placeholder_body_entirely(store):
    reply = reminders.add(store, "remind me at 6", NOW)
    assert reply == "You got it — I'll give you a nudge at 6:00 PM on Monday."


def test_list_pending_reads_naturally_for_a_verb_phrase_body(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reply = reminders.list_pending(store, NOW)
    assert reply == "Here's what you've got: call mom at 6:00 PM on Monday."


def test_list_pending_reads_naturally_for_a_clause_body(store):
    reminders.add(store, "remind me I'm meeting Bob tomorrow at 5 pm", NOW)
    reply = reminders.list_pending(store, NOW)
    assert reply == "Here's what you've got: I'm meeting Bob at 5:00 PM on Tuesday."


def test_list_pending_drops_placeholder_body_entirely(store):
    reminders.add(store, "remind me at 6", NOW)
    reply = reminders.list_pending(store, NOW)
    assert reply == "Here's what you've got: a reminder at 6:00 PM on Monday."
