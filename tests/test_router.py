from datetime import datetime

import pytest

from jarvis_nlu.config import Config
from jarvis_nlu.model import FakeClassifier, Thresholds
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)
THRESHOLDS = Thresholds(defer=0.6, confirm=0.85)


def build(scripted, tmp_path):
    storage = Storage(tmp_path / "t.db")
    return Assistant(config=Config(), storage=storage,
                     classifier=FakeClassifier(scripted), thresholds=THRESHOLDS)


def test_low_confidence_defers_to_the_llm(tmp_path):
    a = build({"who won the world cup": ("out_of_scope", 0.2)}, tmp_path)
    result = a.handle("who won the world cup", NOW)
    assert result.handled is False
    assert result.reply is None


def test_out_of_scope_defers_even_when_confident(tmp_path):
    a = build({"explain recursion": ("out_of_scope", 0.99)}, tmp_path)
    assert a.handle("explain recursion", NOW).handled is False


def test_confident_chore_is_handled_locally(tmp_path):
    a = build({"remind me to call mom at 6 pm": ("add_reminder", 0.97)}, tmp_path)
    result = a.handle("remind me to call mom at 6 pm", NOW)
    assert result.handled is True
    assert "call mom" in result.reply
    assert result.intent == "add_reminder"


def test_calendar_query_is_not_shadowed_by_event_add(tmp_path):
    """The confirmed regex bug. A classifier cannot shadow, but assert it."""
    a = build({
        "add dentist today at 4 pm to my calendar": ("add_calendar_event", 0.96),
        "what is on my calendar today": ("query_calendar", 0.95),
    }, tmp_path)
    a.handle("add dentist today at 4 pm to my calendar", NOW)
    result = a.handle("what is on my calendar today", NOW)
    assert result.intent == "query_calendar"
    assert "dentist" in result.reply


def test_destructive_intent_below_confirm_threshold_asks_first(tmp_path):
    a = build({"drop the mom thing": ("cancel_reminder", 0.7)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    result = a.handle("drop the mom thing", NOW)
    assert result.needs_confirmation is True
    assert result.handled is True
    assert len(a.storage.list_reminders()) == 1  # nothing destroyed yet
    # Full-string pin -- a substring check let two earlier tasks ship
    # malformed spoken text; this is read aloud verbatim.
    assert result.reply == "Just to be sure, sugar — you want me to drop the mom thing?"


def test_confirming_executes_the_pending_action(tmp_path):
    a = build({"drop the mom reminder": ("cancel_reminder", 0.7),
               "yes": ("affirm", 0.99)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    a.handle("drop the mom reminder", NOW)
    result = a.handle("yes", NOW)
    assert a.storage.list_reminders() == []
    # Full-string pin on the post-execution reply too.
    assert result.reply == "Done, I've dropped the reminder to call mom."


def test_denying_abandons_the_pending_action(tmp_path):
    from jarvis_nlu.responses import POOLS
    a = build({"drop the mom reminder": ("cancel_reminder", 0.7),
               "no": ("deny", 0.99)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    a.handle("drop the mom reminder", NOW)
    result = a.handle("no", NOW)
    assert len(a.storage.list_reminders()) == 1
    # The deny reply is pool-randomized -- pin it to the known closed set
    # rather than a single string, but never accept an arbitrary one.
    assert result.reply in POOLS["deny"]


def test_repeat_last_replays_the_previous_reply(tmp_path):
    a = build({"what time is it": ("get_time", 0.99),
               "say that again": ("repeat_last", 0.99)}, tmp_path)
    first = a.handle("what time is it", NOW).reply
    assert a.handle("say that again", NOW).reply == first


def test_repeat_last_with_nothing_said_yet_speaks_up_rather_than_crashing(tmp_path):
    a = build({"say that again": ("repeat_last", 0.99)}, tmp_path)
    result = a.handle("say that again", NOW)
    assert result.reply == "I haven't said anything yet, sugar."


def test_smalltalk_is_handled_with_zero_deferral(tmp_path):
    a = build({"hey there": ("greeting", 0.98)}, tmp_path)
    result = a.handle("hey there", NOW)
    assert result.handled is True
    assert result.reply


def test_empty_input_is_not_handled(tmp_path):
    a = build({}, tmp_path)
    assert a.handle("   ", NOW).handled is False


def test_skill_exception_returns_an_apology_not_a_crash(tmp_path, monkeypatch):
    from jarvis_nlu.skills import clock
    monkeypatch.setattr(clock, "get_time",
                        lambda now: (_ for _ in ()).throw(RuntimeError("boom")))
    a = build({"what time is it": ("get_time", 0.99)}, tmp_path)
    result = a.handle("what time is it", NOW)
    assert result.handled is True
    assert result.reply  # a spoken apology, not an exception
