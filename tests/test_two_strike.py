"""Task F: two-strike deferral policy.

Below `defer`, the FIRST consecutive low-confidence turn is free: the
assistant re-prompts ("say that again, sugar?") rather than paying for an
LLM call to interpret what was probably a garbled Whisper transcript. Only
the SECOND consecutive low-confidence turn escalates to the LLM. A
confidently-predicted `out_of_scope` always escalates immediately -- it
genuinely needs the LLM, no re-prompt involved. Any successfully-handled
turn (a real answer, not a re-prompt) resets the strike counter to zero.

`Result.reprompt` distinguishes "speak this re-prompt" (handled=True,
reprompt=True) from "call the LLM" (handled=False) -- `handled` alone is
never overloaded to carry both meanings.
"""
from __future__ import annotations

from datetime import datetime

from jarvis_nlu.config import Config
from jarvis_nlu.model import FakeClassifier, Thresholds
from jarvis_nlu.responses import POOLS
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 25, 9, 0)
THRESHOLDS = Thresholds(defer=0.6, confirm=0.85)

LOW = ("out_of_scope", 0.2)  # below defer -- classifier is just unsure
HIGH_OOS = ("out_of_scope", 0.95)  # above defer, genuinely out of scope


def build(scripted, tmp_path):
    storage = Storage(tmp_path / "t.db")
    return Assistant(config=Config(), storage=storage,
                     classifier=FakeClassifier(scripted), thresholds=THRESHOLDS)


def test_first_low_confidence_turn_reprompts_without_calling_the_llm(tmp_path):
    a = build({"mumble mumble": LOW}, tmp_path)
    result = a.handle("mumble mumble", NOW)

    # It IS spoken (a re-prompt is a reply), so handled=True -- but reprompt
    # marks WHY, so the caller never mistakes this for "call the LLM".
    assert result.handled is True
    assert result.reprompt is True
    # Full-string membership, not substring -- a malformed spoken string
    # shipped before under a substring check that would have missed it.
    assert result.reply in POOLS["reprompt"]


def test_second_consecutive_low_confidence_turn_signals_the_llm(tmp_path):
    a = build({"mumble mumble": LOW, "still mumbling": LOW}, tmp_path)
    first = a.handle("mumble mumble", NOW)
    second = a.handle("still mumbling", NOW)

    assert first.reprompt is True
    # Second strike: this is the "call the LLM" signal -- handled=False,
    # and definitely not a second re-prompt.
    assert second.handled is False
    assert second.reprompt is False
    assert second.reply is None


def test_a_handled_turn_between_two_low_confidence_turns_resets_the_strike(tmp_path):
    a = build({
        "mumble mumble": LOW,
        "what time is it": ("get_time", 0.99),
        "still mumbling": LOW,
    }, tmp_path)

    first = a.handle("mumble mumble", NOW)
    handled = a.handle("what time is it", NOW)
    third = a.handle("still mumbling", NOW)

    assert first.reprompt is True
    assert handled.handled is True and handled.reprompt is False
    # Strike counter reset by the handled turn -- this is a FIRST strike
    # again, not a second, so it re-prompts rather than calling the LLM.
    assert third.handled is True
    assert third.reprompt is True
    assert third.reply in POOLS["reprompt"]


def test_confident_out_of_scope_signals_the_llm_immediately_with_no_reprompt(tmp_path):
    a = build({"who won the world cup": HIGH_OOS}, tmp_path)
    result = a.handle("who won the world cup", NOW)

    assert result.handled is False
    assert result.reprompt is False
    assert result.reply is None


def test_confident_out_of_scope_does_not_consume_or_leave_a_dangling_strike(tmp_path):
    """A confident out_of_scope turn is a different failure mode from
    'merely unsure' -- it must not itself count as (or be counted as) a
    low-confidence strike. A low-confidence turn right after it must still
    be treated as a FIRST strike."""
    a = build({
        "who won the world cup": HIGH_OOS,
        "mumble mumble": LOW,
    }, tmp_path)

    a.handle("who won the world cup", NOW)
    result = a.handle("mumble mumble", NOW)

    assert result.handled is True
    assert result.reprompt is True


def test_strike_counter_cycles_and_never_grows_unbounded_or_negative(tmp_path):
    a = build({"mumble mumble": LOW}, tmp_path)

    outcomes = [a.handle("mumble mumble", NOW) for _ in range(6)]
    # Strike 1 re-prompts, strike 2 signals the LLM and resets, strike 3
    # re-prompts again, and so on -- the pattern repeats rather than the
    # counter climbing forever.
    assert [(r.handled, r.reprompt) for r in outcomes] == [
        (True, True), (False, False),
        (True, True), (False, False),
        (True, True), (False, False),
    ]
    # The counter itself (internal, but this is exactly what "never
    # unbounded" means) is never allowed to exceed the strike threshold.
    assert 0 <= a._consecutive_unsure <= 1


def test_strike_counter_never_goes_negative_across_many_handled_turns(tmp_path):
    a = build({"what time is it": ("get_time", 0.99)}, tmp_path)
    for _ in range(5):
        result = a.handle("what time is it", NOW)
        assert result.handled is True and result.reprompt is False
        assert a._consecutive_unsure == 0


def test_reprompt_pool_has_at_least_eight_short_spoken_variants():
    variants = POOLS["reprompt"]
    assert len(variants) >= 8
    assert len(set(variants)) == len(variants)  # no accidental duplicates
    for line in variants:
        assert line and line.strip() == line  # no stray leading/trailing space
        assert len(line) <= 60  # short -- these are spoken aloud
