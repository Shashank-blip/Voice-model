from datetime import datetime

import pytest

from jarvis_nlu.slots import extract_time

MORNING = datetime(2026, 8, 24, 9, 0)    # Monday 09:00
EVENING = datetime(2026, 8, 24, 20, 0)   # Monday 20:00


@pytest.mark.parametrize("text,now,expected", [
    # Bare time -> next occurrence of that clock time.
    ("at 6", MORNING, datetime(2026, 8, 24, 18, 0)),
    ("at 6", EVENING, datetime(2026, 8, 25, 6, 0)),
    ("at 10", MORNING, datetime(2026, 8, 24, 10, 0)),
    # Explicit meridiem wins over the next-occurrence rule.
    ("at 6 pm", MORNING, datetime(2026, 8, 24, 18, 0)),
    ("at 6 am", EVENING, datetime(2026, 8, 25, 6, 0)),
    ("at 6:30 pm", MORNING, datetime(2026, 8, 24, 18, 30)),
    # Midnight / noon boundaries.
    ("at 12 am", MORNING, datetime(2026, 8, 25, 0, 0)),
    ("at 12 pm", MORNING, datetime(2026, 8, 24, 12, 0)),
    # Relative day + time.
    ("tomorrow at 6", MORNING, datetime(2026, 8, 25, 18, 0)),
    ("today at 4", MORNING, datetime(2026, 8, 24, 16, 0)),
    ("tonight at 8", MORNING, datetime(2026, 8, 24, 20, 0)),
    # Named weekday. Monday now; Friday is 4 days out.
    ("friday", MORNING, datetime(2026, 8, 28, 9, 0)),
    ("on monday at 9", MORNING, datetime(2026, 8, 31, 9, 0)),
    ("next friday", MORNING, datetime(2026, 9, 4, 9, 0)),
    # ISO date.
    ("2026-08-30 at 4pm", MORNING, datetime(2026, 8, 30, 16, 0)),
    # Offsets.
    ("in 20 minutes", MORNING, datetime(2026, 8, 24, 9, 20)),
    ("in 2 hours", MORNING, datetime(2026, 8, 24, 11, 0)),
    ("in half an hour", MORNING, datetime(2026, 8, 24, 9, 30)),
    # Dayparts.
    ("tomorrow morning", MORNING, datetime(2026, 8, 25, 9, 0)),
    ("tomorrow afternoon", MORNING, datetime(2026, 8, 25, 14, 0)),
    ("tomorrow evening", MORNING, datetime(2026, 8, 25, 19, 0)),
    ("tonight", MORNING, datetime(2026, 8, 24, 20, 0)),
])
def test_resolves_when(text, now, expected):
    slot = extract_time(text, now)
    assert slot is not None, f"no time found in {text!r}"
    assert slot.when == expected


@pytest.mark.parametrize("text,expected_residual", [
    ("call mom at 6 pm", "call mom"),
    ("at 6 pm call mom", "call mom"),
    ("call mom tomorrow at 6", "call mom"),
    ("in 20 minutes take the bread out", "take the bread out"),
    ("dentist appointment on friday at 4pm", "dentist appointment"),
])
def test_strips_time_expression_from_residual(text, expected_residual):
    slot = extract_time(text, MORNING)
    assert slot is not None
    assert slot.residual == expected_residual


@pytest.mark.parametrize("text", [
    "call mom",
    "what is on my calendar",
    "",
    "remind me to buy milk",
])
def test_returns_none_when_no_time_present(text):
    assert extract_time(text, MORNING) is None


def test_named_weekday_matching_today_goes_to_next_week():
    """'monday' said on a Monday means next Monday, not zero days away."""
    slot = extract_time("monday", MORNING)
    assert slot.when.date() == datetime(2026, 8, 31).date()
