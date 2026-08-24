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


# --- Regression tests: defects found in code review of Task 3 ---

def test_invalid_hour_returns_none_not_valueerror():
    """STT noise can produce out-of-range hours; extract_time must return
    None, never raise ValueError out of .replace()."""
    assert extract_time("tomorrow at 25", MORNING) is None
    assert extract_time("on friday at 30", MORNING) is None
    assert extract_time("2026-08-30 at 44", MORNING) is None
    assert extract_time("tomorrow at 13 pm", MORNING) is None


def test_invalid_iso_date_returns_none_not_valueerror():
    """An impossible calendar date (month 13, day 45) must not raise out of
    strptime, and must not fall back to a weaker pattern matching a valid
    time elsewhere in the string."""
    assert extract_time("2026-13-45 at 4pm", MORNING) is None


@pytest.mark.parametrize("text,now,expected", [
    ("at 7", MORNING, datetime(2026, 8, 24, 19, 0)),
    ("tomorrow at 7", MORNING, datetime(2026, 8, 25, 19, 0)),
    ("at 8", MORNING, datetime(2026, 8, 24, 20, 0)),
    ("tomorrow at 8", MORNING, datetime(2026, 8, 25, 20, 0)),
])
def test_ambiguous_hour_7_and_8_agree_with_and_without_a_day(text, now, expected):
    """Hours 7 and 8 must resolve to the same meridiem whether or not a day
    word is attached -- 'at 7' and 'tomorrow at 7' both mean 7 PM."""
    slot = extract_time(text, now)
    assert slot is not None, f"no time found in {text!r}"
    assert slot.when == expected


def test_today_with_explicit_past_time_stays_today_not_rolled_forward():
    """'today' is taken literally: if the stated clock time has already
    passed, the reminder still resolves to today rather than silently
    rolling to tomorrow."""
    slot = extract_time("today at 4", datetime(2026, 8, 24, 20, 0))
    assert slot is not None
    assert slot.when == datetime(2026, 8, 24, 16, 0)

    slot2 = extract_time("today at 9", datetime(2026, 8, 24, 14, 30))
    assert slot2 is not None
    assert slot2.when == datetime(2026, 8, 24, 9, 0)


def test_meridiem_time_picked_up_without_at_keyword():
    """An explicit-meridiem time attached to a day word must be picked up
    even when it isn't introduced by 'at' -- regardless of whether the time
    leads or trails the day word."""
    slot = extract_time("meeting 3pm tomorrow", MORNING)
    assert slot is not None
    assert slot.when == datetime(2026, 8, 25, 15, 0)
    assert slot.residual == "meeting"

    slot2 = extract_time("friday 4pm", MORNING)
    assert slot2 is not None
    assert slot2.when == datetime(2026, 8, 28, 16, 0)
    assert slot2.residual == ""


# NOTE: round-1's test_tonight_at_12_is_midnight_not_noon asserted that
# "tonight at 12" resolves to 2026-08-24 00:00 (today, 9 hours in the past
# relative to MORNING). Round-2 review found that value itself wrong: a
# bare 12 alongside "tonight" is midnight of the FOLLOWING calendar day,
# not today's already-past midnight. That test is superseded by the
# parametrised one below, which pins the corrected value together with the
# rest of the tonight small-hours rollover.
@pytest.mark.parametrize("text,now,expected", [
    ("tonight at 12", MORNING, datetime(2026, 8, 25, 0, 0)),
    ("tonight at 12 am", MORNING, datetime(2026, 8, 25, 0, 0)),
    ("tonight at 1", MORNING, datetime(2026, 8, 25, 1, 0)),
    ("tonight at 3", MORNING, datetime(2026, 8, 25, 3, 0)),
    ("tonight at 6", MORNING, datetime(2026, 8, 24, 18, 0)),
    ("tonight at 11", MORNING, datetime(2026, 8, 24, 23, 0)),
    ("tonight", MORNING, datetime(2026, 8, 24, 20, 0)),
    ("tonight at 8 pm", MORNING, datetime(2026, 8, 24, 20, 0)),
    # Pinned from the other side too: at 23:00, "tonight at 1" is still
    # ~2 hours away, on the following calendar date.
    ("tonight at 1", datetime(2026, 8, 24, 23, 0), datetime(2026, 8, 25, 1, 0)),
])
def test_tonight_spans_into_small_hours_of_the_following_day(text, now, expected):
    """'tonight' spans from evening into the small hours of the FOLLOWING
    calendar day: hours 6-11 stay this evening (PM, today's date); 12 and
    1-5 are that night's midnight/small hours (AM, tomorrow's date), even
    with no meridiem given. An explicit meridiem always wins over this
    default ('tonight at 8 pm' stays today 20:00)."""
    slot = extract_time(text, now)
    assert slot is not None, f"no time found in {text!r}"
    assert slot.when == expected


@pytest.mark.parametrize("text,now,expected", [
    ("at 12", MORNING, datetime(2026, 8, 24, 12, 0)),
    ("at 12", datetime(2026, 8, 24, 22, 0), datetime(2026, 8, 25, 0, 0)),
    ("tomorrow at 12", MORNING, datetime(2026, 8, 25, 12, 0)),
])
def test_bare_and_tomorrow_hour_12_unaffected_by_tonight_fix(text, now, expected):
    """The tonight-specific small-hours rollover must not leak into the
    unrelated bare-time next-occurrence rule or the tomorrow smart default
    -- both continue to treat a bare 12 as noon-ish, not midnight."""
    slot = extract_time(text, now)
    assert slot is not None, f"no time found in {text!r}"
    assert slot.when == expected
