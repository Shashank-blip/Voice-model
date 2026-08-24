"""Rule-based datetime extraction.

Deliberately hand-written rather than `dateparser`: the grammar here is small
and closed, and we need failure modes we control and can test exhaustively.
`extract_time` returns both the resolved datetime and the text with the time
expression removed -- that residual becomes the reminder body.

Two disambiguation rules matter more than the rest of the grammar:

1. Bare time, no day given ("at 6"): resolve to the NEXT occurrence of that
   clock time, choosing whichever of the AM/PM readings comes soonest after
   `now`, rolling into tomorrow if both of today's readings have passed.
   This is the rule that fixes the production bug where "remind me to call
   mom at 6 pm" (or without the "pm") used to be rejected outright.

2. Bare time WITH a day already pinned down ("tomorrow at 6", "on monday at
   9"): there is no "now" to compare against on a day that isn't today, so
   we fall back to a typical-daytime-scheduling default instead: hours 1-8
   mean afternoon/evening, hours 9-11 mean morning. This boundary must agree
   with the no-day next-occurrence rule above at every hour it covers (e.g.
   "at 7" and "tomorrow at 7" must both land on 7 PM) -- that's why it's 8,
   not 6: at MORNING (09:00) a standalone "at 7" or "at 8" has already
   missed its AM reading, so next-occurrence picks PM for both; the fixed
   default has to pick PM for both too, with no `now` to consult.
   "tonight" overrides this to always mean evening, regardless of the hour
   (including 12, which reads as midnight in that context, not noon).

A named weekday that matches today's weekday ("monday" said on a Monday)
always means next week, never today -- `_resolve_weekday` enforces that by
treating a same-day match as 7 days out rather than 0.

"today" and "tonight" are taken literally: if the stated time has already
passed, the reminder still resolves to today rather than silently rolling
to tomorrow. Saying "today" and having it fire immediately is reasonable
feedback, not a bug to route around -- unlike the no-day "at 6" case, where
there's no day word at all to anchor a literal reading, so we do search
forward instead. See `_resolve_relday`.

An explicit-meridiem time doesn't have to be introduced by "at" to be
picked up: "friday 4pm" and "meeting 3pm tomorrow" both carry an
unambiguous time via am/pm alone. `_WEEKDAY_TRAIL` and `_RELDAY_LEAD` catch
those two shapes; ambiguous bare times (no meridiem) still require "at" to
avoid treating a random trailing number as a clock time.

`extract_time` never raises: STT input is arbitrary noise, and an
out-of-range hour or an impossible calendar date must resolve to `None`
rather than propagate a `ValueError`. If the highest-priority pattern that
matches turns out to be semantically invalid, the whole utterance is
treated as unparseable rather than falling back to guess with a
lower-priority pattern against what's left of the text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}

DAYPARTS = {"morning": (9, 0), "afternoon": (14, 0), "evening": (19, 0),
            "night": (20, 0), "tonight": (20, 0)}

# Shared clock fragment: hour is required, minute and meridiem are optional.
_TIME = r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)?"
# Same, but meridiem is required -- used where "at" is absent, so only an
# unambiguous (am/pm-qualified) time is allowed to count as a time at all.
_TIME_REQ_MERIDIEM = r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)"

_OFFSET = (r"\bin\s+(?:(?P<half>half\s+an?\s+hour)"
           r"|(?P<amount>a|an|\d+)\s+(?P<unit>minutes?|mins?|hours?|hrs?|days?))\b")
_ISO = r"\b(?P<date>\d{4}-\d{2}-\d{2})\b(?:\s+at\s+" + _TIME + r")?"
_WEEKDAY = (r"\b(?P<next>next\s+)?(?:on\s+)?(?P<weekday>" + "|".join(WEEKDAYS) + r")\b"
            r"(?:\s+at\s+" + _TIME + r")?")
# "friday 4pm" -- day word directly followed by an unambiguous time, no "at".
_WEEKDAY_TRAIL = (r"\b(?P<next>next\s+)?(?:on\s+)?(?P<weekday>" + "|".join(WEEKDAYS) + r")\b"
                  r"\s+" + _TIME_REQ_MERIDIEM)
_RELDAY = (r"\b(?P<relday>today|tomorrow|tonight)\b"
           r"(?:\s+(?P<daypart>morning|afternoon|evening|night))?"
           r"(?:\s+at\s+" + _TIME + r")?")
# "meeting 3pm tomorrow" -- unambiguous time directly followed by the day word.
_RELDAY_LEAD = r"\b" + _TIME_REQ_MERIDIEM + r"\s+(?P<relday>today|tomorrow|tonight)\b"
_BARETIME = r"\bat\s+" + _TIME + r"\b"


@dataclass(frozen=True)
class TimeSlot:
    when: datetime
    residual: str


def _apply_meridiem(hour: int, meridiem: str) -> int:
    """Fold a 12-hour `hour` (1-12) into 24-hour form given an explicit
    am/pm. 12 am is midnight (0); 12 pm is noon (12)."""
    meridiem = meridiem.replace(".", "").lower()
    if meridiem == "am":
        return 0 if hour == 12 else hour
    if meridiem == "pm":
        return 12 if hour == 12 else hour + 12
    return hour


def _smart_default_hour(hour: int) -> int:
    """No meridiem given, but the day is already pinned down some other way
    (a weekday name, 'today', 'tomorrow'): assume typical daytime scheduling
    rather than comparing against `now`. 1-8 reads as afternoon/evening;
    9-11 (and 12) stay as given, reading as morning/noon. The boundary sits
    at 8 (not 6) so it agrees with the no-day next-occurrence rule at every
    hour: at a typical 09:00 `now`, a bare "at 7" or "at 8" has already
    missed its AM reading and resolves to PM, so this fixed default must
    pick PM for 7 and 8 too."""
    if 1 <= hour <= 8:
        return hour + 12
    return hour


def _next_occurrence(now: datetime, hour: int, minute: int) -> datetime:
    """No meridiem, no day: pick whichever of the AM/PM readings of `hour`
    comes soonest strictly after `now`, rolling into tomorrow if both of
    today's readings have already passed."""
    am_hour, pm_hour = hour % 12, hour % 12 + 12
    candidates = [
        now.replace(hour=h, minute=minute, second=0, microsecond=0) + timedelta(days=offset)
        for offset in (0, 1) for h in (am_hour, pm_hour)
    ]
    return min(c for c in candidates if c > now)


def _roll_if_past(candidate: datetime, now: datetime) -> datetime:
    return candidate if candidate > now else candidate + timedelta(days=1)


def _safe_datetime(base: datetime, hour: int, minute: int) -> datetime | None:
    """`.replace(hour=.., minute=..)` raises ValueError on out-of-range STT
    noise (e.g. the 44 in a garbled 'tomorrow at 44'). Voice input is
    arbitrary noise, so treat that as unparseable instead of letting the
    exception escape `extract_time`."""
    try:
        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError:
        return None


def _clock_from(match: re.Match, *, ambiguous_pm: bool = False) -> tuple[int, int] | None:
    """Read hour/minute/meridiem off `match`, returning None if no time was
    captured at all. `ambiguous_pm` biases an unmarked hour towards PM (used
    for 'tonight'), otherwise the smart daytime default is applied."""
    if not match.group("hour"):
        return None
    hour, minute = int(match.group("hour")), int(match.group("minute") or 0)
    meridiem = match.group("meridiem")
    if meridiem:
        hour = _apply_meridiem(hour, meridiem)
    elif ambiguous_pm:
        # 'tonight' implies evening; an unqualified 12 in that context reads
        # as midnight (matching explicit '12 am'), not noon.
        if hour == 12:
            hour = 0
        elif hour < 12:
            hour += 12
    else:
        hour = _smart_default_hour(hour)
    return hour, minute


def _resolve_offset(match: re.Match, now: datetime) -> datetime:
    if match.group("half"):
        return now + timedelta(minutes=30)
    raw = match.group("amount").lower()
    amount = 1 if raw in ("a", "an") else int(raw)
    unit = match.group("unit").lower()
    if unit.startswith(("minute", "min")):
        return now + timedelta(minutes=amount)
    if unit.startswith(("hour", "hr")):
        return now + timedelta(hours=amount)
    return now + timedelta(days=amount)


def _resolve_iso(match: re.Match, now: datetime) -> datetime | None:
    try:
        day = datetime.strptime(match.group("date"), "%Y-%m-%d").date()
    except ValueError:
        return None  # e.g. "2026-13-45" -- syntactically date-shaped, not a real date
    clock = _clock_from(match)
    hour, minute = clock if clock else (now.hour, now.minute)
    return _safe_datetime(datetime.combine(day, datetime.min.time()), hour, minute)


def _resolve_weekday(match: re.Match, now: datetime) -> datetime | None:
    target = WEEKDAYS[match.group("weekday").lower()]
    ahead = (target - now.weekday()) % 7
    # Same weekday as today means next week, never today.
    ahead = ahead or 7
    if match.groupdict().get("next"):
        ahead += 7
    day = now + timedelta(days=ahead)
    clock = _clock_from(match)
    hour, minute = clock if clock else (now.hour, now.minute)
    return _safe_datetime(day, hour, minute)


def _resolve_relday(match: re.Match, now: datetime) -> datetime | None:
    word = match.group("relday").lower()
    # "today"/"tonight" are literal: if the stated time already passed, we
    # still resolve to today rather than silently rolling to tomorrow (see
    # module docstring for why that's the right call here specifically).
    base_day = now + timedelta(days=1) if word == "tomorrow" else now
    daypart = match.groupdict().get("daypart")
    if daypart:
        hour, minute = DAYPARTS[daypart.lower()]
    else:
        clock = _clock_from(match, ambiguous_pm=(word == "tonight"))
        if clock:
            hour, minute = clock
        elif word == "tonight":
            hour, minute = DAYPARTS["tonight"]
        else:
            hour, minute = 9, 0
    return _safe_datetime(base_day, hour, minute)


def _resolve_baretime(match: re.Match, now: datetime) -> datetime | None:
    hour, minute = int(match.group("hour")), int(match.group("minute") or 0)
    if hour > 23:
        return None
    meridiem = match.group("meridiem")
    if meridiem:
        if hour > 12:
            return None
        candidate = now.replace(hour=_apply_meridiem(hour, meridiem), minute=minute,
                                second=0, microsecond=0)
        return _roll_if_past(candidate, now)
    if hour > 12:
        # Already unambiguous 24-hour form (e.g. "at 13") -- no AM/PM to pick.
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return _roll_if_past(candidate, now)
    return _next_occurrence(now, hour, minute)


_MATCHERS = [
    (re.compile(_OFFSET, re.I), _resolve_offset),
    (re.compile(_ISO, re.I), _resolve_iso),
    # The no-"at" loose forms must be tried before their at-based/bare
    # counterparts: both can start at the same position ("friday" is a
    # prefix of "friday 4pm"), and the loose form is only correct when an
    # unambiguous meridiem time is actually attached.
    (re.compile(_WEEKDAY_TRAIL, re.I), _resolve_weekday),
    (re.compile(_WEEKDAY, re.I), _resolve_weekday),
    (re.compile(_RELDAY_LEAD, re.I), _resolve_relday),
    (re.compile(_RELDAY, re.I), _resolve_relday),
    (re.compile(_BARETIME, re.I), _resolve_baretime),
]


def _residual(text: str, match: re.Match) -> str:
    stripped = text[:match.start()] + " " + text[match.end():]
    stripped = re.sub(r"\s+", " ", stripped).strip()
    # Drop dangling prepositions left behind by the removal.
    stripped = re.sub(r"^(at|on|by|in)\s+", "", stripped, flags=re.I)
    stripped = re.sub(r"\s+(at|on|by|in)$", "", stripped, flags=re.I)
    return stripped.strip(" ,.")


def extract_time(text: str, now: datetime) -> TimeSlot | None:
    if not text:
        return None
    for pattern, resolver in _MATCHERS:
        match = pattern.search(text)
        if not match:
            continue
        when = resolver(match, now)
        if when is None:
            # The highest-priority pattern that matched turned out to be
            # semantically invalid (bad hour, impossible date) -- treat the
            # whole utterance as unparseable rather than falling through to
            # guess with a lower-priority pattern against what's left.
            return None
        return TimeSlot(when=when, residual=_residual(text, match))
    return None
