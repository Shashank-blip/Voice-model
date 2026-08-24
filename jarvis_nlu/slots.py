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
   we fall back to a typical-daytime-scheduling default instead: hours 1-6
   mean afternoon/evening, hours 7-11 mean morning. "tonight" overrides this
   to always mean evening, regardless of the hour.

A named weekday that matches today's weekday ("monday" said on a Monday)
always means next week, never today -- `_resolve_weekday` enforces that by
treating a same-day match as 7 days out rather than 0.
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

_OFFSET = (r"\bin\s+(?:(?P<half>half\s+an?\s+hour)"
           r"|(?P<amount>a|an|\d+)\s+(?P<unit>minutes?|mins?|hours?|hrs?|days?))\b")
_ISO = r"\b(?P<date>\d{4}-\d{2}-\d{2})\b(?:\s+at\s+" + _TIME + r")?"
_WEEKDAY = (r"\b(?P<next>next\s+)?(?:on\s+)?(?P<weekday>" + "|".join(WEEKDAYS) + r")\b"
            r"(?:\s+at\s+" + _TIME + r")?")
_RELDAY = (r"\b(?P<relday>today|tomorrow|tonight)\b"
           r"(?:\s+(?P<daypart>morning|afternoon|evening|night))?"
           r"(?:\s+at\s+" + _TIME + r")?")
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
    rather than comparing against `now`. 1-6 reads as afternoon/evening;
    7-11 (and 12) stay as given, reading as morning/noon."""
    if 1 <= hour <= 6:
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
        hour = hour if hour >= 12 else hour + 12
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
    day = datetime.strptime(match.group("date"), "%Y-%m-%d").date()
    clock = _clock_from(match)
    hour, minute = clock if clock else (now.hour, now.minute)
    return datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)


def _resolve_weekday(match: re.Match, now: datetime) -> datetime:
    target = WEEKDAYS[match.group("weekday").lower()]
    ahead = (target - now.weekday()) % 7
    # Same weekday as today means next week, never today.
    ahead = ahead or 7
    if match.group("next"):
        ahead += 7
    day = now + timedelta(days=ahead)
    clock = _clock_from(match)
    hour, minute = clock if clock else (now.hour, now.minute)
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _resolve_relday(match: re.Match, now: datetime) -> datetime:
    word = match.group("relday").lower()
    base_day = now + timedelta(days=1) if word == "tomorrow" else now
    daypart = match.group("daypart")
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
    return base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)


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
    (re.compile(_WEEKDAY, re.I), _resolve_weekday),
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
        if when is not None:
            return TimeSlot(when=when, residual=_residual(text, match))
    return None
