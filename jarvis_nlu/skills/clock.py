from __future__ import annotations

from datetime import datetime


def get_time(now: datetime) -> str:
    # %#I is the Windows form of %-I (no zero padding); fall back for other platforms.
    try:
        pretty = now.strftime("%#I:%M %p")
    except ValueError:
        pretty = now.strftime("%-I:%M %p")
    return f"It's {pretty}."


def get_date(now: datetime) -> str:
    return f"It's {now.strftime('%A, %B')} {now.day}."
