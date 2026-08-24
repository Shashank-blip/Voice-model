from __future__ import annotations

import re

from jarvis_nlu.responses import ResponsePool

_TIRED = re.compile(r"\b(tired|exhausted|sleepy|knackered|wiped|drained|beat|worn out)\b", re.I)
_NEGATIVE = re.compile(
    r"\b(bad|awful|terrible|rough|stressed|anxious|sad|down|lousy|"
    r"not great|not good|could be better|meh|struggling|overwhelmed)\b", re.I)


def classify_mood(text: str) -> str:
    """Coarse polarity for `user_mood`. Tired is checked first because
    'tired' and 'not great' often co-occur and tired is the more actionable
    reading. Defaults to positive -- an over-cheery reply is a smaller failure
    than falsely commiserating."""
    if _TIRED.search(text):
        return "tired"
    if _NEGATIVE.search(text):
        return "negative"
    return "positive"


def respond(intent_value: str, text: str, pool: ResponsePool) -> str:
    if intent_value == "user_mood":
        return pool.pick(f"user_mood_{classify_mood(text)}")
    return pool.pick(intent_value)
