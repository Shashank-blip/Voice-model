import pytest

from jarvis_nlu.responses import ResponsePool
from jarvis_nlu.skills import smalltalk


@pytest.mark.parametrize("text,expected", [
    ("i'm good thanks", "positive"),
    ("doing great", "positive"),
    ("pretty bad honestly", "negative"),
    ("i'm stressed out", "negative"),
    ("i'm exhausted", "tired"),
    ("so sleepy today", "tired"),
    ("bit tired honestly", "tired"),
])
def test_classify_mood(text, expected):
    assert smalltalk.classify_mood(text) == expected


def test_unknown_mood_defaults_to_positive():
    assert smalltalk.classify_mood("mm") == "positive"


def test_respond_uses_mood_specific_pool():
    pool = ResponsePool({"user_mood_tired": ["rest up"]}, history=1)
    assert smalltalk.respond("user_mood", "i'm exhausted", pool) == "rest up"


def test_respond_uses_plain_pool_for_non_mood_intents():
    pool = ResponsePool({"greeting": ["hey there"]}, history=1)
    assert smalltalk.respond("greeting", "hello", pool) == "hey there"
