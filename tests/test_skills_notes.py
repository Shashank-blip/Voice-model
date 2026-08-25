from datetime import datetime

import pytest

from jarvis_nlu.skills import notes
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_strips_the_lead_in(store):
    notes.add(store, "take a note buy milk", NOW)
    assert store.list_notes()[0].text == "buy milk"


@pytest.mark.parametrize("phrasing", [
    "note: the wifi password is hunter2",
    "make a note the wifi password is hunter2",
    "remember that the wifi password is hunter2",
])
def test_add_handles_several_lead_ins(store, phrasing):
    notes.add(store, phrasing, NOW)
    assert store.list_notes()[0].text == "the wifi password is hunter2"


def test_list_recent_says_so_when_empty(store):
    assert "no" in notes.list_recent(store, NOW).lower()


def test_list_recent_returns_newest_first(store):
    notes.add(store, "note first thing", NOW)
    notes.add(store, "note second thing", NOW)
    reply = notes.list_recent(store, NOW)
    assert reply.index("second thing") < reply.index("first thing")


def test_search_falls_back_to_substring_without_an_embedder(store):
    notes.add(store, "note the wifi password is hunter2", NOW)
    notes.add(store, "note buy the oat milk", NOW)
    reply = notes.search(store, "what did I note about the wifi")
    assert "hunter2" in reply
    assert "oat milk" not in reply


def test_search_reports_no_match(store):
    notes.add(store, "note buy milk", NOW)
    assert "couldn't find" in notes.search(store, "what did I note about taxes").lower()


def test_add_full_reply_string(store):
    reply = notes.add(store, "take a note buy milk", NOW)
    assert reply == "Noted, sugar."


def test_list_recent_full_reply_string(store):
    notes.add(store, "note first thing", NOW)
    notes.add(store, "note second thing", NOW)
    reply = notes.list_recent(store, NOW)
    assert reply == "Here's what you've noted: second thing; first thing."


def test_search_no_match_full_reply_string(store):
    notes.add(store, "note buy milk", NOW)
    reply = notes.search(store, "what did I note about taxes")
    assert reply == "I couldn't find a note about that, sugar."


def test_search_fallback_ignores_stopwords(store):
    notes.add(store, "note buy milk", NOW)
    notes.add(store, "note the wifi password is hunter2", NOW)
    notes.add(store, "note the bins go out tuesday", NOW)
    reply = notes.search(store, "what did I note about the wifi")
    assert reply == "You noted: the wifi password is hunter2."


def test_search_all_stopwords_asks_instead_of_matching_everything(store):
    notes.add(store, "note buy milk", NOW)
    notes.add(store, "note the wifi password is hunter2", NOW)
    reply = notes.search(store, "what did I note about that")
    assert reply == "What should I look for, sugar?"
