from datetime import datetime

import pytest

from jarvis_nlu.skills import calendar
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_event_strips_lead_in_and_time(store):
    calendar.add(store, "add dentist appointment today at 4 pm to my calendar", NOW)
    events = store.events_on(NOW.date())
    assert len(events) == 1
    assert events[0].title == "dentist appointment"
    assert events[0].starts_at == datetime(2026, 8, 24, 16, 0)


def test_add_event_without_time_asks_instead_of_saving(store):
    reply = calendar.add(store, "put lunch with Sam on my calendar", NOW)
    assert store.events_on(NOW.date()) == []
    assert "when" in reply.lower()


def test_query_reports_clear_day(store):
    assert "clear" in calendar.query(store, "what is on my calendar today", NOW).lower()


def test_query_lists_todays_events_in_order(store):
    calendar.add(store, "add standup today at 9:30 am to my calendar", NOW)
    calendar.add(store, "add dentist today at 4 pm to my calendar", NOW)
    reply = calendar.query(store, "what is on my calendar today", NOW)
    assert reply.index("standup") < reply.index("dentist")


def test_query_respects_tomorrow(store):
    calendar.add(store, "add dentist tomorrow at 4 pm to my calendar", NOW)
    assert "dentist" in calendar.query(store, "what's on my calendar tomorrow", NOW)
    assert "clear" in calendar.query(store, "what's on my calendar today", NOW).lower()


def test_add_event_full_reply_string(store):
    reply = calendar.add(store, "add dentist appointment today at 4 pm to my calendar", NOW)
    assert reply == "Got it — dentist appointment at 4:00 PM on Monday."


def test_add_event_without_time_full_reply_string(store):
    reply = calendar.add(store, "put lunch with Sam on my calendar", NOW)
    assert reply == "Sure thing — when's that happenin', sugar?"


def test_query_full_reply_string_with_events(store):
    calendar.add(store, "add dentist appointment today at 4 pm to my calendar", NOW)
    reply = calendar.query(store, "what is on my calendar today", NOW)
    assert reply == "Today you've got: dentist appointment at 4:00 PM."


def test_query_full_reply_string_clear_day(store):
    reply = calendar.query(store, "what is on my calendar today", NOW)
    assert reply == "Today's lookin' clear, sugar."
