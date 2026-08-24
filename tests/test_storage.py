from datetime import date, datetime

import pytest

from jarvis_nlu.storage import Storage


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


def test_add_and_list_reminder(store):
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    reminders = store.list_reminders()
    assert len(reminders) == 1
    assert reminders[0].text == "call mom"
    assert reminders[0].due_at == datetime(2026, 8, 24, 18, 0)
    assert reminders[0].delivered_at is None


def test_reminder_without_due_is_allowed(store):
    store.add_reminder("buy milk", None)
    assert store.list_reminders()[0].due_at is None


def test_due_reminders_excludes_future_delivered_and_cancelled(store):
    past = store.add_reminder("past", datetime(2026, 8, 24, 9, 0))
    store.add_reminder("future", datetime(2026, 8, 24, 23, 0))
    delivered = store.add_reminder("done", datetime(2026, 8, 24, 8, 0))
    cancelled = store.add_reminder("gone", datetime(2026, 8, 24, 8, 0))
    store.mark_delivered(delivered, datetime(2026, 8, 24, 8, 1))
    store.cancel_reminder(cancelled)

    due = store.due_reminders(datetime(2026, 8, 24, 12, 0))
    assert [r.id for r in due] == [past]


def test_undated_reminder_never_becomes_due(store):
    store.add_reminder("someday", None)
    assert store.due_reminders(datetime(2030, 1, 1)) == []


def test_cancel_returns_false_for_unknown_id(store):
    assert store.cancel_reminder(999) is False


def test_find_reminders_by_text_is_case_insensitive(store):
    store.add_reminder("Call Mom", datetime(2026, 8, 24, 18, 0))
    assert len(store.find_reminders_by_text("call mom")) == 1


def test_events_on_filters_by_day(store):
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    store.add_event("gym", datetime(2026, 8, 25, 7, 0))
    events = store.events_on(date(2026, 8, 24))
    assert [e.title for e in events] == ["dentist"]


def test_events_on_returns_chronological_order(store):
    store.add_event("late", datetime(2026, 8, 24, 18, 0))
    store.add_event("early", datetime(2026, 8, 24, 8, 0))
    assert [e.title for e in store.events_on(date(2026, 8, 24))] == ["early", "late"]


def test_notes_roundtrip_with_embedding(store):
    note_id = store.add_note("wifi password is hunter2")
    assert store.list_notes()[0].embedding is None
    store.set_note_embedding(note_id, b"\x00\x01")
    assert store.notes_with_embeddings()[0].embedding == b"\x00\x01"


def test_meta_roundtrip_and_overwrite(store):
    assert store.get_meta("last_brief_date") is None
    store.set_meta("last_brief_date", "2026-08-24")
    store.set_meta("last_brief_date", "2026-08-25")
    assert store.get_meta("last_brief_date") == "2026-08-25"


def test_reopening_database_is_idempotent(tmp_path):
    path = tmp_path / "test.db"
    first = Storage(path)
    first.add_reminder("persisted", None)
    first.close()

    second = Storage(path)
    assert len(second.list_reminders()) == 1
    second.close()


@pytest.mark.xfail(reason="needs Task 3 slots")
def test_migrates_legacy_minutes_schema(tmp_path):
    """miss-minutes' existing db has free-text `due` and no events table."""
    import sqlite3
    path = tmp_path / "minutes.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, due TEXT, created_at TEXT NOT NULL)"
    )
    legacy.execute(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    legacy.execute("INSERT INTO reminders (text, due, created_at) VALUES (?,?,?)",
                   ("call mom", "sometime next week", "2026-08-01T10:00:00"))
    legacy.execute("INSERT INTO notes (text, created_at) VALUES (?,?)",
                   ("old note", "2026-08-01T10:00:00"))
    legacy.commit()
    legacy.close()

    store = Storage(path)
    # No data discarded; unparseable due becomes NULL rather than dropping the row.
    assert [r.text for r in store.list_reminders()] == ["call mom"]
    assert store.list_reminders()[0].due_at is None
    assert [n.text for n in store.list_notes()] == ["old note"]
    store.close()

    # Migration is idempotent.
    again = Storage(path)
    assert len(again.list_reminders()) == 1
    again.close()
