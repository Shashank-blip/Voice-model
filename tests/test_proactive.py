import threading
from datetime import datetime

import pytest

from jarvis_nlu.proactive import Scheduler
from jarvis_nlu.storage import Storage


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def build(store, spoken, lock=None):
    return Scheduler(storage=store, speak=spoken.append,
                     turn_lock=lock or threading.Lock(),
                     daily_brief_at=None)


def test_due_reminder_is_spoken_once(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    scheduler = build(store, spoken)
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    scheduler.tick(datetime(2026, 8, 24, 18, 2))
    assert len(spoken) == 1
    assert "call mom" in spoken[0]


def test_future_reminder_is_not_spoken(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 23, 0))
    build(store, spoken).tick(datetime(2026, 8, 24, 18, 0))
    assert spoken == []


def test_delivery_is_recorded_only_after_speaking(store):
    """A crash mid-announcement must replay the reminder, not swallow it."""
    reminder_id = store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))

    def exploding_speak(_message):
        raise RuntimeError("tts died")

    scheduler = Scheduler(storage=store, speak=exploding_speak,
                          turn_lock=threading.Lock(), daily_brief_at=None)
    with pytest.raises(RuntimeError):
        scheduler.tick(datetime(2026, 8, 24, 18, 1))

    assert store.list_reminders()[0].delivered_at is None
    assert len(store.due_reminders(datetime(2026, 8, 24, 18, 2))) == 1


def test_tick_skips_while_a_turn_is_in_progress(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    lock = threading.Lock()
    scheduler = build(store, spoken, lock)
    lock.acquire()                                  # user is mid-turn
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    assert spoken == []
    lock.release()
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    assert len(spoken) == 1


def test_daily_brief_fires_once_per_day(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    scheduler.tick(datetime(2026, 8, 24, 9, 0))
    assert len(spoken) == 1
    assert "dentist" in spoken[0]


def test_daily_brief_stays_quiet_with_nothing_to_report(store):
    spoken = []
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    assert spoken == []


def test_daily_brief_does_not_fire_before_its_time(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 7, 0))
    assert spoken == []


# ---------- pinned full-string assertions ----------
# Earlier tasks in this plan shipped malformed spoken text that still passed
# substring assertions. These pin the exact literal wording so that kind of
# regression cannot slip through silently again.

def test_due_reminder_announcement_is_pinned_verbatim(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    build(store, spoken).tick(datetime(2026, 8, 24, 18, 1))
    assert spoken == ["Heads up, sugar — you asked me to remind you to call mom."]


def test_daily_brief_with_event_and_pending_reminder_is_pinned_verbatim(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    store.add_reminder("buy milk", datetime(2026, 8, 24, 20, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    assert spoken == [
        "Mornin', sugar. Here's your day — on your calendar: dentist at 4:00 PM, "
        "and 1 reminder pendin'."
    ]


def test_daily_brief_time_formatting_at_noon_and_midnight(store):
    spoken = []
    store.add_event("noon standup", datetime(2026, 8, 24, 12, 0))
    store.add_event("midnight snack", datetime(2026, 8, 24, 0, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    assert len(spoken) == 1
    message = spoken[0]
    assert "midnight snack at 12:00 AM" in message
    assert "noon standup at 12:00 PM" in message
    # No leading zero on any hour, ever.
    assert "at 0" not in message


def test_daily_brief_multiple_pending_reminders_pluralizes(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    store.add_reminder("buy milk", datetime(2026, 8, 24, 20, 0))
    store.add_reminder("call mom", datetime(2026, 8, 24, 21, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    assert spoken[0].endswith("and 2 reminders pendin'.")


def test_start_and_stop_run_the_background_thread_deterministically(store):
    """The thread is a thin wrapper around tick(); start/stop must not hang
    the suite and must not die if a tick raises."""
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    ticked = threading.Event()

    def fake_clock():
        ticked.set()
        return datetime(2026, 8, 24, 18, 1)

    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at=None,
                          tick_seconds=0.01, clock=fake_clock)
    scheduler.start()
    try:
        assert ticked.wait(timeout=5)
    finally:
        scheduler.stop()

    assert scheduler._thread is None
    assert len(spoken) == 1


def test_start_thread_survives_a_raising_tick(store):
    """A bad tick must not kill the background thread."""
    def exploding_speak(_message):
        raise RuntimeError("boom")

    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    ticks = []

    def fake_clock():
        ticks.append(1)
        return datetime(2026, 8, 24, 18, 1)

    scheduler = Scheduler(storage=store, speak=exploding_speak,
                          turn_lock=threading.Lock(), daily_brief_at=None,
                          tick_seconds=0.01, clock=fake_clock)
    scheduler.start()
    try:
        deadline = threading.Event()
        deadline.wait(timeout=0.5)
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
        assert len(ticks) >= 1
    finally:
        scheduler.stop()
    assert scheduler._thread is None
