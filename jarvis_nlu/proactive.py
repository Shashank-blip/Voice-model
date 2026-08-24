"""The only unprompted speech in the system: due reminders and one daily brief.

`tick` is synchronous and takes `now`, so the whole policy is testable on a
fake clock without spawning threads."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable

from jarvis_nlu.storage import Storage

BRIEF_KEY = "last_brief_date"


class Scheduler:
    def __init__(self, storage: Storage, speak: Callable[[str], None],
                 turn_lock: threading.Lock, daily_brief_at: str | None = "08:30",
                 tick_seconds: int = 20, clock: Callable[[], datetime] = datetime.now):
        self.storage = storage
        self.speak = speak
        self.turn_lock = turn_lock
        self.daily_brief_at = daily_brief_at
        self.tick_seconds = tick_seconds
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- policy ----------

    def tick(self, now: datetime) -> None:
        # Never talk over the user: skip this tick rather than queue behind them.
        if not self.turn_lock.acquire(blocking=False):
            return
        try:
            self._speak_due_reminders(now)
            self._maybe_daily_brief(now)
        finally:
            self.turn_lock.release()

    def _speak_due_reminders(self, now: datetime) -> None:
        for reminder in self.storage.due_reminders(now):
            message = f"Heads up, sugar — you asked me to remind you to {reminder.text}."
            print(message)
            self.speak(message)
            # Recorded only after speak() returns: if speak() raises, this
            # line never runs and delivered_at stays NULL, so the reminder
            # replays on the next tick instead of being silently eaten.
            self.storage.mark_delivered(reminder.id, now)

    def _maybe_daily_brief(self, now: datetime) -> None:
        if not self.daily_brief_at:
            return
        hour, minute = (int(p) for p in self.daily_brief_at.split(":"))
        if (now.hour, now.minute) < (hour, minute):
            return
        today = now.date().isoformat()
        if self.storage.get_meta(BRIEF_KEY) == today:
            return

        events = self.storage.events_on(now.date())
        pending = [r for r in self.storage.list_reminders() if r.due_at]
        if not events and not pending:
            self.storage.set_meta(BRIEF_KEY, today)  # nothing to say, but don't retry all day
            return

        parts = []
        if events:
            listed = "; ".join(f"{e.title} at {e.starts_at.strftime('%I:%M %p').lstrip('0')}"
                               for e in events)
            parts.append(f"on your calendar: {listed}")
        if pending:
            parts.append(f"and {len(pending)} reminder{'s' if len(pending) != 1 else ''} pendin'")
        message = "Mornin', sugar. Here's your day — " + ", ".join(parts) + "."
        print(message)
        self.speak(message)
        self.storage.set_meta(BRIEF_KEY, today)

    # ---------- thread ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="jarvis-proactive")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_seconds):
            try:
                self.tick(self.clock())
            except Exception:  # a bad tick must not kill the thread
                logging.getLogger(__name__).exception("proactive tick failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.tick_seconds)
            self._thread = None
