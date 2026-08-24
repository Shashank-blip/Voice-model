from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .storage import JsonStore


class PersonalServices:
    def __init__(self, data_dir: Path, search_roots: list[Path]):
        self.reminders = JsonStore(data_dir / "reminders.json", [])
        self.calendar = JsonStore(data_dir / "calendar.json", [])
        self.notes = JsonStore(data_dir / "notes.json", [])
        self.search_roots = [root.resolve() for root in search_roots if root.exists()]

    def add_reminder(self, text: str, when: datetime) -> str:
        records = self.reminders.read()
        records.append({"id": str(uuid4()), "text": text, "when": when.isoformat(), "completed": False})
        self.reminders.write(records)
        return f"I will remind you to {text} at {when.strftime('%I:%M %p on %A, %d %B')}"

    def due_reminders(self, now: datetime) -> list[dict]:
        records, due = self.reminders.read(), []
        for record in records:
            if not record["completed"] and datetime.fromisoformat(record["when"]) <= now:
                record["completed"] = True
                due.append(record)
        self.reminders.write(records)
        return due

    def list_reminders(self) -> str:
        active = [r for r in self.reminders.read() if not r["completed"]]
        if not active:
            return "You have no pending reminders."
        return "Your reminders are: " + "; ".join(f"{r['text']} at {datetime.fromisoformat(r['when']).strftime('%I:%M %p, %d %B')}" for r in active[:5])

    def add_event(self, title: str, when: datetime) -> str:
        events = self.calendar.read()
        events.append({"id": str(uuid4()), "title": title, "when": when.isoformat()})
        self.calendar.write(events)
        return f"Added {title} to your local calendar for {when.strftime('%I:%M %p on %A, %d %B')}"

    def todays_events(self, today: datetime) -> str:
        events = [e for e in self.calendar.read() if datetime.fromisoformat(e["when"]).date() == today.date()]
        if not events:
            return "Your local calendar is clear today."
        events.sort(key=lambda e: e["when"])
        return "Today you have: " + "; ".join(f"{e['title']} at {datetime.fromisoformat(e['when']).strftime('%I:%M %p')}" for e in events)

    def add_note(self, text: str) -> str:
        notes = self.notes.read()
        notes.append({"id": str(uuid4()), "text": text, "created_at": datetime.now().isoformat()})
        self.notes.write(notes)
        return "Saved your note."

    def find_files(self, query: str) -> str:
        cleaned = re.sub(r"[^\w .-]", "", query).strip().lower()
        if not cleaned:
            return "Tell me the name of the file to find."
        results: list[Path] = []
        for root in self.search_roots:
            try:
                results.extend(path for path in root.rglob("*") if path.is_file() and cleaned in path.name.lower())
            except PermissionError:
                continue
            if len(results) >= 5:
                break
        if not results:
            return "I could not find a matching file in your approved folders."
        return "I found: " + "; ".join(str(path) for path in results[:5])
