from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from .services import PersonalServices

WAKE_PATTERN = re.compile(r"\b(?:hey\s+)?jarvis\b[,.! ]*", re.I)


def parse_when(words: str, now: datetime) -> tuple[datetime | None, str]:
    match = re.search(r"\b(today|tomorrow|\d{4}-\d{2}-\d{2})\b(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", words, re.I)
    if not match:
        return None, words
    day, hour, minute, meridiem = match.groups()
    date = now.date() + timedelta(days=1 if day.lower() == "tomorrow" else 0)
    if day[:2].isdigit():
        date = datetime.strptime(day, "%Y-%m-%d").date()
    hour, minute = int(hour), int(minute or 0)
    if meridiem:
        hour = (hour % 12) + (12 if meridiem.lower() == "pm" else 0)
    when = datetime.combine(date, datetime.min.time()).replace(hour=hour, minute=minute)
    return when, (words[:match.start()] + words[match.end():]).strip(" ,")


class JarvisAssistant:
    def __init__(self, services: PersonalServices, name: str = "Jarvis"):
        self.services, self.name, self.awake = services, name, False
        self.last_greeting: datetime | None = None

    def handle_transcript(self, transcript: str, now: datetime | None = None) -> str | None:
        now = now or datetime.now()
        text = transcript.strip()
        if not text:
            return None
        wake = WAKE_PATTERN.search(text)
        if wake:
            self.awake = True
            text = text[wake.end():].strip()
            if not text:
                self.last_greeting = now
                return f"Hello. How are you doing?"
        if not self.awake:
            return None
        answer = self._route(text, now)
        return answer

    def _route(self, text: str, now: datetime) -> str:
        lowered = text.lower().strip()
        if re.search(r"\b(hi|hello|how are you|i am|i'm)\b", lowered):
            return "I am here and ready to help. What would you like to do?"
        reminder = re.match(r"remind me to (.+)", text, re.I)
        if reminder:
            when, task = parse_when(reminder.group(1), now)
            return self.services.add_reminder(task, when) if when else "Please include a time, such as tomorrow at 6 PM."
        event = re.match(r"(?:add )?(.+?) (?:to )?my calendar", text, re.I)
        if event:
            when, title = parse_when(event.group(1), now)
            return self.services.add_event(title, when) if when and title else "Please include the event and time, such as add dentist appointment tomorrow at 4 PM to my calendar."
        if "calendar" in lowered and any(word in lowered for word in ("what", "show", "today", "schedule")):
            return self.services.todays_events(now)
        note = re.match(r"(?:take |save |add )?(?:a )?note[: ]+(.+)", text, re.I)
        if note:
            return self.services.add_note(note.group(1))
        if "reminder" in lowered and any(word in lowered for word in ("read", "show", "what", "list")):
            return self.services.list_reminders()
        found = re.match(r"(?:find|search for) (?:my |the )?(.+)", text, re.I)
        if found:
            return self.services.find_files(found.group(1))
        self._log_unknown(text, now)
        return "I do not know that chore yet. I saved the command pattern for review. You can ask me to set a reminder, add a calendar event, save a note, or find a file."

    def _log_unknown(self, text: str, now: datetime) -> None:
        log = self.services.reminders.path.parent / "learning_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": now.isoformat(), "transcript": text, "outcome": "unknown_command"}) + "\n")


def load_config(root: Path) -> dict:
    path = root / "jarvis.config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"assistant_name": "Jarvis", "search_roots": [], "reminder_poll_seconds": 15}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first Jarvis")
    parser.add_argument("--text", action="store_true", help="Run the keyboard transcript interface")
    parser.add_argument("--voice", action="store_true", help="Run the safe console voice-adapter fallback")
    args = parser.parse_args()
    root, config = Path.cwd(), load_config(Path.cwd())
    services = PersonalServices(root / "data", [Path(item) for item in config.get("search_roots", [])])
    assistant = JarvisAssistant(services, config.get("assistant_name", "Jarvis"))
    if args.voice:
        from .voice import LocalVoiceLoop
        try:
            LocalVoiceLoop(assistant).run()
        except RuntimeError as error:
            print(f"Jarvis: {error}")
        return
    print("Jarvis is ready. Type 'hey jarvis' to wake it; Ctrl+C exits.")
    try:
        while True:
            for reminder in services.due_reminders(datetime.now()):
                print(f"Jarvis: Reminder — {reminder['text']}")
            transcript = input("You: ")
            response = assistant.handle_transcript(transcript)
            if response:
                print(f"Jarvis: {response}")
            time.sleep(0.05)
    except (KeyboardInterrupt, EOFError):
        print("\nJarvis: Going dormant.")
