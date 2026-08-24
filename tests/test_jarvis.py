import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from jarvis.app import JarvisAssistant
from jarvis.services import PersonalServices


class JarvisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.services = PersonalServices(Path(self.directory.name), [])
        self.jarvis = JarvisAssistant(self.services)
        self.now = datetime(2026, 8, 24, 9, 0)

    def tearDown(self):
        self.directory.cleanup()

    def test_stays_dormant_without_wake_word(self):
        self.assertIsNone(self.jarvis.handle_transcript("what is on my calendar", self.now))

    def test_wake_then_reminder(self):
        answer = self.jarvis.handle_transcript("hey jarvis remind me to call Alex tomorrow at 6 pm", self.now)
        self.assertIn("call Alex", answer)
        self.assertIn("remind", answer)

    def test_note_is_local(self):
        self.jarvis.handle_transcript("hey jarvis", self.now)
        self.assertEqual("Saved your note.", self.jarvis.handle_transcript("take a note buy milk", self.now))
        self.assertEqual("buy milk", self.services.notes.read()[0]["text"])

    def test_add_calendar_event_then_list_today(self):
        answer = self.jarvis.handle_transcript("hey jarvis add dentist appointment today at 4 pm to my calendar", self.now)
        self.assertIn("Added dentist appointment", answer)
        self.assertIn("dentist appointment", self.jarvis.handle_transcript("what is on my calendar today", self.now))
