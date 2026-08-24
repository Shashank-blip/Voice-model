"""Local-first NLU for the Jarvis / Miss Minutes voice assistant.

This package makes no network calls. Exports are added as tasks land.
"""
from jarvis_nlu.config import Config
from jarvis_nlu.intents import Intent

__all__ = ["Config", "Intent"]
