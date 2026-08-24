"""Runtime configuration. Every field has a default, so a missing file is
valid. Model thresholds deliberately live in the model manifest, not here --
a retrain must not leave a stale hand-tuned value behind."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_NAME = "jarvis.config.json"


@dataclass(frozen=True)
class Config:
    model_dir: Path = Path("models")
    db_path: Path = Path("data/jarvis.db")
    tick_seconds: int = 20
    daily_brief_at: str | None = "08:30"
    persona_path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_NAME
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        defaults = cls()

        def _or_default(key: str, default):
            # A missing key and an explicit JSON null both fall back to the
            # dataclass default -- required scalar/Path fields have no valid
            # "null" of their own, unlike daily_brief_at/persona_path below.
            value = raw.get(key)
            return default if value is None else value

        return cls(
            model_dir=Path(_or_default("model_dir", defaults.model_dir)),
            db_path=Path(_or_default("db_path", defaults.db_path)),
            tick_seconds=int(_or_default("tick_seconds", defaults.tick_seconds)),
            daily_brief_at=raw.get("daily_brief_at", defaults.daily_brief_at),
            persona_path=Path(raw["persona_path"]) if raw.get("persona_path") else None,
        )
