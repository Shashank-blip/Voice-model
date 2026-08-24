from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonStore:
    """Small, dependency-free persistent store for a single local user."""

    def __init__(self, path: Path, default: Any):
        self.path, self.default, self.lock = path, default, Lock()

    def read(self) -> Any:
        with self.lock:
            if not self.path.exists():
                return self.default
            return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, value: Any) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
            temporary.replace(self.path)
