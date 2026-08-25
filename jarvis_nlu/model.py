"""Classifier boundary.

The router depends on the `Classifier` protocol, never on onnxruntime, so the
whole assistant is testable with `FakeClassifier` and fast unit tests before
any model is trained."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from jarvis_nlu.intents import Intent


@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str) -> tuple[str, float]:
        """Return (intent_value, calibrated_confidence)."""
        ...


@dataclass(frozen=True)
class Thresholds:
    defer: float
    confirm: float

    @classmethod
    def from_manifest(cls, path: Path) -> "Thresholds":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model manifest at {path}. Build the model first:\n"
                f"  python -m training.generate && python -m training.train "
                f"&& python -m training.export_onnx")
        raw = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
        return cls(defer=float(raw["defer"]), confirm=float(raw["confirm"]))


class FakeClassifier:
    """Test double. Scripted exact matches, out_of_scope for everything else."""

    def __init__(self, scripted: dict[str, tuple[str, float]],
                 default: tuple[str, float] = (Intent.OUT_OF_SCOPE.value, 0.1)):
        self._scripted = {k.strip().lower(): v for k, v in scripted.items()}
        self._default = default

    def classify(self, text: str) -> tuple[str, float]:
        return self._scripted.get(text.strip().lower(), self._default)
