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


class OnnxClassifier:
    """Runtime classifier. onnxruntime only -- torch is a training-time dep."""

    def __init__(self, model_dir: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_path = model_dir / "intent.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No model at {onnx_path}. Build it:\n"
                f"  python training/generate.py && python training/train.py "
                f"&& python training/export_onnx.py")

        self._np = np
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self._labels: list[str] = manifest["labels"]
        self._temperature: float = float(manifest.get("temperature", 1.0))
        self._max_length: int = int(manifest["max_length"])
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer" / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=self._max_length)
        # Single request, batch=1, <=48 tokens: intra/inter-op thread-pool
        # negotiation costs more than it saves on a graph this small, and
        # its cost is *unpredictable* -- it depends on what else the process
        # is doing (e.g. another onnxruntime session's thread pool, a busy
        # test suite). Pinning to one thread and sequential execution trades
        # a little best-case latency for a p95 that holds steady regardless
        # of what else is running in-process.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=session_options,
            providers=["CPUExecutionProvider"])

    def classify(self, text: str) -> tuple[str, float]:
        np = self._np
        encoded = self._tokenizer.encode(text)
        logits = self._session.run(None, {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        })[0][0]
        scaled = logits / self._temperature
        exponentials = np.exp(scaled - scaled.max())
        probabilities = exponentials / exponentials.sum()
        index = int(probabilities.argmax())
        return self._labels[index], float(probabilities[index])
