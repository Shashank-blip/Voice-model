"""Regression guard against eval/heldout.yaml, the hand-written held-out set.

This is NOT a gate (see training/evaluate.py GATES, which the user owns).
It is a floor: a number picked from what was actually measured against the
current shipped model, so a future change that silently tanks real-world
accuracy fails CI even though no formal gate exists yet at 300-ish samples
per run. If retraining moves the number, update FLOOR deliberately and say
why -- don't raise it reflexively to whatever the new run happens to score.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_nlu.model import OnnxClassifier
from training.evaluate import load_heldout

MODELS = Path(__file__).parent.parent / "models"

# Measured overall accuracy on eval/heldout.yaml (337 examples) against the
# model shipped as of this commit was 77.4% (261/337; see
# .superpowers/sdd/2026-08-24-jarvis-local-nlu/task-14-report.md for the
# full run). FLOOR is set ~12 points below that -- enough headroom that
# normal retraining noise (a reshuffled split, a slightly different
# template mix) won't trip it, while still catching a real regression
# (e.g. a training bug that silently drops a class, or a threshold change
# that starts answering with the wrong intent instead of deferring).
FLOOR = 0.65

pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(), reason="model not built")


@pytest.fixture(scope="module")
def classifier():
    return OnnxClassifier(MODELS)


def test_heldout_accuracy_stays_above_floor(classifier):
    examples = load_heldout()
    correct = sum(
        1 for example in examples
        if classifier.classify(example.text)[0] == example.intent)
    accuracy = correct / len(examples)
    assert accuracy >= FLOOR, (
        f"held-out accuracy {accuracy:.3f} ({correct}/{len(examples)}) "
        f"dropped below the regression floor {FLOOR}")
