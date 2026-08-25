from pathlib import Path

import pytest
import yaml

from jarvis_nlu.model import OnnxClassifier

MODELS = Path(__file__).parent.parent / "models"
GOLDEN = yaml.safe_load((Path(__file__).parent / "golden.yaml").read_text())

pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(), reason="model not built")


@pytest.fixture(scope="module")
def classifier():
    return OnnxClassifier(MODELS)


@pytest.mark.parametrize("case", GOLDEN, ids=[c["text"][:40] for c in GOLDEN])
def test_golden_case(classifier, case):
    predicted, confidence = classifier.classify(case["text"])
    assert predicted == case["intent"], (
        f"{case['text']!r} -> {predicted} ({confidence:.2f}), "
        f"expected {case['intent']}")
