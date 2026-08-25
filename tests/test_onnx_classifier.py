import time
from pathlib import Path

import pytest

from jarvis_nlu.model import Classifier, OnnxClassifier

MODELS = Path(__file__).parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(),
    reason="model not built; run training/train.py and training/export_onnx.py")


@pytest.fixture(scope="module")
def classifier():
    return OnnxClassifier(MODELS)


def test_satisfies_the_classifier_protocol(classifier):
    assert isinstance(classifier, Classifier)


def test_returns_a_known_label_and_a_probability(classifier):
    from jarvis_nlu.intents import ALL_INTENTS
    intent, confidence = classifier.classify("remind me to call mom at six")
    assert intent in {i.value for i in ALL_INTENTS}
    assert 0.0 <= confidence <= 1.0


def test_p95_latency_under_25ms(classifier):
    classifier.classify("warm up")
    samples = []
    for _ in range(50):
        start = time.perf_counter()
        classifier.classify("what is on my calendar today")
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    assert samples[int(len(samples) * 0.95)] < 25.0


def test_missing_model_directory_raises_with_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx.py"):
        OnnxClassifier(tmp_path)
