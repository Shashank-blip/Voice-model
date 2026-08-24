import json

import pytest

from jarvis_nlu.model import FakeClassifier, Thresholds


def test_fake_returns_scripted_result():
    fake = FakeClassifier({"hello there": ("greeting", 0.99)})
    assert fake.classify("hello there") == ("greeting", 0.99)


def test_fake_falls_back_to_out_of_scope():
    fake = FakeClassifier({})
    intent, confidence = fake.classify("anything at all")
    assert intent == "out_of_scope"
    assert confidence < 0.5


def test_fake_matching_is_case_and_space_insensitive():
    fake = FakeClassifier({"hello there": ("greeting", 0.99)})
    assert fake.classify("  Hello There  ")[0] == "greeting"


def test_thresholds_load_from_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"thresholds": {"defer": 0.62, "confirm": 0.85}}))
    thresholds = Thresholds.from_manifest(manifest)
    assert thresholds.defer == 0.62
    assert thresholds.confirm == 0.85


def test_missing_manifest_raises_with_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="training/train.py"):
        Thresholds.from_manifest(tmp_path / "absent.json")
