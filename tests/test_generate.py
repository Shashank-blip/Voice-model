from pathlib import Path

import pytest

from jarvis_nlu.intents import ALL_INTENTS
from training.generate import Example, add_stt_noise, build_dataset

TEMPLATES = Path(__file__).parent.parent / "training" / "templates"


def test_every_intent_has_a_template_file():
    for intent in ALL_INTENTS:
        assert (TEMPLATES / f"{intent.value}.yaml").exists(), f"missing {intent.value}.yaml"


def test_dataset_covers_every_intent():
    dataset = build_dataset(TEMPLATES, seed=0)
    covered = {e.intent for e in dataset}
    assert covered == {i.value for i in ALL_INTENTS}


def test_dataset_is_large_enough_to_train_on():
    assert len(build_dataset(TEMPLATES, seed=0)) >= 4000


def test_every_intent_has_at_least_a_hundred_examples():
    dataset = build_dataset(TEMPLATES, seed=0)
    counts = {}
    for example in dataset:
        counts[example.intent] = counts.get(example.intent, 0) + 1
    thin = {k: v for k, v in counts.items() if v < 100}
    assert not thin, f"under-represented intents: {thin}"


def test_out_of_scope_is_well_represented():
    """The negative class is what protects the LLM fallback path."""
    dataset = build_dataset(TEMPLATES, seed=0)
    out_of_scope = [e for e in dataset if e.intent == "out_of_scope"]
    assert len(out_of_scope) >= 400


def test_generation_is_deterministic_for_a_seed():
    first = [e.text for e in build_dataset(TEMPLATES, seed=7)]
    second = [e.text for e in build_dataset(TEMPLATES, seed=7)]
    assert first == second


def test_every_example_carries_a_template_id():
    """Template ids drive grouped splitting so paraphrases can't straddle
    train and test and inflate the score."""
    for example in build_dataset(TEMPLATES, seed=0):
        assert example.template_id


def test_stt_noise_changes_text_but_keeps_it_recognisable():
    import random
    rng = random.Random(0)
    noisy = [add_stt_noise("remind me to call mom at six", rng) for _ in range(20)]
    assert any(n != "remind me to call mom at six" for n in noisy)
    assert all(len(n) > 5 for n in noisy)


def test_learning_log_examples_are_merged_with_higher_weight(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    log.write_text('{"transcript": "ping me about the thing at four", '
                   '"intent": "add_reminder"}\n')
    dataset = build_dataset(TEMPLATES, learning_log=log, seed=0)
    merged = [e for e in dataset if e.text == "ping me about the thing at four"]
    assert len(merged) == 1
    assert merged[0].weight > 1.0


def test_learning_log_examples_are_weighted_at_the_documented_constant(tmp_path):
    """LEARNING_LOG_WEIGHT is the whole point of Task B's up-weighting --
    pin the exact value, not just 'higher than 1.0'."""
    from training.generate import LEARNING_LOG_WEIGHT
    log = tmp_path / "learning_log.jsonl"
    log.write_text('{"transcript": "ping me about the thing at four", '
                   '"intent": "add_reminder"}\n')
    dataset = build_dataset(TEMPLATES, learning_log=log, seed=0)
    merged = [e for e in dataset if e.text == "ping me about the thing at four"]
    assert merged[0].weight == LEARNING_LOG_WEIGHT == 3.0


def test_unlabelled_and_correction_rows_are_safely_ignored(tmp_path):
    """The core Task B safety property: nothing without a human-confirmed
    `intent` may ever become a training example. This must hold both for a
    plain unreviewed prediction row and for a correction/back-annotation
    row (e.g. a repeat_within_seconds signal), which carries no `intent`
    or `transcript` at all."""
    log = tmp_path / "learning_log.jsonl"
    log.write_text(
        '{"id": "a1", "transcript": "wake me at seven", '
        '"predicted_intent": "add_reminder", "confidence": 0.4}\n'
        '{"id": "a2", "correction_for": "a1", "repeat_within_seconds": 1.2}\n'
        '{"id": "a3", "transcript": "cancel the mom reminder", '
        '"predicted_intent": "cancel_reminder", "confidence": 0.91, '
        '"intent": "cancel_reminder"}\n'
    )
    dataset = build_dataset(TEMPLATES, learning_log=log, seed=0)
    texts = {e.text for e in dataset}
    assert "wake me at seven" not in texts
    assert "cancel the mom reminder" in texts


def test_unlabelled_row_with_empty_string_intent_is_also_ignored(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    log.write_text('{"transcript": "some noise", "predicted_intent": "out_of_scope", '
                   '"intent": ""}\n')
    dataset = build_dataset(TEMPLATES, learning_log=log, seed=0)
    assert "some noise" not in {e.text for e in dataset}
