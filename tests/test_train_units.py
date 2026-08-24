import numpy as np
import pytest

from training.generate import Example
from training.train import fit_temperature, grouped_split


def _dataset():
    examples = []
    for intent in ("add_reminder", "greeting", "out_of_scope"):
        for template in range(10):
            for variant in range(10):
                examples.append(
                    Example(f"{intent} t{template} v{variant}", intent,
                            f"{intent}:{template}"))
    return examples


def test_split_proportions_are_roughly_eighty_ten_ten():
    train, val, test = grouped_split(_dataset(), seed=0)
    total = len(train) + len(val) + len(test)
    assert 0.7 <= len(train) / total <= 0.9
    assert len(val) > 0 and len(test) > 0


def test_no_template_id_straddles_splits():
    """Paraphrases of one template must not appear in both train and test."""
    train, val, test = grouped_split(_dataset(), seed=0)
    ids = [{e.template_id for e in split} for split in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_every_intent_appears_in_every_split():
    train, val, test = grouped_split(_dataset(), seed=0)
    for split in (train, val, test):
        assert {e.intent for e in split} == {"add_reminder", "greeting", "out_of_scope"}


def test_split_is_deterministic():
    assert ([e.text for e in grouped_split(_dataset(), seed=3)[0]]
            == [e.text for e in grouped_split(_dataset(), seed=3)[0]])


def test_temperature_softens_overconfident_logits():
    # Wildly overconfident logits where 30% of predictions are wrong.
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, size=(200, 3)) * 10
    labels = logits.argmax(axis=1)
    labels[:60] = (labels[:60] + 1) % 3        # inject 30% error
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0                    # >1 means confidence is reduced


def test_temperature_is_positive_for_well_calibrated_input():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=(200, 3))
    labels = logits.argmax(axis=1)
    assert fit_temperature(logits, labels) > 0
