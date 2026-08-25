"""Task A: post-hoc threshold refit on eval/heldout.yaml.

The shipped manifest's thresholds were fit inside training/train.py on the
template-derived test split, which overstates accuracy: every example there
came out of the same generator the model trained on, just not that exact
string. eval/heldout.yaml was authored independently of the generator (see
its own header + training/evaluate.py's overlap check), so refitting the
SAME cost-minimisation logic (`_choose_thresholds`, imported -- never
duplicated) on held-out scores instead is the honest number.

These tests need the real trained model + tokenizer under models/, exactly
like tests/test_heldout.py, and skip if it hasn't been built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(), reason="model not built")


def _classify_to_rows(classifier, examples, labels_list):
    """Turn (text, intent) examples into the (predicted_index, confidence)
    rows _choose_thresholds needs, by running them through the real
    classifier -- the same construction training/fit_thresholds.py uses."""
    rows, label_indices = [], []
    for example in examples:
        predicted, confidence = classifier.classify(example.text)
        rows.append((labels_list.index(predicted), confidence))
        label_indices.append(labels_list.index(example.intent))
    return rows, label_indices


def _probabilities_from_rows(rows, n_labels):
    """Synthesize a probabilities matrix whose per-row argmax/max match the
    given (predicted_index, confidence) pairs exactly -- _choose_thresholds
    reads only those two derived values off `probabilities`, so this
    reproduces what the classifier reported without needing its internal
    softmax vector. Same technique as tests/test_train_units.py."""
    probabilities = np.zeros((len(rows), n_labels))
    for row, (predicted_index, confidence) in enumerate(rows):
        remainder = (1.0 - confidence) / (n_labels - 1)
        probabilities[row, :] = remainder
        probabilities[row, predicted_index] = confidence
    return probabilities


@pytest.fixture(scope="module")
def classifier():
    from jarvis_nlu.model import OnnxClassifier
    return OnnxClassifier(MODELS)


@pytest.fixture(scope="module")
def template_fitted_thresholds(classifier):
    """The SAME cost-minimisation logic, fit on the template-derived test
    split -- the split training/train.py fits thresholds on today.
    Recomputed here (not read off manifest.json) so this comparison holds
    even after training/fit_thresholds.py has already overwritten the
    manifest with the held-out fit."""
    from training.generate import build_dataset
    from training.train import LABELS, _choose_thresholds, grouped_split

    examples = build_dataset(ROOT / "training" / "templates")
    _train, _val, test = grouped_split(examples, seed=0)
    rows, label_indices = _classify_to_rows(classifier, test, LABELS)
    probabilities = _probabilities_from_rows(rows, len(LABELS))
    return _choose_thresholds(probabilities, np.array(label_indices))


def test_heldout_fitted_thresholds_differ_from_template_fitted(
        classifier, template_fitted_thresholds):
    from training.fit_thresholds import score_heldout
    from training.train import _choose_thresholds

    probabilities, labels = score_heldout(classifier)
    heldout_fitted = _choose_thresholds(probabilities, labels)

    assert heldout_fitted != template_fitted_thresholds, (
        "held-out-fitted thresholds must differ from the template-split "
        "fit -- if they match, the held-out set isn't measuring anything "
        "the (optimistic) template split didn't already say")


def test_heldout_fitted_confirm_band_is_never_empty(classifier):
    """_choose_thresholds floors confirm at defer + CONFIRM_MARGIN, but that
    margin is capped at 0.99 -- so it is not guaranteed verbatim once defer
    itself lands above ~0.89. The invariant that must always hold is the
    strict inequality: an empty [defer, confirm) band lets a destructive
    intent execute without ever asking."""
    from training.fit_thresholds import score_heldout
    from training.train import _choose_thresholds

    probabilities, labels = score_heldout(classifier)
    thresholds = _choose_thresholds(probabilities, labels)
    assert thresholds["confirm"] > thresholds["defer"]


def test_heldout_fitted_defer_keeps_confidently_wrong_in_scope_under_six_percent(
        classifier):
    """The whole point of refitting on held-out data: at the shipped
    template-fit threshold (0.67), held-out confidently-wrong in-scope
    answers run ~12-13%. The held-out cost-argmin must bring that under 6%."""
    from training.fit_thresholds import rates_at, score_heldout
    from training.train import _choose_thresholds

    probabilities, labels = score_heldout(classifier)
    thresholds = _choose_thresholds(probabilities, labels)
    rates = rates_at(probabilities, labels, thresholds["defer"])
    assert rates["confidently_wrong_rate"] < 0.06


def test_fit_thresholds_main_rewrites_only_the_thresholds_block(tmp_path, monkeypatch):
    """main() must touch nothing but manifest['thresholds'] -- weights,
    temperature, labels, and seed are untouched. Runs against a throwaway
    copy of models/ so the real manifest isn't mutated as a side effect of
    running the test suite.

    The fake manifest's thresholds are deliberately overwritten with an
    obviously-wrong sentinel first: the real manifest may already be sitting
    at its converged held-out fit (this script is idempotent once run for
    real), and comparing against that would make this assertion pass
    trivially on a second run instead of proving main() actually rewrote
    anything."""
    import shutil

    import training.fit_thresholds as fit_thresholds_module

    fake_models = tmp_path / "models"
    shutil.copytree(MODELS, fake_models)
    manifest_path = fake_models / "manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    before["thresholds"] = {"defer": 0.01, "confirm": 0.02}
    manifest_path.write_text(json.dumps(before), encoding="utf-8")

    monkeypatch.setattr(fit_thresholds_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(fit_thresholds_module, "MODEL_DIR", fake_models)
    fit_thresholds_module.main()

    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after["thresholds"] != before["thresholds"]
    for key in ("labels", "base_model", "max_length", "temperature", "seed"):
        assert after[key] == before[key]


def test_constrained_defer_keeps_held_out_reprompt_rate_within_budget(classifier):
    """The re-prompt rate under the two-strike policy (Task F) has no term
    in the cost function -- it prices a deferral purely as a wrong-answer
    avoided, never as user patience spent. The constrained selection must
    keep the held-out in-scope re-prompt rate at or under MAX_REPROMPT_RATE."""
    from training.fit_thresholds import (
        MAX_REPROMPT_RATE, reprompt_rate_in_scope, score_heldout,
        choose_thresholds_constrained,
    )

    probabilities, labels = score_heldout(classifier)
    thresholds = choose_thresholds_constrained(probabilities, labels)
    rate = reprompt_rate_in_scope(probabilities, labels, thresholds["defer"])
    assert rate <= MAX_REPROMPT_RATE + 1e-9


def test_constrained_defer_still_keeps_confirm_band_non_empty(classifier):
    from training.fit_thresholds import choose_thresholds_constrained, score_heldout

    probabilities, labels = score_heldout(classifier)
    thresholds = choose_thresholds_constrained(probabilities, labels)
    assert thresholds["confirm"] > thresholds["defer"]


def test_unsatisfiable_constraint_falls_back_to_unconstrained_argmin_with_a_warning(
        classifier, capsys):
    """A negative budget can never be satisfied by any rate (rates are
    always >= 0), so this deterministically forces the empty-feasible-set
    path regardless of what the held-out data happens to look like near
    zero confidence -- unlike an arbitrarily small positive budget, which
    could still be met if very few examples score that low. Must fall back
    to the plain cost argmin, and must say so loudly, never silently
    ignore the constraint."""
    from training.fit_thresholds import choose_thresholds_constrained, score_heldout
    from training.train import _choose_thresholds

    probabilities, labels = score_heldout(classifier)
    unconstrained = _choose_thresholds(probabilities, labels)

    fallback = choose_thresholds_constrained(
        probabilities, labels, max_reprompt_rate=-0.01)
    captured = capsys.readouterr()

    assert fallback == unconstrained
    assert "warning" in captured.out.lower()
    assert "re-prompt" in captured.out.lower() or "reprompt" in captured.out.lower()
