import numpy as np
import pytest

from training.generate import Example
from training.train import CONFIRM_MARGIN, LABELS, _choose_thresholds, fit_temperature, grouped_split


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


def test_temperature_sharpens_underconfident_logits():
    """Companion to the overconfident case above: fit_temperature must also
    move T BELOW 1 when the split is under-confident, not just above 1 when
    it is overconfident -- the docstring claims the fit corrects
    miscalibration in either direction, so both directions need coverage."""
    rng = np.random.default_rng(2)
    # Small-magnitude logits -> low-confidence softmax, but every prediction
    # is nonetheless correct: NLL is minimised by sharpening (T < 1).
    logits = rng.normal(0, 1, size=(200, 3)) * 0.2
    labels = logits.argmax(axis=1)
    temperature = fit_temperature(logits, labels)
    assert temperature < 1.0                    # <1 means confidence is raised


def test_temperature_is_positive_for_well_calibrated_input():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=(200, 3))
    labels = logits.argmax(axis=1)
    assert fit_temperature(logits, labels) > 0


def test_thin_intent_raises_instead_of_silently_starving_train_split():
    """An intent with only 2 templates sends n_val=1, n_test=1, leaving zero
    for train. compute_class_weights would then silently assign that class
    a default weight (masking the missing training data) instead of
    failing loudly -- grouped_split must raise first."""
    examples = [
        Example("thin phrase one", "thin_intent", "thin_intent:0"),
        Example("thin phrase two", "thin_intent", "thin_intent:1"),
    ]
    for template in range(10):
        for variant in range(5):
            examples.append(Example(
                f"healthy t{template} v{variant}", "healthy_intent",
                f"healthy_intent:{template}"))
    with pytest.raises(ValueError, match="thin_intent"):
        grouped_split(examples, seed=0)


def _probabilities_from_rows(rows: list[tuple[int, float]], n_labels: int) -> np.ndarray:
    """Build a probabilities matrix whose argmax/max per row are exactly the
    given (predicted_index, confidence) pairs -- the only two things
    _choose_thresholds reads off `probabilities`."""
    probabilities = np.zeros((len(rows), n_labels))
    for row, (predicted_index, confidence) in enumerate(rows):
        remainder = (1.0 - confidence) / (n_labels - 1)
        probabilities[row, :] = remainder
        probabilities[row, predicted_index] = confidence
    return probabilities


def test_choose_thresholds_raises_tau_defer_when_confidently_wrong_predictions_appear():
    """Regression test for the degenerate-objective defect: the old
    two-gate implementation never measured a confidently WRONG in-scope
    answer, so injecting one couldn't move tau_defer up. The cost-based
    replacement must move it up, because a confidently wrong answer costs
    MISCLASSIFY_COST times a mere deferral."""
    n_labels = len(LABELS)
    add_reminder_index = LABELS.index("add_reminder")
    add_note_index = LABELS.index("add_note")
    out_of_scope_index = LABELS.index("out_of_scope")

    n_in_scope = 200
    in_scope_confidence = np.linspace(0.15, 0.90, n_in_scope)
    out_of_scope_confidence = np.linspace(0.15, 0.45, 60)
    labels = np.array([add_reminder_index] * n_in_scope + [out_of_scope_index] * 60)

    # Clean: every in-scope row is correctly predicted; out_of_scope never
    # leaks. The only cost is deferral, so the cheapest tau is the lowest.
    clean_rows = ([(add_reminder_index, c) for c in in_scope_confidence]
                  + [(out_of_scope_index, c) for c in out_of_scope_confidence])
    clean = _choose_thresholds(
        _probabilities_from_rows(clean_rows, n_labels), labels)

    # Injected: 40 of those in-scope rows are now confidently WRONG (high
    # confidence, wrong predicted intent) instead of correct.
    injected_predicted = [add_reminder_index] * n_in_scope
    injected_confidence = in_scope_confidence.copy()
    for index in range(150, 190):
        injected_predicted[index] = add_note_index
        injected_confidence[index] = 0.85
    injected_rows = ([(injected_predicted[i], injected_confidence[i])
                      for i in range(n_in_scope)]
                     + [(out_of_scope_index, c) for c in out_of_scope_confidence])
    injected = _choose_thresholds(
        _probabilities_from_rows(injected_rows, n_labels), labels)

    assert injected["defer"] > clean["defer"]


def test_confirm_band_is_never_empty_for_a_high_precision_model():
    """Regression test for task-12 review round 2, Important 1: when every
    in-scope prediction is correct, the 95%-precision criterion for
    tau_confirm is already satisfied at or below tau_defer, which used to
    collapse tau_confirm down to exactly tau_defer -- an empty
    [defer, confirm) band that let destructive intents execute without
    ever asking. tau_confirm must now be floored at
    tau_defer + CONFIRM_MARGIN."""
    n_labels = len(LABELS)
    add_reminder_index = LABELS.index("add_reminder")
    out_of_scope_index = LABELS.index("out_of_scope")

    n_in_scope, n_out_of_scope = 200, 60
    in_scope_confidence = np.linspace(0.15, 0.90, n_in_scope)
    out_of_scope_confidence = np.linspace(0.15, 0.45, n_out_of_scope)
    rows = ([(add_reminder_index, c) for c in in_scope_confidence]
            + [(out_of_scope_index, c) for c in out_of_scope_confidence])
    labels = np.array([add_reminder_index] * n_in_scope
                      + [out_of_scope_index] * n_out_of_scope)

    thresholds = _choose_thresholds(_probabilities_from_rows(rows, n_labels), labels)
    assert thresholds["confirm"] > thresholds["defer"]
    assert thresholds["confirm"] >= thresholds["defer"] + CONFIRM_MARGIN - 1e-9


def test_confirm_band_is_never_empty_for_a_low_precision_model():
    """Companion to the high-precision case above: when in-scope precision
    only clears 95% well above tau_defer + CONFIRM_MARGIN, tau_confirm must
    track that higher precision-driven threshold (not collapse to the
    floor) -- the confirmation band still must not be empty."""
    n_labels = len(LABELS)
    add_reminder_index = LABELS.index("add_reminder")
    add_note_index = LABELS.index("add_note")
    out_of_scope_index = LABELS.index("out_of_scope")

    n_right, n_wrong, n_out_of_scope = 300, 20, 40
    right_confidence = np.linspace(0.10, 0.95, n_right)
    # A small but persistent share of wrong predictions survives up to
    # moderately high confidence, so cost-minimisation settles on a low
    # tau_defer while the stricter 95%-precision bar for confirm is only
    # cleared much later.
    wrong_confidence = np.linspace(0.10, 0.60, n_wrong)
    out_of_scope_confidence = np.linspace(0.10, 0.30, n_out_of_scope)

    rows = ([(add_reminder_index, c) for c in right_confidence]
            + [(add_note_index, c) for c in wrong_confidence]
            + [(out_of_scope_index, c) for c in out_of_scope_confidence])
    labels = np.array([add_reminder_index] * (n_right + n_wrong)
                      + [out_of_scope_index] * n_out_of_scope)

    thresholds = _choose_thresholds(_probabilities_from_rows(rows, n_labels), labels)
    assert thresholds["confirm"] > thresholds["defer"]
    # This scenario is deliberately built so the precision bar binds well
    # above the floor -- assert that actually happened, or the test would
    # not be distinguishing this code path from the high-precision case.
    assert thresholds["confirm"] - thresholds["defer"] > CONFIRM_MARGIN + 1e-9
