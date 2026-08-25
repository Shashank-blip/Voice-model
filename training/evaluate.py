"""Enforce the spec's evaluation gates.

The confusion matrix is a required artifact, not a nicety: shadowing between
add_calendar_event and query_calendar is exactly the bug class that shipped
undetected in the regex router, and it appears here as an off-diagonal cell."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from jarvis_nlu.intents import CHORE_INTENTS
from training.generate import Example, build_dataset

ROOT = Path(__file__).parent.parent

GATES = {"macro_f1": 0.95, "max_confusion": 0.02,
         "deferral_rate": 0.05, "leak_rate": 0.02}


@dataclass
class Report:
    macro_f1: float
    confusion: dict[tuple[str, str], float] = field(default_factory=dict)
    deferral_rate: float = 0.0
    leak_rate: float = 0.0
    per_intent: dict[str, dict[str, float]] = field(default_factory=dict)

    def worst_confusion(self) -> tuple[str, str, float]:
        chores = {i.value for i in CHORE_INTENTS}
        pairs = [(t, p, r) for (t, p), r in self.confusion.items()
                 if t in chores and p in chores]
        if not pairs:
            return ("", "", 0.0)
        return max(pairs, key=lambda item: item[2])


def score(classifier, examples: list[Example], defer_threshold: float = 0.6) -> Report:
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)
    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    per_true_total: dict[str, int] = defaultdict(int)

    deferred = in_scope = leaked = out_of_scope = 0

    for example in examples:
        predicted, confidence = classifier.classify(example.text)
        actual = example.intent
        per_true_total[actual] += 1

        if predicted == actual:
            true_positive[actual] += 1
        else:
            false_positive[predicted] += 1
            false_negative[actual] += 1
            confusion_counts[(actual, predicted)] += 1

        if actual == "out_of_scope":
            out_of_scope += 1
            if confidence >= defer_threshold and predicted != "out_of_scope":
                leaked += 1
        else:
            in_scope += 1
            if confidence < defer_threshold or predicted == "out_of_scope":
                deferred += 1

    per_intent, f1_scores = {}, []
    for intent in per_true_total:
        tp, fp, fn = true_positive[intent], false_positive[intent], false_negative[intent]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent[intent] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    confusion = {pair: count / per_true_total[pair[0]]
                 for pair, count in confusion_counts.items()}

    return Report(
        macro_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        confusion=confusion,
        deferral_rate=deferred / in_scope if in_scope else 0.0,
        leak_rate=leaked / out_of_scope if out_of_scope else 0.0,
        per_intent=per_intent)


def deferral_tradeoff(classifier, examples: list[Example],
                      thresholds: list[float]) -> dict[float, dict[str, float]]:
    """For each candidate defer threshold, report the in-scope deferral rate
    alongside the rate of CONFIDENTLY-WRONG in-scope answers (predicted !=
    actual, in-scope, and confidence >= threshold) -- the failure a lower
    threshold trades more of in exchange for fewer deferrals. Classifies each
    example once and reuses the (predicted, confidence) pairs for every
    threshold, so this is O(examples) + O(thresholds), not O(examples *
    thresholds)."""
    predictions = [(example, *classifier.classify(example.text))
                   for example in examples]
    in_scope = [(example, predicted, confidence)
                for example, predicted, confidence in predictions
                if example.intent != "out_of_scope"]
    n_in_scope = len(in_scope)

    table: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        deferred = wrong_confident = 0
        for example, predicted, confidence in in_scope:
            answered = confidence >= threshold and predicted != "out_of_scope"
            if not answered:
                deferred += 1
            elif predicted != example.intent:
                wrong_confident += 1
        table[threshold] = {
            "deferral_rate": deferred / n_in_scope if n_in_scope else 0.0,
            "confidently_wrong_rate": wrong_confident / n_in_scope if n_in_scope else 0.0,
        }
    return table


def main() -> int:
    from jarvis_nlu.model import OnnxClassifier
    from training.train import grouped_split

    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    defer_threshold = manifest["thresholds"]["defer"]
    _train, _val, test_rows = grouped_split(
        build_dataset(ROOT / "training" / "templates"))
    classifier = OnnxClassifier(ROOT / "models")
    report = score(classifier, test_rows, defer_threshold)

    print("\nPer-intent F1")
    for intent, metrics in sorted(report.per_intent.items()):
        print(f"  {intent:22} P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    print("\nTop confusions (true -> predicted)")
    for (actual, predicted), rate in sorted(
            report.confusion.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {actual:22} -> {predicted:22} {rate:.1%}")

    worst_true, worst_pred, worst_rate = report.worst_confusion()
    results = {
        "macro_f1": (report.macro_f1, GATES["macro_f1"], report.macro_f1 >= GATES["macro_f1"]),
        "max_chore_confusion": (worst_rate, GATES["max_confusion"],
                                worst_rate <= GATES["max_confusion"]),
        "deferral_rate": (report.deferral_rate, GATES["deferral_rate"],
                          report.deferral_rate < GATES["deferral_rate"]),
        "leak_rate": (report.leak_rate, GATES["leak_rate"],
                      report.leak_rate < GATES["leak_rate"]),
    }

    print("\nGates")
    failed = False
    for name, (value, gate, passed) in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name:22} {value:.4f} (gate {gate})")
        failed = failed or not passed
    if worst_rate > GATES["max_confusion"]:
        print(f"\n  worst chore confusion: {worst_true} -> {worst_pred} ({worst_rate:.1%})")

    # The deferral_rate gate below assumes defer ~= 0.6; the trained manifest
    # uses defer = 0.83, chosen by cost-minimisation to trade more deferrals
    # for fewer confidently-wrong local answers (see training/train.py
    # MISCLASSIFY_COST). If the gate fails at 0.83, this table is what
    # decides whether that's the right trade -- do NOT use it to retune the
    # gate or the shipped threshold.
    print("\nDeferral / confidently-wrong trade-off by threshold")
    print(f"  {'threshold':>10} {'deferral_rate':>15} {'confidently_wrong_rate':>24}")
    tradeoff = deferral_tradeoff(classifier, test_rows, [0.5, 0.6, 0.7, defer_threshold])
    for threshold in sorted(tradeoff):
        row = tradeoff[threshold]
        print(f"  {threshold:>10.2f} {row['deferral_rate']:>15.1%} "
              f"{row['confidently_wrong_rate']:>24.1%}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
