"""Enforce the spec's evaluation gates.

The confusion matrix is a required artifact, not a nicety: shadowing between
add_calendar_event and query_calendar is exactly the bug class that shipped
undetected in the regex router, and it appears here as an off-diagonal cell.

Two evaluation sets are supported:

- The template-derived test split (`grouped_split` over `build_dataset`),
  the default -- fast, no authoring cost, but every in-scope intent gets
  exactly 2 template families in the test split, so per-intent recall
  rests on very few phrasings.
- `eval/heldout.yaml` (`--heldout`), a hand-written set independent of the
  generator -- see the file's own header comment for how independence is
  maintained and verified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from jarvis_nlu.intents import CHORE_INTENTS
from training.generate import Example, build_dataset

ROOT = Path(__file__).parent.parent
HELDOUT_PATH = ROOT / "eval" / "heldout.yaml"

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


def load_heldout(path: Path = HELDOUT_PATH) -> list[Example]:
    """Load eval/heldout.yaml into Examples. template_id is synthesised as
    'heldout:<index>' -- unique per row, matching the contract build_dataset
    documents (grouped_split is not used for this set; every row is scored
    directly, there is no train/val/test split to assign templates to)."""
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [Example(row["text"], row["intent"], f"heldout:{index}")
            for index, row in enumerate(rows)]


def check_heldout_overlap(heldout: list[Example], templates_dir: Path,
                          learning_log: Path | None = None) -> list[str]:
    """Return the heldout texts (case-insensitive) that also appear in the
    generated training corpus. Must be empty -- a leaked eval set measures
    nothing, since the model has seen those exact strings during training."""
    generated = build_dataset(templates_dir, learning_log=learning_log)
    generated_texts = {example.text.strip().lower() for example in generated}
    return [example.text for example in heldout
            if example.text.strip().lower() in generated_texts]


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


def _print_report(report: Report, classifier, examples: list[Example],
                  defer_threshold: float, tradeoff_thresholds: list[float]) -> bool:
    """Print the per-intent table, confusions, gates and trade-off table
    shared by both evaluation modes. Returns True iff every gate passed.
    GATES is never modified here -- gate values are the user's decision."""
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

    # This table is diagnostic only. It never changes a gate value or the
    # shipped threshold -- it exists so a human can see the trade being made
    # at each candidate defer threshold before deciding whether GATES or
    # models/manifest.json's thresholds.defer should change.
    print("\nDeferral / confidently-wrong trade-off by threshold")
    print(f"  {'threshold':>10} {'deferral_rate':>15} {'confidently_wrong_rate':>24}")
    tradeoff = deferral_tradeoff(classifier, examples, tradeoff_thresholds)
    for threshold in sorted(tradeoff):
        row = tradeoff[threshold]
        print(f"  {threshold:>10.2f} {row['deferral_rate']:>15.1%} "
              f"{row['confidently_wrong_rate']:>24.1%}")

    return not failed


def main() -> int:
    from jarvis_nlu.model import OnnxClassifier
    from training.train import grouped_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout", action="store_true",
        help="Score against the hand-written eval/heldout.yaml set instead "
             "of the template-derived grouped_split test split.")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    defer_threshold = manifest["thresholds"]["defer"]
    classifier = OnnxClassifier(ROOT / "models")

    if args.heldout:
        print(f"Mode: held-out set ({HELDOUT_PATH})")
        examples = load_heldout()
        overlap = check_heldout_overlap(examples, ROOT / "training" / "templates")
        print(f"Held-out size: {len(examples)}")
        print(f"Overlap with generated training corpus: {len(overlap)} "
              f"(must be 0)")
        if overlap:
            print("  Overlapping texts (leaked -- remove or reword these "
                  "in eval/heldout.yaml):")
            for text in overlap:
                print(f"    {text!r}")
        tradeoff_thresholds = sorted({0.50, 0.60, 0.71, 0.80, defer_threshold})
    else:
        print("Mode: template-derived test split (grouped_split)")
        _train, _val, examples = grouped_split(
            build_dataset(ROOT / "training" / "templates"))
        tradeoff_thresholds = sorted({0.5, 0.6, 0.7, defer_threshold})

    report = score(classifier, examples, defer_threshold)
    passed = _print_report(report, classifier, examples, defer_threshold,
                           tradeoff_thresholds)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
