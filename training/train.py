"""Fine-tune MiniLM for intent classification, then calibrate its confidence.

The encoder is NOT frozen -- this is a genuine fine-tune, which is what lets
the model generalise to phrasings absent from the templates.

Two independent loss weightings are combined:

1. Per-class weights, computed from inverse training-set frequency. The
   corpus is imbalanced 7x (add_calendar_event has 722 examples, sleep has
   103); without this, the thin small-talk/meta intents are systematically
   under-predicted and Task 14's macro-F1 gate (which weights every class
   equally) fails on exactly those classes.
2. Per-example weights (``Example.weight``), which up-weight real phrases
   merged in from the learning log.

These address different problems and are multiplied together, not used as
alternatives to one another.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python training/train.py` (script's own directory on sys.path[0],
# not the repo root) as well as `python -m training.train` (repo root
# already on sys.path). Must run before the local-package imports below.
_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORTS))

import json
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from jarvis_nlu.intents import ALL_INTENTS
from training.generate import Example, build_dataset

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 48
EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
ROOT = Path(__file__).parent.parent
LABELS = [i.value for i in ALL_INTENTS]

# Fixes DataLoader shuffling, dropout, and head-init randomness so
# intent.pt, temperature, and thresholds are reproducible run-to-run --
# Task 14's gates assert on these artifacts. Recorded in the manifest.
SEED = 0

# One confidently-wrong local answer (a bad autonomous action, or a wrong
# answer presented as fact) costs roughly as much user trust/support burden
# as five wasted deferrals to the cloud API. Used by _choose_thresholds.
MISCLASSIFY_COST = 5.0


def grouped_split(examples: list[Example], seed: int = 0
                  ) -> tuple[list[Example], list[Example], list[Example]]:
    """Split by template_id, not by row. Splitting by row would put paraphrases
    of the same template in both train and test and inflate every metric."""
    rng = random.Random(seed)
    by_intent: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        if example.template_id not in by_intent[example.intent]:
            by_intent[example.intent].append(example.template_id)

    assignment: dict[str, str] = {}
    for intent, template_ids in by_intent.items():
        ids = sorted(template_ids)
        rng.shuffle(ids)
        # Guarantee at least one template per split so every intent is present
        # in val and test even when an intent has few templates.
        n_val = max(1, round(len(ids) * 0.1)) if len(ids) >= 3 else 1
        n_test = max(1, round(len(ids) * 0.1)) if len(ids) >= 3 else 1
        n_train = len(ids) - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Intent {intent!r} has only {len(ids)} template(s), which "
                f"leaves zero for training after reserving {n_val} for val "
                f"and {n_test} for test. Add at least "
                f"{n_val + n_test + 1} templates for this intent -- "
                f"otherwise compute_class_weights silently assigns it a "
                f"default weight and masks the missing training data.")
        for index, template_id in enumerate(ids):
            if index < n_val:
                assignment[template_id] = "val"
            elif index < n_val + n_test:
                assignment[template_id] = "test"
            else:
                assignment[template_id] = "train"

    buckets: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for example in examples:
        buckets[assignment.get(example.template_id, "train")].append(example)
    return buckets["train"], buckets["val"], buckets["test"]


def compute_class_weights(examples: list[Example]) -> torch.Tensor:
    """Inverse-frequency class weights over the training split, normalised so
    the mean weight is 1.0 (keeps the mean loss scale comparable to an
    unweighted cross-entropy, so LEARNING_RATE does not need retuning)."""
    counts = Counter(example.intent for example in examples)
    weights = torch.tensor(
        [1.0 / counts.get(label, 1) for label in LABELS], dtype=torch.float32)
    weights *= len(LABELS) / weights.sum()
    return weights


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: divide logits by a learned scalar T before the
    softmax to correct miscalibration, in EITHER direction. T > 1 spreads
    probability mass (softens overconfident logits); T < 1 concentrates it
    (sharpens underconfident logits). T is fit by minimising NLL on a
    held-out split, so it moves whichever way that split's miscalibration
    demands -- it is not assumed to be > 1.

    Measured on this corpus's val split: mean max-softmax-prob 0.8811 vs.
    accuracy 0.8855, i.e. the raw model is marginally UNDER-confident, so
    NLL is genuinely minimised by sharpening (T fit to ~0.877, not >1).
    That fit improves val NLL 0.3124 -> 0.2991 and generalises to test."""
    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(logit_tensor / log_temperature.exp(), label_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.exp().item())


class IntentModel(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(self.dropout(pooled))


class _Rows(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples, self.tokenizer = examples, tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        encoded = self.tokenizer(example.text, truncation=True, padding="max_length",
                                 max_length=MAX_LENGTH, return_tensors="pt")
        return (encoded["input_ids"][0], encoded["attention_mask"][0],
                LABELS.index(example.intent), example.weight)


def _logits_for(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, _weight in loader:
            out = model(input_ids.to(device), attention_mask.to(device))
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


CANDIDATE_THRESHOLDS = np.arange(0.10, 0.95, 0.01)

# The confirmation band [tau_defer, tau_confirm) must never collapse to
# empty -- see the comment in _choose_thresholds. A precise model can
# satisfy the 95%-precision criterion for tau_confirm at or below
# tau_defer, so tau_confirm is floored at tau_defer + CONFIRM_MARGIN.
CONFIRM_MARGIN = 0.10


def _choose_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Choose tau_defer by minimising an explicit expected-cost objective.
    The caller MUST pass a held-out split the model's confidence was not
    calibrated on (the TEST split -- never the split temperature was fitted
    on: see Important 3 in the task-12 review).

        cost(tau) = MISCLASSIFY_COST * P(in_scope and conf>=tau and pred!=y)
                  + MISCLASSIFY_COST * P(oos      and conf>=tau and pred!=oos)
                  + 1.0               * P(in_scope and conf<tau)

    (All three P(...) terms are joint probabilities over the whole split --
    they share one denominator, len(labels).)

    The previous implementation returned the LOWEST tau clearing two
    independent gates (in-scope deferral rate under 5%, out-of-scope leakage
    under 2%). Those two gates move in OPPOSITE directions as tau rises --
    deferral rises, leakage falls -- so "first tau clearing both" always
    collapses to the most permissive tau in range, and it never measured the
    failure that actually matters: a CONFIDENTLY WRONG in-scope answer.
    Explicit cost minimisation scores that failure directly, at
    MISCLASSIFY_COST times the cost of a mere deferral. Ties break toward
    the higher threshold (more conservative).

    tau_confirm is chosen independently, not as a fixed offset from
    tau_defer: the smallest threshold at which locally-answered in-scope
    predictions are right at least 95% of the time. That precision
    criterion alone is not enough, though: a sufficiently precise model
    satisfies it at or below tau_defer, which would collapse the
    confirmation band [tau_defer, tau_confirm) to empty and let destructive
    intents execute without ever asking. So tau_confirm is additionally
    floored at tau_defer + CONFIRM_MARGIN (see that constant) and capped at
    0.99.
    """
    out_of_scope_index = LABELS.index("out_of_scope")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    in_scope = labels != out_of_scope_index
    out_of_scope = ~in_scope
    n_total = max(len(labels), 1)

    best_tau, best_cost = float(CANDIDATE_THRESHOLDS[0]), float("inf")
    for tau in CANDIDATE_THRESHOLDS:
        answered = confidence >= tau
        wrong_in_scope = in_scope & answered & (predicted != labels)
        wrong_oos = out_of_scope & answered & (predicted != out_of_scope_index)
        deferred_in_scope = in_scope & ~answered

        cost = (MISCLASSIFY_COST * wrong_in_scope.sum() / n_total
                + MISCLASSIFY_COST * wrong_oos.sum() / n_total
                + 1.0 * deferred_in_scope.sum() / n_total)
        if cost <= best_cost:  # <= : ties break toward the HIGHER threshold
            best_cost = cost
            best_tau = float(tau)

    tau_defer = round(best_tau, 3)

    precision_threshold = 0.95
    for tau in CANDIDATE_THRESHOLDS:
        answered_in_scope = in_scope & (confidence >= tau)
        if answered_in_scope.sum() == 0:
            continue
        precision = (predicted[answered_in_scope]
                     == labels[answered_in_scope]).mean()
        if precision >= 0.95:
            precision_threshold = float(tau)
            break
    # Guarantee a non-empty confirmation band: [defer, confirm) must never
    # collapse to empty. The router only asks a destructive intent to
    # confirm when defer <= confidence < confirm -- if confirm == defer
    # that band is empty and destructive intents execute silently the
    # instant they clear defer. This is the only thing standing between a
    # misheard destructive command and silent data loss, so confirm is
    # floored at tau_defer + CONFIRM_MARGIN even when the precision
    # criterion is already satisfied at or below tau_defer (a precise
    # model collapses precision_threshold down to tau_defer otherwise).
    tau_confirm = round(min(max(precision_threshold, tau_defer + CONFIRM_MARGIN), 0.99), 3)

    return {"defer": tau_defer, "confirm": tau_confirm}


def cost_curve(probabilities: np.ndarray, labels: np.ndarray,
               taus: list[float]) -> dict[float, float]:
    """Expose the per-tau cost from _choose_thresholds's objective, for
    reporting/debugging. Not used by main(); kept small and side-effect
    free so it is safe to import from a one-off analysis script."""
    out_of_scope_index = LABELS.index("out_of_scope")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    in_scope = labels != out_of_scope_index
    out_of_scope = ~in_scope
    n_total = max(len(labels), 1)

    curve = {}
    for tau in taus:
        answered = confidence >= tau
        wrong_in_scope = in_scope & answered & (predicted != labels)
        wrong_oos = out_of_scope & answered & (predicted != out_of_scope_index)
        deferred_in_scope = in_scope & ~answered
        curve[tau] = (MISCLASSIFY_COST * wrong_in_scope.sum() / n_total
                     + MISCLASSIFY_COST * wrong_oos.sum() / n_total
                     + 1.0 * deferred_in_scope.sum() / n_total)
    return curve


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    _seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = build_dataset(ROOT / "training" / "templates",
                             learning_log=ROOT / "data" / "learning_log.jsonl")
    train_rows, val_rows, test_rows = grouped_split(examples)
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    class_weights = compute_class_weights(train_rows).to(device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = IntentModel(len(LABELS)).to(device)
    loader = DataLoader(_Rows(train_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # weight=class_weights offsets intent-frequency imbalance (inverse
    # training-set frequency, mean-normalised); reduction="none" lets us also
    # multiply by the per-example weight (learning-log up-weighting) below --
    # the two mechanisms are independent and both apply.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for input_ids, attention_mask, labels, weights in loader:
            optimiser.zero_grad()
            logits = model(input_ids.to(device), attention_mask.to(device))
            losses = loss_fn(logits, labels.to(device))
            loss = (losses * weights.float().to(device)).mean()
            loss.backward()
            optimiser.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{EPOCHS} loss={total / max(len(loader), 1):.4f}")

    # Temperature is fit on val (correct: it needs a held-out split, and
    # test must stay untouched for threshold selection below).
    val_loader = DataLoader(_Rows(val_rows, tokenizer), batch_size=BATCH_SIZE)
    val_logits, val_labels = _logits_for(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)

    # Thresholds are chosen on test -- a split neither the weights nor the
    # temperature were fitted on -- so the cost estimate isn't optimistic.
    test_loader = DataLoader(_Rows(test_rows, tokenizer), batch_size=BATCH_SIZE)
    test_logits, test_labels = _logits_for(model, test_loader, device)
    test_probabilities = torch.softmax(
        torch.tensor(test_logits) / temperature, dim=1).numpy()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "intent.pt")
    (models_dir / "manifest.json").write_text(json.dumps({
        "labels": LABELS,
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "temperature": temperature,
        "thresholds": _choose_thresholds(test_probabilities, test_labels),
        "seed": SEED,
        "trained_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"saved model + manifest to {models_dir}")


if __name__ == "__main__":
    main()
