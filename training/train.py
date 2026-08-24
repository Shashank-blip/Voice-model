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

import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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
    """Temperature scaling. Raw softmax is overconfident; a temperature above
    1 spreads probability mass so the deferral threshold means something."""
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


def _choose_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Pick tau_defer as the lowest threshold meeting the spec's two gates:
    under 5% of in-scope deferred, under 2% of out_of_scope handled."""
    out_of_scope_index = LABELS.index("out_of_scope")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    in_scope = labels != out_of_scope_index
    out_of_scope = ~in_scope

    best = 0.5
    for candidate in np.arange(0.30, 0.95, 0.01):
        deferred_in_scope = (confidence[in_scope] < candidate).mean()
        leaked = ((confidence[out_of_scope] >= candidate)
                  & (predicted[out_of_scope] != out_of_scope_index)).mean()
        if deferred_in_scope < 0.05 and leaked < 0.02:
            best = float(candidate)
            break
    return {"defer": round(best, 3), "confirm": round(min(best + 0.20, 0.95), 3)}


def main() -> None:
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

    val_loader = DataLoader(_Rows(val_rows, tokenizer), batch_size=BATCH_SIZE)
    val_logits, val_labels = _logits_for(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    probabilities = torch.softmax(
        torch.tensor(val_logits) / temperature, dim=1).numpy()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "intent.pt")
    (models_dir / "manifest.json").write_text(json.dumps({
        "labels": LABELS,
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "temperature": temperature,
        "thresholds": _choose_thresholds(probabilities, val_labels),
        "trained_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"saved model + manifest to {models_dir}")


if __name__ == "__main__":
    main()
