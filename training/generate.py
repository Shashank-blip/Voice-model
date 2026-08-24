"""Synthesise the training corpus.

Three properties matter more than raw volume: an out_of_scope negative class
(without it the model cannot express 'defer'), STT-noise augmentation (the real
input is Whisper output, not typed text), and merged real phrases from the
learning log."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

LEARNING_LOG_WEIGHT = 3.0
NOISE_FRACTION = 0.25

HOMOPHONES = [("to", "two"), ("for", "four"), ("their", "there"),
              ("your", "you're"), ("its", "it's"), ("won", "one"),
              ("ate", "eight"), ("by", "buy")]
DROPPABLE = {"the", "a", "an", "please", "just", "some"}


@dataclass(frozen=True)
class Example:
    text: str
    intent: str
    template_id: str
    weight: float = 1.0


def add_stt_noise(text: str, rng: random.Random) -> str:
    """Perturb text the way Whisper realistically would."""
    words = text.split()
    choice = rng.random()
    if choice < 0.3 and len(words) > 3:
        candidates = [i for i, w in enumerate(words) if w.lower() in DROPPABLE]
        if candidates:
            words.pop(rng.choice(candidates))
    elif choice < 0.6:
        for index, word in enumerate(words):
            for left, right in HOMOPHONES:
                if word.lower() == left:
                    words[index] = right
                    break
    result = " ".join(words)
    if rng.random() < 0.5:
        result = result.rstrip(".?!")
    return result.lower() if rng.random() < 0.7 else result


def _expand(pattern: str, slots: dict[str, list[str]], rng: random.Random) -> list[str]:
    names = re.findall(r"\{(\w+)\}", pattern)
    if not names:
        return [pattern]
    value_lists = [slots.get(name, [""]) for name in names]
    combos = list(product(*value_lists))
    rng.shuffle(combos)
    combos = combos[:40]  # cap the blow-up from multi-slot patterns
    results = []
    for combo in combos:
        text = pattern
        for name, value in zip(names, combo):
            text = text.replace("{" + name + "}", value)
        results.append(re.sub(r"\s+", " ", text).strip())
    return results


def build_dataset(templates_dir: Path, learning_log: Path | None = None,
                  seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []

    for path in sorted(Path(templates_dir).glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        intent, slots = spec["intent"], spec.get("slots") or {}
        for index, pattern in enumerate(spec["patterns"]):
            template_id = f"{intent}:{index}"
            for text in _expand(pattern, slots, rng):
                examples.append(Example(text, intent, template_id))
                if rng.random() < NOISE_FRACTION:
                    examples.append(
                        Example(add_stt_noise(text, rng), intent, template_id))

    if learning_log and Path(learning_log).exists():
        for line in Path(learning_log).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("intent"):
                examples.append(Example(record["transcript"], record["intent"],
                                        "learning_log", LEARNING_LOG_WEIGHT))

    # Deduplicate on (text, intent), keeping the highest weight.
    best: dict[tuple[str, str], Example] = {}
    for example in examples:
        key = (example.text, example.intent)
        if key not in best or example.weight > best[key].weight:
            best[key] = example
    return sorted(best.values(), key=lambda e: (e.intent, e.text))


if __name__ == "__main__":
    root = Path(__file__).parent
    dataset = build_dataset(root / "templates",
                            learning_log=root.parent / "data" / "learning_log.jsonl")
    out = root / "data" / "dataset.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in dataset:
            handle.write(json.dumps(example.__dict__) + "\n")
    print(f"wrote {len(dataset)} examples to {out}")
