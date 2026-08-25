"""Semantic note search. `encoder.onnx` holds the same MiniLM base checkpoint
already resident for classification, so retrieval needs no separate model
download -- embeddings are a BLOB column, not a vector database. It is a
*fresh* copy of that checkpoint though, not the classification-fine-tuned
weights: fine-tuning on the 20-way intent boundary collapses the
mean-pooled embedding geometry (measured similarity for unrelated sentence
pairs, ~0.6, ends up *higher* than for genuinely related ones, ~0.4), which
makes it useless for ranking. The untouched checkpoint keeps the
contrastively-trained sentence-embedding geometry it was built for, where
unrelated pairs land near 0 and related pairs land well above the floor
below. See `training/export_onnx.py` for the export.

SIMILARITY_FLOOR was picked empirically against this checkpoint's own
mean-pooled + L2-normalised cosine scores, not copied from a generic
sentence-similarity convention: unrelated note/query pairs measured here
top out around 0.2, genuinely related pairs measured here start around
0.3-0.5, so the floor sits at the midpoint with margin on both sides."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from jarvis_nlu.storage import Note, Storage

SIMILARITY_FLOOR = 0.27
MODELS_DIR = Path(__file__).parent.parent / "models"


@lru_cache(maxsize=1)
def _encoder():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODELS_DIR / "tokenizer" / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=64)
    tokenizer.enable_padding(length=64)
    session = ort.InferenceSession(str(MODELS_DIR / "encoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    return tokenizer, session


def embed(text: str) -> bytes:
    tokenizer, session = _encoder()
    encoded = tokenizer.encode(text)
    hidden = session.run(None, {
        "input_ids": np.array([encoded.ids], dtype=np.int64),
        "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
    })[0]
    mask = np.array(encoded.attention_mask, dtype=np.float32)[None, :, None]
    pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
    vector = pooled[0] / (np.linalg.norm(pooled[0]) + 1e-9)
    return vector.astype(np.float32).tobytes()


def _vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def rank_notes(storage: Storage, query: str, embedder) -> list[Note]:
    candidates = storage.notes_with_embeddings()
    if not candidates:
        return []
    query_vector = _vector(embedder(query))
    scored = [(float(np.dot(query_vector, _vector(n.embedding))), n) for n in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [note for score, note in scored if score >= SIMILARITY_FLOOR]
