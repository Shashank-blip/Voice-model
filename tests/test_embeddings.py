from datetime import datetime
from pathlib import Path

import pytest

from jarvis_nlu.storage import Storage

MODELS = Path(__file__).parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS / "encoder.onnx").exists() or not (MODELS / "tokenizer").exists(),
    reason="encoder not built; run training/export_onnx.py")


def test_semantic_search_beats_keyword_overlap(tmp_path):
    """The point of the embedding path: no shared words between query and note."""
    from jarvis_nlu.embeddings import embed, rank_notes
    store = Storage(tmp_path / "t.db")
    store.add_note("the router password is hunter2", embed("the router password is hunter2"))
    store.add_note("buy oat milk and bread", embed("buy oat milk and bread"))
    ranked = rank_notes(store, "what did I write about the wifi", embed)
    assert "hunter2" in ranked[0].text
    store.close()


def test_rank_notes_returns_empty_when_nothing_is_close(tmp_path):
    from jarvis_nlu.embeddings import embed, rank_notes
    store = Storage(tmp_path / "t.db")
    store.add_note("buy oat milk", embed("buy oat milk"))
    assert rank_notes(store, "quarterly tax filing deadlines", embed) == []
    store.close()
