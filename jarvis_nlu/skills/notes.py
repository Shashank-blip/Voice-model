"""Note skills: add, list recent, search.

`search` takes an optional `embedder: Callable[[str], bytes] | None`. Task 13
supplies a real one backed by `jarvis_nlu.embeddings.rank_notes`; until then,
every caller passes `embedder=None` and `search` falls back to substring
matching over note text, so this skill is fully testable standalone.

The substring fallback filters out common English stopwords (see
`_STOPWORDS`) in addition to its length-3+ filter -- without that, a query
like "what did I note about the wifi" reduces to the terms {"the", "wifi"},
and "the" alone matches nearly every note in the store, surfacing unrelated
results ahead of the actual match. If stopword filtering leaves no terms at
all, the fallback asks what to search for rather than matching everything.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

from jarvis_nlu.storage import Storage

_ADD_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?"
    r"(?:take|make|save|add|write|jot)?\s*(?:me\s+)?(?:a\s+|down\s+a\s+)?"
    r"note(?:\s+(?:that|about|down))?[:,]?\s*"
    r"|^\s*remember(?:\s+that)?\s*", re.I)

_SEARCH_LEAD_IN = re.compile(
    r"^\s*(?:what did i|what'd i|did i)?\s*"
    r"(?:note|write|say|save|jot)?\s*(?:down)?\s*(?:about|regarding|on)\s*", re.I)

# Stopwords excluded from the substring fallback's search terms -- a query
# like "what did I note about the wifi" reduces (after `_SEARCH_LEAD_IN`
# strips its lead-in) to "the wifi"; without this filter "the" alone would
# match nearly every note in the store.
_STOPWORDS = {
    "the", "and", "for", "about", "that", "this", "with", "from", "was",
    "were", "what", "when", "where", "did", "does", "note", "notes",
    "have", "has", "had", "you", "your", "out", "get", "got",
}

Embedder = Callable[[str], bytes]


def add(storage: Storage, text: str, now: datetime,
        embedder: Embedder | None = None) -> str:
    body = _ADD_LEAD_IN.sub("", text).strip(" ,.")
    if not body:
        return "What would you like me to note down, sugar?"
    storage.add_note(body, embedder(body) if embedder else None, now=now)
    return "Noted, sugar."


def list_recent(storage: Storage, now: datetime) -> str:
    saved = storage.list_notes(limit=5)
    if not saved:
        return "You've got no notes saved yet."
    return "Here's what you've noted: " + "; ".join(n.text for n in saved) + "."


def search(storage: Storage, text: str, embedder: Embedder | None = None) -> str:
    query = _SEARCH_LEAD_IN.sub("", text).strip(" ,.?")
    if not query:
        return "What should I look for, sugar?"

    if embedder is not None:
        from jarvis_nlu.embeddings import rank_notes  # provided by Task 13
        ranked = rank_notes(storage, query, embedder)
        if ranked:
            return "You noted: " + "; ".join(n.text for n in ranked[:3]) + "."
        return "I couldn't find a note about that, sugar."

    # Substring fallback: keeps this skill usable before the embedder exists.
    # Stopwords are dropped on top of the length filter -- see `_STOPWORDS`.
    terms = [t for t in re.findall(r"\w+", query.lower())
             if len(t) > 2 and t not in _STOPWORDS]
    if not terms:
        return "What should I look for, sugar?"
    hits = [n for n in storage.list_notes(limit=100)
            if any(t in n.text.lower() for t in terms)]
    if not hits:
        return "I couldn't find a note about that, sugar."
    return "You noted: " + "; ".join(n.text for n in hits[:3]) + "."
