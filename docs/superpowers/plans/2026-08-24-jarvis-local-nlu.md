# Jarvis Local NLU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `RAG-Voice_model`'s shadowing regex router with a locally trained MiniLM intent classifier, and have `miss-minutes` consult it before spending an OpenRouter request.

**Architecture:** `RAG-Voice_model/` becomes `jarvis_nlu`, an installable package with no network calls: a fine-tuned MiniLM intent classifier (ONNX, int8), a rule-based datetime slot extractor, SQLite storage, deterministic skill modules, and a proactive scheduler thread. `miss-minutes` installs it editable and calls `Assistant.handle(text)`; only `handled=False` falls through to OpenRouter. The classifier sits behind a `Protocol`, so tasks 1–10 are built and tested against a fake before any training runs.

**Tech Stack:** Python 3.13, PyTorch 2.9 (CPU), transformers 4.57, sentence-transformers 5.1, onnxruntime 1.23, scikit-learn 1.6, SQLite (stdlib), pytest 8.3.

**Spec:** `docs/superpowers/specs/2026-08-24-jarvis-local-nlu-design.md`

## Global Constraints

- **`jarvis_nlu` makes no network calls at runtime.** No `requests`, no `httpx`, no API clients. Training scripts may download the base encoder; the runtime package may not.
- **`intents.py` is the single source of truth.** Training, evaluation, routing and tests all read the roster from it. Never hardcode an intent string list anywhere else.
- **All datetimes are stored as ISO-8601 strings and are timezone-naive local time.** Convert at the storage boundary, never above it.
- **Every function that depends on the current time takes an injectable `now: datetime` parameter.** No bare `datetime.now()` below the router boundary.
- **v1 intent count is exactly 20.** The v2 backlog (file search, system info, apps, media) is out of scope.
- **Thresholds `τ_defer` and `τ_confirm` live in the model manifest**, never in `Config` and never hardcoded.
- **Test framework is pytest**, run from `RAG-Voice_model/` as `python -m pytest`.
- **Python version floor: 3.11** (uses `X | Y` unions and `tomllib`-era stdlib).

---

## File Structure

**`RAG-Voice_model/` (created/rewritten)**

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `jarvis_nlu/__init__.py` | Public exports: `Assistant`, `Result`, `Config` |
| `jarvis_nlu/intents.py` | Intent enum + intent group sets |
| `jarvis_nlu/config.py` | `Config` dataclass + JSON loading |
| `jarvis_nlu/storage.py` | SQLite schema, migration, row DAOs |
| `jarvis_nlu/slots.py` | Rule-based datetime grammar |
| `jarvis_nlu/responses.py` | Small-talk pools, no-repeat selection |
| `jarvis_nlu/model.py` | `Classifier` protocol, `Thresholds`, `OnnxClassifier` |
| `jarvis_nlu/router.py` | `Result`, `Assistant.handle` |
| `jarvis_nlu/proactive.py` | Scheduler thread |
| `jarvis_nlu/skills/clock.py` | `get_time`, `get_date` |
| `jarvis_nlu/skills/smalltalk.py` | 5 small-talk intents |
| `jarvis_nlu/skills/reminders.py` | add / list / cancel |
| `jarvis_nlu/skills/calendar.py` | add event / query day |
| `jarvis_nlu/skills/notes.py` | add / list / semantic search |
| `training/templates/*.yaml` | One template file per intent |
| `training/generate.py` | Dataset synthesis + STT-noise augmentation |
| `training/train.py` | Fine-tune + temperature calibration |
| `training/export_onnx.py` | ONNX export + int8 quantization |
| `training/evaluate.py` | Metrics, confusion matrix, gate enforcement |
| `tests/` | One test module per source module |

**Deleted:** `jarvis/app.py`, `jarvis/services.py`, `jarvis/storage.py`, `jarvis/voice.py`, `jarvis/__main__.py`, `tests/test_jarvis.py`. The regex router is removed wholesale, not patched — its ordering bugs are structural.

**`miss-minutes/` (modified)**

| Path | Change |
|---|---|
| `brain/assistant.py` | `ask()` consults `jarvis_nlu` first, LLM only on deferral |
| `main.py` | Construct `Assistant`, start scheduler, pass `speak` callback |
| `requirements.txt` | Add `-e ../RAG-Voice_model` |
| `brain/tools.py` | **Deleted** — superseded by `jarvis_nlu.skills` |

---

### Task 1: Package scaffold, intents registry, config

**Files:**
- Create: `RAG-Voice_model/pyproject.toml`, `jarvis_nlu/__init__.py`, `jarvis_nlu/intents.py`, `jarvis_nlu/config.py`
- Delete: `RAG-Voice_model/jarvis/` (whole directory), `RAG-Voice_model/tests/test_jarvis.py`
- Test: `tests/test_intents.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Intent` (str enum, 20 members), `ALL_INTENTS: tuple[Intent, ...]`, `CHORE_INTENTS`, `SMALLTALK_INTENTS`, `DESTRUCTIVE_INTENTS` (all `frozenset[Intent]`), `Config` frozen dataclass with `.load(path=None) -> Config`.

- [ ] **Step 1: Initialise git and scaffold**

The repo is not under version control, and every task ends in a commit.

```bash
cd "d:/Miss Minutes"
git init
printf '%s\n' 'data/' '__pycache__/' '*.py[cod]' '.pytest_cache/' '.hypothesis/' \
  'jarvis.config.json' 'RAG-Voice_model/models/' 'miss-minutes/.env' > .gitignore
mkdir -p RAG-Voice_model/jarvis_nlu/skills RAG-Voice_model/tests
rm -rf RAG-Voice_model/jarvis RAG-Voice_model/tests/test_jarvis.py
```

- [ ] **Step 2: Write the failing tests**

`tests/test_intents.py`:

```python
from jarvis_nlu.intents import (
    ALL_INTENTS, CHORE_INTENTS, DESTRUCTIVE_INTENTS, Intent, SMALLTALK_INTENTS,
)


def test_v1_has_exactly_twenty_intents():
    assert len(ALL_INTENTS) == 20


def test_intent_values_are_unique():
    assert len({i.value for i in ALL_INTENTS}) == 20


def test_out_of_scope_is_not_a_chore_or_smalltalk():
    assert Intent.OUT_OF_SCOPE not in CHORE_INTENTS
    assert Intent.OUT_OF_SCOPE not in SMALLTALK_INTENTS


def test_cancel_reminder_is_destructive():
    assert Intent.CANCEL_REMINDER in DESTRUCTIVE_INTENTS


def test_groups_are_subsets_of_all_intents():
    for group in (CHORE_INTENTS, SMALLTALK_INTENTS, DESTRUCTIVE_INTENTS):
        assert group <= set(ALL_INTENTS)
```

`tests/test_config.py`:

```python
import json
from pathlib import Path

from jarvis_nlu.config import Config


def test_missing_file_yields_defaults(tmp_path):
    config = Config.load(tmp_path / "absent.json")
    assert config.tick_seconds == 20
    assert config.daily_brief_at == "08:30"
    assert config.persona_path is None


def test_file_overrides_defaults(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"tick_seconds": 5, "daily_brief_at": None}))
    config = Config.load(path)
    assert config.tick_seconds == 5
    assert config.daily_brief_at is None


def test_paths_are_resolved_to_path_objects(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"db_path": "data/custom.db"}))
    assert isinstance(Config.load(path).db_path, Path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd RAG-Voice_model && python -m pytest tests/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu'`

- [ ] **Step 4: Write the implementation**

`jarvis_nlu/intents.py`:

```python
"""The v1 intent roster. This module is the single source of truth --
training, evaluation, routing and tests all read the roster from here."""
from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    # Reminders
    ADD_REMINDER = "add_reminder"
    LIST_REMINDERS = "list_reminders"
    CANCEL_REMINDER = "cancel_reminder"
    # Calendar
    ADD_CALENDAR_EVENT = "add_calendar_event"
    QUERY_CALENDAR = "query_calendar"
    # Notes
    ADD_NOTE = "add_note"
    LIST_NOTES = "list_notes"
    SEARCH_NOTES = "search_notes"
    # Clock
    GET_TIME = "get_time"
    GET_DATE = "get_date"
    # Small talk
    GREETING = "greeting"
    HOW_ARE_YOU = "how_are_you"
    USER_MOOD = "user_mood"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    # Meta
    AFFIRM = "affirm"
    DENY = "deny"
    REPEAT_LAST = "repeat_last"
    SLEEP = "sleep"
    # Fallback
    OUT_OF_SCOPE = "out_of_scope"


ALL_INTENTS: tuple[Intent, ...] = tuple(Intent)

CHORE_INTENTS: frozenset[Intent] = frozenset({
    Intent.ADD_REMINDER, Intent.LIST_REMINDERS, Intent.CANCEL_REMINDER,
    Intent.ADD_CALENDAR_EVENT, Intent.QUERY_CALENDAR,
    Intent.ADD_NOTE, Intent.LIST_NOTES, Intent.SEARCH_NOTES,
    Intent.GET_TIME, Intent.GET_DATE,
})

SMALLTALK_INTENTS: frozenset[Intent] = frozenset({
    Intent.GREETING, Intent.HOW_ARE_YOU, Intent.USER_MOOD,
    Intent.THANKS, Intent.GOODBYE,
})

META_INTENTS: frozenset[Intent] = frozenset({
    Intent.AFFIRM, Intent.DENY, Intent.REPEAT_LAST, Intent.SLEEP,
})

# Intents that destroy user data and therefore require confirmation when the
# classifier is merely probable rather than confident.
DESTRUCTIVE_INTENTS: frozenset[Intent] = frozenset({Intent.CANCEL_REMINDER})

# Intents whose meaning depends on a resolved time slot.
TIME_SLOT_INTENTS: frozenset[Intent] = frozenset({
    Intent.ADD_REMINDER, Intent.ADD_CALENDAR_EVENT,
})
```

`jarvis_nlu/config.py`:

```python
"""Runtime configuration. Every field has a default, so a missing file is
valid. Model thresholds deliberately live in the model manifest, not here --
a retrain must not leave a stale hand-tuned value behind."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_NAME = "jarvis.config.json"


@dataclass(frozen=True)
class Config:
    model_dir: Path = Path("models")
    db_path: Path = Path("data/jarvis.db")
    tick_seconds: int = 20
    daily_brief_at: str | None = "08:30"
    persona_path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_NAME
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        defaults = cls()
        return cls(
            model_dir=Path(raw.get("model_dir", defaults.model_dir)),
            db_path=Path(raw.get("db_path", defaults.db_path)),
            tick_seconds=int(raw.get("tick_seconds", defaults.tick_seconds)),
            daily_brief_at=raw.get("daily_brief_at", defaults.daily_brief_at),
            persona_path=Path(raw["persona_path"]) if raw.get("persona_path") else None,
        )
```

`jarvis_nlu/__init__.py`:

```python
"""Local-first NLU for the Jarvis / Miss Minutes voice assistant.

This package makes no network calls. Exports are added as tasks land.
"""
from jarvis_nlu.config import Config
from jarvis_nlu.intents import Intent

__all__ = ["Config", "Intent"]
```

`pyproject.toml`:

```toml
[project]
name = "jarvis-nlu"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "onnxruntime>=1.20", "PyYAML>=6.0"]

[project.optional-dependencies]
training = ["torch>=2.4", "transformers>=4.50", "sentence-transformers>=5.0",
            "scikit-learn>=1.5", "onnx>=1.17"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["jarvis_nlu*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Note the runtime deps exclude torch entirely — inference is onnxruntime only.

- [ ] **Step 5: Install, run tests, commit**

```bash
cd RAG-Voice_model && pip install -e . && python -m pytest tests/ -v
```
Expected: 8 passed.

```bash
git add -A && git commit -m "feat: scaffold jarvis_nlu with intent registry and config"
```

---

### Task 2: Storage layer

**Files:**
- Create: `jarvis_nlu/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Reminder`, `Event`, `Note` frozen dataclasses; `Storage` class with `add_reminder(text, due_at) -> int`, `list_reminders(include_delivered=False) -> list[Reminder]`, `find_reminders_by_text(needle) -> list[Reminder]`, `cancel_reminder(reminder_id) -> bool`, `due_reminders(now) -> list[Reminder]`, `mark_delivered(reminder_id, when) -> None`, `add_event(title, starts_at) -> int`, `events_on(day) -> list[Event]`, `add_note(text, embedding=None) -> int`, `list_notes(limit=20) -> list[Note]`, `notes_with_embeddings() -> list[Note]`, `set_note_embedding(note_id, embedding) -> None`, `get_meta(key) -> str | None`, `set_meta(key, value) -> None`, `close() -> None`; module constant `SCHEMA_VERSION = 1`.

- [ ] **Step 1: Write the failing tests**

`tests/test_storage.py`:

```python
from datetime import date, datetime

import pytest

from jarvis_nlu.storage import Storage


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


def test_add_and_list_reminder(store):
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    reminders = store.list_reminders()
    assert len(reminders) == 1
    assert reminders[0].text == "call mom"
    assert reminders[0].due_at == datetime(2026, 8, 24, 18, 0)
    assert reminders[0].delivered_at is None


def test_reminder_without_due_is_allowed(store):
    store.add_reminder("buy milk", None)
    assert store.list_reminders()[0].due_at is None


def test_due_reminders_excludes_future_delivered_and_cancelled(store):
    past = store.add_reminder("past", datetime(2026, 8, 24, 9, 0))
    store.add_reminder("future", datetime(2026, 8, 24, 23, 0))
    delivered = store.add_reminder("done", datetime(2026, 8, 24, 8, 0))
    cancelled = store.add_reminder("gone", datetime(2026, 8, 24, 8, 0))
    store.mark_delivered(delivered, datetime(2026, 8, 24, 8, 1))
    store.cancel_reminder(cancelled)

    due = store.due_reminders(datetime(2026, 8, 24, 12, 0))
    assert [r.id for r in due] == [past]


def test_undated_reminder_never_becomes_due(store):
    store.add_reminder("someday", None)
    assert store.due_reminders(datetime(2030, 1, 1)) == []


def test_cancel_returns_false_for_unknown_id(store):
    assert store.cancel_reminder(999) is False


def test_find_reminders_by_text_is_case_insensitive(store):
    store.add_reminder("Call Mom", datetime(2026, 8, 24, 18, 0))
    assert len(store.find_reminders_by_text("call mom")) == 1


def test_events_on_filters_by_day(store):
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    store.add_event("gym", datetime(2026, 8, 25, 7, 0))
    events = store.events_on(date(2026, 8, 24))
    assert [e.title for e in events] == ["dentist"]


def test_events_on_returns_chronological_order(store):
    store.add_event("late", datetime(2026, 8, 24, 18, 0))
    store.add_event("early", datetime(2026, 8, 24, 8, 0))
    assert [e.title for e in store.events_on(date(2026, 8, 24))] == ["early", "late"]


def test_notes_roundtrip_with_embedding(store):
    note_id = store.add_note("wifi password is hunter2")
    assert store.list_notes()[0].embedding is None
    store.set_note_embedding(note_id, b"\x00\x01")
    assert store.notes_with_embeddings()[0].embedding == b"\x00\x01"


def test_meta_roundtrip_and_overwrite(store):
    assert store.get_meta("last_brief_date") is None
    store.set_meta("last_brief_date", "2026-08-24")
    store.set_meta("last_brief_date", "2026-08-25")
    assert store.get_meta("last_brief_date") == "2026-08-25"


def test_reopening_database_is_idempotent(tmp_path):
    path = tmp_path / "test.db"
    first = Storage(path)
    first.add_reminder("persisted", None)
    first.close()

    second = Storage(path)
    assert len(second.list_reminders()) == 1
    second.close()


def test_migrates_legacy_minutes_schema(tmp_path):
    """miss-minutes' existing db has free-text `due` and no events table."""
    import sqlite3
    path = tmp_path / "minutes.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, due TEXT, created_at TEXT NOT NULL)"
    )
    legacy.execute(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    legacy.execute("INSERT INTO reminders (text, due, created_at) VALUES (?,?,?)",
                   ("call mom", "sometime next week", "2026-08-01T10:00:00"))
    legacy.execute("INSERT INTO notes (text, created_at) VALUES (?,?)",
                   ("old note", "2026-08-01T10:00:00"))
    legacy.commit()
    legacy.close()

    store = Storage(path)
    # No data discarded; unparseable due becomes NULL rather than dropping the row.
    assert [r.text for r in store.list_reminders()] == ["call mom"]
    assert store.list_reminders()[0].due_at is None
    assert [n.text for n in store.list_notes()] == ["old note"]
    store.close()

    # Migration is idempotent.
    again = Storage(path)
    assert len(again.list_reminders()) == 1
    again.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.storage'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/storage.py`:

```python
"""SQLite persistence. Datetimes cross this boundary as `datetime` objects and
are stored as ISO-8601 strings; nothing above this module handles the string
form. WAL mode is enabled because the proactive scheduler thread writes
concurrently with the main thread."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Reminder:
    id: int
    text: str
    due_at: datetime | None
    created_at: datetime
    delivered_at: datetime | None
    cancelled: bool


@dataclass(frozen=True)
class Event:
    id: int
    title: str
    starts_at: datetime


@dataclass(frozen=True)
class Note:
    id: int
    text: str
    created_at: datetime
    embedding: bytes | None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Storage:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the scheduler thread shares this connection,
        # serialised by SQLite's own locking plus the busy timeout.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    # ---------- schema ----------

    def _migrate(self) -> None:
        cur = self._conn
        cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                due_at TEXT,
                created_at TEXT NOT NULL,
                delivered_at TEXT,
                cancelled INTEGER NOT NULL DEFAULT 0)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                starts_at TEXT NOT NULL,
                created_at TEXT NOT NULL)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB)
        """)
        self._upgrade_legacy_reminders()
        self._upgrade_legacy_notes()
        cur.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),))
        self._conn.commit()

    def _upgrade_legacy_notes(self) -> None:
        """Same story as reminders: miss-minutes' notes table predates the
        `embedding` column, and CREATE TABLE IF NOT EXISTS will not add it."""
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(notes)")}
        if "embedding" not in columns:
            self._conn.execute("ALTER TABLE notes ADD COLUMN embedding BLOB")

    def _upgrade_legacy_reminders(self) -> None:
        """miss-minutes shipped `reminders` with a free-text `due` and none of
        our newer columns. `CREATE TABLE IF NOT EXISTS` above is a no-op on an
        existing table, so the missing columns must be added explicitly --
        otherwise every later query dies on `no such column: delivered_at`.
        Carry every row over; a `due` we cannot parse becomes NULL, keeping the
        reminder visible in listings but stopping it firing proactively.
        Idempotent: each step is guarded on the current column set."""
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(reminders)")}

        for name, ddl in (("due_at", "TEXT"),
                          ("delivered_at", "TEXT"),
                          ("cancelled", "INTEGER NOT NULL DEFAULT 0")):
            if name not in columns:
                self._conn.execute(f"ALTER TABLE reminders ADD COLUMN {name} {ddl}")

        if "due" not in columns:
            return
        from jarvis_nlu.slots import extract_time  # local import avoids a cycle

        now = datetime.now()
        rows = self._conn.execute(
            "SELECT id, due FROM reminders WHERE due IS NOT NULL").fetchall()
        for row in rows:
            slot = extract_time(row["due"], now)
            if slot is not None:
                self._conn.execute("UPDATE reminders SET due_at=? WHERE id=?",
                                   (slot.when.isoformat(), row["id"]))
        # SQLite 3.35+ (Python 3.13 ships 3.45) supports DROP COLUMN.
        self._conn.execute("ALTER TABLE reminders DROP COLUMN due")

    # ---------- reminders ----------

    def add_reminder(self, text: str, due_at: datetime | None) -> int:
        cur = self._conn.execute(
            "INSERT INTO reminders (text, due_at, created_at) VALUES (?,?,?)",
            (text, due_at.isoformat() if due_at else None, datetime.now().isoformat()))
        self._conn.commit()
        return int(cur.lastrowid)

    def _rows_to_reminders(self, rows) -> list[Reminder]:
        return [Reminder(r["id"], r["text"], _dt(r["due_at"]), _dt(r["created_at"]),
                         _dt(r["delivered_at"]), bool(r["cancelled"])) for r in rows]

    def list_reminders(self, include_delivered: bool = False) -> list[Reminder]:
        sql = "SELECT * FROM reminders WHERE cancelled=0"
        if not include_delivered:
            sql += " AND delivered_at IS NULL"
        sql += " ORDER BY due_at IS NULL, due_at ASC, id ASC"
        return self._rows_to_reminders(self._conn.execute(sql))

    def find_reminders_by_text(self, needle: str) -> list[Reminder]:
        return self._rows_to_reminders(self._conn.execute(
            "SELECT * FROM reminders WHERE cancelled=0 AND delivered_at IS NULL "
            "AND LOWER(text) LIKE ? ORDER BY id",
            (f"%{needle.lower()}%",)))

    def cancel_reminder(self, reminder_id: int) -> bool:
        cur = self._conn.execute("UPDATE reminders SET cancelled=1 WHERE id=? AND cancelled=0",
                                 (reminder_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def due_reminders(self, now: datetime) -> list[Reminder]:
        return self._rows_to_reminders(self._conn.execute(
            "SELECT * FROM reminders WHERE cancelled=0 AND delivered_at IS NULL "
            "AND due_at IS NOT NULL AND due_at <= ? ORDER BY due_at",
            (now.isoformat(),)))

    def mark_delivered(self, reminder_id: int, when: datetime) -> None:
        self._conn.execute("UPDATE reminders SET delivered_at=? WHERE id=?",
                           (when.isoformat(), reminder_id))
        self._conn.commit()

    # ---------- events ----------

    def add_event(self, title: str, starts_at: datetime) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (title, starts_at, created_at) VALUES (?,?,?)",
            (title, starts_at.isoformat(), datetime.now().isoformat()))
        self._conn.commit()
        return int(cur.lastrowid)

    def events_on(self, day: date) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE DATE(starts_at)=? ORDER BY starts_at",
            (day.isoformat(),))
        return [Event(r["id"], r["title"], _dt(r["starts_at"])) for r in rows]

    # ---------- notes ----------

    def add_note(self, text: str, embedding: bytes | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO notes (text, created_at, embedding) VALUES (?,?,?)",
            (text, datetime.now().isoformat(), embedding))
        self._conn.commit()
        return int(cur.lastrowid)

    def _rows_to_notes(self, rows) -> list[Note]:
        return [Note(r["id"], r["text"], _dt(r["created_at"]), r["embedding"]) for r in rows]

    def list_notes(self, limit: int = 20) -> list[Note]:
        return self._rows_to_notes(self._conn.execute(
            "SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)))

    def notes_with_embeddings(self) -> list[Note]:
        return self._rows_to_notes(self._conn.execute(
            "SELECT * FROM notes WHERE embedding IS NOT NULL"))

    def set_note_embedding(self, note_id: int, embedding: bytes) -> None:
        self._conn.execute("UPDATE notes SET embedding=? WHERE id=?", (embedding, note_id))
        self._conn.commit()

    # ---------- meta ----------

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
        self._conn.commit()
```

Note `_upgrade_legacy_reminders` imports `slots` lazily — Task 3 provides it, so run Task 3 before the legacy migration test passes. Mark that one test `@pytest.mark.xfail(reason="needs Task 3 slots")` until then.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_storage.py -v`
Expected: 11 passed, 1 xfail (legacy migration, unblocked by Task 3).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: SQLite storage with legacy minutes.db migration"
```

---

### Task 3: Datetime slot extraction

**Files:**
- Create: `jarvis_nlu/slots.py`
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TimeSlot` frozen dataclass with fields `when: datetime` and `residual: str`; `extract_time(text: str, now: datetime) -> TimeSlot | None`.

This is the highest-value task in the plan: it fixes the confirmed bug where `"remind me to call mom at 6 pm"` is rejected for lacking a day.

- [ ] **Step 1: Write the failing tests**

`tests/test_slots.py`:

```python
from datetime import datetime

import pytest

from jarvis_nlu.slots import extract_time

MORNING = datetime(2026, 8, 24, 9, 0)    # Monday 09:00
EVENING = datetime(2026, 8, 24, 20, 0)   # Monday 20:00


@pytest.mark.parametrize("text,now,expected", [
    # Bare time -> next occurrence of that clock time.
    ("at 6", MORNING, datetime(2026, 8, 24, 18, 0)),
    ("at 6", EVENING, datetime(2026, 8, 25, 6, 0)),
    ("at 10", MORNING, datetime(2026, 8, 24, 10, 0)),
    # Explicit meridiem wins over the next-occurrence rule.
    ("at 6 pm", MORNING, datetime(2026, 8, 24, 18, 0)),
    ("at 6 am", EVENING, datetime(2026, 8, 25, 6, 0)),
    ("at 6:30 pm", MORNING, datetime(2026, 8, 24, 18, 30)),
    # Midnight / noon boundaries.
    ("at 12 am", MORNING, datetime(2026, 8, 25, 0, 0)),
    ("at 12 pm", MORNING, datetime(2026, 8, 24, 12, 0)),
    # Relative day + time.
    ("tomorrow at 6", MORNING, datetime(2026, 8, 25, 18, 0)),
    ("today at 4", MORNING, datetime(2026, 8, 24, 16, 0)),
    ("tonight at 8", MORNING, datetime(2026, 8, 24, 20, 0)),
    # Named weekday. Monday now; Friday is 4 days out.
    ("friday", MORNING, datetime(2026, 8, 28, 9, 0)),
    ("on monday at 9", MORNING, datetime(2026, 8, 31, 9, 0)),
    ("next friday", MORNING, datetime(2026, 9, 4, 9, 0)),
    # ISO date.
    ("2026-08-30 at 4pm", MORNING, datetime(2026, 8, 30, 16, 0)),
    # Offsets.
    ("in 20 minutes", MORNING, datetime(2026, 8, 24, 9, 20)),
    ("in 2 hours", MORNING, datetime(2026, 8, 24, 11, 0)),
    ("in half an hour", MORNING, datetime(2026, 8, 24, 9, 30)),
    # Dayparts.
    ("tomorrow morning", MORNING, datetime(2026, 8, 25, 9, 0)),
    ("tomorrow afternoon", MORNING, datetime(2026, 8, 25, 14, 0)),
    ("tomorrow evening", MORNING, datetime(2026, 8, 25, 19, 0)),
    ("tonight", MORNING, datetime(2026, 8, 24, 20, 0)),
])
def test_resolves_when(text, now, expected):
    slot = extract_time(text, now)
    assert slot is not None, f"no time found in {text!r}"
    assert slot.when == expected


@pytest.mark.parametrize("text,expected_residual", [
    ("call mom at 6 pm", "call mom"),
    ("at 6 pm call mom", "call mom"),
    ("call mom tomorrow at 6", "call mom"),
    ("in 20 minutes take the bread out", "take the bread out"),
    ("dentist appointment on friday at 4pm", "dentist appointment"),
])
def test_strips_time_expression_from_residual(text, expected_residual):
    slot = extract_time(text, MORNING)
    assert slot is not None
    assert slot.residual == expected_residual


@pytest.mark.parametrize("text", [
    "call mom",
    "what is on my calendar",
    "",
    "remind me to buy milk",
])
def test_returns_none_when_no_time_present(text):
    assert extract_time(text, MORNING) is None


def test_named_weekday_matching_today_goes_to_next_week():
    """'monday' said on a Monday means next Monday, not zero days away."""
    slot = extract_time("monday", MORNING)
    assert slot.when.date() == datetime(2026, 8, 31).date()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.slots'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/slots.py`:

```python
"""Rule-based datetime extraction.

Deliberately hand-written rather than `dateparser`: the grammar here is small
and closed, and we need failure modes we control and can test exhaustively.
Returns both the resolved datetime and the text with the time expression
removed -- that residual becomes the reminder body."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}
DAYPARTS = {"morning": (9, 0), "afternoon": (14, 0), "evening": (19, 0),
            "night": (20, 0), "tonight": (20, 0), "noon": (12, 0), "midnight": (0, 0)}

_TIME = r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)?"

# Ordered most-specific first; the first pattern that matches wins.
_PATTERNS: list[tuple[str, str]] = [
    ("offset",   r"\bin\s+(?P<amount>a|an|half\s+an|\d+)\s+(?P<unit>minutes?|mins?|hours?|hrs?|days?)\b"),
    ("iso",      r"\b(?P<date>\d{4}-\d{2}-\d{2})\b(?:\s+at)?(?:\s+" + _TIME + r")?"),
    ("weekday",  r"\b(?P<next>next\s+)?(?:on\s+)?(?P<weekday>" + "|".join(WEEKDAYS) + r")\b(?:\s+at)?(?:\s+" + _TIME + r")?"),
    ("relday",   r"\b(?P<relday>today|tomorrow|tonight)\b(?:\s+(?P<daypart>morning|afternoon|evening|night))?(?:\s+at)?(?:\s+" + _TIME + r")?"),
    ("daypart",  r"\bthis\s+(?P<daypart2>morning|afternoon|evening)\b"),
    ("baretime", r"\bat\s+" + _TIME + r"\b"),
    ("looseti",  r"\b" + _TIME.replace("(?P<meridiem>am|pm|a\\.m\\.|p\\.m\\.)?", "(?P<meridiem>am|pm|a\\.m\\.|p\\.m\\.)") + r"\b"),
]


@dataclass(frozen=True)
class TimeSlot:
    when: datetime
    residual: str


def _clock(match: re.Match, default: tuple[int, int]) -> tuple[int, int]:
    """Resolve hour/minute from a match, falling back to `default`."""
    if not match.groupdict().get("hour"):
        return default
    hour, minute = int(match.group("hour")), int(match.group("minute") or 0)
    meridiem = (match.groupdict().get("meridiem") or "").replace(".", "").lower()
    if meridiem == "am":
        hour = 0 if hour == 12 else hour
    elif meridiem == "pm":
        hour = 12 if hour == 12 else hour + 12
    return hour, minute


def _next_occurrence(now: datetime, hour: int, minute: int) -> datetime:
    """Bare-time rule: no meridiem given, so pick the NEXT time the clock reads
    this. At 09:00 'at 6' is 18:00 today; at 20:00 it is 06:00 tomorrow."""
    for candidate_hour in (hour, hour + 12 if hour < 12 else hour - 12):
        candidate = now.replace(hour=candidate_hour % 24, minute=minute,
                                second=0, microsecond=0)
        if candidate > now:
            return candidate
    return now.replace(hour=hour % 24, minute=minute, second=0,
                       microsecond=0) + timedelta(days=1)


def _residual(text: str, match: re.Match) -> str:
    stripped = (text[:match.start()] + " " + text[match.end():])
    stripped = re.sub(r"\s+", " ", stripped).strip()
    # Drop dangling prepositions left behind by the removal.
    stripped = re.sub(r"\s+(at|on|by|in)$", "", stripped).strip()
    stripped = re.sub(r"^(at|on|by|in)\s+", "", stripped).strip()
    return stripped.strip(" ,.")


def extract_time(text: str, now: datetime) -> TimeSlot | None:
    for kind, pattern in _PATTERNS:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        when = _resolve(kind, match, now)
        if when is not None:
            return TimeSlot(when=when, residual=_residual(text, match))
    return None


def _resolve(kind: str, match: re.Match, now: datetime) -> datetime | None:
    groups = match.groupdict()

    if kind == "offset":
        raw = groups["amount"].lower()
        amount = 0.5 if "half" in raw else (1 if raw in ("a", "an") else int(raw))
        unit = groups["unit"].lower()
        if unit.startswith(("minute", "min")):
            return now + timedelta(minutes=amount)
        if unit.startswith(("hour", "hr")):
            return now + timedelta(hours=amount)
        return now + timedelta(days=amount)

    if kind == "iso":
        day = datetime.strptime(groups["date"], "%Y-%m-%d").date()
        hour, minute = _clock(match, (9, 0))
        return datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)

    if kind == "weekday":
        target = WEEKDAYS[groups["weekday"].lower()]
        ahead = (target - now.weekday()) % 7
        # 'monday' said on a Monday means next Monday, never today.
        ahead = ahead or 7
        if groups.get("next"):
            ahead += 7 if ahead <= 7 else 0
        hour, minute = _clock(match, (now.hour, now.minute))
        return (now + timedelta(days=ahead)).replace(hour=hour, minute=minute,
                                                     second=0, microsecond=0)

    if kind == "relday":
        word = groups["relday"].lower()
        day = now + timedelta(days=1 if word == "tomorrow" else 0)
        if groups.get("daypart"):
            hour, minute = DAYPARTS[groups["daypart"].lower()]
        elif groups.get("hour"):
            hour, minute = _clock(match, (9, 0))
        elif word == "tonight":
            hour, minute = DAYPARTS["tonight"]
        else:
            hour, minute = 9, 0
        return day.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if kind == "daypart":
        hour, minute = DAYPARTS[groups["daypart2"].lower()]
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if kind in ("baretime", "looseti"):
        hour, minute = int(match.group("hour")), int(match.group("minute") or 0)
        if hour > 23:
            return None
        meridiem = (groups.get("meridiem") or "").replace(".", "").lower()
        if meridiem:
            hour, minute = _clock(match, (hour, minute))
            candidate = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
            return candidate if candidate > now else candidate + timedelta(days=1)
        return _next_occurrence(now, hour, minute)

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_slots.py tests/test_storage.py -v`
Expected: all pass. Remove the `xfail` marker added in Task 2 — the legacy migration test now works.

If a parametrised case fails, fix the pattern rather than the expectation: the expectations encode the spec's §6 grammar.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: rule-based datetime slot extraction with next-occurrence rule"
```

---

### Task 4: Response pools

**Files:**
- Create: `jarvis_nlu/responses.py`
- Test: `tests/test_responses.py`

**Interfaces:**
- Consumes: `Intent` from Task 1.
- Produces: `ResponsePool` class with `__init__(pools: dict[str, list[str]] | None = None, history: int = 3)` and `pick(key: str) -> str`; module constant `POOLS: dict[str, list[str]]` keyed by intent value.

**Tone requirement (spec §7.1):** pools are written in the Miss Minutes voice — warm, folksy, "sugar", "mm-hmm". Local and LLM replies alternate inside one conversation; a neutral register makes the seam audible. Read `miss-minutes/config/persona.md` before writing them.

- [ ] **Step 1: Write the failing test**

`tests/test_responses.py`:

```python
import pytest

from jarvis_nlu.intents import SMALLTALK_INTENTS
from jarvis_nlu.responses import POOLS, ResponsePool


def test_every_smalltalk_intent_has_a_pool():
    for intent in SMALLTALK_INTENTS:
        assert intent.value in POOLS, f"missing pool for {intent.value}"


def test_pools_meet_minimum_variant_count():
    for key, variants in POOLS.items():
        assert len(variants) >= 8, f"{key} has only {len(variants)} variants"


def test_pool_variants_are_unique():
    for key, variants in POOLS.items():
        assert len(set(variants)) == len(variants), f"{key} has duplicates"


def test_pick_avoids_recent_repeats():
    pool = ResponsePool({"greeting": ["a", "b", "c", "d"]}, history=3)
    seen = [pool.pick("greeting") for _ in range(4)]
    # With history=3, the first four picks cannot repeat.
    assert len(set(seen)) == 4


def test_pick_recycles_once_history_rolls_over():
    pool = ResponsePool({"greeting": ["a", "b", "c", "d"]}, history=3)
    picks = [pool.pick("greeting") for _ in range(20)]
    assert set(picks) == {"a", "b", "c", "d"}


def test_pick_handles_pool_smaller_than_history():
    pool = ResponsePool({"thanks": ["only"]}, history=3)
    assert pool.pick("thanks") == "only"
    assert pool.pick("thanks") == "only"


def test_pick_raises_for_unknown_key():
    with pytest.raises(KeyError):
        ResponsePool({"greeting": ["a"]}).pick("nope")


def test_history_is_per_key():
    pool = ResponsePool({"a": ["x", "y"], "b": ["x", "y"]}, history=1)
    first = pool.pick("a")
    assert pool.pick("b") in {"x", "y"}  # b's history is independent
    assert pool.pick("a") != first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_responses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.responses'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/responses.py`. Write **at least 8 variants per key** in the persona voice. The keys below are required; `user_mood` takes three sub-keys because the skill branches on mood polarity.

```python
"""Small-talk response pools. Plain data -- editing these needs no retrain.

Voice matches miss-minutes/config/persona.md so that local replies and LLM
replies don't sound like two different assistants mid-conversation."""
from __future__ import annotations

import random
from collections import defaultdict, deque

POOLS: dict[str, list[str]] = {
    "greeting": [
        "Well hey there, sugar. What're we doin' today?",
        "Mornin'! What can I get sorted for you?",
        "Hey you. I'm all ears.",
        "There you are. What's on your mind?",
        "Howdy. What're we tacklin'?",
        "Hey, sugar. Whatcha need?",
        "Look who it is. What's up?",
        "I'm here. What's the plan?",
        "Hey! Good to hear you. What's first?",
        "Mm-hmm, I'm listenin'.",
    ],
    "how_are_you": [
        "Oh, I'm just fine, sugar. Tickin' along. How 'bout you?",
        "Can't complain — every second's right where it oughta be. You?",
        "Doin' good! Better now you're here. How're you holdin' up?",
        "Right as rain. What about yourself?",
        "I'm peachy. How's your day treatin' you?",
        "All good on my end. You doin' alright?",
        "Never better. How you feelin'?",
        "Just dandy, thanks for askin'. And you?",
    ],
    "user_mood_positive": [
        "Well that's just lovely to hear!",
        "Glad to hear it, sugar.",
        "Now that's what I like to hear.",
        "Good! Let's keep that goin'.",
        "Happy to hear it. What's next?",
        "That's the spirit.",
        "Love that for you. What're we doin'?",
        "Wonderful. Put me to work.",
    ],
    "user_mood_negative": [
        "Aw, sorry to hear that, sugar. Anything I can take off your plate?",
        "That's rough. Want me to keep things simple today?",
        "Sorry, hon. I'm here if you need somethin' handled.",
        "Rough one, huh? Let me know what'd help.",
        "That's no fun. Want me to hold onto anything for you?",
        "Sorry to hear it. I'll keep out of your hair unless you need me.",
        "Hang in there. What can I do?",
        "Well shoot. Anything I can help with?",
    ],
    "user_mood_tired": [
        "Sounds like you need a break, sugar. Want me to keep it light?",
        "Rough night? I'll keep things short.",
        "Tired's allowed. What's the one thing that's gotta get done?",
        "Go easy on yourself today. What d'you need?",
        "Mm, I hear that. Want me to handle the small stuff?",
        "Sounds like a long one. What can I take off you?",
        "Get yourself some rest soon. Anything first?",
        "Runnin' on empty, huh? Tell me what's urgent.",
    ],
    "thanks": [
        "Anytime, sugar.",
        "You bet.",
        "'Course. That's what I'm here for.",
        "Happy to help.",
        "Don't mention it.",
        "My pleasure.",
        "Anytime at all.",
        "That's what I'm for.",
    ],
    "goodbye": [
        "See you 'round, sugar.",
        "Catch you later!",
        "I'll be right here when you need me.",
        "Bye now. Holler if you need somethin'.",
        "Take care, hon.",
        "Later! I'll keep an eye on things.",
        "See ya. I'll hold down the fort.",
        "Alright, talk soon.",
    ],
    "sleep": [
        "Goin' quiet. Say the word when you need me.",
        "I'll hush up. Just holler.",
        "Alright, I'll sit tight.",
        "Standin' by, sugar.",
        "Quiet mode. I'm still here.",
        "Say no more. I'll wait.",
        "Zippin' it. Call when you need me.",
        "Restin' up. Just say the wake word.",
    ],
    "affirm": [
        "You got it.", "Alright then.", "Done and done.", "Sure thing, sugar.",
        "Consider it handled.", "Okay!", "On it.", "Mm-hmm, will do.",
    ],
    "deny": [
        "No worries, sugar.", "Alright, leavin' it be.", "Okay, scratch that.",
        "Fair enough.", "Understood.", "No problem at all.",
        "Alright, forget I asked.", "Sure, we'll skip it.",
    ],
    "unknown_error": [
        "Well shoot, somethin' went sideways on my end.",
        "Hm, that didn't take. Try me again?",
        "Somethin' snagged, sugar. Give it another go.",
        "That one got away from me. Mind repeatin'?",
        "Well that's embarrassin' — didn't work. Try again?",
        "My wires crossed. One more time?",
        "Didn't quite land. Say it again for me?",
        "Somethin's gummed up. Try once more?",
    ],
}


class ResponsePool:
    """Random selection that avoids the last `history` picks per key, so
    repeated greetings don't sound like a phone tree."""

    def __init__(self, pools: dict[str, list[str]] | None = None, history: int = 3):
        self._pools = pools if pools is not None else POOLS
        self._history = history
        self._recent: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=history))

    def pick(self, key: str) -> str:
        variants = self._pools[key]  # KeyError is intentional: an unknown key is a bug
        recent = self._recent[key]
        candidates = [v for v in variants if v not in recent] or list(variants)
        choice = random.choice(candidates)
        recent.append(choice)
        return choice
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_responses.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: small-talk response pools with no-repeat selection"
```

---

### Task 5: Clock and small-talk skills

**Files:**
- Create: `jarvis_nlu/skills/__init__.py`, `jarvis_nlu/skills/clock.py`, `jarvis_nlu/skills/smalltalk.py`
- Test: `tests/test_skills_clock.py`, `tests/test_skills_smalltalk.py`

**Interfaces:**
- Consumes: `ResponsePool` (Task 4).
- Produces: `clock.get_time(now: datetime) -> str`, `clock.get_date(now: datetime) -> str`; `smalltalk.respond(intent_value: str, text: str, pool: ResponsePool) -> str`, `smalltalk.classify_mood(text: str) -> str` returning one of `"positive" | "negative" | "tired"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skills_clock.py`:

```python
from datetime import datetime

from jarvis_nlu.skills import clock

NOW = datetime(2026, 8, 24, 14, 5)


def test_get_time_is_conversational_not_iso():
    reply = clock.get_time(NOW)
    assert "2:05" in reply
    assert "14:05" not in reply


def test_get_date_names_weekday_and_month():
    reply = clock.get_date(NOW)
    assert "Monday" in reply
    assert "August" in reply
    assert "24" in reply


def test_midnight_and_noon_read_naturally():
    assert "12:00 AM" in clock.get_time(datetime(2026, 8, 24, 0, 0))
    assert "12:00 PM" in clock.get_time(datetime(2026, 8, 24, 12, 0))
```

`tests/test_skills_smalltalk.py`:

```python
import pytest

from jarvis_nlu.responses import ResponsePool
from jarvis_nlu.skills import smalltalk


@pytest.mark.parametrize("text,expected", [
    ("i'm good thanks", "positive"),
    ("doing great", "positive"),
    ("pretty bad honestly", "negative"),
    ("i'm stressed out", "negative"),
    ("i'm exhausted", "tired"),
    ("so sleepy today", "tired"),
    ("bit tired honestly", "tired"),
])
def test_classify_mood(text, expected):
    assert smalltalk.classify_mood(text) == expected


def test_unknown_mood_defaults_to_positive():
    assert smalltalk.classify_mood("mm") == "positive"


def test_respond_uses_mood_specific_pool():
    pool = ResponsePool({"user_mood_tired": ["rest up"]}, history=1)
    assert smalltalk.respond("user_mood", "i'm exhausted", pool) == "rest up"


def test_respond_uses_plain_pool_for_non_mood_intents():
    pool = ResponsePool({"greeting": ["hey there"]}, history=1)
    assert smalltalk.respond("greeting", "hello", pool) == "hey there"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skills_clock.py tests/test_skills_smalltalk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.skills'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/skills/__init__.py`:

```python
"""Deterministic skill implementations. Skills receive storage and plain
arguments, return spoken strings, and never touch the classifier."""
```

`jarvis_nlu/skills/clock.py`:

```python
from __future__ import annotations

from datetime import datetime


def get_time(now: datetime) -> str:
    # %#I is the Windows form of %-I (no zero padding); fall back for other platforms.
    try:
        pretty = now.strftime("%#I:%M %p")
    except ValueError:
        pretty = now.strftime("%-I:%M %p")
    return f"It's {pretty}."


def get_date(now: datetime) -> str:
    return f"It's {now.strftime('%A, %B')} {now.day}."
```

`jarvis_nlu/skills/smalltalk.py`:

```python
from __future__ import annotations

import re

from jarvis_nlu.responses import ResponsePool

_TIRED = re.compile(r"\b(tired|exhausted|sleepy|knackered|wiped|drained|beat|worn out)\b", re.I)
_NEGATIVE = re.compile(
    r"\b(bad|awful|terrible|rough|stressed|anxious|sad|down|lousy|"
    r"not great|not good|could be better|meh|struggling|overwhelmed)\b", re.I)


def classify_mood(text: str) -> str:
    """Coarse polarity for `user_mood`. Tired is checked first because
    'tired' and 'not great' often co-occur and tired is the more actionable
    reading. Defaults to positive -- an over-cheery reply is a smaller failure
    than falsely commiserating."""
    if _TIRED.search(text):
        return "tired"
    if _NEGATIVE.search(text):
        return "negative"
    return "positive"


def respond(intent_value: str, text: str, pool: ResponsePool) -> str:
    if intent_value == "user_mood":
        return pool.pick(f"user_mood_{classify_mood(text)}")
    return pool.pick(intent_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skills_clock.py tests/test_skills_smalltalk.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: clock and small-talk skills"
```

---

### Task 6: Reminder skills

**Files:**
- Create: `jarvis_nlu/skills/reminders.py`
- Test: `tests/test_skills_reminders.py`

**Interfaces:**
- Consumes: `Storage`, `Reminder` (Task 2); `extract_time`, `TimeSlot` (Task 3).
- Produces: `reminders.add(storage, text, now) -> str`, `reminders.list_pending(storage, now) -> str`, `reminders.cancel(storage, text, now) -> str`, `reminders.needs_time(text, now) -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skills_reminders.py`:

```python
from datetime import datetime

import pytest

from jarvis_nlu.skills import reminders
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_strips_the_lead_in_and_the_time(store):
    reply = reminders.add(store, "remind me to call mom at 6 pm", NOW)
    saved = store.list_reminders()[0]
    assert saved.text == "call mom"
    assert saved.due_at == datetime(2026, 8, 24, 18, 0)
    assert "call mom" in reply


def test_add_handles_bare_time_without_a_day(store):
    """The bug: this was rejected outright before."""
    reminders.add(store, "remind me to call mom at 6", NOW)
    assert store.list_reminders()[0].due_at == datetime(2026, 8, 24, 18, 0)


def test_add_keeps_apostrophes_in_the_body(store):
    """The greeting-hijack bug phrase must survive as a real reminder."""
    reminders.add(store, "remind me I'm meeting Bob tomorrow at 5 pm", NOW)
    saved = store.list_reminders()[0]
    assert "meeting Bob" in saved.text
    assert saved.due_at == datetime(2026, 8, 25, 17, 0)


def test_needs_time_detects_a_missing_slot():
    assert reminders.needs_time("remind me to buy milk", NOW) is True
    assert reminders.needs_time("remind me to buy milk at 6", NOW) is False


def test_add_without_a_time_saves_undated(store):
    reply = reminders.add(store, "remind me to buy milk", NOW)
    saved = store.list_reminders()[0]
    assert saved.text == "buy milk"
    assert saved.due_at is None
    assert "when" in reply.lower()


def test_list_pending_says_so_when_empty(store):
    assert "no" in reminders.list_pending(store, NOW).lower()


def test_list_pending_includes_saved_items(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reminders.add(store, "remind me to pay rent tomorrow at 9 am", NOW)
    reply = reminders.list_pending(store, NOW)
    assert "call mom" in reply
    assert "pay rent" in reply


def test_cancel_removes_a_matching_reminder(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reply = reminders.cancel(store, "forget the mom reminder", NOW)
    assert store.list_reminders() == []
    assert "call mom" in reply


def test_cancel_reports_when_nothing_matches(store):
    reply = reminders.cancel(store, "cancel the dentist reminder", NOW)
    assert "couldn't find" in reply.lower() or "could not find" in reply.lower()


def test_cancel_asks_when_multiple_match(store):
    reminders.add(store, "remind me to call mom at 6 pm", NOW)
    reminders.add(store, "remind me to call mom back tomorrow at 9 am", NOW)
    reply = reminders.cancel(store, "cancel the call mom reminder", NOW)
    assert len(store.list_reminders()) == 2  # nothing destroyed on ambiguity
    assert "which" in reply.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skills_reminders.py -v`
Expected: FAIL — `ImportError: cannot import name 'reminders'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/skills/reminders.py`:

```python
from __future__ import annotations

import re
from datetime import datetime

from jarvis_nlu.slots import extract_time
from jarvis_nlu.storage import Storage

# Lead-ins stripped before the reminder body is stored.
_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:remind me(?:\s+(?:to|that|about))?|set a reminder(?:\s+(?:to|for|about))?|"
    r"don'?t let me forget(?:\s+to)?|nudge me(?:\s+(?:to|about))?)\s*", re.I)

_CANCEL_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:cancel|delete|remove|forget|drop|scrap)\s+"
    r"(?:the\s+|my\s+)?", re.I)
_CANCEL_TRAILER = re.compile(r"\s*reminder(?:s)?\s*$", re.I)


def _body(text: str) -> str:
    return _LEAD_IN.sub("", text).strip(" ,.")


def needs_time(text: str, now: datetime) -> bool:
    return extract_time(text, now) is None


def add(storage: Storage, text: str, now: datetime) -> str:
    slot = extract_time(text, now)
    # Strip the time first so the lead-in regex sees a clean body.
    body = _body(slot.residual if slot else text)
    if not body:
        body = "that"
    storage.add_reminder(body, slot.when if slot else None)
    if slot is None:
        return f"Saved — {body}. When should I remind you, sugar?"
    return f"You got it. I'll remind you to {body} at {_pretty(slot.when)}."


def _pretty(when: datetime) -> str:
    try:
        return when.strftime("%#I:%M %p on %A")
    except ValueError:
        return when.strftime("%-I:%M %p on %A")


def list_pending(storage: Storage, now: datetime) -> str:
    pending = storage.list_reminders()
    if not pending:
        return "You've got no reminders pendin', sugar."
    parts = []
    for reminder in pending[:5]:
        when = f" at {_pretty(reminder.due_at)}" if reminder.due_at else " (no time set)"
        parts.append(f"{reminder.text}{when}")
    return "Here's what you've got: " + "; ".join(parts) + "."


def cancel(storage: Storage, text: str, now: datetime) -> str:
    needle = _CANCEL_TRAILER.sub("", _CANCEL_LEAD_IN.sub("", text)).strip(" ,.")
    matches = storage.find_reminders_by_text(needle) if needle else []
    if not matches:
        return "I couldn't find a reminder matchin' that, sugar."
    if len(matches) > 1:
        listed = "; ".join(m.text for m in matches[:3])
        return f"I've got more than one of those — which did you mean? {listed}"
    storage.cancel_reminder(matches[0].id)
    return f"Done, I've dropped the reminder to {matches[0].text}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skills_reminders.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: reminder skills with ambiguity-safe cancellation"
```

---

### Task 7: Calendar and note skills

**Files:**
- Create: `jarvis_nlu/skills/calendar.py`, `jarvis_nlu/skills/notes.py`
- Test: `tests/test_skills_calendar.py`, `tests/test_skills_notes.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `extract_time` (Task 3).
- Produces: `calendar.add(storage, text, now) -> str`, `calendar.query(storage, text, now) -> str`; `notes.add(storage, text, now, embedder=None) -> str`, `notes.list_recent(storage, now) -> str`, `notes.search(storage, text, embedder=None) -> str`.

`embedder` is `Callable[[str], bytes] | None`; Task 13 supplies a real one. When `None`, `notes.search` falls back to substring matching so this task is testable standalone.

- [ ] **Step 1: Write the failing tests**

`tests/test_skills_calendar.py`:

```python
from datetime import datetime

import pytest

from jarvis_nlu.skills import calendar
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_event_strips_lead_in_and_time(store):
    calendar.add(store, "add dentist appointment today at 4 pm to my calendar", NOW)
    events = store.events_on(NOW.date())
    assert len(events) == 1
    assert events[0].title == "dentist appointment"
    assert events[0].starts_at == datetime(2026, 8, 24, 16, 0)


def test_add_event_without_time_asks_instead_of_saving(store):
    reply = calendar.add(store, "put lunch with Sam on my calendar", NOW)
    assert store.events_on(NOW.date()) == []
    assert "when" in reply.lower()


def test_query_reports_clear_day(store):
    assert "clear" in calendar.query(store, "what is on my calendar today", NOW).lower()


def test_query_lists_todays_events_in_order(store):
    calendar.add(store, "add standup today at 9:30 am to my calendar", NOW)
    calendar.add(store, "add dentist today at 4 pm to my calendar", NOW)
    reply = calendar.query(store, "what is on my calendar today", NOW)
    assert reply.index("standup") < reply.index("dentist")


def test_query_respects_tomorrow(store):
    calendar.add(store, "add dentist tomorrow at 4 pm to my calendar", NOW)
    assert "dentist" in calendar.query(store, "what's on my calendar tomorrow", NOW)
    assert "clear" in calendar.query(store, "what's on my calendar today", NOW).lower()
```

`tests/test_skills_notes.py`:

```python
from datetime import datetime

import pytest

from jarvis_nlu.skills import notes
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def test_add_strips_the_lead_in(store):
    notes.add(store, "take a note buy milk", NOW)
    assert store.list_notes()[0].text == "buy milk"


@pytest.mark.parametrize("phrasing", [
    "note: the wifi password is hunter2",
    "make a note the wifi password is hunter2",
    "remember that the wifi password is hunter2",
])
def test_add_handles_several_lead_ins(store, phrasing):
    notes.add(store, phrasing, NOW)
    assert store.list_notes()[0].text == "the wifi password is hunter2"


def test_list_recent_says_so_when_empty(store):
    assert "no" in notes.list_recent(store, NOW).lower()


def test_list_recent_returns_newest_first(store):
    notes.add(store, "note first thing", NOW)
    notes.add(store, "note second thing", NOW)
    reply = notes.list_recent(store, NOW)
    assert reply.index("second thing") < reply.index("first thing")


def test_search_falls_back_to_substring_without_an_embedder(store):
    notes.add(store, "note the wifi password is hunter2", NOW)
    notes.add(store, "note buy oat milk", NOW)
    reply = notes.search(store, "what did I note about the wifi")
    assert "hunter2" in reply
    assert "oat milk" not in reply


def test_search_reports_no_match(store):
    notes.add(store, "note buy milk", NOW)
    assert "couldn't find" in notes.search(store, "what did I note about taxes").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skills_calendar.py tests/test_skills_notes.py -v`
Expected: FAIL — `ImportError: cannot import name 'calendar'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/skills/calendar.py`:

```python
from __future__ import annotations

import re
from datetime import datetime, timedelta

from jarvis_nlu.slots import extract_time
from jarvis_nlu.storage import Storage

_ADD_LEAD_IN = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?(?:add|put|schedule|book|create|stick)\s+", re.I)
_ADD_TRAILER = re.compile(
    r"\s*(?:to|on|in|into)\s+(?:my\s+|the\s+)?calendar\s*$", re.I)


def _pretty(when: datetime) -> str:
    try:
        return when.strftime("%#I:%M %p")
    except ValueError:
        return when.strftime("%-I:%M %p")


def add(storage: Storage, text: str, now: datetime) -> str:
    slot = extract_time(text, now)
    if slot is None:
        return "Sure thing — when's that happenin', sugar?"
    title = _ADD_TRAILER.sub("", _ADD_LEAD_IN.sub("", slot.residual)).strip(" ,.")
    if not title:
        return "What should I call that one?"
    storage.add_event(title, slot.when)
    return f"Got it — {title} at {_pretty(slot.when)} on {slot.when.strftime('%A')}."


def query(storage: Storage, text: str, now: datetime) -> str:
    """Which day the user means. `extract_time` resolves bare 'tomorrow' to a
    daypart default, which is fine -- we only use the date part here."""
    lowered = text.lower()
    if "tomorrow" in lowered:
        day = (now + timedelta(days=1)).date()
        label = "Tomorrow"
    else:
        slot = extract_time(text, now)
        if slot is not None and "today" not in lowered:
            day, label = slot.when.date(), slot.when.strftime("%A")
        else:
            day, label = now.date(), "Today"

    events = storage.events_on(day)
    if not events:
        return f"{label}'s lookin' clear, sugar."
    listed = "; ".join(f"{e.title} at {_pretty(e.starts_at)}" for e in events)
    return f"{label} you've got: {listed}."
```

`jarvis_nlu/skills/notes.py`:

```python
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

Embedder = Callable[[str], bytes]


def add(storage: Storage, text: str, now: datetime,
        embedder: Embedder | None = None) -> str:
    body = _ADD_LEAD_IN.sub("", text).strip(" ,.")
    if not body:
        return "What would you like me to note down, sugar?"
    storage.add_note(body, embedder(body) if embedder else None)
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
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    hits = [n for n in storage.list_notes(limit=100)
            if any(t in n.text.lower() for t in terms)]
    if not hits:
        return "I couldn't find a note about that, sugar."
    return "You noted: " + "; ".join(n.text for n in hits[:3]) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skills_calendar.py tests/test_skills_notes.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: calendar and note skills"
```

---

### Task 8: Classifier protocol, thresholds, and fake

**Files:**
- Create: `jarvis_nlu/model.py`
- Test: `tests/test_model_protocol.py`

**Interfaces:**
- Consumes: `Intent` (Task 1).
- Produces: `Classifier` protocol with `classify(text: str) -> tuple[str, float]`; `Thresholds` frozen dataclass with `defer: float`, `confirm: float`, and `Thresholds.from_manifest(path: Path) -> Thresholds`; `FakeClassifier(scripted: dict[str, tuple[str, float]], default=(Intent.OUT_OF_SCOPE.value, 0.1))` for tests. `OnnxClassifier` arrives in Task 13.

This task is what lets Tasks 9–12 be built and tested before any training runs.

- [ ] **Step 1: Write the failing test**

`tests/test_model_protocol.py`:

```python
import json

import pytest

from jarvis_nlu.model import FakeClassifier, Thresholds


def test_fake_returns_scripted_result():
    fake = FakeClassifier({"hello there": ("greeting", 0.99)})
    assert fake.classify("hello there") == ("greeting", 0.99)


def test_fake_falls_back_to_out_of_scope():
    fake = FakeClassifier({})
    intent, confidence = fake.classify("anything at all")
    assert intent == "out_of_scope"
    assert confidence < 0.5


def test_fake_matching_is_case_and_space_insensitive():
    fake = FakeClassifier({"hello there": ("greeting", 0.99)})
    assert fake.classify("  Hello There  ")[0] == "greeting"


def test_thresholds_load_from_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"thresholds": {"defer": 0.62, "confirm": 0.85}}))
    thresholds = Thresholds.from_manifest(manifest)
    assert thresholds.defer == 0.62
    assert thresholds.confirm == 0.85


def test_missing_manifest_raises_with_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="training/train.py"):
        Thresholds.from_manifest(tmp_path / "absent.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.model'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/model.py`:

```python
"""Classifier boundary.

The router depends on the `Classifier` protocol, never on onnxruntime, so the
whole assistant is testable with `FakeClassifier` and fast unit tests before
any model is trained."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from jarvis_nlu.intents import Intent


@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str) -> tuple[str, float]:
        """Return (intent_value, calibrated_confidence)."""
        ...


@dataclass(frozen=True)
class Thresholds:
    defer: float
    confirm: float

    @classmethod
    def from_manifest(cls, path: Path) -> "Thresholds":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model manifest at {path}. Build the model first:\n"
                f"  python training/generate.py && python training/train.py "
                f"&& python training/export_onnx.py")
        raw = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
        return cls(defer=float(raw["defer"]), confirm=float(raw["confirm"]))


class FakeClassifier:
    """Test double. Scripted exact matches, out_of_scope for everything else."""

    def __init__(self, scripted: dict[str, tuple[str, float]],
                 default: tuple[str, float] = (Intent.OUT_OF_SCOPE.value, 0.1)):
        self._scripted = {k.strip().lower(): v for k, v in scripted.items()}
        self._default = default

    def classify(self, text: str) -> tuple[str, float]:
        return self._scripted.get(text.strip().lower(), self._default)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_protocol.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: classifier protocol, thresholds, and test fake"
```

---

### Task 9: Router

**Files:**
- Create: `jarvis_nlu/router.py`
- Modify: `jarvis_nlu/__init__.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `Result` frozen dataclass (`handled: bool`, `reply: str | None`, `intent: str`, `confidence: float`, `slots: dict`, `needs_confirmation: bool = False`); `Assistant(config, storage, classifier, thresholds, pool=None, embedder=None)` with `handle(text: str, now: datetime | None = None) -> Result` and attribute `last_reply: str | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_router.py`:

```python
from datetime import datetime

import pytest

from jarvis_nlu.config import Config
from jarvis_nlu.model import FakeClassifier, Thresholds
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)
THRESHOLDS = Thresholds(defer=0.6, confirm=0.85)


def build(scripted, tmp_path):
    storage = Storage(tmp_path / "t.db")
    return Assistant(config=Config(), storage=storage,
                     classifier=FakeClassifier(scripted), thresholds=THRESHOLDS)


def test_low_confidence_defers_to_the_llm(tmp_path):
    a = build({"who won the world cup": ("out_of_scope", 0.2)}, tmp_path)
    result = a.handle("who won the world cup", NOW)
    assert result.handled is False
    assert result.reply is None


def test_out_of_scope_defers_even_when_confident(tmp_path):
    a = build({"explain recursion": ("out_of_scope", 0.99)}, tmp_path)
    assert a.handle("explain recursion", NOW).handled is False


def test_confident_chore_is_handled_locally(tmp_path):
    a = build({"remind me to call mom at 6 pm": ("add_reminder", 0.97)}, tmp_path)
    result = a.handle("remind me to call mom at 6 pm", NOW)
    assert result.handled is True
    assert "call mom" in result.reply
    assert result.intent == "add_reminder"


def test_calendar_query_is_not_shadowed_by_event_add(tmp_path):
    """The confirmed regex bug. A classifier cannot shadow, but assert it."""
    a = build({
        "add dentist today at 4 pm to my calendar": ("add_calendar_event", 0.96),
        "what is on my calendar today": ("query_calendar", 0.95),
    }, tmp_path)
    a.handle("add dentist today at 4 pm to my calendar", NOW)
    result = a.handle("what is on my calendar today", NOW)
    assert result.intent == "query_calendar"
    assert "dentist" in result.reply


def test_destructive_intent_below_confirm_threshold_asks_first(tmp_path):
    a = build({"drop the mom thing": ("cancel_reminder", 0.7)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    result = a.handle("drop the mom thing", NOW)
    assert result.needs_confirmation is True
    assert result.handled is True
    assert len(a.storage.list_reminders()) == 1  # nothing destroyed yet


def test_confirming_executes_the_pending_action(tmp_path):
    a = build({"drop the mom reminder": ("cancel_reminder", 0.7),
               "yes": ("affirm", 0.99)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    a.handle("drop the mom reminder", NOW)
    a.handle("yes", NOW)
    assert a.storage.list_reminders() == []


def test_denying_abandons_the_pending_action(tmp_path):
    a = build({"drop the mom reminder": ("cancel_reminder", 0.7),
               "no": ("deny", 0.99)}, tmp_path)
    a.storage.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    a.handle("drop the mom reminder", NOW)
    a.handle("no", NOW)
    assert len(a.storage.list_reminders()) == 1


def test_repeat_last_replays_the_previous_reply(tmp_path):
    a = build({"what time is it": ("get_time", 0.99),
               "say that again": ("repeat_last", 0.99)}, tmp_path)
    first = a.handle("what time is it", NOW).reply
    assert a.handle("say that again", NOW).reply == first


def test_smalltalk_is_handled_with_zero_deferral(tmp_path):
    a = build({"hey there": ("greeting", 0.98)}, tmp_path)
    result = a.handle("hey there", NOW)
    assert result.handled is True
    assert result.reply


def test_empty_input_is_not_handled(tmp_path):
    a = build({}, tmp_path)
    assert a.handle("   ", NOW).handled is False


def test_skill_exception_returns_an_apology_not_a_crash(tmp_path, monkeypatch):
    from jarvis_nlu.skills import clock
    monkeypatch.setattr(clock, "get_time",
                        lambda now: (_ for _ in ()).throw(RuntimeError("boom")))
    a = build({"what time is it": ("get_time", 0.99)}, tmp_path)
    result = a.handle("what time is it", NOW)
    assert result.handled is True
    assert result.reply  # a spoken apology, not an exception
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.router'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/router.py`:

```python
"""Intent dispatch. Replaces the ordered-regex chain whose earlier patterns
shadowed later ones -- there is exactly one scored decision here, so no branch
can silently swallow another."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from jarvis_nlu.config import Config
from jarvis_nlu.intents import DESTRUCTIVE_INTENTS, Intent
from jarvis_nlu.model import Classifier, Thresholds
from jarvis_nlu.responses import ResponsePool
from jarvis_nlu.skills import calendar, clock, notes, reminders, smalltalk
from jarvis_nlu.storage import Storage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    handled: bool
    reply: str | None
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    needs_confirmation: bool = False


DEFERRED = Result(handled=False, reply=None,
                  intent=Intent.OUT_OF_SCOPE.value, confidence=0.0)


class Assistant:
    def __init__(self, config: Config, storage: Storage, classifier: Classifier,
                 thresholds: Thresholds, pool: ResponsePool | None = None,
                 embedder: Callable[[str], bytes] | None = None):
        self.config = config
        self.storage = storage
        self.classifier = classifier
        self.thresholds = thresholds
        self.pool = pool or ResponsePool()
        self.embedder = embedder
        self.last_reply: str | None = None
        # Held for the duration of a turn so the proactive scheduler never
        # speaks over the user mid-exchange.
        self.turn_lock = threading.Lock()
        self._pending: tuple[str, str] | None = None  # (intent_value, original text)

    def handle(self, text: str, now: datetime | None = None) -> Result:
        now = now or datetime.now()
        text = (text or "").strip()
        if not text:
            return DEFERRED

        with self.turn_lock:
            intent_value, confidence = self.classifier.classify(text)

            # Deferral is the designed path, not an error.
            if intent_value == Intent.OUT_OF_SCOPE.value or confidence < self.thresholds.defer:
                return Result(handled=False, reply=None,
                              intent=intent_value, confidence=confidence)

            if self._pending and intent_value in (Intent.AFFIRM.value, Intent.DENY.value):
                return self._resolve_pending(intent_value, confidence, now)

            # A destructive intent we are merely probable about asks first.
            if (intent_value in {i.value for i in DESTRUCTIVE_INTENTS}
                    and confidence < self.thresholds.confirm):
                self._pending = (intent_value, text)
                return self._reply(f"Just to be sure, sugar — you want me to {text}?",
                                   intent_value, confidence, needs_confirmation=True)

            return self._dispatch(intent_value, text, confidence, now)

    def _resolve_pending(self, answer: str, confidence: float, now: datetime) -> Result:
        pending_intent, pending_text = self._pending
        self._pending = None
        if answer == Intent.DENY.value:
            return self._reply(self.pool.pick("deny"), answer, confidence)
        return self._dispatch(pending_intent, pending_text, confidence, now)

    def _reply(self, text: str, intent_value: str, confidence: float,
               needs_confirmation: bool = False, slots: dict | None = None) -> Result:
        self.last_reply = text
        return Result(handled=True, reply=text, intent=intent_value,
                      confidence=confidence, slots=slots or {},
                      needs_confirmation=needs_confirmation)

    def _dispatch(self, intent_value: str, text: str, confidence: float,
                  now: datetime) -> Result:
        try:
            reply = self._run_skill(intent_value, text, now)
        except Exception:
            # A skill fault must never kill the assistant.
            log.exception("skill %s failed on %r", intent_value, text)
            return self._reply(self.pool.pick("unknown_error"), intent_value, confidence)
        return self._reply(reply, intent_value, confidence)

    def _run_skill(self, intent_value: str, text: str, now: datetime) -> str:
        store = self.storage
        if intent_value == Intent.ADD_REMINDER.value:
            return reminders.add(store, text, now)
        if intent_value == Intent.LIST_REMINDERS.value:
            return reminders.list_pending(store, now)
        if intent_value == Intent.CANCEL_REMINDER.value:
            return reminders.cancel(store, text, now)
        if intent_value == Intent.ADD_CALENDAR_EVENT.value:
            return calendar.add(store, text, now)
        if intent_value == Intent.QUERY_CALENDAR.value:
            return calendar.query(store, text, now)
        if intent_value == Intent.ADD_NOTE.value:
            return notes.add(store, text, now, self.embedder)
        if intent_value == Intent.LIST_NOTES.value:
            return notes.list_recent(store, now)
        if intent_value == Intent.SEARCH_NOTES.value:
            return notes.search(store, text, self.embedder)
        if intent_value == Intent.GET_TIME.value:
            return clock.get_time(now)
        if intent_value == Intent.GET_DATE.value:
            return clock.get_date(now)
        if intent_value == Intent.REPEAT_LAST.value:
            return self.last_reply or "I haven't said anything yet, sugar."
        # Small talk, affirm/deny with nothing pending, and sleep.
        return smalltalk.respond(intent_value, text, self.pool)
```

Update `jarvis_nlu/__init__.py`:

```python
"""Local-first NLU for the Jarvis / Miss Minutes voice assistant.

This package makes no network calls.
"""
from jarvis_nlu.config import Config
from jarvis_nlu.intents import Intent
from jarvis_nlu.model import Classifier, Thresholds
from jarvis_nlu.router import Assistant, Result
from jarvis_nlu.storage import Storage

__all__ = ["Assistant", "Classifier", "Config", "Intent", "Result",
           "Storage", "Thresholds"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all tests from Tasks 1–9 pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: intent router with confirmation flow and skill fault isolation"
```

---

### Task 10: Proactive scheduler

**Files:**
- Create: `jarvis_nlu/proactive.py`
- Test: `tests/test_proactive.py`

**Interfaces:**
- Consumes: `Storage` (Task 2), `Assistant` (Task 9).
- Produces: `Scheduler(storage, speak, turn_lock, daily_brief_at="08:30", tick_seconds=20, clock=datetime.now)` with `tick(now) -> None`, `start() -> None`, `stop() -> None`.

`tick(now)` is public and synchronous precisely so tests drive it on a fake clock without threads.

- [ ] **Step 1: Write the failing tests**

`tests/test_proactive.py`:

```python
import threading
from datetime import datetime

import pytest

from jarvis_nlu.proactive import Scheduler
from jarvis_nlu.storage import Storage


@pytest.fixture
def store(tmp_path):
    s = Storage(tmp_path / "t.db")
    yield s
    s.close()


def build(store, spoken, lock=None):
    return Scheduler(storage=store, speak=spoken.append,
                     turn_lock=lock or threading.Lock(),
                     daily_brief_at=None)


def test_due_reminder_is_spoken_once(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    scheduler = build(store, spoken)
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    scheduler.tick(datetime(2026, 8, 24, 18, 2))
    assert len(spoken) == 1
    assert "call mom" in spoken[0]


def test_future_reminder_is_not_spoken(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 23, 0))
    build(store, spoken).tick(datetime(2026, 8, 24, 18, 0))
    assert spoken == []


def test_delivery_is_recorded_only_after_speaking(store):
    """A crash mid-announcement must replay the reminder, not swallow it."""
    reminder_id = store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))

    def exploding_speak(_message):
        raise RuntimeError("tts died")

    scheduler = Scheduler(storage=store, speak=exploding_speak,
                          turn_lock=threading.Lock(), daily_brief_at=None)
    with pytest.raises(RuntimeError):
        scheduler.tick(datetime(2026, 8, 24, 18, 1))

    assert store.list_reminders()[0].delivered_at is None
    assert len(store.due_reminders(datetime(2026, 8, 24, 18, 2))) == 1


def test_tick_skips_while_a_turn_is_in_progress(store):
    spoken = []
    store.add_reminder("call mom", datetime(2026, 8, 24, 18, 0))
    lock = threading.Lock()
    scheduler = build(store, spoken, lock)
    lock.acquire()                                  # user is mid-turn
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    assert spoken == []
    lock.release()
    scheduler.tick(datetime(2026, 8, 24, 18, 1))
    assert len(spoken) == 1


def test_daily_brief_fires_once_per_day(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    scheduler.tick(datetime(2026, 8, 24, 9, 0))
    assert len(spoken) == 1
    assert "dentist" in spoken[0]


def test_daily_brief_stays_quiet_with_nothing_to_report(store):
    spoken = []
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 8, 31))
    assert spoken == []


def test_daily_brief_does_not_fire_before_its_time(store):
    spoken = []
    store.add_event("dentist", datetime(2026, 8, 24, 16, 0))
    scheduler = Scheduler(storage=store, speak=spoken.append,
                          turn_lock=threading.Lock(), daily_brief_at="08:30")
    scheduler.tick(datetime(2026, 8, 24, 7, 0))
    assert spoken == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proactive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_nlu.proactive'`

- [ ] **Step 3: Write the implementation**

`jarvis_nlu/proactive.py`:

```python
"""The only unprompted speech in the system: due reminders and one daily brief.

`tick` is synchronous and takes `now`, so the whole policy is testable on a
fake clock without spawning threads."""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable

from jarvis_nlu.storage import Storage

BRIEF_KEY = "last_brief_date"


class Scheduler:
    def __init__(self, storage: Storage, speak: Callable[[str], None],
                 turn_lock: threading.Lock, daily_brief_at: str | None = "08:30",
                 tick_seconds: int = 20):
        self.storage = storage
        self.speak = speak
        self.turn_lock = turn_lock
        self.daily_brief_at = daily_brief_at
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- policy ----------

    def tick(self, now: datetime) -> None:
        # Never talk over the user: skip this tick rather than queue behind them.
        if not self.turn_lock.acquire(blocking=False):
            return
        try:
            self._speak_due_reminders(now)
            self._maybe_daily_brief(now)
        finally:
            self.turn_lock.release()

    def _speak_due_reminders(self, now: datetime) -> None:
        for reminder in self.storage.due_reminders(now):
            self.speak(f"Heads up, sugar — you asked me to remind you to {reminder.text}.")
            # Recorded only after speak() returns: a crash mid-announcement
            # replays the reminder rather than silently eating it.
            self.storage.mark_delivered(reminder.id, now)

    def _maybe_daily_brief(self, now: datetime) -> None:
        if not self.daily_brief_at:
            return
        hour, minute = (int(p) for p in self.daily_brief_at.split(":"))
        if (now.hour, now.minute) < (hour, minute):
            return
        today = now.date().isoformat()
        if self.storage.get_meta(BRIEF_KEY) == today:
            return

        events = self.storage.events_on(now.date())
        pending = [r for r in self.storage.list_reminders() if r.due_at]
        if not events and not pending:
            self.storage.set_meta(BRIEF_KEY, today)  # nothing to say, but don't retry all day
            return

        parts = []
        if events:
            listed = "; ".join(f"{e.title} at {e.starts_at.strftime('%I:%M %p').lstrip('0')}"
                               for e in events)
            parts.append(f"on your calendar: {listed}")
        if pending:
            parts.append(f"and {len(pending)} reminder{'s' if len(pending) != 1 else ''} pendin'")
        self.speak("Mornin', sugar. Here's your day — " + ", ".join(parts) + ".")
        self.storage.set_meta(BRIEF_KEY, today)

    # ---------- thread ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="jarvis-proactive")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_seconds):
            try:
                self.tick(datetime.now())
            except Exception:  # a bad tick must not kill the thread
                import logging
                logging.getLogger(__name__).exception("proactive tick failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.tick_seconds)
            self._thread = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proactive.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: proactive scheduler with crash-safe reminder delivery"
```

---

### Task 11: Training dataset generation

**Files:**
- Create: `training/templates/*.yaml` (20 files), `training/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `ALL_INTENTS` (Task 1).
- Produces: `generate.build_dataset(templates_dir, learning_log=None, seed=0) -> list[Example]`; `Example` frozen dataclass (`text: str`, `intent: str`, `template_id: str`, `weight: float`); `generate.add_stt_noise(text, rng) -> str`; CLI writing `training/data/dataset.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:

```python
from pathlib import Path

import pytest

from jarvis_nlu.intents import ALL_INTENTS
from training.generate import Example, add_stt_noise, build_dataset

TEMPLATES = Path(__file__).parent.parent / "training" / "templates"


def test_every_intent_has_a_template_file():
    for intent in ALL_INTENTS:
        assert (TEMPLATES / f"{intent.value}.yaml").exists(), f"missing {intent.value}.yaml"


def test_dataset_covers_every_intent():
    dataset = build_dataset(TEMPLATES, seed=0)
    covered = {e.intent for e in dataset}
    assert covered == {i.value for i in ALL_INTENTS}


def test_dataset_is_large_enough_to_train_on():
    assert len(build_dataset(TEMPLATES, seed=0)) >= 4000


def test_every_intent_has_at_least_a_hundred_examples():
    dataset = build_dataset(TEMPLATES, seed=0)
    counts = {}
    for example in dataset:
        counts[example.intent] = counts.get(example.intent, 0) + 1
    thin = {k: v for k, v in counts.items() if v < 100}
    assert not thin, f"under-represented intents: {thin}"


def test_out_of_scope_is_well_represented():
    """The negative class is what protects the LLM fallback path."""
    dataset = build_dataset(TEMPLATES, seed=0)
    out_of_scope = [e for e in dataset if e.intent == "out_of_scope"]
    assert len(out_of_scope) >= 400


def test_generation_is_deterministic_for_a_seed():
    first = [e.text for e in build_dataset(TEMPLATES, seed=7)]
    second = [e.text for e in build_dataset(TEMPLATES, seed=7)]
    assert first == second


def test_every_example_carries_a_template_id():
    """Template ids drive grouped splitting so paraphrases can't straddle
    train and test and inflate the score."""
    for example in build_dataset(TEMPLATES, seed=0):
        assert example.template_id


def test_stt_noise_changes_text_but_keeps_it_recognisable():
    import random
    rng = random.Random(0)
    noisy = [add_stt_noise("remind me to call mom at six", rng) for _ in range(20)]
    assert any(n != "remind me to call mom at six" for n in noisy)
    assert all(len(n) > 5 for n in noisy)


def test_learning_log_examples_are_merged_with_higher_weight(tmp_path):
    log = tmp_path / "learning_log.jsonl"
    log.write_text('{"transcript": "ping me about the thing at four", '
                   '"intent": "add_reminder"}\n')
    dataset = build_dataset(TEMPLATES, learning_log=log, seed=0)
    merged = [e for e in dataset if e.text == "ping me about the thing at four"]
    assert len(merged) == 1
    assert merged[0].weight > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Write the templates and generator**

Create `training/__init__.py` (empty) so `training` is importable.

Each `training/templates/<intent>.yaml` has this shape. Write **at least 12 patterns per intent**, and enough slot values that expansion yields 100+ examples per intent.

`training/templates/add_reminder.yaml`:

```yaml
intent: add_reminder
slots:
  task:
    - call mom
    - pay the rent
    - take the bread out
    - email Priya
    - book the dentist
    - water the plants
    - submit the timesheet
    - pick up the parcel
  time:
    - at 6
    - at 6 pm
    - tomorrow at 9
    - tonight at 8
    - in 20 minutes
    - on friday at 4
    - tomorrow morning
    - in two hours
patterns:
  - remind me to {task} {time}
  - remind me {time} to {task}
  - can you remind me to {task} {time}
  - set a reminder to {task} {time}
  - set a reminder for {task} {time}
  - don't let me forget to {task} {time}
  - nudge me to {task} {time}
  - remind me about {task} {time}
  - i need a reminder to {task} {time}
  - make sure i {task} {time}
  - give me a nudge to {task} {time}
  - remember to remind me to {task} {time}
```

`training/templates/query_calendar.yaml` — note these patterns must **not** overlap `add_calendar_event`; that overlap is the confirmed shadowing bug:

```yaml
intent: query_calendar
slots:
  day:
    - today
    - tomorrow
    - on friday
    - this afternoon
    - ""
patterns:
  - what is on my calendar {day}
  - what's on my calendar {day}
  - whats my schedule {day}
  - what does my day look like {day}
  - do i have anything {day}
  - am i free {day}
  - what have i got on {day}
  - read me my calendar {day}
  - what's my day looking like {day}
  - anything on the calendar {day}
  - run me through {day}
  - what am i doing {day}
```

`training/templates/out_of_scope.yaml` — the negative class. Use literal phrases, not slot expansion, and write **at least 150 distinct patterns** spanning general knowledge, coding questions, weather, maths, opinions, and open-ended chat:

```yaml
intent: out_of_scope
slots: {}
patterns:
  - who won the world cup
  - explain recursion to me
  - what's the weather in tokyo
  - write me a python function that sorts a list
  - what's the capital of australia
  - how do i fix a merge conflict
  - what does this error mean
  - tell me a story about a dragon
  - what's 17 times 23
  - who wrote pride and prejudice
  - summarise the news for me
  - what's the difference between tcp and udp
  - how far away is the moon
  - give me a recipe for carbonara
  - what should i name my cat
  # ... continue to at least 150
```

`training/generate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate.py -v && python training/generate.py`
Expected: 9 passed; generator reports ≥4000 examples. If an intent is thin, add patterns or slot values to its YAML — do not lower the threshold.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: training data generation with STT-noise augmentation"
```

---

### Task 12: Train, calibrate, and export

**Files:**
- Create: `training/train.py`, `training/export_onnx.py`
- Test: `tests/test_train_units.py`

**Interfaces:**
- Consumes: `build_dataset` (Task 11), `ALL_INTENTS` (Task 1).
- Produces: `train.grouped_split(examples, seed=0) -> tuple[list, list, list]`; `train.fit_temperature(logits, labels) -> float`; `train.main()` writing `models/intent.pt` and `models/manifest.json`; `export_onnx.main()` writing `models/intent.onnx` (int8). Manifest keys: `labels` (ordered list), `thresholds.defer`, `thresholds.confirm`, `base_model`, `max_length`, `trained_at`.

- [ ] **Step 1: Write the failing unit tests**

Only the pure logic is unit-tested; the fit itself is validated by Task 14's gates.

`tests/test_train_units.py`:

```python
import numpy as np
import pytest

from training.generate import Example
from training.train import fit_temperature, grouped_split


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


def test_temperature_is_positive_for_well_calibrated_input():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=(200, 3))
    labels = logits.argmax(axis=1)
    assert fit_temperature(logits, labels) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train_units.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: Write the implementation**

`training/train.py`:

```python
"""Fine-tune MiniLM for intent classification, then calibrate its confidence.

The encoder is NOT frozen -- this is a genuine fine-tune, which is what lets
the model generalise to phrasings absent from the templates."""
from __future__ import annotations

import json
import random
from collections import defaultdict
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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = IntentModel(len(LABELS)).to(device)
    loader = DataLoader(_Rows(train_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

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
```

`training/export_onnx.py`:

```python
"""Export the fine-tuned model to int8 ONNX so the runtime needs onnxruntime
only -- no torch in the deployed package."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from training.train import BASE_MODEL, MAX_LENGTH, IntentModel

ROOT = Path(__file__).parent.parent


def main() -> None:
    models_dir = ROOT / "models"
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))

    model = IntentModel(len(manifest["labels"]))
    model.load_state_dict(torch.load(models_dir / "intent.pt", map_location="cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(models_dir / "tokenizer")

    dummy = tokenizer("hello", return_tensors="pt", padding="max_length",
                      truncation=True, max_length=MAX_LENGTH)
    fp32 = models_dir / "intent.fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17)

    quantize_dynamic(fp32, models_dir / "intent.onnx", weight_type=QuantType.QInt8)
    fp32.unlink()
    size_mb = (models_dir / "intent.onnx").stat().st_size / 1e6
    print(f"exported models/intent.onnx ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, then train**

```bash
python -m pytest tests/test_train_units.py -v
python training/generate.py && python training/train.py && python training/export_onnx.py
```
Expected: 6 unit tests pass. Training takes roughly 5–15 minutes on CPU. `models/intent.onnx` should land near 23MB.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: MiniLM fine-tuning, temperature calibration, ONNX export"
```

`models/` is gitignored — the artifacts are rebuilt, not committed.

---

### Task 13: ONNX classifier and note embeddings

**Files:**
- Modify: `jarvis_nlu/model.py`
- Create: `jarvis_nlu/embeddings.py`
- Test: `tests/test_onnx_classifier.py`, `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `models/intent.onnx`, `models/manifest.json`, `models/tokenizer/` (Task 12).
- Produces: `OnnxClassifier(model_dir: Path)` implementing `Classifier`; `embeddings.embed(text: str) -> bytes`, `embeddings.rank_notes(storage, query, embedder) -> list[Note]`.

Tests here require the trained artifacts and are skipped when absent, so the suite still runs on a clean checkout.

- [ ] **Step 1: Write the failing tests**

`tests/test_onnx_classifier.py`:

```python
import time
from pathlib import Path

import pytest

from jarvis_nlu.model import Classifier, OnnxClassifier

MODELS = Path(__file__).parent.parent / "models"
pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(),
    reason="model not built; run training/train.py and training/export_onnx.py")


@pytest.fixture(scope="module")
def classifier():
    return OnnxClassifier(MODELS)


def test_satisfies_the_classifier_protocol(classifier):
    assert isinstance(classifier, Classifier)


def test_returns_a_known_label_and_a_probability(classifier):
    from jarvis_nlu.intents import ALL_INTENTS
    intent, confidence = classifier.classify("remind me to call mom at six")
    assert intent in {i.value for i in ALL_INTENTS}
    assert 0.0 <= confidence <= 1.0


def test_p95_latency_under_25ms(classifier):
    classifier.classify("warm up")
    samples = []
    for _ in range(50):
        start = time.perf_counter()
        classifier.classify("what is on my calendar today")
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    assert samples[int(len(samples) * 0.95)] < 25.0


def test_missing_model_directory_raises_with_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx.py"):
        OnnxClassifier(tmp_path)
```

`tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_onnx_classifier.py -v`
Expected: FAIL — `ImportError: cannot import name 'OnnxClassifier'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis_nlu/model.py`:

```python
class OnnxClassifier:
    """Runtime classifier. onnxruntime only -- torch is a training-time dep."""

    def __init__(self, model_dir: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_path = model_dir / "intent.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No model at {onnx_path}. Build it:\n"
                f"  python training/generate.py && python training/train.py "
                f"&& python training/export_onnx.py")

        self._np = np
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self._labels: list[str] = manifest["labels"]
        self._temperature: float = float(manifest.get("temperature", 1.0))
        self._max_length: int = int(manifest["max_length"])
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer" / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=self._max_length)
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])

    def classify(self, text: str) -> tuple[str, float]:
        np = self._np
        encoded = self._tokenizer.encode(text)
        logits = self._session.run(None, {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        })[0][0]
        scaled = logits / self._temperature
        exponentials = np.exp(scaled - scaled.max())
        probabilities = exponentials / exponentials.sum()
        index = int(probabilities.argmax())
        return self._labels[index], float(probabilities[index])
```

Add `tokenizers>=0.20` to the runtime dependencies in `pyproject.toml`.

`jarvis_nlu/embeddings.py`:

```python
"""Semantic note search. Reuses the same MiniLM encoder already resident for
classification, so retrieval costs no extra model -- embeddings are a BLOB
column, not a vector database."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from jarvis_nlu.storage import Note, Storage

SIMILARITY_FLOOR = 0.35
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
```

`export_onnx.py` must also export the bare encoder. Add before `main()`'s final print:

```python
    class _Encoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state

    torch.onnx.export(
        _Encoder(model.encoder), (dummy["input_ids"], dummy["attention_mask"]),
        models_dir / "encoder.onnx",
        input_names=["input_ids", "attention_mask"], output_names=["hidden"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "hidden": {0: "batch", 1: "seq"}},
        opset_version=17)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python training/export_onnx.py
python -m pytest tests/test_onnx_classifier.py tests/test_embeddings.py -v
```
Expected: 6 passed. If p95 latency exceeds 25ms, lower `MAX_LENGTH` to 32 and re-export.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: ONNX classifier and semantic note search"
```

---

### Task 14: Evaluation gates and golden regression suite

**Files:**
- Create: `training/evaluate.py`, `tests/test_golden.py`, `tests/golden.yaml`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `OnnxClassifier` (Task 13), `grouped_split` (Task 12), `build_dataset` (Task 11).
- Produces: `evaluate.score(classifier, examples) -> Report`; `Report` dataclass with `macro_f1: float`, `confusion: dict[tuple[str, str], float]`, `deferral_rate: float`, `leak_rate: float`, `worst_confusion() -> tuple[str, str, float]`; `evaluate.main()` printing the matrix and exiting non-zero on any failed gate.

- [ ] **Step 1: Write the failing tests**

`tests/golden.yaml` — seeded with every phrase confirmed broken on 2026-08-24:

```yaml
# Every bug fixed from here on adds a row.
- text: what is on my calendar today
  intent: query_calendar          # regression: matched add_calendar_event
- text: what's on my calendar
  intent: query_calendar
- text: remind me I'm meeting Bob tomorrow at 5 pm
  intent: add_reminder            # regression: matched greeting
- text: remind me to say hi to Alex tomorrow at 6 pm
  intent: add_reminder            # regression: matched greeting
- text: remind me to call mom at 6 pm
  intent: add_reminder            # regression: rejected for lacking a day
- text: who won the world cup
  intent: out_of_scope
- text: explain recursion to me
  intent: out_of_scope
- text: hey there
  intent: greeting
- text: how are you doing
  intent: how_are_you
- text: take a note buy milk
  intent: add_note
- text: what did I note about the wifi
  intent: search_notes
- text: read me my reminders
  intent: list_reminders
- text: cancel the mom reminder
  intent: cancel_reminder
- text: add dentist appointment tomorrow at 4 pm to my calendar
  intent: add_calendar_event
- text: what time is it
  intent: get_time
- text: what's the date
  intent: get_date
```

`tests/test_golden.py`:

```python
from pathlib import Path

import pytest
import yaml

from jarvis_nlu.model import OnnxClassifier

MODELS = Path(__file__).parent.parent / "models"
GOLDEN = yaml.safe_load((Path(__file__).parent / "golden.yaml").read_text())

pytestmark = pytest.mark.skipif(
    not (MODELS / "intent.onnx").exists(), reason="model not built")


@pytest.fixture(scope="module")
def classifier():
    return OnnxClassifier(MODELS)


@pytest.mark.parametrize("case", GOLDEN, ids=[c["text"][:40] for c in GOLDEN])
def test_golden_case(classifier, case):
    predicted, confidence = classifier.classify(case["text"])
    assert predicted == case["intent"], (
        f"{case['text']!r} -> {predicted} ({confidence:.2f}), "
        f"expected {case['intent']}")
```

`tests/test_evaluate.py`:

```python
from training.evaluate import Report, score
from training.generate import Example


class _Stub:
    def __init__(self, mapping):
        self._mapping = mapping

    def classify(self, text):
        return self._mapping.get(text, ("out_of_scope", 0.9))


def test_perfect_classifier_scores_one():
    examples = [Example("a", "greeting", "g:0"), Example("b", "get_time", "t:0")]
    report = score(_Stub({"a": ("greeting", 0.99), "b": ("get_time", 0.99)}), examples)
    assert report.macro_f1 == 1.0


def test_confusion_matrix_records_the_mistake():
    examples = [Example("a", "query_calendar", "q:0")]
    report = score(_Stub({"a": ("add_calendar_event", 0.99)}), examples)
    assert report.confusion[("query_calendar", "add_calendar_event")] == 1.0
    assert report.worst_confusion()[:2] == ("query_calendar", "add_calendar_event")


def test_deferral_rate_counts_low_confidence_in_scope():
    examples = [Example("a", "greeting", "g:0"), Example("b", "greeting", "g:1")]
    report = score(_Stub({"a": ("greeting", 0.99), "b": ("greeting", 0.2)}),
                   examples, defer_threshold=0.6)
    assert report.deferral_rate == 0.5


def test_leak_rate_counts_confident_out_of_scope_mislabels():
    examples = [Example("a", "out_of_scope", "o:0")]
    report = score(_Stub({"a": ("greeting", 0.99)}), examples, defer_threshold=0.6)
    assert report.leak_rate == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.evaluate'`

- [ ] **Step 3: Write the implementation**

`training/evaluate.py`:

```python
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


def main() -> int:
    from jarvis_nlu.model import OnnxClassifier
    from training.train import grouped_split

    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    defer_threshold = manifest["thresholds"]["defer"]
    _train, _val, test_rows = grouped_split(
        build_dataset(ROOT / "training" / "templates"))
    report = score(OnnxClassifier(ROOT / "models"), test_rows, defer_threshold)

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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and the gates**

```bash
python -m pytest tests/test_evaluate.py tests/test_golden.py -v
python training/evaluate.py
```
Expected: unit tests pass; all 16 golden cases pass; `evaluate.py` exits 0 with every gate PASS.

If a gate fails, the fix is **more or better training templates for the confused intents** — never a lowered threshold. If two intents confuse persistently, their template patterns overlap in meaning and one set needs rewording.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: evaluation gates and golden regression suite"
```

---

### Task 15: miss-minutes integration

**Files:**
- Modify: `miss-minutes/brain/assistant.py`, `miss-minutes/main.py`, `miss-minutes/requirements.txt`
- Delete: `miss-minutes/brain/tools.py`
- Create: `miss-minutes/tests/test_routing.py`
- Test: `miss-minutes/tests/test_routing.py`

**Interfaces:**
- Consumes: `Assistant`, `Result`, `Config`, `Storage`, `OnnxClassifier`, `Thresholds`, `Scheduler`.
- Produces: `brain.assistant.respond(text, nlu, now=None) -> str`; `brain.assistant.ask_llm(text) -> str` (the existing OpenRouter path, renamed from `ask`).

- [ ] **Step 1: Write the failing test**

`miss-minutes/tests/test_routing.py`:

```python
"""The test that proves the rate-limit fix: handled intents must not touch
the network."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from jarvis_nlu.config import Config
from jarvis_nlu.model import FakeClassifier, Thresholds
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage

NOW = datetime(2026, 8, 24, 9, 0)


@pytest.fixture
def nlu(tmp_path):
    storage = Storage(tmp_path / "t.db")
    classifier = FakeClassifier({
        "remind me to call mom at 6 pm": ("add_reminder", 0.97),
        "hey there": ("greeting", 0.98),
        "what time is it": ("get_time", 0.99),
        "who won the world cup": ("out_of_scope", 0.2),
    })
    yield Assistant(config=Config(), storage=storage, classifier=classifier,
                    thresholds=Thresholds(defer=0.6, confirm=0.85))
    storage.close()


@pytest.mark.parametrize("text", [
    "remind me to call mom at 6 pm", "hey there", "what time is it",
])
def test_handled_intents_make_zero_llm_calls(nlu, text, monkeypatch):
    from brain import assistant

    llm = MagicMock(return_value="should not be called")
    monkeypatch.setattr(assistant, "ask_llm", llm)

    reply = assistant.respond(text, nlu, now=NOW)
    assert reply
    llm.assert_not_called()


def test_out_of_scope_makes_exactly_one_llm_call(nlu, monkeypatch):
    from brain import assistant

    llm = MagicMock(return_value="Brazil won in 2002.")
    monkeypatch.setattr(assistant, "ask_llm", llm)

    reply = assistant.respond("who won the world cup", nlu, now=NOW)
    assert reply == "Brazil won in 2002."
    llm.assert_called_once_with("who won the world cup")


def test_llm_failure_still_returns_something_speakable(nlu, monkeypatch):
    from brain import assistant

    monkeypatch.setattr(assistant, "ask_llm",
                        MagicMock(side_effect=RuntimeError("429 rate limit")))
    reply = assistant.respond("who won the world cup", nlu, now=NOW)
    assert reply  # never propagates; the voice loop must not die
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd miss-minutes && python -m pytest tests/ -v`
Expected: FAIL — `ImportError: cannot import name 'respond'`

- [ ] **Step 3: Write the implementation**

Add to `miss-minutes/requirements.txt`:

```
-e ../RAG-Voice_model
```

In `miss-minutes/brain/assistant.py`: rename `ask` to `ask_llm`, change its import of `brain.tools` to drop the tool schemas entirely (the LLM no longer needs tools — chores are handled locally), and append:

```python
def respond(text: str, nlu, now=None) -> str:
    """Local NLU first; the network only on deferral.

    This function is the whole rate-limit fix: every intent the classifier
    handles confidently costs zero OpenRouter requests."""
    result = nlu.handle(text, now)
    if result.handled:
        return result.reply
    try:
        return ask_llm(text)
    except Exception as error:  # never let a network fault kill the voice loop
        print(f"[brain] llm failed: {error}")
        if "429" in str(error) or "rate" in str(error).lower():
            return "Sugar, I've hit my daily thinkin' limit. Try me again in a bit."
        return "My thoughts aren't reachin' me right now. Try again in a moment."
```

Because chores no longer reach the model, delete `brain/tools.py` and remove `tools=TOOL_SCHEMAS` and the `_run_tool` loop from `_complete`/`ask_llm`.

Rewrite `miss-minutes/main.py`:

```python
"""Minutes -- voice assistant main loop.
Wake word -> record -> transcribe -> local NLU (or LLM) -> speak."""
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from jarvis_nlu.config import Config
from jarvis_nlu.model import OnnxClassifier, Thresholds
from jarvis_nlu.proactive import Scheduler
from jarvis_nlu.router import Assistant
from jarvis_nlu.storage import Storage

from brain.assistant import respond
from stt.transcribe import record_utterance, transcribe
from tts.speak import speak
from wakeword.listen import wait_for_wake_word

NLU_ROOT = Path(__file__).parent.parent / "RAG-Voice_model"


def build_nlu() -> tuple[Assistant, Scheduler]:
    config = Config.load(NLU_ROOT / "jarvis.config.json")
    storage = Storage(NLU_ROOT / config.db_path)
    models = NLU_ROOT / config.model_dir
    assistant = Assistant(
        config=config, storage=storage,
        classifier=OnnxClassifier(models),
        thresholds=Thresholds.from_manifest(models / "manifest.json"))
    scheduler = Scheduler(storage=storage, speak=speak,
                          turn_lock=assistant.turn_lock,
                          daily_brief_at=config.daily_brief_at,
                          tick_seconds=config.tick_seconds)
    return assistant, scheduler


def main():
    nlu, scheduler = build_nlu()
    scheduler.start()
    speak("I'm up and listenin' whenever you need me, sugar.")
    try:
        while True:
            wait_for_wake_word()
            speak("Mm-hmm?")
            text = transcribe(record_utterance())
            if not text:
                speak("Didn't quite catch that, try me again.")
                continue
            print(f"[you] {text}")
            reply = respond(text, nlu)
            print(f"[minutes] {reply}")
            speak(reply)
    except KeyboardInterrupt:
        scheduler.stop()
        print("\nGoin' quiet.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

```bash
cd RAG-Voice_model && python -m pytest tests/ -v
cd ../miss-minutes && python -m pytest tests/ -v
```
Expected: the full suite passes in both projects.

Then a manual smoke test with the microphone: say the wake phrase, greet it, set a reminder for two minutes out, ask what's on the calendar, and confirm the reminder fires unprompted.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: route miss-minutes through local NLU before the LLM"
```

---

## Verification checklist

Before calling this done, all of the following must hold:

- [ ] `cd RAG-Voice_model && python -m pytest tests/ -v` — all pass, none skipped
- [ ] `cd miss-minutes && python -m pytest tests/ -v` — all pass
- [ ] `python training/evaluate.py` — exits 0, every gate PASS
- [ ] All 16 golden regression cases pass
- [ ] `models/intent.onnx` exists and is under 30MB
- [ ] Manual smoke test: wake word → greeting → reminder → calendar query → proactive reminder fires
- [ ] `grep -rn "requests\|httpx\|openai" RAG-Voice_model/jarvis_nlu/` returns nothing
