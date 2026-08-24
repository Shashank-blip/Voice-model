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
