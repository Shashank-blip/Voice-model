"""Task D: the review CLI.

Without this, `data/learning_log.jsonl` accumulates forever and never
becomes training data -- `training/generate.py` only merges rows that
carry a ground-truth `intent`, and nothing else in this project ever adds
one. This is the only place a human turns a raw spoken turn into a label.

Run as:

    python -m training.review

SAFETY -- the log is irreplaceable user data, and it may be being
appended to live by a running assistant while a human reviews it:

  * The rewrite is atomic. We write the full new content to a temp file
    in the SAME directory, fsync it, then `os.replace` it over the
    original in one step. If the process dies at any point before that
    replace, the original file was never touched -- there is no window
    where it is truncated or partially written.
  * Malformed lines (anything that fails `json.loads`) are preserved
    verbatim, byte-for-byte, rather than dropped. They are simply never
    offered as review candidates.
  * The file is read once at the start of a session (`ReviewSession.load`)
    and again, fresh, right before writing (`ReviewSession.save`). If the
    file changed in the meantime in a way that is a pure append (the
    assistant logged more turns while the human was reviewing), those new
    lines are merged in untouched. If it changed any other way -- an
    existing line edited or removed, e.g. by a second review session
    running concurrently -- we refuse to write at all and raise
    `ConcurrentModificationError` rather than risk silently clobbering
    data. A refusal is an acceptable outcome; silent data loss is not.

See also jarvis_nlu/turnlog.py, which writes this file, and
training/generate.py, which is the only other reader of `intent`.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Allow `python training/review.py` (script's own directory on sys.path[0],
# not the repo root) as well as `python -m training.review`, matching the
# other training/*.py entry points.
_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORTS))

from jarvis_nlu.intents import ALL_INTENTS, Intent

PROMPT = "[Enter] confirm / number to correct / s skip / q quit: "

# Weights for the likely-miss-first ordering. A correction record
# (repeat or negation, discovered from the NEXT turn) is the strongest
# available evidence of a miss -- see the plan's Task B rationale -- so it
# outranks everything else. `reprompt`/`deferred` are the router's own
# admission it wasn't sure. `skill_error` means the prediction was
# plausible enough to dispatch but broke on execution. Confidence is a
# soft tiebreaker among entries carrying none of the above, worth at most
# CONFIDENCE_WEIGHT so it can never outrank a real signal.
CORRECTION_WEIGHT = 1000.0
REPROMPT_WEIGHT = 500.0
DEFERRED_WEIGHT = 300.0
SKILL_ERROR_WEIGHT = 200.0
CONFIDENCE_WEIGHT = 100.0


class ConcurrentModificationError(RuntimeError):
    """Raised by `ReviewSession.save` when the log file changed underneath
    the session in a way that cannot be safely merged -- something other
    than a pure append (an existing line was edited, removed, or the file
    was rewritten by another process/review session). The fix is to stop
    whatever else is touching the file and re-run the review; this is
    never raised as a result of writing anything."""


@dataclass
class Candidate:
    """One unlabelled turn, ready to show a human."""
    id: str
    line_index: int
    obj: dict
    signals: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def transcript(self) -> str:
        return self.obj.get("transcript", "")

    @property
    def predicted_intent(self) -> str:
        return self.obj.get("predicted_intent", "")

    @property
    def confidence(self) -> float | None:
        return self.obj.get("confidence")


@dataclass
class SaveResult:
    written: bool
    labelled: int
    appended_by_others: int


@dataclass
class ReviewSummary:
    total_candidates: int = 0
    confirmed: int = 0
    corrected: int = 0
    skipped: int = 0
    saved: bool = False


def _signal_labels(obj: dict, corrections: list[dict]) -> list[str]:
    """Human-readable miss signals for display -- kept separate from
    `_miss_score` so the two can evolve independently (a signal can be
    worth showing without changing the sort weight, or vice versa)."""
    signals: list[str] = []
    if any(c.get("repeat_within_seconds") is not None for c in corrections):
        signals.append("repeated")
    if any(c.get("user_negation") for c in corrections):
        signals.append("negated")
    if obj.get("reprompt"):
        signals.append("reprompt")
    if obj.get("deferred"):
        signals.append("deferred")
    if obj.get("skill_error"):
        signals.append(f"skill_error={obj['skill_error']!r}")
    confidence = obj.get("confidence")
    if confidence is not None and confidence < 0.5:
        signals.append(f"low_confidence={confidence:.2f}")
    return signals


def _miss_score(obj: dict, corrections: list[dict]) -> float:
    """Higher = more likely the model got this one wrong = show first."""
    score = 0.0
    if corrections:
        score += CORRECTION_WEIGHT
    if obj.get("reprompt"):
        score += REPROMPT_WEIGHT
    if obj.get("deferred"):
        score += DEFERRED_WEIGHT
    if obj.get("skill_error"):
        score += SKILL_ERROR_WEIGHT
    confidence = obj.get("confidence")
    if confidence is not None:
        score += (1.0 - confidence) * CONFIDENCE_WEIGHT
    return score


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Write `lines` (no trailing newlines) to `path` atomically: full
    content to a temp file in the same directory, fsync'd, then a single
    `os.replace`. On any failure before the replace, the temp file is
    removed and the original is left completely untouched."""
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


class ReviewSession:
    """Loads candidates once, accumulates label decisions in memory, and
    writes them back with the safety properties described in this
    module's docstring. One session is meant to cover one run of the CLI
    from load() through a single save()."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._raw_lines: list[str] = []
        self._parsed: list[dict | None] = []
        self._labels: dict[str, str] = {}

    def load(self) -> list[Candidate]:
        self._raw_lines = []
        self._parsed = []
        self._labels = {}
        if not self.path.exists():
            return []

        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        parsed: list[dict | None] = []
        for line in raw_lines:
            if not line.strip():
                parsed.append(None)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parsed.append(None)
                continue
            parsed.append(obj if isinstance(obj, dict) else None)

        corrections_by_target: dict[str, list[dict]] = {}
        for obj in parsed:
            if obj and "correction_for" in obj:
                corrections_by_target.setdefault(obj["correction_for"], []).append(obj)

        candidates: list[Candidate] = []
        for index, obj in enumerate(parsed):
            if obj is None:
                continue
            if "correction_for" in obj:
                continue  # correction records are never candidates
            if "intent" in obj:
                continue  # already labelled
            if "predicted_intent" not in obj or "id" not in obj:
                continue  # not a recognizable turn entry
            entry_corrections = corrections_by_target.get(obj["id"], [])
            candidates.append(Candidate(
                id=obj["id"], line_index=index, obj=obj,
                signals=_signal_labels(obj, entry_corrections),
                score=_miss_score(obj, entry_corrections),
            ))

        self._raw_lines = raw_lines
        self._parsed = parsed
        # Stable sort: entries with equal score keep their original
        # (chronological) file order, so ties among "plain confident"
        # entries still show oldest-first.
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def record(self, entry_id: str, intent_value: str) -> None:
        self._labels[entry_id] = intent_value

    def save(self) -> SaveResult:
        """Re-read the file fresh, merge in any lines appended since
        load(), apply recorded labels, and write atomically. Raises
        ConcurrentModificationError (without writing anything) if the
        file changed in a way other than a pure append."""
        if not self._labels:
            return SaveResult(written=False, labelled=0, appended_by_others=0)

        if not self.path.exists():
            raise ConcurrentModificationError(
                f"{self.path} no longer exists. Refusing to write -- stop "
                "the running assistant (or whatever removed the file) and "
                "re-run `python -m training.review`."
            )

        fresh_lines = self.path.read_text(encoding="utf-8").splitlines()
        original = self._raw_lines
        if fresh_lines[:len(original)] != original:
            raise ConcurrentModificationError(
                f"{self.path} changed underneath this review session in a "
                "way that isn't a plain append (an existing line was "
                "edited, removed, or something else rewrote the file). "
                "Refusing to write and risk losing data -- stop the "
                "running assistant (and any other review session) and "
                "re-run `python -m training.review`."
            )
        appended = fresh_lines[len(original):]

        output_lines: list[str] = []
        for index, raw in enumerate(original):
            obj = self._parsed[index]
            entry_id = obj.get("id") if obj is not None else None
            if obj is not None and entry_id in self._labels:
                updated = dict(obj)
                updated["intent"] = self._labels[entry_id]
                output_lines.append(json.dumps(updated))
            else:
                output_lines.append(raw)
        output_lines.extend(appended)

        _atomic_write_lines(self.path, output_lines)

        labelled = len(self._labels)
        self._labels = {}
        # Keep the session consistent with what's now on disk, in case the
        # caller reuses it (load() is normally called again for that, but
        # this keeps _raw_lines/_parsed from silently going stale).
        self._raw_lines = output_lines
        self._parsed = None  # type: ignore[assignment]  # not reused without a fresh load()
        return SaveResult(written=True, labelled=labelled, appended_by_others=len(appended))


def _build_menu() -> list[tuple[int, Intent]]:
    """The numbered correction menu, built from `jarvis_nlu.intents.ALL_INTENTS`
    -- the single source of truth for this project. Never hardcode this list."""
    return list(enumerate(ALL_INTENTS, start=1))


def _format_menu(menu: list[tuple[int, Intent]]) -> str:
    return "\n".join(f"  {n:2d}. {intent.value}" for n, intent in menu)


def _format_candidate(candidate: Candidate, position: int, total: int) -> str:
    lines = [f"--- entry {position}/{total} ---",
             f"transcript: {candidate.transcript!r}"]
    confidence = candidate.confidence
    if confidence is not None:
        lines.append(f"predicted:  {candidate.predicted_intent}  "
                     f"(confidence {confidence:.2f})")
    else:
        lines.append(f"predicted:  {candidate.predicted_intent}")
    if candidate.signals:
        lines.append(f"signals:    {', '.join(candidate.signals)}")
    return "\n".join(lines)


def run_review(path: Path | str,
               input_func: Callable[[str], str] = input,
               print_func: Callable[..., None] = print) -> ReviewSummary:
    """Drive one interactive review session over `path`. `input_func` and
    `print_func` are injectable so tests never need real stdin/stdout."""
    session = ReviewSession(path)
    candidates = session.load()
    summary = ReviewSummary(total_candidates=len(candidates))

    if not candidates:
        print_func("Nothing to review -- every logged turn is already labelled.")
        return summary

    menu = _build_menu()
    menu_text = _format_menu(menu)
    valid_numbers = {n: intent for n, intent in menu}

    for position, candidate in enumerate(candidates, start=1):
        print_func("")
        print_func(_format_candidate(candidate, position, len(candidates)))
        print_func(menu_text)
        answer = input_func(PROMPT).strip()

        if answer.lower() == "q":
            break
        if answer == "":
            session.record(candidate.id, candidate.predicted_intent)
            summary.confirmed += 1
        elif answer.lower() == "s":
            summary.skipped += 1
        elif answer.isdigit() and int(answer) in valid_numbers:
            session.record(candidate.id, valid_numbers[int(answer)].value)
            summary.corrected += 1
        else:
            print_func(f"'{answer}' not understood -- skipping this entry.")
            summary.skipped += 1
            continue

        labelled_so_far = summary.confirmed + summary.corrected
        remaining = len(candidates) - position
        print_func(f"labelled {labelled_so_far} so far, {remaining} remaining")

    try:
        result = session.save()
    except ConcurrentModificationError as exc:
        print_func(f"REFUSING TO SAVE: {exc}")
        summary.saved = False
    else:
        summary.saved = result.written
        if result.appended_by_others:
            print_func(f"({result.appended_by_others} new line(s) logged by "
                       "the running assistant during this session were "
                       "preserved untouched.)")

    print_func("")
    print_func(f"Session summary: confirmed={summary.confirmed} "
               f"corrected={summary.corrected} skipped={summary.skipped} "
               f"(of {summary.total_candidates} candidates reviewed this session)")
    return summary


def main() -> None:
    root = Path(__file__).parent.parent
    log_path = root / "data" / "learning_log.jsonl"
    run_review(log_path)


if __name__ == "__main__":
    main()
