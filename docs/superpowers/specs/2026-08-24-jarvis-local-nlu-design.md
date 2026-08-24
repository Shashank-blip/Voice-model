# Jarvis Local NLU — Design

**Date:** 2026-08-24
**Status:** Approved for planning
**Scope:** `RAG-Voice_model/` (new trained model) + `miss-minutes/` (integration)

## 1. Problem

`miss-minutes` sends every utterance to an OpenRouter free-tier model capped at
20 requests/minute and 50/day. Routine chores — "remind me to call mom at six",
"what's on my calendar" — spend that budget on work that needs no reasoning.
When the cap is hit the assistant stops functioning entirely.

`RAG-Voice_model` already attempts these chores without an LLM, but routes
intent through an ordered regex chain in which earlier patterns shadow later
ones. Confirmed by execution on 2026-08-24:

- `"what is on my calendar today"` matches the event-*add* pattern and returns
  an error. `todays_events()` is unreachable dead code.
- `"remind me I'm meeting Bob tomorrow at 5 pm"` matches the greeting pattern
  and returns a greeting.
- The test covering the calendar path passes vacuously: it asserts
  `"dentist appointment" in answer`, and that substring appears in the *error
  message*.

## 2. Goal

Replace regex routing with a **locally trained intent classifier**, and have
`miss-minutes` consult it before the LLM. The LLM is then reserved for
genuinely open-ended questions.

Target: **under 5% of in-scope utterances reach the network.**

## 3. Non-goals for v1

Deferred to v2 (explicitly descoped by the user on 2026-08-24): file search and
open, battery/CPU/RAM/disk queries, volume control, application launching,
media control. Also out of scope: fine-tuning a generative model, document RAG
over projects, replacing `miss-minutes` voice I/O or persona.

## 4. Architecture

```
wake word --> STT --> jarvis_nlu.Assistant.handle(text)
                        |
                        +-- handled=True  --> speak(result.reply)    0 API calls
                        +-- handled=False --> OpenRouter --> speak() 1 API call
```

`RAG-Voice_model/` becomes `jarvis_nlu`, an installable package holding the
model, slot extraction, skills, storage, and the proactive scheduler. It makes
**no network calls of any kind**. `miss-minutes` installs it editable
(`pip install -e ../RAG-Voice_model`) and keeps its wake word, STT, TTS and
persona. One copy of the logic, two consumers.

### 4.1 Public API

The entire contract between the packages:

```python
@dataclass(frozen=True)
class Result:
    handled: bool          # False => caller should fall back to the LLM
    reply: str | None      # spoken text when handled
    intent: str
    confidence: float
    slots: dict[str, object]
    needs_confirmation: bool = False

class Assistant:
    def __init__(self, config: Config) -> None: ...
    def handle(self, text: str, now: datetime | None = None) -> Result: ...
```

`now` is injectable so every time-dependent test runs on a fixed clock.

### 4.2 Configuration

`Config` is a frozen dataclass loaded from `jarvis.config.json` with defaults
for every field, so a missing file is valid:

| Field | Default | Purpose |
|---|---|---|
| `model_dir` | `models/` | ONNX model + manifest location |
| `db_path` | `data/jarvis.db` | SQLite database |
| `tick_seconds` | `20` | Proactive scheduler poll interval |
| `daily_brief_at` | `"08:30"` | Daily brief time; `null` disables it |
| `persona_path` | `null` | Optional persona file for tone (see §7.1) |

Thresholds (`τ_defer`, `τ_confirm`) are **not** config — they are training
outputs and live in the model manifest, so a retrain cannot leave a stale
hand-tuned value behind.

### 4.3 Package layout

```
RAG-Voice_model/
├── jarvis_nlu/
│   ├── __init__.py       # exports Assistant, Result, Config
│   ├── intents.py        # intent registry — single source of truth
│   ├── model.py          # ONNX session, softmax, calibrated confidence
│   ├── slots.py          # datetime + entity extraction (rule-based)
│   ├── responses.py      # small-talk pools, no-repeat selection
│   ├── storage.py        # SQLite schema, migrations, DAOs
│   ├── router.py         # Assistant.handle
│   ├── proactive.py      # scheduler thread
│   └── skills/           # reminders, calendar, notes, clock, smalltalk
├── training/
│   ├── templates/        # one YAML per intent
│   ├── generate.py       # dataset synthesis + STT-noise augmentation
│   ├── train.py          # fine-tune + temperature calibration
│   ├── export_onnx.py    # ONNX export + int8 quantization
│   └── evaluate.py       # metrics, confusion matrix, gate enforcement
├── tests/
├── models/               # build output (gitignored)
└── pyproject.toml
```

Each module has one responsibility and is testable without the others.
`skills/` modules receive a storage handle and return strings; they never touch
the model.

## 5. The trained model

### 5.1 Dataset

`training/generate.py` synthesises ~6,000 labelled examples from per-intent YAML
templates with slot placeholders, expanded combinatorially across paraphrase
variants, filler words ("um", "hey", "can you"), and contractions.

Three properties matter more than raw volume:

1. **An `out_of_scope` class.** Sampled open-domain questions ("who won the
   world cup", "explain recursion", "what's the weather in Tokyo"). Without a
   negative class the model cannot express "I should defer" and will confidently
   mislabel everything. This class is what protects the LLM fallback path.
2. **STT-noise augmentation.** The model's real input is Whisper output, not
   typed text. A fraction of examples are perturbed with realistic transcription
   errors — dropped articles, homophone substitution ("to"/"two", "for"/"four"),
   missing terminal punctuation, lowercase-only. Training on clean text alone is
   a distribution mismatch with the deployed input.
3. **Real phrases.** Any entries in `data/learning_log.jsonl` are merged in with
   higher sample weight, so the model improves on the user's actual phrasing.

Splits are 80/10/10 train/val/test, stratified, with template-level grouping so
that paraphrases of one template cannot straddle train and test and inflate the
score.

### 5.2 Model and training

`sentence-transformers/all-MiniLM-L6-v2` encoder plus a classification head,
fine-tuned **end-to-end** — the encoder's weights update; it is not frozen.
Cross-entropy with class weights to offset intent frequency imbalance.

Exported to ONNX and int8-quantized: **~23MB, p95 under 25ms on CPU.**

### 5.3 Confidence and deferral

Raw softmax is overconfident. Temperature scaling is fitted on the validation
split so the score is calibrated, then two thresholds apply:

- `confidence < τ_defer` → `handled=False`, caller falls back to the LLM.
- `τ_defer ≤ confidence < τ_confirm` **and** the intent is destructive
  (`cancel_reminder`) → `needs_confirmation=True`; the assistant asks before
  acting.

Both thresholds are chosen on the validation split to satisfy the gates in §9,
and are stored in the model manifest rather than hardcoded.

## 6. Slot extraction

Deliberately **rule-based, not learned.** Learned NER underperforms a focused
rule engine on datetimes, and datetimes are the whole ballgame for reminders.
A hand-written extractor is also fast, debuggable, and has failure modes we
control. `dateparser` is rejected as a dependency: it is large, slow, and its
ambiguity handling is hard to pin down.

The grammar must handle, at minimum:

| Input | Resolution |
|---|---|
| `at 6`, `at 6pm`, `6:30`, `at 6:30 pm` | time today or tomorrow (see rule below) |
| `today at 4`, `tomorrow at 6`, `tonight at 8` | relative day + time |
| `friday`, `next friday`, `on monday at 9` | named weekday |
| `2026-08-30 at 4pm` | ISO date |
| `in 20 minutes`, `in 2 hours`, `in half an hour` | offset from now |
| `tomorrow morning` / `afternoon` / `evening` / `tonight` | daypart → 09:00 / 14:00 / 19:00 / 20:00 |

**Bare-time rule (explicit, because it is the most common phrasing and is
currently rejected):** when no meridiem is given, resolve to the *next*
occurrence of that clock time. At 09:00, `"at 6"` means 18:00 today. At 20:00,
`"at 6"` means 06:00 tomorrow. This fixes `"remind me to call mom at 6 pm"`,
which today returns *"Please include a time"*.

The extractor returns the resolved `datetime` **and** the residual text with the
time expression removed, which becomes the reminder body.

## 7. Intents — v1 (20 classes)

| Group | Intents |
|---|---|
| Reminders | `add_reminder`, `list_reminders`, `cancel_reminder` |
| Calendar | `add_calendar_event`, `query_calendar` |
| Notes | `add_note`, `list_notes`, `search_notes` |
| Clock | `get_time`, `get_date` |
| Small talk | `greeting`, `how_are_you`, `user_mood`, `thanks`, `goodbye` |
| Meta | `affirm`, `deny`, `repeat_last`, `sleep` |
| Fallback | `out_of_scope` |

`intents.py` is the single source of truth; training, evaluation, routing and
tests all read the roster from it, so adding an intent cannot leave one layer
out of sync.

### 7.1 Small talk

Handled locally with **zero API calls**. Each small-talk intent owns a pool of
8–12 persona-written variants, selected at random excluding the last 3 used, so
repeated greetings do not sound like a phone tree. `user_mood` additionally
branches on a coarse positive/negative/tired slot.

Pools live in `responses.py` as plain data, editable without retraining.

**Tone must match the persona.** `miss-minutes` has an established voice in
`config/persona.md` (warm, folksy — "sugar", "mm-hmm"). Local replies and
LLM replies alternate within a single conversation, so if the pools are written
in a neutral register the seam will be audible. The pools are authored in that
same voice, and this is a review criterion when they are written, not an
afterthought.

### 7.2 Semantic note search

The MiniLM encoder is already resident for classification, so `search_notes`
embeds note text at write time and retrieves by cosine similarity at query time.
*"What did I note about the wifi"* works without keyword overlap. This is the
retrieval capability the folder name refers to, at near-zero marginal cost —
no vector database, embeddings stored as a BLOB column.

## 8. Storage

Unify on **SQLite**, replacing `RAG-Voice_model`'s `JsonStore` (whose `read()`
returns the shared mutable default, so callers appending to it corrupt the
default) and absorbing `miss-minutes`' existing `data/minutes.db`.

```sql
reminders(id, text, due_at TEXT NULL, created_at, delivered_at TEXT NULL,
          cancelled INTEGER DEFAULT 0)
events   (id, title, starts_at TEXT, created_at)
notes    (id, text, created_at, embedding BLOB NULL)
meta     (key PRIMARY KEY, value)      -- schema_version, last_brief_date
```

**Migration.** `miss-minutes`' current `reminders.due` is free text. On first
run, existing rows are carried over and each `due` is best-effort parsed by the
new extractor; unparseable values leave `due_at` NULL, which surfaces the
reminder in listings but never fires proactively. No data is discarded. The
migration is idempotent and versioned via `meta.schema_version`.

## 9. Evaluation

Gates enforced by `training/evaluate.py`; the build fails if any is missed.

| Gate | Threshold |
|---|---|
| Macro F1 (held-out test) | ≥ 0.95 |
| Any off-diagonal confusion cell between chore intents | ≤ 2% |
| In-scope utterances deferred to LLM | < 5% |
| `out_of_scope` wrongly handled locally | < 2% |
| p95 inference latency, CPU | < 25ms |
| Golden regression suite | 100% |

The **confusion matrix is a required artifact**, not a nicety: shadowing between
`add_calendar_event` and `query_calendar` is precisely the bug class that
shipped undetected in the regex router, and it shows up as an off-diagonal cell.

### 9.1 Golden regression suite

Seeded with every phrase confirmed broken on 2026-08-24. Each must pass:

| Utterance | Expected |
|---|---|
| `what is on my calendar today` | `query_calendar` (**not** `add_calendar_event`) |
| `remind me I'm meeting Bob tomorrow at 5 pm` | `add_reminder` (**not** `greeting`) |
| `remind me to say hi to Alex tomorrow at 6 pm` | `add_reminder` |
| `remind me to call mom at 6 pm` | `add_reminder`, `due_at` resolved |
| `who won the world cup` | `out_of_scope` → deferred |

Every future bug fix adds a row here.

## 10. Proactive layer

A daemon thread in `proactive.py` ticking every 20s (configurable), holding an
injected `speak: Callable[[str], None]`.

- **Due reminders.** `due_at <= now AND delivered_at IS NULL AND NOT cancelled`
  → speak → **then** set `delivered_at`. Writing the timestamp only after
  `speak()` returns means a crash mid-announcement replays the reminder rather
  than silently swallowing it. The current implementation marks reminders
  complete as a side effect of *reading* them, and loses them on crash.
- **Daily brief.** Once per calendar day at a configured time (default 08:30),
  tracked in `meta.last_brief_date`, and only when there is something to report.

A conversation lock is acquired for the duration of each turn; the scheduler
skips a tick rather than talking over the user. Nothing else is proactive — no
idle nudges, no battery warnings.

## 11. Error handling

- **Model file missing or corrupt** — `Assistant` construction raises with the
  exact command to build it. It does not silently degrade to regex.
- **Low confidence** — deferral, not a guess. This is the designed path, not an
  error.
- **Slot extraction fails on an intent that needs one** — ask a targeted
  question ("When should I remind you?") rather than rejecting the utterance.
- **Skill raises** — caught at the router boundary, logged with traceback,
  returns a spoken apology. The assistant never dies on a skill fault.
- **Storage locked** — SQLite in WAL mode with a busy timeout; the scheduler
  thread and main thread both write.
- **LLM unreachable in `miss-minutes`** — unchanged from today: the existing
  handler already distinguishes rate-limit from other failures.

## 12. Testing

- **Unit** — slot extraction (large table-driven suite over §6's grammar,
  including the bare-time rule at several times of day); no-repeat response
  selection; storage DAOs and migration idempotency; each skill against a
  temp database.
- **Model** — the §9 gates, run as tests.
- **Integration** — `Assistant.handle` end-to-end for all 20 intents on a fixed
  clock.
- **Proactive** — fake clock: assert a due reminder speaks exactly once, that
  `delivered_at` is written only after `speak` returns, and that a raise inside
  `speak` leaves the reminder undelivered.
- **`miss-minutes` routing** — with a **mocked** OpenRouter client, assert that
  handled intents produce **zero** client calls, and that `out_of_scope`
  produces exactly one. This is the test that proves the rate-limit fix.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data ≠ real speech | STT-noise augmentation (§5.1); learning log feeds retraining |
| Retrain needed after real use | Expected. `training/` is a first-class, re-runnable pipeline, not a one-shot script |
| Threshold tuned on synthetic data may be optimistic in practice | Thresholds live in the manifest and are adjustable without retraining |
| First-run model download (~90MB base encoder) | Documented in README; training is a separate step from running |

## 14. v2 backlog

File search and open; battery, CPU/RAM, disk; volume; app launch; media
control. These reuse the same intent registry and skills pattern — adding them
is templates plus a skill module plus a retrain, with no architectural change.
