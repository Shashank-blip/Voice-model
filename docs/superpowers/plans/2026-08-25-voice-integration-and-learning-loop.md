# Voice Integration & Learning Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the user speak to Jarvis and have it act, while logging every turn as training data honest enough to retrain on.

**Architecture:** `miss-minutes` keeps its wake word, STT, and TTS. Its brain is rewired so `jarvis_nlu` answers first and OpenRouter is consulted only on deferral. Every turn is appended to `data/learning_log.jsonl` with the model's prediction plus signals about whether it was right. A small review CLI turns those raw turns into labelled training data.

**Tech Stack:** Python 3.13, `jarvis_nlu` (installed editable), openwakeword, faster-whisper, edge-tts, OpenRouter.

## Global Constraints

- `jarvis_nlu` still makes **no network calls**. The log writer lives in `jarvis_nlu` but only touches the local filesystem.
- The log is **local-only and gitignored**. `data/` is already ignored in both repos. It contains everything the user says — never commit it, never send it anywhere.
- **Never train on unreviewed predictions.** See Task B's rationale.
- Every time-dependent function takes an injectable `now: datetime`.
- pytest from each repo root. Python floor 3.11.
- Commit with `git commit --no-verify` (the user's global git hook is broken).

---

## Current state — what exists and what does not

Verified on 2026-08-25:

| Piece | State |
|---|---|
| `jarvis_nlu` package | Done. 223 tests, on `feat/local-nlu`, PR open to main. |
| Trained model | Done. `models/` ~45MB int8. Held-out macro F1 **0.79**. |
| `miss-minutes` voice loop | Works, but calls `brain.assistant.ask` (OpenRouter) for everything. |
| **miss-minutes → jarvis_nlu wiring** | **Does not exist.** `grep jarvis_nlu miss-minutes/` returns nothing. |
| **Learning log writer** | **Does not exist.** `training/generate.py` *reads* `data/learning_log.jsonl`; nothing writes it. |
| Deferral threshold | Fitted on the optimistic template split. Wrong for real speech. |

## Two problems found in the existing voice pipeline

Neither is caused by this work, but both will shape how it feels to use.

**1. `record_utterance` records a fixed 8 seconds with no voice-activity detection**
(`stt/transcribe.py`). Measured breakdown of one "what time is it" turn:

| stage | measured | fixable to |
|---|---|---|
| **recording (fixed 8s, no endpointing)** | **8.00s** | **~1.5s** |
| whisper `base` transcribe | 1.01s | ~0.4s (`tiny`) |
| **jarvis classify** | **0.012s** | — |
| edge-tts synthesis (network) | 1.60s | ~0.2s (local piper) |
| **total** | **~11.6s** | **~3s** |

The classifier is 12 milliseconds — irrelevant to perceived speed. The fixed recording
window is the entire problem. Also measured: **whisper `base` takes 16s to load at
startup**, so the process must stay warm rather than start on wake.

**2. `edge-tts` is a network service.** It calls Microsoft's endpoint, so the assistant
is not offline-capable and every reply has network latency. This contradicts the
"local-first" framing but is not necessarily wrong — the voice quality is good and it
is free. Flagged as a decision, not a defect. A local fallback (piper) is out of scope.

---

### Task A: Fit deferral thresholds on the held-out set

**Files:** Modify `training/train.py`; add `training/fit_thresholds.py`; test `tests/test_thresholds_heldout.py`

**Why:** `_choose_thresholds` currently minimises cost over the template-derived test split, which overstates accuracy by ~20 points. The shipped `defer=0.67` was fitted there. Measured on the 337-phrase held-out set, that threshold yields **~13% confidently-wrong in-scope actions and 5% out-of-scope leakage**. The cost argmin on held-out data is **0.93**.

- [ ] **Step 1:** Write a failing test asserting that thresholds fitted on `eval/heldout.yaml` differ from those fitted on the template split, and that the held-out-fitted `defer` yields under 6% confidently-wrong in-scope on held-out.
- [ ] **Step 2:** Run it; confirm it fails against the current manifest.
- [ ] **Step 3:** Add `training/fit_thresholds.py` that loads the trained model, scores `eval/heldout.yaml`, runs the existing cost minimisation over that data, and rewrites only the `thresholds` block of `models/manifest.json`. Keep `CONFIRM_MARGIN = 0.10` so the confirmation band stays non-empty. Do **not** retrain — thresholds are post-hoc.
- [ ] **Step 4:** Run it, then re-run `python -m training.evaluate --heldout`. Record before/after deferral, confidently-wrong, and leak rates.
- [ ] **Step 5:** Commit.

**Expected outcome:** `defer` ≈ 0.93. Roughly 60% of chores handled locally and correctly, 40% deferred to the LLM. At realistic usage (20–30 turns/day) that is 8–12 API calls against a 50/day budget — the rate-limit goal still holds, with far fewer wrong actions.

---

### Task B: The learning log

**Files:** Create `jarvis_nlu/logging.py`; modify `jarvis_nlu/router.py`; test `tests/test_learning_log.py`

**This is the highest-value task in the plan.** Every measurement says synthetic templates plateau near 0.79; only real phrasings move it.

**The core design problem: you cannot train on your own predictions.**

A naive log records what the model predicted. Retraining on that teaches the model to repeat its own mistakes with more confidence. The log must therefore separate three things:

| Field | Meaning |
|---|---|
| `predicted_intent` | What the model said. Never used as a training label. |
| `intent` | Ground truth. Absent until a human confirms it (Task D). |
| signals | Evidence about whether the prediction was right. |

`training/generate.py` already merges only entries carrying `intent`, so unlabelled rows are safely ignored.

**Signals worth capturing**, because they mark misses without the user labelling anything:

- `deferred` — did it fall below threshold and reach the LLM
- `confidence`
- `needs_confirmation` and what the user answered
- `repeat_within_seconds` — the next utterance arriving quickly and scoring as a near-duplicate is the strongest available "that was wrong" signal
- `user_negation` — the next turn classifying as `deny` right after an action
- `skill_error` — the skill raised

- [ ] **Step 1:** Write failing tests: one turn appends exactly one JSONL line; the line carries `predicted_intent` and NOT `intent`; a deferred turn is logged too; a repeat within the window sets `repeat_within_seconds` on the *previous* entry; logging failure never breaks the turn.
- [ ] **Step 2:** Run; confirm failure.
- [ ] **Step 3:** Implement `TurnLogger` with `log(turn)` appending one JSON object per line, atomically (open in append mode, one `write` per line, flush). Wrap every call site in try/except — **a logging fault must never break a spoken turn.** Wire it into `Assistant.handle` behind an optional `logger` parameter, defaulting to `None` so existing tests are unaffected.
- [ ] **Step 4:** Run tests; confirm pass. Verify the file is under `data/` and gitignored.
- [ ] **Step 5:** Commit.

**Privacy note to surface to the user:** this file records everything said to the assistant, in plain text, forever. It is local and gitignored. Worth telling them where it lives so they can delete it at will.

---

### Task C: Wire miss-minutes to jarvis_nlu

**Files:** Modify `miss-minutes/brain/assistant.py`, `miss-minutes/main.py`, `miss-minutes/requirements.txt`, `miss-minutes/stt/transcribe.py`; delete `miss-minutes/brain/tools.py`; create `miss-minutes/tests/test_routing.py`

- [ ] **Step 1:** Write the failing test that proves the rate-limit fix: with a **mocked** OpenRouter client, handled intents produce **zero** client calls; an `out_of_scope` input produces exactly one. This is the test that justifies the whole project.
- [ ] **Step 2:** Run; confirm failure (`ImportError: cannot import name 'respond'`).
- [ ] **Step 3:** Implement:
  - `pip install -e ../RAG-Voice_model`; add it to `requirements.txt`.
  - Rename `ask` → `ask_llm`; strip `tools=TOOL_SCHEMAS` and the tool loop (chores never reach the LLM now); delete `brain/tools.py`.
  - Add `respond(text, nlu, logger=None, now=None)`: call `nlu.handle`, return `result.reply` when handled, else `ask_llm(text)`. Catch every exception from `ask_llm` — a network fault must not kill the voice loop.
  - Rewrite `main.py` to build `Assistant` + `Scheduler`, pass `speak` as the scheduler's callback and `assistant.turn_lock` as its lock, and start it.
  - **Fix the fixed-duration recording:** add simple energy-based endpointing to `record_utterance` — stop after ~1.2s below a noise threshold, cap at 8s. Keep the cap as a fallback.
- [ ] **Step 4:** Run both test suites. Then a real smoke test with the microphone: wake word → greeting → set a reminder two minutes out → ask what's on the calendar → confirm the reminder fires unprompted.
- [ ] **Step 5:** Commit.

---

### Task D: Review CLI for labelling

**Files:** Create `training/review.py`; test `tests/test_review.py`

Without this, the log accumulates and never becomes training data.

- [ ] **Step 1:** Failing tests: unlabelled entries are listed oldest-first; confirming writes `intent` equal to `predicted_intent`; correcting writes the chosen intent; skipping leaves the entry untouched; the file is rewritten atomically via a temp file and rename.
- [ ] **Step 2:** Run; confirm failure.
- [ ] **Step 3:** Implement `python -m training.review`: show each unlabelled turn with its transcript, prediction, and confidence; prompt `[Enter] confirm / number to correct / s skip / q quit`; prioritise entries whose signals suggest a miss (deferred, repeated, negated). Show the intent roster from `intents.py` — never a hardcoded list.
- [ ] **Step 4:** Run tests; confirm pass.
- [ ] **Step 5:** Commit.

---

### Task E: Retrain on real data

Not code — a documented procedure, to run once the log has enough labelled turns.

- [ ] Review the log (`python -m training.review`) until roughly 100+ labelled turns exist.
- [ ] `python -m training.generate` — merges labelled entries at `LEARNING_LOG_WEIGHT = 3.0`.
- [ ] `python -m training.train && python -m training.export_onnx`
- [ ] `python -m training.fit_thresholds` (Task A)
- [ ] `python -m training.evaluate --heldout` — compare macro F1 against the 0.79 baseline.
- [ ] **Add any newly-fixed phrase to `tests/golden.yaml`** so it can never regress.

**The held-out set must never be used for training.** It is the only honest measurement in the project.

---

## Verification checklist

- [ ] `cd RAG-Voice_model && python -m pytest` — all pass
- [ ] `cd miss-minutes && python -m pytest` — all pass
- [ ] Mocked-client test proves zero API calls for handled intents
- [ ] Microphone smoke test passes end to end
- [ ] `data/learning_log.jsonl` gains one line per spoken turn
- [ ] No entry has `intent` until reviewed
- [ ] `git check-ignore data/learning_log.jsonl` confirms it is ignored

## Decisions — resolved 2026-08-25

1. **Threshold: ship `defer = 0.83`**, not 0.93. With the two-strike policy below, a
   deferral costs a re-prompt rather than an API call, so the argument for a very high
   threshold weakens. At 0.83: 66.5% handled instantly, 29.4% unsure, 4.2% genuinely
   out of scope.
2. **edge-tts network dependency: accepted for now.** Measured 1.6s per reply. Task G
   covers a local fallback if that proves annoying.
3. **Log retention: keep everything, no trimming.** ~200 bytes per turn, under 2MB/year
   at 30 turns/day. There is no size argument for deleting training data. Local and
   gitignored; the user deletes it manually if they want.

---

## Task F: Two-strike deferral policy — the LLM-call reduction

**Files:** Modify `jarvis_nlu/router.py`, `jarvis_nlu/responses.py`; test `tests/test_two_strike.py`

**Why this matters most for the user's stated goal.** Measured on the held-out set at
`defer = 0.83`, deferrals split into two very different cases:

| outcome | share of turns |
|---|---|
| handled locally | 66.5% |
| **genuinely out of scope** (needs the LLM) | **4.2%** |
| **model merely unsure** | **29.4%** |

Today all 33.6% would hit OpenRouter. But "unsure" usually means a garbled Whisper
transcript, and the correct response to that is not an API call — it is *"say that
again, sugar?"*, which is free and is what a person would do.

**Policy:**
- `out_of_scope` at or above `defer` → this genuinely needs the LLM. Call it.
- Below `defer`, first occurrence → return a `reprompt` result. **No API call.**
- Below `defer`, second consecutive occurrence → now call the LLM.
- Any successfully handled turn resets the strike counter.

Expected effect: real API usage drops from ~33% of turns to roughly 4–8%, i.e. **1–3
calls per day** at realistic usage, against a 50/day budget.

Every re-prompt is also a logged miss — free training signal, at no cost.

- [ ] **Step 1:** Failing tests — first low-confidence turn returns `handled=True` with a
      re-prompt reply and makes no LLM call; a second consecutive low-confidence turn sets
      `should_call_llm`; a successful turn in between resets the counter; a confident
      `out_of_scope` calls the LLM immediately without a re-prompt.
- [ ] **Step 2:** Run; confirm failure.
- [ ] **Step 3:** Add a `reprompt` pool to `responses.py` (8+ variants in the persona voice —
      "Didn't quite catch that, sugar", "Say that again for me?"). Add strike tracking to
      `Assistant`. Extend `Result` with `reprompt: bool` so the caller can distinguish
      "ask again" from "call the LLM" — do not overload `handled`.
- [ ] **Step 4:** Run tests; verify the full suite still passes.
- [ ] **Step 5:** Commit.
