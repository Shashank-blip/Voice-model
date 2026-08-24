# Jarvis local voice assistant

Jarvis is a local-first assistant for common voice chores. Its normal commands do not call an LLM: reminders, notes, a local calendar, safe file search, and greetings are handled deterministically.

## Start here

```powershell
cd RAG-Voice_model
python -m jarvis --text
```

Text mode is the reliable first-run experience and uses only the Python standard library. Say or type `hey jarvis`, then try:

- `remind me to call Alex tomorrow at 6 pm`
- `add dentist appointment on 2026-08-30 at 4 pm to my calendar`
- `what is on my calendar today`
- `take a note buy milk`
- `find my resume`
- `read my reminders`

Data is stored locally in `data/` (which is ignored by Git). Configure allowed file-search folders in `jarvis.config.json`; Jarvis will never search outside them.

## Voice mode

Voice mode is local: openWakeWord listens for the wake phrase, faster-whisper transcribes locally, and Windows uses its built-in speech synthesizer for responses. `--text` remains useful for validating commands before enabling an always-on microphone.

```powershell
pip install -r requirements-voice.txt
python -m jarvis --voice
```

The first voice install may download local wake-word/STT model files. Review the microphone permission requested by your operating system. Voice mode currently targets English and expects an openWakeWord-compatible wake model; configure a custom model for the exact phrase “Hey Jarvis” if the default model does not recognize it reliably.

## LLM and training strategy

An LLM is **not needed** for supported chores. Unknown requests are logged in `data/learning_log.jsonl` (without recording audio) so you can review the phrases you actually use. Add those phrases to the rule set or use the log later as a labelled dataset. Fine-tuning should only be considered after enough reviewed examples exist.

## Safety

- File search is read-only and restricted to configured folders.
- Calendar and reminder changes require an explicit command; no external calendar is modified.
- This MVP keeps its own local calendar. Connecting Google/Outlook is a separate, consent-based adapter.
