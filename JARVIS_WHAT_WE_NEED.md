# JARVIS — What We Need

## 1. Core Runtime

- Python or TypeScript service for the assistant core.
- Local configuration system.
- Process manager.
- Structured logging.
- Environment/secrets management.
- Cross-platform abstraction layer.

## 2. Audio Stack

### Input
- Microphone capture.
- Voice Activity Detection (VAD).
- Local wake-word engine.
- Wake-word model.
- Streaming audio pipeline.

### Speech-to-Text
- STT engine/API.
- Streaming transcription support.
- Silence/end-of-utterance detection.
- Error and timeout handling.

### Output
- TTS engine.
- Streaming audio playback.
- Interruptible playback / barge-in.
- Queue management for responses.

## 3. Jarvis Core

Build an orchestration layer containing:

- Intent router.
- Context manager.
- Planner.
- Tool selector.
- Tool executor.
- Response generator.
- Error/retry manager.
- Model router.
- Conversation state manager.

Suggested internal flow:

`request -> intent -> context -> plan -> tools -> results -> response`

## 4. Tool Registry

Every capability should be exposed through a typed tool interface.

### Filesystem
- list_directory
- search_files
- read_file
- create_file
- edit_file
- move_file
- delete_file

### Terminal
- execute_command
- start_process
- stop_process
- process_status

### Git
- status
- diff
- log
- branch
- checkout
- commit
- push

### System
- CPU usage
- RAM usage
- disk usage
- network status
- running processes

### Agent Bridge
- list_sessions
- get_session
- get_session_state
- send_input
- start_session
- stop_session
- receive_events

### Future
- Docker
- Kubernetes
- AWS
- Browser
- Calendar
- Email
- Discord

## 5. Permission System

Define tool risk classes:

### Safe
Read-only operations such as listing files, reading files, Git status, and system metrics.

### Confirmation Required
Editing files, installing packages, starting processes, committing changes, and similar operations.

### High Risk
Deleting files, force pushing, arbitrary destructive shell commands, shutdown/reboot, and other system-impacting operations.

Each tool should declare its risk level and required confirmation policy.

## 6. RAG System

Components:

- File/document discovery.
- Parsers for Markdown, PDF, text, source code, etc.
- Code-aware chunking.
- Metadata extraction.
- Embedding model.
- Vector database.
- Retriever.
- Reranker if needed.
- Context builder.
- Citation/source tracking.

Suggested collections:

- Agent Bridge
- SystemCraft
- Portfolio
- Notes
- Documents
- Other projects

Do not use RAG for rapidly changing state.

## 7. Memory System

Separate memory from knowledge retrieval.

### Short-term
- Current conversation.
- Current task.
- Current tool results.

### Long-term
- Explicit user memories.
- Preferences.
- Important decisions.
- Project decisions.
- Conversation summaries.

### Project memory
- Architecture decisions.
- Known constraints.
- Development history.
- Important implementation notes.

## 8. Live State System

Maintain authoritative state for:

- Active coding-agent sessions.
- Agent status.
- Current projects.
- Processes.
- Git repositories/branches.
- Background tasks.
- Relevant system metrics.

This should be updated from events and direct system queries, not inferred from RAG.

## 9. Event System

Create an event bus or equivalent event-driven layer.

Potential events:

- agent.started
- agent.working
- agent.waiting
- agent.completed
- agent.failed
- build.failed
- build.completed
- file.changed
- task.completed
- system.threshold_reached

The event system should update state and only invoke an LLM when semantic reasoning is actually required.

## 10. Agent Bridge Integration

Reuse the existing Agent Bridge instead of rebuilding agent control.

Required interface:

`Jarvis -> Agent Bridge -> agent session`

Capabilities:

- Session discovery.
- Session-aware routing.
- Claude integration.
- Future Codex/other-agent adapters.
- Remote input injection.
- Event consumption.
- Concurrent session handling.
- Resume-session support.

## 11. Storage

Potential architecture:

### PostgreSQL
Structured persistent data:
- settings
- sessions
- tasks
- permissions
- memories
- audit records

### Vector database / pgvector
- RAG embeddings.
- Semantic memory.

### Redis
- ephemeral state.
- queues.
- event processing.
- caching.

### Filesystem
- logs.
- indexed artifacts.
- local models.
- configuration.

## 12. LLM Layer

Build a model abstraction rather than hard-coding one provider.

Requirements:

- Provider adapters.
- Model routing.
- Token accounting.
- Rate-limit handling.
- Retry/backoff.
- Request caching.
- Context trimming.
- Streaming responses.

Use:
- local/deterministic logic for simple state queries,
- cheaper models for simple reasoning,
- stronger models for complex planning and analysis.

## 13. Vision

Later subsystem:

- Screenshot capture.
- Vision model.
- Screen-region analysis.
- Optional mouse/keyboard control.
- UI element detection.

## 14. Character/UI

States:

- Dormant
- Listening
- Thinking
- Using tool
- Waiting for confirmation
- Speaking
- Error

The character is the interface layer; it should not contain core assistant logic.

## 15. Background Tasks

Need:

- Task queue.
- Worker process.
- Task status.
- Retry policy.
- Completion events.
- User notification.

Example:

`Run tests -> background task -> completion event -> Jarvis reports result`

## 16. Observability

Track:

- LLM latency.
- Token usage.
- API errors.
- Rate-limit events.
- Tool execution time.
- STT latency.
- TTS latency.
- Wake detection.
- RAG retrieval latency.
- Failed tool calls.
- Agent-session events.

## 17. Testing

### Unit
- Routing.
- Permission logic.
- Tool validation.
- Memory.
- Retrieval.

### Integration
- Jarvis -> tool.
- Jarvis -> Agent Bridge.
- Jarvis -> RAG -> LLM.
- Event -> state update.

### End-to-End
- Wake -> STT -> reasoning -> tool -> result -> TTS.

## 18. Deployment / Packaging

Eventually provide:

- Linux installer.
- Windows installer.
- Service/background startup.
- Configuration wizard.
- Secure API-key storage.
- Upgrade mechanism.
- Log collection/debug mode.
