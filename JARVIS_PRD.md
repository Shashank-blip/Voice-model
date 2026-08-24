# JARVIS — Product Requirements Document

## 1. Vision

Build a personal, Jarvis-style AI assistant that is always available locally, wakes when called, understands the user's environment, can retrieve knowledge from their files and projects, can execute controlled actions on the computer, can interact with AI coding agents, and can communicate through voice and a visual character interface.

The assistant should be an **agentic control layer**, not merely a voice chatbot.

## 2. Core Goals

- Wake on a locally detected wake word.
- Convert natural speech to text and respond through speech.
- Understand the user's projects, files, notes, and explicitly stored memories.
- Maintain live awareness of relevant system and development state.
- Execute tools for filesystem, terminal, Git, system, and coding-agent operations.
- Integrate with Agent Bridge to monitor and control concurrent Claude/Codex sessions.
- Ask for confirmation before risky actions.
- Support RAG for project/document knowledge.
- Support long-term and short-term memory.
- Support asynchronous/background tasks.
- Provide an animated desktop character/interface.
- Minimize LLM usage by using local state, deterministic tools, caching, and model routing.

## 3. Non-Goals for V1

- Fully autonomous unrestricted computer control.
- Training a foundation LLM from scratch.
- Blindly indexing the entire filesystem.
- Sending every system event to an LLM.
- Replacing the operating system shell or developer tools.

## 4. Primary User Flow

1. Assistant runs locally in a dormant state.
2. Local wake-word engine detects the configured wake word.
3. Assistant activates microphone/STT.
4. User gives a request.
5. Jarvis Core determines intent, required context, and required tools.
6. The system retrieves live state, RAG context, memory, or tool information as needed.
7. The LLM reasons only when reasoning is required.
8. Tools execute approved actions.
9. Results are returned to the assistant.
10. TTS produces the response.
11. Assistant returns to dormant mode.

## 5. Major Capabilities

### Voice
- Local wake-word detection.
- Streaming or low-latency STT.
- TTS.
- Interruptible/barge-in speech.
- Voice activity detection.

### Intelligence
- Intent routing.
- Tool selection.
- Multi-step planning.
- Context assembly.
- Model routing.
- Error recovery.

### Knowledge
- Project/document ingestion.
- Embeddings and vector retrieval.
- Code/document-aware chunking.
- Source-aware answers.
- Conversation and project memory.

### Computer Control
- Filesystem operations.
- Terminal/process operations.
- Git operations.
- System information.
- Controlled application integrations.

### AI Agent Control
- List active agent sessions.
- Inspect agent state.
- Send input to an agent.
- Start/stop agent sessions.
- Receive agent events.
- Route actions to the correct concurrent session.
- Integrate existing Agent Bridge infrastructure.

### Safety
- Tool permission levels.
- Confirmation for destructive/high-impact actions.
- Command validation.
- Workspace restrictions.
- Audit logs.

### Proactive/Event-Driven Behavior
- Consume events from Agent Bridge and other integrations.
- Maintain live state without constant LLM calls.
- Run background tasks.
- Notify the user when configured conditions occur.

## 6. Architecture Principles

1. The LLM is the reasoning layer, not the event loop.
2. Live state is stored separately from RAG.
3. RAG supplies knowledge; tools perform actions.
4. The always-on path should be local wherever practical.
5. Expensive models are reserved for complex reasoning.
6. Every tool invocation is explicit and auditable.
7. Agent sessions must remain independently addressable.
8. Components should be replaceable through interfaces/adapters.

## 7. Success Criteria

V1 is successful when the user can say a wake phrase, ask a natural-language question, retrieve information about their development environment, perform safe computer actions, and control Agent Bridge sessions without manually opening a terminal.

The system should remain responsive, avoid unnecessary LLM calls, preserve session identity, and safely request confirmation for risky actions.

## 8. Future Extensions

- Windows/ConPTY support.
- Additional coding-agent adapters.
- Screen understanding and computer vision.
- Browser control.
- Calendar/email integrations.
- Mobile/Discord remote interface.
- Local LLM inference.
- Advanced background agents.
- Rich animated character behavior.
