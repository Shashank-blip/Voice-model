# JARVIS — Training and Learning Plan

## Important Terminology

RAG is not model training.

- **RAG** gives the assistant access to external knowledge.
- **Fine-tuning** changes model behavior.
- **Memory** preserves selected information across interactions.
- **Tools** give the assistant capabilities.
- **System state** provides current facts about the computer/environment.

Do not begin by training a foundation model from scratch.

## 1. Knowledge Preparation

Before any fine-tuning, build a high-quality knowledge corpus.

Sources may include:

- Agent Bridge README and documentation.
- Source code and architecture notes.
- SystemCraft documentation.
- Other personal projects.
- Explicitly provided notes.
- Development conventions.
- Tool documentation.

Each indexed document should have metadata such as:

- project
- file path
- file type
- repository
- language
- section
- last modified time

## 2. RAG Training/Indexing Pipeline

Build:

`documents -> parsing -> cleaning -> chunking -> embeddings -> vector store`

For source code, use structure-aware chunking where possible.

Store metadata so retrieval can distinguish:

- project
- source file
- function/class
- documentation
- version/time

Evaluate retrieval separately from generation.

## 3. Retrieval Evaluation

Create a small benchmark containing realistic questions.

Examples:

- "How does Agent Bridge route concurrent sessions?"
- "Where is the Discord response injected?"
- "How does SystemCraft handle autosave races?"
- "What is the Windows support plan?"

Measure:

- Recall@K.
- Precision of retrieved chunks.
- Context relevance.
- Answer faithfulness.
- Source attribution.

Do not fine-tune the LLM until retrieval quality is acceptable.

## 4. Memory Learning

Create explicit memory categories:

### User preferences
Only store information the user wants remembered.

### Project memory
Architecture decisions, constraints, and implementation decisions.

### Conversation summaries
Compress completed conversations into durable summaries.

### Task memory
Active tasks, background jobs, and their results.

Memory should be retrievable and editable rather than permanently hidden inside model weights.

## 5. Tool-Use Training

The assistant needs to learn when to call tools.

Start with structured tool schemas rather than fine-tuning.

Example:

`get_agent_sessions()`

`read_file(path)`

`search_files(query)`

`execute_command(command)`

`send_agent_input(session_id, input)`

The model should receive tool descriptions and return structured tool calls.

## 6. Build a Tool-Calling Dataset

Once the system works, collect examples of:

- User request.
- Relevant context.
- Correct tool.
- Correct arguments.
- Expected result.
- Whether confirmation was required.

Example:

User:
"Which Claude sessions are waiting?"

Expected tool:
`list_sessions`

User:
"Tell PromptWall to continue."

Expected tool:
`send_agent_input(session_id="PromptWall", input="continue")`

The dataset should contain both successful examples and negative examples where the assistant must refuse or request clarification.

## 7. Planning Dataset

For multi-step tasks, create examples containing:

- goal
- intermediate steps
- required tools
- dependencies
- final result

Example:

"Find why Agent Bridge tests are failing and fix them."

Possible plan:

1. Inspect Git status.
2. Run tests.
3. Read failing test output.
4. Inspect relevant source.
5. Propose or make a change.
6. Run tests again.
7. Report result.

The assistant should not execute destructive steps without authorization.

## 8. Safety Training

Create examples covering:

- safe read-only actions.
- confirmation-required actions.
- dangerous commands.
- ambiguous commands.
- requests outside allowed workspaces.

Examples:

"Read bot.py" -> execute.

"Delete the repository" -> confirmation/high-risk policy.

"Run rm -rf on the project directory" -> block or require explicit high-risk authorization according to policy.

The permission engine should enforce safety independently of the LLM.

## 9. Personality / Behavior

Do not fine-tune personality first.

Define a system behavior specification covering:

- concise responses.
- respectful tone.
- when to ask clarification.
- when to report tool results.
- how to announce actions.
- how to handle failure.
- how to behave when interrupted.
- how to address the user.
- how proactive the assistant should be.

A prompt/policy layer is easier to iterate than fine-tuning.

## 10. Voice Training

Wake-word detection is separate from LLM training.

Potential local training data:

- positive wake-word samples.
- background-noise samples.
- negative speech samples.
- different microphone distances.
- different speaking speeds.

Evaluate:

- false activation rate.
- missed wake rate.
- detection latency.

STT and TTS can initially use existing models/services; custom training is optional.

## 11. Fine-Tuning — Later

Only consider fine-tuning after the following work:

- RAG works.
- Tool calling works.
- Memory works.
- Permission system works.
- Model routing works.
- You have a meaningful dataset of real interactions.

Potential fine-tuning goals:

- Better tool selection.
- Better structured tool arguments.
- Better planning.
- Better concise assistant behavior.
- Domain-specific coding-agent interaction.

Fine-tuning should target a smaller model when possible to reduce inference cost.

## 12. Local Model Strategy

Eventually evaluate local models for:

- intent classification.
- simple routing.
- summarization.
- lightweight tool selection.
- wake/voice-related tasks.

Reserve powerful hosted models for difficult reasoning if needed.

The architecture should support replacing the model without rewriting the assistant.

## 13. Continuous Improvement Loop

Use a feedback pipeline:

`interaction -> log -> evaluate -> label failure -> improve prompt/RAG/tool/model -> regression test`

Track failures such as:

- wrong tool.
- wrong arguments.
- missing context.
- hallucinated file.
- stale information.
- unnecessary LLM call.
- unnecessary confirmation.
- insufficient confirmation.
- failed recovery.

Every fixed failure should become a regression test.

## 14. Training Roadmap

### Stage 1
No custom model training.

Build:
- wake word.
- STT/TTS.
- LLM.
- tools.
- permissions.

### Stage 2
Build:
- RAG.
- project indexing.
- memory.
- tool-call datasets.

### Stage 3
Collect:
- real interaction traces.
- tool-call successes/failures.
- retrieval failures.
- planning failures.

### Stage 4
Evaluate:
- retrieval.
- tool accuracy.
- planning.
- latency.
- token usage.
- safety.

### Stage 5
Experiment with:
- smaller local models.
- fine-tuning for tool calling/behavior.
- local inference.

### Stage 6
Continuously regression-test every change.

## 15. Core Principle

Do not try to make the model memorize the entire computer.

Instead:

`LLM + RAG + Memory + Live State + Tools + Events`

should work together.

The assistant should **retrieve knowledge, query live state, and use tools when needed**, rather than relying on model weights to know or control everything.
