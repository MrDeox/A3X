# Design Notes: Agent Evolution and Inter-Fragment Communication

## Goal

To establish a robust framework that supports the exponential evolutionary growth of the A³X agent, particularly by enabling richer forms of communication and collaboration between its internal components (Fragments).

## Core Principles

*   **Incremental Evolution:** Introduce new capabilities gradually, building upon existing mechanisms.
*   **Observability & Learnability:** Ensure that new interaction patterns can be logged, reflected upon, and learned from by the agent's meta-learning cycles.
*   **Maintain Modularity (Initially):** Avoid tight coupling between fragments early on. Prefer mediated communication or shared context over direct calls initially.
*   **Emergence through Structure:** Provide structured mechanisms (like the Shared Task Context) that enable, rather than dictate, complex collaborative behaviors.

## Implemented Mechanisms

### 1. Shared Task Context (`a3x.core.context.SharedTaskContext`)

*   **Purpose:** Acts as a transient "whiteboard" for a single task execution (`run_task`). Allows fragments to share structured data, intermediate results, and status updates indirectly.
*   **Structure:** Stores data as `ContextEntry` objects, each containing:
    *   `value`: The actual data.
    *   `source`: Identifier of the fragment/skill providing the data.
    *   `timestamp`: Time of creation/update.
    *   `tags`: List of strings for categorization.
    *   `metadata`: Dictionary for arbitrary metadata (e.g., `{ "priority": 0.8, "status": "provisional" }`).
*   **Access:** Passed via `_ToolExecutionContext` to skills and fragments. Provides methods like `set()`, `get()`, `get_entry()`, `get_by_tag()`, `get_by_source()`.
*   **Evolutionary Potential:**
    *   Fragments learn *what* information is useful to share/retrieve via the context.
    *   Fragments learn to interpret/use `tags` and `metadata`.
    *   The reflection cycle analyzes context usage patterns to identify effective collaboration strategies.

### 2. Debugger Fragment (`a3x.fragments.debugger.DebuggerFragment`)

*   **Purpose:** Analyzes persistent *failures* in sub-tasks delegated by the Orchestrator. Uses `llm_error_diagnosis` skill with failure history.
*   **Trigger:** Invoked by the Orchestrator (`run_task`) after a configurable number of consecutive failures for the same sub-task.
*   **Output:** Provides a diagnosis and suggested actions.
*   **Evolutionary Potential:**
    *   Learns to provide better diagnoses over time.
    *   Orchestrator learns how to effectively use the debugger's suggestions (e.g., retry with modification, change fragment, abandon sub-task).

## Future Evolutionary Paths (Ideas Discussed)

These build upon the `SharedTaskContext` and reflection mechanisms:

1.  **"Attention" via Metadata:** Standardize specific keys within `ContextEntry.metadata` (e.g., `priority`, `confidence`, `requires_review`). Fragments and the Orchestrator learn to act on these signals.
2.  **"Hypothesis Board":** Designate a specific key in `SharedTaskContext` (e.g., `_hypotheses: List[Dict]`) where fragments can post structured hypotheses (plan snippets, potential solutions). Other fragments or the Orchestrator can interact with these (validate, execute, refine). Requires standardization of the hypothesis structure.
3.  **Service Discovery via Context:** Fragments could register temporary capabilities by setting specific keys (e.g., `context.set("_service:image_analyzer", True, source="VisionFragment")`). Other fragments check the context before defaulting to Orchestrator delegation for that capability.
4.  **Resource Management via Context:** Orchestrator sets initial resource limits (e.g., `_limits: {"api_calls": 10, "max_tokens": 50000}`). Skills/fragments check and decrement these values in the context. Requires skills to be adapted.
5.  **Enhanced Reflection on Context:** Explicitly train the reflection models to identify patterns in `SharedTaskContext` usage (e.g., "Data tagged 'critical' from Fragment X often leads to success when used by Fragment Y").
6.  **State Tagging:** Use `ContextEntry.tags` more formally for high-level task state (e.g., `data_ingested`, `analysis_complete`, `code_generated`, `validation_failed`). Orchestrator uses these tags for better situational awareness.
7.  **Mediated Communication (Orchestrator as Router):** Introduce a skill like `request_fragment_communication(target_fragment, message_content)`. The Orchestrator intercepts this, validates, and potentially forwards the message or incorporates it into the next sub-task delegation for the target fragment. Maintains central control while allowing more direct intent expression.

## Next Steps

1.  Integrate analysis of the `SharedTaskContext` (specifically the metadata and tags) into the post-task reflection cycle.
2.  Adapt key skills/fragments to demonstrate reading from and writing to the `SharedTaskContext` with relevant metadata/tags.
3.  Enhance the Orchestrator's delegation logic (`_get_next_step_delegation`) to incorporate information from the `SharedTaskContext`.
4.  Begin implementing one of the "Future Evolutionary Paths" (e.g., Hypothesis Board or Attention via Metadata) once the base context usage is stable and understood through reflection. 