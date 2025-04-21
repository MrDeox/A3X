import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock # For mocking tools
from pathlib import Path
import signal # For graceful shutdown
from typing import List, Tuple, Dict
import importlib # <<< ADD IMPORT >>>

from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
from a3x.fragments.mutator import MutatorFragment
from a3x.fragments.anomaly_detector import AnomalyDetectorFragment
from a3x.fragments.evaluator import EvaluatorFragment
from a3x.fragments.goal_manager import GoalManagerFragment # <<< Import GoalManager >>>
from a3x.fragments.self_evolver import SelfEvolverFragment # <<< Import SelfEvolver >>>
from a3x.fragments.architect import ArchitectFragment # <<< Import Architect >>>
from a3x.fragments.fragment_manager import FragmentManagerFragment # <<< Import FragmentManager >>>
from a3x.fragments.performance_monitor import PerformanceMonitorFragment # <<< Import PerformanceMonitor >>>
from a3x.fragments.deactivator import DeactivatorFragment # <<< Import Deactivator >>>
from a3x.fragments.consolidator import ConsolidatorFragment # <<< Import Consolidator >>>
from a3x.fragments.strategist import StrategistFragment # <<< Import Strategist >>>
from a3x.fragments.reflector import ReflectorFragment, ReflectorFragmentDef # <<< IMPORT ReflectorFragmentDef >>>
from a3x.fragments.planner import PlannerFragment, PlannerFragmentDef # <<< IMPORT PlannerFragmentDef >>>
from a3x.fragments.executor import ExecutorFragment # ExecutorFragmentDef might be in base or defined here
from a3x.fragments.base import FragmentDef, BaseFragment # Import BaseFragment along with FragmentDef
# Import FragmentContext and SharedTaskContext from core
from a3x.core.context import FragmentContext, SharedTaskContext
from a3x.core.tool_registry import ToolRegistry # Import ToolRegistry
# Assume Skill imports would be here in a real scenario, we'll mock them
# from a3x.skills.code_analyzer import CodeAnalyzerSkill
# from a3x.skills.file_manager import FileManagerSkill
# etc.
from a3x.fragments.coordinator_fragment import CoordinatorFragment, CoordinatorFragmentDef # <<< IMPORT NEW FRAGMENT >>>
from a3x.fragments.knowledge_synthesizer import KnowledgeSynthesizerFragment, KnowledgeSynthesizerFragmentDef # <<< IMPORT NEW FRAGMENT >>>
# from a3x.api.realtime_processing import handle_realtime_chat # <<< COMMENT OUT - Module Not Found >>>

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Fragment Definitions (already exists)
# ==========================================
# Hardcoded list of fragments for the test run
fragment_defs = [
    FragmentDef(
        name="StructureAutoRefactor",
        description="Responsible for high-level code structure modifications like creating new modules.",
        fragment_class=StructureAutoRefactorFragment,
        skills=["CodeAnalyzerSkill", "FileManagerSkill", "SandboxSkill", "generate_module_from_directive", "write_file", "execute_python_in_sandbox"], # Include specific tools it needs
    ),
    FragmentDef(
        name="Mutator",
        description="Applies small, random mutations to the codebase to explore variations.",
        fragment_class=MutatorFragment,
        skills=["CodeAnalyzerSkill", "FileManagerSkill", "SandboxSkill", "GitSkill"],
    ),
    FragmentDef(
        name="AnomalyDetector",
        description="Monitors system messages for patterns indicating potential problems.",
        fragment_class=AnomalyDetectorFragment,
        skills=[],
    ),
    FragmentDef(
        name="Evaluator",
        description="Observes system messages and provides rewards for successful actions.",
        fragment_class=EvaluatorFragment,
        skills=[],
    ),
    # <<< Add GoalManagerFragment Definition >>>
    FragmentDef(
        name="GoalManager",
        description="Periodically injects new goals into the system.",
        fragment_class=GoalManagerFragment,
        skills=[], # Does not require active skills
    ),
    # <<< Add SelfEvolverFragment Definition >>>
    FragmentDef(
        name="SelfEvolver",
        description="Observes system behavior and suggests new fragments to improve efficiency or coverage.",
        fragment_class=SelfEvolverFragment,
        category="Meta", # Meta-level fragment
        skills=[], # No direct skills needed, observes messages
    ),
    # <<< Add ArchitectFragment Definition >>>
    FragmentDef(
        name="Architect",
        description="Creates new fragments based on 'create_fragment' directives.",
        fragment_class=ArchitectFragment,
        category="Meta",
        skills=["generate_module_from_directive", "write_file"], # Needs these skills
    ),
    # <<< Add FragmentManagerFragment Definition >>>
    FragmentDef(
        name="FragmentManager",
        description="Listens for 'register_fragment' directives and simulates registration.",
        fragment_class=FragmentManagerFragment,
        category="Meta",
        skills=["FileManagerSkill"], # Needs FileManagerSkill to access read_file
    ),
    # <<< Add PerformanceMonitorFragment Definition >>>
    FragmentDef(
        name="PerformanceMonitor",
        description="Monitors fragment performance and suggests deactivating ineffective ones.",
        fragment_class=PerformanceMonitorFragment,
        category="Meta",
        skills=[], # Observes messages, no direct skills needed
    ),
    # <<< Add DeactivatorFragment Definition >>>
    FragmentDef(
        name="Deactivator",
        description="Handles 'deactivate_fragment' directives to simulate removing fragments.",
        fragment_class=DeactivatorFragment,
        category="Meta",
        skills=[], # No skills needed for simulation
    ),
    # <<< Add ConsolidatorFragment Definition >>>
    FragmentDef(
        name="Consolidator",
        description="Observes successes and consolidates recurring patterns into heuristics.",
        fragment_class=ConsolidatorFragment,
        category="Meta",
        skills=[], # Observes messages, no direct skills needed
    ),
    # <<< Add StrategistFragment Definition >>>
    FragmentDef(
        name="Strategist",
        description="Analyzes learned heuristics and proposes new goals based on success patterns.",
        fragment_class=StrategistFragment,
        category="Meta",
        skills=[], # Observes messages, no direct skills needed
    ),
    # <<< Add ReflectorFragment Definition >>>
    FragmentDef(
        name="Reflector",
        description="Observes system activity and periodically posts reflective summaries.",
        fragment_class=ReflectorFragment,
        category="Meta",
        skills=[], # Observes messages, no direct skills needed
    ),
    # <<< Add PlannerFragment Definition >>>
    FragmentDef(
        name="Planner",
        description="Analyzes reflections and summaries to propose action sequences.",
        fragment_class=PlannerFragment,
        category="Meta",
        skills=[], # Observes messages, no direct skills needed
    ),
    # <<< Add ExecutorFragment Definition >>>
    FragmentDef(
        name="Executor",
        fragment_class=ExecutorFragment,
        description="Executes planned sequences of actions.",
        category="Orchestration",
        skills=[], # Orchestrates via chat, no direct skills needed
    ),
    CoordinatorFragmentDef, # <<< ADD FRAGMENT DEF >>>
    KnowledgeSynthesizerFragmentDef # <<< ADD FRAGMENT DEF >>>
]

# ==========================================
# Central Message Dispatcher (Moved to Module Level)
# ==========================================
async def message_dispatcher(queue: asyncio.Queue, fragment_context: FragmentContext, shared_context: SharedTaskContext):
    """The original message dispatcher loop."""
    logger.info(f"[{__name__}] Starting original message dispatcher loop...")
    while True:
        logger.info(f"[{__name__}] Attempting queue.get()...")
        message = await queue.get() # Blocks until a message is available
        logger.info(f"[{__name__}] Got message from queue: {message.get('type', 'N/A')}")
        if message.get("type") == "STOP_DISPATCHER":
            logger.info(f"[{__name__}] STOP_DISPATCHER received. Exiting loop.")
            queue.task_done()
            break

        # <<< COMMENT OUT SECTION HANDLING 'internal_chat' via handle_realtime_chat >>>
        # if message.get("type") == "internal_chat":
        #     logger.info(f"[{__name__}] Dispatching internal_chat to handle_realtime_chat (original behavior)")
        #     # Assuming handle_realtime_chat is imported correctly at module level
        #     asyncio.create_task(handle_realtime_chat(message['data'], fragment_context, shared_context))
        # elif message.get("type") == "tool_response":
        
        # <<< ADJUST elif after commenting out previous block >>>
        if message.get("type") == "tool_response":
             logger.info(f"[{__name__}] Dispatching tool_response (original behavior)")
             response_id = message.get("response_id")
             if response_id and response_id in shared_context.tool_response_futures:
                 shared_context.tool_response_futures[response_id].set_result(message.get("data"))
                 del shared_context.tool_response_futures[response_id]
             else:
                 logger.warning(f"[{__name__}] Received tool response for unknown/expired id: {response_id}")

        # Add handling for other message types if necessary
        elif message.get("type") == "test_message": # <<< ADD HANDLING for isolated test message
            logger.info(f"[{__name__}] Received test message: {message.get('content')}")
            # No further action needed for this test message

        else:
            logger.warning(f"[{__name__}] Unknown message type received: {message.get('type')}")

        queue.task_done()
    logger.info(f"[{__name__}] Message dispatcher loop finished.")

# ==========================================
# Main Execution Logic
# ==========================================
async def main():
    logger.info("Starting Continuous Emergent Test Runner...")
    task_id = f"emergent_test_{uuid.uuid4()}"
    shared_context = SharedTaskContext(task_id=task_id)
    tool_registry = ToolRegistry()

    # --- Mock and Register Skill Instances ---
    logger.info("Setting up mock skill instances...")
    required_skills = set()
    for frag_def in fragment_defs:
        required_skills.update(frag_def.skills)

    mock_tools = {}
    registered_skill_instances = set()
    for skill_name in required_skills:
        # Skip if it's a specific tool we register later
        if skill_name in ["generate_module_from_directive", "write_file", "execute_python_in_sandbox"]:
            continue 
        # Avoid re-registering if multiple fragments list the same base skill
        if skill_name in registered_skill_instances:
            continue
            
        mock_skill_instance = AsyncMock(name=skill_name)
        # Assign mock methods to the instance
        if skill_name == "SandboxSkill":
             mock_skill_instance.execute_code_in_sandbox = AsyncMock(return_value={"status": "success", "output": "Mock execution successful", "exit_code": 0})
        elif skill_name == "FileManagerSkill":
             mock_skill_instance.read_file = AsyncMock(return_value="mock file content")
             mock_skill_instance.write_file = AsyncMock(return_value=True)
             mock_skill_instance.create_directory = AsyncMock(return_value=True)
        elif skill_name == "CodeAnalyzerSkill":
             mock_skill_instance.analyze_code = AsyncMock(return_value={"dependencies": [], "complexity": 1})
             mock_skill_instance.find_definitions = AsyncMock(return_value=[])
        elif skill_name == "GitSkill":
             mock_skill_instance.get_changed_files = AsyncMock(return_value=["mock/file.py"])
             mock_skill_instance.get_file_diff = AsyncMock(return_value="mock diff")
        
        mock_tools[skill_name] = mock_skill_instance
        mock_schema_for_instance = {
            "name": skill_name,
            "description": f"Mock schema for Skill Class {skill_name}",
            "parameters": {}
        }
        try:
            # Register the instance itself under the skill name
            tool_registry.register_tool(skill_name, mock_skill_instance, mock_skill_instance, mock_schema_for_instance)
            logger.info(f"Registered mock skill instance: '{skill_name}'")
            registered_skill_instances.add(skill_name)
        except Exception as e:
             logger.error(f"Failed to register mock skill instance '{skill_name}': {e}")
             raise

    # --- Register Specific Mock Tools Needed Directly ---
    logger.info("Registering specific mock tools needed directly by fragments...")
    # (Keep existing specific tool mocks: generate_module_from_directive, write_file, execute_python_in_sandbox)
    generate_mock = AsyncMock(name="generate_module_from_directive_mock", return_value={
        "status": "success",
        "data": { "code": "# Mock generated code\ndef add(a, b):\n    return a + b" }
    })
    generate_schema = { "name": "generate_module_from_directive", "description": "...", "parameters": {...} } # Keep schema details
    tool_registry.register_tool("generate_module_from_directive", None, generate_mock, generate_schema)
    logger.info("Registered specific tool: generate_module_from_directive")

    write_mock = AsyncMock(name="write_file_mock", return_value={ "status": "success" })
    write_schema = { "name": "write_file", "description": "...", "parameters": {...} } # Keep schema details
    tool_registry.register_tool("write_file", None, write_mock, write_schema)
    logger.info("Registered specific tool: write_file")

    sandbox_mock = AsyncMock(name="execute_python_in_sandbox_mock", return_value={ "status": "success", "output": "...", "exit_code": 0 })
    sandbox_schema = { "name": "execute_python_in_sandbox", "description": "...", "parameters": {...} } # Keep schema details
    tool_registry.register_tool("execute_python_in_sandbox", None, sandbox_mock, sandbox_schema)
    logger.info("Registered specific tool: execute_python_in_sandbox")

    # <<< ADD Mock for modify_code used by Mutator >>>
    async def modify_code_mock(original_code: str, instructions: str, **kwargs) -> Dict:
        # Simple mock: Add a comment indicating modification
        modified = f"# Code modified by Mutator based on: {instructions[:50]}...\n{original_code}"
        return {
            "status": "success",
            "data": {"modified_code": modified}
        }
    modify_schema = {
        "name": "modify_code",
        "description": "Mock modify_code skill. Returns slightly altered code.",
        "parameters": {
            "type": "object",
            "properties": {
                "original_code": {"type": "string"},
                "instructions": {"type": "string"},
                "file_path": {"type": "string"} # Optional param
            },
            "required": ["original_code", "instructions"]
        }
    }
    tool_registry.register_tool(
        name="modify_code",
        instance=None, # Standalone mock function
        tool=modify_code_mock, # Use the async def mock function
        schema=modify_schema
    )
    logger.info("Registered specific tool: modify_code")
    # <<< END ADD >>>

    # --- ADD Mock Research Skill (Initially Failing) ---
    # mock_research_state = {"fail": True} # State to control mock behavior -- REMOVED, logic is now topic-based

    async def research_topic_mock(topic: str, **kwargs) -> Dict:
        # Specific failure for the first topic
        if topic == "digital marketing basics":
            logger.warning(f"[Mock Research] Simulating FAILURE for specific topic: {topic}")
            return {
                "status": "error",
                "reason": "Simulated research API timeout",
                "details": f"Could not retrieve information for initial topic: {topic}"
            }
        # Success for other topics (like the adaptively chosen 'python basics')
        else:
            logger.info(f"[Mock Research] Simulating SUCCESS for topic: {topic}")
            # Provide more generic success data relevant to any topic
            return {
                "status": "success",
                "data": {
                    "summary": f"Successfully retrieved basic information about {topic}. Key aspects include A, B, and C.",
                    "key_points": [
                        f"Key point 1 about {topic}",
                        f"Key point 2 about {topic}",
                        f"Key point 3 about {topic}"
                    ],
                    "source": "mock_research_api"
                }
            }
            
    research_schema = {
        "name": "research_topic",
        "description": "Mock skill to research a topic. Fails for 'digital marketing basics', succeeds otherwise.", # Updated description
        "parameters": {
            "type": "object",
            "properties": { "topic": {"type": "string"} },
            "required": ["topic"]
        }
    }
    tool_registry.register_tool("research_topic", None, research_topic_mock, research_schema)
    logger.info("Registered mock skill: research_topic (set to fail initially)")
    # --- END Mock Research Skill ---

    # --- Instantiate Fragments ---
    logger.info("Instantiating fragments...")
    fragments: List[BaseFragment] = []
    fragment_map: Dict[str, BaseFragment] = {}

    # <<< REVERTED INSTANTIATION LOGIC >>>
    for fragment_def in fragment_defs:
        try:
            FragmentClass = fragment_def.fragment_class # Get the actual class
            
            # <<< ADD DEBUG LOGGING >>>
            logger.debug(f"Attempting to instantiate '{fragment_def.name}': class={FragmentClass}, type={type(FragmentClass)}")
            # <<< END DEBUG LOGGING >>>

            # Check if the class requires tool_registry
            if fragment_def.name in ["StructureAutoRefactor", "Mutator", "Architect", "FragmentManager"]:
                 fragment_instance = FragmentClass(fragment_def=fragment_def, tool_registry=tool_registry)
            else:
                 fragment_instance = FragmentClass(fragment_def=fragment_def)
                 
            fragments.append(fragment_instance)
            fragment_map[fragment_instance.get_name()] = fragment_instance
            logger.info(f"Instantiated Fragment: {fragment_instance.get_name()}")

        except Exception as e:
            logger.error(f"Failed to instantiate fragment {fragment_def.name}: {e}", exc_info=True)
    # <<< END REVERTED INSTANTIATION LOGIC >>>

    # Check if essential fragments were loaded (e.g., Executor, Coordinator)
    if "Executor" not in fragment_map:
        logger.critical("ExecutorFragment failed to instantiate. Cannot run test plan.")
        return
    if "Coordinator" not in fragment_map:
        logger.warning("CoordinatorFragment failed to instantiate. Failure coordination disabled.")
        
    # Create the shared context (using the actual class now)
    # ... (existing context creation)

    # Set context for all instantiated fragments
    logger.info("Setting context for all fragments...")
    fragments_with_context: List[Tuple[BaseFragment, FragmentContext]] = [] # Ensure list is initialized
    for fragment in fragments:
        ctx = shared_context # Each fragment gets the shared context
        try:
            fragment.set_context(ctx)
            await asyncio.sleep(0) # <<< ADDED: Yield control to event loop >>>
            fragments_with_context.append((fragment, ctx)) # Populate the list
        except Exception as e:
            logger.error(f"Error setting context for fragment {fragment.get_name()}: {e}", exc_info=True)

    # --- Start Background Fragment Tasks ---
    # Start background tasks for fragments that need them
    background_tasks = []
    logger.info("Starting background fragment tasks...")
    # PerformanceMonitor starts its own loop internally via set_context
    # FragmentManager doesn't have the specified loop
    # GoalManager is intentionally disabled for this test
    # Coordinator starts its own loop internally via set_context / start()
    # Only add tasks here for fragments that DON'T start their own loop.

    # (Currently, no fragments need manual task starting here)

    # --- Start Central Message Dispatcher (uses module-level function) ---
    logger.info("Starting central message dispatcher...")
    logger.info("Starting the original message dispatcher task...")
    # Now directly call the module-level message_dispatcher
    dispatcher_task = asyncio.create_task(message_dispatcher(shared_context.internal_chat_queue, shared_context, shared_context)) # Use original

    # --- Inject Initial Plan (Learning Cycle Test) ---
    logger.warning("Injecting LEARNING CYCLE test plan (Step 1 designed to fail)...")
    
    # <<< ADD SMALL DELAY >>>
    await asyncio.sleep(0.1) # Give the dispatcher a moment to start properly
    # <<< END SMALL DELAY >>>
    
    learning_plan_message = {
      "type": "plan_sequence", # Assuming Executor handles this type
      "sender": "TestHarness",
      "content": {
          "plan_id": f"learn_digital_marketing_{str(uuid.uuid4())[:8]}",
          "objective": "Learn digital marketing basics", # Added objective for context
          "actions": [
            {
                # <<< NEW DIRECTIVE: Learn from Professor LLM >>>
                "type": "directive",
                "action": "learn_from_professor",
                "professor_id": "prof_learn_test",
                "context_fragment_id": "learner_ctx", # Assuming Executor or Professor knows how to interpret this
                "question": "Como posso aprender fundamentos de marketing digital?",
                "target": "ExecutorFragment", # Explicitly target Executor
                "message": "Initiate learning cycle with professor LLM."
            },
            {
              # Action for the Executor to execute using a skill
              "type": "directive",
              "action": "execute_skill", # Executor needs to know how to handle this
              "skill": "research_topic", # The skill to call
              "parameters": {           # Parameters for the skill
                  "topic": "digital marketing basics"
              },
              "message": "Execute first step: Research digital marketing basics." # Descriptive message
            },
            {
              # Placeholder for a subsequent action (e.g., summarize research)
              "type": "directive",
              "action": "summarize_research", # Assumes another skill/fragment handles this
              "target": "research_summary.md",
              "message": "Summarize the findings from the initial research."
            }
          ]
      }
    }
    await shared_context.internal_chat_queue.put(learning_plan_message)
    logger.info("LEARNING CYCLE test plan injected.")
    # --- End Test Injection ---

    # --- Run Indefinitely & Handle Shutdown ---
    logger.info("Runner started. Fragments are active. Press Ctrl+C to stop.")
    stop_event = asyncio.Event()

    def handle_sigint(*args):
        logger.warning("SIGINT received, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutdown sequence initiated...")
        # --- Cleanup ---
        # Cancel the dispatcher task
        logger.info("Cancelling message dispatcher task...")
        if dispatcher_task and not dispatcher_task.done():
            dispatcher_task.cancel()
            await asyncio.gather(dispatcher_task, return_exceptions=True) # Wait for it
            logger.info("Message dispatcher task finished gathering.")
        
        # Call shutdown on fragments (GoalManager's internal loop needs this)
        logger.info("Calling fragment shutdown methods...")
        shutdown_tasks = []
        for frag_instance in fragments:
            if hasattr(frag_instance, 'shutdown'):
                logger.debug(f"Calling shutdown for {frag_instance.get_name()}")
                shutdown_tasks.append(asyncio.create_task(frag_instance.shutdown()))
        if shutdown_tasks:
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            logger.info("Fragment shutdown methods finished.")
            # Log any errors during fragment shutdown
            for i, result in enumerate(results):
                 if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                     # Find corresponding fragment instance if possible (assuming order is maintained)
                     frag_name = fragments[i].get_name() if i < len(fragments) else "UnknownFragment"
                     logger.error(f"Error during shutdown of {frag_name}: {result}")

        logger.info("Emergent Test Runner finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Runner stopped by KeyboardInterrupt.")

# Ensure schema placeholders are filled/removed if copying
# Example adjustment for generate_schema:
generate_schema = {
    "name": "generate_module_from_directive", 
    "description": "Mock skill to generate a module from a directive.", 
    "parameters": {
        "type": "object", 
        "properties": {
            "message": {"type": "string"},
            "target_path": {"type": "string"}
        },
        "required": ["message", "target_path"]
    }
}
# Apply similar fixes for write_schema and sandbox_schema if placeholders were used.

write_schema = { 
    "name": "write_file", 
    "description": "Mock write_file skill.", 
    "parameters": {
        "type": "object", 
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["file_path", "content"]
    }
}
sandbox_schema = { 
    "name": "execute_python_in_sandbox", 
    "description": "Mock execute_python_in_sandbox skill.", 
    "parameters": {
        "type": "object", 
        "properties": {
            "code": {"type": "string"},
            "file_path_to_execute": {"type": "string"} 
        },
        "required": [] 
    }
} 