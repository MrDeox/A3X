import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- A³X Core Imports ---
try:
    from a3x.core.tool_registry import ToolRegistry
    from a3x.core.context import FragmentContext, SharedTaskContext # Base context
    from a3x.core.llm_interface import LLMInterface # For type hinting
    from a3x.fragments.base import FragmentDef, BaseFragment
    from a3x.fragments.registry import FragmentRegistry # For type hinting or mocking
    # Import Fragments to test
    from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
    from a3x.fragments.mutator import MutatorFragment
    from a3x.fragments.anomaly_detector import AnomalyDetectorFragment
    # Assume PROJECT_ROOT is available via config or determined path
    from a3x.core.config import PROJECT_ROOT
except ImportError as e:
    logger.error(f"Failed to import A³X components: {e}. Ensure PYTHONPATH is set or run from root.")
    pytest.skip("Skipping tests due to import errors.", allow_module_level=True)

# --- Test Message Queue --- #
# Simple asyncio queue for inter-fragment communication simulation
@pytest.fixture
def message_queue() -> asyncio.Queue:
    return asyncio.Queue()

# --- Patched Test Context --- #
# We need a context that fragments can use to post messages to our queue.
# Instead of a full mock, let's slightly modify the EmergentTestFragmentContext
# from the old runner, or create a similar one.

class IntegratedTestFragmentContext:
    """Context for integration tests using a shared message queue."""
    logger: logging.Logger
    llm_interface: LLMInterface
    tool_registry: ToolRegistry
    fragment_registry: FragmentRegistry
    shared_task_context: SharedTaskContext
    workspace_root: Path
    message_queue: asyncio.Queue # The shared queue
    # Store which fragment instance this context belongs to, needed for sender in post_chat_message
    _fragment_instance: Optional[BaseFragment] = None 

    def __init__(self,
                 logger: logging.Logger,
                 llm_interface: LLMInterface,
                 tool_registry: ToolRegistry,
                 fragment_registry: FragmentRegistry,
                 shared_task_context: SharedTaskContext,
                 workspace_root: Path,
                 message_queue: asyncio.Queue):
        self.logger = logger
        self.llm_interface = llm_interface
        self.tool_registry = tool_registry
        self.fragment_registry = fragment_registry
        self.shared_task_context = shared_task_context
        self.workspace_root = workspace_root
        self.message_queue = message_queue
        self.logger.debug("IntegratedTestFragmentContext initialized.")

    def set_fragment_instance(self, fragment: BaseFragment):
        """Sets the fragment instance associated with this context."""
        self._fragment_instance = fragment

    async def post_chat_message(
        self,
        # Parameters as expected by BaseFragment.post_chat_message might vary slightly
        # Let's assume it takes context, message_type, content, target_fragment
        # The actual call from fragment is often: `await self.post_chat_message(context=self_context, ...)`
        # But the context object itself might be simpler. Adapting...
        # Let's align with the signature used in the fragment tests context:
        message_type: str,
        content: Any,
        # We get the sender from the _fragment_instance set for this context
        # sender: Optional[str] = None, 
        target_fragment: Optional[str] = None,
    ):
        """Puts a message onto the shared asyncio queue."""
        sender_name = self._fragment_instance.get_name() if self._fragment_instance else "UnknownContextSender"
        message = {
            "type": message_type,
            "sender": sender_name,
            "content": content,
            "task_id": self.shared_task_context.task_id,
            "target": target_fragment,
            "timestamp": asyncio.get_event_loop().time()
        }
        self.logger.debug(f"Context (for {sender_name}) queueing message: Type='{message_type}', Target='{target_fragment}'")
        await self.message_queue.put(message)

# --- Pytest Fixtures (Integration Scope) ---

@pytest.fixture
def mock_llm_interface() -> MagicMock:
    mock = MagicMock(spec=LLMInterface)
    mock.call_llm = AsyncMock(return_value=iter(["mock llm response"]))
    return mock

@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()

@pytest.fixture
def fragment_registry() -> MagicMock:
    return MagicMock(spec=FragmentRegistry)

@pytest.fixture
def shared_task_context() -> SharedTaskContext:
    return SharedTaskContext(
        task_id='integration_test_001',
        initial_objective="Test emergent error correction cycle."
    )

@pytest.fixture
def workspace_root(tmp_path) -> Path:
    (tmp_path / "a3x" / "modules" / "temp").mkdir(parents=True, exist_ok=True)
    # Patch PROJECT_ROOT used within fragments if they import it directly
    # Note: Need to patch for each fragment module if they import separately
    patches = [
        patch('a3x.fragments.structure_auto_refactor.PROJECT_ROOT', str(tmp_path)),
        patch('a3x.fragments.mutator.PROJECT_ROOT', str(tmp_path), create=True), # Use create=True if not present
        patch('a3x.fragments.anomaly_detector.PROJECT_ROOT', str(tmp_path), create=True)
    ]
    for p in patches:
        p.start()
    yield tmp_path
    for p in patches:
        p.stop()

# Fixture for the base context components
@pytest.fixture
def base_context_args(mock_llm_interface, tool_registry, fragment_registry, shared_task_context, workspace_root, message_queue):
    return {
        "logger": logging.getLogger("IntegrationContext"),
        "llm_interface": mock_llm_interface,
        "tool_registry": tool_registry,
        "fragment_registry": fragment_registry,
        "shared_task_context": shared_task_context,
        "workspace_root": workspace_root,
        "message_queue": message_queue
    }

# Fixture for creating the specific context instance
@pytest.fixture
def integration_context(base_context_args) -> IntegratedTestFragmentContext:
    return IntegratedTestFragmentContext(**base_context_args)


# --- Mock Skills Fixture (Integration Scope) ---
@pytest.fixture
def mock_skills(tool_registry): # Register in the shared tool_registry
    mocks = {
        "generate_module_from_directive": AsyncMock(name="generate_module_from_directive"),
        "write_file": AsyncMock(name="write_file"),
        "execute_python_in_sandbox": AsyncMock(name="execute_python_in_sandbox"),
        "read_file": AsyncMock(name="read_file"),
        "modify_code": AsyncMock(name="modify_code"),
        "learn_from_correction_result": AsyncMock(name="learn_from_correction_result"),
    }
    schemas = {
        "generate_module_from_directive": {"name": "generate_module_from_directive", "description": "mock", "parameters": {}},
        "write_file": {"name": "write_file", "description": "mock", "parameters": {}},
        "execute_python_in_sandbox": {"name": "execute_python_in_sandbox", "description": "mock", "parameters": {}},
        "read_file": {"name": "read_file", "description": "mock", "parameters": {}},
        "modify_code": {"name": "modify_code", "description": "mock", "parameters": {}},
        "learn_from_correction_result": {"name": "learn_from_correction_result", "description": "mock", "parameters": {}},
    }
    for name, mock_func in mocks.items():
        # Default to success, configure specifics in the test
        if name == "generate_module_from_directive":
            mock_func.return_value = {"status": "success", "content": "# Default mock code", "path": "default_mock_path.py"}
        elif name == "learn_from_correction_result":
            mock_func.return_value = {"status": "skipped"}
        else:
            mock_func.return_value = {"status": "success", "exit_code": 0}
        tool_registry.register_tool(name=name, instance=None, tool=mock_func, schema=schemas[name])
    logger.debug(f"Registered integration mock skills: {list(mocks.keys())}")
    return mocks

# --- Fragment Fixtures (Integration Scope) ---
# Use the shared tool_registry containing mocks

@pytest.fixture
def structure_fragment(tool_registry) -> StructureAutoRefactorFragment:
    metadata = {"name": "StructureAutoRefactor", "description": "Integration Test", "skills": ["generate_module_from_directive", "write_file", "execute_python_in_sandbox", "read_file", "modify_code", "learn_from_correction_result"]}
    frag_def = FragmentDef(**metadata, fragment_class=StructureAutoRefactorFragment)
    fragment = StructureAutoRefactorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    return fragment

@pytest.fixture
def mutator_fragment(tool_registry) -> MutatorFragment:
    metadata = {"name": "Mutator", "description": "Integration Test", "skills": ["read_file", "modify_code", "write_file"]}
    frag_def = FragmentDef(**metadata, fragment_class=MutatorFragment)
    fragment = MutatorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    return fragment

@pytest.fixture
def anomaly_fragment(tool_registry) -> AnomalyDetectorFragment:
    metadata = {"name": "AnomalyDetector", "description": "Integration Test", "skills": []}
    frag_def = FragmentDef(**metadata, fragment_class=AnomalyDetectorFragment)
    fragment = AnomalyDetectorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    return fragment

# --- Test Helper --- #
async def process_queue_until_empty(queue: asyncio.Queue, handlers: Dict[str, List[BaseFragment]], context: IntegratedTestFragmentContext, timeout=5.0):
    """Processes messages from the queue, routing them to fragment handlers."""
    start_time = asyncio.get_event_loop().time()
    processed_count = 0
    while True:
        try:
            # Wait for a message with a timeout
            message = await asyncio.wait_for(queue.get(), timeout=0.1) 
            logger.debug(f"Queue Processor: Got message type '{message.get('type')}' from '{message.get('sender')}'")
            processed_count += 1
            # Find relevant handlers (all fragments listen in this simple setup)
            # A real system might have targeted delivery or topic subscriptions
            active_fragments = handlers.get("all", [])
            for fragment in active_fragments:
                 # Provide the context specific to this fragment
                 context.set_fragment_instance(fragment) # Let context know who is posting
                 # Patch post_chat_message for this fragment instance to use the integration context
                 # This ensures messages go back to the queue
                 with patch.object(fragment, 'post_chat_message', new=context.post_chat_message):
                    if hasattr(fragment, 'handle_realtime_chat'):
                         logger.debug(f"Routing message to {fragment.get_name()}.handle_realtime_chat")
                         # Pass the *integration* context
                         await fragment.handle_realtime_chat(message, context)
                    else:
                         logger.warning(f"Fragment {fragment.get_name()} has no handle_realtime_chat method.")
            queue.task_done()
            start_time = asyncio.get_event_loop().time() # Reset timeout if message processed
        except asyncio.TimeoutError:
            logger.debug("Queue processing timed out (queue likely empty).")
            break # Queue is empty or timeout reached
        except Exception as e:
            logger.exception(f"Error processing message queue: {e}")
            break # Stop processing on error
        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning(f"Queue processing exceeded global timeout ({timeout}s). Processed {processed_count} messages.")
            break
    logger.info(f"Finished queue processing. Processed {processed_count} messages.")

# --- Integration Test Case --- #

@pytest.mark.asyncio
async def test_emergent_correction_cycle(
    structure_fragment: StructureAutoRefactorFragment,
    mutator_fragment: MutatorFragment,
    anomaly_fragment: AnomalyDetectorFragment,
    integration_context: IntegratedTestFragmentContext,
    message_queue: asyncio.Queue,
    mock_skills: Dict[str, AsyncMock],
    workspace_root: Path # Needed for path generation
):
    """Tests the full cycle: create->fail->correct->succeed with multiple fragments."""
    target_path_rel = "a3x/modules/temp/emergent_fail_correct.py"
    target_path_abs = workspace_root / target_path_rel
    directive_message = "Create emergent test module (will fail first)."
    initial_directive_content = {
        "type": "directive",
        "action": "create_helper_module",
        "target": target_path_rel,
        "message": directive_message
    }
    initial_suggestion_message = {
        "type": "architecture_suggestion",
        "sender": "IntegrationTestRunner",
        "content": initial_directive_content,
        "task_id": integration_context.shared_task_context.task_id,
        "timestamp": asyncio.get_event_loop().time()
    }

    # Configure mock skills for the flow:
    # 1. Generate fails
    failing_code = "print(some_undefined_variable)"
    sandbox_error = "NameError: name 'some_undefined_variable' is not defined"
    mock_skills["generate_module_from_directive"].return_value = {"status": "success", "content": failing_code, "path": str(target_path_abs)}
    # 2. Write succeeds
    mock_skills["write_file"].return_value = {"status": "success"}
    # 3. First sandbox run FAILS
    mock_skills["execute_python_in_sandbox"].side_effect = [
        # First call (after create_helper_module) fails
        {"status": "error", "exit_code": 1, "stderr": sandbox_error},
        # Second call (after successful refactor_module) succeeds
        {"status": "success", "exit_code": 0, "stdout": "Corrected!", "stderr": ""}
    ]
    # 4. Read file succeeds (for refactor)
    mock_skills["read_file"].return_value = {"status": "success", "data": {"content": failing_code}}
    # 5. Modify code succeeds
    corrected_code = "print('Corrected!')"
    mock_skills["modify_code"].return_value = {"status": "success", "data": {"modified_code": corrected_code}}
    # 6. Learning can be skipped
    mock_skills["learn_from_correction_result"].return_value = {"status": "skipped"}

    # --- Act ---
    # Put initial message in queue
    await message_queue.put(initial_suggestion_message)

    # Process the queue
    # Need to pass fragments to the processor so it knows who can handle messages
    all_fragments = [structure_fragment, mutator_fragment, anomaly_fragment]
    handlers = {"all": all_fragments} # Simple broadcast

    await process_queue_until_empty(message_queue, handlers, integration_context, timeout=10.0)

    # --- Assert --- 
    # Check key skill calls happened in order
    call_list = [
        call("generate_module_from_directive", ANY, ANY), # ANY for args dict for simplicity
        call("write_file", ANY, ANY), 
        call("execute_python_in_sandbox", ANY, ANY), # First (failing) call
        # Correction cycle starts
        call("read_file", ANY, ANY),
        call("modify_code", ANY, ANY),
        call("write_file", ANY, ANY), # Overwrite with corrected code
        call("execute_python_in_sandbox", ANY, ANY), # Second (succeeding) call
        call("learn_from_correction_result", ANY, ANY)
    ]
    # This requires more precise mocking/spying on the ToolRegistry execution if possible
    # For now, check if they were called at least once in the expected phases
    mock_skills["generate_module_from_directive"].assert_called_once()
    # Write might be called twice (initial create, then overwrite during correct)
    assert mock_skills["write_file"].call_count >= 2
    assert mock_skills["execute_python_in_sandbox"].call_count == 2
    mock_skills["read_file"].assert_called_once()
    mock_skills["modify_code"].assert_called_once()
    mock_skills["learn_from_correction_result"].assert_called_once()

    # Check messages in the context (which simulates the queue output)
    # Note: process_queue_until_empty doesn't store messages itself,
    # the context mock passed to fragments does.
    captured_messages = integration_context.posted_messages

    # Expected messages (approximate order):
    # 1. refactor_result (error from initial create failure)
    # 2. architecture_suggestion (correction directive from StructureAutoRefactor)
    # 3. reward (from StructureAutoRefactor for successful correction)
    # 4. refactor_result (success from correction)
    # Might also include messages from AnomalyDetector or Mutator if triggered

    assert len(captured_messages) >= 4, f"Expected at least 4 messages, got {len(captured_messages)}"

    # Find specific messages
    error_result = find_message(captured_messages, "refactor_result", lambda m: m['content']['status'] == 'error')
    correction_suggestion = find_message(captured_messages, "architecture_suggestion")
    reward = find_message(captured_messages, "reward")
    success_result = find_message(captured_messages, "refactor_result", lambda m: m['content']['status'] == 'success')

    assert error_result is not None
    assert error_result["sender"] == structure_fragment.get_name()
    assert error_result["content"]["original_action"] == "create_helper_module"
    assert sandbox_error in error_result["content"]["details"]

    assert correction_suggestion is not None
    assert correction_suggestion["sender"] == structure_fragment.get_name()
    assert correction_suggestion["content"]["action"] == "refactor_module"
    assert sandbox_error in correction_suggestion["content"]["message"]

    assert reward is not None
    assert reward["sender"] == structure_fragment.get_name()
    assert reward["content"]["target"] == structure_fragment.get_name()

    assert success_result is not None
    assert success_result["sender"] == structure_fragment.get_name()
    assert success_result["content"]["original_action"] == "refactor_module"
    assert "Successfully corrected" in success_result["content"]["summary"]

# Helper refinement for find_message
def find_message(messages: List[Dict], msg_type: str, condition: Optional[callable] = None) -> Optional[Dict]:
    """Helper to find the first message of a specific type matching an optional condition."""
    for msg in messages:
        if msg.get("type") == msg_type:
            if condition is None or condition(msg):
                return msg
    return None 