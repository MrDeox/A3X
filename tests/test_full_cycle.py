import pytest
import asyncio
import sqlite3
import os
import uuid
import json
from unittest.mock import patch, AsyncMock, MagicMock, call
from typing import List, Dict, AsyncGenerator
from pathlib import Path
import logging
import time # Import time for potential delays/checks
import inspect # Added inspect import
import shutil

# Imports from project
from a3x.core.agent import ReactAgent
from a3x.core.llm_interface import LLMInterface
from a3x.core.db_utils import (
    initialize_database, 
    load_agent_state, 
    get_db_connection, 
    save_agent_state, # May be needed for setup/verification
    DATABASE_PATH, # Import to potentially patch
    close_db_connection,
)
from a3x.core.tool_executor import _ToolExecutionContext # For potential mocking
from a3x.core.tool_registry import ToolRegistry
from a3x.core.orchestrator import TaskOrchestrator
from a3x.fragments.registry import FragmentRegistry
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.fragments.registry import fragment
from a3x.core.skills import skill
from a3x.skills.file_manager import FileManagerSkill
from a3x.fragments.file_manager_fragment import FileOpsManager # Import the real manager
# from a3x.core.memory import EpisodicMemory, VectorMemory, SimpleVectorDB, BaseMemory # <<< REMOVE THIS LINE
from a3x.core.context import SharedTaskContext # Import SharedTaskContext
from a3x.fragments.basic_fragments import FinalAnswerProvider, PlannerFragment
from a3x.skills.final_answer import final_answer
from a3x.skills.planning import hierarchical_planner
from a3x.core.config import PROJECT_ROOT
from a3x.core.skill_management import SKILL_REGISTRY # <<< IMPORT the global registry dict

# Configure logger
logger = logging.getLogger(__name__)

# Fixture for temporary database
@pytest.fixture
def temp_db():
    """Creates a temporary SQLite database for testing and cleans it up afterward."""
    # Use a unique filename for parallel tests if needed
    temp_db_path = f"./test_db_{uuid.uuid4()}.sqlite"
    
    # Ensure DB is initialized *before* patching DATABASE_PATH for agent usage
    # We need a direct connection here before patching affects get_db_connection globally
    try:
        conn = sqlite3.connect(temp_db_path) 
        conn.close() # Just create the file
        
        # Patch the global DATABASE_PATH used by db_utils functions
        with patch('a3x.core.db_utils.DATABASE_PATH', temp_db_path):
            initialize_database() # Initialize schema using the patched path
            yield temp_db_path # Provide the path to the test
            
    finally:
        # Cleanup: remove the temporary database file
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            print(f"\nRemoved temporary database: {temp_db_path}")

# --- Mock LLM Interface ---
class MockLLMInterface(LLMInterface):
    """Mock LLMInterface that returns predefined responses."""
    def __init__(self, responses=None):
        self.responses = responses if responses else []
        self.response_iterator = None
        self._reset_iterator()
        self._call_count = 0
        self.llm_url = "mock_url" # Add required attribute

    def _reset_iterator(self):
        if self.responses:
            if isinstance(self.responses, (list, tuple)):
                self.response_iterator = iter(self.responses)
            elif callable(self.responses): # Support callable side_effect
                 self.response_iterator = self.responses
            else:
                 # Assume single response
                 self.response_iterator = iter([self.responses])
        else:
            self.response_iterator = iter([])

    async def call_llm(self, messages: list, stream: bool = True, **kwargs) -> any:
        self._call_count += 1
        call_num = self._call_count
        response = "No more mock responses." # Default if iterator is exhausted
        try:
            if callable(self.response_iterator):
                # Directly call if it's a callable (like a mock side_effect)
                # Ensure the callable itself is awaited if it's async
                resp_obj = self.response_iterator(messages=messages, stream=stream, **kwargs)
                # Check if the result is awaitable (a coroutine)
                if asyncio.iscoroutine(resp_obj):
                     response = await resp_obj
                else:
                     response = resp_obj
                logger.debug(f"[MockLLM call {call_num}] Callable response: {response}")
            elif self.response_iterator:
                response = next(self.response_iterator)
                logger.debug(f"[MockLLM call {call_num}] Yielding response: {response}")
            else:
                 logger.warning(f"[MockLLM call {call_num}] No more responses in iterator.")

        except StopIteration:
            logger.warning(f"[MockLLM call {call_num}] Iterator exhausted.")
        except Exception as e:
            logger.exception(f"[MockLLM call {call_num}] Error getting mock response: {e}")
            response = f"Error in mock: {e}"

        # Define the async generator function *once*
        async def response_generator():
            if isinstance(response, str):
                yield response
            # Add handling for potential streaming responses if mock needs it later
            # elif hasattr(response, '__aiter__'): # If the mock response is already an async iterator
            #    async for chunk in response:
            #        yield chunk
            else: # Handle non-string potentially complex responses if needed
                 yield str(response) # Default string conversion

        # Always return the result of CALLING the generator function
        return response_generator()

    def add_responses(self, responses):
        """Adds more responses to the queue."""
        if isinstance(responses, (list, tuple)):
            self.responses.extend(responses)
        else:
            self.responses.append(responses)
        self._reset_iterator() # Reset iterator to include new responses

    def get_call_count(self):
        return self._call_count

# --- Fixtures ---

@pytest.fixture(scope="function")
def db_conn(tmp_path):
    """Fixture to set up and tear down the test database."""
    # Use tmp_path provided by pytest for a unique temp directory per test
    db_path_for_test = tmp_path / "test_agent_state.db"
    logger.info(f"DB Fixture: Setting up database at {db_path_for_test}")

    # Ensure the directory exists (though tmp_path should handle this)
    db_path_for_test.parent.mkdir(parents=True, exist_ok=True)

    # Patch DATABASE_PATH used by db_utils functions like get_db_connection, save_agent_state, etc.
    with patch("a3x.core.db_utils.DATABASE_PATH", str(db_path_for_test)) as mock_db_path:
        try:
            # Initialize the database schema using the *patched* path
            logger.info(f"DB Fixture: Initializing database schema for {db_path_for_test}")
            initialize_database() # db_utils functions will now use the patched path
            logger.info(f"DB Fixture: Database schema initialized.")
            yield str(db_path_for_test) # Yield the path for potential direct use if needed

        finally:
            logger.info(f"DB Fixture: Cleaning up resources for database at {db_path_for_test}")
            # Close any global connection if it exists (optional, depends on db_utils implementation)
            # close_db_connection(get_db_connection()) # Might be needed if get_db_connection caches
            # tmp_path fixture handles cleanup of the directory and file automatically
            pass # No explicit connection closing needed here as functions manage their own

@pytest.fixture(scope="function")
def tool_registry_data(tmp_path):
    """
    Provides a ToolRegistry instance with file manager skills registered
    AND the workspace path used for the FileManagerSkill.
    """
    registry = ToolRegistry()
    agent_workspace = tmp_path / "agent_workspace"
    agent_workspace.mkdir(exist_ok=True, parents=True)

    file_manager = FileManagerSkill(workspace_root=agent_workspace)
    tool_registry = ToolRegistry()
    for name, method in inspect.getmembers(FileManagerSkill, predicate=inspect.isfunction):
        skill_name = getattr(method, '_skill_name', None)
        print(f"[DEBUG TEST] Método: {name}, skill_name: {skill_name}")
        if skill_name:
            bound_method = method.__get__(file_manager, FileManagerSkill)
            description = getattr(method, '_skill_description', method.__doc__ or f"Skill: {skill_name}")
            tool_registry.register_tool(name=skill_name, instance=file_manager, tool=bound_method, description=description)

# --- Register final_answer skill ---
    tool_registry.register_tool(
        name="final_answer",
        instance=None,
        tool=final_answer,
        description="Provides the final answer or summary to the user."
    )

    fragment_registry = FragmentRegistry()
    from a3x.fragments.file_manager_fragment import FileOpsManager
    print("[DEBUG TEST] ToolRegistry _tools:", tool_registry._tools)
    fragment_def = fragment_registry.get_fragment_definition("FileOpsManager")
    fragment_registry._fragments["FileOpsManager"] = FileOpsManager(fragment_def=fragment_def, tool_registry=tool_registry)
    return {"registry": tool_registry, "workspace": agent_workspace}

@pytest.fixture
def fragment_registry(tool_registry_data): # Needs tool_registry to validate managed skills
    registry = FragmentRegistry(tool_registry_data["registry"])
    # Register the actual FileOpsManager fragment
    # registry.register_fragment(FileOpsManager) # <<< REMOVE: Fragments are auto-discovered
    return registry

# @pytest.fixture
# def tool_executor(db_conn, tool_registry, fragment_registry): # Add fragment_registry
#     # ToolExecutor needs access to the fragment registry to potentially execute fragments if needed
#     # (though typically orchestrator handles fragment execution)
#     # ToolExecutor also needs llm_interface but we mock it per test usually
#     # Let's initialize with None and tests can provide mock if needed by tools
#     return ToolExecutor(db_path=":memory:", tool_registry=tool_registry, fragment_registry=fragment_registry, llm_interface=None)

# @pytest.fixture
# def episodic_memory(db_conn):
#     return EpisodicMemory(db_path=":memory:")

# @pytest.fixture
# def vector_memory():
#     return VectorMemory(vector_db=SimpleVectorDB(embedding_dim=3)) # Example dim

@pytest.fixture
def agent_dependencies(tmp_path, mock_llm_interface):
    """Provides common dependencies for agent initialization using the global skill registry."""
    # Create a separate workspace for agent dependencies if needed
    agent_dep_workspace = tmp_path / "agent_dep_workspace"
    agent_dep_workspace.mkdir(exist_ok=True)

    # Get the globally populated skill registry
    # Assumes load_skills() in a3x/skills/__init__.py has run during import
    global_skill_registry = SKILL_REGISTRY # Use the imported dictionary directly
    logger.info(f"Using global SKILL_REGISTRY with {len(global_skill_registry)} skills.")

    # Initialize skills (example with FileManagerSkill)
    file_manager_skill = FileManagerSkill(workspace_root=agent_dep_workspace)

    # Create ToolRegistry
    tool_registry = ToolRegistry()
    tool_registry.register_tool(
        name="read_file",
        instance=file_manager_skill, 
        tool=file_manager_skill.read_file, 
        description=file_manager_skill.read_file.__doc__ or "Reads a file."
    )
    tool_registry.register_tool(
        name="write_file", 
        instance=file_manager_skill,
        tool=file_manager_skill.write_file,
        description=file_manager_skill.write_file.__doc__ or "Writes to a file."
    )
    tool_registry.register_tool(
        name="list_directory", 
        instance=file_manager_skill,
        tool=file_manager_skill.list_directory, 
        description=file_manager_skill.list_directory.__doc__ or "Lists directory contents."
    )

    # Initialize FragmentRegistry
    fragment_registry = FragmentRegistry()

    # Return dependencies expected by ReactAgent.__init__
    return {
        "agent_id": "1", # Example agent ID
        "llm_interface": mock_llm_interface,
        "skill_registry": global_skill_registry, # Pass the global registry dict
        "tool_registry": tool_registry,
        "fragment_registry": fragment_registry,
        # "workspace_root" will be set in the test itself
    }

# --- Test Cases ---

@pytest.mark.asyncio
async def test_agent_full_cycle_with_persistence(tmp_path, agent_dependencies, db_conn):
    """
    Tests a full agent cycle: user input -> orchestration -> fragment -> skill -> persistence -> loading.
    Uses the refactored FileOpsManager.
    """
    workspace_root = tmp_path / "test_workspace"
    workspace_root.mkdir()
    output_file = workspace_root / "output.txt"

    # --- ToolRegistry, FileManagerSkill e FragmentRegistry alinhados ---
    from a3x.core.tool_registry import ToolRegistry
    from a3x.skills.file_manager import FileManagerSkill
    from a3x.fragments.registry import FragmentRegistry
    import inspect
    file_manager = FileManagerSkill(workspace_root=workspace_root)
    tool_registry = ToolRegistry()
    for name, method in inspect.getmembers(FileManagerSkill, predicate=inspect.isfunction):
        skill_name = getattr(method, '_skill_name', None)
        print(f"[DEBUG TEST] Método: {name}, skill_name: {skill_name}")
        if skill_name:
            bound_method = method.__get__(file_manager, FileManagerSkill)
            description = getattr(method, '_skill_description', method.__doc__ or f"Skill: {skill_name}")
            tool_registry.register_tool(name=skill_name, instance=file_manager, tool=bound_method, description=description)

    tool_registry.register_tool(
        name="final_answer",
        instance=None,
        tool=final_answer,
        description="Provides the final answer or summary to the user."
    )

    fragment_registry = FragmentRegistry()
    from a3x.fragments.file_manager_fragment import FileOpsManager
    fragment_registry.register_fragment_class("FileOpsManager", FileOpsManager)
    # Step 1: Orchestrator delegates to FileOpsManager
    response1_delegate = json.dumps({
        "component": "FileOpsManager",
        "sub_task": "Write the content 'Olá Mundo' to the file named 'output.txt'"
    })
    # Step 2: FileOpsManager asks LLM to determine action and parameters
    response2_action = json.dumps({
        "skill_name": "write_file",
        "parameters": {
            "filename": "output.txt",
            "content": "Olá Mundo",
            "overwrite": True
        }
    })
    # Step 3: Orchestrator sees write success, asks LLM for next step (Final Answer)
    response3_final_answer = json.dumps({
        "component": "FinalAnswerProvider",
        "sub_task": "I have written 'Olá Mundo' to output.txt."
    })
    response4_dummy_end = json.dumps({"component": "End", "sub_task": "Task finished."})

    mock_llm = MockLLMInterface(responses=[
        response1_delegate,
        response2_action,
        response3_final_answer,
        response4_dummy_end
    ])

    # --- Agent Initialization ---
    agent = ReactAgent(
        agent_id="test_agent_persistence",
        llm_interface=mock_llm,
        skill_registry=None,  # Usar ToolRegistry, não dict
        tool_registry=tool_registry,
        fragment_registry=fragment_registry,
        workspace_root=workspace_root,
    )
    task = "Write 'Olá Mundo' to output.txt"
    agent_id = "test_agent_persistence"

    # --- Run Task ---
    final_result = await agent.run_task(task)
    logger.debug(f"Agent final result: {final_result}")
    logger.debug(f"Agent history: {agent._history}") # Use private attribute

    # Find the last message that represents the final answer
    final_answer_message = None
    for message in reversed(agent._history):
        content = message.get("content")
        if isinstance(content, str) and content.strip().startswith("Final Answer:"):
            final_answer_message = message
            break

    assert final_answer_message is not None, "Could not find final answer message in history."
    final_answer_content = content.strip()
    expected_answer_content = "Final Answer: I have written 'Olá Mundo' to output.txt."
    expected_answer_content_alt = "Final Answer: I have written 'Olá Mundo' to output.txt" # Without trailing period

    assert final_answer_content == expected_answer_content or \
           final_answer_content == expected_answer_content_alt, \
           f"Final answer content mismatch. Got: '{final_answer_content}'"

    assert output_file.exists(), "Output file was not created."
    assert output_file.read_text() == "Olá Mundo", "Output file content mismatch."

    # Check LLM calls (Orchestrator -> Manager -> Orchestrator)
    assert mock_llm.get_call_count() == 2, "Expected 2 LLM calls"
    assert mock_llm.responses[0] == response1_delegate
    assert mock_llm.responses[1] == response2_action

    loaded_state = load_agent_state(agent_id, db_path=":memory:")
    assert loaded_state is not None, "Agent state should have been saved"
    assert loaded_state["agent_id"] == agent_id
    # History comparison (ensure it's serializable and loadable)
    # Convert loaded history (JSON strings) back to dicts for comparison
    loaded_history = [json.loads(entry) if isinstance(entry, str) else entry for entry in loaded_state['history']]
    original_history = agent._history # Get the final history state
    # Compare essential parts (role, content) ignoring potential subtle diffs
    assert len(loaded_history) == len(original_history)
    for loaded, original in zip(loaded_history, original_history):
        assert loaded.get('role') == original.get('role')
        assert loaded.get('content') == original.get('content')

    # Memory comparison
    assert loaded_state['memory'] == agent._memory.export_memory(), "Agent memory state mismatch" # Use private attribute

@pytest.mark.asyncio
async def test_agent_read_and_write_cycle(tmp_path, agent_dependencies, db_conn):
    """
    Tests a cycle involving reading a file and then writing based on its content.
    Uses the refactored FileOpsManager.
    """
    workspace_root = tmp_path / "read_write_workspace"
    workspace_root.mkdir()
    input_file = workspace_root / "input_read_test.txt"
    output_file = workspace_root / "output_read_write.txt"
    input_content = "Conteúdo para ler."
    input_file.write_text(input_content)

    # --- Create ToolRegistry specific to this test's workspace --- 
    test_tool_registry = ToolRegistry()
    file_manager = FileManagerSkill(workspace_root=workspace_root) 
    # Register necessary file manager skills for this test
    test_tool_registry.register_tool(
        name="read_file",
        instance=file_manager,
        tool=file_manager.read_file,
        description="Reads a file."
    )
    test_tool_registry.register_tool(
        name="write_file",
        instance=file_manager,
        tool=file_manager.write_file,
        description="Writes to a file."
    )
    # Add final_answer if needed by the flow
    test_tool_registry.register_tool(
        name="final_answer",
        instance=None,
        tool=final_answer,
        description="Provides final answer."
    )

    # Update agent dependencies for this specific test
    agent_dependencies["workspace_root"] = workspace_root
    agent_dependencies["tool_registry"] = test_tool_registry # Use test-specific registry

    # --- Mock LLM Responses ---
    # 1. Orchestrator: Delegate Read
    resp1_delegate_read = json.dumps({
        "component": "FileOpsManager",
        "sub_task": f"Read the content of {input_file.name}"
    })
    # 2. FileOpsManager: Determine Read Action
    resp2_action_read = json.dumps({
        "skill_name": "read_file",
        "parameters": {"path": input_file.name}
    })
    # 3. Orchestrator: Delegate Write (using read content)
    resp3_delegate_write = json.dumps({
        "component": "FileOpsManager",
        "sub_task": f"Write the read content '{input_content}' to {output_file.name}"
    })
    # 4. FileOpsManager: Determine Write Action
    resp4_action_write = json.dumps({
        "skill_name": "write_file",
        "parameters": {"filename": output_file.name, "content": input_content, "overwrite": True}
    })
    # 5. Orchestrator: Final Answer
    resp5_final_answer_signal = json.dumps({
        "component": "FinalAnswerProvider",
        "sub_task": f"I have read '{input_content}' from {input_file.name} and written it to {output_file.name}."
    })
    response6_dummy_end = json.dumps({"component": "End", "sub_task": "Task finished."})

    mock_llm = MockLLMInterface(responses=[
        resp1_delegate_read,
        resp2_action_read,
        resp3_delegate_write,
        resp4_action_write,
        resp5_final_answer_signal,
        response6_dummy_end
    ])
    agent_dependencies["llm_interface"] = mock_llm

    # --- Agent Initialization ---
    agent = ReactAgent(**agent_dependencies)
    task = f"Read {input_file.name} and write its content to {output_file.name}"
    agent_id = "test_agent_read_write"

    # --- Run Task ---
    final_result = await agent.run_task(task)
    logger.debug(f"Agent final result: {final_result}")
    logger.debug(f"Agent history: {agent._history}")

    # Find the last message that represents the final answer
    final_answer_message = None
    for message in reversed(agent._history):
        content = message.get("content")
        if isinstance(content, str) and content.strip().startswith("Final Answer:"):
            final_answer_message = message
            break

    assert final_answer_message is not None, "Could not find final answer message in history."
    final_answer_content = content.strip()
    expected_answer_content = f"Final Answer: I have read '{input_content}' from {input_file.name} and written it to {output_file.name}."
    expected_answer_content_alt = f"Final Answer: I have read '{input_content}' from {input_file.name} and written it to {output_file.name}" # Without trailing period

    assert final_answer_content == expected_answer_content or \
           final_answer_content == expected_answer_content_alt, \
            f"Final answer content mismatch. Got: '{final_answer_content}'"

    assert output_file.exists(), "Output file was not created."
    assert output_file.read_text() == input_content, "Output file content mismatch."
    assert mock_llm.get_call_count() == 4, "Expected 4 LLM calls"

@pytest.mark.asyncio
async def test_agent_full_cycle_with_critical_fragment_failure(
    agent_dependencies, tmp_path, mocker, fragment_registry # Added fragment_registry fixture
):
    """
    Tests the agent cycle when a critical fragment's execution fails.
    The agent should stop and return an error state.
    """
    workspace_root = tmp_path / "critical_fail_workspace"
    workspace_root.mkdir()
    agent_dependencies["workspace_root"] = workspace_root

    # --- Mock LLM Responses ---
    # Step 1: Orchestrator delegates to FileOpsManager (which will fail)
    response1_delegate = json.dumps({
        "component": "FileOpsManager",
        "sub_task": "Perform a critical file operation"
    })
    # Step 2: FileOpsManager determines action (this might still succeed before execution fails)
    response2_action = json.dumps({
        "skill_name": "write_file",
        "parameters": {"filename": "fail.txt", "content": "should fail"}
    })
    # No further responses expected as execution should halt

    mock_llm = MockLLMInterface(responses=[response1_delegate, response2_action])
    agent_dependencies["llm_interface"] = mock_llm

    # --- Mock Tool Execution to Fail --- 
    async def mock_execute_tool_fail(*args, **kwargs):
        logger.debug(f"Mock execute_tool entered: args={args}, kwargs={kwargs}") # Changed log message
        tool_name = kwargs.get("tool_name")
        if tool_name == "write_file":
            logger.warning(f"Simulating critical failure during {tool_name} execution")
            raise ValueError("Simulated critical skill failure")
        logger.debug(f"Mock execute_tool called with non-failing skill ({tool_name}), passing through...")
        # If not the failing tool, simulate a generic success to avoid unrelated errors
        # You might need to return a more specific structure depending on what execute_tool normally returns
        return {
            "status": "success", 
            "result": {"status": "success", "message": f"Mock pass-through for {tool_name}"}
        } 

    # --- Patching --- 
    # Apply the patch *before* initializing the agent
    # Target execute_tool where it's *used* (in the fragment module)
    patcher = mocker.patch('a3x.fragments.file_manager_fragment.execute_tool', side_effect=mock_execute_tool_fail)

    # --- Agent Initialization --- 
    agent = ReactAgent(**agent_dependencies)

    # --- Run Agent --- 
    initial_goal = "Perform a critical file operation that will fail."
    final_result = await agent.run_task(initial_goal) # Correct call
    agent.last_run_context = final_result.get("shared_task_context") # <<< Store context

    # --- Stop Patcher (optional but good practice) ---
    patcher.stop() 

    # --- Assertions --- 
    assert final_result is not None, "Agent should return a result even on failure"
    assert final_result["status"] == "error", "Agent status should be 'error'"
    # Check the message from the orchestrator's error handling
    assert "Error during FileOpsManager execution" in final_result["message"], "Error message should indicate fragment failure"
    # Check the main message for the original exception string
    assert "Simulated critical skill failure" in final_result["message"], "Error message should contain the original exception" \
            # Modify this check based on actual structure of final_result on error
    # On fragment failure, the orchestrator puts the error in final_answer too
    assert "Error during FileOpsManager execution" in final_result.get("final_answer", ""), "Final answer should contain error on failure"
    assert "Simulated critical skill failure" in final_result.get("final_answer", ""), "Final answer should contain the original exception on failure"

    # Use the correct context attribute for history if it exists
    history_attr = agent.last_run_context # <<< Use stored context
    orchestration_history = getattr(history_attr, 'orchestration_history', None) if history_attr else None
    assert orchestration_history, "Orchestration history should not be empty"
    # Check history reflects the attempted steps and the failure
    assert any("FileOpsManager" in step.get("component_name", "") for step in orchestration_history), "FileOpsManager should be in history"
    assert any("Simulated critical skill failure" in step.get("error", "") for step in orchestration_history), "Failure event should be logged in history"

    assert not (workspace_root / "fail.txt").exists(), "File should not have been created due to failure"

    # Verify LLM calls
    assert mock_llm.call_count == 2, "LLM should have been called twice (delegate + action)"

# Add more tests:
# - Test with multiple non-critical fragments
# - Test max_iterations limit
# - Test error handling during LLM calls (e.g., invalid JSON response)
# - Test memory integration (though basic check is in persistence test)

# Mark file as containing async tests
pytestmark = pytest.mark.asyncio 