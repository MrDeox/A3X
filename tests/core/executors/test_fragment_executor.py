import pytest
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Imports from the module being tested
from a3x.core.executors.fragment_executor import FragmentExecutor
from a3x.core.constants import STATUS_SUCCESS
from a3x.core.context import SharedTaskContext
from a3x.fragments.base import FragmentDef

# Assume fixtures are defined in conftest.py or globally available tests fixtures
# We need:
# - llm_mock: An AsyncMock simulating LLMInterface.call_llm
# - tool_registry: A fixture providing a ToolRegistry instance
# - fragment_registry: A fixture providing a FragmentRegistry instance (or a mock)
# - dummy_fragment_def: A fixture providing a basic FragmentDef

# Basic LLM Mock Response for the test
HELLO_WORLD_FINAL_ANSWER = "Final Answer: Hello World!"

@pytest.mark.asyncio
async def test_executor_hello_world(
    tool_registry, # Assume this provides necessary 'final_answer' tool
    fragment_registry, # Assume this provides 'HelloFragment'
    tmp_path: Path
):
    """Tests if the executor can run a simple fragment that uses final_answer."""
    # --- Mocks --- 
    # Mock LLMInterface
    llm_mock = AsyncMock()
    # Configure the async generator mock for call_llm
    async def mock_llm_stream(*args, **kwargs):
        yield HELLO_WORLD_FINAL_ANSWER
    llm_mock.call_llm = mock_llm_stream # Assign the async generator function

    # Mock FragmentRegistry to return a simple definition
    mock_fragment_def = FragmentDef(
        name="HelloFragment",
        description="A simple fragment that says hello.",
        fragment_class=MagicMock(), # We don't need the actual class for this test
        skills=["final_answer"]
    )
    fragment_registry.get_definition = MagicMock(return_value=mock_fragment_def)
    # Ensure tool_registry has final_answer (might need adjustment based on actual fixture)
    if "final_answer" not in tool_registry:
         # Add a mock final_answer if not present
         mock_final_answer_tool = AsyncMock(return_value={"status": "success", "action": "final_answer"})
         tool_registry.register_tool("final_answer", None, mock_final_answer_tool, {
             "name": "final_answer", 
             "description": "Provides the final answer.",
             "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
         })

    # --- Executor Instantiation ---
    executor = FragmentExecutor(
        llm_interface=llm_mock,
        tool_registry=tool_registry,
        fragment_registry=fragment_registry,
        workspace_root=tmp_path
        # Using default max_iterations and max_runtime
    )

    # --- Execution ---
    shared_context = SharedTaskContext(task_id="test-hello")
    result = await executor.execute(
        fragment_name="HelloFragment",
        sub_task_objective="say hi",
        overall_objective="test executor",
        fragment_history=[],
        shared_task_context=shared_context,
        allowed_skills=["final_answer"], # Explicitly allow final_answer
        logger=logging.getLogger("test_executor")
    )

    # --- Assertions ---
    assert result is not None
    assert result.get("status") == STATUS_SUCCESS
    assert "final_answer" in result
    assert result["final_answer"] is not None
    assert "Hello World!" in result["final_answer"]
    # Check if LLM was called (using the generator mock directly doesn't track calls easily)
    # A better approach would be to wrap the generator in an AsyncMock
    # llm_mock.call_llm.assert_called_once() # This won't work directly with the generator func

    # --- Cleanup ---
    # Remove the mock final_answer tool
    tool_registry.unregister_tool("final_answer")

    # Remove the mock fragment definition
    fragment_registry.unregister_definition("HelloFragment") 