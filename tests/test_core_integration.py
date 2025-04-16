import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
from pathlib import Path
import logging

from a3x.core.agent import ReactAgent
from a3x.core.llm_interface import LLMInterface
from a3x.core.tool_registry import ToolRegistry
from a3x.fragments.registry import FragmentRegistry
from a3x.core.orchestrator import TaskOrchestrator  # Import for type hinting

# Configure logging for tests (optional, but helpful)
logging.basicConfig(level=logging.DEBUG)


# Fixture to create a mock LLM Interface
@pytest.fixture
def mock_llm_interface():
    mock = AsyncMock(spec=LLMInterface)
    mock.llm_url = "http://mock-llm-url"
    return mock


# Fixture to create a mock Tool Registry
@pytest.fixture
def mock_tool_registry():
    mock = MagicMock(spec=ToolRegistry)
    # Add mock methods/attributes if needed for registration/lookup
    mock.get_tool_schema = MagicMock(return_value={"type": "function", "function": {"name": "mock_tool", "description": "A mock tool", "parameters": {}}})
    return mock

# Fixture to create a mock Fragment Registry
@pytest.fixture
def mock_fragment_registry():
    mock = MagicMock(spec=FragmentRegistry)
    # Add mock methods/attributes if needed
    mock.get_fragment = MagicMock(return_value=MagicMock()) # Return a generic mock fragment
    return mock

# Fixture for the TaskOrchestrator mock
@pytest.fixture
def mock_orchestrator_instance():
    # Create an AsyncMock instance directly
    mock = AsyncMock(spec=TaskOrchestrator)
    # Mock the orchestrate method to return a predefined result
    async def mock_orchestrate(*args, **kwargs):
        print(f"Mock Orchestrator called with args: {args}, kwargs: {kwargs}")
        # Simulate returning a final answer structure
        return {
            "status": "success",
            "result": {
                "final_answer": "The task is complete according to the mock orchestrator.",
                "result_details": "Mock details",
                "intermediate_steps": []
            }
        }
    mock.orchestrate.side_effect = mock_orchestrate
    return mock

# Updated Test Fixture for ReactAgent
@pytest.fixture
@patch("a3x.core.db_utils.load_agent_state", return_value=None) # Mock DB load
@patch("a3x.core.db_utils.save_agent_state") # Mock DB save
@patch("a3x.core.orchestrator.TaskOrchestrator", new_callable=AsyncMock) # Correct patch target
def test_agent(
    mock_task_orchestrator_class: AsyncMock, # Patched Class
    mock_save_state: MagicMock,
    mock_load_state: MagicMock,
    mock_llm_interface: AsyncMock,
    mock_tool_registry: MagicMock,
    mock_fragment_registry: MagicMock,
    mock_orchestrator_instance: AsyncMock, # Use the instance fixture
    tmp_path: Path
):
    """Fixture to create a ReactAgent instance with mocked dependencies."""
    # Configure the *class* mock to return our specific *instance* mock
    mock_task_orchestrator_class.return_value = mock_orchestrator_instance

    agent = ReactAgent(
        agent_id="test_agent_001",
        llm_interface=mock_llm_interface,
        skill_registry={}, # Keep empty or provide mock skills if needed
        tool_registry=mock_tool_registry,
        fragment_registry=mock_fragment_registry,
        workspace_root=tmp_path,
        logger=logging.getLogger("test_agent")
    )
    # <<< Ensure the agent uses the correct mock orchestrator instance >>>
    agent.orchestrator = mock_orchestrator_instance

    return agent, mock_orchestrator_instance # Return both agent and the specific mock

# Test the core cognitive flow (delegation to orchestrator)
@pytest.mark.asyncio
async def test_core_cognitive_flow(test_agent):
    """Tests that the agent correctly delegates the task to the TaskOrchestrator."""
    agent, mock_orchestrator = test_agent # Unpack the agent and the mock
    objective = "Write a simple hello world script."

    print(f"Agent Orchestrator before run: {agent.orchestrator}")
    print(f"Mock Orchestrator used in assert: {mock_orchestrator}")

    # Run the agent
    result = await agent.run_task(objective)

    # Assertions
    # 1. Check that the orchestrator's orchestrate method was called exactly once
    mock_orchestrator.orchestrate.assert_called_once()

    # 2. Check the arguments passed to orchestrate
    # Use mock_orchestrator directly for assertion
    call_args, call_kwargs = mock_orchestrator.orchestrate.call_args
    # Check the first positional argument for the objective
    assert call_args[0] == objective
    # Add more specific checks for other args/kwargs if needed
    # e.g., assert call_kwargs.get('agent_id') == agent.agent_id # Check if agent_id is passed via kwargs
    # e.g., assert call_args[1] == some_other_positional_arg

    # 3. Check the final result structure (assuming orchestrator returns a dict)
    assert isinstance(result, dict)
    assert "status" in result
    assert "result" in result
    assert "final_answer" in result["result"]
    assert result["result"]["final_answer"] == "The task is complete according to the mock orchestrator."

    print("Test test_core_cognitive_flow completed successfully.")

# --- Add more tests as needed ---
# Example: Test error handling
# Example: Test different objectives
# Example: Test state loading/saving interaction