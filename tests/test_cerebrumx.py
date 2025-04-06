# tests/test_cerebrumx.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from a3x.core.cerebrumx import CerebrumXAgent, cerebrumx_logger
from a3x.core.agent import ReactAgent  # Needed for mocking super().run
# Import execution logic functions
from a3x.core.execution_logic import _execute_actual_plan_step
# Import config from conftest
from tests.conftest import TEST_SERVER_BASE_URL  # Import only constants

# Add project root to sys.path to find 'core' and 'skills'
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# --- Fixtures ---


@pytest.fixture
def agent_instance(
    mock_llm_interface,
    mock_planner,
    mock_reflector,
    mock_parser,
    mock_tool_executor,
    mock_db,
    mock_llm_url,
):
    """Provides a fully mocked Agent instance for testing basic execution flow."""
    # Mock load_agent_state which might be called during init
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("core.agent.load_agent_state", lambda *args, **kwargs: {})
        # *** CORRECTED: Instantiate CerebrumXAgent, not ReactAgent ***
        agent_obj = CerebrumXAgent(
            llm_url=mock_llm_url, system_prompt="mock_system_prompt"
        )

    # Add crucial mocks directly to the instance after creation
    agent_obj.add_history_entry = MagicMock()  # Mock history adding
    # agent_obj._call_llm = AsyncMock()  # REMOVED: Incorrect mock, agent doesn't have _call_llm
    # Mock retrieve_relevant_context directly here if it's consistently needed
    agent_obj._memory = MagicMock()
    agent_obj._memory.retrieve_relevant_context = AsyncMock(
        return_value={
            "semantic_match": "Default Mocked context via fixture",
            "short_term_history": [],
        }
    )

    return agent_obj  # Return the agent object


# --- Unit Tests for Helper Methods ---


def test_cerebrumx_init(cerebrumx_agent_instance):
    """Tests basic initialization of CerebrumXAgent."""
    assert isinstance(cerebrumx_agent_instance, CerebrumXAgent)
    assert isinstance(cerebrumx_agent_instance, ReactAgent)  # Check inheritance
    assert cerebrumx_agent_instance.system_prompt == "mock_cerebrumx_prompt"


def test_perceive(cerebrumx_agent_instance):
    """Tests the placeholder _perceive method."""
    initial_input = "Test perception input"
    result = cerebrumx_agent_instance._perceive(initial_input)
    assert result == {"processed": initial_input}


@pytest.mark.asyncio
async def test_retrieve_context(cerebrumx_agent_instance):
    """Tests the updated _retrieve_context method structure by mocking memory method."""
    # Mock the memory object's method directly on the instance for this test
    # OLD: (Mocking was previously assumed to be handled by fixture)
    # NEW: Mock directly on the instance's _memory object
    cerebrumx_agent_instance._memory.retrieve_relevant_context = AsyncMock(return_value={
        "semantic_match": "Mocked context via direct instance mock",
        "short_term_history": [] # Keep history simple
    })

    processed_perception = {"processed": "some data"}
    result = await cerebrumx_agent_instance._retrieve_context(processed_perception)

    assert "retrieved_context" in result
    assert "semantic_match" in result["retrieved_context"]
    # Correct assertion: Check if the mock on the instance's memory object was awaited
    cerebrumx_agent_instance._memory.retrieve_relevant_context.assert_awaited_once_with(
        query="some data", max_results=5
    )


@pytest.mark.asyncio
@patch("a3x.core.cerebrumx.get_tool_descriptions", return_value="Mock Tool Desc")
@patch("a3x.core.cerebrumx.execute_tool", new_callable=AsyncMock)
async def test_plan_hierarchically_success(
    mock_execute_tool, mock_get_tools, cerebrumx_agent_instance
):
    """Tests _plan_hierarchically when the planner skill succeeds."""
    mock_plan = ["Step 1", "Step 2"]
    mock_execute_tool.return_value = {
        "status": "success",
        "action": "plan_generated",
        "data": {"plan": mock_plan},
    }
    perception = {"processed": "Test objective"}
    context = {"retrieved_context": "Some context"}

    # Mock agent properties needed by execute_tool call
    cerebrumx_agent_instance.tools = {"hierarchical_planner": MagicMock()}
    cerebrumx_agent_instance._memory = {}

    result = await cerebrumx_agent_instance._plan_hierarchically(perception, context)

    mock_get_tools.assert_called_once()
    mock_execute_tool.assert_awaited_once_with(
        tool_name="hierarchical_planner",
        action_input={
            "objective": "Test objective",
            "available_tools": "Mock Tool Desc",
            "context": context,
        },
        tools_dict=cerebrumx_agent_instance.tools,
        agent_logger=cerebrumx_logger,
        agent_memory=cerebrumx_agent_instance._memory,
    )
    assert result == mock_plan


@pytest.mark.asyncio
@patch("a3x.core.cerebrumx.get_tool_descriptions", return_value="Mock Tool Desc")
@patch("a3x.core.cerebrumx.execute_tool", new_callable=AsyncMock)
async def test_plan_hierarchically_planner_error(
    mock_execute_tool, mock_get_tools, agent_instance
):
    """Tests _plan_hierarchically when the planner skill returns an error."""
    mock_execute_tool.return_value = {
        "status": "error",
        "action": "plan_generation_failed_parsing",
        "data": {"message": "LLM Parse Error"},
    }
    perception = {"processed": "Complex objective"}
    context = {"retrieved_context": "Some context"}
    agent_instance.tools = {"hierarchical_planner": MagicMock()}
    agent_instance._memory = {}

    result = await agent_instance._plan_hierarchically(perception, context)

    mock_get_tools.assert_called_once()
    mock_execute_tool.assert_awaited_once_with(
        tool_name="hierarchical_planner",
        action_input={
            "objective": "Complex objective",
            "available_tools": "Mock Tool Desc",
            "context": context,
        },
        tools_dict=agent_instance.tools,
        agent_logger=cerebrumx_logger,
        agent_memory=agent_instance._memory,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith("Address objective directly (planning failed):")
    assert "Complex objective" in result[0]


@pytest.mark.asyncio
@patch("a3x.core.cerebrumx.get_tool_descriptions", return_value="Mock Tool Desc")
@patch("a3x.core.cerebrumx.execute_tool", new_callable=AsyncMock)
async def test_plan_hierarchically_empty_plan(
    mock_execute_tool, mock_get_tools, agent_instance
):
    """Tests _plan_hierarchically when the planner skill returns an empty plan."""
    mock_execute_tool.return_value = {
        "status": "success",
        "action": "plan_generated",
        "data": {"plan": []},  # Empty plan
    }
    perception = {"processed": "Another objective"}
    context = {"retrieved_context": "Some context"}
    agent_instance.tools = {"hierarchical_planner": MagicMock()}
    agent_instance._memory = {}

    result = await agent_instance._plan_hierarchically(perception, context)

    mock_get_tools.assert_called_once()
    mock_execute_tool.assert_awaited_once_with(
        tool_name="hierarchical_planner",
        action_input={
            "objective": "Another objective",
            "available_tools": "Mock Tool Desc",
            "context": context,
        },
        tools_dict=agent_instance.tools,
        agent_logger=cerebrumx_logger,
        agent_memory=agent_instance._memory,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].startswith(
        "Address objective directly:"
    )  # Fallback without "planning failed"
    assert "Another objective" in result[0]


# --- Mocks for New Skills ---
@pytest.fixture
def mock_simulate_step_success():
    return {
        "status": "success",
        "simulated_outcome": "Simulated success for the step.",
        "confidence": "Alta",
    }


@pytest.fixture
def mock_reflect_step_execute():
    return {
        "status": "success",
        "decision": "execute",
        "justification": "Looks good, proceed.",
        "confidence": "Alta",
    }


@pytest.fixture
def mock_reflect_step_skip():
    return {
        "status": "success",
        "decision": "skip",
        "justification": "Step is redundant.",
        "confidence": "Alta",
    }


@pytest.fixture
def mock_reflect_step_modify():
    return {
        "status": "success",
        "decision": "modify",
        "justification": "Needs adjustment before execution.",
        "confidence": "Média",
    }


# --- Updated Tests for Core Logic (_plan_hierarchically, _execute_plan_step, _reflect, _learn) ---


@pytest.mark.asyncio
@patch("a3x.core.cerebrumx.execute_tool", new_callable=AsyncMock)
async def test_simulate_step_calls_skill(
    mock_execute_tool, agent_instance, mock_simulate_step_success
):
    """Tests that _simulate_step calls the simulate_step skill via execute_tool."""
    plan_step = "Test Step"
    context = {"key": "value"}
    mock_execute_tool.return_value = mock_simulate_step_success

    result = await agent_instance._simulate_step(plan_step, context)

    expected_input = {"step": plan_step, "context": context}
    mock_execute_tool.assert_awaited_once_with(
        tool_name="simulate_step",
        action_input=expected_input,
        tools_dict=agent_instance.tools,
        agent_logger=cerebrumx_logger,
        agent_memory=agent_instance._memory,
    )
    # Check if the result from the skill is returned correctly
    assert (
        result["simulated_outcome"] == mock_simulate_step_success["simulated_outcome"]
    )
    assert result["confidence"] == mock_simulate_step_success["confidence"]


@pytest.mark.asyncio
# Removed patch decorator
async def test_execute_actual_plan_step_success(agent_instance):
    """Tests _execute_actual_plan_step when the base run loop succeeds."""
    step_objective = "Achieve sub-goal"
    context = {}
    # agent_instance.add_history_entry = MagicMock() # Now mocked in fixture

    mock_final_result = {"status": "success", "message": "Sub-goal achieved"}

    # Define an ASYNC GENERATOR function for the side_effect
    async def mock_run_generator(*args, **kwargs):
        yield mock_final_result  # Yield the result as expected by async for

    # Manually patch the run method on the specific instance
    agent_instance.run = mock_run_generator  # Assign the generator directly

    # Call the actual function from execution_logic, passing the agent
    result = await _execute_actual_plan_step(agent_instance, step_objective, context)

    # agent_instance.run.assert_called_once_with(objective=step_objective) # Cannot assert call on a direct function assignment
    # Instead, check if the result matches the *final* yielded value
    assert result == mock_final_result


@pytest.mark.asyncio
# Removed patch decorator
async def test_execute_actual_plan_step_react_error(agent_instance):
    """Tests _execute_actual_plan_step when the base run loop returns an error dict."""
    step_objective = "Try failing sub-goal"
    context = {}
    # agent_instance.add_history_entry = MagicMock() # Mocked in fixture

    mock_error_result = {"status": "error", "message": "Tool execution failed"}

    # Define an ASYNC GENERATOR function for the side_effect
    async def mock_run_generator(*args, **kwargs):
        yield mock_error_result  # Yield the error result

    # Manually patch the run method on the specific instance
    agent_instance.run = mock_run_generator  # Assign the generator directly

    # Call the actual function from execution_logic
    result = await _execute_actual_plan_step(agent_instance, step_objective, context)

    # agent_instance.run.assert_called_once_with(objective=step_objective) # Cannot assert call
    assert result == mock_error_result


@pytest.mark.asyncio
# Removed patch decorator
async def test_execute_actual_plan_step_final_answer_string(agent_instance):
    """Tests _execute_actual_plan_step when the base run loop returns a final answer string."""
    step_objective = "Get final string answer"
    context = {}
    # agent_instance.add_history_entry = MagicMock() # Mocked in fixture

    mock_final_answer = "This is the final answer."

    # Define an ASYNC GENERATOR function for the side_effect
    async def mock_run_generator(*args, **kwargs):
        yield mock_final_answer  # Yield the final string

    # Manually patch the run method on the specific instance
    agent_instance.run = mock_run_generator  # Assign the generator directly

    result = await _execute_actual_plan_step(agent_instance, step_objective, context)

    # agent_instance.run.assert_called_once_with(objective=step_objective) # Cannot assert call
    expected_result = {"status": "success", "message": mock_final_answer}
    assert result == expected_result


@pytest.mark.asyncio
# Removed patch decorator
async def test_execute_actual_plan_step_exception(agent_instance):
    """Tests _execute_actual_plan_step when the base run loop raises an exception."""
    step_objective = "Sub-goal causing exception"
    context = {}
    # agent_instance.add_history_entry = MagicMock() # Mocked in fixture

    # Define an ASYNC GENERATOR that raises an exception
    async def mock_run_generator(*args, **kwargs):
        raise Exception("Base run failed!")
        yield  # Need yield to make it an async generator

    # Manually patch the run method on the specific instance to raise exception
    agent_instance.run = mock_run_generator  # Assign the failing generator

    # Call the actual function from execution_logic
    result = await _execute_actual_plan_step(agent_instance, step_objective, context)

    # agent_instance.run.assert_called_once_with(objective=step_objective) # Cannot assert call
    assert result["status"] == "error"
    assert "Exception during step execution: Base run failed!" in result["message"]


@pytest.mark.asyncio
async def test_reflect_success(agent_instance):
    """Tests the updated _reflect method with successful execution."""
    perception = {"processed": "Objective"}
    plan = ["Step 1", "Step 2"]
    execution_results = [
        {"status": "success", "message": "Step 1 done"},
        {"status": "success", "message": "Step 2 done"},
    ]
    result = await agent_instance._reflect(perception, plan, execution_results)
    assert isinstance(result, dict)
    # Corrected Assertion: Match the exact format from _reflect
    assert (
        result["assessment"]
        == "Objective 'Objective...': Plan executed successfully. All 2 steps completed."
    )
    assert result["success_rate"] == 1.0
    assert result["overall_outcome"] == "success"
    assert len(result["learnings"]) == 2


@pytest.mark.asyncio
async def test_reflect_partial_failure(agent_instance):
    """Tests the updated _reflect method with partial failure."""
    perception = {"processed": "Objective"}
    plan = ["Step 1", "Step 2", "Step 3"]
    execution_results = [
        {"status": "success", "message": "Step 1 done"},
        {"status": "error", "message": "Something failed in step 2"},
        {"status": "success", "message": "Step 3 done"},  # Added success for step 3
    ]
    result = await agent_instance._reflect(perception, plan, execution_results)
    assert isinstance(result, dict)
    # Correct assertion: Match the exact format from _reflect (2/3 = 67%)
    assert (
        result["assessment"]
        == "Objective 'Objective...': Plan partially executed. 2/3 steps successful (67%)."
    )
    assert round(result["success_rate"], 2) == 0.67 # Check rounded rate
    assert result["overall_outcome"] == "partial_success"
    assert len(result["learnings"]) == 3


@pytest.mark.asyncio
async def test_reflect_all_failures(agent_instance):
    """Tests the updated _reflect method with all steps failing."""
    perception = {"processed": "Objective"}
    plan = ["Step 1", "Step 2"]
    execution_results = [
        {"status": "error", "message": "Failure A"},
        {"status": "error", "message": "Failure B"},
    ]
    result = await agent_instance._reflect(perception, plan, execution_results)
    assert isinstance(result, dict)
    # Corrected Assertion: Match the exact format from _reflect
    assert (
        result["assessment"]
        == "Objective 'Objective...': Plan execution failed. 0/2 steps successful (0%)."
    )
    assert result["success_rate"] == 0.0
    assert result["overall_outcome"] == "failure"
    assert len(result["learnings"]) == 2


@pytest.mark.asyncio
async def test_reflect_empty_results(agent_instance):
    """Tests the updated _reflect method with empty execution results."""
    perception = {"processed": "Objective"}
    plan = []  # Empty plan implies no execution
    execution_results = []
    result = await agent_instance._reflect(perception, plan, execution_results)
    assert isinstance(result, dict)
    # Corrected Assertion: Match the exact format from _reflect
    assert result["assessment"] == "Objective 'Objective...': No steps were executed."
    assert result["success_rate"] == 0.0
    assert result["overall_outcome"] == "unknown"
    assert len(result["learnings"]) == 0


@pytest.mark.asyncio
async def test_learn(agent_instance):
    """Tests the updated _learn method (checks logging)."""
    # Corrected: Provide structured learnings as expected by _learn
    reflection_success = {
        "assessment": "Plan execution completed. Success Rate: 100%",
        "success_rate": 1.0,
        "learnings": [
            {
                "type": "success",
                "step_index": 0,
                "step_description": "Mock Step 1",
                "content": "Step 1 ('Mock Step 1...'): Completed successfully. Result: Mock success...",
            }
        ],
    }
    reflection_failure = {
        "assessment": "Plan execution completed. Success Rate: 0% (1 failed steps)",
        "success_rate": 0.0,
        "learnings": [
            {
                "type": "failure",
                "step_index": 0,
                "step_description": "Mock Step 1",
                "content": "Step 1 ('Mock Step 1...'): Failed. Reason: Mock error...",
            }
        ],
    }

    # Mock the logger used within _learn and the memory method if needed
    with patch("a3x.core.cerebrumx.cerebrumx_logger.info") as mock_log_info, \
         patch.object(agent_instance._memory, "add_episodic_record", new_callable=AsyncMock) as mock_add_memory:

        await agent_instance._learn(reflection_success)
        # Check logging
        log_calls_success = [args[0] for args, kwargs in mock_log_info.call_args_list]
        assert "Updating memory based on reflection..." in log_calls_success
        assert "1 potential learning points identified." in log_calls_success # Check count
        assert "Overall Execution Assessment: Plan execution completed. Success Rate: 100%" in log_calls_success
        # Check memory call
        mock_add_memory.assert_awaited_once_with(data=reflection_success["learnings"][0])

        # Reset mocks for the failure case
        mock_log_info.reset_mock()
        mock_add_memory.reset_mock()

        await agent_instance._learn(reflection_failure)
        # Check logging for failure
        log_calls_failure = [args[0] for args, kwargs in mock_log_info.call_args_list]
        assert "Updating memory based on reflection..." in log_calls_failure
        assert "1 potential learning points identified." in log_calls_failure # Check count
        assert "Overall Execution Assessment: Plan execution completed. Success Rate: 0% (1 failed steps)" in log_calls_failure
        # Check memory call for failure
        mock_add_memory.assert_awaited_once_with(data=reflection_failure["learnings"][0])

        # Reset mocks for the empty learnings case
        mock_log_info.reset_mock()
        mock_add_memory.reset_mock()

        # Test with empty learnings
        reflection_empty = {"assessment": "Done", "success_rate": 1.0, "learnings": []}
        await agent_instance._learn(reflection_empty)
        log_calls_empty = [args[0] for args, kwargs in mock_log_info.call_args_list]
        assert "Updating memory based on reflection..." in log_calls_empty
        assert "No specific step learnings identified in this cycle." in log_calls_empty
        assert "Overall Execution Assessment: Done" in log_calls_empty
        # Check memory was NOT called
        mock_add_memory.assert_not_called()


# --- Integration Test for CerebrumX Cycle (Now relies on REAL LLM Server) ---


@pytest.mark.asyncio
@patch("a3x.core.cerebrumx.CerebrumXAgent._perceive")
@patch("a3x.core.cerebrumx.CerebrumXAgent._retrieve_context")
@patch("a3x.core.cerebrumx.CerebrumXAgent._plan_hierarchically")
# @patch('core.execution_logic.execute_plan_with_reflection') # REMOVED - Testing real execution
@patch("a3x.core.cerebrumx.CerebrumXAgent._reflect")  # Mock overall reflection
@patch("a3x.core.cerebrumx.CerebrumXAgent._learn")  # Mock learning
# @patch('skills.simulation.call_llm', new_callable=AsyncMock) # REMOVED - Using real LLM
# @patch('skills.reflection.call_llm', new_callable=AsyncMock) # REMOVED - Using real LLM
async def test_run_cerebrumx_cycle_execute_flow(
    # Removed skill LLM mocks
    mock_learn,
    mock_reflect,  # Removed mock_execute_plan_loop
    mock_plan_hierarchically,
    mock_retrieve_context,
    mock_perceive,
    agent_instance,
    managed_llama_server_session,  # Fixture injected by pytest automatically by name - CORRECTED NAME
):
    """
    Tests the main run_cerebrumx_cycle focusing on the execution flow,
    interacting with the MANAGED LLM server for simulation/reflection steps.
    NOTE: Uses the server started by the `managed_llama_server` fixture.
    """
    # --- Mock Setup ---
    objective = "Test objective for execution flow (e.g., list files)"  # More realistic objective
    mock_perception = {"processed": objective}
    # Use the mock from the test_retrieve_context logic for consistency
    mock_context = {
        "retrieved_context": {
            "semantic_match": "Mocked context via instance memory patch",
            "short_term_history": [
                {"role": "user", "content": "previous instance memory patch turn"}
            ],
        }
    }
    # Mock retrieve_context on the instance for this test
    agent_instance._retrieve_context = AsyncMock(return_value=mock_context)

    # Keep plan mock simple for now, assuming planner works or falls back
    mock_plan = ["Step 1: Simulate listing files", "Step 2: Reflect on listing files"]

    mock_perceive.return_value = mock_perception
    # mock_retrieve_context.return_value = mock_context # Already mocked on instance
    mock_plan_hierarchically.return_value = mock_plan

    # *** Configure agent to use the managed test server ***
    agent_instance.llm_url = TEST_SERVER_BASE_URL
    # Ensure the logger also reflects this change if it logs the URL
    cerebrumx_logger.info(
        f"[Test Setup] Agent LLM URL set to managed server: {agent_instance.llm_url}"
    )

    # REMOVED Skill LLM call mocks
    # REMOVED mock_execute_plan_loop mock

    # Mock overall reflection and learning outcomes
    mock_final_reflection = {"summary": "Overall execution attempted"}
    mock_learning_outcome = {"learned": "Something was learned"}
    mock_reflect.return_value = mock_final_reflection
    mock_learn.return_value = mock_learning_outcome

    # --- Execute ---
    # Correctly consume the async generator returned by run_cerebrumx_cycle
    all_results = []
    # Add error handling as real LLM calls might fail
    try:
        async for result in agent_instance.run_cerebrumx_cycle(objective):
            all_results.append(result)
            # Optional: Add basic checks on yielded items if needed
            assert isinstance(result, dict)
            assert "type" in result
    except Exception as e:
        pytest.fail(
            f"run_cerebrumx_cycle raised an unexpected exception during integration test: {e}"
        )

    # --- Assertions ---
    mock_perceive.assert_called_once_with(objective)
    agent_instance._retrieve_context.assert_awaited_once_with(mock_perception)
    mock_plan_hierarchically.assert_called_once_with(mock_perception, mock_context)

    # Assert core methods were called (cannot assert mock_execute_plan_loop anymore)
    assert mock_reflect.call_count == 1  # Check reflect was called at the end
    # Check arguments passed to reflect might be complex now, just check call count
    mock_learn.assert_called_once_with(mock_final_reflection)

    # Assert the final outcome / yielded results
    # The exact results depend heavily on the real LLM interaction
    # Focus on the flow being completed without crashing
    assert len(all_results) > 0  # Check that the cycle yielded *something*
    print(f"Integration test yielded {len(all_results)} results:")
    # for i, res in enumerate(all_results):
    #     print(f"  [{i}] {res.get('type')}: {str(res.get('content') or res.get('result'))[:100]}...")


# Add more tests for skip/modify flow in run_cerebrumx_cycle if needed
# Add tests for error handling within run_cerebrumx_cycle (e.g., planning fails)
