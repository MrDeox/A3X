# tests/test_agent_reflector.py
import pytest
import logging
from unittest.mock import MagicMock

# Import the function to test and the Decision type
from core.agent_reflector import reflect_on_observation, Decision 

# --- Fixtures --- 

@pytest.fixture
def mock_agent_logger():
    """Fixture for a mocked logger."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_context(mock_agent_logger): 
    """Provides a base context dictionary for calling reflect_on_observation."""
    return {
        "objective": "Test objective",
        "plan": ["Step 1", "Step 2"],
        "current_step_index": 0,
        "action_name": "some_tool",
        "action_input": {"param": "value"},
        "history": ["Human: Test objective", "LLM Response"],
        "memory": {"key": "value"},
        "agent_logger": mock_agent_logger
    }

# --- Test Cases --- 

def test_reflect_success_final_answer(base_context):
    """Test reflection when status is success and action is final_answer."""
    observation_dict = {
        "status": "success", 
        "action": "final_answer", 
        "data": {"answer": "All done!"}
    }
    context = base_context.copy()
    context["action_name"] = "final_answer"
    decision, new_plan = reflect_on_observation(**context, observation_dict=observation_dict)
    assert decision == "plan_complete"
    assert new_plan is None
    base_context["agent_logger"].info.assert_any_call("[Reflector] Final Answer provided. Plan complete.")

def test_reflect_success_tool_step(base_context):
    """Test reflection when status is success for a regular tool step."""
    observation_dict = {
        "status": "success", 
        "action": "some_tool_success", 
        "data": {"output": "Tool worked"}
    }
    decision, new_plan = reflect_on_observation(**base_context, observation_dict=observation_dict)
    assert decision == "continue_plan"
    assert new_plan is None
    base_context["agent_logger"].info.assert_any_call("[Reflector] Action 'some_tool' completed successfully.")

def test_reflect_no_change(base_context):
    """Test reflection when status is no_change."""
    observation_dict = {"status": "no_change", "action": "some_tool_no_change", "data": {}}
    decision, new_plan = reflect_on_observation(**base_context, observation_dict=observation_dict)
    assert decision == "continue_plan"
    assert new_plan is None
    base_context["agent_logger"].info.assert_any_call("[Reflector] Action 'some_tool' resulted in no change. Continuing plan.")

def test_reflect_error_tool_not_found(base_context):
    """Test reflection when status is error and action is tool_not_found."""
    observation_dict = {
        "status": "error", 
        "action": "tool_not_found", 
        "data": {"message": "Tool 'non_existent_tool' not found."}
    }
    context = base_context.copy()
    context["action_name"] = "non_existent_tool"
    decision, new_plan = reflect_on_observation(**context, observation_dict=observation_dict)
    assert decision == "stop_plan"
    assert new_plan is None
    base_context["agent_logger"].warning.assert_any_call("[Reflector] Tool 'non_existent_tool' not found. Stopping plan.")

def test_reflect_error_execution_failed(base_context):
    """Test reflection for execution_failed error (current placeholder behavior)."""
    observation_dict = {
        "status": "error", 
        "action": "execution_failed", 
        "data": {"message": "Code failed: ZeroDivisionError"}
    }
    context = base_context.copy()
    context["action_name"] = "execute_code"
    decision, new_plan = reflect_on_observation(**context, observation_dict=observation_dict)
    assert decision == "stop_plan" # Current behavior is stop_plan
    assert new_plan is None
    base_context["agent_logger"].warning.assert_any_call("[Reflector] Code execution failed for action 'execute_code'.")
    base_context["agent_logger"].info.assert_any_call("[Reflector] Auto-correction for execution_failed not implemented yet. Stopping plan.")

def test_reflect_error_parsing_failed(base_context):
    """Test reflection for parsing_failed error."""
    observation_dict = {
        "status": "error", 
        "action": "parsing_failed", 
        "data": {"message": "Could not parse JSON"}
    }
    context = base_context.copy()
    context["action_name"] = "_parse_llm"
    decision, new_plan = reflect_on_observation(**context, observation_dict=observation_dict)
    assert decision == "stop_plan"
    assert new_plan is None
    base_context["agent_logger"].error.assert_any_call("[Reflector] Internal agent error detected (parsing_failed). Stopping plan.")

def test_reflect_error_llm_call_failed(base_context):
    """Test reflection for llm_call_failed error."""
    observation_dict = {
        "status": "error", 
        "action": "llm_call_failed", 
        "data": {"message": "Timeout connecting to LLM"}
    }
    context = base_context.copy()
    context["action_name"] = "_llm_call"
    decision, new_plan = reflect_on_observation(**context, observation_dict=observation_dict)
    assert decision == "stop_plan"
    assert new_plan is None
    base_context["agent_logger"].error.assert_any_call("[Reflector] Internal agent error detected (llm_call_failed). Stopping plan.")

def test_reflect_error_unhandled_action(base_context):
    """Test reflection for an unhandled error action type."""
    observation_dict = {
        "status": "error", 
        "action": "some_unexpected_error_action", 
        "data": {"message": "Something weird happened"}
    }
    decision, new_plan = reflect_on_observation(**base_context, observation_dict=observation_dict)
    assert decision == "stop_plan"
    assert new_plan is None
    base_context["agent_logger"].error.assert_any_call("[Reflector] Unhandled error type (some_unexpected_error_action). Stopping plan.")

def test_reflect_unknown_status(base_context):
    """Test reflection when the observation status is unknown."""
    observation_dict = {"status": "maybe_success?", "action": "some_action", "data": {}}
    decision, new_plan = reflect_on_observation(**base_context, observation_dict=observation_dict)
    assert decision == "stop_plan"
    assert new_plan is None
    base_context["agent_logger"].warning.assert_any_call("[Reflector] Unknown status 'maybe_success?' in observation. Stopping plan as a precaution.") 