# tests/test_agent_run_errors.py
import pytest
import json
from unittest.mock import MagicMock, call, AsyncMock

# Import necessary components
from core.agent import ReactAgent
from core.config import MAX_REACT_ITERATIONS
# Import exception types if needed
from requests.exceptions import RequestException, HTTPError, Timeout

# Fixtures are typically in conftest.py, but if specific mocks are needed here:
# @pytest.fixture
# def mock_specific_tool(mocker):
#     # ... setup specific tool mock ...
#     pass

# --- Error Handling Specific Tests ---

@pytest.mark.asyncio
async def test_react_agent_run_handles_parsing_error(
    agent_instance, mock_db, mocker, INVALID_JSON_STRING
):
    """Testa se o agente lida com JSON inválido na resposta do LLM."""
    agent, _ = agent_instance

    objective = "Do something that results in bad JSON"
    mock_plan = [objective]
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    # <<< CORRECT Patch target and use AsyncMock >>>
    mock_call_llm = mocker.patch('core.agent.call_llm', new_callable=AsyncMock)
    mock_call_llm.return_value = INVALID_JSON_STRING # LLM returns invalid JSON

    # Mock reflector (assume it stops on parsing error)
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("stop_plan", None))

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    mock_call_llm.assert_awaited_once()
    mock_reflector.assert_awaited_once()
    reflector_call_args = mock_reflector.await_args[1]
    assert reflector_call_args['action_name'] == '_parse_llm' # Internal action for parsing
    assert reflector_call_args['observation_dict']['status'] == 'error'
    assert reflector_call_args['observation_dict']['action'] == 'parsing_failed'
    assert "Erro: Falha ao processar resposta do LLM" in final_response
    mock_db.assert_called_once()

@pytest.mark.asyncio
async def test_react_agent_run_handles_max_iterations(
    agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_LIST_FILES
):
    """Testa se o agente para após atingir o número máximo de iterações."""
    agent, _ = agent_instance
    agent.max_iterations = 2 # Set a low max iteration for testing

    objective = "List files repeatedly (will hit max iterations)"
    # Simulate a plan that doesn't finish
    mock_plan = ["Step 1: List files", "Step 2: List files again"]
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    # <<< CORRECT Patch target and use AsyncMock >>>
    mock_call_llm = mocker.patch('core.agent.call_llm', new_callable=AsyncMock)
    mock_call_llm.return_value = LLM_JSON_RESPONSE_LIST_FILES

    # Mock the tool executor to always return success (but doesn't lead to final_answer)
    list_success_result = {"status": "success", "action": "list_files_success", "data": {"files": ["file.txt"]}}
    mock_executor = mocker.patch('core.agent.tool_executor.execute_tool', return_value=list_success_result)

    # Mock reflector to always continue
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("continue_plan", mock_plan))

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    # Check LLM was called max_iterations times ( planner + agent cycles)
    assert mock_call_llm.await_count == agent.max_iterations
    assert mock_executor.call_count == agent.max_iterations
    assert mock_reflector.await_count == agent.max_iterations
    assert f"Erro: Maximum total iterations ({agent.max_iterations}) reached" in final_response
    mock_db.assert_called_once() # Should still save state

@pytest.mark.asyncio
async def test_react_agent_run_handles_failed_planning(
    agent_instance, mock_db, mocker
):
    """Testa se o agente lida com a falha na geração do plano inicial."""
    agent, _ = agent_instance

    objective = "Do something complex"
    # Mock planner to return None (failure)
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=None)

    # Mock LLM to respond as if it received the objective directly as the first step
    # <<< CORRECT Patch target and use AsyncMock >>>
    mock_call_llm = mocker.patch('core.agent.call_llm', new_callable=AsyncMock)
    mock_call_llm.return_value = json.dumps({
        "Thought": "Planning failed, I'll try the objective directly. User wants something complex, I'll provide a generic final answer.",
        "Action": "final_answer",
        "Action Input": {"answer": "Could not determine steps for the complex objective."}
    })

    # Mock reflector
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("plan_complete", None)) # Assume reflector sees final_answer

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    mock_call_llm.assert_awaited_once() # Called once with the objective as the 'plan'
    mock_reflector.assert_awaited_once()
    assert "Could not determine steps for the complex objective." in final_response
    mock_db.assert_called_once()
