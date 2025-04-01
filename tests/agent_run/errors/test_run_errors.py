# tests/test_agent_run_errors.py
import pytest
import json
from unittest.mock import AsyncMock, patch

# Import necessary components
# Import exception types if needed


# Add a fixture for the mock URL
@pytest.fixture
def mock_llm_url():
    return "http://mock-llm-errors/v1/chat/completions"


# --- Fixtures ---


@pytest.fixture
def INVALID_JSON_STRING():
    return "This is not valid JSON { maybe ```json ... nope ```"


@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES():  # Needed for max_iterations test
    return json.dumps(
        {
            "thought": "Still listing files...",
            "Action": "list_files",
            "action_input": {"directory": "."},
        }
    )


# --- Error Handling Specific Tests ---


@pytest.mark.asyncio
async def test_react_agent_run_handles_parsing_error(
    agent_instance, mock_db, mocker, INVALID_JSON_STRING
):
    """Testa se o agente lida com JSON inválido na resposta do LLM."""
    agent = agent_instance

    objective = "Do something that results in bad JSON"
    mock_plan = [objective]
    mock_planner = mocker.patch(
        "core.planner.generate_plan", new_callable=AsyncMock, return_value=mock_plan
    )

    with patch("core.agent.call_llm", new_callable=AsyncMock) as mock_call_llm:

        async def mock_llm_invalid_json():
            yield INVALID_JSON_STRING

        mock_call_llm.return_value = mock_llm_invalid_json()

    # Mock reflector (assume it stops on parsing error)
    mock_reflector = mocker.patch(
        "core.agent_reflector.reflect_on_observation",
        new_callable=AsyncMock,
        return_value=("stop_plan", None),
    )

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    mock_call_llm.assert_called_once()
    mock_reflector.assert_called_once()
    reflector_call_args = mock_reflector.call_args.kwargs
    assert (
        reflector_call_args.get("action_name") == "_parse_llm"
    )  # Internal action for parsing
    assert reflector_call_args.get("observation_dict", {}).get("status") == "error"
    assert (
        reflector_call_args.get("observation_dict", {}).get("action")
        == "parsing_failed"
    )
    assert "Erro: Falha ao processar resposta do LLM" in final_response
    mock_db.assert_called_once()


@pytest.mark.asyncio
async def test_react_agent_run_handles_max_iterations(
    agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_LIST_FILES
):
    """Testa se o agente para após atingir o número máximo de iterações."""
    agent = agent_instance
    agent.max_iterations = 2  # Set a low max iteration for testing

    objective = "List files repeatedly (will hit max iterations)"
    # Simulate a plan that doesn't finish
    mock_plan = ["Step 1: List files", "Step 2: List files again"]
    mock_planner = mocker.patch(
        "core.planner.generate_plan", new_callable=AsyncMock, return_value=mock_plan
    )

    with patch("core.agent.call_llm", new_callable=AsyncMock) as mock_call_llm:

        async def mock_llm_repeat_list():
            # Yield the same response multiple times
            for _ in range(agent.max_iterations):
                yield LLM_JSON_RESPONSE_LIST_FILES

        mock_call_llm.return_value = mock_llm_repeat_list()

    # Mock the tool executor to always return success
    list_success_result = {
        "status": "success",
        "action": "list_files_success",
        "data": {"files": ["file.txt"]},
    }
    mock_executor = mocker.patch(
        "core.tool_executor.execute_tool", return_value=list_success_result
    )

    # Mock reflector to always continue
    mock_reflector = mocker.patch(
        "core.agent_reflector.reflect_on_observation",
        new_callable=AsyncMock,
        return_value=("continue_plan", mock_plan),
    )

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    # Check LLM was called max_iterations times
    assert mock_call_llm.call_count == agent.max_iterations
    assert mock_executor.call_count == agent.max_iterations
    assert mock_reflector.call_count == agent.max_iterations
    assert (
        f"Erro: Maximum total iterations ({agent.max_iterations}) reached"
        in final_response
    )
    mock_db.assert_called_once()  # Should still save state


@pytest.mark.asyncio
async def test_react_agent_run_handles_failed_planning(agent_instance, mock_db, mocker):
    """Testa se o agente lida com a falha na geração do plano inicial."""
    agent = agent_instance  # Use configured agent

    objective = "Objective that causes planning failure"
    # Mock planner to return None (failure)
    mock_planner = mocker.patch(
        "core.planner.generate_plan", new_callable=AsyncMock, return_value=None
    )

    with patch("core.agent.call_llm", new_callable=AsyncMock) as mock_call_llm:

        async def mock_llm_failed_plan_response():
            yield json.dumps(
                {
                    "Thought": "Planning failed, I'll try the objective directly...",
                    "Action": "final_answer",
                    "Action Input": {
                        "answer": "Could not determine steps for the complex objective."
                    },
                }
            )

        mock_call_llm.return_value = mock_llm_failed_plan_response()

    # Mock reflector
    mock_reflector = mocker.patch(
        "core.agent_reflector.reflect_on_observation",
        new_callable=AsyncMock,
        return_value=("plan_complete", None),
    )  # Assume reflector sees final_answer

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response_dict = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    mock_call_llm.assert_called_once()
    mock_reflector.assert_called_once()
    reflector_call_args = mock_reflector.call_args.kwargs
    assert (
        reflector_call_args.get("action_name") == "_parse_llm"
    )  # Internal action for parsing
    assert reflector_call_args.get("observation_dict", {}).get("status") == "error"
    assert (
        reflector_call_args.get("observation_dict", {}).get("action")
        == "parsing_failed"
    )
    assert "Erro: Falha ao processar resposta do LLM" in final_response_dict
    mock_db.assert_called_once()
