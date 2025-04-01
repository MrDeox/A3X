import json
from unittest.mock import MagicMock, AsyncMock
import pytest
from core.agent_reflector import Decision


# Assuming APIError is defined somewhere, e.g., in a shared exceptions module
class APIError(Exception):
    def __init__(self, message, request, body):
        super().__init__(message)
        self.request = request
        self.body = body


# --- Agent Run Basic Tests ---
@pytest.mark.asyncio
async def test_react_agent_run_list_files(
    agent_instance,
    mock_db,
    mocker,
    LLM_JSON_RESPONSE_LIST_FILES,
    LLM_JSON_RESPONSE_LIST_FILES_FINAL,
    LIST_FILES_RESULT_JSON,
):
    """Testa um fluxo básico com a ferramenta list_files e final_answer, agora com planejamento."""
    agent, _ = agent_instance  # Don't need the pre-mocked _call_llm from fixture

    mock_plan = [
        "Step 1: List files in current directory.",
        "Step 2: Provide the final answer.",
    ]
    # <<< Use returned mock >>>
    mock_planner = mocker.patch(
        "core.agent.planner.generate_plan", return_value=mock_plan
    )

    mock_call_llm = mocker.patch.object(agent, "_call_llm", new_callable=AsyncMock)
    mock_call_llm.side_effect = [
        LLM_JSON_RESPONSE_LIST_FILES,
        LLM_JSON_RESPONSE_LIST_FILES_FINAL,
    ]

    # <<< Patch the TOOLS dictionary entry >>>
    mock_list_files_func = MagicMock(return_value=json.loads(LIST_FILES_RESULT_JSON))
    mocker.patch.dict("core.agent.TOOLS", {"list_files": mock_list_files_func})

    objective = "List the files in my current directory."
    final_response = await agent.run(objective)

    # Assertions
    mock_planner.assert_called_once()  # <<< Use the mock planner object
    assert mock_call_llm.call_count == 2
    mock_list_files_func.assert_called_once()  # <<< Use the mock tool func
    assert len(agent._history) == 5  # Human, Plan, LLM1, Obs1, LLM2, FinalAnswer(Obs2)
    assert "Files listed: mock_file.txt" in final_response


@pytest.mark.asyncio
async def test_react_agent_run_final_answer_direct(
    agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_HELLO_FINAL
):
    """Testa o caso onde o LLM retorna final_answer diretamente (agora após plano de 1 passo)."""
    agent, _ = agent_instance

    objective = "Just give me the final answer immediately."
    mock_plan = [objective]
    # <<< Use returned mock >>>
    mock_planner = mocker.patch(
        "core.agent.planner.generate_plan", return_value=mock_plan
    )

    mock_call_llm = mocker.patch.object(agent, "_call_llm", new_callable=AsyncMock)
    mock_call_llm.return_value = LLM_JSON_RESPONSE_HELLO_FINAL

    final_response = await agent.run(objective)

    # Assertions
    mock_planner.assert_called_once()  # <<< Use the mock planner object
    mock_call_llm.assert_called_once()
    assert "Hello there!" in final_response
    assert len(agent._history) == 3  # Human, Plan, LLM1, FinalAnswer(Obs1)


# --- Agent Run Error Tests ---
@pytest.mark.asyncio
async def test_react_agent_run_handles_llm_call_error(
    agent_instance, mock_code_tools, mock_db, mocker
):
    """Testa se o agente lida com um erro de chamada LLM (agora com plano)."""
    agent, _ = agent_instance

    objective = "Test LLM call error handling......"
    mock_plan = ["Step 1: This step's LLM call will fail."]
    # <<< Use returned mock >>>
    mock_planner = mocker.patch(
        "core.agent.planner.generate_plan", return_value=mock_plan
    )

    error_message = "LLM API call failed"
    llm_error = APIError(message=error_message, request=None, body=None)
    mock_call_llm = mocker.patch.object(agent, "_call_llm", new_callable=AsyncMock)
    mock_call_llm.side_effect = llm_error

    # Mock reflector (Decision import is needed here)
    mock_reflector = mocker.patch(
        "core.agent.agent_reflector.reflect_on_observation", new_callable=AsyncMock
    )
    # <<< Simulate reflector deciding retry_step for LLM error >>>
    mock_reflector.return_value = (Decision.retry_step, None)

    final_response = await agent.run(objective=objective)

    # Assertions
    mock_planner.assert_called_once()  # <<< Use the mock planner object
    mock_call_llm.assert_called_once()
    mock_reflector.assert_called_once()
    # <<< Check final response when reflector suggests retry_step (agent currently stops) >>>
    assert (
        "Erro: Falha ao comunicar com LLM" in final_response
    )  # Agent stops on LLM error before reflector decision matters here
    assert error_message in final_response
    # History length should reflect the attempt and the error observation
    assert len(agent._history) == 3  # Human, Plan, Obs(LLM Error)


@pytest.mark.asyncio
async def test_react_agent_run_handles_tool_execution_error(
    agent_instance,
    mock_code_tools,
    mock_db,
    mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE,
    EXECUTE_CODE_RESULT_ERROR_JSON,
):
    """Testa se o agente lida com um erro retornado pela execução de uma tool (agora com plano e reflector)."""
    agent, _ = agent_instance
    # mock_execute = mock_code_tools # Need to mock the actual function

    objective = "Execute este código Python: print(1/0)"
    mock_plan = ["Step 1: Execute the failing code."]
    # <<< Use returned mock >>>
    mock_planner = mocker.patch(
        "core.agent.planner.generate_plan", return_value=mock_plan
    )

    mock_call_llm = mocker.patch.object(agent, "_call_llm", new_callable=AsyncMock)
    mock_call_llm.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE

    # <<< Patch TOOLS dictionary entry >>>
    mock_execute_func = MagicMock(
        return_value=json.loads(EXECUTE_CODE_RESULT_ERROR_JSON)
    )
    mocker.patch.dict("core.agent.TOOLS", {"execute_code": mock_execute_func})

    # Mock the reflector call
    mock_reflector = mocker.patch(
        "core.agent.agent_reflector.reflect_on_observation", new_callable=AsyncMock
    )
    # <<< Simulate reflector returning stop_plan after failed auto-correct >>>
    mock_reflector.return_value = (Decision.stop_plan, None)

    result = await agent.run(objective)

    # Assertions
    mock_planner.assert_called_once()  # <<< Use the mock planner object
    mock_call_llm.assert_called_once()
    mock_execute_func.assert_called_once()  # <<< Use the mock tool func
    mock_reflector.assert_called_once()

    # Check final result based on reflector deciding stop_plan
    assert "Erro: Plano interrompido pelo Reflector" in result
    # The specific error message depends on what the agent's main loop returns when reflector says stop_plan
    # Let's check for the reflector stopping message.
    assert "Falha na execução do código" in result
    assert "ZeroDivisionError: division by zero" in result
