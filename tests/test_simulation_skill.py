# tests/test_simulation_skill.py
import pytest
from unittest.mock import patch, AsyncMock
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from skills.simulation import simulate_step, SIMULATE_STEP_PROMPT_TEMPLATE

# Mark all tests as async
pytestmark = pytest.mark.asyncio

@patch('skills.simulation.call_llm', new_callable=AsyncMock)
async def test_simulate_step_success(mock_call_llm):
    """Tests successful simulation when LLM returns valid text."""
    step = "Install the 'requests' library using pip."
    context = {"os": "Linux", "python_version": "3.10"}
    mock_llm_response = "O agente provavelmente executará 'pip install requests'. O resultado esperado é a instalação bem-sucedida da biblioteca."
    mock_call_llm.return_value = mock_llm_response

    expected_prompt = SIMULATE_STEP_PROMPT_TEMPLATE.format(step=step, context=context)

    result = await simulate_step(step=step, context=context)

    mock_call_llm.assert_awaited_once_with(expected_prompt, stream=False)
    assert result['status'] == "success"
    assert result['simulated_outcome'] == mock_llm_response
    assert result['confidence'] == "Média" # Default confidence
    assert result['error_message'] is None

@patch('skills.simulation.call_llm', new_callable=AsyncMock)
async def test_simulate_step_success_no_context(mock_call_llm):
    """Tests successful simulation with no context provided."""
    step = "Read the file 'README.md'."
    mock_llm_response = "O agente usará a skill read_file para ler o conteúdo de README.md."
    mock_call_llm.return_value = mock_llm_response

    # Context defaults to {}
    expected_prompt = SIMULATE_STEP_PROMPT_TEMPLATE.format(step=step, context="Nenhum contexto fornecido.")

    result = await simulate_step(step=step)

    mock_call_llm.assert_awaited_once_with(expected_prompt, stream=False)
    assert result['status'] == "success"
    assert result['simulated_outcome'] == mock_llm_response
    assert result['confidence'] == "Média"

@patch('skills.simulation.call_llm', new_callable=AsyncMock)
async def test_simulate_step_llm_empty_response(mock_call_llm):
    """Tests when the LLM call returns an empty string."""
    step = "Analyze the data."
    context = {"data_source": "api"}
    mock_call_llm.return_value = "" # Empty response

    result = await simulate_step(step=step, context=context)

    assert result['status'] == "error"
    assert result['simulated_outcome'] is None
    assert result['confidence'] == "N/A"
    assert "LLM simulation response was empty or not a string" in result['error_message']

@patch('skills.simulation.call_llm', new_callable=AsyncMock)
async def test_simulate_step_llm_invalid_type(mock_call_llm):
    """Tests when the LLM call returns non-string data."""
    step = "Summarize results."
    context = {}
    mock_call_llm.return_value = {"outcome": "summary..."} # Invalid type

    result = await simulate_step(step=step, context=context)

    assert result['status'] == "error"
    assert result['simulated_outcome'] is None
    assert result['confidence'] == "N/A"
    assert "LLM simulation response was empty or not a string" in result['error_message']

@patch('skills.simulation.call_llm', new_callable=AsyncMock)
async def test_simulate_step_llm_exception(mock_call_llm):
    """Tests when call_llm raises an exception."""
    step = "Deploy the application."
    context = {"environment": "production"}
    error_message = "API rate limit exceeded"
    mock_call_llm.side_effect = Exception(error_message)

    result = await simulate_step(step=step, context=context)

    assert result['status'] == "error"
    assert result['simulated_outcome'] is None
    assert result['confidence'] == "N/A"
    assert "Failed to simulate step due to LLM error" in result['error_message']
    assert error_message in result['error_message'] 