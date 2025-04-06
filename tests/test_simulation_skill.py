# tests/test_simulation_skill.py
import pytest
from unittest.mock import patch
from a3x.skills.simulation import simulate_step

# Add project root to sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)


# <<< ADDED: Helper for async generator >>>
async def async_generator_for(item):
    yield item


# Mark all tests as async
pytestmark = pytest.mark.asyncio


@patch("skills.simulation.call_llm")
async def test_simulate_step_success(mock_call_llm):
    """Tests successful simulation when LLM returns valid text."""
    step = "Install the 'requests' library using pip."
    context = {"os": "Linux", "python_version": "3.10"}
    mock_llm_response = "Agent likely runs 'pip install requests'. Success expected."
    mock_call_llm.return_value = async_generator_for(mock_llm_response)

    # F841: expected_prompt = SIMULATE_STEP_PROMPT_TEMPLATE.format(step=step, context=context)

    result = await simulate_step(step=step, context=context)

    mock_call_llm.assert_called_once()
    call_args, call_kwargs = mock_call_llm.call_args
    assert isinstance(call_args[0], list)
    assert not call_kwargs.get("stream")
    assert result["status"] == "success"
    assert result["simulated_outcome"] == mock_llm_response.strip()
    assert result["confidence"] == "Média"  # Default confidence
    assert result["error_message"] is None


@patch("skills.simulation.call_llm")
async def test_simulate_step_success_no_context(mock_call_llm):
    """Tests successful simulation with no context provided."""
    step = "Read the file 'README.md'."
    mock_llm_response = "Agent uses read_file for README.md."
    mock_call_llm.return_value = async_generator_for(mock_llm_response)

    result = await simulate_step(step=step)

    mock_call_llm.assert_called_once()
    call_args, call_kwargs = mock_call_llm.call_args
    assert isinstance(call_args[0], list)
    assert not call_kwargs.get("stream")
    assert result["status"] == "success"
    assert result["simulated_outcome"] == mock_llm_response.strip()
    assert result["confidence"] == "Média"


@patch("skills.simulation.call_llm")
async def test_simulate_step_llm_empty_response(mock_call_llm):
    """Tests when the LLM call returns an empty string."""
    step = "Analyze the data."
    context = {"data_source": "api"}
    mock_llm_response = ""
    mock_call_llm.return_value = async_generator_for(mock_llm_response)

    result = await simulate_step(step=step, context=context)

    mock_call_llm.assert_called_once()
    call_args, call_kwargs = mock_call_llm.call_args
    assert isinstance(call_args[0], list)
    assert not call_kwargs.get("stream")
    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    assert "empty or not a string" in result["error_message"]


@patch("skills.simulation.call_llm")
async def test_simulate_step_llm_invalid_type(mock_call_llm):
    """Tests when the LLM call returns non-string data."""
    step = "Summarize results."
    context = {}
    mock_llm_response = {"outcome": "summary..."}
    mock_call_llm.return_value = async_generator_for(mock_llm_response)

    result = await simulate_step(step=step, context=context)

    mock_call_llm.assert_called_once()
    call_args, call_kwargs = mock_call_llm.call_args
    assert isinstance(call_args[0], list)
    assert not call_kwargs.get("stream")
    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    assert "can only concatenate str" in result["error_message"]


@patch("skills.simulation.call_llm")
async def test_simulate_step_llm_exception(mock_call_llm):
    """Tests when call_llm raises an exception."""
    step = "Deploy the application."
    context = {"environment": "production"}
    error_message = "API rate limit exceeded"
    mock_call_llm.side_effect = Exception(error_message)

    result = await simulate_step(step=step, context=context)

    mock_call_llm.assert_called_once()
    call_args, call_kwargs = mock_call_llm.call_args
    assert isinstance(call_args[0], list)
    assert not call_kwargs.get("stream")
    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    assert "Failed to simulate step due to LLM error" in result["error_message"]
    assert error_message in result["error_message"]
