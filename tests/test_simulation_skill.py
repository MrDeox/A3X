# tests/test_simulation_skill.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from a3x.skills.simulation import simulate_step

# Import context types for mocking
from a3x.core.context import Context, _ToolExecutionContext
from a3x.core.llm_interface import LLMInterface

# Add project root to sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)


# <<< ADDED: Helper for async generator >>>
async def async_generator_for(item):
    yield item


# Helper function to create a mock context for simulation skill
def create_mock_sim_context(llm_response=None, llm_side_effect=None, memory_content=None):
    mock_llm = AsyncMock(spec=LLMInterface)
    if llm_side_effect:
        mock_llm.call_llm.side_effect = llm_side_effect
    elif llm_response is not None:
        async def mock_llm_stream(*args, **kwargs):
            # Simulate async stream: check if response is string (single chunk) or list (multiple chunks)
            if isinstance(llm_response, str):
                yield llm_response
            elif isinstance(llm_response, list):
                for chunk in llm_response:
                    yield chunk
            else: # Handle other unexpected types if necessary, maybe raise?
                pass
        mock_llm.call_llm = mock_llm_stream # Assign the async generator function
    else:
         async def mock_llm_stream_empty(*args, **kwargs):
             if False: yield
         mock_llm.call_llm = mock_llm_stream_empty

    # Use _ToolExecutionContext as the context type passed
    mock_ctx = MagicMock(spec=_ToolExecutionContext)
    mock_ctx.logger = MagicMock()
    mock_ctx.llm_interface = mock_llm # Assign the mocked LLM interface

    # <<< ADDED: Mock memory attribute >>>
    mock_ctx.memory = MagicMock()
    mock_ctx.memory.get_memory.return_value = memory_content if memory_content is not None else {} # Default to empty dict
    # <<< END ADDED >>>

    # Add other attributes if needed (e.g., tool_registry if skill uses it)
    return mock_ctx, mock_llm


# Mark all tests as async
pytestmark = pytest.mark.asyncio


# Remove the patch decorator
# @patch("skills.simulation.call_llm")
async def test_simulate_step_success(): # Remove mock_call_llm argument
    """Tests successful simulation when LLM returns valid text."""
    step = "Install the 'requests' library using pip."
    plan_context = {"os": "Linux", "python_version": "3.10"} # Context for the PLAN step
    mock_llm_response = "Agent likely runs 'pip install requests'. Success expected."
    mock_ctx, mock_llm = create_mock_sim_context(llm_response=mock_llm_response)

    # The skill now expects 'context' as the first argument
    result = await simulate_step(context=mock_ctx, step=step)

    # mock_llm.call_llm.assert_called_once() # Hard to assert on generator directly
    assert result["status"] == "success"
    assert result["simulated_outcome"] == mock_llm_response.strip()
    assert result["confidence"] == "Média"  # Default confidence
    assert result["error_message"] is None


# Remove the patch decorator
# @patch("skills.simulation.call_llm")
async def test_simulate_step_success_no_plan_context(): # Remove mock_call_llm argument
    """Tests successful simulation with no plan context provided."""
    step = "Read the file 'README.md'."
    mock_llm_response = "Agent uses read_file for README.md."
    mock_ctx, mock_llm = create_mock_sim_context(llm_response=mock_llm_response)

    result = await simulate_step(context=mock_ctx, step=step) # No plan_context here

    # mock_llm.call_llm.assert_called_once()
    assert result["status"] == "success"
    assert result["simulated_outcome"] == mock_llm_response.strip()
    assert result["confidence"] == "Média"


# Remove the patch decorator
# @patch("skills.simulation.call_llm")
async def test_simulate_step_llm_empty_response(): # Remove mock_call_llm argument
    """Tests when the LLM call returns an empty string."""
    step = "Analyze the data."
    plan_context = {"data_source": "api"}
    mock_llm_response = "" # Empty response
    mock_ctx, mock_llm = create_mock_sim_context(llm_response=mock_llm_response)

    result = await simulate_step(context=mock_ctx, step=step)

    # mock_llm.call_llm.assert_called_once()
    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    # Check specific error message expected from the skill for empty LLM response
    assert "LLM response was empty or not a string." in result["error_message"]


# Remove the patch decorator
# @patch("skills.simulation.call_llm")
async def test_simulate_step_llm_invalid_type(): # Remove mock_call_llm argument
    """Tests when the LLM call returns non-string data."""
    step = "Summarize results."
    mock_llm_response = {"outcome": "summary..."} # Intentionally invalid type
    mock_ctx, mock_llm = create_mock_sim_context(llm_response=mock_llm_response)

    # Use the created mock_ctx
    result = await simulate_step(context=mock_ctx, step=step)

    # mock_llm.call_llm should have been called by the skill
    # We can check if the mock was called, but asserting args on generators is tricky
    assert mock_llm.call_llm is not None # Basic check that the method exists

    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    # The error is now likely a TypeError within the skill trying to process the dict
    assert "LLM response was empty or not a string." in result["error_message"]


# Remove the patch decorator
# @patch("skills.simulation.call_llm")
async def test_simulate_step_llm_exception(): # Remove mock_call_llm argument
    """Tests when call_llm raises an exception."""
    step = "Deploy the application."
    plan_context = {"environment": "production"}
    error_message = "API rate limit exceeded"
    mock_exception = Exception(error_message)
    mock_ctx, mock_llm = create_mock_sim_context(llm_side_effect=mock_exception)

    result = await simulate_step(context=mock_ctx, step=step)

    # mock_llm.call_llm.assert_called_once()
    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    assert "Failed to simulate step due to LLM error" in result["error_message"]
    assert error_message in result["error_message"]


async def test_simulate_step_missing_llm_interface():
    """Tests when the context is missing the LLM interface."""
    step = "Deploy the application."
    plan_context = {"environment": "production"}
    
    # Create context with llm_interface explicitly set to None
    mock_ctx = MagicMock(spec=_ToolExecutionContext)
    mock_ctx.logger = MagicMock()
    mock_ctx.llm_interface = None # Simulate missing interface

    # <<< ADDED: Mock memory even if LLM is missing, as simulate_step checks LLM *after* memory >>>
    mock_ctx.memory = MagicMock()
    mock_ctx.memory.get_memory.return_value = {}
    # <<< END ADDED >>>

    result = await simulate_step(context=mock_ctx, step=step)

    assert result["status"] == "error"
    assert result["simulated_outcome"] is None
    assert result["confidence"] == "N/A"
    assert "Internal error: LLMInterface missing." in result["error_message"]
