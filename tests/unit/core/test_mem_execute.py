# tests/unit/core/test_mem_execute.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Ensure a3x can be imported (adjust path as needed)
# Assuming tests are run from the project root
sys.path.insert(0, ".") 

# Import the function to test AFTER modifying path
from a3x.core.mem import execute

@pytest.mark.asyncio
async def test_execute_calls_a3lang_interpreter():
    """Tests if mem.execute correctly calls the a3lang interpreter by default."""
    command_to_run = "RESPONDER 'teste'"
    mock_context = MagicMock()
    mock_context.tool_executor = AsyncMock()
    mock_context.llm_interface = AsyncMock()
    expected_result = {"status": "success", "message": "Comando executado"}
    mock_parse_execute = AsyncMock(return_value=expected_result)
    with patch('a3x.a3lang.interpreter.parse_and_execute', mock_parse_execute): 
        actual_result = await execute(command_to_run, context=mock_context)
    mock_parse_execute.assert_awaited_once_with(command_to_run, execution_context=mock_context)
    assert actual_result == expected_result

@pytest.mark.asyncio
async def test_execute_handles_interpreter_import_error():
    """Tests if mem.execute handles ImportError gracefully."""
    command_to_run = "COMANDO_QUALQUER"
    mock_context = MagicMock()
    expected_error_result = {"status": "error", "message": "Interpreter import failed: No module named 'a3x.a3lang.nonexistent'"}

    # Patch the import to raise an ImportError
    with patch('a3x.a3lang.interpreter.parse_and_execute', side_effect=ImportError("No module named 'a3x.a3lang.nonexistent'")):
        actual_result = await execute(command_to_run, context=mock_context)
    assert actual_result["status"] == "error"
    assert "Symbolic interpreter import failed" in actual_result["message"]
    assert "No module named 'a3x.a3lang.nonexistent'" in actual_result["message"]

@pytest.mark.asyncio
async def test_execute_handles_interpreter_execution_error():
    """Tests if mem.execute handles exceptions during interpreter execution."""
    command_to_run = "COMANDO_COM_ERRO"
    mock_context = MagicMock()
    runtime_error_message = "Erro simulado na execução"
    expected_error_result = {"status": "error", "message": f"Execution failed: {runtime_error_message}"}

    # Mock the interpreter to raise an exception
    mock_parse_execute = AsyncMock(side_effect=RuntimeError(runtime_error_message))

    with patch('a3x.a3lang.interpreter.parse_and_execute', mock_parse_execute):
        actual_result = await execute(command_to_run, context=mock_context)
    mock_parse_execute.assert_awaited_once_with(command_to_run, execution_context=mock_context)
    assert actual_result["status"] == "error"
    assert actual_result["message"] == f"Symbolic execution failed: {runtime_error_message}"

@pytest.mark.asyncio
async def test_execute_symbolic_basic():
    """Tests the default symbolic execution path with a simple command."""
    command_to_run = "RESPONDER 'olá'"
    expected_result = {"status": "success", "detail": "Responding with 'olá'"} # Example success result
    mock_context = MagicMock()
    mock_parse_execute = AsyncMock(return_value=expected_result)
    with patch('a3x.a3lang.interpreter.parse_and_execute', mock_parse_execute):
        actual_result = await execute(command_to_run, context=mock_context)
    mock_parse_execute.assert_awaited_once_with(command_to_run, execution_context=mock_context)
    assert actual_result == expected_result

@pytest.mark.asyncio
async def test_execute_unknown_mode():
    """Tests that an unknown mode returns an error."""
    command_to_run = "ALGUM_COMANDO"
    unknown_mode = "gibberish"
    
    actual_result = await execute(command_to_run, mode=unknown_mode)
    
    assert actual_result["status"] == "error"
    assert f"Modo desconhecido: {unknown_mode}" in actual_result["message"]

@pytest.mark.asyncio
async def test_execute_neural_mode_not_implemented():
    """Tests that the neural mode raises NotImplementedError."""
    command_to_run = "ALGUM_COMANDO_NEURAL"
    
    with pytest.raises(NotImplementedError, match="Modo 'neural' ainda não implementado."):
        await execute(command_to_run, mode="neural") 