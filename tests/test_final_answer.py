# tests/test_final_answer.py
import pytest
from skills.final_answer import final_answer
from unittest.mock import MagicMock

# Testes para a skill final_answer

@pytest.mark.asyncio
async def test_final_answer_basic():
    """Testa a execução básica da skill final_answer."""
    expected_answer = "The final answer is 42."
    # O input agora é diretamente os argumentos da função
    result = final_answer(answer=expected_answer)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert "final_answer" in result["data"]
    assert result["data"]["final_answer"] == expected_answer

@pytest.mark.asyncio
async def test_final_answer_empty():
    """Testa com uma string vazia (deve funcionar)."""
    expected_answer = ""
    result = final_answer(answer=expected_answer)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert result["data"]["final_answer"] == expected_answer

@pytest.mark.asyncio
async def test_final_answer_long():
    """Testa com uma string longa."""
    expected_answer = "This is a very long answer " * 50
    result = final_answer(answer=expected_answer)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert result["data"]["final_answer"] == expected_answer

# Teste de erro de tipo não é mais necessário aqui,
# pois o Pydantic/decorator deve lidar com isso antes da chamada.

# <<< MODIFIED: These tests check how the function handles bad inputs internally >>>
# <<< The decorator validation prevents these calls in real scenarios >>>
# <<< Keep them to test internal robustness if needed, OR remove them >>>

def test_final_answer_missing_answer_internal():
    """Testa o caso onde 'answer' não está no input (simulando falha no decorador)."""
    # This simulates calling the function directly bypassing the decorator's validation
    # In a real scenario, the decorator would raise a validation error
    # We're testing the function's internal robustness to missing args (which it isn't designed for)
    with pytest.raises(TypeError) as excinfo:
        final_answer() # Call without the required argument
    assert "missing 1 required positional argument: 'answer'" in str(excinfo.value)

def test_final_answer_non_string_answer_internal():
    """Testa o caso onde 'answer' não é uma string (simulando falha no decorador)."""
    # Decorator handles type validation. This tests internal behavior if type check failed.
    # Python's type hinting doesn't prevent passing wrong types, but Pydantic does.
    # The function might still work due to implicit str conversion in logging, but this isn't guaranteed.
    result = final_answer(answer=12345) # Passing int instead of str
    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    # It likely converts to string for the message, but rely on decorator validation primarily.
    assert result["data"]["final_answer"] == "12345"

def test_final_answer_empty_input_internal():
    """Testa o caso com um input vazio (simulando falha no decorador)."""
    with pytest.raises(TypeError) as excinfo:
        final_answer() # Call without the required argument
    assert "missing 1 required positional argument: 'answer'" in str(excinfo.value)

