# tests/test_final_answer.py
import pytest
from skills.final_answer import skill_final_answer

# Testes para a skill final_answer

def test_final_answer_success():
    """Testa o funcionamento básico com uma resposta válida."""
    answer_text = "A operação foi concluída com sucesso e o arquivo foi salvo."
    action_input = {"action": "final_answer", "answer": answer_text}

    result = skill_final_answer(action_input)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert "final_answer" in result["data"]
    assert result["data"]["final_answer"] == answer_text

def test_final_answer_missing_answer():
    """Testa o caso onde a chave 'answer' está ausente no input."""
    action_input = {"action": "final_answer"} # Sem a chave 'answer'

    result = skill_final_answer(action_input)

    assert result["status"] == "success" # A skill em si não falha
    assert result["action"] == "final_answer_provided"
    assert "final_answer" in result["data"]
    assert "N/A - Resposta final não fornecida" in result["data"]["final_answer"]

def test_final_answer_non_string_answer():
    """Testa o caso onde 'answer' não é uma string."""
    action_input = {"action": "final_answer", "answer": 12345}

    result = skill_final_answer(action_input)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert "final_answer" in result["data"]
    assert result["data"]["final_answer"] == "12345" # Deve converter para string

def test_final_answer_empty_input():
    """Testa o caso com um action_input vazio."""
    action_input = {}

    result = skill_final_answer(action_input)

    assert result["status"] == "success"
    assert result["action"] == "final_answer_provided"
    assert "final_answer" in result["data"]
    assert "N/A - Resposta final não fornecida" in result["data"]["final_answer"]

