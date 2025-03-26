"""
Testes para análise de intenção do A³X.
"""

import pytest
from unittest.mock import patch, MagicMock
from core import analyze_intent

def test_question_intent():
    """Testa análise de perguntas."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.return_value = '''{
            "type": "question",
            "action": "ask",
            "target": null,
            "content": "Qual é a capital do Japão?"
        }'''
        
        intent = analyze_intent("me diga a capital do Japão")
        assert intent["type"] == "question"
        assert intent["action"] == "ask"
        assert intent["target"] is None
        assert "capital do Japão" in intent["content"]

def test_memory_store_intent():
    """Testa análise de armazenamento em memória."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.return_value = '''{
            "type": "memory",
            "action": "store",
            "target": "info_japao",
            "content": "Capital do Japão é Tóquio"
        }'''
        
        intent = analyze_intent("lembre que a capital do Japão é Tóquio e salve como info_japao")
        assert intent["type"] == "memory"
        assert intent["action"] == "store"
        assert intent["target"] == "info_japao"
        assert "Tóquio" in intent["content"]

def test_terminal_command_intent():
    """Testa análise de comandos do terminal."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.return_value = '''{
            "type": "terminal",
            "action": "run",
            "target": null,
            "content": "ls -la"
        }'''
        
        intent = analyze_intent("rode o comando ls -la")
        assert intent["type"] == "terminal"
        assert intent["action"] == "run"
        assert intent["target"] is None
        assert intent["content"] == "ls -la"

def test_python_code_intent():
    """Testa análise de código Python."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.return_value = '''{
            "type": "python",
            "action": "generate",
            "target": null,
            "content": "def soma(a, b): return a + b"
        }'''
        
        intent = analyze_intent("crie uma função python que some dois números")
        assert intent["type"] == "python"
        assert intent["action"] == "generate"
        assert intent["target"] is None
        assert "def soma" in intent["content"]

def test_instruction_intent():
    """Testa análise de instruções."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.return_value = '''{
            "type": "instruction",
            "action": "explain",
            "target": null,
            "content": "Diferença entre list e tuple em Python"
        }'''
        
        intent = analyze_intent("qual a diferença entre list e tuple?")
        assert intent["type"] == "instruction"
        assert intent["action"] == "explain"
        assert intent["target"] is None
        assert "list e tuple" in intent["content"]

def test_invalid_json_handling():
    """Testa tratamento de JSON inválido."""
    with patch('core.core.run_llm') as mock_llm:
        # Primeira tentativa retorna JSON inválido
        mock_llm.side_effect = [
            "invalid json",
            '''{
                "type": "question",
                "action": "ask",
                "target": null,
                "content": "Qual é a capital do Japão?"
            }'''
        ]
        
        intent = analyze_intent("me diga a capital do Japão")
        assert intent["type"] == "question"
        assert intent["action"] == "ask"
        assert intent["target"] is None
        assert "capital do Japão" in intent["content"]

def test_error_handling():
    """Testa tratamento de erros."""
    with patch('core.core.run_llm') as mock_llm:
        mock_llm.side_effect = Exception("Erro no LLM")
        
        intent = analyze_intent("comando qualquer")
        assert intent["type"] == "unknown"
        assert intent["action"] == "unknown"
        assert intent["target"] is None
        assert intent["content"] == "comando qualquer" 