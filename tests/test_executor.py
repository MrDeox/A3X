"""
Testes para o Executor Principal do A³X.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.executor import Executor
import time
import os
from pathlib import Path

@pytest.fixture
def executor():
    """Fixture que cria um executor limpo para cada teste."""
    return Executor()

def test_memory_store_command(executor):
    """Testa comando de armazenamento em memória."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.store') as mock_store:
        
        mock_intent.return_value = {
            "type": "memory",
            "action": "store",
            "target": "info_test",
            "content": "valor de teste"
        }
        
        result = executor.process_command("lembre info_test valor de teste")
        assert "sucesso" in result.lower()
        mock_store.assert_called_once_with("info_test", "valor de teste")

def test_memory_retrieve_command(executor):
    """Testa comando de recuperação da memória."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.retrieve') as mock_retrieve:
        
        mock_intent.return_value = {
            "type": "memory",
            "action": "retrieve",
            "target": "info_test",
            "content": "info_test"
        }
        mock_retrieve.return_value = "valor de teste"
        
        result = executor.process_command("recupere info_test")
        assert result == "valor de teste"
        mock_retrieve.assert_called_once_with("info_test")

def test_terminal_command(executor):
    """Testa comando de terminal."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.execute') as mock_execute:
        
        mock_intent.return_value = {
            "type": "terminal",
            "action": "run",
            "target": None,
            "content": "ls -la"
        }
        mock_execute.return_value = "arquivo1.txt\narquivo2.txt"
        
        result = executor.process_command("rode o comando ls -la")
        assert "arquivo1.txt" in result
        mock_execute.assert_called_once_with("ls -la")

def test_python_code_command(executor):
    """Testa comando de código Python."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_python_code') as mock_run_python:
        
        mock_intent.return_value = {
            "type": "python",
            "action": "generate",
            "target": None,
            "content": "print(2 + 2)"
        }
        mock_run_python.return_value = "4"
        
        result = executor.process_command("rode python print(2 + 2)")
        assert result == "4"
        mock_run_python.assert_called_once_with("print(2 + 2)")

def test_question_command(executor):
    """Testa comando de pergunta."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_llm') as mock_llm:
        
        mock_intent.return_value = {
            "type": "question",
            "action": "ask",
            "target": None,
            "content": "Qual é a capital do Japão?"
        }
        mock_llm.return_value = "A capital do Japão é Tóquio."
        
        result = executor.process_command("me diga a capital do Japão")
        assert "tóquio" in result.lower()
        mock_llm.assert_called_once_with("Qual é a capital do Japão?")

def test_instruction_command(executor):
    """Testa comando de instrução."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_llm') as mock_llm:
        
        mock_intent.return_value = {
            "type": "instruction",
            "action": "explain",
            "target": None,
            "content": "Como criar uma lista em Python?"
        }
        mock_llm.return_value = "Para criar uma lista em Python, use colchetes []..."
        
        result = executor.process_command("me explique como criar uma lista em Python")
        assert "lista" in result.lower()
        mock_llm.assert_called_once_with("Como criar uma lista em Python?")

def test_unknown_command(executor):
    """Testa comando desconhecido."""
    with patch('core.executor.analyze_intent') as mock_intent:
        mock_intent.return_value = {
            "type": "unknown",
            "action": "unknown",
            "target": None,
            "content": "comando inválido"
        }
        
        result = executor.process_command("comando inválido")
        assert "não entendi" in result.lower()

def test_auto_store_result(executor):
    """Testa armazenamento automático do resultado."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_llm') as mock_llm, \
         patch('core.executor.store') as mock_store:
        
        mock_intent.return_value = {
            "type": "question",
            "action": "ask",
            "target": "info_japao",
            "content": "Qual é a capital do Japão?"
        }
        mock_llm.return_value = "A capital do Japão é Tóquio."
        
        result = executor.process_command("me diga a capital do Japão e salve como info_japao")
        assert "tóquio" in result.lower()
        mock_store.assert_called_once_with("info_japao", "A capital do Japão é Tóquio.")

def test_memoria_store_and_retrieve(executor):
    """Testa operações de memória via Executor."""
    # Gera uma chave única baseada no timestamp
    chave = f"teste_memoria_{int(time.time())}"
    
    # Armazenar valor
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.store') as mock_store:
        
        mock_intent.return_value = {
            "type": "memory",
            "action": "store",
            "target": chave,
            "content": "isto é um teste"
        }
        
        result_store = executor.process_command(f"lembre {chave} isto é um teste")
        assert "sucesso" in result_store.lower()
        mock_store.assert_called_once_with(chave, "isto é um teste")
    
    # Recuperar valor
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.retrieve') as mock_retrieve:
        
        mock_intent.return_value = {
            "type": "memory",
            "action": "retrieve",
            "target": chave,
            "content": chave
        }
        mock_retrieve.return_value = "isto é um teste"
        
        result_retrieve = executor.process_command(f"recupere {chave}")
        assert "isto é um teste" in result_retrieve.lower()
        mock_retrieve.assert_called_once_with(chave)

def test_codigo_python(executor):
    """Testa execução de código Python via Executor."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_python_code') as mock_run_python:
        
        mock_intent.return_value = {
            "type": "python",
            "action": "run",
            "target": None,
            "content": "print(3 * 7)"
        }
        mock_run_python.return_value = "21"
        
        result = executor.process_command("rode python print(3 * 7)")
        assert "21" in result
        mock_run_python.assert_called_once_with("print(3 * 7)")

def test_comando_terminal(executor):
    """Testa execução de comandos shell via Executor."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.execute') as mock_execute:
        
        mock_intent.return_value = {
            "type": "terminal",
            "action": "run",
            "target": None,
            "content": "echo funcionando"
        }
        mock_execute.return_value = "funcionando"
        
        result = executor.process_command("execute o comando echo funcionando")
        assert "funcionando" in result
        mock_execute.assert_called_once_with("echo funcionando")

# Verifica se o binário do LLM existe
llm_binary = Path('bin/llama-cli')
has_llm = llm_binary.exists()

@pytest.mark.skipif(not has_llm, reason="Binário do LLM não encontrado")
def test_pergunta_llm(executor):
    """Testa processamento de perguntas via LLM."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_llm') as mock_llm:
        
        mock_intent.return_value = {
            "type": "question",
            "action": "ask",
            "target": None,
            "content": "Qual é a capital do Brasil?"
        }
        mock_llm.return_value = "A capital do Brasil é Brasília."
        
        result = executor.process_command("Qual é a capital do Brasil?")
        assert "brasília" in result.lower()
        mock_llm.assert_called_once_with("Qual é a capital do Brasil?")

@pytest.mark.skipif(not has_llm, reason="Binário do LLM não encontrado")
def test_instrucao_llm(executor):
    """Testa geração de código via LLM."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_llm') as mock_llm:
        
        mock_intent.return_value = {
            "type": "instruction",
            "action": "generate",
            "target": None,
            "content": "Crie uma função que calcule o fatorial"
        }
        mock_llm.return_value = "def fatorial(n):\n    return 1 if n <= 1 else n * fatorial(n-1)"
        
        result = executor.process_command("Crie uma função que calcule o fatorial")
        assert "def fatorial" in result
        mock_llm.assert_called_once_with("Crie uma função que calcule o fatorial")

def test_avaliacao_falha(executor):
    """Testa tratamento de erros na execução de código."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.run_python_code') as mock_run_python:
        
        mock_intent.return_value = {
            "type": "python",
            "action": "run",
            "target": None,
            "content": "print(variavel_nao_definida)"
        }
        mock_run_python.return_value = "Erro: Variável não definida: name 'variavel_nao_definida' is not defined"
        
        result = executor.process_command("rode python print(variavel_nao_definida)")
        assert "erro" in result.lower()
        assert "variável não definida" in result.lower()

def test_historico_comandos(executor):
    """Testa se o Executor mantém histórico de comandos."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.store') as mock_store:
        
        mock_intent.return_value = {
            "type": "memory",
            "action": "store",
            "target": "teste_historico",
            "content": "comando 1"
        }
        
        # Executa alguns comandos
        executor.process_command("lembre teste_historico comando 1")
        
        # Verifica histórico
        assert len(executor.history) == 1
        assert executor.history[0]['command'] == "lembre teste_historico comando 1"
        assert executor.history[0]['intent']['type'] == "memory"

def test_contexto_persistente(executor):
    """Testa se o Executor mantém contexto entre comandos."""
    with patch('core.executor.analyze_intent') as mock_intent, \
         patch('core.executor.store') as mock_store, \
         patch('core.executor.retrieve') as mock_retrieve:
        
        # Define contexto
        mock_intent.return_value = {
            "type": "memory",
            "action": "store",
            "target": "contexto_teste",
            "content": "valor inicial"
        }
        executor.process_command("lembre contexto_teste valor inicial")
        
        # Verifica se o contexto foi mantido
        mock_intent.return_value = {
            "type": "memory",
            "action": "retrieve",
            "target": "contexto_teste",
            "content": "contexto_teste"
        }
        mock_retrieve.return_value = "valor inicial"
        
        result = executor.process_command("recupere contexto_teste")
        assert "valor inicial" in result 