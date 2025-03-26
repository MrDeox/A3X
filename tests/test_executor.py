"""
Testes de integração para o Executor Principal do A³X.
"""

from core.executor import Executor
import pytest
import time
import os
from pathlib import Path

executor = Executor()

# Verifica se o binário do LLM existe
llm_binary = Path('bin/llama-cli')
has_llm = llm_binary.exists()

def test_memoria_store_and_retrieve():
    """Testa operações de memória via Executor."""
    # Gera uma chave única baseada no timestamp
    chave = f"teste_memoria_{int(time.time())}"
    
    # Armazenar valor
    result_store = executor.process_command(f"lembre {chave} isto é um teste")
    assert "sucesso" in result_store.lower() or "ok" in result_store.lower()
    
    # Recuperar valor
    result_retrieve = executor.process_command(f"recupere {chave}")
    assert "isto é um teste" in result_retrieve.lower()

def test_codigo_python():
    """Testa execução de código Python via Executor."""
    result = executor.process_command("rode python print(3 * 7)")
    assert "21" in result

def test_comando_terminal():
    """Testa execução de comandos shell via Executor."""
    result = executor.process_command("execute o comando echo funcionando")
    assert "funcionando" in result.lower()

@pytest.mark.skipif(not has_llm, reason="Binário do LLM não encontrado")
def test_pergunta_llm():
    """Testa processamento de perguntas via LLM."""
    result = executor.process_command("Qual é a capital do Brasil?")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "brasília" in result.lower() or "brasilia" in result.lower()

@pytest.mark.skipif(not has_llm, reason="Binário do LLM não encontrado")
def test_instrucao_llm():
    """Testa geração de código via LLM."""
    result = executor.process_command("Crie uma função que calcule o fatorial")
    assert "def" in result or "fatorial" in result.lower()
    assert "return" in result.lower()

def test_avaliacao_falha():
    """Testa tratamento de erros na execução de código."""
    result = executor.process_command("rode python print(variavel_nao_definida)")
    assert "erro" in result.lower() or "não definida" in result.lower()

def test_historico_comandos():
    """Testa se o Executor mantém histórico de comandos."""
    # Executa alguns comandos
    executor.process_command("lembre teste_historico comando 1")
    executor.process_command("rode python print('comando 2')")
    
    # Verifica se os comandos estão no histórico
    assert len(executor.command_history) >= 2
    assert any("comando 1" in str(cmd) for cmd in executor.command_history)
    assert any("comando 2" in str(cmd) for cmd in executor.command_history)

def test_contexto_persistente():
    """Testa se o Executor mantém contexto entre comandos."""
    # Define contexto
    executor.process_command("lembre contexto_teste valor inicial")
    
    # Usa contexto em outro comando
    result = executor.process_command("rode python print('valor inicial')")
    assert "valor inicial" in result.lower()
    
    # Verifica se o contexto foi mantido
    context_result = executor.process_command("recupere contexto_teste")
    assert "valor inicial" in context_result.lower() 