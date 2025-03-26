"""
Testes de integração para comandos em linguagem natural do A³X.
Foca em testes realistas simulando o uso prático do sistema.
"""

from core.executor import Executor
import pytest
import time

executor = Executor()

def test_memoria_natural():
    """Testa operações de memória com comandos naturais."""
    # Gera uma chave única baseada no timestamp
    chave = f"tarefa_{int(time.time())}"
    
    # Armazenar valor
    result_store = executor.process_command(f"lembre {chave} revisar código do módulo core")
    assert "sucesso" in result_store.lower() or "ok" in result_store.lower()
    
    # Recuperar valor
    result_retrieve = executor.process_command(f"recupere {chave}")
    assert "revisar código do módulo core" in result_retrieve.lower()

def test_comandos_terminal_naturais():
    """Testa comandos de terminal com linguagem natural."""
    # Teste simples
    result = executor.process_command("execute o comando echo testando")
    assert "testando" in result.lower()
    
    # Teste com ls
    result = executor.process_command("rode o comando ls -la")
    assert "bin" in result.lower() and "models" in result.lower()

def test_codigo_python_natural():
    """Testa execução de código Python com comandos naturais."""
    # Teste simples
    result = executor.process_command("rode python print('teste ok')")
    assert "teste ok" in result
    
    # Teste com cálculo
    result = executor.process_command("execute python print(2 + 2)")
    assert "4" in result

def test_perguntas_llm_naturais():
    """Testa perguntas ao LLM com linguagem natural."""
    # Pergunta sobre capital
    result = executor.process_command("Qual é a capital do Japão?")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "tóquio" in result.lower() or "tokyo" in result.lower()
    
    # Pergunta sobre programação
    result = executor.process_command("Como criar uma lista em Python?")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "python" in result.lower() or "lista" in result.lower()

def test_instrucoes_llm_naturais():
    """Testa instruções ao LLM com linguagem natural."""
    # Geração de código
    result = executor.process_command("Crie uma função que receba um número e retorne o fatorial")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "def" in result or "fatorial" in result.lower()
    
    # Explicação de conceito
    result = executor.process_command("Explique o que é recursão em programação")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "recursão" in result.lower() or "recursao" in result.lower()

def test_comandos_compostos():
    """Testa comandos que combinam diferentes funcionalidades."""
    # Armazenar resultado de comando
    chave = f"resultado_{int(time.time())}"
    result = executor.process_command(f"lembre {chave} execute o comando date")
    assert "sucesso" in result.lower() or "ok" in result.lower()
    
    # Recuperar e verificar
    result = executor.process_command(f"recupere {chave}")
    assert isinstance(result, str)
    assert len(result.strip()) > 0

def test_comandos_complexos():
    """Testa comandos mais complexos e específicos."""
    # Geração de código com requisitos específicos
    result = executor.process_command("Crie uma função que calcule a média de uma lista de números")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "def" in result or "média" in result.lower() or "media" in result.lower()
    
    # Pergunta técnica
    result = executor.process_command("Qual é a diferença entre list e tuple em Python?")
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "python" in result.lower() 