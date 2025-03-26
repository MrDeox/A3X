"""
Testes do módulo core do A³X.
"""

import pytest
from core import Executor, analyze_intent, run_python_code, execute_terminal_command, run_llm

def test_analyze_intent():
    """Testa a análise de intenções."""
    # Teste de saudação
    intent = analyze_intent("Olá, como vai?")
    assert intent['type'] == 'greeting'
    
    # Teste de armazenamento na memória
    intent = analyze_intent("lembre-se que o céu é azul como cor_ceu")
    assert intent['type'] == 'memory'
    assert intent['action'] == 'store'
    assert intent['content'] == 'o céu é azul'
    assert intent['target'] == 'cor_ceu'
    
    # Teste de recuperação da memória
    intent = analyze_intent("qual era cor_ceu")
    assert intent['type'] == 'memory'
    assert intent['action'] == 'retrieve'
    assert intent['target'] == 'cor_ceu'
    
    # Teste de comando do terminal
    intent = analyze_intent("execute o comando ls -la")
    assert intent['type'] == 'terminal'
    assert intent['action'] == 'run'
    assert intent['content'] == 'ls -la'
    
    # Teste de código Python
    intent = analyze_intent("execute o código python print('hello')")
    assert intent['type'] == 'python'
    assert intent['action'] == 'run'
    assert intent['content'] == "print('hello')"
    
    # Teste de pergunta
    intent = analyze_intent("qual é a capital do Brasil?")
    assert intent['type'] == 'question'
    assert intent['action'] == 'ask'
    assert intent['content'] == "qual é a capital do Brasil?"

def test_run_python_code():
    """Testa a execução de código Python."""
    # Teste de código simples
    result = run_python_code("print('hello')")
    assert result['status'] == 'success'
    assert result['output'] == 'hello\n'
    
    # Teste de código com erro
    result = run_python_code("print(x)")
    assert result['status'] == 'error'
    assert 'NameError' in result['error']

def test_execute_terminal_command():
    """Testa a execução de comandos do terminal."""
    # Teste de comando permitido
    result = execute_terminal_command("ls")
    assert result['status'] == 'success'
    
    # Teste de comando não permitido
    result = execute_terminal_command("rm -rf /")
    assert result['status'] == 'error'
    assert 'não permitido' in result['error']

def test_run_llm():
    """Testa o processamento de linguagem natural."""
    # Teste de geração de texto
    result = run_llm("Olá, como vai?")
    assert result['status'] == 'success'
    assert isinstance(result['response'], str)
    assert len(result['response']) > 0

def test_executor():
    """Testa o Executor."""
    executor = Executor()
    
    # Teste de processamento de comando
    result = executor.process_command("Olá, como vai?")
    assert result['status'] == 'success'
    assert 'Olá' in result['response']
    
    # Teste de comando inválido
    result = executor.process_command("comando inválido")
    assert result['status'] == 'error'
    assert 'não entendi' in result['response'].lower() 