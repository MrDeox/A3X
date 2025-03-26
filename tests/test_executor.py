"""Testes para o Executor."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from core.executor import Executor
from memory.system import MemorySystem
from memory.models import SemanticMemoryEntry

@pytest.fixture
def memory_system():
    """Fixture que cria um MemorySystem mockado."""
    return MagicMock(spec=MemorySystem)

@pytest.fixture
def executor(memory_system):
    """Fixture que cria um Executor com MemorySystem mockado."""
    return Executor(memory_system=memory_system)

def test_initialization():
    """Testa a inicialização do executor."""
    # Sem parâmetros
    executor = Executor()
    assert executor.memory_system is None
    assert executor.command_history == []
    
    # Com memory system mock
    mock_memory = MagicMock(spec=MemorySystem)
    executor = Executor(memory_system=mock_memory)
    assert executor.memory_system == mock_memory
    assert executor.command_history == []

@patch('core.executor.datetime')
def test_execute_memory_store(mock_dt):
    """Testa o armazenamento em memória."""
    # Configura o mock de datetime
    fixed_dt = datetime(2024, 1, 1, 12, 0, 0)
    mock_dt.now.return_value = fixed_dt
    
    mock_memory = MagicMock(spec=MemorySystem)
    executor = Executor(memory_system=mock_memory)
    
    intent = {
        'type': 'memory',
        'action': 'store',
        'target': 'test_key',
        'content': 'test_value'
    }
    
    result = executor.execute(intent)
    assert result['status'] == 'success'
    assert result['action'] == 'store'
    assert result['message'] == 'Valor armazenado em test_key'
    
    mock_memory.store.assert_called_once_with(
        'test_key',
        SemanticMemoryEntry(
            key='test_key',
            value='test_value',
            timestamp=fixed_dt
        )
    )

def test_execute_memory_retrieve():
    """Testa a recuperação de memória."""
    mock_memory = MagicMock(spec=MemorySystem)
    mock_memory.retrieve.return_value = "test_value"
    executor = Executor(memory_system=mock_memory)
    
    intent = {
        'type': 'memory',
        'action': 'retrieve',
        'target': 'test_key'
    }
    
    result = executor.execute(intent)
    assert result['status'] == 'success'
    assert result['action'] == 'retrieve'
    assert result['value'] == 'test_value'
    
    mock_memory.retrieve.assert_called_once_with('test_key')

def test_execute_python():
    """Testa a execução de código Python."""
    executor = Executor()
    
    intent = {
        'type': 'python',
        'content': 'print("test")'
    }
    
    with patch('core.code_runner.run_python_code') as mock_run:
        mock_run.return_value = "test\n"
        result = executor.execute(intent)
        assert result['status'] == 'success'
        assert result['action'] == 'execute_python'
        assert result['output'] == 'test'
        mock_run.assert_called_once_with('print("test")')

def test_execute_terminal():
    """Testa a execução de comando terminal."""
    executor = Executor()
    
    intent = {
        'type': 'terminal',
        'content': 'ls'
    }
    
    with patch('core.code_runner.execute_terminal_command') as mock_run:
        mock_run.return_value = {'output': 'test output\n'}
        result = executor.execute(intent)
        assert result['status'] == 'success'
        assert result['action'] == 'execute_terminal'
        assert result['output'] == 'test output'
        mock_run.assert_called_once_with('ls')

def test_execute_question():
    """Testa o processamento de perguntas."""
    executor = Executor()
    
    intent = {
        'type': 'question',
        'content': 'Como você está?'
    }
    
    with patch('core.llm.run_llm') as mock_run:
        mock_run.return_value = "Estou bem, obrigado!\n"
        result = executor.execute(intent)
        assert result['status'] == 'success'
        assert result['action'] == 'ask'
        assert result['response'] == 'Estou bem, obrigado!'
        mock_run.assert_called_once_with('Como você está?')

def test_execute_memory_retrieve_not_found():
    """Testa a recuperação de memória quando o valor não existe."""
    mock_memory = MagicMock(spec=MemorySystem)
    mock_memory.retrieve.return_value = None
    executor = Executor(memory_system=mock_memory)
    
    intent = {
        'type': 'memory',
        'action': 'retrieve',
        'target': 'nonexistent_key'
    }
    
    result = executor.execute(intent)
    assert result['status'] == 'error'
    assert result['action'] == 'retrieve'
    assert 'não encontrado' in result['message']
    mock_memory.retrieve.assert_called_once_with('nonexistent_key')

def test_execute_python_code_error():
    """Testa a execução de código Python com erro."""
    executor = Executor()
    
    intent = {
        'type': 'python',
        'content': 'print(undefined_variable)'
    }
    
    with patch('core.code_runner.run_python_code') as mock_run:
        mock_run.return_value = {'error': 'NameError: name "undefined_variable" is not defined'}
        result = executor.execute(intent)
        assert result['status'] == 'error'
        assert result['action'] == 'execute_python'
        assert 'undefined_variable' in result['message']

def test_execute_terminal_command_error():
    """Testa a execução de comando de terminal com erro."""
    executor = Executor()
    
    intent = {
        'type': 'terminal',
        'content': 'nonexistent_command'
    }
    
    with patch('core.code_runner.execute_terminal_command') as mock_run:
        mock_run.return_value = {'error': 'Command not found: nonexistent_command'}
        result = executor.execute(intent)
        assert result['status'] == 'error'
        assert result['action'] == 'execute_terminal'
        assert result['message'] == 'Command not found: nonexistent_command'
        mock_run.assert_called_once_with('nonexistent_command')

def test_execute_terminal_command_dangerous():
    """Testa a execução de comando de terminal perigoso."""
    executor = Executor()
    
    intent = {
        'type': 'terminal',
        'content': 'rm -rf /'
    }
    
    with patch('core.code_runner.execute_terminal_command') as mock_run:
        mock_run.return_value = {'error': 'Comando não permitido por questões de segurança'}
        result = executor.execute(intent)
        assert result['status'] == 'error'
        assert result['action'] == 'execute_terminal'
        assert 'segurança' in result['message']

def test_execute_unknown_intent():
    """Testa a execução de uma intenção desconhecida."""
    executor = Executor()
    
    intent = {
        'type': 'unknown',
        'action': 'unknown',
        'content': 'test'
    }
    
    result = executor.execute(intent)
    assert result['status'] == 'error'
    assert result['action'] == 'unknown'
    assert 'não suportado' in result['message']

def test_execute_no_intent():
    """Testa a execução sem intenção."""
    executor = Executor()
    
    result = executor.execute(None)
    assert result['status'] == 'error'
    assert result['action'] == 'unknown'
    assert 'não fornecida' in result['message']

def test_instruction_command():
    """Testa o processamento de instruções."""
    executor = Executor()
    
    intent = {
        'type': 'instruction',
        'content': 'Como criar uma lista em Python?'
    }
    
    with patch('core.llm.run_llm') as mock_run:
        mock_run.return_value = "Para criar uma lista em Python, use colchetes []...\n"
        result = executor.execute(intent)
        assert result['status'] == 'success'
        assert result['action'] == 'instruction'
        assert result['response'] == "Para criar uma lista em Python, use colchetes []..."
        mock_run.assert_called_once_with('Como criar uma lista em Python?') 