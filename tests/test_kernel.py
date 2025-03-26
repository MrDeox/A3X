"""
Testes para o CognitiveKernel.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from core.kernel import CognitiveKernel
from core.executor import Executor
from core.analyzer import analyze_intent
from memory.system import MemorySystem

@pytest.fixture
def memory_system():
    """Fixture que cria um MemorySystem mockado."""
    return MagicMock(spec=MemorySystem)

@pytest.fixture
def executor(memory_system):
    """Fixture que cria um Executor mockado."""
    return MagicMock(spec=Executor)

@pytest.fixture
def kernel(memory_system, executor):
    """Fixture que cria um CognitiveKernel para testes."""
    return CognitiveKernel(memory_system=memory_system, executor=executor)

def test_initialization():
    """Testa a inicialização do kernel."""
    # Sem parâmetros
    kernel = CognitiveKernel()
    assert kernel.executor is None
    assert kernel.memory_system is None
    assert kernel.intent_parser == analyze_intent
    assert kernel.current_context == {}
    
    # Com executor mock
    mock_executor = MagicMock(spec=Executor)
    kernel = CognitiveKernel(executor=mock_executor)
    assert kernel.executor == mock_executor
    assert kernel.memory_system is None
    assert kernel.intent_parser == analyze_intent
    
    # Com memory system mock
    mock_memory = MagicMock(spec=MemorySystem)
    kernel = CognitiveKernel(memory_system=mock_memory)
    assert kernel.executor is not None
    assert kernel.memory_system == mock_memory
    assert kernel.intent_parser == analyze_intent
    
    # Com ambos os mocks
    kernel = CognitiveKernel(executor=mock_executor, memory_system=mock_memory)
    assert kernel.executor == mock_executor
    assert kernel.memory_system == mock_memory
    assert kernel.intent_parser == analyze_intent
    
    # Com parser customizado
    def custom_parser(text):
        return {'type': 'custom', 'content': text}
    kernel = CognitiveKernel(intent_parser=custom_parser)
    assert kernel.intent_parser == custom_parser

def test_process_input():
    """Testa o processamento de input."""
    mock_executor = MagicMock(spec=Executor)
    mock_executor.execute.return_value = {
        'status': 'success',
        'action': 'test',
        'output': 'test output'
    }
    
    kernel = CognitiveKernel(executor=mock_executor)
    
    # Input válido
    input_data = {'content': 'test command'}
    result = kernel.process_input(input_data)
    
    assert result['status'] == 'executed'
    assert result['parsed_intent']['type'] == 'unknown'
    assert result['parsed_intent']['content'] == 'test command'
    assert result['execution_result']['status'] == 'success'
    assert result['execution_result']['output'] == 'test output'
    
    mock_executor.execute.assert_called_once_with(intent={
        'type': 'unknown',
        'action': 'unknown',
        'target': None,
        'content': 'test command'
    })

def test_process_input_empty_content():
    """Testa o processamento de input vazio."""
    kernel = CognitiveKernel()
    
    result = kernel.process_input({})
    assert result['status'] == 'error'
    assert result['message'] == 'Entrada vazia ou inválida'
    assert result['parsed_intent'] is None
    assert result['execution_result'] is None
    
    result = kernel.process_input({'content': ''})
    assert result['status'] == 'error'
    assert result['message'] == 'Conteúdo vazio'
    assert result['parsed_intent'] is None
    assert result['execution_result'] is None

def test_process_input_no_content():
    """Testa o processamento sem input."""
    kernel = CognitiveKernel()
    
    result = kernel.process_input(None)
    assert result['status'] == 'error'
    assert result['message'] == 'Entrada vazia ou inválida'
    assert result['parsed_intent'] is None
    assert result['execution_result'] is None

def test_process_input_executor_error():
    """Testa erro no executor durante processamento."""
    mock_executor = MagicMock(spec=Executor)
    mock_executor.execute.side_effect = Exception("Erro de teste")
    
    kernel = CognitiveKernel(executor=mock_executor)
    
    result = kernel.process_input({'content': 'test'})
    assert result['status'] == 'error'
    assert result['parsed_intent']['type'] == 'unknown'
    assert result['parsed_intent']['content'] == 'test'
    assert result['execution_result']['status'] == 'error'
    assert result['execution_result']['message'] == 'Erro de teste'
    
    mock_executor.execute.assert_called_once_with(intent={
        'type': 'unknown',
        'action': 'unknown',
        'target': None,
        'content': 'test'
    })

def test_process_input_no_executor():
    """Testa processamento sem executor configurado."""
    kernel = CognitiveKernel(executor=None)
    
    result = kernel.process_input({'content': 'test'})
    assert result['status'] == 'error'
    assert result['parsed_intent']['type'] == 'unknown'
    assert result['parsed_intent']['content'] == 'test'
    assert result['execution_result']['status'] == 'error'
    assert result['execution_result']['message'] == 'Executor não configurado'

def test_custom_intent_parser():
    """Testa a inicialização com um parser de intenção customizado."""
    def custom_parser(text):
        return {'type': 'custom', 'content': text}
    
    mock_executor = MagicMock(spec=Executor)
    mock_executor.execute.return_value = {
        'status': 'success',
        'action': 'custom',
        'output': 'test output'
    }
    
    kernel = CognitiveKernel(executor=mock_executor, intent_parser=custom_parser)
    input_data = {'content': 'test'}
    
    result = kernel.process_input(input_data)
    
    assert result['status'] == 'executed'
    assert result['parsed_intent']['type'] == 'custom'
    assert result['parsed_intent']['content'] == 'test'
    assert result['execution_result']['status'] == 'success'
    assert result['execution_result']['output'] == 'test output'
    
    mock_executor.execute.assert_called_once_with(intent={'type': 'custom', 'content': 'test'})
