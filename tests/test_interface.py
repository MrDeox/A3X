"""
Testes para a interface interativa do A³X.
"""

import pytest
from unittest.mock import patch, MagicMock
from main import A3XInterface
import json
from pathlib import Path

@pytest.fixture
def interface():
    """Fixture que cria uma interface limpa para cada teste."""
    interface = A3XInterface()
    interface.history = []
    return interface

def test_help_command(interface):
    """Testa o comando de ajuda."""
    with patch('builtins.print') as mock_print:
        interface._show_help()
        mock_print.assert_called()

def test_clear_screen(interface):
    """Testa limpeza do terminal."""
    with patch('os.system') as mock_system:
        interface._clear_screen()
        mock_system.assert_called_once()

def test_format_time(interface):
    """Testa formatação de tempo."""
    assert interface._format_time(0.5) == "500.0ms"
    assert interface._format_time(1.5) == "1.50s"

def test_history_operations(interface, tmp_path):
    """Testa operações de histórico."""
    # Configura arquivo temporário
    interface.history_file = tmp_path / "test_history.json"
    
    # Testa salvar histórico
    interface.history = [
        {'command': 'test1', 'result': 'ok', 'timestamp': '2024-03-25', 'execution_time': 0.1},
        {'command': 'test2', 'result': 'ok', 'timestamp': '2024-03-25', 'execution_time': 0.2}
    ]
    interface._save_history()
    
    # Verifica se arquivo foi criado
    assert interface.history_file.exists()
    
    # Testa carregar histórico
    interface.history = []
    interface._load_history()
    assert len(interface.history) == 2
    assert interface.history[0]['command'] == 'test1'
    assert interface.history[1]['command'] == 'test2'

def test_multiline_mode(interface):
    """Testa modo multilinha."""
    # Entra no modo multilinha
    interface.is_multiline = True
    interface.multiline_buffer = ['print("test")', 'print("ok")']
    
    # Simula entrada de >>> para sair do modo
    with patch('builtins.input', return_value='>>>'):
        command = interface._get_next_command()
        assert command == 'print("test")\nprint("ok")'
        assert not interface.is_multiline
        assert not interface.multiline_buffer

def test_special_commands(interface):
    """Testa comandos especiais."""
    # Testa !help
    with patch('builtins.print') as mock_print:
        interface._process_special_command('!help')
        mock_print.assert_called()
    
    # Testa !clear
    with patch('os.system') as mock_system:
        interface._process_special_command('!clear')
        mock_system.assert_called_once()
    
    # Testa !exit
    with pytest.raises(SystemExit):
        interface._process_special_command('!exit')

def test_command_processing(interface):
    """Testa processamento de comandos."""
    with patch.object(interface.executor, 'process_command') as mock_process:
        mock_process.return_value = 'test result'
        
        result, time = interface._process_command('test command')
        assert result == 'test result'
        assert isinstance(time, float)
        assert len(interface.history) == 1
        assert interface.history[0]['command'] == 'test command' 