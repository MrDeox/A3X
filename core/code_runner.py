"""
Módulo de execução de código do A³X.
Responsável por executar código Python e comandos do terminal de forma segura.
"""

import logging
import subprocess
import io
import contextlib
from typing import Dict, Union, Any

# Configuração de logging
logger = logging.getLogger(__name__)

def run_python_code(code: str) -> Union[str, Dict[str, str]]:
    """
    Executa código Python em um ambiente seguro.
    
    Args:
        code: Código Python a ser executado
        
    Returns:
        String com a saída ou dicionário com erro
    """
    # Cria um buffer para capturar a saída
    string_io = io.StringIO()
    
    try:
        # Executa o código em um ambiente controlado
        with contextlib.redirect_stdout(string_io):
            # Cria um namespace vazio e remove acesso a built-ins perigosos
            exec(code, {'__builtins__': {}}, {})
            
        # Obtém a saída capturada
        output = string_io.getvalue()
        return output
        
    except Exception as e:
        logger.error(f"Erro ao executar código Python: {str(e)}")
        return {'error': str(e)}

def execute_terminal_command(command: str) -> Dict[str, str]:
    """
    Executa um comando no terminal.
    
    Args:
        command: Comando a ser executado
        
    Returns:
        Dicionário com a saída ou erro
    """
    # Lista de comandos perigosos
    dangerous_commands = ['rm -rf', 'mkfs', 'dd', '> /dev/sda']
    
    # Verifica se é um comando perigoso
    for dangerous in dangerous_commands:
        if dangerous in command:
            return {'error': 'Comando não permitido por questões de segurança'}
            
    try:
        # Executa o comando
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return {'output': result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar comando: {e.stderr}")
        return {'error': f"Erro ao executar comando: {e.stderr}"} 