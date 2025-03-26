#!/usr/bin/env python3
"""
Módulo CLI do A³X - Execução segura de comandos shell.
"""

import subprocess
import re
import logging
import time
from typing import Optional, Tuple
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    filename='logs/cli.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
TIMEOUT = 10  # segundos
BLOCKED_COMMANDS = {
    'rm', 'rmdir', 'shutdown', 'reboot', 'halt',
    'dd', 'mkfs', 'chmod', 'chown', 'sudo', 'su',
    'wget', 'curl', 'nc', 'netcat',
    '> /dev/sda', 'mkfs.ext4', 'dd if=', 'mkfs.'
}

def _validate_command(command: str) -> bool:
    """
    Valida o comando antes da execução.
    
    Args:
        command: Comando a ser validado
        
    Returns:
        bool: True se válido, False caso contrário
    """
    if not command or not isinstance(command, str):
        return False
        
    # Verifica se é ASCII seguro
    if not re.match(r'^[a-zA-Z0-9_\-\.\s\/]+$', command):
        return False
        
    # Verifica comandos bloqueados
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return False
            
    # Verifica redirecionamentos perigosos
    if '>' in command and '/dev/' in command:
        return False
        
    return True

def _format_error(error: Exception) -> str:
    """
    Formata mensagens de erro de forma amigável.
    
    Args:
        error: Exceção a ser formatada
        
    Returns:
        str: Mensagem formatada
    """
    error_types = {
        subprocess.TimeoutExpired: "Tempo limite excedido",
        subprocess.CalledProcessError: "Comando falhou",
        PermissionError: "Comando não permitido",
        ValueError: "Comando inválido"
    }
    
    error_type = type(error)
    message = error_types.get(error_type, "Erro na execução")
    
    if isinstance(error, subprocess.CalledProcessError):
        return f"{message}: {error.stderr.decode() if error.stderr else str(error)}"
        
    return f"{message}: {str(error)}"

def _log_execution(
    command: str,
    success: bool,
    output: Optional[str] = None,
    error: Optional[str] = None,
    duration: Optional[float] = None
) -> None:
    """
    Registra a execução do comando no log.
    
    Args:
        command: Comando executado
        success: Se a execução foi bem sucedida
        output: Saída do comando (se houver)
        error: Mensagem de erro (se houver)
        duration: Tempo de execução em segundos
    """
    log_data = {
        'command': command,
        'success': success,
        'output': output,
        'error': error,
        'duration': duration
    }
    
    if success:
        logging.info(f"Execução bem sucedida: {command}")
    else:
        logging.error(f"Falha na execução: {command} - {error}")

def execute(command: str, capture_output: bool = True) -> str:
    """
    Executa um comando shell de forma segura.
    
    Args:
        command: Comando a ser executado
        capture_output: Se True, retorna stdout/stderr
        
    Returns:
        str: Saída do comando ou mensagem de erro
        
    Raises:
        ValueError: Se o comando for inválido
        subprocess.TimeoutExpired: Se exceder o timeout
    """
    # Validação
    if not _validate_command(command):
        raise ValueError("Comando não permitido por questões de segurança")
        
    try:
        # Executa o comando
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=TIMEOUT
        )
        
        # Log de sucesso
        logging.info(f"Comando executado: {command}")
        
        # Retorna saída
        if capture_output:
            return result.stdout or result.stderr
        return ""
        
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout no comando: {command}")
        raise
        
    except Exception as e:
        logging.error(f"Erro ao executar comando: {command} - {str(e)}")
        return f"Erro ao executar comando: {str(e)}"

# Interface pública
__all__ = ['execute'] 