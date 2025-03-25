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
        capture_output: Se True, retorna stdout/stderr como string
        
    Returns:
        str: Saída do comando ou mensagem de erro
        
    Raises:
        ValueError: Se o comando for inválido
        PermissionError: Se o comando for bloqueado
        subprocess.CalledProcessError: Se o comando falhar
        subprocess.TimeoutExpired: Se exceder o tempo limite
    """
    # Validação inicial
    if not _validate_command(command):
        error_msg = "Comando inválido ou não permitido"
        _log_execution(command, False, error=error_msg)
        raise ValueError(error_msg)
        
    # Prepara o comando
    cmd = command.split()
    
    try:
        # Inicia o timer
        start_time = time.time()
        
        # Executa o comando
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=TIMEOUT,
            shell=False
        )
        
        # Calcula duração
        duration = time.time() - start_time
        
        # Verifica sucesso
        if result.returncode != 0:
            error_msg = _format_error(result)
            _log_execution(command, False, error=error_msg, duration=duration)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
        # Log de sucesso
        _log_execution(
            command,
            True,
            output=result.stdout if capture_output else None,
            duration=duration
        )
        
        # Retorna resultado
        return result.stdout if capture_output else ""
        
    except subprocess.TimeoutExpired:
        error_msg = "Tempo limite excedido"
        _log_execution(command, False, error=error_msg)
        raise
        
    except Exception as e:
        error_msg = _format_error(e)
        _log_execution(command, False, error=error_msg)
        raise

# Interface pública
__all__ = ['execute'] 