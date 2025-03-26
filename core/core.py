#!/usr/bin/env python3
"""
Módulo Core do A³X - Execução segura de código Python.
"""

import re
import time
import json
import logging
import builtins
import contextlib
from typing import Dict, Any, Optional, Union
from pathlib import Path
from io import StringIO
from llm.inference import run_llm

# Configuração de logging
logging.basicConfig(
    filename='logs/core.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
MAX_CODE_LENGTH = 1000
TIMEOUT = 5  # segundos

# Palavras-chave proibidas
FORBIDDEN_KEYWORDS = {
    'import', 'from', 'as', 'with', 'try', 'except', 'finally',
    'raise', 'assert', 'del', 'global', 'nonlocal', 'yield',
    'lambda', 'class', 'def', 'return', 'break', 'continue',
    'pass', 'exec', 'eval', 'compile', 'open', 'file',
    'os', 'sys', 'subprocess', 'shutil', '__import__'
}

# Builtins permitidos
SAFE_BUILTINS = {
    # Tipos básicos
    'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple',
    
    # Funções builtin seguras
    'print', 'len', 'range', 'sum', 'min', 'max', 'abs', 'round',
    'pow', 'divmod', 'bin', 'hex', 'oct', 'chr', 'ord', 'enumerate',
    'zip', 'map', 'filter', 'any', 'all', 'sorted', 'reversed',
    'format', 'repr', 'ascii', 'hash', 'id', 'type', 'isinstance',
    'issubclass', 'super', 'property', 'classmethod', 'staticmethod',
    'getattr', 'setattr', 'hasattr', 'delattr', 'vars', 'locals',
    'globals', 'dir', 'help', 'copyright', 'credits', 'license'
}

def _validate_code(code: str) -> bool:
    """
    Valida o código antes da execução.
    
    Args:
        code: Código a ser validado
        
    Returns:
        bool: True se válido, False caso contrário
    """
    if not code or not isinstance(code, str):
        return False
        
    # Verifica tamanho
    if len(code) > MAX_CODE_LENGTH:
        return False
        
    # Verifica palavras-chave proibidas
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in code:
            return False
            
    return True

def _create_sandbox() -> Dict[str, Any]:
    """
    Cria ambiente isolado para execução.
    
    Returns:
        Dict[str, Any]: Dicionário com globals seguros
    """
    # Cria dicionário com builtins permitidos
    safe_globals = {
        '__builtins__': {
            name: getattr(builtins, name)
            for name in SAFE_BUILTINS
        }
    }
    
    return safe_globals

def _format_error(error: Exception) -> str:
    """
    Formata mensagens de erro de forma amigável.
    
    Args:
        error: Exceção a ser formatada
        
    Returns:
        str: Mensagem formatada
    """
    error_types = {
        SyntaxError: "Erro de sintaxe no código",
        NameError: "Variável não definida",
        TypeError: "Tipo de dado inválido",
        ValueError: "Valor inválido",
        IndexError: "Índice fora dos limites",
        KeyError: "Chave não encontrada",
        ZeroDivisionError: "Divisão por zero",
        TimeoutError: "Tempo limite excedido",
        MemoryError: "Memória insuficiente"
    }
    
    error_type = type(error)
    message = error_types.get(error_type, "Erro na execução")
    
    return f"{message}: {str(error)}"

def _log_execution(
    code: str,
    success: bool,
    output: Optional[str] = None,
    error: Optional[str] = None,
    duration: Optional[float] = None
) -> None:
    """
    Registra a execução do código no log.
    
    Args:
        code: Código executado
        success: Se a execução foi bem sucedida
        output: Saída do código (se houver)
        error: Mensagem de erro (se houver)
        duration: Tempo de execução em segundos
    """
    log_data = {
        'code': code,
        'success': success,
        'output': output,
        'error': error,
        'duration': duration,
        'size': len(code)
    }
    
    if success:
        logging.info(f"Execução bem sucedida: {code[:50]}...")
    else:
        logging.error(f"Falha na execução: {code[:50]}... - {error}")

def run_python_code(code: str) -> str:
    """
    Executa código Python de forma segura.
    
    Args:
        code: Código Python a ser executado
        
    Returns:
        str: Saída do código ou mensagem de erro
        
    Raises:
        ValueError: Se o código for inválido
        TimeoutError: Se exceder o tempo limite
    """
    # Validação inicial
    if not _validate_code(code):
        error_msg = "Código não permitido por questões de segurança"
        _log_execution(code, False, error=error_msg)
        raise ValueError(error_msg)
        
    try:
        # Inicia o timer
        start_time = time.time()
        
        # Cria sandbox
        sandbox = _create_sandbox()
        
        # Captura stdout
        output = []
        with contextlib.redirect_stdout(StringIO()) as stdout:
            # Executa o código
            exec(code, sandbox)
            
            # Recupera saída
            output.append(stdout.getvalue())
            
        # Calcula duração
        duration = time.time() - start_time
        
        # Verifica timeout
        if duration > TIMEOUT:
            raise TimeoutError("Tempo limite excedido")
            
        # Log de sucesso
        _log_execution(
            code,
            True,
            output=''.join(output),
            duration=duration
        )
        
        # Retorna resultado
        return ''.join(output).strip()
        
    except TimeoutError:
        error_msg = "Tempo limite excedido"
        _log_execution(code, False, error=error_msg)
        raise
        
    except Exception as e:
        error_msg = _format_error(e)
        _log_execution(code, False, error=error_msg)
        return error_msg

class Executor:
    """Executor de comandos do A³X."""
    
    def __init__(self):
        """Inicializa o executor."""
        self.memory = {}
        self.history = []
        
    def process_command(self, command: str) -> str:
        """
        Processa um comando e retorna o resultado.
        
        Args:
            command: Comando a ser processado
            
        Returns:
            str: Resultado do comando
        """
        try:
            # Executa o código Python
            result = run_python_code(command)
            return result
        except Exception as e:
            return f"Erro: {_format_error(e)}"

def analyze_intent(command: str) -> Dict[str, Any]:
    """
    Analisa a intenção de um comando em linguagem natural usando LLM.
    
    Args:
        command: Comando em linguagem natural
        
    Returns:
        Dict[str, Any]: Estrutura JSON com a intenção interpretada
        
    Example:
        >>> analyze_intent("me diga a capital do Japão e salve como info_japao")
        {
            "type": "question",
            "action": "ask",
            "target": "info_japao",
            "content": "Qual é a capital do Japão?"
        }
    """
    # Prompt estruturado para o LLM
    prompt = f"""Analise a intenção do seguinte comando e retorne um JSON com a estrutura:
{{
    "type": "tipo do comando",
    "action": "ação específica",
    "target": "alvo ou chave (se aplicável)",
    "content": "conteúdo principal"
}}

Tipos possíveis:
- memory: operações de memória (lembrar, recuperar)
- terminal: comandos do terminal
- python: código Python
- question: perguntas
- instruction: instruções/geração de código
- unknown: não reconhecido

Ações possíveis:
- store: armazenar na memória
- retrieve: recuperar da memória
- run: executar comando
- ask: fazer pergunta
- generate: gerar código
- explain: explicar conceito
- unknown: não reconhecido

Comando: {command}

Retorne APENAS o JSON, sem texto adicional."""

    try:
        # Executa o LLM
        response = run_llm(prompt)
        
        # Tenta extrair o JSON da resposta
        try:
            # Remove possíveis textos antes ou depois do JSON
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1]
            
            # Tenta fazer o parse do JSON
            intent = json.loads(json_str)
            
            # Validação básica da estrutura
            required_keys = {"type", "action", "target", "content"}
            if not all(key in intent for key in required_keys):
                raise ValueError("JSON inválido: faltam chaves obrigatórias")
                
            # Normaliza valores
            intent["type"] = intent["type"].lower()
            intent["action"] = intent["action"].lower()
            intent["target"] = intent["target"] if intent["target"] else None
            intent["content"] = intent["content"].strip()
            
            return intent
            
        except json.JSONDecodeError:
            # Se falhar, tenta uma segunda vez com um prompt mais específico
            retry_prompt = f"""O comando anterior retornou uma resposta inválida.
Por favor, retorne APENAS um JSON válido com a estrutura:
{{
    "type": "tipo do comando",
    "action": "ação específica",
    "target": "alvo ou chave (se aplicável)",
    "content": "conteúdo principal"
}}

Comando: {command}"""
            
            retry_response = run_llm(retry_prompt)
            try:
                intent = json.loads(retry_response.strip())
                return intent
            except json.JSONDecodeError:
                # Se ainda falhar, retorna uma intenção desconhecida
                return {
                    "type": "unknown",
                    "action": "unknown",
                    "target": None,
                    "content": command
                }
                
    except Exception as e:
        logging.error(f"Erro ao analisar intenção: {str(e)}")
        return {
            "type": "unknown",
            "action": "unknown",
            "target": None,
            "content": command
        }

# Interface pública
__all__ = ['run_python_code', 'Executor', 'analyze_intent'] 