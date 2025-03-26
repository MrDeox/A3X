"""
Módulo core do A³X.
Contém as funcionalidades principais do sistema.
"""

from .analyzer import analyze_intent
from .code_runner import run_python_code, execute_terminal_command
from .llm import run_llm
from .executor import Executor
from .kernel import CognitiveKernel

__all__ = [
    'analyze_intent',
    'run_python_code',
    'execute_terminal_command',
    'run_llm',
    'Executor',
    'CognitiveKernel'
] 