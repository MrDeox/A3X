"""
Wrapper Python para execução de modelos GGUF usando llama.cpp com suporte a GPU AMD via ROCm.
"""

from .inference import run_llm, format_prompt

__all__ = ['run_llm', 'format_prompt'] 