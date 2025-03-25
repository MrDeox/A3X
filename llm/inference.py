#!/usr/bin/env python3
"""
Wrapper Python para execução de modelos GGUF usando llama.cpp com suporte a GPU AMD via ROCm.
"""

import subprocess
from pathlib import Path
import os

# Configurações do modelo e binário
LLAMA_CPP_DIR = Path(__file__).parent.parent / "llama.cpp"
MODEL_PATH = LLAMA_CPP_DIR / "models" / "dolphin-2.2.1-mistral-7b.Q4_K_M.gguf"
BINARY_PATH = LLAMA_CPP_DIR / "build" / "bin" / "llama-cli"

def format_prompt(prompt: str) -> str:
    """
    Formata o prompt no estilo chat/instruct do modelo.
    
    Args:
        prompt: O prompt do usuário
        
    Returns:
        str: O prompt formatado
    """
    return f"<|im_start|>user {prompt} <|im_end|> <|im_start|>assistant"

def run_llm(prompt: str, max_tokens: int = 128) -> str:
    """
    Executa o modelo LLM usando o binário llama-cli.
    
    Args:
        prompt: O prompt do usuário
        max_tokens: Número máximo de tokens a gerar (default: 128)
        
    Returns:
        str: A resposta do modelo
        
    Raises:
        FileNotFoundError: Se o binário ou modelo não forem encontrados
        subprocess.CalledProcessError: Se houver erro na execução
    """
    # Verifica se os arquivos necessários existem
    if not BINARY_PATH.exists():
        raise FileNotFoundError(f"Binário não encontrado: {BINARY_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    
    # Formata o prompt
    formatted_prompt = format_prompt(prompt)
    
    # Prepara o comando
    cmd = [
        str(BINARY_PATH),
        "-m", str(MODEL_PATH),
        "-n", str(max_tokens),
        "--gpu-layers", "16",
        "--n-gpu-layers", "16",
        "-p", formatted_prompt
    ]
    
    # Executa o comando
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extrai a resposta do modelo
        output = result.stdout
        
        # Remove o prompt e tokens especiais
        response = output.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
        
        return response
        
    except subprocess.CalledProcessError as e:
        print(f"Erro na execução: {e.stderr}")
        raise

if __name__ == "__main__":
    # Exemplo de uso
    try:
        resposta = run_llm("Qual é a sua missão?")
        print(resposta)
    except Exception as e:
        print(f"Erro: {e}") 