#!/usr/bin/env python3
"""
Módulo LLM do A³X - Interface com modelo de linguagem.
"""

import subprocess
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    filename='logs/llm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Caminhos
BINARY_PATH = Path('bin/llama-cli')
MODEL_PATH = Path('models/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf')

def format_prompt(prompt: str) -> str:
    """Formata o prompt para o modelo."""
    return f"<|im_start|>user\n{prompt}\n<|im_start|>assistant"

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
        
        # Remove tokens especiais e limpa a resposta
        response = output.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("[end of text]", "").strip()
        response = response.replace("<|im_start|>", "").strip()
        response = response.replace("user", "").strip()
        response = response.replace("assistant:", "").strip()
        
        # Remove o prompt original da resposta
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Log de sucesso
        logging.info(f"Resposta gerada para prompt: {prompt[:50]}...")
        
        return response
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro na execução: {e.stderr}")
        raise

if __name__ == "__main__":
    # Exemplo de uso
    try:
        resposta = run_llm("Qual é a sua missão?")
        print(resposta)
    except Exception as e:
        print(f"Erro: {e}") 