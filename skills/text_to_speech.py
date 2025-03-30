# /home/arthur/Projects/A3X/skills/text_to_speech.py
import os
import subprocess
import logging
import pathlib
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (IMPORTANTE PARA VALIDAÇÃO)
WORKSPACE_ROOT = pathlib.Path("/home/arthur/Projects/A3X").resolve()

# Função auxiliar de validação de caminho (similar à de manage_files, mas simplificada)
def _is_path_safe(target_path: pathlib.Path) -> bool:
    """Verifica se o caminho está dentro do WORKSPACE_ROOT."""
    try:
        resolved_path = target_path.resolve()
        return resolved_path.is_relative_to(WORKSPACE_ROOT)
    except Exception as e:
        logger.error(f"Erro ao validar caminho '{target_path}': {e}")
        return False

def skill_text_to_speech(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte texto em fala usando o motor Piper TTS.

    Espera os seguintes parâmetros no action_input:
      - text (str): O texto a ser convertido em fala (obrigatório).
      - voice_model_path (str): Caminho para o arquivo do modelo .onnx do Piper (obrigatório).
      - output_dir (str, opcional): Diretório onde salvar o arquivo .wav. Padrão: '.' (diretório atual de trabalho do A3X).
      - filename (str, opcional): Nome do arquivo de saída sem a extensão .wav. Padrão: 'output_tts'.
    """
    logger.debug(f"Executando skill_text_to_speech com input: {action_input}")

    # >>> READ ENV VAR HERE <<<
    PIPER_EXECUTABLE_PATH = os.environ.get("PIPER_EXECUTABLE", "/path/to/piper/executable")

    text = action_input.get("text")
    voice_model_path = action_input.get("voice_model_path")
    # Usar pathlib para manipulação segura de caminhos
    output_dir_path = pathlib.Path(action_input.get("output_dir", ".")).resolve() # Resolve para absoluto relativo ao CWD atual
    output_filename_base = action_input.get("filename", "output_tts")

    # Validação básica dos parâmetros obrigatórios
    if not text:
        logger.error("Parâmetro 'text' obrigatório não fornecido.")
        return {
            "status": "error",
            "action": "tts_failed",
            "data": {"message": "Erro: Parâmetro 'text' é obrigatório."}
        }
    if not voice_model_path:
        logger.error("Parâmetro 'voice_model_path' obrigatório não fornecido.")
        return {
            "status": "error",
            "action": "tts_failed",
            "data": {"message": "Erro: Parâmetro 'voice_model_path' é obrigatório."}
        }

    # --- Implementação da Chamada ao Piper --- 
    try:
        # 1. Validar caminhos e segurança
        voice_model_file = pathlib.Path(voice_model_path).resolve()
        if not voice_model_file.is_file():
            return {"status": "error", "action": "tts_failed", "data": {"message": f"Arquivo do modelo Piper não encontrado: {voice_model_path}"}}
        # Não validamos se o modelo está no workspace, pode estar em um local compartilhado

        # Garante que o diretório de saída está dentro do workspace
        if not _is_path_safe(output_dir_path):
            return {"status": "error", "action": "tts_failed", "data": {"message": f"Diretório de saída fora do workspace não permitido: {action_input.get('output_dir', '.')}"}}

        # 2. Construir caminho completo do arquivo de saída .wav
        output_wav_path = (output_dir_path / f"{output_filename_base}.wav").resolve()

        # 3. Garantir que diretório de saída exista (já validado que está no workspace)
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_err:
             logger.error(f"Erro ao criar diretório de saída '{output_dir_path}': {mkdir_err}", exc_info=True)
             return {"status": "error", "action": "tts_failed", "data": {"message": f"Não foi possível criar o diretório de saída: {output_dir_path}"}}

        # 4. Construir o comando Piper
        # Usar subprocess.PIPE para passar o texto via stdin
        command = [
            PIPER_EXECUTABLE_PATH, # Now reads the potentially updated path
            "--model", str(voice_model_file),
            "--output_file", str(output_wav_path)
            # Adicionar outros flags do Piper se necessário (ex: --length_scale, --noise_scale)
        ]
        logger.info(f"Executando comando Piper: {' '.join(command)}")

        # 5. Executar com subprocess.run
        # Passa o texto via input=text.encode(...) e captura output
        process = subprocess.run(
            command, 
            input=text.encode('utf-8'), 
            capture_output=True, 
            check=False, # Não lança exceção em erro, verificamos returncode
            timeout=30 # Timeout de 30 segundos para TTS
        )

        # 6. Verificar resultado e retornar
        if process.returncode == 0:
            logger.info(f"Arquivo TTS gerado com sucesso em: {output_wav_path}")
            relative_output_path = str(output_wav_path.relative_to(WORKSPACE_ROOT))
            return {
                "status": "success", 
                "action": "tts_generated", 
                "data": {"message": "Texto convertido em fala com sucesso.", "output_filepath": relative_output_path}
            }
        else:
            stderr_output = process.stderr.decode('utf-8', errors='ignore').strip()
            logger.error(f"Erro ao executar Piper (returncode: {process.returncode}): {stderr_output}")
            return {
                "status": "error", 
                "action": "tts_failed", 
                "data": {"message": f"Falha na execução do Piper TTS: {stderr_output}", "returncode": process.returncode}
            }

    except FileNotFoundError:
        logger.error(f"Executável do Piper não encontrado em: {PIPER_EXECUTABLE_PATH}")
        return {"status": "error", "action": "tts_failed", "data": {"message": f"Erro de configuração: Executável do Piper não encontrado em '{PIPER_EXECUTABLE_PATH}'. Configure a variável de ambiente PIPER_EXECUTABLE."}}
    except subprocess.TimeoutExpired:
        logger.error("Timeout ao executar Piper TTS.")
        return {"status": "error", "action": "tts_failed", "data": {"message": "Timeout: A conversão TTS demorou muito."}}
    except Exception as e:
        logger.exception("Erro inesperado na skill text_to_speech:")
        return {"status": "error", "action": "tts_failed", "data": {"message": f"Erro inesperado durante TTS: {e}"}}
