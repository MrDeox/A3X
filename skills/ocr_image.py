# /home/arthur/Projects/A3X/skills/ocr_image.py
import logging
import pathlib
from typing import Dict, Any
import pytesseract
from PIL import Image

# Configure logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (IMPORTANTE PARA VALIDAÇÃO)
WORKSPACE_ROOT = pathlib.Path("/home/arthur/Projects/A3X").resolve()

# Função auxiliar de validação de caminho (reutilizada ou adaptada)
def _is_path_safe(target_path: pathlib.Path) -> bool:
    """Verifica se o caminho está dentro do WORKSPACE_ROOT."""
    try:
        resolved_path = target_path.resolve()
        # Permite caminhos dentro do workspace
        if WORKSPACE_ROOT in resolved_path.parents or resolved_path == WORKSPACE_ROOT:
             return True
        # Poderíamos permitir outros diretórios seguros se necessário (e.g., /tmp)
        # elif resolved_path.is_relative_to("/tmp"):
        #     return True 
        return False
    except Exception as e:
        logger.error(f"Erro ao validar caminho '{target_path}': {e}")
        return False

def skill_ocr_image(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai texto de uma imagem usando Tesseract OCR.

    Expected action_input parameters:
      - image_path (str): Caminho para o arquivo de imagem (PNG, JPG, etc.) (required).
      - lang (str, optional): Código(s) do idioma para o Tesseract usar (e.g., 'eng', 'por', 'eng+por'). Padrão: 'eng'.
    """
    logger.debug(f"Executing skill_ocr_image with input: {action_input}")

    image_path_str = action_input.get("image_path")
    lang = action_input.get("lang", "eng") # Padrão para inglês se não especificado

    # --- Basic Validation ---
    if not image_path_str:
        return {"status": "error", "action": "ocr_failed", "data": {"message": "Error: 'image_path' parameter is required."}}

    try:
        image_path = pathlib.Path(image_path_str)
        # Resolve o caminho relativo ao workspace se não for absoluto
        if not image_path.is_absolute():
             image_path = (WORKSPACE_ROOT / image_path_str).resolve()
        else:
             image_path = image_path.resolve()

        # Validação de Segurança: Garante que a imagem está em um local permitido
        if not _is_path_safe(image_path):
             return {"status": "error", "action": "ocr_failed", "data": {"message": f"Error: Accessing image outside allowed directories is not permitted: {image_path_str}"}}
        
        if not image_path.is_file():
             return {"status": "error", "action": "ocr_failed", "data": {"message": f"Error: Image file not found at: {image_path}"}}

        # --- OCR Logic ---
        logger.info(f"Performing OCR on image '{image_path}' with language(s) '{lang}'")
        
        # Usa pytesseract para extrair texto da imagem
        # Pode precisar configurar o caminho do Tesseract se não estiver no PATH do sistema:
        # pytesseract.pytesseract.tesseract_cmd = r'/path/to/your/tesseract' # Exemplo
        extracted_text = pytesseract.image_to_string(Image.open(image_path), lang=lang)

        logger.info(f"OCR successful. Extracted {len(extracted_text)} characters.")
        return {
            "status": "success",
            "action": "ocr_completed",
            "data": {"extracted_text": extracted_text.strip()}
        }

    except pytesseract.TesseractNotFoundError:
         logger.error("Tesseract executable not found. Make sure Tesseract OCR is installed and in your system's PATH.")
         return {"status": "error", "action": "ocr_failed", "data": {"message": "Error: Tesseract executable not found. Please install Tesseract OCR."}}
    except FileNotFoundError:
         # Este erro pode ocorrer se o caminho relativo não puder ser resolvido corretamente APESAR da checagem anterior
         # Ou se houve um problema com _is_path_safe ou a construção do path
         logger.error(f"File not found during OCR process, potentially resolved path issue for: {image_path_str}")
         return {"status": "error", "action": "ocr_failed", "data": {"message": f"Error: Image file not found (path resolution issue?): {image_path_str}"}}    
    except Exception as e:
        logger.exception("Unexpected error during OCR:")
        return {"status": "error", "action": "ocr_failed", "data": {"message": f"Unexpected error during OCR: {e}"}}

