# skills/capture_screen.py
import logging
import mss
import mss.tools
import os
import time
from typing import Dict, Any, Optional

# Configurar logger para esta skill
logger = logging.getLogger(__name__)

# Diretório para salvar screenshots temporários (dentro do workspace)
# Certifique-se que este diretório existe ou crie-o.
SCREENSHOT_DIR = "temp/screenshots"
DEFAULT_MONITOR_INDEX = 1  # 1 geralmente é o monitor primário


def _ensure_dir_exists(dir_path: str):
    """Garante que um diretório exista, criando-o se necessário."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logger.info(f"Diretório criado: {dir_path}")
        except OSError as e:
            logger.error(f"Erro ao criar diretório {dir_path}: {e}")
            raise  # Re-levanta o erro se não puder criar


def skill_capture_screen(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Captura a tela inteira ou uma região específica e salva em um arquivo.

    Args:
        action_input (Dict[str, Any]): Dicionário contendo opcionalmente:
            - region (Dict[str, int]): Dicionário com {top, left, width, height}.
            - monitor (int): O índice do monitor a capturar (default: 1).
            - filename (str): Nome opcional para o arquivo (sem extensão).

    Returns:
        Dict[str, Any]: Dicionário com status, action, e data (contendo o path do arquivo).
    """
    logger.info(f"Executando skill_capture_screen com input: {action_input}")

    region: Optional[Dict[str, int]] = action_input.get("region")
    monitor_index: int = action_input.get("monitor", DEFAULT_MONITOR_INDEX)
    filename_base: str = action_input.get("filename", f"screenshot_{int(time.time())}")

    try:
        _ensure_dir_exists(SCREENSHOT_DIR)  # Garante que o diretório existe

        output_filename = f"{filename_base}.png"
        output_path = os.path.join(SCREENSHOT_DIR, output_filename)

        with mss.mss() as sct:
            if region:
                # Validar região (básico)
                if not all(k in region for k in ("top", "left", "width", "height")):
                    raise ValueError(
                        "A região deve conter 'top', 'left', 'width', 'height'."
                    )
                logger.info(f"Capturando região: {region}")
                sct_img = sct.grab(region)
            else:
                # Capturar monitor específico
                if monitor_index < 1 or monitor_index >= len(sct.monitors):
                    logger.warning(
                        f"Índice de monitor inválido ({monitor_index}). Usando monitor primário (1)."
                    )
                    monitor_index = 1  # Fallback para monitor primário
                logger.info(f"Capturando monitor inteiro: {monitor_index}")
                monitor = sct.monitors[monitor_index]
                sct_img = sct.grab(monitor)

            # Salvar a imagem
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)
            logger.info(f"Screenshot salvo em: {output_path}")

        return {
            "status": "success",
            "action": "screen_captured",
            "data": {
                "file_path": output_path,
                "message": f"Screenshot salvo com sucesso em '{output_path}'.",
            },
        }

    except (
        FileNotFoundError
    ):  # mss pode levantar isso se algo estiver errado com monitores?
        logger.error("Erro ao encontrar monitores ou problema com mss.")
        return {
            "status": "error",
            "action": "capture_failed",
            "data": {
                "message": "Erro interno ao acessar informações de tela/monitores."
            },
        }
    except (ValueError, KeyError) as e:  # Erro em parâmetros (e.g., região mal formada)
        logger.error(f"Erro nos parâmetros fornecidos: {e}")
        return {
            "status": "error",
            "action": "capture_failed",
            "data": {"message": f"Erro nos parâmetros: {e}"},
        }
    except Exception as e:
        logger.exception("Erro inesperado ao capturar a tela:")
        return {
            "status": "error",
            "action": "capture_failed",
            "data": {"message": f"Erro inesperado ao capturar tela: {str(e)}"},
        }
