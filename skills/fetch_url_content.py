# skills/fetch_url_content.py
import logging
import requests
from typing import Dict, Any
from urllib.parse import urlparse

# Importar BeautifulSoup e Readability se/quando usados
# from bs4 import BeautifulSoup
# from readability import Document # Requires readability-lxml

# Configurar logger para esta skill
logger = logging.getLogger(__name__)

# Timeout padrão para requisições HTTP (segundos)
DEFAULT_REQUEST_TIMEOUT = 15


def _is_valid_url(url: str) -> bool:
    """Verifica se a URL tem um esquema e netloc válidos e não é local."""
    try:
        parsed = urlparse(url)
        # Requer esquema (http/https) e localização de rede (domínio)
        # Proíbe explicitamente o esquema 'file'
        return all([parsed.scheme in ["http", "https"], parsed.netloc])
    except Exception:
        return False


def skill_fetch_url_content(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Busca o conteúdo HTML de uma URL, extrai o texto principal e o retorna.

    Args:
        action_input (Dict[str, Any]): Dicionário contendo:
            - url (str): A URL da página web a ser buscada (obrigatório).
            - timeout (int, optional): Timeout em segundos para a requisição HTTP.

    Returns:
        Dict[str, Any]: Dicionário com status, action, e data (contendo URL e conteúdo extraído).
    """
    logger.info(f"Executando skill_fetch_url_content com input: {action_input}")

    url = action_input.get("url")
    timeout = action_input.get("timeout", DEFAULT_REQUEST_TIMEOUT)

    # --- Validação do Input ---
    if not url:
        logger.error("Parâmetro obrigatório 'url' não fornecido.")
        return {
            "status": "error",
            "action": "fetch_failed",
            "data": {"message": "Erro: O parâmetro 'url' não foi especificado."},
        }

    if not isinstance(url, str) or not _is_valid_url(url):
        logger.error(f"URL inválida ou não permitida fornecida: {url}")
        return {
            "status": "error",
            "action": "fetch_failed",
            "data": {
                "url": url,
                "message": f"Erro: URL inválida ou não permitida: '{url}'. Use http ou https e inclua o domínio.",
            },
        }

    try:
        timeout_sec = int(timeout)
        if timeout_sec <= 0:
            raise ValueError("Timeout must be positive")
    except (ValueError, TypeError):
        logger.warning(
            f"Timeout inválido fornecido: {timeout}. Usando default: {DEFAULT_REQUEST_TIMEOUT}s"
        )
        timeout_sec = DEFAULT_REQUEST_TIMEOUT

    # --- Lógica de Fetch e Parse (Placeholder) ---
    try:
        logger.info(f"Buscando conteúdo da URL: {url} com timeout {timeout_sec}s")
        headers = {  # Simular um navegador básico
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(
            url, headers=headers, timeout=timeout_sec, allow_redirects=True
        )
        response.raise_for_status()  # Levanta erro para status >= 400

        # Verificar tipo de conteúdo (aceitar apenas HTML por enquanto)
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type:
            logger.warning(f"Conteúdo não é HTML ({content_type}) para URL: {url}")
            return {
                "status": "error",
                "action": "fetch_failed",
                "data": {
                    "url": url,
                    "message": f"Erro: O conteúdo da URL não é HTML (tipo: {content_type}).",
                },
            }

        html_content = response.text
        logger.debug(f"HTML recebido (primeiros 500 chars): {html_content[:500]}...")

        # --- Placeholder para Extração de Conteúdo ---
        # TODO: Usar BeautifulSoup e/ou Readability para extrair texto útil
        extracted_content = f"Placeholder: Conteúdo extraído de {url}"
        # --- Fim do Placeholder ---

        logger.info(f"Conteúdo extraído com sucesso da URL: {url}")
        return {
            "status": "success",
            "action": "content_fetched",
            "data": {
                "url": url,
                "content": extracted_content,  # Retornar o conteúdo extraído
                "message": f"Conteúdo principal extraído com sucesso de '{url}'.",
            },
        }

    except requests.exceptions.Timeout:
        logger.error(f"Timeout ({timeout_sec}s) ao buscar URL: {url}")
        return {
            "status": "error",
            "action": "fetch_failed",
            "data": {
                "url": url,
                "message": f"Erro: Timeout ({timeout_sec}s) ao tentar acessar a URL.",
            },
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de rede ao buscar URL '{url}': {e}")
        return {
            "status": "error",
            "action": "fetch_failed",
            "data": {"url": url, "message": f"Erro de rede ao acessar a URL: {e}"},
        }
    except Exception as e:
        logger.exception(f"Erro inesperado ao processar URL '{url}':")
        return {
            "status": "error",
            "action": "fetch_failed",
            "data": {
                "url": url,
                "message": f"Erro inesperado durante o processamento da URL: {str(e)}",
            },
        }
