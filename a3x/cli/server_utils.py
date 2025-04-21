# a3x/cli/server_utils.py
import logging
import requests
from urllib.parse import urljoin

# Import config directly if needed for default URLs
try:
    from a3x.core.config import LLAVA_API_URL as DEFAULT_LLAVA_API_URL
except ImportError:
    DEFAULT_LLAVA_API_URL = ""

logger = logging.getLogger(__name__)

def check_llm_server_health(url: str, timeout: float = 2.0) -> bool:
    """Checks if the LLM server (like llama.cpp server) is responding."""
    if not url:
        logger.warning("LLM server URL not provided, cannot check health.")
        return False
    health_url = urljoin(url, "/health") # Standard health check endpoint
    logger.debug(f"Checking LLM server health at {health_url}...")
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("status") == "ok":
                    logger.debug("LLM server is healthy.")
                    return True
                else:
                    logger.warning(f"LLM server health check failed: Status is not 'ok'. Response: {data}")
                    return False
            except requests.exceptions.JSONDecodeError:
                logger.warning(f"LLM server health check at {health_url} returned status 200 but non-JSON response: {response.text[:100]}...")
                return False # Treat non-JSON as potentially unhealthy
        else:
            logger.warning(
                f"LLM server health check failed: Status code {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to LLM server at {health_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during LLM health check: {e}", exc_info=True)
        return False

def check_sd_server_health(timeout: float = 5.0) -> bool:
    """Placeholder for checking Stable Diffusion server health."""
    logger.debug("Checking Stable Diffusion server health (placeholder)...")
    logger.warning("SD server health check not implemented.")
    return True # Assume running for now if autostart not used

def check_llava_api_health(url: str = DEFAULT_LLAVA_API_URL, timeout: float = 5.0) -> bool:
    """Checks if the LLaVA OpenAI-compatible API server is running."""
    if not url:
        logger.warning("LLaVA API URL not provided, cannot check health.")
        return False
    models_url = urljoin(url, "/models") # Standard OpenAI API endpoint
    logger.debug(f"Checking LLaVA API server health at {models_url}...")
    try:
        response = requests.get(models_url, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    logger.debug("LLaVA API server is healthy.")
                    return True
                else:
                     logger.warning(f"LLaVA API health check failed: Response format unexpected. Response: {data}")
                     return False
            except requests.exceptions.JSONDecodeError:
                logger.warning(f"LLaVA API health check at {models_url} returned status 200 but non-JSON response: {response.text[:100]}...")
                return False # Treat non-JSON as potentially unhealthy
        else:
            logger.warning(
                f"LLaVA API health check failed: Status code {response.status_code}"
            )
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to LLaVA API server at {models_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during LLaVA API health check: {e}", exc_info=True)
        return False 