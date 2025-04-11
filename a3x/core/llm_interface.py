import logging
import requests
import json
import datetime
import os
from typing import List, Dict, Optional, AsyncGenerator
import time
import httpx

# Initialize logger for this module *before* first use
llm_logger = logging.getLogger(__name__)

# Local imports (assuming config is accessible from core)
# <<< MODIFIED: Handle potential ImportError or missing LLAMA_SERVER_URL >>>
try:
    # from .config import LLAMA_DEFAULT_HEADERS
    from a3x.core.config import LLAMA_DEFAULT_HEADERS

    # Try importing LLAMA_SERVER_URL, but don't fail if it's missing/commented out
    # from .config import LLAMA_SERVER_URL as _CONFIG_LLAMA_URL
    from a3x.core.config import LLAMA_SERVER_URL as _CONFIG_LLAMA_URL
except ImportError:
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}  # Sensible fallback
    _CONFIG_LLAMA_URL = None
    llm_logger.warning(
        "Could not import config from core.config or LLAMA_SERVER_URL is missing. Using defaults."
    )

# Define a default URL within the module
_DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# <<< ADD REACT_SCHEMA Definition for testing >>>
REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "Thought": {"type": "string"},
        "Action": {"type": "string"},
        "Action Input": {"type": "object"},  # Allow any object for input initially
    },
    "required": ["Thought", "Action"],
}

DEFAULT_TIMEOUT = 600 # Increased default timeout

def _determine_llm_url(provided_url: Optional[str]) -> str:
    """Determines the LLM URL to use, checking environment variables and defaults."""
    if provided_url:
        return provided_url
    env_url = os.getenv("LLM_API_URL")
    if env_url:
        return env_url
    return _DEFAULT_LLM_URL

# <<< REPLACE the entire call_llm function body >>>
async def call_llm(
    messages: List[Dict[str, str]],
    llm_url: Optional[str] = None,
    stream: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> AsyncGenerator[str, None]:
    """
    Async LLM call using httpx (supports both streaming and non-streaming modes).
    """
    target_url = _determine_llm_url(llm_url)
    headers = {"Content-Type": "application/json"}
    payload = {"messages": messages, "stream": stream}

    llm_logger.debug(f"[LLM] Sending to {target_url} | Stream: {stream}")
    llm_logger.debug(f"[LLM] Payload (trimmed): {str(payload)[:500]}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url=target_url,
                headers=headers,
                json=payload,
            )

            response.raise_for_status()

            if stream:
                llm_logger.info(f"[LLM] Streaming response started.")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[len("data: "):].strip()
                        if json_str == "[DONE]":
                            llm_logger.debug("[LLM] Stream finished ([DONE])")
                            break
                        try:
                            data = json.loads(json_str)
                            # Safe access to nested content
                            choices = data.get("choices", [])
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                if isinstance(delta, dict):
                                    content = delta.get("content")
                                    if content:
                                        yield content
                        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                            llm_logger.warning(f"[LLM] Streaming decode/access error: {e} | line: {json_str}")
            else:
                data = response.json()
                # Safe access to nested content
                choices = data.get("choices", [])
                content = ""
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                llm_logger.debug(f"[LLM] Non-streaming response content length: {len(content)}")
                yield content

        except httpx.RequestError as e:
            llm_logger.exception(f"[LLM] Request error: {e}")
            yield f"[LLM Error: Request failed - {e}]"
        except httpx.HTTPStatusError as e:
            llm_logger.error(f"[LLM] HTTP error {e.response.status_code}: {e.response.text}")
            yield f"[LLM Error: HTTP {e.response.status_code}]"
        except Exception as e:
            llm_logger.exception(f"[LLM] Unexpected error: {e}")
            yield f"[LLM Error: Unexpected failure - {e}]"
