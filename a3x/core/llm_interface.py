import logging
import requests
import json
import datetime
import os
from typing import List, Dict, Optional, AsyncGenerator
import time
import httpx
from urllib.parse import urlparse, urljoin
from rich.text import Text  # Import Text

# Initialize logger for this module *before* first use
llm_logger = logging.getLogger(__name__)

# Local imports (assuming config is accessible from core)
try:
    from a3x.core.config import LLAMA_DEFAULT_HEADERS, LLAMA_SERVER_URL as _CONFIG_LLAMA_URL
except ImportError:
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}  # Sensible fallback
    _CONFIG_LLAMA_URL = None
    llm_logger.warning(
        "Could not import config from core.config or LLAMA_SERVER_URL is missing. Using defaults."
    )

# Define a default URL within the module
_DEFAULT_LLM_URL = "http://127.0.0.1:8080/completion"

# --- EXPORT THE DEFAULT URL ---
DEFAULT_LLM_URL = _DEFAULT_LLM_URL

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

# --- Define the LLMInterface Class ---
class LLMInterface:
    def __init__(
        self,
        llm_url: Optional[str] = None,
        model_name: str = "default_model",
        context_size: int = 4096,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initializes the LLM Interface."""
        self.llm_url = self._determine_llm_url(llm_url) # Use helper during init
        self.model_name = model_name
        self.context_size = context_size
        self.timeout = httpx.Timeout(timeout, connect=10.0) # Store as httpx.Timeout object
        self.headers = {"Content-Type": "application/json"}
        llm_logger.info(f"LLMInterface initialized. URL: {self.llm_url}, Model: {self.model_name}, Context: {self.context_size}")

    def _determine_llm_url(self, provided_url: Optional[str]) -> str:
        """Determines the LLM URL to use, checking environment variables, defaults, and ensuring /completion is appended for default server."""
        # Prioritize provided URL, then environment variable, then config, then default
        url_to_use = _DEFAULT_LLM_URL # Start with the module default

        if _CONFIG_LLAMA_URL: # Check config import first
             url_to_use = _CONFIG_LLAMA_URL
             llm_logger.debug(f"Using LLM URL from config: {url_to_use}")

        env_url = os.getenv("LLM_API_URL")
        if env_url:
            url_to_use = env_url
            llm_logger.debug(f"Using LLM URL from environment variable (overrides config): {url_to_use}")

        if provided_url: # Provided URL takes highest priority
            url_to_use = provided_url
            llm_logger.debug(f"Using LLM URL provided to constructor (overrides env/config): {url_to_use}")

        # Ensure /completion for default server URL unless path specified
        # Use urljoin for cleaner path handling
        base_url = url_to_use.rstrip('/')
        parsed_url = urlparse(base_url)
        is_default_host_port = parsed_url.hostname == "127.0.0.1" and parsed_url.port == 8080
        has_specific_path = parsed_url.path not in ["/", "", None] # Check if path is more than just root

        if is_default_host_port and not has_specific_path:
            # If it's the default host/port and no specific path is given, append /completion
            final_url = urljoin(base_url + '/', 'completion') # Ensures single slash before completion
            llm_logger.debug(f"Appended /completion to default base URL. Final URL: {final_url}")
            return final_url
        else:
             # If it has a specific path or isn't the default host/port, use as is
             llm_logger.debug(f"Using LLM URL as determined: {base_url}")
             return base_url

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        # timeout is now instance variable self.timeout
        # llm_url is now instance variable self.llm_url
        **kwargs # Allow extra generation parameters
    ) -> AsyncGenerator[str, None]:
        """
        Async LLM call using httpx (supports both streaming and non-streaming modes).
        Uses instance configuration for URL and timeout.
        Allows overriding generation parameters via kwargs.
        """
        target_url = self.llm_url # Use instance URL
        headers = self.headers     # Use instance headers

        # Determine payload format based on the *instance* URL
        if target_url.endswith("/completion"):
            # Format for /completion endpoint (llama.cpp default)
            full_prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
            llm_logger.info(f"[LLM /completion] Full prompt:\n-------\n{full_prompt}\n-------")

            # Base payload with defaults
            payload = {
                "prompt": full_prompt,
                "stream": stream,
                "temperature": 0.7,
                "max_tokens": self.context_size, # Use instance context size
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            # Update with any kwargs provided
            payload.update(kwargs)
            llm_logger.debug("Using /completion payload format with parameters: %s", payload)
        else:
            # Format for /v1/chat/completions endpoint (OpenAI standard)
            payload = {"messages": messages, "stream": stream}
            # Update with any kwargs provided
            payload.update(kwargs)
            llm_logger.debug("Using /v1/chat/completions payload format with parameters: %s", payload)

        llm_logger.debug(f"[LLM] Sending to {target_url} | Stream: {stream}")
        llm_logger.debug(f"[LLM] Payload (first 500 chars): {str(payload)[:500]}")

        # Use instance timeout
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
                                # Handle both /completion and chat/completions streaming formats
                                content = None
                                if 'content' in data: # llama.cpp stream format
                                    content = data.get("content")
                                else: # OpenAI stream format
                                    choices = data.get("choices", [])
                                    if choices and isinstance(choices, list) and len(choices) > 0:
                                        delta = choices[0].get("delta", {})
                                        if isinstance(delta, dict):
                                            content = delta.get("content")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                                llm_logger.warning(f"[LLM] Streaming decode/access error: {e} | line: {json_str}")
                else: # Non-streaming
                    data = response.json()
                    content = "" # Default to empty
                    if target_url.endswith("/completion"):
                        # Extract directly from 'content' key for llama.cpp /completion
                        content = data.get("content", "")
                        llm_logger.debug(f"Extracted content directly for /completion endpoint.")
                    else: # OpenAI format
                        choices = data.get("choices", [])
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            message = choices[0].get("message", {})
                            if isinstance(message, dict):
                                content = message.get("content", "")
                        llm_logger.debug(f"Attempted extraction using OpenAI structure.")

                    llm_logger.debug(f"[LLM] Non-streaming response content length: {len(content)}")
                    # Log the raw response with color if DEBUG level is active
                    if llm_logger.isEnabledFor(logging.DEBUG):
                        colored_response = Text(f"[LLM {target_url.split('/')[-1]}] Response:\n-------\n{content}\n-------", style="magenta")
                        llm_logger.debug(colored_response)
                    else:
                        llm_logger.info(f"[LLM {target_url.split('/')[-1]}] Response:\n-------\n{content}\n-------")
                    yield content # Yield the single complete response

            except httpx.RequestError as e:
                llm_logger.exception(f"[LLM] Request error: {e}")
                yield f"[LLM Error: Request failed - {e}]" # Yield error message
            except httpx.HTTPStatusError as e:
                llm_logger.error(f"[LLM] HTTP error {e.response.status_code}: {e.response.text}")
                yield f"[LLM Error: HTTP {e.response.status_code}]" # Yield error message
            except Exception as e:
                llm_logger.exception(f"[LLM] Unexpected error: {e}")
                yield f"[LLM Error: Unexpected failure - {e}]" # Yield error message
