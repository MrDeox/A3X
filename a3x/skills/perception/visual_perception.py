import logging
import os
import json
import asyncio
import base64
import requests # Added for HTTP calls
from typing import Dict, Any, List, Optional
import datetime # For temporary file naming (alternative)
import importlib.util # To check for playwright
import httpx # Add httpx for async requests
import io
import mss
import mss.tools
from PIL import Image

# --- Playwright Check ---
_playwright_installed = importlib.util.find_spec("playwright")
if _playwright_installed:
    from playwright.async_api import async_playwright, Playwright, Error as PlaywrightError
else:
    async_playwright = None
    Playwright = None
    PlaywrightError = None
# --- End Playwright Check ---

# Core framework imports
from a3x.core.config import (
    LLM_PROVIDER,
    LLAMA_SERVER_URL, # Assuming this might be used for LLaVA
    # Add any specific LLaVA server URL config if needed
)
from a3x.core.llm_interface import call_llm
from a3x.core.skills import skill

logger = logging.getLogger(__name__)

# --- Configuration ---
try:
    # Assume que o arquivo da skill está em a3x/skills/perception/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    # Fallback se __file__ não estiver definido (ex: execução interativa)
    PROJECT_ROOT = os.path.abspath('.')

SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "memory", "screenshots")
SCREENSHOT_FILENAME = "screen.png"
# Directory creation moved inside function

# LLaVA Server Configuration
LLAVA_PORT = 8081
LLAVA_SERVER_URL = f"http://127.0.0.1:{LLAVA_PORT}" # Assuming LLaVA runs on port 8081
# <<< Use New API Port >>>
LLAVA_API_PORT = 9999
LLAVA_API_ENDPOINT = f"http://localhost:{LLAVA_API_PORT}/v1/chat/completions"
LLAVA_MODEL_NAME = "llava-1.5-7b" # Model name expected by the API server
# <<< REMOVE LLaVA API Config - Use default llama.cpp server >>>
# LLAVA_API_PORT = 9999
# LLAVA_API_ENDPOINT = f"http://localhost:{LLAVA_API_PORT}/v1/chat/completions"
# LLAVA_MODEL_NAME = "llava-1.5-7b"
# Model name (can be arbitrary for llama.cpp server, but good for clarity)
OBSIDIAN_MODEL_NAME = "obsidian-3b"
# Target the default llama.cpp server endpoint
LLAMA_CPP_ENDPOINT = f"http://localhost:8080/v1/chat/completions" # Assuming default port 8080

@skill(
    name="visual_perception",
    description="Captura um screenshot da tela usando Playwright, envia para o servidor LLaVA externo para análise e descreve a interface.",
    parameters={
        "ctx": (Context, None)
    }
)
async def visual_perception(ctx) -> Dict[str, Any]:
    """
    Captures a screenshot using Playwright, sends it to the unified llama.cpp server
    (running an Obsidian multimodal model) for analysis, and returns a text description.

    Args:
        ctx: The skill execution context (provides logger).

    Returns:
        A dictionary containing the analysis ('resumo', 'elementos_detectados', 'acoes_possiveis')
        or an error message.
    """
    ctx.logger.info("[VISUAL PERCEPTION] Starting skill execution.")

    # --- Dependency Check ---
    if not _playwright_installed or async_playwright is None or httpx is None: # Check httpx too
        ctx.logger.error("Playwright or httpx library not found. Please install them: pip install playwright httpx")
        ctx.logger.error("You also need to install browser binaries: playwright install chromium")
        return {"error": "Playwright library is required but not installed. Install with 'pip install playwright' and 'playwright install chromium'."}
    # --- End Dependency Check ---

    # Ensure screenshot directory exists
    try:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ctx.logger.debug(f"Ensured screenshot directory exists: {SCREENSHOT_DIR}")
    except Exception as e:
        ctx.logger.exception(f"Failed to create or access screenshot directory {SCREENSHOT_DIR}: ")
        return {"error": f"Failed to create/access screenshot directory: {e}"}

    screenshot_path = os.path.join(SCREENSHOT_DIR, SCREENSHOT_FILENAME)

    # 1. Capture Screenshot using Playwright
    async with async_playwright() as p:
        browser = None
        try:
            ctx.logger.info("Launching headless Chromium browser...")
            # Try launching Chromium first
            try:
                browser = await p.chromium.launch(headless=True)
                ctx.logger.debug("Chromium launched successfully.")
            except PlaywrightError as e:
                 ctx.logger.warning(f"Failed to launch Chromium ({e}), trying system default. Ensure browser is installed ('playwright install chromium').")
                 # Fallback or raise error if needed
                 # For now, let's try the default launch which might pick another browser if installed
                 try:
                     browser = await p.browser.launch(headless=True) # Less specific launch
                     ctx.logger.info("Launched default browser successfully as fallback.")
                 except PlaywrightError as fallback_e:
                     ctx.logger.error(f"Failed to launch any browser: {fallback_e}")
                     ctx.logger.error("Please ensure Playwright browsers are installed: run 'playwright install'")
                     return {"error": f"Failed to launch Playwright browser. Ensure it's installed ('playwright install'). Error: {fallback_e}"}

            # We need a page to take a screenshot, even if it's blank
            page = await browser.new_page()
            ctx.logger.info(f"Capturing screenshot to {screenshot_path}...")

            # Get screen dimensions (optional, might improve accuracy for some setups)
            # screen_size = await page.evaluate('''() => {
            #     return { width: window.screen.width, height: window.screen.height };
            # }''')
            # await page.set_viewport_size(screen_size)
            # ctx.logger.debug(f"Set viewport size to screen dimensions: {screen_size}")

            # Capture the full screen. full_page=True might not capture *everything* outside the browser window itself.
            # Taking a screenshot of the page is the standard Playwright way.
            # A true full *desktop* screenshot might require OS-specific tools or different libraries if Playwright's page screenshot isn't sufficient.
            # For now, we assume the page screenshot is the goal.
            await page.screenshot(path=screenshot_path, full_page=True) # Capture the virtual page

            ctx.logger.info("[VISUAL PERCEPTION] Screenshot captured successfully using Playwright.")

        except PlaywrightError as e:
            ctx.logger.exception("[VISUAL PERCEPTION] Playwright error during screenshot capture:")
            return {"error": f"Playwright error during screenshot: {e}. Ensure browsers are installed ('playwright install')."}
        except Exception as e:
            ctx.logger.exception("[VISUAL PERCEPTION] Failed to capture screenshot using Playwright:")
            return {"error": f"Failed to capture screenshot with Playwright: {e}"}
        finally:
            if browser:
                await browser.close()
                ctx.logger.debug("[VISUAL PERCEPTION] Playwright browser closed.")

    # 2. Prepare and Send to llama.cpp Server
    text_description = None # To store the final text result
    error_message = None
    payload = {} # Define payload scope outside try/except for finally block if needed

    try:
        ctx.logger.info(f"[VISUAL PERCEPTION] Preparing screenshot for llama.cpp multimodal server")
        # Read image and encode
        try:
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            ctx.logger.debug("[VISUAL PERCEPTION] Screenshot read and base64 encoded.")
            image_data_url = f"data:image/png;base64,{base64_image}"
        except Exception as read_err:
             ctx.logger.exception(f"[VISUAL PERCEPTION] Failed to read or encode screenshot {screenshot_path}:")
             raise IOError(f"Failed to read/encode screenshot: {read_err}") # Raise specific error to be caught below

        # Construct prompt text in Obsidian format
        prompt_text_user = "O que há nesta imagem?"
        prompt_text_formatted = f"<|im_start|>user\n{prompt_text_user}\n<image>\n###<|im_start|>assistant"

        # Prepare OpenAI-compatible payload for llama.cpp server
        payload = {
            "model": OBSIDIAN_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text_formatted},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ],
            "stream": False # Important for direct httpx call parsing
            # Add other parameters like temperature if needed
            # "temperature": 0.2,
        }

        # <<< MOVED: Use httpx again for direct call >>>
        ctx.logger.info(f"[VISUAL PERCEPTION] Sending request to llama.cpp server endpoint: {LLAMA_CPP_ENDPOINT}")
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(LLAMA_CPP_ENDPOINT, json=payload)

        ctx.logger.info(f"[VISUAL PERCEPTION] Received response from llama.cpp server (Status: {response.status_code}).")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        ctx.logger.debug(f"[VISUAL PERCEPTION] llama.cpp Raw Response JSON: {response_data}")

        # Extract text content (standard OpenAI format)
        try:
             text_description = response_data['choices'][0]['message']['content']
             if not isinstance(text_description, str) or not text_description.strip():
                 raise ValueError("Invalid or empty content in llama.cpp API response.")
             ctx.logger.info("[VISUAL PERCEPTION] Successfully extracted text description from llama.cpp API response.")
        except (KeyError, IndexError, ValueError, TypeError) as parse_err:
             ctx.logger.error(f"[VISUAL PERCEPTION] Failed to parse expected content from llama.cpp API response: {parse_err}")
             ctx.logger.error(f"[VISUAL PERCEPTION] Full API Response: {response_data}")
             raise ValueError(f"Could not parse description from llama.cpp API response: {parse_err}")

    except IOError as io_err:
        # Error reading/encoding screenshot
        error_message = str(io_err)
    # <<< CORRECTED: httpx Error Handling >>>
    except httpx.HTTPStatusError as http_err:
         # Handle HTTP errors (4xx, 5xx)
         status_code = http_err.response.status_code
         response_text = http_err.response.text
         ctx.logger.error(f"[VISUAL PERCEPTION] llama.cpp server returned error status {status_code}: {response_text[:500]}")
         error_message = f"Backend multimodal (llama.cpp) returned status {status_code}. Check server logs."
    except httpx.RequestError as req_err:
         # Handle connection errors, timeouts, etc.
         ctx.logger.error(f"[VISUAL PERCEPTION] Error during request to llama.cpp server ({LLAMA_CPP_ENDPOINT}): {req_err}")
         error_message = "Backend multimodal (llama.cpp) não disponível ou falhou."
    except ValueError as val_err:
        # Handle errors during content extraction/validation
        error_message = str(val_err)
    except Exception as e:
        # Catch other unexpected issues
        ctx.logger.exception("[VISUAL PERCEPTION] Unexpected error during llama.cpp communication or processing:")
        error_message = f"Unexpected error during visual perception processing: {e}"
    finally:
        # 3. Delete Temporary Screenshot
        try:
            if os.path.exists(screenshot_path): # Check if file exists before deleting
                os.remove(screenshot_path)
                ctx.logger.info(f"[VISUAL PERCEPTION] Temporary screenshot {screenshot_path} deleted.")
            else:
                ctx.logger.warning(f"[VISUAL PERCEPTION] Temporary screenshot {screenshot_path} not found for deletion.")
        except OSError as e:
            ctx.logger.warning(f"[VISUAL PERCEPTION] Could not delete temporary screenshot {screenshot_path}: {e}")

    ctx.logger.info("[VISUAL PERCEPTION] Skill execution finished.")

    # Return either the description or an error dictionary
    if error_message:
        return {"error": error_message}
    elif text_description:
        return {"description": text_description}
    else:
        # Should not happen if error handling is correct, but as a fallback
        return {"error": "Failed to get visual description for unknown reasons."}

# Example usage removed as it's a skill module 