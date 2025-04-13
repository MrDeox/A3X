import requests
import logging
import json
import re
from typing import Dict, Any, Optional

# Use configurações centralizadas
# from a3x.core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS
from a3x.core.skills import skill  # Import skill decorator
from a3x.core.context import Context # Added import

# Configure logger para esta skill
logger = logging.getLogger(__name__)


# Renamed function and added @skill decorator
@skill(
    name="generate_code",
    description="Generates code based on a detailed prompt and optional existing code.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM access."},
        "prompt": {"type": str, "description": "Detailed prompt specifying the code to be generated."},
        "language": {"type": Optional[str], "default": "python", "description": "Programming language for the code (e.g., 'python', 'javascript')."},
        "existing_code": {"type": Optional[str], "default": None, "description": "Optional existing code snippet to modify or build upon."},
        "construct_type": {"type": Optional[str], "default": "function", "description": "Type of code construct (e.g., 'function', 'class', 'script')."}
    }
)
async def generate_code(
    context: Context,
    prompt: str,
    language: Optional[str] = "python",
    existing_code: Optional[str] = None,
    construct_type: Optional[str] = "function"
) -> Dict[str, Any]:
    """
    Generates code using the LLM with chat-style prompting and improved code extraction.

    Args:
        context (Context): Execution context for LLM access.
        prompt (str): Detailed prompt specifying the code to be generated.
        language (str, optional): Programming language. Defaults to 'python'.
        existing_code (str | None, optional): Optional existing code snippet to modify or build upon. Defaults to None.
        construct_type (str | None, optional): Type of code construct (e.g., 'function', 'class', 'script'). Defaults to 'function'.

    Returns:
        Dict[str, Any]: Standardized dictionary with status, action, and data (code).
    """
    logger.info(f"Executing skill 'generate_code' for language '{language}'")
    logger.debug(
        f"Purpose: {prompt}, Construct: {construct_type}, Context provided: {existing_code is not None}"
    )

    # Basic validation (purpose is mandatory)
    if not prompt:
        logger.error("Parameter 'prompt' is mandatory for generate_code.")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": "Error: Code purpose (prompt) was not specified."},
        }

    try:
        # --- NEW CHAT COMPLETIONS LLM CALL LOGIC ---
        logger.debug("Building chat prompt for code generation...")

        # System prompt for the generation skill
        system_prompt = f"You are an expert {language} programming assistant. Your task is to generate concise and functional code."
        # User prompt with the specific purpose and context
        user_prompt_lines = [
            f"Generate ONLY the {language} code for the following task: {prompt}."
        ]
        if construct_type:
            user_prompt_lines.append(f"The code should be a {construct_type}.")
        if existing_code:
            user_prompt_lines.append(
                f"Consider the following context or existing code:\n```\n{existing_code}\n```"
            )
        user_prompt_lines.append(
            "Respond ONLY with the raw code block (e.g., within ```<lang>...``` or just the code itself), without any explanation before or after."
        )
        user_prompt = "\n".join(user_prompt_lines)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use configured URL and Headers
        chat_url = LLAMA_SERVER_URL
        headers = LLAMA_DEFAULT_HEADERS

        # Ensure URL points to chat completions (best effort)
        if not chat_url.endswith("/chat/completions"):
            logger.warning(
                f"Configured LLM URL '{chat_url}' might not be for chat completions. Trying to adjust..."
            )
            if chat_url.endswith("/v1") or chat_url.endswith("/v1/"):
                chat_url = chat_url.rstrip("/") + "/chat/completions"
            else:
                # Assume it's a base URL, append standard path
                chat_url = chat_url.rstrip("/") + "/v1/chat/completions"
            logger.info(f"Adjusted URL for chat: {chat_url}")

        payload = {
            "messages": messages,
            "temperature": 0.2,  # Low temperature for code
            "max_tokens": 2048,  # Increased limit
            "stream": False,
        }

        logger.debug(f"Sending code generation request to: {chat_url}")
        response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        generated_content = (
            response_data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        logger.debug(
            f"Raw LLM Response Content:\n---\n{generated_content[:500]}...\n---"
        )

        # --- Improved Code Extraction Logic ---
        code = generated_content
        extracted_via_markdown = False

        # 1. Try extracting from markdown block
        lang_pattern = (
            language if language else r"\w+"
        )  # Use specific lang or generic if not provided
        code_match = re.search(
            rf"```{lang_pattern}\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```",
            code,
            re.DOTALL,
        )
        if code_match:
            extracted_code = next(
                (group for group in code_match.groups() if group is not None), None
            )
            if extracted_code is not None:
                code = extracted_code.strip()
                extracted_via_markdown = True
                logger.info("Code extracted from markdown block.")

        # 2. Fallback cleanup if not extracted via markdown
        if (
            not extracted_via_markdown and code
        ):  # Only clean if not empty and not from markdown
            logger.debug(
                "Markdown block not found or empty, attempting fallback cleanup."
            )
            # Remove common leading/trailing explanations
            patterns_to_remove_prefix = [
                r"(?im)^\s*(?:aqui está o código|o código é|código solicitado|claro(?:,|,) aqui está|você pode usar|gerando código|defina a função|função python|função javascript|código python|código javascript|// Tarefa:.*?|// Código python:|// Código javascript:)\s*\n?",
                r"^\s*```(?:\w+)?\s*\n",
            ]
            patterns_to_remove_suffix = [
                r"(?m)\n\s*(?:#.*|\/\/.*|Explicação:.*|Nota:.*|```)\s*$"
            ]
            cleaned_code = code
            for pattern in patterns_to_remove_prefix:
                cleaned_code = re.sub(pattern, "", cleaned_code, count=1)
            for pattern in patterns_to_remove_suffix:
                cleaned_code = re.sub(pattern, "", cleaned_code)

            if cleaned_code.strip() != code:
                logger.info("Fallback cleanup removed potential explanatory text.")
                code = cleaned_code.strip()
            else:
                logger.debug("Fallback cleanup did not change the content.")
                code = code.strip()
        elif not code:
            logger.warning("LLM returned empty content initially.")

        # Final check if code is empty after processing
        if not code:
            logger.error("Code generation resulted in empty code after processing.")
            return {
                "status": "error",
                "action": "code_generation_failed",
                "data": {
                    "message": f"LLM response did not contain valid code after extraction/cleanup. Raw response: '{generated_content[:200]}...'"
                },
            }

        logger.info(f"Code generated successfully in {language}.")
        logger.debug(f"Generated Code Snippet:\n---\n{code[:500]}...\n---")
        return {
            "status": "success",
            "action": "code_generated",
            "data": {
                "code": code,
                "language": language,
                "construct_type": construct_type,
                "message": f"Code generated successfully in {language}.",
            },
        }

    # --- Keep Existing Exception Handling ---
    except requests.exceptions.Timeout:
        logger.error("LLM timed out during code generation (>120s).")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": "Timeout: LLM took too long to generate code."},
        }
    except requests.exceptions.RequestException as e:
        error_details = str(e)
        if e.response is not None:
            error_details += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}..."
        logger.error(
            f"HTTP Error communicating with LLM for code generation: {error_details}"
        )
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"LLM Communication Error: {error_details}"},
        }
    except json.JSONDecodeError as e:
        raw_resp_text = "N/A"
        # Ensure response object exists before accessing .text
        if "response" in locals() and hasattr(response, "text"):
            raw_resp_text = response.text[:200]
        logger.error(
            f"Failed to decode JSON response from LLM: {e}. Response text (start): '{raw_resp_text}'"
        )
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {
                "message": f"Invalid JSON response from LLM: {e}. Response start: '{raw_resp_text}'"
            },
        }
    except Exception as e:
        logger.exception("Unexpected error during code generation:")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"Unexpected error during code generation: {str(e)}"},
        }
