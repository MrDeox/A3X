# a3x/skills/core/llm_error_diagnosis.py
import logging
from typing import Dict, Any, List, Optional
import json
import re

from a3x.core.skills import skill
# Import the class and default URL, not the function
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
# Import Context for type hinting
from a3x.core.context import Context

logger = logging.getLogger(__name__)

def _parse_llm_diagnosis_response(response_text: str) -> Dict[str, Any]:
    """
    Parses the LLM response to extract diagnosis and suggested actions.
    Assumes a semi-structured format like:
    Diagnosis: [The diagnosis text]
    Suggested Actions:
    - [Action 1]
    - [Action 2]
    """
    diagnosis = "Could not parse diagnosis from LLM response."
    suggested_actions = []

    try:
        # Extract Diagnosis
        diag_match = re.search(r"Diagnosis:\s*(.*?)(?:\nSuggested Actions:|\Z)", response_text, re.DOTALL | re.IGNORECASE)
        if diag_match:
            diagnosis = diag_match.group(1).strip()

        # Extract Suggested Actions
        actions_match = re.search(r"Suggested Actions:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            # Split actions by newline and strip leading/trailing whitespace and hyphens
            suggested_actions = [action.strip("-* ").strip() for action in actions_text.split('\n') if action.strip("-* ").strip()]

    except Exception as e:
        logger.error(f"Error parsing LLM diagnosis response: {e}\nRaw Response:\n{response_text}")
        # Keep default values

    # Fallback if parsing fails completely
    if diagnosis == "Could not parse diagnosis from LLM response." and not suggested_actions:
         diagnosis = f"LLM response did not follow expected format. Raw response: {response_text[:200]}..."

    return {"diagnosis": diagnosis, "suggested_actions": suggested_actions}


@skill(
    name="llm_error_diagnosis",
    description="Analyzes error messages, tracebacks, and execution context using an LLM to provide a semantic diagnosis and suggest potential corrective actions or a recovery plan.",
    parameters={
        "error_message": (str, ...),
        "traceback": (str, None), # Optional traceback string
        "execution_context": (dict, None), # Optional context (objective, failed_step, last_action, etc.)
        # Removed llm_url parameter, it should be retrieved from context
    },
)
async def llm_error_diagnosis_skill(
    ctx: Context, # Added context parameter
    error_message: str,
    traceback: Optional[str] = None,
    execution_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Uses an LLM to diagnose an error and suggest recovery actions.
    """
    logger.info(f"Initiating LLM error diagnosis for: {error_message[:100]}...")

    # --- Construct Prompt ---
    prompt_lines = [
        "You are an expert system debugger. Analyze the following error details and provide a diagnosis and suggested recovery actions.",
        "Focus on the root cause and provide specific, actionable steps.",
        "\n--- Error Details ---",
        f"Error Message: {error_message}",
    ]
    if traceback:
        prompt_lines.append(f"\nTraceback:\n```\n{traceback}\n```")
    if execution_context:
        # Ensure context is serializable before logging/prompting
        try:
            context_str = json.dumps(execution_context, indent=2, ensure_ascii=False, default=str) # Use default=str for non-serializable
            prompt_lines.append(f"\nExecution Context:\n```json\n{context_str}\n```")
        except Exception as json_err:
            logger.warning(f"Could not serialize execution_context for prompt: {json_err}")
            prompt_lines.append("\nExecution Context: (Could not serialize for display)")

    prompt_lines.extend([
        "\n--- Analysis Request ---",
        "1. Provide a concise Diagnosis of the likely root cause.",
        "2. List specific Suggested Actions for recovery or debugging.",
        "\n--- Response Format ---",
        "Diagnosis: [Your diagnosis here]",
        "Suggested Actions:",
        "- [Action 1]",
        "- [Action 2]",
        "- ...",
    ])

    prompt_text = "\n".join(prompt_lines)
    # Use simple user role for now
    llm_call_params = {
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        # Add other specific params like max_tokens if needed
    }

    # --- Get LLM Interface --- 
    if hasattr(ctx, 'llm_interface') and isinstance(ctx.llm_interface, LLMInterface):
        llm_interface = ctx.llm_interface
        logger.debug("Using LLMInterface from context for error diagnosis.")
    else:
        # Fallback: Use default URL (consider if ctx has llm_url attribute)
        llm_url = getattr(ctx, 'llm_url', DEFAULT_LLM_URL)
        logger.warning(f"LLMInterface not found in context. Creating temporary instance for error diagnosis with URL: {llm_url}")
        llm_interface = LLMInterface(llm_url=llm_url)

    # --- Call LLM --- 
    llm_response_raw = ""
    try:
        logger.info("Calling LLM for error diagnosis...")
        # Use the LLMInterface instance
        async for chunk in llm_interface.call_llm(**llm_call_params):
             llm_response_raw += chunk

        if not llm_response_raw:
             raise ValueError("LLM returned an empty response.")

        logger.info("LLM diagnosis response received.")
        logger.debug(f"Raw LLM Diagnosis Response:\n{llm_response_raw}")

    except Exception as e:
        logger.exception("Error calling LLM for diagnosis:")
        return {"status": "error", "action": "llm_call_failed", "data": {"message": f"Failed to get diagnosis from LLM: {e}"}}

    # --- Parse Response ---
    parsed_result = _parse_llm_diagnosis_response(llm_response_raw)

    # --- Return Result ---
    return {
        "status": "success",
        "action": "diagnosis_provided",
        "data": {
            "diagnosis": parsed_result["diagnosis"],
            "suggested_actions": parsed_result["suggested_actions"],
        }
    }