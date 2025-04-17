import logging
from typing import Dict, Any

from a3x.core.skills import skill
from a3x.core.context import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name="self_expansion",
    description="Generates a related prompt, expanding on the original idea.",
    parameters={
        "prompt": { "type": "string", "description": "The original prompt to expand on." }
    }
)
async def self_expansion(ctx: _ToolExecutionContext, prompt: str) -> Dict[str, Any]:
    """
    Generates a related prompt, expanding on the original idea.

    Args:
        ctx: Execution context.
        prompt: The original prompt to expand on.

    Returns:
        Dict[str, Any]: {'status': 'success'/'error', 'data': {...}}
    """
    logger.info(f"Executing skill: self_expansion with prompt: {prompt}")
    try:
        # Example implementation: Add a detail to the prompt
        expanded_prompt = f"Expand on the following idea: {prompt}.  Consider the implications and potential consequences."
        result_data = {"expanded_prompt": expanded_prompt}
        return { "status": "success", "data": result_data }
    except Exception as e:
        logger.error(f"Error in skill self_expansion: {e}", exc_info=True)
        return { "status": "error", "data": { "message": f"Skill execution failed: {e}" } }