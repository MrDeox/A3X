"""
Skill para chamar dinamicamente outra skill registrada pelo nome.
"""

import logging
from typing import Dict, Any, Optional
import inspect # Added for async check
import asyncio

# Core imports
from a3x.core.skills import skill, get_skill
from a3x.core.context import Context

# Removed assumption comment about SkillContext

logger = logging.getLogger(__name__)

@skill(
    name="call_skill_by_name",
    description="Dynamically calls another registered skill by its name with the given arguments.",
    parameters={
        "skill_name": {"type": str, "description": "The exact name of the skill to call."},
        "skill_args": {"type": Optional[Dict[str, Any]], "default": None, "description": "Dictionary of arguments to pass to the target skill." }
    }
)
async def call_skill_by_name(ctx: Context, skill_name: str, skill_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Dynamically calls another registered skill by its name using the global registry.

    Args:
        ctx: The skill execution context (passed to the target skill).
        skill_name: The name of the skill to call.
        skill_args: A dictionary containing the arguments for the target skill.

    Returns:
        The dictionary result returned by the called skill, or an error dictionary.
    """
    logger.info(f"Attempting to dynamically call skill: '{skill_name}' with args: {skill_args}")

    if skill_args is None:
        skill_args = {} # Default to empty dict if None

    # Use the global skills registry function get_skill
    skill_info = get_skill(skill_name)

    if not skill_info:
        logger.error(f"Skill '{skill_name}' not found in the global registry via get_skill.")
        return {"error": f"Skill '{skill_name}' not found in the global registry."}

    skill_func = skill_info.get("function")
    skill_instance = skill_info.get("instance")

    if not callable(skill_func):
        logger.error(f"Skill '{skill_name}' function found via get_skill is not callable.")
        return {"error": f"Skill '{skill_name}' implementation is invalid (not callable)."}

    try:
        # TODO: Argument validation could still be useful here based on skill metadata if available
        logger.debug(f"Executing skill '{skill_name}'...")

        if skill_instance:
            logger.debug(f"Calling '{skill_name}' as an instance method.")
            result = await skill_func(skill_instance, ctx, **skill_args)
        else:
            logger.debug(f"Calling '{skill_name}' as a standalone function.")
            result = await skill_func(ctx, **skill_args)

        logger.info(f"Skill '{skill_name}' executed successfully.")
        return result
    except Exception as e:
        logger.exception(f"Error executing skill '{skill_name}':")
        return {"error": f"Error during '{skill_name}' execution: {e}"} 