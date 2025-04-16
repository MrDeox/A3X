import logging
from typing import Dict, Any

from a3x.core.skills import skill
from a3x.core.tool_executor import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name="list_skills",
    description="Lists the skills currently allowed for the active fragment/manager, along with their descriptions.",
    parameters={} # No input parameters needed
)
async def list_skills(ctx: _ToolExecutionContext) -> Dict[str, Any]:
    """
    Retrieves and formats the list of allowed skills and their descriptions
    from the execution context.
    """
    log_prefix = "[Skill: list_skills]"
    allowed_skills = ctx.allowed_skills
    all_tools = ctx.tools_dict # The full registry

    if not allowed_skills:
        logger.warning(f"{log_prefix} No allowed skills found in the current context.")
        return {"status": "success", "data": {"message": "No skills are currently allowed or available in this context."}}

    skill_descriptions = []
    for skill_name in sorted(allowed_skills): # Sort for consistent output
        skill_info = all_tools.get(skill_name)
        if skill_info and isinstance(skill_info, dict) and "description" in skill_info:
            description = skill_info.get("description", "No description available.")
            skill_descriptions.append(f"- {skill_name}: {description}")
        else:
            # This shouldn't happen if the context/registry are correct, but log it.
            logger.warning(f"{log_prefix} Allowed skill '{skill_name}' not found or missing description in registry.")
            skill_descriptions.append(f"- {skill_name}: (Description unavailable)")

    formatted_list = "\n".join(skill_descriptions)
    logger.info(f"{log_prefix} Returning list of {len(skill_descriptions)} allowed skills.")
    
    return {"status": "success", "data": {"allowed_skills_list": formatted_list}} 