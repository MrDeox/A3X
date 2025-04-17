# a3x/skills/reload_generated_skills.py
import logging
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Any

from a3x.core.skills import skill
from a3x.core.context import _ToolExecutionContext
from a3x.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Directory where generated skills are expected
GENERATED_SKILLS_DIR = Path("a3x/skills/generated")
GENERATED_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

@skill(
    name="reload_generated_skills",
    description="Scans the generated skills directory and dynamically registers any new, valid skills found.",
    parameters={ # No application-level parameters needed besides context
        # "ctx": {"type": _ToolExecutionContext, "description": "Execution context."}, # Removed ctx from here
    }
)
async def reload_generated_skills(ctx: _ToolExecutionContext) -> Dict[str, Any]:
    """
    Dynamically loads and registers Python skill modules found in the 
    'a3x/skills/generated' directory.

    Args:
        ctx: The execution context, providing access to the ToolRegistry.

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'.
        - data: {'message': 'Summary of actions', 'new_skills_registered': count}
    """
    logger.info("Attempting to reload and register skills from generated directory...")
    
    if not hasattr(ctx, 'tools_dict') or not isinstance(ctx.tools_dict, ToolRegistry):
        msg = "ToolRegistry (via tools_dict) not found or invalid type in execution context."
        logger.error(msg)
        return {"status": "error", "data": {"message": msg, "new_skills_registered": 0}}

    tool_registry = ctx.tools_dict
    generated_module_prefix = "a3x.skills.generated."
    newly_registered_count = 0
    errors_found = 0

    try:
        # Use pkgutil to properly discover modules within the directory
        for module_finder, name, ispkg in pkgutil.iter_modules([str(GENERATED_SKILLS_DIR)]):
            if ispkg:
                continue # Skip packages if any exist

            full_module_name = generated_module_prefix + name
            logger.debug(f"Found potential skill module: {name}. Attempting import as {full_module_name}")

            try:
                # Dynamically import the module
                module = importlib.import_module(full_module_name)
                # Force reload to pick up changes if module was somehow already imported
                module = importlib.reload(module)
                logger.debug(f"Successfully imported/reloaded module: {full_module_name}")

                # Inspect the module for functions with the @skill decorator
                for attribute_name, attribute in inspect.getmembers(module):
                    if inspect.isfunction(attribute) and hasattr(attribute, '_is_skill') and getattr(attribute, '_is_skill') is True:
                        skill_name = getattr(attribute, '_skill_name', None)
                        skill_description = getattr(attribute, '_skill_description', attribute.__doc__)
                        skill_parameters = getattr(attribute, '_skill_parameters', None)
                        
                        if not skill_name:
                            logger.warning(f"Skill function '{attribute_name}' in {full_module_name} is missing a name in its decorator.")
                            continue
                            
                        # Generate schema if not explicitly provided (use existing helper if available)
                        # For simplicity, assume schema might be needed later or handled by registry
                        # if not skill_parameters: # Basic schema generation
                        #    skill_parameters = { ... } # Simplified placeholder 

                        # Check if skill is already registered
                        if tool_registry.get_tool(skill_name):
                            logger.debug(f"Skill '{skill_name}' from {full_module_name} is already registered. Skipping.")
                            continue

                        # Register the new skill (as standalone function, no instance)
                        try:
                            # We need a proper schema here. Let's assume the decorator stored it, 
                            # or we generate a basic one. Using None for now.
                            # TODO: Enhance schema generation/retrieval from decorator
                            tool_registry.register_tool(
                                name=skill_name, 
                                instance=None, 
                                tool=attribute, 
                                schema={'name': skill_name, 'description': skill_description or '', 'parameters': skill_parameters or {}}
                            )
                            logger.info(f"Successfully registered new skill: '{skill_name}' from {full_module_name}")
                            newly_registered_count += 1
                        except Exception as reg_err:
                            logger.error(f"Failed to register skill '{skill_name}' from {full_module_name}: {reg_err}", exc_info=True)
                            errors_found += 1

            except ImportError as e:
                logger.error(f"Failed to import module {full_module_name}: {e}", exc_info=True)
                errors_found += 1
            except Exception as e:
                logger.error(f"Error processing module {full_module_name}: {e}", exc_info=True)
                errors_found += 1

        final_message = f"Skill reload process completed. Registered {newly_registered_count} new skills."
        if errors_found > 0:
            final_message += f" Encountered {errors_found} errors during loading/registration."
            logger.warning(final_message)
            return {"status": "success_with_errors", "data": {"message": final_message, "new_skills_registered": newly_registered_count}}
        else:
            logger.info(final_message)
            return {"status": "success", "data": {"message": final_message, "new_skills_registered": newly_registered_count}}

    except Exception as e:
        logger.exception(f"Unexpected error during skill reloading: {e}")
        return {"status": "error", "data": {"message": f"Failed to reload skills: {e}", "new_skills_registered": newly_registered_count}} 