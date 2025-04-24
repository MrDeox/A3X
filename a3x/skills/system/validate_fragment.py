import logging
import os
import importlib.util
import inspect
import sys
from pydantic import BaseModel, Field
from typing import Dict, Any, Type

from a3x.core.skills import skill, SkillContext  # Use correct import for skill decorator and context
from a3x.fragments.base import BaseFragment  # To check inheritance

logger = logging.getLogger(__name__)

class ValidateFragmentParams(BaseModel):
    fragment_path: str = Field(..., description="The relative path to the Python file containing the fragment class.")

@skill(
    name="validate_fragment",
    description="Validates a newly created fragment file for basic correctness (existence, importability, class structure).",
    parameters={
        "fragment_path": {
            "type": "string",
            "description": "The relative path to the Python file containing the fragment class.",
        }
    }
)
async def skill_validate_fragment(context: Any, action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a fragment Python file based on path.

    Checks:
    1. File existence.
    2. Python syntax validity (via import).
    3. Presence of exactly one class inheriting from BaseFragment.
    4. Presence of an 'execute' method in that class.
    """
    try:
        params = ValidateFragmentParams(**action_input)
        fragment_path_rel = params.fragment_path
        logger.info(f"Validating fragment at path: {fragment_path_rel}")

        # Ensure the path is relative to the workspace root if needed
        # Assuming context.workspace_root provides the absolute path to the workspace
        if hasattr(context, 'workspace_root') and context.workspace_root:
            fragment_path_abs = os.path.join(context.workspace_root, fragment_path_rel)
        else:
            # Fallback if workspace_root isn't available (might need adjustment)
            fragment_path_abs = os.path.abspath(fragment_path_rel)
            logger.warning(f"Workspace root not found in context, using absolute path: {fragment_path_abs}")

        validation_steps = {
            "file_exists": False,
            "can_import": False,
            "found_fragment_class": False,
            "correct_class_count": False,
            "has_execute_method": False
        }
        messages = []

        # 1. Check file existence
        if not os.path.isfile(fragment_path_abs):
            msg = f"Validation failed: Fragment file not found at {fragment_path_abs}"
            logger.error(msg)
            messages.append(msg)
            return {"status": "error", "message": msg, "details": validation_steps}
        validation_steps["file_exists"] = True
        logger.debug(f"File found: {fragment_path_abs}")

        # 2. Check importability (basic syntax check)
        module_name = os.path.splitext(os.path.basename(fragment_path_rel))[0]
        spec = importlib.util.spec_from_file_location(module_name, fragment_path_abs)
        if spec is None or spec.loader is None:
             msg = f"Validation failed: Could not create module spec for {fragment_path_abs}"
             logger.error(msg)
             messages.append(msg)
             return {"status": "error", "message": msg, "details": validation_steps}

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module # Add to sys.modules to handle relative imports within the fragment if any
        try:
            spec.loader.exec_module(module)
            validation_steps["can_import"] = True
            logger.debug(f"Module imported successfully: {module_name}")
        except SyntaxError as e:
            msg = f"Validation failed: Syntax error in {fragment_path_abs}: {e}"
            logger.error(msg)
            messages.append(msg)
            # Clean up failed module load
            if module_name in sys.modules: del sys.modules[module_name]
            return {"status": "error", "message": msg, "details": validation_steps}
        except Exception as e:
            msg = f"Validation failed: Error importing {fragment_path_abs}: {e}"
            logger.exception(msg) # Log with stack trace
            messages.append(msg)
            if module_name in sys.modules: del sys.modules[module_name]
            return {"status": "error", "message": msg, "details": validation_steps}

        # 3. Find classes inheriting from BaseFragment
        fragment_classes = []
        try:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseFragment) and obj is not BaseFragment:
                    fragment_classes.append(obj)
                    logger.debug(f"Found BaseFragment subclass: {name}")

            if fragment_classes:
                 validation_steps["found_fragment_class"] = True

            # 4. Check for exactly one fragment class
            if len(fragment_classes) == 1:
                validation_steps["correct_class_count"] = True
                fragment_class: Type[BaseFragment] = fragment_classes[0]
                logger.debug(f"Found exactly one fragment class: {fragment_class.__name__}")

                # 5. Check for 'execute' method
                if hasattr(fragment_class, 'execute') and callable(getattr(fragment_class, 'execute')):
                    validation_steps["has_execute_method"] = True
                    logger.debug(f"Fragment class '{fragment_class.__name__}' has an 'execute' method.")
                else:
                    msg = f"Validation failed: Fragment class '{fragment_class.__name__}' is missing an 'execute' method."
                    logger.error(msg)
                    messages.append(msg)
            elif len(fragment_classes) == 0:
                 msg = f"Validation failed: No class inheriting from BaseFragment found in {fragment_path_abs}."
                 logger.error(msg)
                 messages.append(msg)
            else:
                 msg = f"Validation failed: Found multiple ({len(fragment_classes)}) classes inheriting from BaseFragment in {fragment_path_abs}. Expected exactly one."
                 logger.error(msg)
                 messages.append(msg)

        finally:
            # Clean up: remove the imported module to avoid side effects
            if module_name in sys.modules:
                del sys.modules[module_name]
                logger.debug(f"Cleaned up module '{module_name}' from sys.modules")


        # Final result
        if all(validation_steps.values()):
            msg = f"Validation successful for fragment: {fragment_path_rel}"
            logger.info(msg)
            return {"status": "success", "message": msg, "details": validation_steps}
        else:
             final_message = f"Validation failed for fragment: {fragment_path_rel}. Issues: {'; '.join(messages)}"
             # Ensure the status reflects the failure even if no specific message was added in the last checks
             return {"status": "error", "message": final_message, "details": validation_steps}

    except Exception as e:
        logger.exception(f"Unexpected error during fragment validation for input {action_input}: {e}")
        return {"status": "error", "message": f"Internal validation error: {e}"} 