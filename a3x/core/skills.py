# core/tools.py
import os
import sys
import logging
import importlib
import inspect
import pkgutil  # <<< ADDED IMPORT >>>
from typing import Dict, Any, Callable, Optional, Union, Type, List, get_origin, get_args, TYPE_CHECKING # Added List, get_origin, get_args, TYPE_CHECKING
from pydantic import BaseModel, create_model, ConfigDict, Field, ValidationError # <<< ADDED ConfigDict, Field, ValidationError >>>
from collections import namedtuple
from pathlib import Path
from pydantic import PydanticUserError # Import the specific error
import json
import asyncio

# <<< Moved Imports into TYPE_CHECKING block >>>
if TYPE_CHECKING:
    from a3x.fragments.base import BaseFragment
    from a3x.fragments.manager_fragment import ManagerFragment
    from a3x.core.context import _ToolExecutionContext, SharedTaskContext, FragmentContext # Added contexts
    import a3x # Import the top-level package for type eval

# Placeholder if Context isn't defined/importable easily
# class ContextPlaceholder:
#    pass
# Context = ContextPlaceholder
# <<< Use specific contexts defined >>>
_ToolExecutionContext = namedtuple("_ToolExecutionContext", ["logger", "workspace_root", "llm_url", "tools_dict", "llm_interface", "fragment_registry", "shared_task_context", "allowed_skills", "skill_instance", "memory_manager"]) # Re-declare for runtime resolution if needed?
SkillContext = _ToolExecutionContext # Alias for common usage

# Initialize logger before potentially using it in print statements
logger = logging.getLogger(__name__)

# <<< ADDED: Base class for Context-Aware Skills >>>
# class ContextAwareSkill:
#     """Base class for skills that need access to a specific execution context ID."""
#     def __init__(self, context_id: str):
#         self._context_id = context_id

#     @property
#     def context_id(self) -> str:
#         return self._context_id

# Global skill registry (Dictionary to store skills)
SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {}
# <<< Global Pydantic Model Registry >>>
PYDANTIC_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def get_skill_registry() -> Dict[str, Dict[str, Any]]:
    """Returns the global skill registry."""
    return SKILL_REGISTRY

def get_skill(name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a skill definition from the registry by name.

    Args:
        name: The name of the skill to retrieve.

    Returns:
        The skill definition dictionary if found, otherwise None.
    """
    return SKILL_REGISTRY.get(name)


def get_skill_descriptions(indent: int = 0, include_params: bool = True) -> str:
    """
    Generates a formatted string describing all registered skills,
    including their parameters and descriptions.

    Args:
        indent: The number of spaces to indent the output.
        include_params: Whether to include parameter details in the description.

    Returns:
        A formatted string listing all skills and their details.
    """
    prefix = " " * indent
    output = ""
    for name, skill_info in SKILL_REGISTRY.items():
        output += f"{prefix}- Name: {name}\n"
        output += f"{prefix}  Description: {skill_info['description']}\n"
        if include_params and 'parameters' in skill_info:
            output += f"{prefix}  Parameters:\n"
            for param_name, param_info in skill_info['parameters'].items():
                param_type = param_info.get('type', 'Any')
                # Format complex types like Optional[str] or List[Dict[str, Any]] better
                param_type_str = str(param_type).replace('typing.', '')
                desc = param_info.get('description', '')
                default_val = param_info.get('default', inspect.Parameter.empty)
                default_str = f" (default: {default_val})" if default_val is not inspect.Parameter.empty else ""
                optional_str = " (optional)" if param_info.get('optional') else "" # Mark optional params clearly
                output += f"{prefix}    - {param_name}: {param_type_str}{optional_str}{default_str} - {desc}\n"
        output += "\n"  # Add a newline between skills
    return output


def skill(name: str, description: str, parameters: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    Decorator to register a function as a skill.

    Args:
        name: The unique name of the skill (used for calling it).
        description: A clear description of what the skill does.
        parameters: A dictionary describing the parameters the skill accepts.
                    Each key is the parameter name.
                    The value is a dictionary with:
                        - 'type': The expected type (e.g., str, int, bool, list, dict, Optional[str]). Use strings for complex/forward refs.
                        - 'description': A clear description of the parameter.
                        - 'optional': (Optional) Boolean, True if the parameter is not required.
                        - 'default': (Optional) The default value if the parameter is optional.
    """
    if parameters is None:
        parameters = {} # Ensure parameters is always a dict

    def decorator(func: Callable):
        logger.debug(f"Registering skill: {name}")

        # --- Enhanced Parameter Introspection ---
        sig = inspect.signature(func)
        func_params = sig.parameters
        processed_params = {} # Store detailed param info for registry and schema

        # 1. Align defined parameters with function signature
        for param_name, param_def in parameters.items():
            if param_name not in func_params:
                logger.warning(f"Skill '{name}': Parameter '{param_name}' defined in decorator but not in function signature. Skipping.")
                continue

            func_param = func_params[param_name]
            # Prioritize decorator definition for type and description
            processed_params[param_name] = {
                'type': param_def.get('type', func_param.annotation if func_param.annotation != inspect.Parameter.empty else Any),
                'description': param_def.get('description', ''),
                'optional': param_def.get('optional', func_param.default != inspect.Parameter.empty),
                'default': param_def.get('default', func_param.default if func_param.default != inspect.Parameter.empty else inspect.Parameter.empty)
            }

        # 2. Add parameters from signature not explicitly defined in decorator (excluding context)
        first_param_name = next(iter(func_params)) if func_params else None
        for param_name, func_param in func_params.items():
            # Skip the first parameter if it's likely context (heuristic)
            if param_name == first_param_name and param_name not in parameters:
                 logger.debug(f"Skill '{name}': Assuming first parameter '{param_name}' is context, skipping explicit registration.")
                 continue
            if param_name not in processed_params:
                # Skip parameters starting with _ (convention for internal/unused)
                if param_name.startswith('_'):
                     logger.debug(f"Skill '{name}': Skipping parameter '{param_name}' starting with underscore from signature.")
                     continue
                
                param_type = func_param.annotation if func_param.annotation != inspect.Parameter.empty else Any
                is_optional = func_param.default != inspect.Parameter.empty
                default_val = func_param.default if is_optional else inspect.Parameter.empty
                processed_params[param_name] = {
                    'type': param_type,
                    'description': '', # No description available from signature alone
                    'optional': is_optional,
                    'default': default_val
                }
                logger.debug(f"Skill '{name}': Parameter '{param_name}' added from function signature.")


        # <<< ADDED DYNAMIC IMPORTS HERE >>>
        # Import types needed for Pydantic model_rebuild resolution dynamically
        # This avoids top-level circular imports but makes the types available at decoration time.
        try:
            from a3x.fragments.base import BaseFragment, FragmentContext
            from a3x.fragments.manager_fragment import ManagerFragment
            from a3x.core.context import Context # Import base Context as it might be used
            # <<< REMOVED AgentContext from dynamic import >>>
            from a3x.core.context import SharedTaskContext, _ToolExecutionContext
            # Add any other types that might appear in skill parameter annotations
            # and cause UndefinedAnnotation errors during rebuild
        except ImportError as e:
            logger.warning(f"Skill '{name}': Could not dynamically import types needed for Pydantic rebuild. Forward references might fail. Error: {e}")
            # Define placeholders if imports fail to prevent crashes later, though functionality might be limited
            BaseFragment = type('BaseFragmentPlaceholder', (), {})
            FragmentContext = type('FragmentContextPlaceholder', (), {})
            ManagerFragment = type('ManagerFragmentPlaceholder', (), {})
            Context = type('ContextPlaceholder', (), {}) # Add placeholder for Context too
            SharedTaskContext = type('SharedTaskContextPlaceholder', (), {})
            # AgentContext = type('AgentContextPlaceholder', (), {}) # Removed
            _ToolExecutionContext = type('ToolExecutionContextPlaceholder', (), {})
        # <<< END ADDED DYNAMIC IMPORTS >>>

        # Create Pydantic schema dynamically
        pydantic_fields: Dict[str, tuple[Any, Any]] = {} # Use Any for type hint flexibility
        # param_details_for_registry = {} # No longer needed for separate storage

        # <<< REVISED SCHEMA GENERATION LOGIC (handle direct dict vs. schema object) >>>
        params_to_process = processed_params # Use the merged params

        # Map string type names to actual types (handle common cases and complex types)
        type_mapping = {
            'str': str, 'string': str,
            'int': int, 'integer': int,
            'float': float,
            'bool': bool, 'boolean': bool,
            'list': List, 'array': List,
            'dict': Dict, 'object': Dict,
            'any': Any,
             # Add mappings for types imported dynamically if needed by string refs
            'BaseFragment': BaseFragment,
            'FragmentContext': FragmentContext,
            'ManagerFragment': ManagerFragment,
            'SharedTaskContext': SharedTaskContext,
            # 'AgentContext': AgentContext, # Removed
            '_ToolExecutionContext': _ToolExecutionContext,
            'Context': Context # Map the base Context class
        }

        for param_name, details in params_to_process.items():
            param_type_repr = details.get('type', Any)
            is_optional = details.get('optional', False)
            default_value = details.get('default', inspect.Parameter.empty)

            actual_type = Any # Default to Any
            try:
                # Try evaluating string annotations (carefully)
                if isinstance(param_type_repr, str):
                     # Simple mapping first
                     if param_type_repr.lower() in type_mapping:
                         actual_type = type_mapping[param_type_repr.lower()]
                     else:
                         # Attempt to evaluate more complex types like Optional[str], List[Dict[str, Any]]
                         # This requires the types (Optional, List, Dict, etc.) and potential custom types
                         # to be available in the evaluation scope.
                         try:
                             # Define a local scope with common types + dynamically imported ones
                             # <<< ADDED a3x to eval_scope >>>
                             eval_scope = {
                                 'Optional': Optional, 'List': List, 'Dict': Dict, 'Any': Any, 'Union': Union,
                                 'str': str, 'int': int, 'float': float, 'bool': bool, 'dict': dict, 'list': list,
                                 'BaseFragment': BaseFragment, 'FragmentContext': FragmentContext,
                                 'ManagerFragment': ManagerFragment, 'SharedTaskContext': SharedTaskContext,
                                 '_ToolExecutionContext': _ToolExecutionContext,
                                 'Context': Context,
                                 'a3x': sys.modules.get('a3x') # Add top-level module if needed
                                 # Add other necessary types here
                             }
                             # Ensure a3x module is actually loaded
                             if 'a3x' not in sys.modules:
                                 try:
                                     import a3x
                                     eval_scope['a3x'] = a3x
                                 except ImportError:
                                      logger.warning("Could not import top-level 'a3x' package for type evaluation.")
                                      eval_scope['a3x'] = None # Set to None if import fails
                             
                             # Only evaluate if a3x is needed and available, or if not needed
                             if 'a3x.' not in param_type_repr or eval_scope.get('a3x'):
                                 actual_type = eval(param_type_repr, globals(), eval_scope)
                             else:
                                 logger.warning(f"Skill '{name}', Param '{param_name}': Cannot evaluate type string '{param_type_repr}' because 'a3x' module is required but not loaded. Defaulting to Any.")
                                 actual_type = Any # Fallback if a3x needed but not loaded

                         except (NameError, SyntaxError, TypeError, AttributeError) as eval_err: # Added AttributeError
                              logger.warning(f"Skill '{name}', Param '{param_name}': Failed to evaluate type string '{param_type_repr}'. Defaulting to Any. Error: {eval_err}")
                              actual_type = Any # Fallback
                elif inspect.isclass(param_type_repr) or get_origin(param_type_repr): # Handle actual types or typing generics (List, Optional)
                    actual_type = param_type_repr
                else:
                     logger.warning(f"Skill '{name}', Param '{param_name}': Unhandled type representation '{param_type_repr}'. Defaulting to Any.")
                     actual_type = Any # Fallback for unknown representations

            except Exception as e:
                logger.error(f"Skill '{name}', Param '{param_name}': Unexpected error processing type '{param_type_repr}'. Defaulting to Any. Error: {e}")
                actual_type = Any

            # <<< ADDED: Explicitly check if the resolved type is a Context class >>>
            # <<< This MUST run AFTER actual_type is resolved >>>
            is_context_type = False
            if inspect.isclass(actual_type):
                try:
                    # Check against the dynamically imported context types
                    if issubclass(actual_type, (Context, FragmentContext, SharedTaskContext, _ToolExecutionContext)):
                        is_context_type = True
                except TypeError: # Handle cases where issubclass gets non-class types (like Any, Optional)
                    pass
                except NameError: # Handle if context types weren't imported successfully
                    logger.warning(f"Skill '{name}', Param '{param_name}': Could not check context type due to missing definitions.")
                    pass # Assume not context if definitions are missing

            # <<< MODIFIED: Skip adding to pydantic_fields if it's a context type >>>
            if is_context_type:
                logger.debug(f"Skill '{name}', Param '{param_name}': Skipping context parameter of type '{actual_type.__name__}' from Pydantic schema.")
                continue # Skip to the next parameter
            # <<< END MODIFICATION >>>

            # --- Construct Pydantic Field ---
            field_args = {
                'description': details.get('description', ''),
                 # Include default only if one was provided and it's not 'empty sentinel'
                **({'default': default_value} if default_value is not inspect.Parameter.empty else {})
            }
            # If optional and no default, use Pydantic's Optional handling
            if is_optional and default_value is inspect.Parameter.empty:
                 final_type = Optional[actual_type]
                 # Pydantic handles Optional[...] implicitly having a default of None
                 pydantic_field = Field(**field_args)
            elif default_value is not inspect.Parameter.empty:
                 final_type = actual_type # Type hint remains the base type
                 # Pydantic infers optionality from the presence of a default value
                 pydantic_field = Field(**field_args)
            else:
                 # Required field (not optional, no default)
                 final_type = actual_type
                 pydantic_field = Field(...) # Ellipsis indicates required field

            pydantic_fields[param_name] = (final_type, pydantic_field)

        # Dynamically create the Pydantic model for input validation
        DynamicInputModel = None
        model_name = f"{func.__name__.capitalize()}InputModel"
        if pydantic_fields:
            try:
                DynamicInputModel = create_model(
                    model_name,
                    **pydantic_fields,
                    # <<< ADDED arbitrary_types_allowed=True >>>
                    __config__=ConfigDict(extra='forbid', arbitrary_types_allowed=True) 
                )
                PYDANTIC_MODEL_REGISTRY[name] = DynamicInputModel
                logger.debug(f"Skill '{name}': Created Pydantic input model '{model_name}'.")
            except (PydanticUserError, TypeError, NameError, Exception) as model_err:
                logger.error(f"Skill '{name}': Failed to create Pydantic model '{model_name}'. Input validation may not work correctly. Error: {model_err}")
                DynamicInputModel = None # Ensure it's None if creation failed
        else:
            logger.debug(f"Skill '{name}': No parameters requiring validation. Skipping Pydantic model creation.")


        # Store skill information in the registry
        SKILL_REGISTRY[name] = {
            'function': func,
            'async': asyncio.iscoroutinefunction(func),
            'description': description,
            'parameters': processed_params, # Store the processed parameters
            'input_schema': DynamicInputModel, # Store the Pydantic model (or None)
            'signature': sig # Store original signature for reference
        }

        logger.info(f"Skill '{name}' registered successfully.")
        return func # Return the original function

    return decorator


# --- Skill Discovery ---

def discover_skills(skill_directory: Union[str, Path] = "a3x/skills") -> None:
    """
    Dynamically discovers and imports skills from subdirectories.

    Assumes skills are defined in Python files within subdirectories
    (e.g., a3x/skills/core/, a3x/skills/file_management/).
    It imports the modules to trigger the @skill decorators.

    Args:
        skill_directory: The root directory containing skill subdirectories.
                         Defaults to "a3x/skills".
    """
    root_dir = Path(skill_directory).resolve()
    if not root_dir.is_dir():
        logger.error(f"Skill directory not found: {root_dir}")
        return
            
    logger.info(f"Starting skill discovery in: {root_dir}")
    # Ensure the root skills directory is treated as a package (might need __init__.py)
    # sys.path.insert(0, str(root_dir.parent)) # Add parent to path if needed

    # <<< Ensure parent directory of skill_directory is in path for correct prefix >>>
    module_prefix = root_dir.name + "."
    package_path = str(root_dir.parent)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
        logger.debug(f"Added {package_path} to sys.path for skill discovery.")
        path_added = True
    else:
        path_added = False
        logger.debug(f"{package_path} already in sys.path.")

    # Iterate through subdirectories and files
    # Use root_dir directly as path, prefix handles the 'a3x.skills.' part
    for module_info in pkgutil.walk_packages(path=[str(root_dir)], prefix=module_prefix):
        if not module_info.ispkg: # Only import modules, not packages themselves initially
            # <<< Construct full module name correctly >>>
            # Example: prefix='a3x.skills.', module_info.name='core.basic_math'
            # Needs to import 'a3x.skills.core.basic_math'
            full_module_name = f"a3x.skills.{module_info.name.split('.', 1)[-1]}" if '.' in module_info.name else f"a3x.skills.{module_info.name}" # Heuristic adjustment
            # Alternative: full_module_name = module_info.name # If prefix already includes the top package?
            # Let's stick to the prefix adding the base 'a3x.skills.' if needed
            # Example: root_dir = /path/to/A3X/a3x/skills -> prefix = 'skills.' -> module_info.name = 'core.basic_math'
            # Need 'a3x.skills.core.basic_math'
            # Let's try prefixing with 'a3x.'
            full_module_name = f"a3x.{module_info.name}"

            try:
                 logger.debug(f"Attempting to import skill module: {full_module_name}")
                 importlib.import_module(full_module_name)
                 logger.debug(f"Successfully imported {full_module_name}")
            except ModuleNotFoundError:
                 # Try original name from pkgutil if the adjusted one fails
                 original_module_name = module_info.name
                 logger.warning(f"Import failed for adjusted name '{full_module_name}', trying original name '{original_module_name}'...")
                 try:
                      importlib.import_module(original_module_name)
                      logger.debug(f"Successfully imported original name {original_module_name}")
                 except Exception as inner_e:
                      logger.error(f"Failed to import skill module using original name {original_module_name} either: {inner_e}")
            except ImportError as e:
                 logger.error(f"Failed to import skill module {full_module_name}: {e}")
            except Exception as e:
                 logger.error(f"Error importing skill module {full_module_name}: {e}", exc_info=True)

    # Clean up path modification if done
    if path_added:
         try:
             sys.path.remove(package_path)
             logger.debug(f"Removed {package_path} from sys.path after skill discovery.")
         except ValueError:
             pass # Path might have been removed elsewhere

    logger.info(f"Skill discovery finished. Total skills registered: {len(SKILL_REGISTRY)}")

# Example usage (Optional: Could be called during application startup)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     # Assume skills are in a directory structure like 'a3x/skills/core', 'a3x/skills/web'
#     # The path should be relative to where this script is run or absolute.
#     # If running from project root, 'a3x/skills' should work if 'a3x' is the package name.
#     script_dir = Path(__file__).parent.parent.resolve() # Go up two levels from core to project root potentially
#     skills_path = script_dir / 'skills'
#     discover_skills(skills_path)
#     print("\n--- Registered Skills ---")
#     print(get_skill_descriptions(include_params=True))

#     # Example of getting a specific skill's schema
#     # math_skill = get_skill("basic_math")
#     # if math_skill and math_skill['input_schema']:
#     #     print("\n--- Basic Math Input Schema ---")
#     #     print(math_skill['input_schema'].model_json_schema())
