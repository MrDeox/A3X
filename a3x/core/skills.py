# core/tools.py
import os
import sys
import logging
import importlib
import inspect
import pkgutil  # <<< ADDED IMPORT >>>
from typing import Dict, Any, Callable, Optional, Union, Type, List, get_origin, get_args # Added List, get_origin, get_args
from pydantic import BaseModel, create_model, ConfigDict # <<< ADDED ConfigDict >>>
from collections import namedtuple
from pathlib import Path
from pydantic import PydanticUserError # Import the specific error

# Initialize logger before potentially using it in print statements
logger = logging.getLogger(__name__)

# <<< ADDED: Base class for Context-Aware Skills >>>
class ContextAwareSkill:
    """Base class for skills that need access to a specific execution context ID."""
    def __init__(self, context_id: str):
        self._context_id = context_id

    @property
    def context_id(self) -> str:
        return self._context_id
# <<< END ADDED >>>

# <<< ADDED: Define SkillContext centrally >>>
SkillContext = namedtuple(
    "SkillContext",
    ["logger", "llm_call", "is_test", "workspace_root", "task"]
)

# <<< LOG HERE >>>
print(f"\n*** Executing module: a3x.core.skills (ID: {id(sys.modules.get('a3x.core.skills'))}) ***\n")

# <<< START PROJECT ROOT PATH CORRECTION >>>
# Calculate the project root (two levels up from core/tools.py)
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_current_file_dir)  # /home/arthur/Projects/A3X

# Add project root to sys.path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Clean up temporary variables
del _current_file_dir
# Keep 'project_root' as it might be useful elsewhere in this module, like in load_skills
# <<< END PROJECT ROOT PATH CORRECTION >>>

# <<< IMPORT ToolRegistry for type hint >>>
from .tool_registry import ToolRegistry, ToolInfo
# <<< IMPORT the singleton instance >>>
from .registry_instance import SKILL_REGISTRY
# <<< ADDED IMPORT for BaseFragment needed by model_rebuild >>>
from a3x.fragments.base import BaseFragment

# <<< REMOVED: Instantiation moved to registry_instance.py >>>
# SKILL_REGISTRY = ToolRegistry()

# --- Novo Registro de Skills e Decorador ---

class SkillInputSchema(BaseModel):
    """Classe base para schemas de input de skill, agora com config.

    Permite tipos arbitrários (como Context) para serem usados nos parâmetros das skills.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)


def skill(name: str, description: str, parameters: Dict[str, Dict[str, Any]]):
    """
    Decorator to register a function as a skill available to the agent.

    Args:
        name: The unique name of the skill (used by the LLM).
        description: A clear description of what the skill does.
        parameters: A dictionary where keys are parameter names and values are
                    dictionaries containing 'type', 'description', and optionally 'default'.
                    Example:
                    {
                        "path": {"type": str, "description": "Path to the file"}, # Required
                        "content": {"type": str, "default": "", "description": "Content to write"} # Optional
                    }
    """

    def decorator(func: Callable):
        logger.debug(f"Registering skill: {name}")

        # <<< ADDED IMPORT INSIDE DECORATOR >>>
        from typing import Union, get_args, get_origin

        # Validate function signature against declared parameters
        sig = inspect.signature(func)
        func_params = sig.parameters

        # <<< CORRECTED AGAIN: Handle both direct param dict and JSON schema structure >>>
        if "properties" in parameters and isinstance(parameters.get("properties"), dict):
            # Assume JSON Schema structure
            declared_param_names = set(parameters["properties"].keys())
        else:
            # Assume direct parameter dictionary structure
            declared_param_names = set(parameters.keys())
        # <<< END CORRECTION >>>

        func_param_names = set(func_params.keys())

        # Define parameters to ignore during signature validation
        ignored_params = {
            "self",
            "agent_memory",
            "agent_history",
            "resolved_path",
            "original_path_str",
            "kwargs",
            "ctx",
        }

        func_params_to_validate = func_param_names - ignored_params
        if not declared_param_names.issubset(func_params_to_validate):
            missing_in_func = declared_param_names - func_params_to_validate
            raise TypeError(
                f"Skill '{name}': Parameters {missing_in_func} declared in decorator but not found in function signature (excluding {ignored_params})."
            )

        extra_in_func = func_params_to_validate - declared_param_names
        if extra_in_func:
            # Allow the first parameter (usually context) implicitly
            first_param_name = next(iter(sig.parameters)) if sig.parameters else None
            extra_in_func_filtered = {p for p in extra_in_func if p != first_param_name}

            if extra_in_func_filtered: # Check if there are still unexpected params after excluding context
                # Allow extra function params ONLY if they have defaults
                has_defaults = all(func_params[p].default != inspect.Parameter.empty for p in extra_in_func_filtered)
                if not has_defaults:
                    params_without_defaults = {p for p in extra_in_func_filtered if func_params[p].default == inspect.Parameter.empty}
                    raise TypeError(
                        f"Skill '{name}': Parameters {params_without_defaults} found in function signature without default values but not declared in decorator (excluding {ignored_params} and first param '{first_param_name}')."
                    )
                else:
                    logger.debug(f"Skill '{name}': Extra parameters {extra_in_func_filtered} found in function signature with defaults but not in decorator. Allowed.")


        # Create Pydantic schema dynamically
        pydantic_fields: Dict[str, tuple[Type[Any], Any]] = {} # Type hint for clarity
        # param_details_for_registry = {} # No longer needed for separate storage

        # <<< REVISED SCHEMA GENERATION LOGIC (handle direct dict vs. schema object) >>>
        params_to_process = {}
        if "properties" in parameters and isinstance(parameters.get("properties"), dict):
            # JSON Schema structure
            params_to_process = parameters["properties"]
        else:
            # Direct parameter dictionary structure
            params_to_process = parameters

        for param_name, param_config in params_to_process.items(): # Iterate over actual params
            # <<< ADD CHECK TO SKIP CONTEXT PARAMS >>>
            if param_name in ['ctx', 'context']:
                logger.debug(f"Skipping context parameter '{param_name}' for Pydantic model generation in skill '{name}'.")
                continue
            # <<< END CHECK >>>
            
            if not isinstance(param_config, dict) or "type" not in param_config or "description" not in param_config:
                raise TypeError(f"Skill '{name}': Invalid parameter config for '{param_name}'. Expected dict with 'type' and 'description'.")

            param_type = param_config["type"]
            param_desc = param_config["description"]
            is_optional = param_config.get("optional", False)
            # Use provided default if exists, otherwise sentinel (...) if required, or None if optional
            param_default_value = param_config.get("default", ...) if not is_optional else param_config.get("default", None)

            # Store details for the registry
            # param_details_for_registry[param_name] = {
            #      "type_obj": param_type,
            #      "type_str": str(param_type.__name__) if hasattr(param_type, '__name__') else str(param_type),
            #      "required": param_default_value is ... and not is_optional,
            #      "default": None if (param_default_value is ...) else param_default_value, # Store None if required, else the default
            #      "description": param_desc,
            #      "optional_flag": is_optional # Store the explicit optional flag
            # }

            # --- Prepare for Pydantic model --- # <-- MODIFIED BLOCK
            pydantic_type = param_type
            pydantic_default = param_default_value

            # If the parameter is explicitly marked as optional, ensure Pydantic type is Optional
            # and the default is None (unless a different default was provided)
            if is_optional:
                # Check if the type is already Optional
                origin = getattr(param_type, '__origin__', None)
                if origin is not Optional and origin is not Union:
                     try:
                        # Attempt to wrap the type in Optional
                        # Handle Union types as well, check if None is already part of it
                        if get_origin(param_type) is Union:
                             args = get_args(param_type)
                             if type(None) not in args:
                                 pydantic_type = Union[param_type, type(None)]
                             else:
                                 pydantic_type = param_type # Already includes None
                        else:
                             # Simple type, wrap in Optional
                             pydantic_type = Optional[param_type]

                     except TypeError as e:
                         logger.warning(f"Skill '{name}': Could not wrap type '{param_type}' in Optional for param '{param_name}'. Type might not support it (e.g., complex alias). Error: {e}. Proceeding with original type.")
                         pydantic_type = param_type # Fallback to original type

                # Set default to None if no other default was specified for an optional field
                if pydantic_default is ...:
                    pydantic_default = None

            pydantic_fields[param_name] = (pydantic_type, pydantic_default)
            # --- End Pydantic Preparation --- #

        # <<< END REVISED SCHEMA GENERATION LOGIC >>>

        # Dynamic schema name
        schema_name = f"{name.capitalize()}SkillSchema"
        dynamic_schema: Optional[Type[BaseModel]] = None
        json_schema_properties = {}
        json_schema = {}
        try:
            dynamic_schema = create_model(
                schema_name,
                __base__=SkillInputSchema,
                __module__=func.__module__,
                **pydantic_fields
            )
            
            # <<< ADDED: Attempt to rebuild the model to resolve forward refs >>>
            try:
                dynamic_schema.model_rebuild(force=True) # Force rebuild
                logger.debug(f"Successfully rebuilt Pydantic model for skill '{name}'.")
            except PydanticUserError as rebuild_err:
                # Log specific Pydantic errors during rebuild but try to continue
                logger.warning(f"PydanticUserError during model_rebuild for skill '{name}': {rebuild_err}. Schema generation might be incomplete.")
            except Exception as rebuild_generic_err:
                # Catch other potential errors during rebuild
                logger.error(f"Unexpected error during model_rebuild for skill '{name}': {rebuild_generic_err}", exc_info=True)
                # Decide if we should raise here or try to continue
            
            # <<< Generate JSON schema AFTER rebuild attempt >>>
            json_schema = dynamic_schema.model_json_schema()
            json_schema_properties = json_schema.get('properties', {})

        except Exception as e:
            # This catches errors during create_model primarily
            logger.error(f"Failed to create Pydantic/JSON schema for skill '{name}'. Error: {e}")
            raise TypeError(f"Invalid Pydantic schema definition for skill '{name}'.") from e

        # <<< MODIFIED: Use SKILL_REGISTRY.register_tool with explicit dict >>>
        instance = None # Assume function
        # TODO: Add logic to detect if func belongs to a class

        takes_context = False
        if func_params:
            first_param_name = next(iter(func_params))
            first_param = func_params[first_param_name]
            if first_param_name in ['ctx', 'context'] or first_param.annotation is SkillContext:
                 takes_context = True
                 logger.debug(f"Skill '{name}' identified as context-aware.")

        # Create the schema dictionary expected by ToolRegistry.register_tool
        tool_schema_for_registry = {
            "name": name,
            "description": description,
            "parameters": json_schema_properties # Pass only the properties
            # Add other top-level schema info if needed, e.g., required fields:
            # "required": list(dynamic_schema.model_fields_set) if dynamic_schema else [] 
            # Or get required fields from the generated json_schema dict if preferred
        }
        # Add optional 'required' list from JSON schema if available
        if dynamic_schema and 'required' in json_schema:
             tool_schema_for_registry['required'] = json_schema['required']

        # Register using the correct method and the structured dictionary
        SKILL_REGISTRY.register_tool(
            name=name, # Pass name separately for potential validation/lookup
            instance=instance,
            tool=func,
            schema=tool_schema_for_registry # Pass the structured dict
        )
        # <<< END MODIFIED REGISTRATION >>>

        logger.info(f"Successfully registered skill: {name}")
        func._skill_name = name
        func._skill_description = description
        return func

    return decorator


# <<< RENOMEADO: Função para carregar um único pacote de skills >>>
def _load_single_skill_package(skill_package_name: str):
    """Loads skills from all modules within a specified package."""
    try:
        if not all(part.isidentifier() for part in skill_package_name.split('.')):
            logger.error(f"Invalid skill package name: {skill_package_name}")
            return

        count_before = len(SKILL_REGISTRY.list_tools())

        spec = importlib.util.find_spec(skill_package_name)

        # <<< REMOVED RELOAD LOGIC - Only import >>>
        if spec is None:
            logger.warning(f"Could not find skill package '{skill_package_name}'. Skipping.")
            return
            
        # Check if already imported (simplest way to avoid re-running module code)
        if skill_package_name in sys.modules:
             logger.info(f"Skills package '{skill_package_name}' already imported. Skipping module re-execution.")
             # NOTE: This assumes skills register on first import via @skill decorator.
             # If re-registration or updates are needed, a different mechanism
             # than simple import/reload is required.
             return # Skip walking and importing modules again
        else:
            logger.info(f"Importing skills package for the first time: {skill_package_name}")
            # Try importing the top-level package itself first
            try:
                 module = importlib.import_module(skill_package_name)
                 # Now iterate through its modules if it's a package
                 if spec.submodule_search_locations:
                     package_path = spec.submodule_search_locations[0]
                     logger.debug(f"Walking package: {skill_package_name} at {package_path}")
                     for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
                         full_module_name = f"{skill_package_name}.{module_name}"
                         if is_pkg:
                              # Optionally recurse into sub-packages
                              logger.debug(f"Found sub-package '{full_module_name}', loading recursively...")
                              _load_single_skill_package(full_module_name) # Recursive call
                         else:
                             if full_module_name not in sys.modules:
                                 try:
                                     logger.debug(f"---> Importing submodule: {full_module_name}")
                                     importlib.import_module(full_module_name)
                                     logger.debug(f"<--- Imported submodule: {full_module_name}")
                                 except ImportError as e:
                                     logger.error(f"Failed to import skill submodule '{full_module_name}': {e}", exc_info=True)
                                 except Exception as e:
                                     logger.error(f"Error loading submodule '{full_module_name}': {e}", exc_info=True)
                             else:
                                 logger.debug(f"Submodule '{full_module_name}' already imported. Skipping.")
                 else:
                     logger.debug(f"'{skill_package_name}' is a single module, not a package. Skills loaded (if any)." )

            except ImportError as e:
                logger.error(f"Failed to import skill package/module '{skill_package_name}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error during skill loading process for '{skill_package_name}':", exc_info=True)

        # <<< END REMOVED RELOAD LOGIC >>>

        count_after = len(SKILL_REGISTRY.list_tools())
        newly_registered = count_after - count_before
        logger.info(f"Finished processing package '{skill_package_name}'. Newly registered: {newly_registered}. Total in registry: {count_after}")

    except Exception as e:
        logger.error(f"Unexpected error in _load_single_skill_package for '{skill_package_name}': {e}", exc_info=True)

# <<< NOVA FUNÇÃO: Para carregar skills de MÚLTIPLOS pacotes >>>
def load_all_skills(skill_package_list: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Loads or reloads skills dynamically from a LIST of Python packages.
    Calls _load_single_skill_package for each package in the list.

    Args:
        skill_package_list (list[str]): A list of Python package names (e.g., ['a3x.skills.core', 'a3x.skills.web']).

    Returns:
        Dict[str, Dict[str, Any]]: The final skill registry after loading all packages.
    """
    logger.info(f"Loading skills from multiple packages: {skill_package_list}")
    initial_count = len(SKILL_REGISTRY.list_tools())

    for package_name in skill_package_list:
        logger.debug(f"--->>> About to load package: {package_name}") # Log antes
        _load_single_skill_package(package_name)
        logger.debug(f"--->>> Finished loading package: {package_name}") # Log depois

    final_count = len(SKILL_REGISTRY.list_tools())
    total_new = final_count - initial_count
    logger.info(f"Finished loading all skill packages. Total newly registered: {total_new}. Final registry size: {final_count}")
    logger.debug(f"--->>> Returning final SKILL_REGISTRY from load_all_skills (size: {final_count})") # Log antes de retornar
    return SKILL_REGISTRY


# --- Funções para acessar informações das Skills ---


def get_skill_registry() -> ToolRegistry:
    """Returns the global skill registry instance."""
    # Returns the imported singleton instance
    return SKILL_REGISTRY


def get_skill_descriptions() -> str:
    """
    Retorna uma string formatada descrevendo todas as skills registradas,
    incluindo seus parâmetros e suas descrições do docstring.
    """
    tools = SKILL_REGISTRY.list_tools() # Use the imported instance

    if not tools:
        return "No skills are currently registered."

    descriptions = []
    for name, skill_info in SKILL_REGISTRY.items():
        func = skill_info.get("function")
        desc = skill_info.get("description", "No description.")
        params_dict = skill_info.get("parameters", {})

        # --- Extract Parameter Descriptions from Docstring --- START
        param_docs = {}
        if func and func.__doc__:
            docstring = inspect.getdoc(func)
            if docstring:
                lines = docstring.split('\n')
                in_args_section = False
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.lower().startswith("args:"):
                        in_args_section = True
                        continue
                    if in_args_section:
                        if not stripped_line: # End of Args section
                            break
                        if ":" in stripped_line:
                            # Simple parse: Assume format 'param_name (type): Description.'
                            # or 'param_name: Description.' (if type is omitted in docstring)
                            parts = stripped_line.split(':', 1)
                            param_part = parts[0].strip()
                            param_desc_text = parts[1].strip()
                            # Extract just the name (handle potential type hints in docstring like 'param (str)')
                            param_name_from_doc = param_part.split('(')[0].strip()
                            param_docs[param_name_from_doc] = param_desc_text
        # --- Extract Parameter Descriptions from Docstring --- END

        # Obter docstring como descrição alternativa para a *skill* (se não houver no decorator)
        if desc == "No description." and func and func.__doc__:
            desc = inspect.getdoc(func).split('\n')[0] if inspect.getdoc(func) else "No description."

        param_strs = []
        if params_dict:
            for param_name, param_info in params_dict.items():
                param_type_str = "Any"
                default_desc = ""
                is_required = False

                if isinstance(param_info, (list, tuple)):
                    if len(param_info) == 1:
                        param_type = param_info[0]
                        is_required = True
                    elif len(param_info) == 2:
                        param_type, default_value = param_info
                        is_required = (default_value is inspect.Parameter.empty)
                        default_desc = f" (default: {default_value})" if not is_required else ""
                    else:
                        continue

                    if hasattr(param_type, '__name__'):
                        param_type_str = param_type.__name__
                    else:
                        param_type_str = str(param_type)
                else:
                    continue

                required_str = " (required)" if is_required else default_desc
                # <<< Include Parameter Description >>>
                param_desc = param_docs.get(param_name, "") # Get extracted description
                desc_str = f" - {param_desc}" if param_desc else ""
                param_strs.append(f"{param_name}: {param_type_str}{required_str}{desc_str}")

        param_section = ", ".join(param_strs)
        if param_section:
            descriptions.append(f"- {name}({param_section}): {desc}")
        else:
            descriptions.append(f"- {name}(): {desc}")

    return "\n".join(descriptions)


def get_skill(skill_name: str) -> Optional[ToolInfo]:
    """
    Retrieves the details (ToolInfo) for a specific skill by name.
    """
    try:
        # Use the imported instance
        return SKILL_REGISTRY.get_tool(skill_name)
    except KeyError:
        logger.warning(
            f"Skill '{skill_name}' not found in registry even after attempting load."
        )
        return None


# É crucial chamar load_skills() no ponto de entrada principal da aplicação
# (provavelmente na inicialização do Agent) para garantir que todas as
# skills sejam registradas antes de serem necessárias.
# Exemplo: remover a chamada automática daqui
