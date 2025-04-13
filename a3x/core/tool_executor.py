import json
import logging
import inspect
import os
import sys
from typing import Dict, Any, Optional, Callable, List, Tuple, Awaitable, Coroutine
import importlib
import pkgutil
from pathlib import Path
from collections import namedtuple
import asyncio

# Import SKILL_REGISTRY and PROJECT_ROOT for instantiation and context
from a3x.core.skill_management import SKILL_REGISTRY
from a3x.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Define a context structure similar to the CLI for consistency
# We might not need all fields here, but workspace_root and logger are key.
_ToolExecutionContext = namedtuple("ToolExecutionContext", ["logger", "workspace_root", "llm_url", "tools_dict"])


# Modified Signature: Accept workspace_root and context elements instead of just logger/memory
async def execute_tool(
    tool_name: str,
    action_input: Dict[str, Any],
    tools_dict: dict,
    context: _ToolExecutionContext
) -> Dict[str, Any]:
    """Executa a ferramenta/skill selecionada, handling both functions and class methods correctly."""
    log_prefix = "[Tool Executor]"
    ctx_logger = context.logger
    ctx_logger.info(
        f"{log_prefix} Attempting tool: '{tool_name}', Input: {action_input}"
    )

    # --- Validation ---
    if tool_name not in tools_dict:
        available_tools = list(tools_dict.keys())
        error_message = (
            f"Tool '{tool_name}' does not exist. Available: {available_tools}"
        )
        ctx_logger.error(f"{log_prefix} {error_message}")
        return {
            "status": "error",
            "action": "tool_not_found",
            "data": {"message": error_message},
        }
    # --- End Validation ---

    ctx_logger.debug(
        f"{log_prefix} Tool '{tool_name}' found. Preparing for execution."
    )
    skill_info = SKILL_REGISTRY.get(tool_name)
    if not skill_info:
        ctx_logger.error(f"{log_prefix} Skill '{tool_name}' not found in live registry despite being in tools_dict snapshot.")
        return {"status": "error", "action": "tool_not_found_in_registry", "data": {"message": f"Skill '{tool_name}' disappeared from registry."}}

    func_obj = skill_info.get("function")
    if not func_obj or not callable(func_obj):
        ctx_logger.error(f"{log_prefix} Skill '{tool_name}' found but has no callable function object in registry.")
        return {"status": "error", "action": "skill_not_callable", "data": {"message": f"Skill '{tool_name}' is improperly registered."}}

    ctx_logger.debug(f"{log_prefix} Retrieved skill function object '{func_obj.__qualname__}'.")

    executable_callable: Callable = None
    instance: Optional[Any] = None
    result = None

    try:
        # --- Detect Type and Prepare Callable ---
        is_method = inspect.isfunction(func_obj) and '.' in func_obj.__qualname__

        if is_method:
            ctx_logger.debug(f"{log_prefix} Skill '{tool_name}' detected as a method: {func_obj.__qualname__}")
            qname = func_obj.__qualname__
            class_name = qname.split('.')[-2]
            method_name = func_obj.__name__

            module = inspect.getmodule(func_obj)
            if not module:
                raise RuntimeError(f"Could not determine the module for {qname}")

            ctx_logger.debug(f"{log_prefix} Attempting to get class '{class_name}' from module '{module.__name__}'")
            SkillClass = getattr(module, class_name)

            ctx_logger.info(f"{log_prefix} Instantiating class '{class_name}' for skill '{tool_name}'.")
            # Inspect the __init__ method of the class
            init_sig = inspect.signature(SkillClass.__init__)
            init_params = init_sig.parameters

            # Prepare arguments for instantiation
            init_args = {}
            if 'workspace_root' in init_params:
                ctx_logger.debug(f"{log_prefix} Passing workspace_root ('{context.workspace_root}') to {class_name}.__init__")
                init_args['workspace_root'] = context.workspace_root
            # Add other potential context arguments needed by __init__ here
            # e.g., if 'logger' in init_params: init_args['logger'] = context.logger

            # Instantiate the class
            # <<< START MODIFICATION: Handle potential __init__ errors >>>
            try:
                instance = SkillClass(**init_args)
            except TypeError as init_err:
                ctx_logger.error(f"{log_prefix} Failed to instantiate '{class_name}' for skill '{tool_name}'. Mismatched __init__ arguments? Error: {init_err}")
                return {"status": "error", "action": f"{tool_name}_instantiation_error", "data": {"message": f"Failed to instantiate skill class: {init_err}"}}
            except Exception as init_gen_err:
                 ctx_logger.error(f"{log_prefix} Failed to instantiate '{class_name}' for skill '{tool_name}'. Error during __init__: {init_gen_err}")
                 return {"status": "error", "action": f"{tool_name}_instantiation_error", "data": {"message": f"Error during skill class initialization: {init_gen_err}"}}
            # <<< END MODIFICATION >>>

            executable_callable = getattr(instance, method_name)
            ctx_logger.debug(f"{log_prefix} Prepared callable: method '{method_name}' of instance {instance}")
        else:
            # It's a regular function
            ctx_logger.debug(f"{log_prefix} Skill '{tool_name}' detected as a regular function: {func_obj.__qualname__}")
            executable_callable = func_obj

        # --- Prepare Arguments for Execution ---
        final_args: List[Any] = []  # For positional args (less common for skills)
        call_args: Dict[str, Any] = {} # For keyword args

        # Get the signature of the target callable (function or method)
        target_sig = inspect.signature(executable_callable)
        target_params = target_sig.parameters

        # --- Argument Validation & Preparation ---
        validated_input = {}
        schema = skill_info.get("schema")
        if schema:
            # *** ADDED: Handle parameter name mismatch for introspect ***
            input_to_validate = action_input.copy() # Work on a copy
            if tool_name == 'introspect' and 'query' in input_to_validate and 'question' not in input_to_validate:
                 ctx_logger.warning(f"{log_prefix} Mapping 'query' parameter to 'question' for introspect skill.")
                 input_to_validate['question'] = input_to_validate.pop('query')
            # *** END ADDED SECTION ***

            try:
                # Validate potentially modified input against Pydantic schema
                validated_data = schema(**input_to_validate)
                validated_input = validated_data.model_dump() # Convert back to dict
                ctx_logger.debug(f"{log_prefix} Action input successfully validated by Pydantic schema for '{tool_name}'. Validated: {validated_input}")
            except Exception as pydantic_err: # Catch Pydantic validation errors
                ctx_logger.error(f"{log_prefix} Pydantic validation failed for '{tool_name}'. Input: {action_input}. Error: {pydantic_err}")
                return {
                    "status": "error",
                    "action": f"{tool_name}_validation_error",
                    "data": {"message": f"Invalid input parameters: {pydantic_err}"},
                }
        else:
            # No schema, use raw input but log a warning
            ctx_logger.warning(f"{log_prefix} No Pydantic schema found for skill '{tool_name}'. Using raw input: {action_input}")
            validated_input = action_input # Use raw input

        # --- Populate call_args based on validated input and function signature ---
        for param_name in target_params:
            if param_name == 'self': # Skip 'self' for method calls
                continue

            # <<< ADDED: Skip decorator-injected args during this phase >>>
            if param_name in ["resolved_path", "original_path_str"]:
                 # These are expected to be injected by the @validate_workspace_path decorator
                 # Do not attempt to populate them from action_input here.
                 ctx_logger.debug(f"Skipping population of decorator-injected arg: {param_name}")
                 continue
            # <<< END ADDED >>>

            # Handle special context arguments
            if param_name == 'workspace_root':
                 call_args[param_name] = context.workspace_root
                 continue
            if param_name == 'logger':
                 call_args[param_name] = context.logger
                 continue
            # <<< ADDED: Handle 'ctx' parameter >>>
            if param_name == 'ctx':
                 call_args[param_name] = context # Pass the whole namedtuple context
                 continue

            # Check if the parameter exists in the *validated* input
            if param_name in validated_input:
                call_args[param_name] = validated_input[param_name]
            # If not in input, check if it has a default value in the signature
            elif target_params[param_name].default != inspect.Parameter.empty:
                # Parameter has a default value, no need to provide it explicitly
                pass # Python handles default values automatically
            # --- REMOVED: Explicit check for decorator default - Relies on schema/signature --- #
            # If it's required but missing from validated input (shouldn't happen if schema validation passed)
            else:
                 # This case should ideally be caught by Pydantic validation if schema is correct
                 ctx_logger.error(f"{log_prefix} Required parameter '{param_name}' for skill '{tool_name}' missing from validated input and has no default value.")
                 return {
                     "status": "error",
                     "action": f"{tool_name}_missing_argument",
                     "data": {"message": f"Required argument '{param_name}' missing."},
                 }

        # --- Execute the Callable ---
        ctx_logger.debug(f"{log_prefix} Executing '{executable_callable.__qualname__}' with validated args: {call_args}")
        if inspect.iscoroutinefunction(executable_callable):
            ctx_logger.debug(f"{log_prefix} Awaiting async skill '{tool_name}'")
            result = await executable_callable(*final_args, **call_args)
        else:
            ctx_logger.debug(f"{log_prefix} Running sync skill '{tool_name}' in executor")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: executable_callable(*final_args, **call_args))

        # --- Process Result ---
        # <<< MODIFIED: Allow non-dict results for simpler skills, but wrap them >>>
        if not isinstance(result, dict):
            ctx_logger.warning(
                f"{log_prefix} Skill '{tool_name}' returned non-dict type: {type(result)}. Wrapping in standard success dict."
            )
            # Wrap simple return values (like strings from introspect) into the standard format
            return {
                "status": "success",
                "action": f"{tool_name}_completed",
                "data": {"result": result} # Store the raw result here
            }
        # <<< END MODIFICATION >>>

        # If it's already a dict, assume it follows the standard format
        ctx_logger.info(
            f"{log_prefix} Skill '{tool_name}' executed. Status: {result.get('status', 'N/A')}"
        )
        ctx_logger.debug(f"{log_prefix} Skill '{tool_name}' result: {result}")

        return result # Return the dictionary as-is

    except TypeError as e:
        ctx_logger.exception(
            f"{log_prefix} TypeError executing skill '{tool_name}': {e}"
        )
        passed_args_str = f"Positional: {len(final_args)}, Keywords: {list(call_args.keys())}"
        return {
            "status": "error",
            "action": f"{tool_name}_argument_error",
            "data": {
                "message": f"Argument mismatch calling skill '{tool_name}': {str(e)}. Passed args: {passed_args_str}"
            },
        }
    except Exception as e:
        ctx_logger.exception(f"{log_prefix} Error executing skill '{tool_name}':")
        return {
            "status": "error",
            "action": f"{tool_name}_failed",
            "data": {
                "message": f"Internal error executing skill '{tool_name}': {str(e)}"
            },
        }

# Helper to parse docstrings (basic Google style)
def _parse_docstring_args(docstring: Optional[str]) -> Dict[str, str]:
    if not docstring:
        return {}

    param_docs = {}
    lines = docstring.split('\n')
    in_args_section = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.lower().startswith("args:"):
            in_args_section = True
            continue
        if in_args_section:
            if not stripped_line:
                break
            if ":" in stripped_line:
                parts = stripped_line.split(':', 1)
                param_part = parts[0].strip()
                param_desc_text = parts[1].strip()
                param_name_from_doc = param_part.split('(')[0].strip()
                param_docs[param_name_from_doc] = param_desc_text
    return param_docs
