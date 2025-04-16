import json
import logging
import inspect
import os
import sys
import time
from typing import Dict, Any, Optional, Callable, List, Tuple, Awaitable, Coroutine, NamedTuple, Union
import importlib
import pkgutil
from pathlib import Path
from collections import namedtuple
import asyncio
import re
import uuid
from a3x.fragments.registry import FragmentRegistry
from .context import SharedTaskContext, _ToolExecutionContext, FragmentContext

# Import SKILL_REGISTRY and PROJECT_ROOT for instantiation and context
from a3x.core.skill_management import SKILL_REGISTRY
from a3x.core.config import PROJECT_ROOT

# --- Import State Update Hooks ---
from a3x.core.hooks.state_updates import (
    notify_tool_execution_start,
    notify_tool_execution_end
)
# --- End Hook Imports ---

logger = logging.getLogger(__name__)

# Modified Signature: Accept workspace_root and context elements instead of just logger/memory
async def execute_tool(
    tool_name: str,
    action_input: Dict[str, Any],
    tools_dict: 'ToolRegistry', # Expect ToolRegistry type
    context: Union[_ToolExecutionContext, FragmentContext] # <<< Accept either context type
) -> Dict[str, Any]:
    """Executa a ferramenta/skill, medindo tempo e status, e retorna resultado + métricas."""
    log_prefix = "[Tool Executor]"
    ctx_logger = context.logger
    ctx_logger.info(f"[DEBUG TEST] execute_tool called with tool_name={tool_name}, action_input={action_input}, tools_dict={tools_dict}, context={type(context).__name__}")
    start_time = time.time()
    result_payload: Dict[str, Any] = {}
    skill_status: str = "error"
    skill_message: str = "Execution did not start or failed before completion."
    # Initialize metrics with default error state
    metrics = {
        "duration_seconds": 0.0,
        "status": "error", 
        "message": "Tool execution did not proceed past initial validation."
    }

    ctx_logger.info(
        f"{log_prefix} Attempting tool: '{tool_name}', Input: {action_input}"
    )

    # Check if the tool exists using the appropriate method if tools_dict is a ToolRegistry
    available_tools = {}
    if hasattr(tools_dict, 'list_tools') and callable(tools_dict.list_tools):
        available_tools = tools_dict.list_tools() # Get the dictionary of available tools
    elif isinstance(tools_dict, dict):
        available_tools = tools_dict # Assume it's already a dictionary
    else:
        ctx_logger.error(f"{log_prefix} Invalid tools_dict type: {type(tools_dict)}")
        result_payload = {"status": "error", "action": "invalid_tool_registry_object", "data": {"message": "Invalid tool registry object provided."}}
        skill_message = "Invalid tool registry object provided."
        # Return metrics along with payload
        metrics["message"] = skill_message
        return {"result": result_payload, "metrics": metrics}

    if tool_name not in available_tools:
        error_message = f"Tool '{tool_name}' not found in the registry."
        ctx_logger.error(f"{log_prefix} {error_message}")
        result_payload = {"status": "error", "action": "tool_not_found_in_registry", "data": {"message": error_message}}
        skill_message = error_message
        # Return metrics along with payload
        metrics["message"] = skill_message
        return {"result": result_payload, "metrics": metrics}

    try:
        # Retrieve the instance and the bound tool method using the updated ToolRegistry
        tool_instance, tool_method = tools_dict.get_instance_and_tool(tool_name)

        if not tool_method or not callable(tool_method):
            error_message = f"Tool '{tool_name}' found but is not callable."
            ctx_logger.error(f"{log_prefix} {error_message}")
            result_payload = {"status": "error", "action": "tool_not_callable", "data": {"message": error_message}}
            skill_message = error_message
            return result_payload

        # --- MODIFIED: Context Handling --- 
        if isinstance(context, FragmentContext):
            # If we received FragmentContext, create _ToolExecutionContext
            ctx_logger.debug(f"{log_prefix} Received FragmentContext, creating _ToolExecutionContext.")
            # Need to get allowed_skills from somewhere if needed by _ToolExecutionContext
            # Placeholder: Get skills from the fragment instance if possible, else None
            # This assumes context might have a reference to the calling fragment or its skills
            # For now, setting allowed_skills to None as it's not directly available in FragmentContext
            # and the original ToolExecutionContext didn't seem to reliably have it populated either.
            effective_context = _ToolExecutionContext(
                context.logger,
                context.workspace_root,
                context.llm_interface.llm_url, # Assuming llm_interface has llm_url
                context.tool_registry,
                context.llm_interface,
                context.fragment_registry,
                context.shared_task_context,
                None, # allowed_skills - Placeholder - where should this come from?
                tool_instance, # skill_instance - Add the instance
                context.memory_manager # memory_manager <<< ADDED: Pass memory_manager from FragmentContext >>>
            )
        elif isinstance(context, _ToolExecutionContext):
            # If we already have _ToolExecutionContext, just update skill_instance
            ctx_logger.debug(f"{log_prefix} Received _ToolExecutionContext, replacing skill_instance.")
            try:
                # <<< ALSO ADD memory_manager update if needed? >>>
                # Assuming _ToolExecutionContext already has memory_manager if created elsewhere
                effective_context = context._replace(skill_instance=tool_instance)
            except ValueError as e:
                 # This should not happen now if check above works, but safety net
                 ctx_logger.error(f"{log_prefix} Error using _replace on _ToolExecutionContext: {e}. Context: {context}")
                 raise # Re-raise the error
        else:
            # Handle unexpected context type
            error_message = f"Received unexpected context type: {type(context).__name__}"
            ctx_logger.error(f"{log_prefix} {error_message}")
            result_payload = {"status": "error", "action": "invalid_context_type", "data": {"message": error_message}}
            skill_message = error_message
            # Need to wrap result here too
            metrics["message"] = skill_message
            return {"result": result_payload, "metrics": metrics}
        # --- END MODIFIED Context Handling ---

        # Check if it's an async function
        is_async = asyncio.iscoroutinefunction(tool_method)
        sig = inspect.signature(tool_method)
        params = sig.parameters # Get parameters from signature
        
        # --- REVISED Argument Passing Logic --- 
        call_kwargs = action_input.copy() # Start with action_input

        # Check if the skill expects a 'context' argument
        if 'context' in params:
            # >>> Pass the FINAL effective_context <<<
            call_kwargs['context'] = effective_context 
            ctx_logger.debug(f"Passing _ToolExecutionContext as 'context' argument to {tool_name}")
        # Check if the skill *specifically* expects 'ctx' (less common, but support for now)
        elif 'ctx' in params:
            # >>> Pass the FINAL effective_context <<<
            call_kwargs['ctx'] = effective_context 
            ctx_logger.debug(f"Passing _ToolExecutionContext as 'ctx' argument to {tool_name}")
        # else: skill expects neither 'context' nor 'ctx'
        
        # Remove parameters from call_kwargs that are not in the function signature 
        # to prevent "unexpected keyword argument" errors for skills that don't 
        # accept arbitrary **kwargs.
        final_call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
        removed_keys = set(call_kwargs.keys()) - set(final_call_kwargs.keys())
        if removed_keys:
            ctx_logger.warning(f"{log_prefix} Removed unexpected arguments for {tool_name}: {removed_keys}")
        
        # --- END REVISED Logic --- 

        ctx_logger.debug(f"{log_prefix} Preparing to call {'async' if is_async else 'sync'} tool '{tool_name}'. Final Kwargs: {list(final_call_kwargs.keys())}")

        # Execute the tool method
        try:
            start_time_exec = time.monotonic()
            if is_async:
                ctx_logger.debug(f"{log_prefix} Awaiting async tool '{tool_name}'")
                # >>> Use final_call_kwargs <<< 
                result = await tool_method(**final_call_kwargs)
            else:
                ctx_logger.debug(f"{log_prefix} Calling sync tool '{tool_name}'")
                # >>> Use final_call_kwargs <<< 
                result = tool_method(**final_call_kwargs)
            
            end_time_exec = time.monotonic()
            duration_exec = end_time_exec - start_time_exec
            ctx_logger.info(f"{log_prefix} Tool '{tool_name}' executed successfully in {duration_exec:.4f}s.")

            # Process result
            if isinstance(result, dict) and 'status' in result:
                skill_status = result.get('status', 'unknown')
                skill_message = result.get('message', 'No message provided.')
                result_payload = result
            else:
                ctx_logger.warning(f"{log_prefix} Tool '{tool_name}' executed but returned an unexpected format: {type(result)}. Wrapping as success.")
                skill_status = "success"
                skill_message = "Execution successful, non-standard return format."
                result_payload = {"status": skill_status, "data": result} # Wrap non-dict results

        except Exception as e:
            end_time_exec = time.monotonic()
            duration_exec = end_time_exec - start_time_exec
            ctx_logger.exception(f"{log_prefix} Error executing tool '{tool_name}': {e}")
            skill_status = "error"
            skill_message = f"Execution failed: {str(e)}"
            result_payload = {"status": skill_status, "message": skill_message}
            # Ensure metrics reflect the execution duration even on error
            metrics = {
                "duration_seconds": duration_exec,
                "status": skill_status,
                "message": skill_message[:500]
            }
            # Exit the outer try block and go to finally
            raise # Re-raise the exception to be caught by the outer handler if needed, or just let finally run


    except Exception as e:
        # This block catches errors during tool lookup or initial setup
        end_time = time.time() # Use time.time() for overall duration
        duration = end_time - start_time
        logger.exception(f"{log_prefix} Unexpected error during tool execution setup for '{tool_name}': {e}")
        skill_status = "error"
        skill_message = f"Tool setup failed: {str(e)}"
        result_payload = {"status": skill_status, "message": skill_message}
        metrics = {
            "duration_seconds": duration, # Use overall duration for setup errors
            "status": skill_status,
            "message": skill_message[:500]
        }
        # No need to return here, let finally handle it

    finally:
        # Calculate final duration based on whether execution happened
        if 'duration_exec' in locals(): # If execution try block was entered
             metrics = {
                "duration_seconds": duration_exec,
                "status": skill_status,
                "message": skill_message[:500]
            }
        else: # Setup failed before execution try block
             metrics = {
                "duration_seconds": time.time() - start_time, # Overall duration
                "status": skill_status,
                "message": skill_message[:500]
            }

        ctx_logger.info(f"{log_prefix} Completed '{tool_name}'. Status: {skill_status}, Duration: {metrics['duration_seconds']:.3f}s")
        # --- Notify Tool Execution End --- << HOOK >>
        final_result_for_hook = {"result": result_payload, "metrics": metrics}
        # Ensure notify gets called even if setup failed
        notify_tool_execution_end(tool_name, final_result_for_hook, source="ToolExecutor")
        # ----------------------------------

    # Combine result payload and metrics for the final return value
    final_return_value = {"result": result_payload, "metrics": metrics}
    ctx_logger.debug(f"{log_prefix} Final return value for '{tool_name}': {final_return_value}")
    return final_return_value

# Helper to parse docstrings (basic Google style)
def _parse_docstring_args(docstring: Optional[str]) -> Dict[str, str]:
    """Parses the Args section of a Google-style docstring."""
    if not docstring:
        return {}

    args_section_match = re.search(r"Args:(.*?)(Returns:|Raises:|$)", docstring, re.DOTALL | re.IGNORECASE)
    if not args_section_match:
        return {}

    args_section = args_section_match.group(1).strip()
    args_dict = {}
    # Regex to capture argument name and its description
    arg_pattern = re.compile(r"^\s*([\w_]+)\s*(?:\([^)]*\))?:\s*(.*?)(?=\n\s*\w+\s*(?:\([^)]*\))?:|\Z)", re.MULTILINE | re.DOTALL)

    for match in arg_pattern.finditer(args_section):
        arg_name = match.group(1)
        description = " ".join(match.group(2).strip().split())
        args_dict[arg_name] = description

    return args_dict
