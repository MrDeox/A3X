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
from a3x.core.context import SharedTaskContext, FragmentContext, _ToolExecutionContext

# Import SKILL_REGISTRY and PROJECT_ROOT for instantiation and context
from a3x.core.registry_instance import SKILL_REGISTRY
from a3x.core.skills import discover_skills
from a3x.core.config import PROJECT_ROOT

# --- Import State Update Hooks ---
from a3x.core.hooks.state_updates import (
    notify_tool_execution_start,
    notify_tool_execution_end
)
# --- End Hook Imports ---

logger = logging.getLogger(__name__)

# Utility function, keep outside the class or make it static
def _parse_docstring_args(docstring: Optional[str]) -> Dict[str, str]:
    """Parses arguments from a Python function's docstring (numpy style)."""
    # Simple placeholder implementation - enhance as needed
    args = {}
    if not docstring:
        return args
    lines = docstring.strip().split('\n')
    in_args_section = False
    for line in lines:
        line = line.strip()
        if line.lower() in ('args:', 'arguments:', 'parameters:'):
            in_args_section = True
            continue
        if in_args_section:
            if not line or line.startswith(('---', '===')):
                break # End of section
            match = re.match(r"^([\w\*]+)\s*\(([^)]*)\)\s*:\s*(.*)$", line)
            if match:
                name, type_info, desc = match.groups()
                args[name.strip()] = desc.strip()
            else:
                # Handle lines without type info
                parts = line.split(':', 1)
                if len(parts) == 2:
                    name, desc = parts
                    args[name.strip()] = desc.strip()
    return args

class ToolExecutor:
    """
    Manages the loading and execution of skills (tools).
    This class is intended to be instantiated and passed within the Context.
    """

    def __init__(self, tool_registry: Any = None): # Made tool_registry optional
        """Initializes the ToolExecutor."""
        resolved_registry = None
        if tool_registry:
            # Check if the provided object looks like our ToolRegistry
            if hasattr(tool_registry, 'get_tool') and callable(tool_registry.get_tool):
                resolved_registry = tool_registry
                logger.info(f"ToolExecutor initialized with provided registry: {type(tool_registry).__name__}")
            else:
                # Warn if an invalid object was passed, before falling back
                logger.warning(f"Provided tool_registry ({type(tool_registry).__name__}) is invalid or lacks 'get_tool' method. Falling back to global SKILL_REGISTRY instance.")
        
        if not resolved_registry:
            # Fallback to the global singleton instance
            logger.info("No valid tool_registry provided or fallback needed. Using global SKILL_REGISTRY instance from registry_instance.py.")
            resolved_registry = SKILL_REGISTRY # Use the imported singleton instance
            
            # Ensure skills are loaded into the global registry if it's empty
            if not resolved_registry.list_tools(): 
                logger.info("Global skill registry appears empty, attempting to load default skills...")
                try:
                    # TODO: Make this configurable
                    default_skill_packages = ['a3x.skills.core', 'a3x.skills.auto_generated'] 
                    discover_skills()
                    logger.info(f"Skills loaded successfully into global registry from {default_skill_packages}.")
                    if not resolved_registry.list_tools():
                        logger.warning("discover_skills completed but global skill registry still seems empty.")
                except Exception as e:
                    logger.exception("Failed to load default skills automatically during ToolExecutor fallback initialization.")
                    # Continuing with potentially empty registry
            
        # Final check on the resolved registry (should be ToolRegistry instance)
        if not hasattr(resolved_registry, 'get_tool') or not callable(resolved_registry.get_tool):
            # This should ideally not happen now if the singleton is imported correctly
            logger.critical(f"Resolved tool registry ({type(resolved_registry).__name__}) lacks 'get_tool' method. ToolExecutor cannot function.")
            raise TypeError("Valid tool registry with 'get_tool' method is required.")
        
        self.tool_registry = resolved_registry
        logger.info(f"ToolExecutor successfully initialized using registry: {type(self.tool_registry).__name__}")

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any], # Renamed from action_input for clarity
        context: Union[_ToolExecutionContext, FragmentContext, Any] # Expect Context object, allow Any for flexibility
    ) -> Dict[str, Any]:
        """Executa a ferramenta/skill, medindo tempo e status, e retorna resultado + métricas."""
        log_prefix = "[Tool Executor]"
        # Attempt to get logger from context, fallback to global logger
        ctx_logger = getattr(context, 'logger', logger)
        ctx_logger.info(f"{log_prefix} Executing tool '{tool_name}' with input keys: {list(tool_input.keys()) if tool_input else []}")
        
        start_time_overall = time.time()
        result_payload: Dict[str, Any] = {}
        skill_status: str = "error"
        skill_message: str = "Execution did not start or failed before completion."
        metrics = {
            "duration_seconds": 0.0,
            "status": "error",
            "message": "Tool execution did not proceed past initial validation."
        }
        duration_exec = 0.0 # Initialize execution duration

        try:
            # Retrieve the instance and the bound tool method from the registry
            tool_instance, tool_method = self.tool_registry.get_tool(tool_name)

            if not tool_method or not callable(tool_method):
                error_message = f"Tool '{tool_name}' found in registry but is not callable."
                ctx_logger.error(f"{log_prefix} {error_message}")
                skill_status = "error"
                skill_message = error_message
                result_payload = {"status": skill_status, "action": "tool_not_callable", "data": {"message": error_message}}
                raise ValueError(error_message) # Raise to go to finally block

            # --- Context Handling --- 
            # Determine the effective context to pass to the skill
            effective_context = context # Default to the received context
            if isinstance(context, FragmentContext):
                # Convert FragmentContext to _ToolExecutionContext if possible
                try:
                    effective_context = _ToolExecutionContext(
                        context.logger,
                        context.workspace_root,
                        getattr(context.llm_interface, 'llm_url', None), 
                        self.tool_registry, 
                        context.llm_interface,
                        context.fragment_registry,
                        context.shared_task_context,
                        None, # allowed_skills - Placeholder
                        tool_instance, 
                        context.memory_manager 
                    )
                except AttributeError as ae:
                    ctx_logger.warning(f"Could not fully create _ToolExecutionContext from FragmentContext: {ae}. Passing FragmentContext directly.")
                    effective_context = context # Pass original if conversion fails
            elif isinstance(context, _ToolExecutionContext):
                # Ensure skill_instance is set correctly
                try:
                    effective_context = context._replace(skill_instance=tool_instance)
                except AttributeError:
                    ctx_logger.warning("Could not use _replace on _ToolExecutionContext. Passing original context.")
                    effective_context = context
            else:
                # If it's neither, pass it as is, the skill must handle it
                ctx_logger.warning(f"ToolExecutor received unexpected context type: {type(context).__name__}. Passing it directly to the skill.")
                effective_context = context
            # --- End Context Handling ---

            # Check if it's an async function
            is_async = asyncio.iscoroutinefunction(tool_method)
            sig = inspect.signature(tool_method)
            params = sig.parameters

            # --- Argument Passing Logic --- 
            call_kwargs = {} 
            
            # 1. Add context/ctx if the skill expects it
            #    Prefer passing the derived 'effective_context' if it matches the expected type hint, otherwise pass the original context.
            if 'context' in params:
                # Check type hint if available
                param_type = params['context'].annotation
                if param_type is inspect.Parameter.empty or isinstance(effective_context, param_type):
                    call_kwargs['context'] = effective_context 
                else:
                    ctx_logger.debug(f"Passing original context to 'context' param due to type mismatch (Expected: {param_type}, Got: {type(effective_context)}).")
                    call_kwargs['context'] = context # Pass original context if type hint doesn't match effective_context
            elif 'ctx' in params:
                param_type = params['ctx'].annotation
                if param_type is inspect.Parameter.empty or isinstance(effective_context, param_type):
                    call_kwargs['ctx'] = effective_context
                else:
                    ctx_logger.debug(f"Passing original context to 'ctx' param due to type mismatch (Expected: {param_type}, Got: {type(effective_context)}).")
                    call_kwargs['ctx'] = context
            
            # 2. Add shared_task_context if expected and available
            if 'shared_task_context' in params and hasattr(effective_context, 'shared_task_context') and effective_context.shared_task_context is not None:
                call_kwargs['shared_task_context'] = effective_context.shared_task_context

            # 3. Add arguments from tool_input
            original_tool_input = tool_input or {}
            provided_unexpected_args = set()
            for k, v in original_tool_input.items():
                if k in params:
                    # Avoid overwriting context arguments passed explicitly above
                    if k not in ('context', 'ctx', 'shared_task_context'): 
                        call_kwargs[k] = v
                    # Do not warn if context keys are present, they are just ignored from input
                else:
                    provided_unexpected_args.add(k)

            if provided_unexpected_args:
                ctx_logger.warning(f"{log_prefix} Tool input provided unexpected arguments for {tool_name}: {provided_unexpected_args}. These were ignored.")
            # --- End Argument Passing Logic ---

            ctx_logger.debug(f"{log_prefix} Preparing to call {'async' if is_async else 'sync'} tool '{tool_name}'. Final Kwargs: {list(call_kwargs.keys())}")

            # --- Execute the tool method --- 
            start_time_exec = time.monotonic()
            # --- Notify Tool Execution Start --- << HOOK >>
            notify_tool_execution_start(tool_name, call_kwargs, source="ToolExecutor")
            # ----------------------------------
            if is_async:
                ctx_logger.debug(f"{log_prefix} Awaiting async tool '{tool_name}'")
                result = await tool_method(**call_kwargs)
            else:
                ctx_logger.debug(f"{log_prefix} Calling sync tool '{tool_name}'")
                result = tool_method(**call_kwargs)
            
            end_time_exec = time.monotonic()
            duration_exec = end_time_exec - start_time_exec
            ctx_logger.info(f"{log_prefix} Tool '{tool_name}' executed successfully in {duration_exec:.4f}s.")

            # --- Process result --- 
            if isinstance(result, dict) and 'status' in result:
                skill_status = result.get('status', 'unknown')
                if skill_status != "success":
                    skill_message = result.get('error', result.get('message', f"Tool reported {skill_status} status with no message/error key."))
                else:
                    skill_message = result.get('message', 'Execution reported success with no message.')
                result_payload = result
            else:
                ctx_logger.warning(f"{log_prefix} Tool '{tool_name}' executed but returned an unexpected format: {type(result)}. Wrapping as success.")
                skill_status = "success"
                skill_message = "Execution successful, non-standard return format."
                result_payload = {"status": skill_status, "data": result} # Wrap non-dict results

        except Exception as e:
            # This block catches errors during tool lookup or execution
            end_time_exec = time.monotonic()
            # Use start_time_exec if it exists, otherwise overall start time for duration
            exec_start = start_time_exec if 'start_time_exec' in locals() else start_time_overall
            duration_exec = end_time_exec - exec_start
            
            ctx_logger.exception(f"{log_prefix} Error during execution/setup for tool '{tool_name}': {e}")
            skill_status = "error"
            # Use the specific error message if available, otherwise the general one
            skill_message = str(e) if str(e) else f"Execution failed: {type(e).__name__}"
            result_payload = {"status": skill_status, "message": skill_message}
            # Set metrics based on execution error
            # Metrics dict is defined outside the try block, update it here
            # metrics = {
            #     "duration_seconds": duration_exec,
            #     "status": skill_status,
            #     "message": skill_message[:500]
            # }
            # Note: No raise here, finally block will handle return

        finally:
            # Calculate final metrics regardless of success or failure point
            metrics = {
                "duration_seconds": duration_exec if duration_exec > 0 else time.time() - start_time_overall, # Use exec time if available, else overall
                "status": skill_status,
                "message": skill_message[:500] # Truncate long messages
            }
            ctx_logger.info(f"{log_prefix} Completed '{tool_name}'. Status: {skill_status}, Duration: {metrics['duration_seconds']:.3f}s")
            # --- Notify Tool Execution End --- << HOOK >>
            final_result_for_hook = {"result": result_payload, "metrics": metrics}
            notify_tool_execution_end(tool_name, final_result_for_hook, source="ToolExecutor")
            # ----------------------------------
            
            ctx_logger.debug(f"{log_prefix} Final Result Payload: {result_payload}")
            ctx_logger.debug(f"{log_prefix} Final Metrics: {metrics}")
            # Return structure consistent with previous function
            return {"result": result_payload, "metrics": metrics}
