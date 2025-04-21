import logging
import json
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import inspect
import asyncio

# Imports from core needed for execution logic
# from a3x.core.llm_interface import LLMInterface # REMOVED
from a3x.core.tool_registry import ToolRegistry
# from a3x.core.tool_executor import execute_tool, _ToolExecutionContext # REMOVED - Handled by Fragment
# from a3x.core.agent_parser import parse_llm_response # REMOVED
# from a3x.core.prompt_builder import build_worker_messages # REMOVED
from a3x.core.context import SharedTaskContext
from a3x.core.config import MAX_FRAGMENT_RUNTIME # Only runtime needed
from a3x.fragments.registry import FragmentRegistry
# Import Status Constants
from a3x.core.constants import (
    STATUS_SUCCESS, STATUS_ERROR, STATUS_TIMEOUT,
    STATUS_NOT_FOUND, STATUS_NOT_ALLOWED,
    REASON_FRAGMENT_NOT_FOUND, REASON_TIMEOUT, REASON_UNKNOWN,
    # Removed LLM/ReAct specific reasons
    REASON_ACTION_FAILED, REASON_ACTION_NOT_FOUND, REASON_ACTION_NOT_ALLOWED
)

class FragmentExecutionError(Exception):
    """Custom exception for errors during fragment execution."""
    def __init__(self, message, status=STATUS_ERROR, reason=REASON_UNKNOWN):
        super().__init__(message)
        self.status = status
        self.reason = reason

class FragmentExecutor:
    """
    Handles loading a Fragment instance and calling its execute method.
    Does NOT contain internal ReAct logic anymore.
    """
    def __init__(
        self,
        # llm_interface: LLMInterface, # REMOVED
        tool_registry: ToolRegistry, # Still needed for validation?
        fragment_registry: FragmentRegistry, # Needed to get fragment instance
        memory_manager: 'MemoryManager', # <<< ADDED memory_manager >>>
        logger: logging.Logger, # Receive logger directly
        max_runtime: int = MAX_FRAGMENT_RUNTIME,
    ):
        # self.llm_interface = llm_interface # REMOVED
        self.tool_registry = tool_registry
        self.fragment_registry = fragment_registry
        self.memory_manager = memory_manager # <<< STORE memory_manager >>>
        self.max_runtime = max_runtime # Store max_runtime
        self.logger = logger # Store logger

    # --- REMOVED ReAct Cycle Helpers (_process_llm_response, _execute_action) ---

    # --- Main Execution Method ---

    async def execute(
        self,
        fragment_name: str,
        sub_task: str, # Renamed from sub_task_objective for clarity
        objective: str, # Renamed from overall_objective
        fragment_history: List[Tuple[str, str]], # History specific to this fragment context
        shared_context: SharedTaskContext, # Renamed from shared_task_context
        allowed_skills: List[str], # Skills allowed for this fragment execution
    ) -> Dict[str, Any]:
        """
        Loads and executes the specified fragment's main logic.

        Args:
            fragment_name: Name of the fragment to execute.
            sub_task: The specific goal for this fragment instance.
            objective: The main task goal for broader context.
            fragment_history: History specific to this fragment's attempts.
            shared_context: The shared context object for the overall task.
            allowed_skills: List of skill names this fragment is allowed to use.

        Returns:
            A dictionary containing the execution result directly from the fragment:
            {
                "status": "success" | "error" | ..., # Status reported by the fragment
                "message": Optional[str], # Message from the fragment
                "final_answer": Optional[str], # If the fragment provided a final answer
                "data": Optional[Any], # Any data returned by the fragment (e.g., plan, history)
                "reason": Optional[str] # Description if status is error
            }
        """
        log_prefix = f"[FragmentExecutor|{fragment_name}]"
        self.logger.info(f"{log_prefix} Preparing to execute fragment for sub-task: {sub_task}")
        self.logger.info(f"{log_prefix} Allowed skills: {allowed_skills}")
        self.logger.info(f"{log_prefix} Max runtime: {self.max_runtime}s")

        start_time = time.monotonic()

        try:
            # 1. Load the fragment instance
            self.logger.debug(f"{log_prefix} Loading fragment instance...")
            fragment = self.fragment_registry.get_fragment(fragment_name)
            if not fragment:
                 self.logger.error(f"{log_prefix} Fragment instance '{fragment_name}' not found or failed to load.")
                 # Use the custom exception for consistency?
                 # raise FragmentExecutionError(f"Fragment '{fragment_name}' not found.", status=STATUS_NOT_FOUND, reason=REASON_FRAGMENT_NOT_FOUND)
                 return {
                     "status": STATUS_NOT_FOUND,
                     "reason": REASON_FRAGMENT_NOT_FOUND,
                     "message": f"Fragment '{fragment_name}' could not be loaded.",
                     "data": None
                 }
            
            # --- Pass dependencies to Fragment if needed (e.g., logger, registries) ---
            # This assumes the BaseFragment or the specific fragment's __init__ handles
            # receiving necessary context/dependencies when loaded by the registry.
            # Alternatively, pass them explicitly here if needed by the execute method.
            # fragment.set_dependencies(...) # If such a method exists

            # 2. Execute the fragment's main logic within a timeout
            self.logger.info(f"{log_prefix} Calling fragment.execute()...")
            
            # Use asyncio.wait_for for timeout
            result_dict = await asyncio.wait_for(
                fragment.execute(
                    objective=objective,
                    sub_task=sub_task,
                    history=fragment_history, # Pass the history
                    shared_context=shared_context,
                    allowed_skills=allowed_skills,
                    # Pass logger if fragment.execute needs it?
                ),
                timeout=self.max_runtime
            )

            # 3. Basic validation of the result from the fragment
            if not isinstance(result_dict, dict):
                self.logger.error(f"{log_prefix} Fragment {fragment_name} execute() returned non-dict result: {type(result_dict)}")
                return {
                    "status": STATUS_ERROR,
                    "reason": "fragment_invalid_return",
                    "message": f"Fragment {fragment_name} returned an invalid result type.",
                    "data": None
                }

            # Ensure essential keys are present?
            # status = result_dict.get("status", STATUS_ERROR) # Default to error if status missing
            # result_dict["status"] = status # Ensure status is set

            self.logger.info(f"{log_prefix} Fragment execution finished. Status: {result_dict.get('status', 'N/A')}")
            self.logger.debug(f"{log_prefix} Fragment result: {result_dict}")
            return result_dict # Return the fragment's result directly

        except asyncio.TimeoutError:
             self.logger.error(f"{log_prefix} Fragment execution exceeded maximum runtime ({self.max_runtime}s).")
             return {
                 "status": STATUS_TIMEOUT,
                 "reason": REASON_TIMEOUT,
                 "message": f"Fragment '{fragment_name}' exceeded maximum runtime.",
                 "data": None
             }
        # Catch specific errors from fragment loading/execution if FragmentExecutionError is used
        except FragmentExecutionError as fee:
             self.logger.error(f"{log_prefix} FragmentExecutionError during execution of '{fragment_name}': {fee.message} (Status: {fee.status}, Reason: {fee.reason})", exc_info=True)
             return {"status": fee.status, "reason": fee.reason, "message": fee.message, "data": None}
        except Exception as e:
            self.logger.exception(f"{log_prefix} Unexpected error executing fragment '{fragment_name}':")
            return {
                "status": STATUS_ERROR,
                "reason": REASON_UNKNOWN,
                "message": f"Unexpected error during fragment execution: {e}",
                "data": {"traceback": traceback.format_exc()}
            } 