import logging
import json
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import inspect

# Imports from core needed for execution logic
from a3x.core.llm_interface import LLMInterface
from a3x.core.tool_registry import ToolRegistry
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext # Need context class
from a3x.core.agent_parser import parse_llm_response
from a3x.core.prompt_builder import build_worker_messages # Worker prompts are built here
from a3x.core.context import SharedTaskContext
from a3x.core.config import MAX_REACT_ITERATIONS, MAX_FRAGMENT_RUNTIME # Import timeout config
from a3x.core.utils.param_normalizer import normalize_action_input
# Import FragmentRegistry to get fragment instances if needed (alternative: pass instance)
from a3x.fragments.registry import FragmentRegistry
# Import Status Constants
from a3x.core.constants import (
    STATUS_SUCCESS, STATUS_ERROR, STATUS_MAX_ITERATIONS, STATUS_TIMEOUT,
    STATUS_NOT_FOUND, STATUS_NOT_ALLOWED,
    REASON_LLM_ERROR, REASON_LLM_PROCESSING_ERROR, REASON_PROMPT_BUILD_FAILED,
    REASON_FRAGMENT_NOT_FOUND, REASON_ACTION_FAILED, REASON_ACTION_NOT_FOUND,
    REASON_ACTION_NOT_ALLOWED, REASON_NO_ACTION_LOOP, REASON_MAX_ITERATIONS,
    REASON_TIMEOUT, REASON_UNKNOWN
)

class FragmentExecutionError(Exception):
    """Custom exception for errors during fragment execution."""
    def __init__(self, message, status=STATUS_ERROR, reason=REASON_UNKNOWN):
        super().__init__(message)
        self.status = status
        self.reason = reason

class FragmentExecutor:
    """
    Handles the execution lifecycle of a single Fragment, including its internal ReAct cycle.
    """
    def __init__(
        self,
        llm_interface: LLMInterface,
        tool_registry: ToolRegistry,
        fragment_registry: FragmentRegistry, # Needed to get fragment instance
        memory_manager: 'MemoryManager', # <<< ADDED memory_manager >>>
        workspace_root: Path,
        max_iterations: int = MAX_REACT_ITERATIONS,
        max_runtime: int = MAX_FRAGMENT_RUNTIME, # Add max_runtime
    ):
        self.llm_interface = llm_interface
        self.tool_registry = tool_registry
        self.fragment_registry = fragment_registry
        self.memory_manager = memory_manager # <<< STORE memory_manager >>>
        self.workspace_root = workspace_root
        self.max_iterations = max_iterations
        self.max_runtime = max_runtime # Store max_runtime
        # Logger will be passed during execution call

    # --- ReAct Cycle Helpers (Moved from Orchestrator) ---

    async def _process_llm_response(
        self, messages: List[Dict[str, str]], log_prefix: str, logger: logging.Logger
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Calls LLM and parses Thought/Action/Input format (ReAct cycle).
           Returns (thought, action_name, action_input_dict) or (None, None, None)."""
        llm_response_raw = ""
        try:
            logger.debug(f"{log_prefix} Calling llm_interface.call_llm (stream=False expected for ReAct cycle)")
            # Ensure stop sequence is passed if needed by the worker prompt/model
            async for chunk in self.llm_interface.call_llm(
                messages=messages,
                stream=False,
                stop=["\nObservation:"] # Standard ReAct stop sequence
            ):
                llm_response_raw += chunk

            if not llm_response_raw:
                 logger.warning(f"{log_prefix} LLM call returned empty response for ReAct cycle.")
                 return None, None, None # Indicate non-actionable response

            thought, action_name, action_input_str = parse_llm_response(llm_response_raw, logger)

            if not action_name:
                 # Handle cases where only Final Answer is given or no action
                 if thought and thought.startswith("Final Answer:"):
                      final_answer_content = thought.split("Final Answer:", 1)[1].strip()
                      logger.info(f"{log_prefix} LLM provided Final Answer directly.")
                      # Synthesize 'final_answer' action for consistent handling
                      return None, "final_answer", {"answer": final_answer_content}
                 elif thought:
                     # If there's thought but no action, it might be reflection or just thinking.
                     # Return thought to be added to history, but no action to execute.
                     logger.info(f"{log_prefix} LLM provided Thought but no Action.")
                     return thought, None, None
                 else:
                     logger.warning(f"{log_prefix} Could not parse Action or Thought from LLM response: {llm_response_raw[:500]}")
                     return None, None, None # Indicate non-actionable response

            # Attempt to parse action input
            action_input_dict = None
            if action_input_str:
                try:
                    # Handle potential markdown code blocks around JSON
                    if action_input_str.strip().startswith("```json"):
                        action_input_str = action_input_str.strip()[7:-3].strip()
                    elif action_input_str.strip().startswith("```"):
                         action_input_str = action_input_str.strip()[3:-3].strip()

                    action_input_dict = json.loads(action_input_str)
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix} Failed to parse Action Input JSON: {action_input_str}")
                    # Proceed with the action name but None input, executor can decide how to handle
                    return thought, action_name, None
            else:
                 action_input_dict = {} # No input provided is valid

            return thought, action_name, action_input_dict

        except Exception as e:
            logger.exception(f"{log_prefix} Error processing LLM response for ReAct: {e}")
            # Indicate error during LLM processing
            raise FragmentExecutionError(f"LLM response processing failed: {e}", reason=REASON_LLM_ERROR) from e

    async def _execute_action(
        self,
        action_name: str,
        action_input: Optional[Dict[str, Any]],
        allowed_skills: List[str], # Use the specific list for this fragment
        log_prefix: str,
        shared_task_context: SharedTaskContext,
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        """Executes the chosen action (skill) using the tool registry and returns the observation."""
        if action_input is None:
             action_input = {} # Ensure action_input is a dict for normalization
             logger.warning(f"{log_prefix} Action input was None, using empty dict for {action_name}.")

        logger.info(
            f"{log_prefix} Executing Action: {action_name} with input: {json.dumps(action_input)}"
        )

        # 1. Check if skill is allowed for this fragment
        if action_name not in allowed_skills:
            logger.error(f"{log_prefix} Skill '{action_name}' is not allowed for this fragment/task step.")
            return {
                "status": STATUS_NOT_ALLOWED, # Use Constant
                "action": f"{action_name}_not_allowed",
                "reason": REASON_ACTION_NOT_ALLOWED, # Use Constant
                "data": {"message": f"Skill '{action_name}' is not in the allowed list for this step."},
            }

        # 2. Check if skill exists in the registry
        if action_name not in self.tool_registry.list_tools():
            logger.error(f"{log_prefix} Skill '{action_name}' not found in tool registry.")
            return {
                "status": STATUS_NOT_FOUND, # Use Constant
                "action": f"{action_name}_not_found",
                "reason": REASON_ACTION_NOT_FOUND, # Use Constant
                "data": {"message": f"Skill '{action_name}' not found in the registry."},
            }

        try:
            # 3. Normalize input
            normalized_action_input = normalize_action_input(action_name, action_input)
            if normalized_action_input != action_input:
                logger.info(f"{log_prefix} Action input normalized to: {json.dumps(normalized_action_input)}")

            # <<< ADDED: Argument Validation >>>
            try:
                _, tool_callable = self.tool_registry.get_instance_and_tool(action_name)
                tool_sig = inspect.signature(tool_callable)
                required_params = {
                    name
                    for name, param in tool_sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                    and name not in ["self", "ctx", "context", "shared_task_context"]
                }
                provided_params = set(normalized_action_input.keys())
                missing_params = required_params - provided_params
                if missing_params:
                    error_msg = f"Missing required arguments for tool '{action_name}': {missing_params}"
                    logger.error(f"{log_prefix} {error_msg}. Input received: {normalized_action_input}")
                    return {
                        "status": STATUS_ERROR,
                        "action": f"{action_name}_missing_args",
                        "reason": REASON_ACTION_FAILED, # Or a more specific reason
                        "data": {"message": error_msg}
                    }
            except KeyError: # Should be caught earlier, but safeguard
                logger.error(f"{log_prefix} Tool '{action_name}' not found during argument validation (this shouldn't happen).", exc_info=True)
                return {"status": STATUS_NOT_FOUND, "action": f"{action_name}_not_found", "reason": REASON_ACTION_NOT_FOUND, "data": {"message": "Tool disappeared during validation?"}}
            except Exception as val_err:
                 logger.error(f"{log_prefix} Unexpected error during argument validation for '{action_name}': {val_err}", exc_info=True)
                 return {"status": STATUS_ERROR, "action": f"{action_name}_validation_error", "reason": REASON_ACTION_FAILED, "data": {"message": f"Argument validation error: {val_err}"}}
            # <<< END: Argument Validation >>>

            # 4. Create execution context
            # Note: Allowed skills passed here might be redundant if check is done above,
            # but keep for potential future use within execute_tool or skill itself.
            exec_context = _ToolExecutionContext(
                logger=logger,
                workspace_root=self.workspace_root,
                llm_url=self.llm_interface.llm_url if self.llm_interface else None,
                tools_dict=self.tool_registry, # Pass registry
                llm_interface=self.llm_interface,
                fragment_registry=self.fragment_registry, # Pass fragment registry too
                shared_task_context=shared_task_context,
                allowed_skills=allowed_skills, # Pass allowed_skills list
                skill_instance=None, # execute_tool will find/create the instance
                memory_manager=self.memory_manager # Assuming memory_manager is available on FragmentExecutor
            )

            # 5. Execute the tool
            tool_result = await execute_tool(
                tool_name=action_name,
                action_input=normalized_action_input,
                tools_dict=self.tool_registry, # Pass registry instance
                context=exec_context,
            )
            logger.info(
                f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}"
            )
            # Ensure result is serializable
            try:
                 json.dumps(tool_result)
            except (TypeError, OverflowError) as e:
                 logger.error(f"{log_prefix} Tool result for {action_name} is not JSON serializable: {e}. Replacing complex data.")
                 # Simplify or truncate complex/non-serializable data
                 simplified_data = {}
                 if isinstance(tool_result.get("data"), dict):
                      for k, v in tool_result["data"].items():
                           try:
                                json.dumps({k: v})
                                simplified_data[k] = v
                           except (TypeError, OverflowError):
                                simplified_data[k] = f"<non-serializable data type: {type(v).__name__}>"
                 else:
                      simplified_data = f"<non-serializable data type: {type(tool_result.get('data')).__name__}>"

                 tool_result = {
                     "status": tool_result.get("status", "unknown"),
                     "action": tool_result.get("action", action_name),
                     "data": simplified_data,
                     "error_details": "Original data was not JSON serializable"
                 }


            return tool_result
        except Exception as tool_err:
            logger.exception(f"{log_prefix} Error executing tool '{action_name}':")
            # Wrap the error in a standard format
            return {
                "status": STATUS_ERROR, # Use Constant
                "action": f"{action_name}_failed",
                "reason": REASON_ACTION_FAILED, # Use Constant
                "data": {"message": f"Error during tool execution: {tool_err}", "traceback": traceback.format_exc()},
            }

    # --- Main Execution Method ---

    async def execute(
        self,
        fragment_name: str,
        sub_task_objective: str,
        overall_objective: str, # Pass the main goal for context
        fragment_history: List[Tuple[str, str]], # Specific history for this fragment's execution
        shared_task_context: SharedTaskContext,
        allowed_skills: List[str], # Skills allowed for this fragment execution
        logger: logging.Logger, # Use logger passed from orchestrator
    ) -> Dict[str, Any]:
        """
        Executes the ReAct cycle for a given fragment to achieve its sub-task objective.

        Args:
            fragment_name: Name of the fragment to execute.
            sub_task_objective: The specific goal for this fragment instance.
            overall_objective: The main task goal for broader context.
            fragment_history: History specific to this fragment's attempts (Observation/Thought pairs).
            shared_task_context: The shared context object for the overall task.
            allowed_skills: List of skill names this fragment is allowed to use.
            logger: Logger instance for this execution.

        Returns:
            A dictionary containing the execution result:
            {
                "status": "success" | "error" | "max_iterations_reached",
                "final_answer": Optional[str], # If the fragment provided a final answer
                "result_data": Optional[Any], # Any data returned by the last successful action
                "history": List[Tuple[str, str]], # Updated fragment history
                "reason": Optional[str] # Description if status is error
            }
        """
        log_prefix = f"[FragmentExecutor|{fragment_name}]"
        logger = logging.getLogger(f"a3x.assistant_cli.{log_prefix}")

        logger.info(f"{log_prefix} Starting execution for sub-task: {sub_task_objective}")
        logger.info(f"{log_prefix} Allowed skills: {allowed_skills}")
        logger.info(f"{log_prefix} Max iterations: {self.max_iterations}, Max runtime: {self.max_runtime}s")

        start_time = time.monotonic()
        current_history = []

        # Get fragment definition (needed for prompt building potentially)
        try:
            fragment_definition = self.fragment_registry.get_fragment_definition(fragment_name)
            if not fragment_definition:
                 raise FragmentExecutionError(f"Fragment definition not found for {fragment_name}", status=STATUS_NOT_FOUND, reason=REASON_FRAGMENT_NOT_FOUND)
        except Exception as e:
            logger.error(f"{log_prefix} Failed to get fragment definition: {e}", exc_info=True)
            return {
                "status": STATUS_ERROR,
                "reason": REASON_FRAGMENT_NOT_FOUND,
                "message": f"Could not retrieve definition for fragment '{fragment_name}'.",
                "data": None
            }

        # Get and filter tool schemas/descriptions
        try:
            all_tool_schemas = self.tool_registry.list_tools()
            allowed_tool_schemas = {name: schema for name, schema in all_tool_schemas.items() if name in allowed_skills}
            if not allowed_tool_schemas and allowed_skills:
                 logger.warning(f"{log_prefix} Allowed skills specified ({allowed_skills}), but no matching schemas found in ToolRegistry.")
                 # Decide how to handle: error or proceed with empty tools?
                 # For now, proceed but log warning.
            elif not allowed_skills:
                 logger.warning(f"{log_prefix} No allowed skills provided. Fragment execution might be limited.")
                 allowed_tool_schemas = {} # Ensure it's an empty dict
        except Exception as e:
            logger.error(f"{log_prefix} Error retrieving or filtering tool schemas: {e}", exc_info=True)
            return {"status": STATUS_ERROR, "reason": REASON_ACTION_NOT_FOUND, "message": "Failed to retrieve tool information.", "data": None}

        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"{log_prefix} Iteration {iterations}/{self.max_iterations}")

            # --- Check Timeout --- 
            current_time = time.monotonic()
            elapsed_time = current_time - start_time
            if elapsed_time > self.max_runtime:
                logger.error(f"{log_prefix} Maximum runtime ({self.max_runtime}s) exceeded.")
                return {"status": STATUS_TIMEOUT, "reason": REASON_TIMEOUT, "message": "Fragment exceeded maximum runtime.", "data": {"history": current_history, "iterations": iterations}}
            # --- End Timeout Check --- 

            # 1. Build Worker Prompt using allowed tools
            try:
                messages = build_worker_messages(
                    sub_task=sub_task_objective, # Renamed argument
                    history=current_history,
                    allowed_skills=allowed_skills, # Pass only allowed skill names
                    all_tools=self.tool_registry.list_tools() # Corrected method name
                )
            except Exception as e:
                logger.error(f"{log_prefix} Failed to build worker prompt: {e}", exc_info=True)
                return {"status": STATUS_ERROR, "reason": REASON_PROMPT_BUILD_FAILED, "message": f"Error building prompt: {e}", "data": {"history": current_history, "iterations": iterations}}

            # 2. Call LLM and Parse Response
            try:
                 thought, action_name, action_input = await self._process_llm_response(
                     messages, log_prefix, logger
                 )
            except FragmentExecutionError as e:
                 logger.error(f"{log_prefix} Error processing LLM response: {e}")
                 return {"status": e.status, "reason": e.reason, "history": current_history, "result_data": {"message": str(e)}}
            except Exception as e: # Catch unexpected errors during LLM processing
                 logger.exception(f"{log_prefix} Unexpected error during LLM processing step:")
                 return {"status": STATUS_ERROR, "reason": REASON_LLM_PROCESSING_ERROR, "history": current_history, "result_data": {"message": str(e)}}


            observation_str = ""
            action_result = None

            if thought:
                 logger.info(f"{log_prefix} Thought: {thought}")
                 # Add thought to temporary history for context, but observation is the key pair
                 # We'll add the full Thought/Action/Observation later if an action occurs

            if action_name:
                 logger.info(f"{log_prefix} Action: {action_name}, Input: {json.dumps(action_input)}")

                 # --- Execute Action ---
                 action_result = await self._execute_action(
                     action_name=action_name,
                     action_input=action_input,
                     allowed_skills=allowed_skills,
                     log_prefix=log_prefix,
                     shared_task_context=shared_task_context,
                     logger=logger,
                 )

                 # --- Handle Final Answer Action ---
                 if action_name == "final_answer":
                     final_answer_content = action_input.get("answer", "No answer provided.") if isinstance(action_input, dict) else "Invalid input for final_answer."
                     logger.info(f"{log_prefix} Fragment concluded with Final Answer: {final_answer_content}")
                     # Add the final thought/action to history before returning
                     final_thought = thought if thought else f"Providing final answer for sub-task '{sub_task_objective}'"
                     current_history.append((f"Thought: {final_thought}\nAction: {action_name}\nAction Input: {json.dumps(action_input)}", f"Observation: Final answer submitted by fragment."))
                     return {
                         "status": STATUS_SUCCESS, # Use Constant
                         "final_answer": final_answer_content,
                         "result_data": action_result, # Include result of final_answer call if any
                         "history": current_history,
                     }

                 # --- Format Observation ---
                 # Ensure observation is a string, truncate if necessary
                 try:
                     # Attempt to pretty-print JSON if data is dict/list
                     if isinstance(action_result, (dict, list)):
                         observation_content = json.dumps(action_result, indent=2)
                     else:
                         observation_content = str(action_result)

                     # Simple truncation for safety
                     max_obs_len = 2000
                     if len(observation_content) > max_obs_len:
                         observation_content = observation_content[:max_obs_len] + "... [truncated]"
                     observation_str = f"Observation: {observation_content}"
                 except Exception as e:
                     logger.error(f"{log_prefix} Failed to format observation: {e}. Action Result: {action_result}")
                     observation_str = f"Observation: [Error formatting observation: {e}]"

                 logger.info(f"{log_prefix} {observation_str}")

                 # --- Add completed step to history ---
                 # Combine thought, action, and observation for the history record
                 history_entry_prompt = f"Thought: {thought if thought else '(No thought provided)'}\nAction: {action_name}\nAction Input: {json.dumps(action_input)}"
                 current_history.append((history_entry_prompt, observation_str))

                 # Check if the action result indicates an error that should stop the loop
                 if isinstance(action_result, dict) and action_result.get("status") == STATUS_ERROR: # Use Constant
                      logger.error(f"{log_prefix} Action '{action_name}' failed: {action_result.get('data', {}).get('message', 'No details')}. Stopping fragment execution.")
                      return {
                          "status": STATUS_ERROR, # Use Constant
                          "reason": f"{REASON_ACTION_FAILED}:{action_name}", # Use Constant
                          "result_data": action_result,
                          "history": current_history,
                      }

            else:
                 # No action was decided by the LLM
                 logger.info(f"{log_prefix} No action taken in this iteration.")
                 # Should we stop here or let it try again?
                 # For now, let's assume if there's no action after a thought, it might be stuck.
                 # Add the thought to history if it exists, otherwise add a note.
                 if thought:
                      current_history.append((f"Thought: {thought}", "Observation: No action was taken by the agent."))
                 else:
                      # This case (no thought, no action) should ideally be handled in _process_llm_response
                      logger.warning(f"{log_prefix} No thought and no action in iteration. Ending fragment execution.")
                      return {"status": STATUS_ERROR, "reason": REASON_NO_ACTION_LOOP, "history": current_history}


            # Check history length (optional, maybe handled by prompt builder)
            # current_history = trim_history(current_history, MAX_HISTORY_TURNS)

        # Loop finished without final_answer
        logger.warning(f"{log_prefix} Reached maximum iterations ({self.max_iterations}) without a final answer.")
        return {
            "status": STATUS_MAX_ITERATIONS, # Use Constant
            "reason": REASON_MAX_ITERATIONS, # Use Constant
            "history": current_history,
            "result_data": action_result # Return the result of the last action if any
        } 