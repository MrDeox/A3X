import logging
import json
import re
import uuid
from typing import Dict, Any, List, AsyncGenerator, Optional, Union, Set, Tuple, Type
import traceback
import time
import inspect  # Add this import at the top of the file
from pathlib import Path # Added import for Path
import asyncio # <<< ADDED import >>>

# Package imports (copy necessary ones from agent.py)
from a3x.core.config import (
    MAX_REACT_ITERATIONS, # Maybe define orchestrator-specific limits?
    MAX_HISTORY_TURNS,
)
from a3x.core.skills import get_skill_descriptions # Needed? Maybe just pass descriptions
from a3x.core.db_utils import add_episodic_record
from a3x.core.prompt_builder import (
    build_orchestrator_messages, 
    build_worker_messages,
)
from a3x.core.agent_parser import parse_llm_response
from a3x.core.history_manager import trim_history
from a3x.core.tool_executor import execute_tool
from a3x.core.llm_interface import LLMInterface
from a3x.fragments.registry import FragmentRegistry
from a3x.core.context import SharedTaskContext, _ToolExecutionContext, Context
from a3x.fragments.base import FragmentContext, BaseFragment
from a3x.core.utils.param_normalizer import normalize_action_input
from a3x.core.tool_registry import ToolRegistry # Import ToolRegistry
# Remove unused import causing circular dependency
# from a3x.core.agent import ReactAgent, AgentState, AgentMetadata, AgentOutput
# from .persistence import PersistenceManager # Removed unused import
# from .task_manager import TaskManager # Removed unused import
# from .state_manager import AgentStateManager # Removed unused import
# from .memory import AgentMemory # Removed unused import

# --- Import State Update Hooks ---
from a3x.core.hooks.state_updates import (
    notify_new_task,
    notify_fragment_selection,
    notify_task_completion,
    notify_task_error,
    # notify_task_planning_complete, # Not used directly here yet
    notify_react_step,
    notify_tool_execution_start,
    notify_tool_execution_end
)
# --- End Hook Imports ---

# Placeholder for now:
def get_skills_for_fragment(fragment_name: str) -> List[str]:
    # Replace with actual logic later, maybe access registry?
    logging.warning(f"Placeholder get_skills_for_fragment called for {fragment_name}")
    if fragment_name == "PlannerFragment":
        return ["plan_task", "final_answer"]
    if fragment_name == "FileOpsManager":
         return ["read_file", "write_file", "list_files", "final_answer"]
    # Add other fragments or a default
    return ["final_answer"] # Default safe skill

# <<< ADDED import for chat monitor >>>
from .chat_monitor import chat_monitor_task

class TaskOrchestrator:
    """Handles the step-by-step orchestration of a task using Fragments and Skills."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        fragment_registry: FragmentRegistry,
        tool_registry: ToolRegistry, # Use ToolRegistry
        memory_manager: 'MemoryManager', # <<< ADD memory_manager parameter >>>
        workspace_root: Path,
        agent_logger: logging.Logger,
        # Potentially add agent_id if needed for state saving?
    ):
        self.llm_interface = llm_interface
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry # Store ToolRegistry
        self.memory_manager = memory_manager # <<< Store memory_manager >>>
        self.workspace_root = workspace_root
        self.logger = agent_logger # Use the passed logger
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if "Error" in self.fragment_descriptions:
             self.logger.error(f"[Orchestrator INIT] Failed to get fragment descriptions: {self.fragment_descriptions}")
             self.fragment_descriptions = "- FinalAnswerProvider: Provides final answer." # Fallback

    # --- LLM Interaction Helper (Moved from Agent) ---
    async def _process_llm_response(self, messages: List[Dict[str, str]], log_prefix: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Calls LLM and parses Thought/Action/Input format (ReAct cycle).
           Returns (thought, action_name, action_input_dict) or (None, None, None)."""
        llm_response_raw = ""
        try:
            self.logger.debug(f"{log_prefix} Calling llm_interface.call_llm (stream=False expected for ReAct cycle)")
            async for chunk in self.llm_interface.call_llm(
                messages=messages,
                stream=False,
                stop=["\nObservation:"] # Standard ReAct stop sequence
            ):
                llm_response_raw += chunk
                
            if not llm_response_raw:
                 self.logger.warning(f"{log_prefix} LLM call returned empty response for ReAct cycle.")
                 return None, None, None

            thought, action_name, action_input_str = parse_llm_response(llm_response_raw, self.logger)
            
            if not action_name:
                 # Handle cases where only Final Answer is given
                 if thought and thought.startswith("Final Answer:"):
                      final_answer_content = thought.split("Final Answer:", 1)[1].strip()
                      self.logger.info(f"{log_prefix} LLM provided Final Answer directly.")
                      return None, "final_answer", {"answer": final_answer_content} # Synthesize action
                 
                 self.logger.warning(f"{log_prefix} Could not parse Action from LLM response: {llm_response_raw[:500]}")
                 # Return thought if available, even without action
                 return (thought if thought and not thought.startswith("Final Answer:") else None), None, None 
            try:
                 action_input_dict = json.loads(action_input_str) if action_input_str else {}
            except json.JSONDecodeError:
                 self.logger.warning(f"{log_prefix} Failed to parse Action Input JSON: {action_input_str}")
                 # Return action name even if input fails parsing, let execution handle it
                 return thought, action_name, None 
            return thought, action_name, action_input_dict

        except Exception as e:
            self.logger.exception(f"{log_prefix} Error processing LLM response for ReAct: {e}")
            return None, None, None
            
    # --- Action Execution Helper (Moved from Agent) ---
    async def _execute_action(
        self, action_name: str, action_input: Optional[Dict[str, Any]], 
        log_prefix: str, shared_task_context: SharedTaskContext,
        allowed_skills: List[str] # Allowed skills for this step
    ) -> Dict[str, Any]:
        """Executes the chosen action (skill) and returns the observation."""
        if action_input is None:
             action_input = {} # Ensure action_input is a dict for normalization
             
        self.logger.info(
            f"{log_prefix} Executing Action: {action_name} with input: {action_input}"
        )

        if action_name not in self.tool_registry:
            # Handle missing skill (maybe invoke learning later?)
            self.logger.error(f"{log_prefix} Skill '{action_name}' not found in tool registry.")
            return {
                "status": "error", "action": f"{action_name}_not_found",
                "data": {"message": f"Skill '{action_name}' not found."}
            }
        try:
            normalized_action_input = normalize_action_input(action_name, action_input)
            if normalized_action_input != action_input:
                self.logger.info(f"{log_prefix} Action input normalized to: {normalized_action_input}")
            
            # Use self.tool_registry here
            if not isinstance(self.tool_registry, ToolRegistry):
                self.logger.error(f"{log_prefix} Invalid tool_registry type: {type(self.tool_registry)}")
                return {"status": "error", "action": "internal_error", "data": {"message": "Invalid tool registry configuration."}}
                
            # Create execution context
            exec_context = _ToolExecutionContext(
                logger=self.logger,
                workspace_root=self.workspace_root,
                llm_url=self.llm_interface.llm_url, # Assumes llm_interface has url
                tools_dict=self.tool_registry, # Pass ToolRegistry instance
                llm_interface=self.llm_interface,
                fragment_registry=self.fragment_registry,
                shared_task_context=shared_task_context,
                allowed_skills=allowed_skills, # Pass allowed_skills here
                skill_instance=None # execute_tool will fill this in
            )
            
            tool_result = await execute_tool(
                tool_name=action_name,
                action_input=normalized_action_input,
                tools_dict=self.tool_registry, # Pass ToolRegistry instance
                context=exec_context
            )
            self.logger.info(
                f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}"
            )
            return tool_result
        except Exception as tool_err:
            self.logger.exception(f"{log_prefix} Error executing tool '{action_name}':\"")
            return {
                "status": "error", "action": f"{action_name}_failed",
                "data": {"message": f"Error during tool execution: {tool_err}"},
            }

    # --- Orchestration Step Helper (Moved from Agent) ---
    async def _get_next_step_delegation(
        self, objective: str, history: List[Tuple[str, str]], shared_task_context: SharedTaskContext
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determines the next component (Fragment) and sub-task using an LLM call."""
        log_prefix = "[Orchestrator LLM]"
        self.logger.info(f"{log_prefix} Determining next step delegation for objective: {objective}")

        # 1. Build the Orchestrator Prompt Messages
        # Update fragment descriptions in case they changed (e.g., dynamic loading)
        # --- BEGIN DEBUG LOGGING ---
        all_defs = self.fragment_registry.get_all_definitions()
        self.logger.debug(f"{log_prefix} DEBUG: Available fragment def keys for prompt: {list(all_defs.keys())}")
        # --- END DEBUG LOGGING ---
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if "Error" in self.fragment_descriptions:
             self.logger.error(f"{log_prefix} Failed to get valid fragment descriptions for prompt: {self.fragment_descriptions}")
             # Use last known good or fallback? For now, error out.
             return None, None
             
        messages = await build_orchestrator_messages(
            objective=objective, 
            history=history, 
            fragment_descriptions=self.fragment_descriptions, 
            shared_task_context=shared_task_context
        )
        if not messages:
            self.logger.error(f"{log_prefix} Failed to build orchestrator prompt messages.")
            return None, None

        # 2. Call the LLM
        self.logger.debug(f"{log_prefix} Calling LLM with Orchestrator prompt...")
        llm_response_raw = ""
        try:
            # Await the call to get the async generator, then iterate
            llm_response_generator = self.llm_interface.call_llm(
                messages=messages,
                stream=False, # Expecting a single JSON response for delegation
                temperature=0.2,
                # response_format={"type": "json_object"} # If supported
            )
            async for chunk in llm_response_generator:
                 llm_response_raw += chunk
            
            if not llm_response_raw:
                self.logger.warning(f"{log_prefix} LLM call returned empty response for delegation.")
                return None, None
            
            self.logger.debug(f"{log_prefix} LLM Raw Response: {llm_response_raw}")

            # 3. Interpret the JSON Response
            try:
                json_match = re.search(r'{\s*.*\s*}', llm_response_raw, re.DOTALL)
                if not json_match:
                    self.logger.error(f"{log_prefix} Could not find valid JSON object {{...}} in LLM response. Response: {llm_response_raw}")
                    return None, None
                json_str = json_match.group(0)
                delegation_decision = json.loads(json_str)
                component_name = delegation_decision.get("component")
                sub_task = delegation_decision.get("sub_task")

                if not component_name or not sub_task:
                    self.logger.warning(f"{log_prefix} LLM response missing 'component' or 'sub_task'. Response: {json_str}")
                    return None, None
                
                # Validate component existence (optional but good)
                if component_name not in self.fragment_registry.get_all_definitions():
                    self.logger.warning(f"{log_prefix} LLM suggested unknown component '{component_name}'. Available: {list(self.fragment_registry.get_all_definitions().keys())}")
                    return None, None 

                self.logger.info(f"{log_prefix} Delegation decision: Component='{component_name}', Sub-task='{sub_task}'")
                return component_name, sub_task

            except json.JSONDecodeError as e:
                self.logger.error(f"{log_prefix} Failed to parse LLM JSON response: {e}. Response: {llm_response_raw}")
                return None, None
            except Exception as e:
                 self.logger.error(f"{log_prefix} Error processing delegation response: {e}. Response: {llm_response_raw}")
                 return None, None

        except TypeError as e:
            # Catch the specific error we saw
            self.logger.error(f"{log_prefix} TypeError calling LLM (check async generator handling?): {e}", exc_info=True)
            return None, None
        except Exception as e:
            self.logger.exception(f"{log_prefix} Error calling LLM for delegation: {e}")
            return None, None

    # --- Fragment Execution Helper (Moved from Agent) ---
    async def _execute_fragment_task(
        self,
        component_name: str,
        sub_task: str,
        objective: str,
        shared_task_context: SharedTaskContext,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Executes a task using the specified fragment or manager.
           Returns a tuple: (success_status: bool, result_dict: Dict)
        """
        log_prefix = f"[Fragment Runner: {component_name}]"
        start_time = time.time()
        self.logger.info(f"{log_prefix} Invoking execute on '{component_name}' for sub-task: {sub_task[:100]}...")
        success_status = False
        result_dict = {}

        try:
            fragment_instance = self.fragment_registry.get_fragment(component_name)
            if not fragment_instance:
                 raise ValueError(f"Fragment/Manager '{component_name}' could not be retrieved or instantiated.")

            # <<< ADDED: Register active fragment instance >>>
            await shared_task_context.register_active_fragment(component_name, fragment_instance)
            # <<< END ADDED >>>

            # >>> Create FragmentContext for the fragment call <<<
            fragment_context = FragmentContext(
                logger=self.logger,
                llm_interface=self.llm_interface,
                tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry,
                shared_task_context=shared_task_context,
                workspace_root=self.workspace_root,
                memory_manager=self.memory_manager # <<< RE-ADD memory_manager from self >>>
            )

            # --- Prepare arguments for the fragment's execute method ---
            call_kwargs = {}
            fragment_execute_sig = inspect.signature(fragment_instance.execute)
            fragment_params = fragment_execute_sig.parameters

            # Always pass the FragmentContext if the method expects 'context'
            if 'context' in fragment_params:
                param_type = fragment_params['context'].annotation
                if param_type == FragmentContext or param_type == inspect.Parameter.empty:
                     call_kwargs['context'] = fragment_context
                     self.logger.debug(f"Passing FragmentContext as 'context' to {component_name}")
                # Handle legacy _ToolExecutionContext or ctx for now? Or enforce FragmentContext?
                # For now, let's log a warning if it expects something else but pass FragmentContext
                elif param_type == _ToolExecutionContext:
                     self.logger.warning(f"Parameter 'context' in {component_name}.execute expects _ToolExecutionContext, but passing FragmentContext.")
                     call_kwargs['context'] = fragment_context # Still pass FragmentContext
                else:
                     self.logger.warning(f"Parameter 'context' in {component_name}.execute has unexpected type {param_type}. Passing FragmentContext anyway.")
                     call_kwargs['context'] = fragment_context
            # Handle legacy 'ctx' parameter - pass FragmentContext here too for standardization
            elif 'ctx' in fragment_params:
                 self.logger.warning(f"Parameter 'ctx' in {component_name}.execute is deprecated. Passing FragmentContext.")
                 call_kwargs['ctx'] = fragment_context # Pass FragmentContext even if named ctx
            else:
                 self.logger.warning(f"{component_name}.execute does not seem to accept 'context' or 'ctx'. Context will not be passed.")

            # Pass objective or sub_task based on expected parameters
            if 'objective' in fragment_params:
                call_kwargs['objective'] = objective
                self.logger.debug(f"Passing 'objective' to {component_name}")
            elif 'sub_task' in fragment_params:
                call_kwargs['sub_task'] = sub_task
                self.logger.debug(f"Passing 'sub_task' to {component_name}")
            # Handle final_answer specific case if needed (or rely on sub_task)
            elif 'answer' in fragment_params and component_name == 'FinalAnswerProvider':
                 call_kwargs['answer'] = sub_task # Map sub_task to answer for FinalAnswerProvider
                 self.logger.debug(f"Mapping 'sub_task' to 'answer' for {component_name}")
            # Add more specific argument mappings if other fragments have unique needs
            
            # Execute the fragment/manager task
            result_dict = await fragment_instance.execute(**call_kwargs)

            # Determine success based on fragment's result
            if isinstance(result_dict, Dict) and result_dict.get("status") == "success":
                success_status = True
            else:
                 # Log if result is not dict or status is not success
                 if not isinstance(result_dict, Dict):
                      self.logger.error(f"{log_prefix} Fragment returned non-dictionary result: {type(result_dict)}")
                 else:
                      self.logger.warning(f"{log_prefix} Fragment returned non-success status: {result_dict.get('status')}")
                 success_status = False # Explicitly set to False
                 # Ensure message exists in error case
                 if isinstance(result_dict, Dict) and "message" not in result_dict:
                      result_dict["message"] = f"Fragment {component_name} failed with status {result_dict.get('status')}"
                 elif not isinstance(result_dict, Dict):
                      result_dict = {"status": "error", "message": f"Fragment {component_name} returned invalid type {type(result_dict)}"}

            # >>> RETURN tuple (success_status, result_dict) <<<
            return success_status, result_dict 

        except Exception as e:
            self.logger.exception(f"{log_prefix} Exception during execution: {e}")
            # Prepare standard error dict
            error_result = {
                "status": "error",
                "message": f"Error during {component_name} execution: {e}",
                "fragment_error_details": traceback.format_exc()
            }
            # >>> RETURN tuple (False, error_result) in case of exception <<<
            return False, error_result 
        # Removed finally block for simplicity, relying on try/except returns

    # --- Learning Cycle Helper (Moved from Agent) ---
    async def _invoke_learning_cycle(self, objective: str, main_history: List, final_status: str, shared_context: SharedTaskContext):
        """Invokes a separate learning/reflection skill if available."""
        log_prefix = "[Learning Cycle Invoker]"
        self.logger.info(f"{log_prefix} Invoking learning cycle skill for objective: {objective[:100]}... Status: {final_status}")
        # Check if a specific learning skill exists using ToolRegistry method
        if "learning_cycle_skill" in self.tool_registry.list_tools(): # CORRECT: Check keys of the dict returned by list_tools
            try:
                learning_input = {
                    "objective": objective,
                    "final_status": final_status,
                    "full_history": main_history, # Pass the full history
                    # Pass necessary context data
                    "context_data": shared_context.get_all_data()
                }
                # Execute learning skill (fire and forget for now?)
                exec_context = _ToolExecutionContext(
                    logger=self.logger,
                    workspace_root=self.workspace_root,
                    llm_url=self.llm_interface.llm_url,
                    tools_dict=self.tool_registry,
                    llm_interface=self.llm_interface,
                    fragment_registry=self.fragment_registry,
                    shared_task_context=shared_context,
                    allowed_skills=list(self.tool_registry.keys()) # Allow all skills for learning?
                )
                await execute_tool(
                    tool_name="learning_cycle_skill", 
                    action_input=learning_input, 
                    tools_dict=self.tool_registry,
                    context=exec_context
                )
                self.logger.info(f"{log_prefix} Learning cycle skill executed.")
            except Exception as learn_err:
                    self.logger.error(f"{log_prefix} Error executing learning cycle skill: {learn_err}")
        # else:
            # self.logger.warning(f"{log_prefix} learning_cycle_skill not found in tools. Skipping.")

    # --- Main Orchestration Method (Logic from Agent.run_task) ---
    async def orchestrate(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Orchestrates task execution by delegating to Fragments."""
        log_prefix = "[TaskOrchestrator]"
        self.logger.info(f"{log_prefix} Starting task orchestration: {objective}")

        # --- Initialization ---
        task_id = str(uuid.uuid4())
        shared_task_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        self.logger.info(f"{log_prefix} Initialized SharedTaskContext with ID: {task_id}")
        notify_new_task(objective)

        # <<< ADDED: Start Chat Monitor Task >>>
        monitor_task = asyncio.create_task(
            chat_monitor_task(
                task_id=task_id,
                shared_task_context=shared_task_context,
                # Pass dependencies needed by the monitor and handler context
                llm_interface=self.llm_interface,
                tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry,
                memory_manager=self.memory_manager,
                workspace_root=self.workspace_root
            ),
            name=f"ChatMonitor-{task_id[:8]}" # Name the task for debugging
        )
        self.logger.info(f"{log_prefix} Started Chat Monitor task.")
        # <<< END ADDED >>>

        full_task_history: List[Dict[str, str]] = [] # Stores detailed steps from fragments
        orchestration_history: List[Tuple[str, str]] = [] # High-level orchestrator decisions
        current_step = 0
        max_orchestration_steps = max_steps or 20 # Define a max loop count
        self.logger.info(f"{log_prefix} Maximum orchestration steps allowed: {max_orchestration_steps}")
        final_result: Dict = {}
        final_answer_received = False # Flag to indicate final answer processed

        # --- Orchestration Loop ---
        # Loop continues as long as max steps not reached AND final answer not received
        while current_step < max_orchestration_steps and not final_answer_received:
            current_step += 1
            self.logger.info(f"{log_prefix} Orchestration step {current_step}/{max_orchestration_steps}")

            # <<< ADDED: Check for forced step >>>
            if hasattr(self, '_next_forced_component') and self._next_forced_component:
                component_name = self._next_forced_component
                sub_task = self._next_forced_sub_task
                self.logger.info(f"{log_prefix} Using forced step: Component={component_name}, Sub-task={sub_task[:50]}...")
                # Clear the forced step flags
                self._next_forced_component = None
                self._next_forced_sub_task = None
            else:
                # <<< Original Logic: Determine Component and Sub-task >>>
                component_name, sub_task = await self._get_next_step_delegation(
                    objective=objective, 
                    history=orchestration_history, # Pass high-level history for decisions
                    shared_task_context=shared_task_context
                )
            # <<< END ADDED Check >>>

            if not component_name or not sub_task:
                error_msg = "Failed to get delegation decision from LLM."
                self.logger.error(f"{log_prefix} {error_msg}")
                final_result = {
                    "status": "error", 
                    "message": error_msg,
                    "final_answer": error_msg, # Provide error as final answer
                    "shared_task_context": shared_task_context
                }
                notify_task_error(task_id, "Delegation Failed", final_result)
                break # Exit loop on delegation failure

            # Add delegation decision to high-level history
            orchestration_history.append((f"Step {current_step}: Delegate to {component_name}", sub_task))
            notify_fragment_selection(component_name)

            # 2. Execute the Fragment Task
            fragment_success, fragment_result = await self._execute_fragment_task(
                component_name, 
                sub_task, 
                objective,
                shared_task_context
            )
            
            # Add fragment's detailed history to the main history
            if isinstance(fragment_result, dict) and "full_history" in fragment_result:
                 self.logger.info(f"{log_prefix} Appending {len(fragment_result['full_history'])} items from {component_name} history.")
                 full_task_history.extend(fragment_result["full_history"])
            elif isinstance(fragment_result, dict) and "history" in fragment_result: # Fallback for older structure
                 self.logger.warning(f"{log_prefix} Fragment {component_name} returned 'history' instead of 'full_history'. Appending anyway.")
                 full_task_history.extend(fragment_result["history"])

            # <<< ADDED: Check for replan request status >>>
            if isinstance(fragment_result, dict) and fragment_result.get("status") == "request_replan":
                self.logger.warning(f"{log_prefix} Received 'request_replan' status from {component_name}. Triggering replan.")
                # Prepare for the next iteration to call the planner
                original_component = component_name # Remember who requested it
                reason = fragment_result.get("message", "No reason provided")
                component_name = "PlannerFragment" # Force planner next
                sub_task = f"Replan task. Triggered by: {original_component}. Reason: {reason}. Original objective: {objective}"
                # Add history entry for this decision
                orchestration_history.append((f"Step {current_step}: Replan Requested", f"Received 'request_replan' from {original_component}. Forcing call to {component_name}."))
                # Store the result that requested the replan (might be useful context for planner)
                await shared_task_context.update_data("last_execution_result", fragment_result) 
                # Continue to the next loop iteration, which will now execute the PlannerFragment
                # We need to reset component_name and sub_task *for the next loop's delegation call* if we don't break here
                # Let's just set the component_name/sub_task and let the *next* iteration's _get_next_step_delegation be skipped implicitly?
                # No, the current logic runs _get_next_step_delegation *before* executing. We need to adjust the flow slightly.
                # Let's modify the logic: If replan is requested, we set the *next* target and continue.
                # The next iteration's _get_next_step_delegation will then be skipped or overridden.

                # *** Revised approach: Set flags for next iteration ***
                self._next_forced_component = "PlannerFragment"
                self._next_forced_sub_task = sub_task
                self.logger.info(f"{log_prefix} Setting forced next step to PlannerFragment for replan.")
                continue # Continue loop, next iteration will use the forced step.
            # <<< END ADDED >>>

            # 3. Process Fragment Result (if not replan request)
            if fragment_success and isinstance(fragment_result, dict) and fragment_result.get("final_answer"):
                # Fragment provided the final answer for the *overall objective*
                self.logger.info(f"{log_prefix} Final answer received from {component_name}: {fragment_result['final_answer']}")
                final_result = {
                    "status": "success",
                    "message": f"Orchestration completed by {component_name}.",
                    "final_answer": fragment_result["final_answer"],
                    "full_history": full_task_history, # Include full history
                    "shared_task_context": shared_task_context
                }
                final_answer_received = True # Set the flag
                notify_task_completion(final_result.get('final_answer', 'Task completed successfully.'))
                break # Exit loop, task complete
            elif not fragment_success:
                 # Fragment execution failed
                 error_msg = fragment_result.get("message", f"Fragment {component_name} failed without a specific message.")
                 self.logger.error(f"{log_prefix} Fragment {component_name} execution failed: {error_msg}")
                 final_result = {
                     "status": "error", 
                     "message": f"Error during {component_name} execution: {error_msg}",
                     "final_answer": f"Error during {component_name} execution: {error_msg}",
                     "full_history": full_task_history,
                     "shared_task_context": shared_task_context
                 }
                 notify_task_error(final_result.get('message', 'Unknown fragment error'), source=component_name, error_details=final_result)
                 break # Exit loop on fragment failure
            else:
                 # Fragment completed its sub-task successfully, but no final answer yet.
                 # Update shared context if necessary (result might contain updates)
                 if isinstance(fragment_result, dict) and fragment_result.get("context_updates"):
                      # await shared_task_context.update_data(fragment_result["context_updates"]) # This is likely wrong - update_data expects key, value
                      # Instead, update individual keys if specified, or store the whole result?
                      # For now, let's explicitly store the last result for the prompt builder
                      self.logger.warning(f"{log_prefix} 'context_updates' key found in result from {component_name}, but logic not fully implemented.")
                      # Fall through to store the main result below

                 # >>> ADDED: Store the successful intermediate result <<< 
                 await shared_task_context.update_data("last_execution_result", fragment_result)
                 # Optionally, update last file written/read based on result content
                 if isinstance(fragment_result, dict):
                     action = fragment_result.get("action")
                     data = fragment_result.get("data", {})
                     if action == "file_written" and isinstance(data, dict) and "filename" in data:
                         await shared_task_context.update_data("last_file_written_path", data["filename"])
                     elif action == "file_read" and isinstance(data, dict) and "filepath" in data:
                         await shared_task_context.update_data("last_file_read_path", data["filepath"])
                 # <<< END ADDED >>>

                 self.logger.info(f"{log_prefix} {component_name} completed sub-task. Continuing orchestration.")
                 # Loop continues for the next delegation decision

        # --- End of Orchestration Loop ---

        # Handle loop exit due to max steps
        if not final_result and current_step >= max_orchestration_steps:
            self.logger.warning(f"{log_prefix} Task exceeded max steps ({max_orchestration_steps}). Objective: {objective}")
            final_result = {
                "status": "error", 
                "message": f"Task exceeded maximum steps ({max_orchestration_steps}).",
                "final_answer": f"Task exceeded maximum steps ({max_orchestration_steps}).", # Provide error as final answer
                "full_history": full_task_history,
                "shared_task_context": shared_task_context
            }
            ep_context = f"Orchestrator reached max steps ({max_orchestration_steps}) for objective: {objective}"
            ep_action = "orchestrator_max_steps_reached"
            ep_outcome = json.dumps({"last_history": full_task_history[-3:]})
            ep_metadata = {"status": "max_steps_error"}
            add_episodic_record(ep_context, ep_action, ep_outcome, ep_metadata)
            notify_task_error(final_result.get('message', 'Max steps exceeded'), source="Orchestrator", error_details=final_result)

        # Ensure some result is set if loop finished unexpectedly
        if not final_result:
             self.logger.error(f"{log_prefix} Orchestration loop finished without a definitive result. Objective: {objective}")
             final_result = {
                 "status": "error", 
                 "message": "Orchestration finished without a result.",
                 "final_answer": "Orchestration finished without a result.", # Provide error as final answer
                 "full_history": full_task_history,
                 "shared_task_context": shared_task_context
            }
             ep_context = f"Orchestrator finished unknown state for objective: {objective}"
             ep_action = "orchestrator_unknown_end_state"
             ep_outcome = json.dumps({"last_history": full_task_history[-3:]})
             ep_metadata = {"status": "unknown_error"}
             add_episodic_record(ep_context, ep_action, ep_outcome, ep_metadata)
             notify_task_error(final_result.get('message', 'Unknown orchestration error'), source="Orchestrator", error_details=final_result)

        # --- Finalization ---
        # <<< ADDED: Ensure Chat Monitor is stopped >>>
        if 'monitor_task' in locals() and not monitor_task.done():
            self.logger.info(f"{log_prefix} Stopping Chat Monitor task...")
            monitor_task.cancel()
            try:
                await asyncio.gather(monitor_task, return_exceptions=True)
                self.logger.info(f"{log_prefix} Chat Monitor task stopped.")
            except asyncio.CancelledError:
                 self.logger.info(f"{log_prefix} Chat Monitor task already cancelled.")
            # Clear active fragments for this task (optional cleanup)
            if shared_task_context:
                 active_names = await shared_task_context.get_all_active_fragment_names()
                 for name in active_names:
                      await shared_task_context.unregister_active_fragment(name)
                 self.logger.debug(f"{log_prefix} Cleared active fragments for task {task_id}.")
        # <<< END ADDED >>>
        
        self.logger.info(f"{log_prefix} Recorded final episodic memory state for task {task_id}")
        await self._invoke_learning_cycle(objective, full_task_history, final_result.get("status", "unknown"), shared_task_context)

        # Ensure full history is in the final result
        final_result["full_history"] = full_task_history

        self.logger.info(f"{log_prefix} Orchestration finished for objective '{objective}'. Final Status: {final_result.get('status')}. Result Keys: {list(final_result.keys())}")
        return final_result 