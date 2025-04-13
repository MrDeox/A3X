import logging
import json
import re
import uuid
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple
from pathlib import Path

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
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext
from a3x.core.llm_interface import LLMInterface
from a3x.core.fragment_registry import FragmentRegistry
from a3x.core.context import SharedTaskContext
from a3x.core.utils.param_normalizer import normalize_action_input
# Assume get_skills_for_fragment is defined elsewhere or passed if needed
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


class TaskOrchestrator:
    """Handles the step-by-step orchestration of a task using Fragments and Skills."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        fragment_registry: FragmentRegistry,
        tools: Dict, # Skill registry
        workspace_root: Path,
        agent_logger: logging.Logger,
        # Potentially add agent_id if needed for state saving?
    ):
        self.llm_interface = llm_interface
        self.fragment_registry = fragment_registry
        self.tools = tools
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

            thought, action_name, action_input_str = parse_llm_response(llm_response_raw)
            
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
        self, action_name: str, action_input: Optional[Dict[str, Any]], log_prefix: str, shared_task_context: SharedTaskContext
    ) -> Dict[str, Any]:
        """Executes the chosen action (skill) and returns the observation."""
        if action_input is None:
             action_input = {} # Ensure action_input is a dict for normalization
             
        self.logger.info(
            f"{log_prefix} Executing Action: {action_name} with input: {action_input}"
        )

        if action_name not in self.tools:
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
            
            # Create execution context
            exec_context = _ToolExecutionContext(
                logger=self.logger,
                workspace_root=self.workspace_root,
                llm_url=self.llm_interface.llm_url, # Assumes llm_interface has url
                tools_dict=self.tools,
                llm_interface=self.llm_interface,
                fragment_registry=self.fragment_registry,
                shared_task_context=shared_task_context
            )
            
            tool_result = await execute_tool(
                tool_name=action_name,
                action_input=normalized_action_input,
                tools_dict=self.tools,
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
    async def _get_next_step_delegation(self, objective: str, history: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
        """Determines the next component (Fragment) and sub-task using an LLM call."""
        log_prefix = "[Orchestrator LLM]"
        self.logger.info(f"{log_prefix} Determining next step delegation for objective: {objective}")

        # 1. Build the Orchestrator Prompt Messages
        # Update fragment descriptions in case they changed (e.g., dynamic loading)
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if "Error" in self.fragment_descriptions:
             self.logger.error(f"{log_prefix} Failed to get valid fragment descriptions for prompt: {self.fragment_descriptions}")
             # Use last known good or fallback? For now, error out.
             return None, None
             
        messages = build_orchestrator_messages(
            objective, history, self.fragment_descriptions
        )
        if not messages:
            self.logger.error(f"{log_prefix} Failed to build orchestrator prompt messages.")
            return None, None

        # 2. Call the LLM
        self.logger.debug(f"{log_prefix} Calling LLM with Orchestrator prompt...")
        llm_response_raw = ""
        try:
            async for chunk in self.llm_interface.call_llm(
                messages=messages,
                stream=False,
                temperature=0.2,
                # response_format={"type": "json_object"} # If supported
            ):
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

        except Exception as e:
            self.logger.exception(f"{log_prefix} Error calling LLM for delegation: {e}")
            return None, None

    # --- Fragment Execution Helper (Moved from Agent) ---
    async def _execute_fragment_task(
        self, 
        fragment_name: str, 
        sub_task: str, 
        allowed_skills: List[str],
        # parent_history: List[Tuple[str, str]], # No longer needed? Fragment manages own history
        shared_task_context: SharedTaskContext
    ) -> Dict:
        """
        Executes the ReAct cycle for a specific Fragment worker until the sub-task is completed.
        Uses only the allowed_skills for this fragment.
        Returns a dict with status and final answer/observation.
        """
        log_prefix = f"[{fragment_name} Fragment]"
        self.logger.info(f"{log_prefix} Starting execution for sub-task: '{sub_task}' with skills: {allowed_skills}")
        
        current_task_history: List[Tuple[str, str]] = [] # History *within* this sub-task
        iterations = 0
        max_fragment_iterations = 10 # Limit iterations *per fragment task* 

        while iterations < max_fragment_iterations:
            iterations += 1
            self.logger.info(f"{log_prefix} Iteration {iterations}/{max_fragment_iterations}")

            # Build prompt for the worker fragment using its focused history
            worker_messages = build_worker_messages(
                sub_task, current_task_history, allowed_skills, self.tools
            )
            
            self.logger.info(f"{log_prefix} Calling LLM...")
            parsed_result = await self._process_llm_response(worker_messages, log_prefix=log_prefix)
            
            if not parsed_result:
                 error_msg = "LLM processing failed completely."
                 self.logger.error(f"{log_prefix} Error processing LLM response: {error_msg}")
                 return {"status": "error", "message": error_msg, "fragment_history": current_task_history}
            
            thought, action, action_input = parsed_result

            if thought:
                self.logger.info(f"{log_prefix} Thought: {thought}")
            else:
                self.logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")

            if not action:
                self.logger.error(f"{log_prefix} No Action specified by LLM.")
                observation = {"status": "error", "message": "LLM Fragment did not specify an action."}
                action_str_for_history = "Error: No Action"
            else: 
                action_str_for_history = action 
                if action not in allowed_skills:
                    self.logger.error(f"{log_prefix} Action '{action}' is NOT in allowed skills {allowed_skills}!")
                    observation = {"status": "error", "message": f"Action '{action}' is not allowed for this Fragment."}
                    action_str_for_history = f"Error: Disallowed Action ({action})"
                elif action not in self.tools:
                    self.logger.error(f"{log_prefix} Action '{action}' not found in skill registry.")
                    observation = {"status": "error", "message": f"Skill '{action}' not found."}
                    action_str_for_history = f"Error: Unknown Skill ({action})"
                else:
                    self.logger.info(f"{log_prefix} Action ({fragment_name}): {action}({json.dumps(action_input)}) ")
                    try:
                        observation = await self._execute_action(
                            action, action_input, log_prefix, shared_task_context
                        )
                        self.logger.info(f"{log_prefix} Observation: {str(observation)[:500]}...")
                    except Exception as e:
                        self.logger.error(f"{log_prefix} Error executing action '{action}': {e}", exc_info=True)
                        observation = {"status": "error", "message": f"Error executing action '{action}': {e}"}
                        action_str_for_history = f"Error: Execution Failed ({action})"
            
            current_task_history.append((action_str_for_history, observation))

            # --- Log step to Episodic Memory DB --- 
            try:
                context_str = f"Sub-task: {sub_task} | History: {str(current_task_history[:-1])[-200:]}"
                action_str = action_str_for_history 
                outcome_str = json.dumps(observation, ensure_ascii=False) 
                metadata = {
                    "fragment": fragment_name, "thought": thought, "action": action, 
                    "action_input": action_input, "iteration": iterations
                }
                add_episodic_record(context_str, action_str, outcome_str, metadata)
                self.logger.debug(f"{log_prefix} Step logged to episodic memory.")
            except Exception as db_err:
                 self.logger.error(f"{log_prefix} Failed to log step to episodic memory: {db_err}")

            # --- Check for Sub-Task Completion --- 
            if action == "final_answer":
                 final_answer_content = action_input.get("answer", "(No answer content provided)") if isinstance(action_input, dict) else "(Answer format unexpected)"
                 self.logger.info(f"{log_prefix} Sub-task '{sub_task}' completed with final answer.")
                 return {
                     "status": "success", "final_answer": final_answer_content, 
                     "observation": observation, "fragment_history": current_task_history
                 }
            
            if isinstance(observation, dict) and observation.get("status") == "error":
                 self.logger.warning(f"{log_prefix} An error occurred during action execution or validation. Stopping fragment task.")
                 return {
                      "status": "error", 
                      "message": observation.get("message", "Error during action execution/validation"),
                      "fragment_history": current_task_history
                 }

        self.logger.warning(f"{log_prefix} Max iterations ({max_fragment_iterations}) reached for sub-task '{sub_task}' without completion.")
        return {
             "status": "error", "message": f"Max iterations reached for sub-task", 
             "fragment_history": current_task_history
        }

    # --- Learning Cycle Helper (Moved from Agent) ---
    async def _invoke_learning_cycle(self, objective: str, main_history: List, final_status: str, shared_context: SharedTaskContext):
        """Invokes the learning cycle skill with the final task state."""
        log_prefix = "[Learning Cycle Invoker]"
        try:
            self.logger.info(f"{log_prefix} Invoking learning cycle skill for objective: {objective[:100]}... Status: {final_status}")
            
            # Note: The learning_cycle_skill needs to be in self.tools
            if "learning_cycle_skill" not in self.tools:
                 self.logger.warning(f"{log_prefix} learning_cycle_skill not found in tools. Skipping.")
                 return

            # Execute the skill using the internal helper
            await self._execute_action(
                action_name="learning_cycle_skill", 
                action_input={
                    "objective": objective,
                    "plan": [], # Plan needs to be captured if used
                    "execution_results": main_history, 
                    "final_status": final_status,
                    "agent_tools": self.tools, # Pass available tools
                    "agent_workspace": str(self.workspace_root.resolve()),
                    "agent_llm_url": self.llm_interface.llm_url, # Pass LLM URL
                    "shared_task_context": shared_context # Pass context
                },
                log_prefix=log_prefix, 
                shared_task_context=shared_context 
            )
            self.logger.info(f"{log_prefix} Learning cycle skill executed.")
        except Exception as learn_err:
            self.logger.error(f"{log_prefix} Error executing learning cycle: {learn_err}", exc_info=True)

    # --- Main Orchestration Method (Logic from Agent.run_task) ---
    async def orchestrate(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Orchestrates Fragments to achieve the overall objective."""
        log_prefix = "[TaskOrchestrator]"
        self.logger.info(f"{log_prefix} Starting task orchestration: {objective}")
        
        task_id = str(uuid.uuid4())
        shared_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        self.logger.info(f"{log_prefix} Initialized SharedTaskContext with ID: {task_id}")

        main_history: List[Tuple[str, str]] = [] 
        orchestrator_iterations = 0
        max_orchestrator_iterations = max_steps if max_steps is not None else 20 
        self.logger.info(f"{log_prefix} Maximum orchestration steps allowed: {max_orchestrator_iterations}")

        last_failed_sub_task: Optional[str] = None
        consecutive_failures: int = 0
        MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG: int = 2 
        failure_history_for_debugger: List[Dict] = []
        
        final_result = None 
        final_status = "unknown"

        while orchestrator_iterations < max_orchestrator_iterations:
            orchestrator_iterations += 1
            self.logger.info(f"{log_prefix} Orchestration Cycle {orchestrator_iterations}/{max_orchestrator_iterations}")

            # Use the moved method
            chosen_fragment, sub_task = await self._get_next_step_delegation(objective, main_history)

            if not chosen_fragment or not sub_task:
                self.logger.error(f"{log_prefix} Orchestrator failed to delegate. Aborting task.")
                final_status = "error_orchestrator_delegation"
                final_result = {"status": "error", "message": "Orchestrator failed to delegate next step."}
                break 

            if sub_task != last_failed_sub_task:
                self.logger.debug(f"{log_prefix} New sub-task '{sub_task}'. Resetting consecutive failure count.")
                last_failed_sub_task = sub_task
                consecutive_failures = 0
                failure_history_for_debugger = [] # Reset history for debugger too

            if chosen_fragment == "FinalAnswerProvider":
                self.logger.info(f"{log_prefix} Orchestrator chose FinalAnswerProvider. Task complete.")
                final_status = "completed"
                # Execute final_answer skill to potentially format the answer nicely
                final_answer_observation = await self._execute_action(
                     "final_answer", {"answer": sub_task}, log_prefix, shared_context
                )
                final_result = {"status": "success", "final_answer": sub_task, "observation": final_answer_observation}
                main_history.append(("FinalAnswerProvider", final_answer_observation))
                break 
            
            fragment_def = self.fragment_registry.get_fragment_definition(chosen_fragment)
            allowed_skills = []
            if fragment_def:
                 # Use combined list of managed_skills and skills for the worker
                 allowed_skills = (fragment_def.managed_skills or []) + (fragment_def.skills or [])
                 # Ensure final_answer is always allowed if not present? Or rely on definition?
                 # For now, rely on definition. Add final_answer if needed.
                 if "final_answer" not in allowed_skills:
                     allowed_skills.append("final_answer") # Ensure fragments can finish
                 self.logger.info(f"{log_prefix} Retrieved allowed skills for {chosen_fragment}: {allowed_skills}")
            else:
                 self.logger.error(f"{log_prefix} Could not find definition for fragment '{chosen_fragment}'. Aborting task.")
                 final_status = "error_fragment_definition_not_found"
                 final_result = {"status": "error", "message": f"Fragment definition for {chosen_fragment} not found in registry."}
                 break
            
            # Use the moved method
            fragment_result = await self._execute_fragment_task(
                fragment_name=chosen_fragment,
                sub_task=sub_task,
                allowed_skills=list(set(allowed_skills)), # Ensure unique skills
                shared_task_context=shared_context
            )
            
            fragment_status = fragment_result.get("status")
            fragment_final_answer = fragment_result.get("final_answer")
            fragment_message = fragment_result.get("message")
            internal_history = fragment_result.get("fragment_history", [])
            
            # Append summary to main history
            summary_action = f"Fragment {chosen_fragment} completed sub-task: '{sub_task}'" if fragment_status == "success" else f"Fragment {chosen_fragment} failed sub-task: '{sub_task}'"
            summary_observation = { 
                "status": fragment_status,
                "sub_task_result": fragment_final_answer or fragment_message,
                # Optionally include fragment_history summary? Keep it simple for now.
            }
            main_history.append((summary_action, summary_observation))

            if fragment_status == "error":
                self.logger.error(f"{log_prefix} Fragment {chosen_fragment} failed sub-task '{sub_task}'. Message: {fragment_message}")
                consecutive_failures += 1
                if internal_history: 
                    failure_history_for_debugger.extend(internal_history) 
                self.logger.warning(f"{log_prefix} Consecutive failures for sub-task '{sub_task}': {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG}")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG:
                    self.logger.warning(f"{log_prefix} Failure limit reached. Attempting to trigger DebuggerFragment...")
                    # Check if DebuggerFragment exists and is allowed (or bypass?)
                    if "DebuggerFragment" in self.fragment_registry.get_all_definitions():
                        debugger_sub_task = f"Analyze the following failure history for sub-task '{sub_task}' and suggest a fix or next step: {json.dumps(failure_history_for_debugger[-5:])}" # Pass recent failure history
                        debugger_allowed_skills = get_skills_for_fragment("DebuggerFragment") # Get its allowed skills
                        
                        debug_result = await self._execute_fragment_task(
                             fragment_name="DebuggerFragment",
                             sub_task=debugger_sub_task,
                             allowed_skills=debugger_allowed_skills,
                             shared_task_context=shared_context
                        )
                        # Append debugger result to main history
                        debug_summary_action = f"DebuggerFragment attempted analysis for failed sub-task: '{sub_task}'"
                        debug_summary_observation = {
                             "status": debug_result.get("status"),
                             "debugger_output": debug_result.get("final_answer") or debug_result.get("message")
                        }
                        main_history.append((debug_summary_action, debug_summary_observation))
                        self.logger.info(f"{log_prefix} DebuggerFragment finished. Result: {debug_summary_observation['status']}")
                        # Reset failure count after debugger runs, let orchestrator decide next
                        consecutive_failures = 0
                        failure_history_for_debugger = []
                    else:
                        self.logger.error(f"{log_prefix} DebuggerFragment not found or available. Aborting task after repeated failures.")
                        final_status = "error_consecutive_failures_no_debugger"
                        final_result = {"status": "error", "message": f"Task failed after {consecutive_failures} consecutive errors on sub-task '{sub_task}' and DebuggerFragment was unavailable."}
                        break # Exit loop
            else: # fragment_status == "success"
                if sub_task == last_failed_sub_task: # Check if the previously failed task succeeded now
                     self.logger.debug(f"{log_prefix} Sub-task '{sub_task}' succeeded after previous failure. Resetting failure count.")
                     consecutive_failures = 0
                     failure_history_for_debugger = []
                self.logger.info(f"{log_prefix} Fragment {chosen_fragment} completed sub-task successfully.")


        # === Loop End / Task Conclusion ===
        if final_result is None: 
            self.logger.warning(f"{log_prefix} Max orchestrator iterations ({max_orchestrator_iterations}) reached or loop exited without success for '{objective}'.")
            final_status = "error_max_iterations_or_failed"
            final_result = {"status": "error", "message": "Max orchestrator iterations reached or task failed to complete."} 
        
        # --- Invoke Learning Cycle --- 
        # Use the moved method
        await self._invoke_learning_cycle(objective, main_history, final_status, shared_context)

        # --- Return Final Result (and potentially history if needed) --- 
        self.logger.info(f"{log_prefix} Orchestration finished for objective '{objective}'. Final Status: {final_status}. Result: {final_result}")
        # Return just the result dict, history is logged via DB/learning cycle
        return final_result 