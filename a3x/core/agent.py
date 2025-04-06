import logging
import json
import os
from typing import Dict, Any, List, AsyncGenerator, Optional

# Package imports
from a3x.core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS
from a3x.core.tools import get_tool_descriptions, get_skill_registry
from a3x.core.db_utils import save_agent_state, load_agent_state
from a3x.core.prompt_builder import build_react_prompt
from a3x.core.agent_parser import parse_llm_response
from a3x.core.history_manager import trim_history
from a3x.core.tool_executor import execute_tool
from a3x.core.llm_interface import call_llm
from a3x.core.planner import generate_plan

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente
AGENT_STATE_ID = 1


# --- Helper for Simple Task Detection ---
def _is_simple_list_files_task(objective: str) -> bool:
    """Checks if the objective is likely a simple request to list files."""
    # Simple keyword check for now, can be improved
    objective_lower = objective.lower().strip()
    keywords = ["liste", "listar", "lista", "mostre", "arquivos", "diretório", "pasta"]
    # Check if it contains list keywords and not complex actions like "read and list" or "delete"
    if (
        any(kw in objective_lower for kw in keywords)
        and "ler" not in objective_lower
        and "conteúdo" not in objective_lower
        and "criar" not in objective_lower
        and "deletar" not in objective_lower
        and "apagar" not in objective_lower
        and "executar" not in objective_lower
    ):
        # Basic check, assumes simple listing if these keywords are present
        # And keywords for other actions are absent.
        # A more robust check might involve simple NLP or regex.
        return True
    return False


# --- Classe ReactAgent ---
class ReactAgent:
    def __init__(self, system_prompt: str, llm_url: Optional[str] = None):
        """Inicializa o Agente ReAct."""
        self.llm_url = llm_url
        self.system_prompt = system_prompt
        self.tools = get_skill_registry()
        self._history = []  # Histórico de Thought, Action, Observation
        self.agent_id = "1"  # TODO: Make agent ID configurable/dynamic
        self.agent_logger = agent_logger
        self._memory: Dict[str, Any] = {}
        self.llm_url = llm_url or os.getenv(
            "LLM_API_URL"
        )  # Use env var if not provided

        # Load agent state if exists
        # <<< FIX: Use string "1" for agent_id >>>
        loaded_state = load_agent_state(agent_id="1")
        if loaded_state:
            self._history = loaded_state.get("history", [])
            self._memory = loaded_state.get("memory", {})
            agent_logger.info(
                f"[ReactAgent INIT] Estado do agente carregado para ID '1'. Memória: {list(self._memory.keys())}"
            )
        self.max_iterations = MAX_REACT_ITERATIONS
        self._current_plan = None  # <<< Initialize plan >>>
        agent_logger.info(
            f"[ReactAgent INIT] Agente inicializado. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}"
        )

    # <<< NEW: Method for Plan Generation >>>
    async def _generate_plan(self, objective: str) -> List[str]:
        """Gera um plano de execução para o objetivo dado."""
        agent_logger.info("--- Generating Plan ---")
        plan_to_execute: List[str] = []

        if _is_simple_list_files_task(objective):
            agent_logger.info(
                "[Planner] Detected simple list_files task. Skipping complex planning."
            )
            plan_to_execute = [
                f"Use the list_files tool for the objective: '{objective}'",
                "Use the final_answer tool to provide the list of files.",
            ]
            plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
            agent_logger.info(f"Simple Plan Generated:\n{plan_str}")
        else:
            tool_desc = get_tool_descriptions()
            generated_plan = await generate_plan(
                objective, tool_desc, agent_logger, self.llm_url
            )
            if generated_plan:
                plan_to_execute = generated_plan
                plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
                agent_logger.info(f"Plan Generated:\n{plan_str}")
            else:
                agent_logger.warning(
                    "Failed to generate a plan. Proceeding with objective as single step."
                )
                plan_to_execute = [objective]  # Fallback to objective as the only step
        return plan_to_execute

    # <<< NEW: Method to Call LLM and Parse Response >>>
    async def _process_llm_response(
        self, prompt: List[Dict[str, str]], log_prefix: str
    ) -> Optional[Dict[str, Any]]:
        """Chama o LLM com o prompt, parseia a resposta e retorna um dicionário estruturado ou None em caso de erro fatal."""
        agent_logger.info(f"{log_prefix} Calling LLM...")
        llm_response_raw = ""
        try:
            # Use call_llm (non-streaming for react cycle response)
            # <<< REVERT: Use async for as call_llm likely always returns async generator >>>
            # llm_response_raw = await call_llm(prompt, llm_url=self.llm_url, stream=False) # OLD CORRECTION
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                llm_response_raw += chunk
            agent_logger.info(f"{log_prefix} LLM Response received.")
            agent_logger.debug(f"{log_prefix} Raw LLM Response:\\n{llm_response_raw}")

            # Parse the response
            parsed_output_tuple = parse_llm_response(llm_response_raw, agent_logger)
            if parsed_output_tuple is None:
                agent_logger.error(
                    f"{log_prefix} Failed to parse LLM response (parse_llm_response returned None). Raw: '{llm_response_raw[:100]}...'"
                )
                # Return an error structure instead of None for consistency
                return {
                    "type": "error",
                    "content": "Failed to parse LLM response (parser returned None).",
                }

            thought, action_name, action_input = parsed_output_tuple
            parsed_output = {}
            if thought:
                parsed_output["thought"] = thought
            if action_name:
                parsed_output["action_name"] = action_name
            parsed_output["action_input"] = (
                action_input if action_input is not None else {}
            )
            return parsed_output  # Return structured dictionary

        except json.JSONDecodeError as parse_err:
            agent_logger.error(
                f"{log_prefix} Failed to parse LLM response (JSONDecodeError). Raw: '{llm_response_raw[:100]}...'"
            )
            agent_logger.exception(f"{log_prefix} JSON Parsing Traceback:")
            return {
                "type": "error",
                "content": f"Failed to parse LLM response: {parse_err}",
            }
        except (
            Exception
        ) as llm_err:  # Catch other errors like connection issues during call_llm or general parsing
            agent_logger.exception(f"{log_prefix} Error during LLM call or processing:")
            return {
                "type": "error",
                "content": f"Failed to get or process LLM response: {llm_err}",
            }

    # <<< NEW: Method to Execute Action >>>
    async def _execute_action(
        self, action_name: str, action_input: Dict[str, Any], log_prefix: str
    ) -> Dict[str, Any]:
        """Executa a ferramenta especificada com os inputs fornecidos."""
        agent_logger.info(
            f"{log_prefix} Executing Action: {action_name} with input: {action_input}"
        )
        try:
            # <<< CORRECTED CALL: Pass arguments in the correct order >>>
            tool_result = await execute_tool(
                tool_name=action_name,
                action_input=action_input,
                tools_dict=self.tools,  # Pass self.tools
                agent_logger=agent_logger,  # Pass the module logger
                agent_memory=self._memory,  # Pass agent memory
            )
            agent_logger.info(
                f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}"
            )
            return tool_result
        except Exception as tool_err:
            agent_logger.exception(
                f"{log_prefix} Error executing tool '{action_name}':"
            )
            return {
                "status": "error",
                "action": f"{action_name}_failed",
                "data": {"message": f"Error during tool execution: {tool_err}"},
            }

    # <<< NEW: Method to Handle Observation >>>
    def _handle_observation(
        self, observation_data: Dict[str, Any], log_prefix: str
    ) -> str:
        """Processa os dados da observação, formata para o histórico e log."""
        try:
            # Format observation for history (compact JSON or string fallback)
            observation_content = json.dumps(observation_data, ensure_ascii=False)
            agent_logger.info(
                f"{log_prefix} Observation: {observation_content[:150]}..."
            )
            self._history.append(f"Observation: {observation_content}")
            return observation_content  # Return the string version for potential reflection
        except TypeError:
            # Fallback if data is not JSON serializable
            observation_content = str(observation_data)
            agent_logger.warning(
                f"{log_prefix} Observation data not JSON serializable. Using str()."
            )
            agent_logger.info(
                f"{log_prefix} Observation (str): {observation_content[:150]}..."
            )
            self._history.append(f"Observation: {observation_content}")
            return observation_content
        except Exception as obs_err:
            agent_logger.exception(f"{log_prefix} Error handling observation:")
            error_content = f"Error processing observation: {obs_err}"
            self._history.append(
                f"Observation: {{'status': 'error', 'message': '{error_content}'}}"
            )
            return error_content

    # <<< NEW: Extracted ReAct Iteration Logic >>>
    async def _perform_react_iteration(
        self, step_objective: str, log_prefix: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Performs a single Thought-Action-Observation cycle for a given step objective.

        Args:
            step_objective (str): The objective for the current step.
            log_prefix (str): Prefix for logging messages.

        Yields:
            Dict[str, Any]: Dictionaries for thought, action, observation, or error.
        """
        # Trim history before building prompt
        self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

        # Build Prompt
        prompt = build_react_prompt(
            objective=step_objective,
            history=self._history,
            system_prompt=self.system_prompt,
            tool_descriptions=get_tool_descriptions(),
            agent_logger=agent_logger,
        )

        # Call LLM and Parse Response
        parsed_output = await self._process_llm_response(prompt, log_prefix)

        # Check for processing errors
        if not parsed_output or parsed_output.get("type") == "error":
            yield parsed_output or {
                "type": "error",
                "content": "Unknown error processing LLM response.",
            }
            # Indicate failure to the caller (e.g., by returning False or raising exception?)
            # For now, yielding error and letting caller handle it.
            return

        # Yield Thought
        if parsed_output.get("thought"):
            thought = parsed_output["thought"]
            self._history.append(f"Thought: {thought}")
            yield {"type": "thought", "content": thought}
        else:
            agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")

        # Handle Action or Final Answer
        action_name = parsed_output.get("action_name")
        action_input = parsed_output.get("action_input", {})

        if action_name == "final_answer":
            final_answer = action_input.get(
                "answer", "No final answer provided."
            )  # Get answer from input dict
            agent_logger.info(
                f"{log_prefix} Final Answer received for step: '{final_answer[:100]}...'"
            )
            self._history.append(f"Final Answer: {final_answer}")
            # Yield a special type indicating the step finished with an answer
            yield {"type": "step_final_answer", "content": final_answer}
            return  # Iteration ends here for this step

        if not action_name:
            agent_logger.error(
                f"{log_prefix} No Action specified by LLM (and not Final Answer). Yielding error."
            )
            yield {"type": "error", "content": "Agent did not specify an action."}
            return

        # Yield Action
        self._history.append(f"Action: {action_name}")
        try:
            action_input_json = json.dumps(action_input, ensure_ascii=False)
            self._history.append(f"Action Input: {action_input_json}")
        except TypeError:
            agent_logger.warning(
                f"{log_prefix} Action input not JSON serializable for history. Using str()."
            )
            self._history.append(f"Action Input: {str(action_input)}")
        yield {"type": "action", "tool_name": action_name, "tool_input": action_input}

        # Execute Action
        observation_data = await self._execute_action(
            action_name, action_input, log_prefix
        )

        # Handle and Yield Observation
        _ = self._handle_observation(observation_data, log_prefix)  # Adds to history
        yield {"type": "observation", "content": observation_data}

        # Check if tool execution failed, potentially yield error
        if observation_data.get("status") == "error":
            yield {
                "type": "error",
                "content": f"Tool execution failed: {observation_data.get('data', {}).get('message', 'Unknown tool error')}",
            }
            # Decide if the loop should stop? For now, let the main loop decide based on error.

    # --- run (Refactored to use _perform_react_iteration) ---
    async def run(self, objective: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes the plan-driven ReAct cycle to achieve the objective,
        generating each step's events (Thought, Action, Observation, Final Answer) as a dictionary.
        """
        log_prefix_base = "[ReactAgent]"
        self._current_plan = None
        final_answer_yielded = (
            False  # Flag to control if the overall objective final answer was yielded
        )

        try:
            # --- Setup History ---
            self._history = []
            self._history.append(f"Human: {objective}")
            agent_logger.info(
                f"{log_prefix_base} Objetivo Inicial: '{objective[:100]}...'"
            )

            # --- Planning Phase ---
            plan_to_execute = await self._generate_plan(objective)
            self._current_plan = plan_to_execute  # Store the plan
            yield {"type": "plan", "content": plan_to_execute}

            agent_logger.info("--- Starting Plan Execution ---")

            # --- Plan Execution Loop ---
            current_step_index = 0
            total_iterations = 0  # Track overall iterations across all steps
            last_error_message = None  # <<< INITIALIZE last_error_message >>>
            max_total_iterations = self.max_iterations * len(
                plan_to_execute
            )  # Adjust max iterations based on plan length

            # <<< ADD FLAG and CONTENT STORE >>>
            max_total_reached = False
            final_answer_content = None  # Store content from last step's final answer

            # <<< MAIN PLAN EXECUTION LOOP - ADD ITERATION CHECK BACK >>>
            # while current_step_index < len(plan_to_execute):
            while (
                current_step_index < len(plan_to_execute)
                and total_iterations < max_total_iterations
            ):
                # <<< REMOVED CHECK FROM HERE >>>
                step_objective = plan_to_execute[current_step_index]
                log_prefix_step = f"{log_prefix_base} (Plan Step {current_step_index + 1}/{len(plan_to_execute)})"
                agent_logger.info(
                    f"\n{log_prefix_step} (Objective: '{step_objective[:50]}...')"
                )

                step_completed = False
                react_iterations_this_step = 0

                # <<< INNER ReAct LOOP FOR THE CURRENT STEP >>>
                while (
                    not step_completed
                    and react_iterations_this_step < self.max_iterations
                ):
                    # <<< REMOVED CHECK FROM HERE >>>
                    total_iterations += 1
                    react_iterations_this_step += 1
                    log_prefix_react = (
                        f"{log_prefix_step} React Iter {react_iterations_this_step}"
                    )

                    # Handle actual ReAct iteration
                    async for event in self._perform_react_iteration(
                        step_objective, log_prefix_react
                    ):
                        # <<< YIELD ONLY INTERMEDIATE EVENTS, NOT step_final_answer >>>
                        if event.get("type") != "step_final_answer":
                            yield event  # Pass through non-final step events

                        if event.get("type") == "step_final_answer":
                            step_completed = True
                            # Store content if it's the last step
                            if current_step_index == len(plan_to_execute) - 1:
                                final_answer_content = event.get(
                                    "content"
                                )  # Store content
                                final_answer_yielded = (
                                    True  # Mark that we *have* a final answer
                                )
                            break  # Break inner loop
                        elif event.get("type") == "error":
                            last_error_message = event.get("content")
                            agent_logger.error(
                                f"{log_prefix_react} Error during iteration: {last_error_message}"
                            )
                            step_completed = (
                                True  # Mark step as completed (due to error)
                            )
                            break  # Break inner loop

                # Check iterations *per step*
                if (
                    not step_completed
                    and react_iterations_this_step >= self.max_iterations
                ):
                    agent_logger.warning(
                        f"{log_prefix_step} Max iterations ({self.max_iterations}) reached for step {current_step_index + 1}. Moving to next step."
                    )
                    last_error_message = (
                        f"Max iterations reached for step {current_step_index + 1}"
                    )

                if max_total_reached:
                    break  # Break outer loop if total limit was hit inside inner loop

                current_step_index += 1

            # --- End MAIN PLAN EXECUTION LOOP ---

            # <<< CHECK IF LOOP EXITED DUE TO MAX ITERATIONS >>>
            if total_iterations >= max_total_iterations and not final_answer_yielded:
                agent_logger.warning(
                    f"--- Max total iterations ({max_total_iterations}) reached. Stopping execution. ---"
                )
                yield {"type": "error", "content": "Max total iterations reached."}
                # No need to set final_answer_yielded = True here, summary will run if needed

            # <<< ADJUST FINAL YIELD LOGIC >>>
            # If loop finished AND we captured a final answer from the last step
            if final_answer_yielded and final_answer_content is not None:
                yield {"type": "final_answer", "content": final_answer_content}
            # Otherwise, if loop finished or broke early without a specific final answer
            elif not final_answer_yielded:
                agent_logger.warning(
                    f"{log_prefix_base} Plan execution finished, but no overall Final Answer was yielded during steps. Attempting to summarize."
                )
                last_thought_or_obs = (
                    self._history[-1] if self._history else "No history available."
                )
                final_content = (
                    f"Plan execution concluded. Last state: {last_thought_or_obs}"
                )
                if last_error_message:
                    final_content += f". Encountered error: {last_error_message}"
                yield {
                    "type": "final_answer",
                    "content": final_content,
                }

        except Exception as e:
            agent_logger.exception(
                f"{log_prefix_base} Unhandled exception during agent run:"
            )
            yield {"type": "error", "content": f"Agent execution failed: {e}"}
        finally:
            # --- Save State ---
            self._save_state()
            agent_logger.info(f"{log_prefix_base} Agent state saved.")

    def get_history(self):
        """Retorna o histórico interno para possível inspeção ou passagem para outros componentes."""
        return self._history

    # <<< ADDED Method to add history externally >>>
    def add_history_entry(self, role: str, content: str):
        """Adiciona uma entrada ao histórico interno (usado por CerebrumX para registrar resultados)."""
        # Simple append for now, role might be used for formatting later
        if role.lower() == "human" or role.lower() == "user":
            self._history.append(f"Human: {content}")
        elif role.lower() == "assistant":
            # Format based on likely type (Thought, Observation, etc.)
            if (
                content.startswith("Thought:")
                or content.startswith("Observation:")
                or content.startswith("Final Answer:")
                or content.startswith("Execution result for")
            ):
                self._history.append(content)
            else:  # Generic assistant message
                self._history.append(f"Assistant: {content}")
        else:
            self._history.append(f"{role.capitalize()}: {content}")
        # Optional: Trim history after adding
        # self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

    # <<< NEW: Save State Method (Example using file storage) >>>
    def _save_state(self):
        state = {"history": self._history, "memory": self._memory}
        # <<< FIX: Use string "1" for agent_id >>>
        save_agent_state(agent_id="1", state=state)  # Use self.agent_id later
        agent_logger.info("[ReactAgent] Agent state saved for ID '1'.")
