import logging
import re
import json
import os
from typing import Dict, Any, List, AsyncGenerator, Optional

# Local imports
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS, LLAMA_DEFAULT_HEADERS, MAX_META_DEPTH, MAX_TOKENS_FALLBACK, CONTEXT_SIZE
from core.tools import TOOLS, get_tool_descriptions
from core.db_utils import save_agent_state, load_agent_state
from core.prompt_builder import build_react_prompt
from core.agent_parser import parse_llm_response
from core.history_manager import trim_history
from core.tool_executor import execute_tool
from core.llm_interface import call_llm
from core.agent_reflector import reflect_on_observation
from core.planner import generate_plan

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
    if (any(kw in objective_lower for kw in keywords) and
            "ler" not in objective_lower and
            "conteúdo" not in objective_lower and
            "criar" not in objective_lower and
            "deletar" not in objective_lower and
            "apagar" not in objective_lower and
            "executar" not in objective_lower):
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
        self.tools = TOOLS  # Carrega as ferramentas
        self._history = [] # Histórico de Thought, Action, Observation
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega estado/memória inicial
        self.max_iterations = MAX_REACT_ITERATIONS
        self._current_plan = None # <<< Initialize plan >>>
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}")

    # <<< NEW: Method for Plan Generation >>>
    async def _generate_plan(self, objective: str) -> List[str]:
        """Gera um plano de execução para o objetivo dado."""
        agent_logger.info("--- Generating Plan ---")
        plan_to_execute: List[str] = []

        if _is_simple_list_files_task(objective):
            agent_logger.info("[Planner] Detected simple list_files task. Skipping complex planning.")
            plan_to_execute = [
                f"Use the list_files tool for the objective: '{objective}'",
                "Use the final_answer tool to provide the list of files."
            ]
            plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
            agent_logger.info(f"Simple Plan Generated:\n{plan_str}")
        else:
            tool_desc = get_tool_descriptions()
            generated_plan = await generate_plan(objective, tool_desc, agent_logger, self.llm_url)
            if generated_plan:
                plan_to_execute = generated_plan
                plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
                agent_logger.info(f"Plan Generated:\n{plan_str}")
            else:
                agent_logger.warning("Failed to generate a plan. Proceeding with objective as single step.")
                plan_to_execute = [objective] # Fallback to objective as the only step
        return plan_to_execute

    # <<< NEW: Method to Call LLM and Parse Response >>>
    async def _process_llm_response(self, prompt: List[Dict[str, str]], log_prefix: str) -> Optional[Dict[str, Any]]:
        """Chama o LLM com o prompt, parseia a resposta e retorna um dicionário estruturado ou None em caso de erro fatal."""
        agent_logger.info(f"{log_prefix} Calling LLM...")
        llm_response_raw = ""
        try:
            # Use call_llm (non-streaming for react cycle response)
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                llm_response_raw += chunk
            agent_logger.info(f"{log_prefix} LLM Response received.")
            agent_logger.debug(f"{log_prefix} Raw LLM Response:\n{llm_response_raw}")

            # Parse the response
            parsed_output_tuple = parse_llm_response(llm_response_raw, agent_logger)
            if parsed_output_tuple is None:
                agent_logger.error(f"{log_prefix} Failed to parse LLM response (parse_llm_response returned None). Raw: '{llm_response_raw[:100]}...'")
                # Return an error structure instead of None for consistency
                return {"type": "error", "content": "Failed to parse LLM response (parser returned None)."}

            thought, action_name, action_input = parsed_output_tuple
            parsed_output = {}
            if thought: parsed_output["thought"] = thought
            if action_name: parsed_output["action_name"] = action_name
            parsed_output["action_input"] = action_input if action_input is not None else {}
            return parsed_output # Return structured dictionary

        except json.JSONDecodeError as parse_err:
            agent_logger.error(f"{log_prefix} Failed to parse LLM response (JSONDecodeError). Raw: '{llm_response_raw[:100]}...'")
            agent_logger.exception(f"{log_prefix} JSON Parsing Traceback:")
            return {"type": "error", "content": f"Failed to parse LLM response: {parse_err}"}
        except Exception as llm_err: # Catch other errors like connection issues during call_llm or general parsing
            agent_logger.exception(f"{log_prefix} Error during LLM call or processing:")
            return {"type": "error", "content": f"Failed to get or process LLM response: {llm_err}"}

    # <<< NEW: Method to Execute Action >>>
    async def _execute_action(self, action_name: str, action_input: Dict[str, Any], log_prefix: str) -> Dict[str, Any]:
        """Executa a ferramenta especificada com os inputs fornecidos."""
        agent_logger.info(f"{log_prefix} Executing Action: {action_name} with input: {action_input}")
        try:
            # <<< CORRECTED CALL: Pass arguments in the correct order >>>
            tool_result = execute_tool(
                tool_name=action_name, 
                action_input=action_input, 
                tools_dict=self.tools, # Pass self.tools
                agent_logger=agent_logger, # Pass the module logger
                agent_memory=self._memory # Pass agent memory
            )
            agent_logger.info(f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}")
            return tool_result
        except Exception as tool_err:
            agent_logger.exception(f"{log_prefix} Error executing tool '{action_name}':")
            return {
                "status": "error",
                "action": f"{action_name}_failed",
                "data": {"message": f"Error during tool execution: {tool_err}"}
            }

    # <<< NEW: Method to Handle Observation >>>
    def _handle_observation(self, observation_data: Dict[str, Any], log_prefix: str) -> str:
        """Processa os dados da observação, formata para o histórico e log."""
        try:
            # Format observation for history (compact JSON or string fallback)
            observation_content = json.dumps(observation_data, ensure_ascii=False)
            agent_logger.info(f"{log_prefix} Observation: {observation_content[:150]}...")
            self._history.append(f"Observation: {observation_content}")
            return observation_content # Return the string version for potential reflection
        except TypeError:
            # Fallback if data is not JSON serializable
            observation_content = str(observation_data)
            agent_logger.warning(f"{log_prefix} Observation data not JSON serializable. Using str().")
            agent_logger.info(f"{log_prefix} Observation (str): {observation_content[:150]}...")
            self._history.append(f"Observation: {observation_content}")
            return observation_content
        except Exception as obs_err:
            agent_logger.exception(f"{log_prefix} Error handling observation:")
            error_content = f"Error processing observation: {obs_err}"
            self._history.append(f"Observation: {{'status': 'error', 'message': '{error_content}'}}")
            return error_content

    # --- run (Refatorado para ser um AsyncGenerator) ---
    async def run(self, objective: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executa o ciclo ReAct (agora orientado por plano) para atingir o objetivo,
        gerando cada passo (Thought, Action, Observation, Final Answer) como um dicionário.
        """
        log_prefix_base = "[ReactAgent]"
        self._current_plan = None
        final_answer_yielded = False # Flag para controlar se a resposta final foi gerada

        try:
            # --- Setup History --- 
            self._history = []
            self._history.append(f"Human: {objective}")
            agent_logger.info(f"{log_prefix_base} Objetivo Inicial: '{objective[:100]}...'" )

            # --- Planning Phase --- 
            plan_to_execute = await self._generate_plan(objective)
            self._current_plan = plan_to_execute # Store the plan

            agent_logger.info("--- Starting Plan Execution ---")

            # --- Plan Execution Loop / ReAct Cycle ---
            current_step_index = 0
            total_iterations = 0
            max_total_iterations = self.max_iterations

            while current_step_index < len(plan_to_execute) and total_iterations < max_total_iterations:
                current_step_objective = plan_to_execute[current_step_index]
                log_prefix = f"{log_prefix_base} Cycle {total_iterations + 1} (Step {current_step_index + 1}/{len(plan_to_execute)})"
                agent_logger.info(f"\n{log_prefix} (Step Objective: '{current_step_objective[:60]}...')" )

                # Trim history
                self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

                # Build Prompt
                prompt = build_react_prompt(
                    objective=current_step_objective, # <<< Use step objective for react cycle
                    history=self._history,
                    system_prompt=self.system_prompt,
                    tool_descriptions=get_tool_descriptions(),
                    agent_logger=agent_logger
                )

                # <<< CALL _process_llm_response >>>
                parsed_output = await self._process_llm_response(prompt, log_prefix)

                # Check for processing errors
                if parsed_output.get("type") == "error":
                    yield parsed_output # Yield the error dict
                    return # Stop execution on critical LLM/parsing error

                # Yield Thought
                if parsed_output.get("thought"):
                    thought = parsed_output["thought"]
                    self._history.append(f"Thought: {thought}")
                    yield {"type": "thought", "content": thought}
                else:
                    agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")
                    # Pode acontecer, continua para Action

                # Handle Action or Final Answer
                action_name = parsed_output.get("action_name")
                action_input = parsed_output.get("action_input") # Already ensured to be a dict

                if action_name == "final_answer":
                    final_answer = action_input.get("answer", "No final answer provided.") # Get answer from input dict
                    agent_logger.info(f"{log_prefix} Final Answer received: '{final_answer[:100]}...'")
                    self._history.append(f"Final Answer: {final_answer}")
                    yield {"type": "final_answer", "content": final_answer}
                    final_answer_yielded = True
                    # Decide if we break the loop or move to the next plan step
                    # For now, assume final_answer for a step completes that step
                    current_step_index += 1 # Move to next plan step
                    continue # Skip action execution for this cycle

                if not action_name:
                    agent_logger.error(f"{log_prefix} No Action or Final Answer specified by LLM.")
                    yield {"type": "error", "content": "Agent did not specify an action or final answer."}
                    # Decide if we retry or stop. Stopping for now.
                    return

                # Yield Action
                self._history.append(f"Action: {action_name}")
                # Ensure action_input is serializable for logging/history if needed
                self._history.append(f"Action Input: {json.dumps(action_input, ensure_ascii=False)}")
                yield {"type": "action", "tool_name": action_name, "tool_input": action_input}

                # <<< CALL _execute_action >>>
                observation_data = await self._execute_action(action_name, action_input, log_prefix)

                # <<< CALL _handle_observation >>>
                # observation_content = self._handle_observation(observation_data, log_prefix) # Handles history append
                self._handle_observation(observation_data, log_prefix) # Call to handle history etc.

                # Yield Observation (yield the raw data dict)
                yield {"type": "observation", "content": observation_data}

                # --- Check if step is complete based on observation? ---
                # TODO: Implement logic to decide if the current step objective is met
                # Check if the tool execution resulted in an error that should stop the plan step
                if observation_data.get("status") == "error":
                     agent_logger.warning(f"{log_prefix} Tool execution failed for step. Stopping current plan step.")
                     # Decide if we stop the whole plan or just this step. Stopping plan for now.
                     # yield {"type": "error", "content": f"Step failed due to tool error: {observation_data.get('data', {}).get('message', 'Unknown tool error')}"}
                     # return # Stop the entire run
                     # Or, simply break the inner loop if we had one, and let the outer loop decide?
                     # For now, let's just log and continue to the next step/iteration count check
                     pass # Allow loop condition to check iterations/steps

                # For now, assume one ReAct cycle per plan step unless final_answer is given
                current_step_index += 1
                total_iterations += 1

            # --- End of Loop Handling ---
            if total_iterations >= max_total_iterations:
                agent_logger.warning(f"{log_prefix_base} Reached max iterations ({max_total_iterations}).")
                if not final_answer_yielded:
                     yield {"type": "error", "content": "Agent reached max iterations without a final answer."}

            if current_step_index >= len(plan_to_execute) and not final_answer_yielded:
                 agent_logger.warning(f"{log_prefix_base} Plan completed, but no final answer was explicitly generated.")
                 # Optionally yield a generic completion message
                 # yield {"type": "final_answer", "content": "Plan execution finished."}

        except Exception as e:
            agent_logger.exception(f"{log_prefix_base} Unhandled exception during agent run:")
            if not final_answer_yielded:
                 yield {"type": "error", "content": f"An unexpected error occurred: {e}"}
        finally:
            # Save final state regardless of how run exits
            save_agent_state(AGENT_STATE_ID, self._memory)
            agent_logger.info(f"{log_prefix_base} Agent run finished.")
