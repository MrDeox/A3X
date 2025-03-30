import logging
import re
import json
import datetime
import os
import sys
import traceback
from typing import Tuple, Optional, Dict, Any, List, AsyncGenerator

import requests

# Local imports
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS, MAX_META_DEPTH, MAX_TOKENS_FALLBACK, CONTEXT_SIZE
from core.tools import TOOLS, get_tool_descriptions
# Removed memory skill import as it's not directly used here anymore
from core.db_utils import save_agent_state, load_agent_state
from core.prompt_builder import build_react_prompt
from core.agent_parser import parse_llm_response
# <<< CORRECTED IMPORTS (Replaced core.utils) >>>
# from core.utils import agent_logger, setup_agent_logger, history_manager, tool_executor, llm_client, agent_parser, agent_reflector
import logging # Already imported, but ensure it is
from core.history_manager import trim_history # <-- CORRECTED import
from core.tool_executor import execute_tool # <-- CORRECTED import
from core.llm_interface import call_llm # <-- CORRECTED import
# from core.agent_parser import AgentParser # Assuming class name <-- REMOVE IMPORT
from core.agent_reflector import reflect_on_observation # Assuming class name <-- CORRECTED import name
from core.planner import generate_plan # <<< ADDED planner import >>>
from core.tools import get_tool_descriptions, TOOLS # <<< ADDED get_tool_descriptions import >>>

# <<< REMOVE unused imports >>>
# from core import agent_error_handler, agent_autocorrect

# <<< IMPORT NEW INTERFACE >>>
# from .llm_interface import call_llm # This might be redundant if LLMClient is used

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Instantiate necessary components (assuming they are classes)
# We might need to adjust this based on actual implementations
# history_manager = HistoryManager() # Example instantiation
# tool_executor = ToolExecutor() <-- REMOVED instantiation
# llm_client = LLMClient(base_url=LLAMA_SERVER_URL, headers=LLAMA_DEFAULT_HEADERS) # Example with config <-- REMOVED instantiation
# agent_parser = AgentParser() <-- REMOVE INSTANTIATION
# agent_reflector = AgentReflector() <-- REMOVE INSTANTIATION

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
    def __init__(self, llm_url: str, system_prompt: str):
        """Inicializa o Agente ReAct."""
        self.llm_url = llm_url
        self.system_prompt = system_prompt
        self.tools = TOOLS  # Carrega as ferramentas
        self._history = [] # Histórico de Thought, Action, Observation
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega estado/memória inicial
        self.max_iterations = MAX_REACT_ITERATIONS
        self._current_plan = None # <<< Initialize plan >>>
        # Removed _last_error_type, not needed with new handlers
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. Memória carregada: {list(self._memory.keys())}")

    # --- run (Refatorado para ser um AsyncGenerator) ---
    async def run(self, objective: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executa o ciclo ReAct (agora orientado por plano) para atingir o objetivo,
        gerando cada passo (Thought, Action, Observation, Final Answer) como um dicionário.
        """
        log_prefix_base = "[ReactAgent]"
        self._current_plan = None
        plan_to_execute: List[str] = []
        final_answer_yielded = False # Flag para controlar se a resposta final foi gerada

        try:
            # --- Setup History --- 
            self._history = []
            self._history.append(f"Human: {objective}")
            agent_logger.info(f"{log_prefix_base} Objetivo Inicial: '{objective[:100]}...'" )

            # --- Planning Phase --- 
            agent_logger.info("--- Generating Plan ---")
            if _is_simple_list_files_task(objective):
                agent_logger.info("[Planner] Detected simple list_files task. Skipping complex planning.")
                # Create a direct plan: 1. list_files, 2. final_answer
                plan_to_execute = [
                    f"Use the list_files tool for the objective: '{objective}'", # Step description for clarity
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
                    plan_to_execute = [objective]
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

                # Call LLM
                agent_logger.info(f"{log_prefix} Calling LLM...")
                llm_response_raw = ""
                try:
                    # Usa a função call_llm (não geradora aqui)
                    async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                        llm_response_raw += chunk # Acumula a resposta não-streamed
                    agent_logger.info(f"{log_prefix} LLM Response received.")
                    agent_logger.debug(f"{log_prefix} Raw LLM Response:\n{llm_response_raw}")
                except Exception as llm_err:
                    agent_logger.exception(f"{log_prefix} Error calling LLM:")
                    yield {"type": "error", "content": f"Failed to get response from LLM: {llm_err}"}
                    return # Encerra o gerador em erro de LLM

                # Parse LLM Response
                try:
                    # <<< CORRECTED: Use parse_llm_response from agent_parser >>>
                    parsed_output_tuple = parse_llm_response(llm_response_raw, agent_logger)
                    # Handle potential None return from parser if JSON is invalid but doesn't raise
                    if parsed_output_tuple is None:
                         agent_logger.error(f"{log_prefix} Failed to parse LLM response (parse_llm_response returned None). Raw: '{llm_response_raw[:100]}...'")
                         yield {"type": "error", "content": "Failed to parse LLM response."}
                         return # Stop if parsing fundamentally failed

                    # Convert tuple to dict for consistent handling downstream
                    thought, action_name, action_input = parsed_output_tuple
                    parsed_output = {}
                    if thought: parsed_output["thought"] = thought
                    if action_name: parsed_output["action_name"] = action_name
                    # Ensure action_input is always a dict, even if None from parser (e.g., for final_answer)
                    parsed_output["action_input"] = action_input if action_input is not None else {}

                except json.JSONDecodeError as parse_err:
                    agent_logger.error(f"{log_prefix} Failed to parse LLM response (JSONDecodeError). Raw: '{llm_response_raw[:100]}...'")
                    # Log the full error and traceback
                    agent_logger.exception(f"{log_prefix} JSON Parsing Traceback:")
                    # Consider adding reflection/recovery logic here if desired
                    yield {"type": "error", "content": f"Failed to parse LLM response: {parse_err}"}
                    return # Stop if parsing failed

                # Yield Thought
                if parsed_output.get("thought"):
                    thought = parsed_output["thought"]
                    self._history.append(f"Thought: {thought}")
                    yield {"type": "thought", "content": thought}
                else:
                    agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")
                    # Pode acontecer, continua para Action

                # Handle Action or Final Answer
                if "action_name" in parsed_output and parsed_output["action_name"] == "final_answer":
                    final_answer = parsed_output.get("action_input", {}).get("answer", "No final answer provided.")
                    agent_logger.info(f"{log_prefix} Final Answer received: '{final_answer[:100]}...'")
                    self._history.append(f"Final Answer: {final_answer}")
                    yield {"type": "final_answer", "content": final_answer}
                    final_answer_yielded = True
                    break # Fim do ciclo e do loop while

                elif "action_name" in parsed_output:
                    action_name = parsed_output["action_name"]
                    action_input = parsed_output.get("action_input", {})
                    agent_logger.info(f"{log_prefix} Action: {action_name}, Input: {action_input}")

                    # Yield Action
                    yield {"type": "action", "tool_name": action_name, "tool_input": action_input}

                    # Execute Action
                    tool_result = execute_tool(action_name, action_input, TOOLS, agent_logger)
                    agent_logger.info(f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}")
                    # <<< ERRO AQUI: observation_str não é usado em yield, e history deve ter string <<< CORREÇÃO ABAIXO
                    # observation_str = json.dumps(tool_result, indent=2, ensure_ascii=False) # Format result for history/yield
                    try:
                         observation_str_for_history = json.dumps(tool_result) # Compacto para histórico
                    except TypeError:
                         observation_str_for_history = str(tool_result) # Fallback se não serializável

                    # Yield Observation (yield o dicionário completo)
                    self._history.append(f"Observation: {observation_str_for_history}")
                    yield {"type": "observation", "content": tool_result} # Yield the dict result

                    # Reflection (Optional, pode ser habilitado/desabilitado)
                    # reflection = reflect_on_observation(objective, self._history, agent_logger, self.llm_url)
                    # if reflection:
                    #    agent_logger.info(f"{log_prefix} Reflection: {reflection}")
                    #    self._history.append(f"Reflection: {reflection}") # Add reflection to history if needed

                else:
                    agent_logger.error(f"{log_prefix} No valid Action or Final Answer found in parsed output.")
                    yield {"type": "error", "content": "LLM did not provide a valid action or final answer."}
                    break # Parar se o LLM não der uma ação válida

                # Increment counters and move to next step if applicable
                total_iterations += 1
                # A lógica de avançar `current_step_index` pode ser mais complexa,
                # dependendo se a ação atual cumpriu o passo do plano.
                # Por simplicidade, avançamos o passo do plano após cada ciclo de ação bem-sucedido.
            current_step_index += 1

            # --- After Loop --- 
            if not final_answer_yielded: # Se saiu do loop sem dar yield em final_answer
                if total_iterations >= max_total_iterations:
                    msg = f"Agent stopped: Maximum total iterations ({max_total_iterations}) reached."
                    agent_logger.warning(f"{log_prefix_base} {msg}")
                    yield {"type": "error", "content": msg}
                elif current_step_index >= len(plan_to_execute):
                    msg = "Agent stopped: Plan execution completed, but no 'final_answer' action was triggered."
                    agent_logger.warning(f"{log_prefix_base} {msg}")
                    yield {"type": "info", "content": msg} # Talvez não seja um erro, apenas terminou o plano
                else:
                     msg = "Agent stopped unexpectedly after loop."
                     agent_logger.error(f"{log_prefix_base} {msg}")
                     yield {"type": "error", "content": msg}

        except Exception as main_loop_err:
            agent_logger.exception(f"{log_prefix_base} Unexpected exception during agent run: {main_loop_err}")
            yield {"type": "error", "content": f"Critical Agent Error: {main_loop_err}"}

        finally:
            # --- Save State --- 
            agent_logger.info(f"{log_prefix_base} Saving final agent state (ID: {AGENT_STATE_ID}).")
            save_agent_state(AGENT_STATE_ID, self._memory)
            agent_logger.info(f"{log_prefix_base} Agent run finished.")

    # --- REMOVED _execute_react_cycle ---
