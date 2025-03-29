import logging
import re
import json
import datetime
import os
import sys
import traceback
from typing import Tuple, Optional, Dict, Any

import requests

# Local imports
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS, MAX_META_DEPTH
from core.tools import TOOLS, get_tool_descriptions
# Removed memory skill import as it's not directly used here anymore
from core.db_utils import save_agent_state, load_agent_state

# <<< NEW IMPORTS for modular functions >>>
from core import agent_parser, prompt_builder, history_manager, tool_executor, planner
# <<< REMOVE unused imports >>>
# from core import agent_error_handler, agent_autocorrect

# <<< IMPORT NEW INTERFACE >>>
from .llm_interface import call_llm

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente
AGENT_STATE_ID = 1

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

    # --- run (Refatorado para usar _execute_react_cycle) ---
    def run(self, objective: str) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""
        final_response = f"Erro: O agente não conseguiu completar o objetivo após {self.max_iterations} iterações." # Default error
        log_prefix_base = "[ReactAgent]"
        self._current_plan = None # Ensure plan is reset for each run

        try: # Wrap main loop in try...finally to ensure state saving
            # --- Setup History --- (Simplified)
            self._history = []
            self._history.append(f"Human: {objective}")
            current_objective = objective # Keep original objective for now

            agent_logger.info(f"{log_prefix_base} Objetivo Inicial: '{current_objective[:100]}...'" )

            # <<< START Planning Phase >>>
            agent_logger.info("--- Generating Plan ---")
            tool_desc = get_tool_descriptions()
            # Pass llm_url to generate_plan
            self._current_plan = planner.generate_plan(objective, tool_desc, agent_logger, self.llm_url)
            if self._current_plan:
                # Use json.dumps for pretty printing the plan list - FIX f-string syntax
                plan_str = json.dumps(self._current_plan, indent=2, ensure_ascii=False)
                agent_logger.info(f"Plan Generated:\n{plan_str}") # Single line f-string
                # TODO: Implementar iteração sobre self._current_plan no loop principal (usando current_objective por enquanto)
            else:
                agent_logger.error("Failed to generate a plan. Proceeding with the original objective.")
                # O agente continuará com o objetivo original se o plano falhar
            agent_logger.info("--- Starting ReAct Cycles (Using Original Objective) ---")
            # <<< END Planning Phase >>>

            agent_logger.debug(f"{log_prefix_base} Histórico Inicial (após plano): {self._history}")

            # --- Ciclo ReAct Loop (delegado para _execute_react_cycle) ---
            for i in range(self.max_iterations):
                should_break, final_response_from_cycle = self._execute_react_cycle(
                    cycle_num=i + 1,
                    log_prefix_base=log_prefix_base,
                    # Pass the original objective for now, plan iteration comes later
                    current_objective=objective
                )
                if should_break:
                    final_response = final_response_from_cycle # Atualiza a resposta final (pode ser None se break for por erro)
                    break # Sai do loop for
            else: # Executado se o loop for concluído sem break (max iterations)
                agent_logger.warning(f"{log_prefix_base} Máximo de iterações ({self.max_iterations}) atingido.")
                # Define a mensagem de erro final padrão (pode incluir a última observação)
                last_observation = self._history[-1] if self._history and self._history[-1].startswith("Observation:") else "Nenhuma observação disponível."
                final_response = f"Erro: Máximo de iterações ({self.max_iterations}) atingido. Última observação: {last_observation}"

            # --- End of ReAct Cycle Loop ---

        except Exception as main_loop_err:
            agent_logger.exception(f"{log_prefix_base} Exceção inesperada no loop principal ReAct: {main_loop_err}")
            final_response = f"Erro Interno Crítico no Agente: {main_loop_err}"

        finally:
            # --- Save State --- 
            agent_logger.info(f"{log_prefix_base} Salvando estado final do agente (ID: {AGENT_STATE_ID}).")
            save_agent_state(AGENT_STATE_ID, self._memory)
            agent_logger.info(f"{log_prefix_base} Ciclo ReAct finalizado. Retornando: '{final_response[:100]}...'" if final_response else "[No Final Response]")

        return final_response if final_response is not None else "Erro: O agente finalizou sem uma resposta definida."

    # --- _execute_react_cycle (NOVO MÉTODO PRIVADO) ---
    def _execute_react_cycle(self, cycle_num: int, log_prefix_base: str, current_objective: str) -> Tuple[bool, Optional[str]]:
        """Executa um único ciclo do loop ReAct.
        Retorna (should_break_loop, final_response_if_break)
        """
        log_prefix = f"{log_prefix_base} Cycle {cycle_num}/{self.max_iterations}"
        agent_logger.info(f"\n{log_prefix} (Objetivo: '{current_objective[:60]}...' Inicio: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]})" )

        # --- START ADDED DEBUG LOGGING (Keep for now) ---
        agent_logger.debug(f"{log_prefix} --- Start Cycle Input ---")
        agent_logger.debug(f"{log_prefix} Objective: {current_objective}")
        try:
            history_to_log = self._history[-(MAX_HISTORY_TURNS * 2):]
            history_log_str = json.dumps(history_to_log, indent=2, ensure_ascii=False)
            agent_logger.debug(f"{log_prefix} Relevant History (Last ~{MAX_HISTORY_TURNS} turns):\n{history_log_str}")
        except Exception as log_err:
            agent_logger.error(f"{log_prefix} Failed to serialize history for logging: {log_err}")
        agent_logger.debug(f"{log_prefix} --- End Cycle Input ---")
        # --- END ADDED DEBUG LOGGING ---

        start_cycle_time = datetime.datetime.now()
        # 1. Construir Prompt para LLM
        tool_desc = get_tool_descriptions()
        prompt_messages = prompt_builder.build_react_prompt(
            current_objective, self._history, self.system_prompt, tool_desc, agent_logger
        )

        # 2. Chamar LLM
        try:
            # Usa o _call_llm interno que agora força JSON
            llm_response_raw = self._call_llm(prompt_messages)
            agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
        except Exception as e:
            agent_logger.exception(f"{log_prefix} Exceção durante a chamada LLM: {e}")
            llm_response_raw = f"Erro: Falha na chamada LLM: {e}" # Define a mensagem de erro padrão

        log_llm_response = llm_response_raw[:1000] + ('...' if len(llm_response_raw) > 1000 else '')
        agent_logger.debug(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{log_llm_response}\n---")

        # --- Handle LLM Call Errors (Simplified) ---
        if not llm_response_raw or llm_response_raw.startswith("Erro:"):
            agent_logger.error(f"{log_prefix} LLM Call Failed: {llm_response_raw}")
            self._history.append(f"Observation: Erro interno ao comunicar com o LLM: {llm_response_raw}")
            return False, None # Continue loop

        self._history.append(llm_response_raw) # Append raw response only if no fatal error

        # 3. Parsear Resposta LLM
        try:
            thought, action_name, action_input = agent_parser.parse_llm_response(llm_response_raw, agent_logger)
            if not action_name:
                raise ValueError("JSON válido, mas chave 'Action' obrigatória está ausente.")
        except (json.JSONDecodeError, ValueError) as parse_error:
            agent_logger.error(f"{log_prefix} Parsing Failed: {parse_error}")
            self._history.append(f"Observation: Erro interno ao processar a resposta do LLM: {parse_error}")
            return False, None # Continue loop

        agent_logger.info(f"{log_prefix} Ação Decidida: {action_name}, Input: {action_input}")

        # 4. Executar Ação ou Finalizar
        observation_str = None # Initialize observation
        tool_result = None # Initialize tool result

        if action_name == "final_answer":
            final_answer_text = action_input.get("answer", "Finalizado sem resposta específica.")
            agent_logger.info(f"{log_prefix} Ação Final: {final_answer_text}")
            self._history.append(f"Final Answer: {final_answer_text}")
            return True, final_answer_text # Break loop with final response

        elif action_name in self.tools:
            tool_result = tool_executor.execute_tool(
                tool_name=action_name,
                action_input=action_input,
                tools_dict=self.tools,
                agent_logger=agent_logger
            )
            # Lógica de guardar código em memória
            if tool_result.get('status') == 'success' and action_name in ["generate_code", "modify_code"]:
                 code = tool_result.get('data', {}).get('code') or tool_result.get('data', {}).get('modified_code')
                 if code:
                     self._memory['last_code'] = code
                     # TODO: Considerar guardar linguagem também?
                     # self._memory['last_lang'] = action_input.get('language')
                     agent_logger.info(f"{log_prefix} Saved code from '{action_name}' to memory['last_code'].")
                 else:
                     agent_logger.warning(f"{log_prefix} Action '{action_name}' succeeded but no 'code' or 'modified_code' found in data.")

            # --- Simplified Observation Formatting ---
            try:
                observation_str = f"Observation: {json.dumps(tool_result, ensure_ascii=False)}"
            except TypeError as json_err:
                agent_logger.error(f"{log_prefix} Failed to serialize tool_result to JSON: {json_err}. Result: {tool_result}")
                observation_str = f"Observation: [Error serializing tool result: {json_err}]"

        else:
            # Ferramenta não encontrada
            tool_result = {"status": "error", "action": "tool_not_found", "data": {"message": f"A ferramenta '{action_name}' não existe."}}
            observation_str = f"Observation: {json.dumps(tool_result)}" # Format as observation

        # --- Append Observation to History ---
        if observation_str:
            agent_logger.debug(f"[DEBUG HISTORY APPEND] Appending observation: '{observation_str[:200]}...'")
            self._history.append(observation_str)
        else:
             agent_logger.error(f"{log_prefix} Observation string was unexpectedly None after tool execution.")
             self._history.append("Observation: [Internal Error: Failed to generate observation]")

        # --- Trim History --- (Do this at the end of the cycle)
        self._history = history_manager.trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

        # No break condition met in this cycle
        return False, None

    # --- _call_llm (Remains the same, delegates) ---
    def _call_llm(self, messages: list[dict]) -> str:
        """Chama o LLM local através da interface centralizada, forçando JSON para o ReAct."""
        # <<< DELEGATE TO LLM INTERFACE >>>
        return call_llm(
            llm_url=self.llm_url,
            messages=messages,
            force_json_output=True # <<< Always force JSON for ReAct cycle >>>
        )

    # <<< REMOVED METHOD: _execute_tool >>>
    # <<< REMOVED METHOD: _trim_history >>>
