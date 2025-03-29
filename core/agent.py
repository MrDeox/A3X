\
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
from core import agent_parser, prompt_builder, history_manager, tool_executor
from core import agent_error_handler, agent_autocorrect # <<< ADDED new imports

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente
AGENT_STATE_ID = 1

# <<< CARREGAR JSON SCHEMA (Mantido como antes) >>>
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'react_output_schema.json')
LLM_RESPONSE_SCHEMA = None
try:
    with open(SCHEMA_FILE_PATH, 'r', encoding='utf-8') as f:
        LLM_RESPONSE_SCHEMA = json.load(f)
    agent_logger.info(f"[ReactAgent INIT] JSON Schema carregado de {SCHEMA_FILE_PATH}")
except FileNotFoundError:
    agent_logger.error(f"[ReactAgent INIT ERROR] Arquivo JSON Schema não encontrado em {SCHEMA_FILE_PATH}. A saída do LLM não será forçada.")
except json.JSONDecodeError as e:
    agent_logger.error(f"[ReactAgent INIT ERROR] Erro ao decodificar JSON Schema de {SCHEMA_FILE_PATH}: {e}. A saída do LLM não será forçada.")
except Exception as e:
    agent_logger.error(f"[ReactAgent INIT ERROR] Erro inesperado ao carregar JSON Schema de {SCHEMA_FILE_PATH}: {e}. A saída do LLM não será forçada.")

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
        # Removed _last_error_type, not needed with new handlers
        self._last_executed_code = None
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. Memória carregada: {list(self._memory.keys())}")

    # --- run (Refatorado para usar funções modulares) ---
    def run(self, objective: str, is_meta_objective: bool = False, meta_depth: int = 0) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""
        final_response = f"Erro: O agente não conseguiu completar o objetivo após {self.max_iterations} iterações." # Default error

        try: # Wrap main loop in try...finally to ensure state saving
            # --- Limite de Profundidade Meta ---
            if meta_depth > MAX_META_DEPTH:
                agent_logger.warning(f"[ReactAgent META-{meta_depth}] Max meta depth ({MAX_META_DEPTH}) reached. Aborting meta-cycle for objective: '{objective[:100]}...'")
                return f"Erro: Profundidade máxima de auto-correção ({MAX_META_DEPTH}) atingida." # Return directly

            # --- Logging & Setup ---
            if is_meta_objective:
                log_prefix_base = f"[ReactAgent META-{meta_depth}]"
                current_objective = objective
                # Meta cycles inherit history, no reset needed
            else:
                log_prefix_base = "[ReactAgent]"
                self._history = [] # Reset history only for non-meta objectives
                self._history.append(f"Human: {objective}")
                current_objective = objective

            agent_logger.info(f"\n{log_prefix_base} Iniciando ciclo ReAct (Objetivo: '{current_objective[:100]}...' Profundidade: {meta_depth})")
            agent_logger.debug(f"{log_prefix_base} Histórico Inicial: {self._history}")

            # --- Ciclo ReAct ---
            for i in range(self.max_iterations):
                cycle_num = i + 1
                log_prefix = f"{log_prefix_base} Cycle {cycle_num}/{self.max_iterations}"
                agent_logger.info(f"\n{log_prefix} (Objetivo: '{current_objective[:60]}...' Inicio: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

                start_cycle_time = datetime.datetime.now()
                # 1. Construir Prompt para LLM
                tool_desc = get_tool_descriptions()
                prompt_messages = prompt_builder.build_react_prompt(\
                    current_objective, self._history, self.system_prompt, tool_desc, agent_logger\
                )

                # 2. Chamar LLM
                try:
                    llm_response_raw = self._call_llm(prompt_messages)
                    agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
                except Exception as e:
                    agent_logger.exception(f"{log_prefix} Exceção durante a chamada LLM: {e}")
                    llm_response_raw = f"Erro: Falha na chamada LLM: {e}" # Define a mensagem de erro padrão

                log_llm_response = llm_response_raw[:1000] + ('...' if len(llm_response_raw) > 1000 else '')
                agent_logger.debug(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{log_llm_response}\n---")
                # --- Handle LLM Call Errors (Using handler) ---
                if not llm_response_raw or llm_response_raw.startswith("Erro:"):
                    should_continue, error_msg = agent_error_handler.handle_llm_call_error(\
                        agent_instance=self,\
                        llm_error_msg=llm_response_raw,\
                        current_history=self._history,\
                        current_iteration=cycle_num,\
                        max_iterations=self.max_iterations,\
                        is_meta_objective=is_meta_objective,\
                        meta_depth=meta_depth\
                    )
                    if not should_continue:
                        final_response = error_msg
                        break # Exit loop
                    else:
                        continue # Proceed to next iteration

                self._history.append(llm_response_raw) # Append raw response only if no fatal error

                # 3. Parsear Resposta LLM
                try:
                    thought, action_name, action_input = agent_parser.parse_llm_response(llm_response_raw, agent_logger)
                    if not action_name:
                        raise ValueError("JSON válido, mas chave 'Action' obrigatória está ausente.")
                except (json.JSONDecodeError, ValueError) as parse_error:
                    # --- Handle Parsing Errors (Using handler) ---
                    should_continue, error_msg = agent_error_handler.handle_parsing_error(\
                        agent_instance=self,\
                        parse_error=parse_error,\
                        current_history=self._history,\
                        current_iteration=cycle_num,\
                        max_iterations=self.max_iterations,\
                        is_meta_objective=is_meta_objective,\
                        meta_depth=meta_depth\
                    )
                    if not should_continue:
                        final_response = error_msg
                        break # Exit loop
                    else:
                        continue # Proceed to next iteration

                agent_logger.info(f"{log_prefix} Ação Decidida: {action_name}, Input: {action_input}")

                # 4. Executar Ação ou Finalizar
                observation_str = None # Initialize observation
                tool_result = None # Initialize tool result

                if action_name == "final_answer":
                    final_answer_text = action_input.get("answer", "Finalizado sem resposta específica.")
                    agent_logger.info(f"{log_prefix} Ação Final: {final_answer_text}")
                    self._history.append(f"Final Answer: {final_answer_text}")
                    final_response = final_answer_text # Set the actual final response
                    break # Exit loop successfully

                elif action_name in self.tools:
                    # Store code before execution attempt
                    if action_name == "execute_code":
                        self._last_executed_code = action_input.get("code")
                    else:
                        self._last_executed_code = None # Reset if not executing code

                    # Execute tool
                    tool_result = tool_executor.execute_tool(\
                        tool_name=action_name,\
                        action_input=action_input,\
                        tools_dict=self.tools,\
                        agent_logger=agent_logger\
                    )

                    # Lógica de guardar código em memória (mantida aqui para simplicidade)
                    if tool_result.get('status') == 'success' and action_name in ["generate_code", "modify_code"]:
                        code = tool_result.get('data', {}).get('code') or tool_result.get('data', {}).get('modified_code')
                        if code:
                            self._memory['last_code'] = code
                            agent_logger.info(f"{log_prefix} Saved code from '{action_name}' to memory['last_code'].")
                        else:
                            agent_logger.warning(f"{log_prefix} Action '{action_name}' succeeded but no 'code' or 'modified_code' found in data.")

                    # --- Attempt Auto-Correction (Using handler) ---
                    # Pass meta_depth, not meta_depth + 1, as it's the *current* depth check
                    autocorrect_observation = agent_autocorrect.try_autocorrect(\
                        agent_instance=self,\
                        tool_result=tool_result,\
                        last_executed_code=self._last_executed_code,\
                        current_history=self._history, # Pass history *before* adding current observation
                        meta_depth=meta_depth
                    )

                    if autocorrect_observation:
                        observation_str = autocorrect_observation # Use the result from auto-correct
                        self._last_executed_code = None # Reset last executed code after correction attempt
                    else:
                        # If auto-correct didn't run or apply, format the original tool result
                        try:
                            observation_str = f"Observation: {json.dumps(tool_result, ensure_ascii=False)}"
                        except TypeError as json_err:
                            agent_logger.error(f"{log_prefix} Failed to serialize original tool_result to JSON: {json_err}. Result: {tool_result}")
                            observation_str = f"Observation: [Error serializing tool result: {json_err}]"
                else:
                    # Ferramenta não encontrada
                    tool_result = {"status": "error", "action": "tool_not_found", "data": {"message": f"A ferramenta '{action_name}' não existe."}}
                    observation_str = f"Observation: {json.dumps(tool_result)}" # Format as observation

                # --- Append Observation to History ---
                if observation_str: # Ensure we have an observation string
                    # The handlers/formatters should already include "Observation: " prefix
                    agent_logger.debug(f"[DEBUG HISTORY APPEND] Appending observation: '{observation_str[:200]}...'")
                    self._history.append(observation_str)
                else:
                    agent_logger.error(f"{log_prefix} Observation string was unexpectedly None after tool execution/auto-correct attempt.")
                    self._history.append("Observation: [Internal Error: Failed to generate observation]")

                # --- Trim History ---
                self._history = history_manager.trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

                # Check iteration limits AFTER appending the observation
                if cycle_num >= self.max_iterations:
                    agent_logger.warning(f"{log_prefix} Máximo de iterações ({self.max_iterations}) atingido. Finalizando ciclo.")
                    last_observation = self._history[-1] if self._history else "Nenhuma observação disponível."
                    final_response = f"Erro: Máximo de iterações ({self.max_iterations}) atingido. Última observação: {last_observation}"
                    break # Exit loop

            # --- End of ReAct Cycle Loop ---

        except Exception as main_loop_err:
            agent_logger.exception(f"{log_prefix_base} Exceção inesperada no loop principal ReAct: {main_loop_err}")
            final_response = f"Erro Interno Crítico no Agente: {main_loop_err}"

        finally:
            # --- Save State --- 
            # Always save state, regardless of whether it's a meta-cycle or how the loop ended
            agent_logger.info(f"{log_prefix_base} Salvando estado final do agente (ID: {AGENT_STATE_ID}).")
            save_agent_state(AGENT_STATE_ID, self._memory)
            agent_logger.info(f"{log_prefix_base} Ciclo ReAct finalizado. Retornando: '{final_response[:100]}...'")

        return final_response # Return the final determined response

    # --- _call_llm (Mantido como método interno) ---
    def _call_llm(self, messages: list[dict]) -> str:
        """Chama o LLM local com a lista de mensagens e força a saída JSON usando o schema."""
        # (Lógica de _call_llm permanece a mesma)
        payload = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1500,
            "stream": False
        }

        if LLM_RESPONSE_SCHEMA:
            payload["response_format"] = {
                "type": "json_object",
                "schema": LLM_RESPONSE_SCHEMA
            }
        else:
            agent_logger.warning("[ReactAgent LLM Call] JSON Schema não carregado. A saída do LLM não será forçada.")

        headers = LLAMA_DEFAULT_HEADERS
        try:
            response = requests.post(self.llm_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()

            if 'choices' in response_data and response_data['choices']:
                message = response_data['choices'][0].get('message', {})
                content = message.get('content', '').strip()
                if content:
                    # Ensure the content is a valid JSON string if schema was used
                    if LLM_RESPONSE_SCHEMA:
                        try:
                            json.loads(content) # Try parsing to validate
                        except json.JSONDecodeError as e:
                            agent_logger.error(f"[ReactAgent LLM Call ERROR] LLM response is not valid JSON despite schema enforcement: {e}. Content: {content[:500]}...")
                            return f"Erro: LLM retornou JSON inválido: {e}" # Return specific error
                    return content
                else:
                    agent_logger.error(f"[ReactAgent LLM Call ERROR] Resposta LLM OK, mas 'content' está vazio. Resposta: {response_data}")
                    return "Erro: LLM retornou resposta sem conteúdo."
            else:
                agent_logger.error(f"[ReactAgent LLM Call ERROR] Resposta LLM OK, mas formato inesperado. Resposta: {response_data}")
                return "Erro: LLM retornou resposta em formato inesperado."
        except requests.exceptions.RequestException as e:
            agent_logger.error(f"[ReactAgent LLM Call ERROR] Falha ao conectar/comunicar com LLM em {self.llm_url}: {e}")
            return f"Erro: Falha ao conectar com o servidor LLM ({e})."
        except Exception as e:
            agent_logger.exception("[ReactAgent LLM Call ERROR] Erro inesperado ao chamar LLM:")
            return f"Erro: Erro inesperado durante a chamada do LLM ({e})."

    # <<< REMOVED METHOD: _execute_tool >>>
    # <<< REMOVED METHOD: _trim_history >>>
