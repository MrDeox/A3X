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
from skills.memory import skill_recall_memory
from core.db_utils import save_agent_state, load_agent_state

# <<< NEW IMPORTS for modular functions >>>
from core import agent_parser, prompt_builder, history_manager, tool_executor

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
        self._last_error_type = None # <<< REMOVED _last_skill_file, not used >>>
        self._last_executed_code = None
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. Memória carregada: {list(self._memory.keys())}")

    # <<< REMOVED METHOD: _build_react_messages >>>
    # <<< REMOVED METHOD: _parse_llm_response >>>

    # --- run (Refatorado para usar funções modulares) ---
    def run(self, objective: str, is_meta_objective: bool = False, meta_depth: int = 0) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""

        # --- Limite de Profundidade Meta ---
        if meta_depth > MAX_META_DEPTH:
            agent_logger.warning(f"[ReactAgent META-{meta_depth}] Max meta depth ({MAX_META_DEPTH}) reached. Aborting meta-cycle for objective: '{objective[:100]}...'")
            return f"Erro: Profundidade máxima de auto-correção ({MAX_META_DEPTH}) atingida."

        # --- Logging & Setup ---
        if is_meta_objective:
            log_prefix_base = f"[ReactAgent META-{meta_depth}]"
            current_objective = objective
        else:
            log_prefix_base = "[ReactAgent]"
            self._history = []
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

            # 1. Construir Prompt para LLM (usando função externa)
            tool_desc = get_tool_descriptions() # Obter descrições das ferramentas
            prompt_messages = prompt_builder.build_react_prompt(
                current_objective, self._history, self.system_prompt, tool_desc, agent_logger
            )

            # 2. Chamar LLM (método interno mantido)
            try:
                llm_response_raw = self._call_llm(prompt_messages)
                agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
            except Exception as e:
                agent_logger.exception(f"{log_prefix} Exceção durante a chamada LLM: {e}")
                llm_response_raw = f"Erro: Falha na chamada LLM: {e}" # Define a mensagem de erro padrão

            log_llm_response = llm_response_raw[:1000] + ('...' if len(llm_response_raw) > 1000 else '')
            agent_logger.debug(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{log_llm_response}\n---")

            if not llm_response_raw or llm_response_raw.startswith("Erro:"):
                agent_logger.error(f"{log_prefix} Erro Fatal: _call_llm retornou erro ou resposta vazia: '{llm_response_raw}'")
                self._history.append(f"Observation: Erro crítico na comunicação com o LLM: {llm_response_raw}")
                if i == self.max_iterations - 1:
                    return f"Desculpe, ocorreu um erro na comunicação com o LLM: {llm_response_raw}"
                else:
                    continue

            agent_logger.debug(f"[HISTORY DEBUG] Appending LLM Response. Current len: {len(self._history)}")
            self._history.append(llm_response_raw)

            # 3. Parsear Resposta LLM (usando função externa)
            try:
                thought, action_name, action_input = agent_parser.parse_llm_response(llm_response_raw, agent_logger)
                if not action_name:
                     agent_logger.error(f"{log_prefix} Parsing ok, mas 'Action' não encontrada no JSON.")
                     raise ValueError("JSON válido, mas chave 'Action' obrigatória está ausente.")
            except (json.JSONDecodeError, ValueError) as parse_error:
                agent_logger.error(f"{log_prefix} Falha ao parsear resposta JSON do LLM: {parse_error}")
                observation_msg = f"Observation: Erro crítico - sua resposta anterior não estava no formato JSON esperado ou faltava a chave 'Action'. Verifique o formato. Detalhe: {parse_error}"
                self._history.append(observation_msg)
                agent_logger.debug(f"{log_prefix} Added parse error observation: {observation_msg}")
                if i == self.max_iterations - 1:
                    agent_logger.warning(f"{log_prefix} Última iteração falhou no parsing. Finalizando com erro.")
                    save_agent_state(AGENT_STATE_ID, self._memory)
                    return f"Desculpe, falha ao processar a resposta do LLM após {self.max_iterations} tentativas."
                else:
                    agent_logger.info(f"{log_prefix} Tentando continuar após erro de parsing.")
                    continue

            agent_logger.info(f"{log_prefix} Ação Decidida: {action_name}, Input: {action_input}")

            # 4. Executar Ação ou Finalizar
            if action_name == "final_answer":
                final_answer_text = action_input.get("answer", "Finalizado sem resposta específica.")
                agent_logger.info(f"{log_prefix} Ação Final: {final_answer_text}")
                agent_logger.debug(f"[HISTORY DEBUG] Appending Final Answer. Current len: {len(self._history)}")
                self._history.append(f"Final Answer: {final_answer_text}")
                save_agent_state(AGENT_STATE_ID, self._memory)
                return final_answer_text
            elif action_name in self.tools:
                 # Executar ferramenta (usando função externa)
                 tool_result = tool_executor.execute_tool(
                     tool_name=action_name,
                     action_input=action_input,
                     tools_dict=self.tools,
                     agent_logger=agent_logger
                 )

                 # Lógica de guardar código em memória (mantida aqui)
                 if tool_result.get('status') == 'success' and action_name in ["generate_code", "modify_code"]:
                     code = tool_result.get('data', {}).get('code') or tool_result.get('data', {}).get('modified_code')
                     if code:
                         self._memory['last_code'] = code
                         agent_logger.info(f"{log_prefix} Saved code from '{action_name}' to memory['last_code'].")
                     else:
                         agent_logger.warning(f"{log_prefix} Action '{action_name}' succeeded but no 'code' or 'modified_code' found in data.")

                 if action_name == "execute_code":
                     self._last_executed_code = action_input.get("code")

                 observation = json.dumps(tool_result, ensure_ascii=False, default=lambda o: '<not serializable>')

                 # Lógica de Auto-Correção (mantida aqui, usa self.run)
                 is_execution_error = (
                     action_name == "execute_code" and
                     tool_result.get("status") == "error" and
                     tool_result.get("action") == "execution_failed"
                 )
                 if is_execution_error and self._last_executed_code and not is_meta_objective and meta_depth < MAX_META_DEPTH:
                     agent_logger.warning(f"{log_prefix} Erro detectado na execução do código. Iniciando ciclo de auto-correção (Profundidade: {meta_depth + 1}).")
                     error_message = tool_result.get("data", {}).get("message", "Erro desconhecido")
                     stderr_output = tool_result.get("data", {}).get("stderr", "")
                     full_error_details = f"Error Message: {error_message}\nStderr:\n{stderr_output}".strip()

                     meta_objective = f"""AUTO-CORRECTION STEP 1: MODIFY CODE...
                         # (O prompt meta aqui continua o mesmo)
                         {full_error_details}
                         ... [restante do prompt meta]
                         {self._last_executed_code}
                         ..."""
                         # --- Simplified Meta-Objective (Step 1: Modify Only) --- #
                     meta_objective = f"""AUTO-CORRECTION STEP 1: MODIFY CODE

An error occurred while executing the previous code block.

**Error Details:**
---
{full_error_details}
---

**Code that Failed:**
```python
{self._last_executed_code}
```

**Your ONLY task is to propose the correction:**

1.  **Analyze:** Carefully understand the error and the code that caused it.
2.  **Modify:** Use the `modify_code` tool *immediately*. Provide the *entire* original code block above in the `code_to_modify` parameter and describe the *specific change* needed in the `modification` parameter.

**IMPORTANT:** Do NOT use any other tools like `execute_code` or `final_answer` in this step. Your goal is *only* to call `modify_code`.
"""

                     meta_modify_result_dict = self.run(
                         objective=meta_objective,
                         is_meta_objective=True,
                         meta_depth=meta_depth + 1
                     )
                     # ... [lógica de processar meta_modify_result_dict como antes] ...
                     try:
                         if isinstance(meta_modify_result_dict, str) and meta_modify_result_dict.startswith("Erro:"):
                              raise ValueError(f"Meta-cycle returned an error string: {meta_modify_result_dict}")
                         if isinstance(meta_modify_result_dict, str):
                             parsed_result = json.loads(meta_modify_result_dict)
                         elif isinstance(meta_modify_result_dict, dict):
                              parsed_result = meta_modify_result_dict
                         else:
                              raise TypeError(f"Unexpected type returned from meta-cycle: {type(meta_modify_result_dict)}")

                         if parsed_result.get("status") == "success" and parsed_result.get("action") == "code_modified":
                             modified_code = parsed_result.get("data", {}).get("modified_code")
                             if modified_code:
                                 self._memory['last_code'] = modified_code
                                 agent_logger.info(f"{log_prefix} Código modificado obtido do meta-ciclo e salvo em memory['last_code'].")
                                 observation_msg = f"Observation: An execution error occurred. Auto-correction proposed a modification (now stored as last_code):\n```python\n{modified_code}\n```"
                             else:
                                 observation_msg = "Observation: An execution error occurred. Auto-correction attempted modification, but the 'modify_code' skill did not return the modified code."
                                 agent_logger.warning(f"{log_prefix} Meta-ciclo modify_code success status, but no modified_code in data.")
                         else:
                             failure_reason = parsed_result.get("data", {}).get("message", "unknown reason")
                             observation_msg = f"Observation: An execution error occurred. Auto-correction attempt failed during the modification step. Reason: {failure_reason}"
                             agent_logger.warning(f"{log_prefix} Meta-ciclo para modify_code falhou: {failure_reason}")
                     except (json.JSONDecodeError, ValueError, TypeError) as e:
                         agent_logger.error(f"{log_prefix} Falha ao processar resultado do meta-ciclo de modificação: {e}. Raw result: {meta_modify_result_dict}")
                         observation_msg = f"Observation: An execution error occurred. Auto-correction attempt failed due to an internal error processing the modification result: {e}"

                     observation = observation_msg
                     self._last_executed_code = None
            else:
                # Ferramenta não encontrada (já tratada por execute_tool, mas como fallback)
                 tool_result = {"status": "error", "action": "tool_not_found", "data": {"message": f"A ferramenta '{action_name}' não existe."}}
                 observation = json.dumps(tool_result)

            # --- Append Observation to History (mantido como antes) ---
            # Ensure observation is a string before logging/appending
            if isinstance(observation, str) and not observation.startswith("Observation: "):
                observation = f"Observation: {observation}"
            elif not isinstance(observation, str):
                # If it's not a string (e.g., dict from tool), convert it to JSON string with prefix
                try:
                    observation_str = json.dumps(observation)
                    observation = f"Observation: {observation_str}"
                except TypeError as json_err:
                    agent_logger.error(f"[ReactAgent] Failed to serialize observation to JSON before history append: {json_err}. Observation: {observation}")
                    observation = f"Observation: [Error serializing observation: {json_err}]"

            agent_logger.debug(f"[DEBUG HISTORY APPEND] Appending observation: '{str(observation)[:200]}...'") # <<< Log the final string
            self._history.append(observation)

            # --- Trim History (usando função externa) ---
            self._history = history_manager.trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

            # Check iteration limits AFTER appending the observation
            if cycle_num >= self.max_iterations:
                agent_logger.warning(f"[ReactAgent] Cycle {cycle_num}/{self.max_iterations} Máximo de iterações ({self.max_iterations}) atingido. Finalizando ciclo.")
                # Provide the last observation in the error message
                last_observation = self._history[-1] if self._history else "Nenhuma observação disponível."
                final_response = f"Erro: Máximo de iterações ({self.max_iterations}) atingido. Última observação: {last_observation}"
                break # Exit loop

        # Se sair do loop sem final_answer (improvável com boa lógica)
        agent_logger.error(f"{log_prefix_base} Saiu do loop principal sem \"final_answer\" após {self.max_iterations} iterações.")
        save_agent_state(AGENT_STATE_ID, self._memory)
        return final_response

    # --- _call_llm (Mantido como método interno) ---
    def _call_llm(self, messages: list[dict]) -> str:
        """Chama o LLM local com a lista de mensagens e força a saída JSON usando o schema."""
        # (Lógica de _call_llm permanece a mesma)
        # Substituir o placeholder de descrição de ferramentas já é feito no prompt_builder
        payload = {
            "messages": messages, # Usa as mensagens já processadas pelo prompt_builder
            "temperature": 0.5,
            "max_tokens": 1500,
            "stream": False
        }

        if LLM_RESPONSE_SCHEMA:
            payload["response_format"] = {
                "type": "json_object",
                "schema": LLM_RESPONSE_SCHEMA
            }
            # agent_logger.info("[ReactAgent LLM Call] Usando JSON Schema para forçar output.") # Log menos verboso aqui
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
            agent_logger.exception(f"[ReactAgent LLM Call ERROR] Erro inesperado ao chamar LLM:")
            return f"Erro: Erro inesperado durante a chamada do LLM ({e})."

    # <<< REMOVED METHOD: _execute_tool >>>
    # <<< REMOVED METHOD: _trim_history >>>
