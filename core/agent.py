import logging
import re
import json
import datetime
import os
from typing import Tuple, Optional, Dict, Any

import requests

# Ajuste para importar skills e core do diretório pai (se necessário, ajuste conforme sua estrutura)
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Local imports (ajuste os caminhos se necessário)
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS
from core.tools import TOOLS
# from core.llm_client import call_llm # Comentei pois _call_llm está definido na classe
from skills.memory import skill_recall_memory
from core.db_utils import save_agent_state, load_agent_state # AGENT_STATE_ID é carregado depois

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente (pode vir de config ou DB utils)
AGENT_STATE_ID = 1 # Ou carregue de outra forma


# --- Classe ReactAgent ---
class ReactAgent:
    def __init__(self, llm_url: str, system_prompt: str):
        """Inicializa o Agente ReAct."""
        self.llm_url = llm_url
        self.system_prompt = system_prompt
        self._history = [] # Histórico de Thought, Action, Observation
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega estado/memória inicial
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. Memória carregada: {list(self._memory.keys())}")

    def _build_react_messages(self, objective: str) -> list[dict]:
        """Constrói a lista de mensagens para o LLM com base no objetivo e histórico."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Adiciona objetivo (pode ser principal ou meta)
        messages.append({"role": "user", "content": f"Meu objetivo atual é: {objective}"})

        # Processa histórico ReAct
        if self._history:
             assistant_turn_parts = []
             for entry in self._history:
                 # Agrupa Thought/Action/Input como 'assistant'
                 if entry.startswith("Thought:") or entry.startswith("Action:") or entry.startswith("Action Input:"):
                     assistant_turn_parts.append(entry)
                 # Trata Observation como 'user' (input do ambiente)
                 elif entry.startswith("Observation:"):
                     if assistant_turn_parts:
                          messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})
                          assistant_turn_parts = []
                     messages.append({"role": "user", "content": entry}) # Observation vem do 'user' (ambiente)

             # Adiciona partes restantes do assistente se o histórico não terminar com Observation
             if assistant_turn_parts:
                  messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})

        return messages

    def _parse_llm_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Parses the LLM's raw response string to extract the thought, action, action input, or final answer.
        Returns: thought, action, action_input, final_answer
        """
        agent_logger.debug(f"[Agent Parse DEBUG] Raw LLM Response:\n{response}")
        thought = None
        action = None
        action_input = None
        final_answer = None

        # Extrai Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|Final Answer:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        else: agent_logger.warning("[Agent Parse WARN] Could not extract Thought.")

        # Extrai Action e Action Input (prioriza JSON em blocos)
        action_match = re.search(r"Action:\s*([\w_]+)", response, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.*)", response, re.DOTALL | re.IGNORECASE)

        if action_match:
            action = action_match.group(1).strip()
            agent_logger.info(f"[Agent Parse INFO] Action extracted: '{action}'")

            if action_input_match:
                raw_action_input_str = action_input_match.group(1).strip()
                agent_logger.debug(f"[Agent Parse DEBUG] Raw Action Input string: '{raw_action_input_str}'")

                json_str_to_parse = None
                json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_action_input_str, re.DOTALL)
                curly_brace_match = re.search(r"(\{.*?\})\s*$", raw_action_input_str, re.DOTALL)

                if json_block_match: json_str_to_parse = json_block_match.group(1)
                elif curly_brace_match: json_str_to_parse = curly_brace_match.group(1)

                if json_str_to_parse:
                    agent_logger.debug(f"[Agent Parse DEBUG] JSON String found: '{json_str_to_parse}'")
                    # Limpeza de quotes (mantida)
                    cleaned_json_str = re.sub(r'[""‟„〝〞〟＂]', '"', json_str_to_parse)
                    cleaned_json_str = re.sub(r"['‛‚‵′＇]", "'", cleaned_json_str) # Menos comum, mas seguro
                    if cleaned_json_str != json_str_to_parse:
                        agent_logger.info(f"[Agent Parse INFO] Quotes cleaned. String now: '{cleaned_json_str}'")
                        json_str_to_parse = cleaned_json_str

                    try:
                        action_input = json.loads(json_str_to_parse)
                        agent_logger.info("[Agent Parse INFO] JSON parsed successfully.")
                    except json.JSONDecodeError as json_e:
                        agent_logger.warning(f"[Agent Parse WARN] Failed JSON decode: {json_e}. Trying fix...")
                        # Fallback: Remove trailing comma
                        json_str_fixed = re.sub(r",\s*(\}|\])$", r"\1", json_str_to_parse.strip())
                        if json_str_fixed != json_str_to_parse.strip():
                            agent_logger.info("[Agent Parse INFO] Attempting parse after removing trailing comma.")
                            try:
                                action_input = json.loads(json_str_fixed)
                                agent_logger.info("[Agent Parse INFO] JSON parsed successfully after fix.")
                            except Exception as json_e_fix:
                                agent_logger.error(f"[Agent Parse ERROR] Failed JSON decode even after fix: {json_e_fix}")
                        else:
                            agent_logger.warning("[Agent Parse WARN] No trailing comma or other JSON error persists.")
                else:
                    # Se Action é final_answer, permite input não-JSON como fallback
                    if action == "final_answer":
                         # Usa o raw_action_input_str como a resposta se não for JSON
                         action_input = {"answer": raw_action_input_str}
                         agent_logger.info("[Agent Parse INFO] Treated non-JSON Action Input as final answer text.")
                    else:
                         agent_logger.warning("[Agent Parse WARN] Action Input found, but not in JSON format. Treating as None for non-final_answer.")
            else:
                 agent_logger.info("[Agent Parse INFO] Action found, but no Action Input provided.")
                 # Se for final_answer sem input, é um erro de formato do LLM
                 if action == "final_answer":
                      agent_logger.warning("[Agent Parse WARN] 'final_answer' action received without Action Input.")
                      action_input = {"answer": "Erro: Recebi instrução para finalizar, mas sem resposta."} # Define uma resposta de erro
                 else:
                      action_input = {} # Assume que outras actions podem não precisar de input

        # Verifica se o Action Input deveria ser um dict (quase sempre, exceto talvez em erros de parse)
        if action_input is not None and not isinstance(action_input, dict):
             agent_logger.error(f"[Agent Parse ERROR] Parsed action_input is not None but not a dict: {type(action_input)}. Resetting to None.")
             action_input = None # Reseta para None se não for dict, para evitar erros

        # Detecção de Final Answer (se Action não foi 'final_answer') - Pode ser redundante agora
        # final_answer_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        # if final_answer_match and action != "final_answer":
        #    final_answer = final_answer_match.group(1).strip()
        #    agent_logger.info(f"[Agent Parse INFO] Standalone Final Answer detected: '{final_answer}'")
        #    # Anula action/input se final_answer for encontrado separadamente? Ou deixa o parser tratar?
        #    # Por ora, vamos assumir que se action for parseado, ele tem prioridade.

        # Se nenhuma ação foi parseada, verifica se a resposta inteira pode ser a resposta final
        if not action and not final_answer:
             # Verifica se a resposta NÃO se parece com o formato Thought/Action
             if not response.strip().startswith("Thought:") and not "Action:" in response:
                  agent_logger.info("[Agent Parse INFO] Assuming entire response is Final Answer (no Thought/Action found).")
                  final_answer = response.strip()
                  # Cria um action_input correspondente
                  action = "final_answer"
                  action_input = {"answer": final_answer}
             else:
                  agent_logger.warning("[Agent Parse WARN] Could not extract Action or Final Answer, and response seems structured.")

        return thought, action, action_input, final_answer

    # --- run (Versão Híbrida com Injeção de Meta-Objetivo) ---
    def run(self, objective: str, is_meta_objective: bool = False, meta_depth: int = 0) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""
        log_prefix_base = f"[ReactAgent{' META' if is_meta_objective else ''}]"

        # Log inicial e limpeza de histórico apenas para o objetivo principal
        if not is_meta_objective:
             agent_logger.info(f"\n{log_prefix_base} Iniciando ciclo ReAct para objetivo principal: '{objective}'")
             self._history = [] # Limpa histórico ReAct

             # --- CONSULTA PRÉVIA À MEMÓRIA (Apenas para objetivo principal) ---
             initial_memory_observation = None
             is_question = objective.strip().endswith("?") or objective.lower().strip().startswith(("qual", "quais", "quem", "onde", "como", "por que", "quando"))

             if is_question:
                 agent_logger.info(f"{log_prefix_base} [PRE-FETCH] Objetivo parece pergunta. Consultando memória...")
                 try:
                     memory_result = skill_recall_memory(
                         action_input={"query": objective, "max_results": 2},
                         agent_memory=self._memory,
                         agent_history=None
                     )
                     if memory_result.get("status") == "success":
                         results = memory_result.get("data", {}).get("results", [])
                         if results:
                              formatted_contents = [f"  - (Dist: {item.get('distance', 'N/A'):.4f}): {item.get('content', 'N/A')}" for item in results]
                              initial_memory_observation = "Contexto Preliminar da Memória:\n" + "\n".join(formatted_contents)
                         else:
                              initial_memory_observation = "Contexto Preliminar da Memória: Nenhuma informação relevante encontrada."
                     else:
                          initial_memory_observation = f"Contexto Preliminar da Memória: Erro ao consultar ({memory_result.get('error', '?')})."
                 except Exception as pre_mem_err:
                     initial_memory_observation = f"Contexto Preliminar da Memória: Exceção ({pre_mem_err})."

                 agent_logger.info(f"{log_prefix_base} [PRE-FETCH] {initial_memory_observation}")
                 self._history.append(f"Observation: {initial_memory_observation}")
             # --- FIM DA CONSULTA PRÉVIA ---

        # O objetivo atual pode ser o original ou um meta-objetivo injetado
        current_objective = objective

        # --- CICLO ReAct SIMPLES ---
        for i in range(MAX_REACT_ITERATIONS):
            start_cycle_time = datetime.datetime.now()
            log_prefix = f"{log_prefix_base} Cycle {i+1}/{MAX_REACT_ITERATIONS}" # Adiciona número do ciclo
            agent_logger.info(f"\n{log_prefix} (Objetivo: '{current_objective[:60]}...' Inicio: {start_cycle_time.strftime('%H:%M:%S.%f')[:-3]})")

            # 1. Construir Prompt
            messages = self._build_react_messages(current_objective)

            # 2. Chamar LLM
            response_text = ""
            try:
                 start_llm_time = datetime.datetime.now()
                 response_text = self._call_llm(messages)
                 end_llm_time = datetime.datetime.now()
                 agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(end_llm_time - start_llm_time).total_seconds():.3f}s)")
                 agent_logger.info(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{response_text}\n---")
            except Exception as llm_err:
                 agent_logger.error(f"{log_prefix} Erro na chamada LLM: {llm_err}")
                 return f"Erro fatal na comunicação com o LLM: {llm_err}"

            # 3. Parsear Resposta
            thought, action_name, action_input, final_answer = self._parse_llm_response(response_text)

            # Adiciona Thought/Action/Input ao histórico
            if thought: self._history.append(f"Thought: {thought}")
            if action_name:
                 self._history.append(f"Action: {action_name}")
                 try: action_input_json = json.dumps(action_input, ensure_ascii=False) if action_input is not None else "{}"
                 except Exception: action_input_json = str(action_input)
                 self._history.append(f"Action Input: {action_input_json}")
            elif final_answer:
                 self._history.append(f"Final Answer (detected directly): {final_answer}")
                 # Se temos final_answer direto, forçamos a ação e input correspondentes
                 action_name = "final_answer"
                 if not isinstance(action_input, dict) or "answer" not in action_input:
                      action_input = {"answer": final_answer}
            else:
                 agent_logger.error(f"{log_prefix} Falha ao parsear Ação ou Resposta Final.")
                 self._history.append(f"LLM Raw Response (unparseable): {response_text}")
                 self._history.append("Observation: Erro: Não consegui entender a resposta do LLM.")
                 continue

            # 4. Executar Ação ou Finalizar
            if action_name == "final_answer":
                 final_response = action_input.get("answer", "Erro: Resposta final vazia.")
                 agent_logger.info(f"\n{log_prefix} Resposta Final Decidida pelo LLM: {final_response}")
                 self._history.append("Observation: Resposta final fornecida.")
                 # Retorna resultado (pode ser de meta-objetivo ou principal)
                 if is_meta_objective:
                      if "corrigido" in final_response.lower() or "sucesso" in final_response.lower():
                           return f"Meta-Correção CONCLUÍDA: {final_response}"
                      else:
                           return f"Meta-Correção FALHOU: {final_response}"
                 else:
                      return final_response

            # 5. Executar Ferramenta (Se não for final_answer)
            observation_str = self._execute_tool(
                tool_name=action_name,
                action_input=action_input if action_input is not None else {},
                current_objective=current_objective,
                current_history=self._history
            )

            # Adiciona Observação (será analisada no próximo ciclo ou pela injeção de meta-objetivo)
            agent_logger.info(f"{log_prefix} Observação: {observation_str}")
            self._history.append(f"Observation: {observation_str}")


            # <<< INÍCIO: DETECÇÃO DE ERRO INTERNO E INJEÇÃO DE META-OBJETIVO >>>
            # Executa APENAS se NÃO estivermos já em um meta-objetivo E se houve erro
            if not is_meta_objective and observation_str.startswith("Erro ao executar a ferramenta"):
                 # Heurística para detectar erro interno de skill Python
                 match_internal_error = re.search(r"(TypeError|AttributeError|NameError|IndexError|KeyError|OperationalError|ValueError|FileNotFoundError).*?(skills/[\w_/]+\.py)", observation_str, re.IGNORECASE | re.DOTALL)
                 if match_internal_error:
                     error_type = match_internal_error.group(1)
                     skill_file = match_internal_error.group(2)
                     agent_logger.warning(f"{log_prefix} Detectado erro interno ({error_type}) em '{skill_file}'. Tentando Meta-Correção.")

                     # Define o meta-objetivo
                     meta_objective = f"CORRIGIR BUG: A ferramenta '{action_name}' falhou com erro '{error_type}' originado em '{skill_file}'. Analise a observação anterior (erro), leia '{skill_file}', identifique a causa e use 'modify_code' para corrigir."

                     # Chama recursivamente run() com o meta-objetivo
                     MAX_META_DEPTH = 1 # Limita a 1 nível de correção por erro original
                     if meta_depth < MAX_META_DEPTH:
                          agent_logger.info(f"{log_prefix} Iniciando sub-ciclo ReAct para: '{meta_objective}'")
                          # Passa o histórico atual; o sub-ciclo adicionará a ele
                          correction_result = self.run(objective=meta_objective, is_meta_objective=True, meta_depth=meta_depth + 1)

                          # Adiciona o resultado da correção como uma nova observação
                          meta_obs = f"Observation from Meta-Correction: {correction_result}"
                          self._history.append(meta_obs)
                          agent_logger.info(f"{log_prefix} Sub-ciclo Meta-Correção concluído. Resultado adicionado à observação.")

                          # CONTINUA o loop principal. O LLM verá o resultado da correção.
                          agent_logger.info(f"{log_prefix} Retomando objetivo original: '{current_objective[:60]}...'")
                          continue # Pula para o próximo ciclo do objetivo original
                     else:
                          agent_logger.error(f"{log_prefix} Profundidade máxima de meta-correção ({MAX_META_DEPTH}) atingida. Correção abortada.")
                          # Modifica a observação de erro para indicar falha na correção
                          self._history[-1] = observation_str + f" (Falha na tentativa de auto-correção: profundidade máxima {MAX_META_DEPTH} atingida.)"
                          # Continua o ciclo normal com a observação de erro modificada

            # <<< FIM: DETECÇÃO DE ERRO INTERNO E INJEÇÃO DE META-OBJETIVO >>>


            end_cycle_time = datetime.datetime.now()
            cycle_duration = end_cycle_time - start_cycle_time
            agent_logger.info(f"--- Fim {log_prefix} (Duração Total: {cycle_duration.total_seconds():.3f}s) ---")
            # O loop continua para o próximo ciclo do objetivo ATUAL

        # Se sair do loop por limite de iterações
        agent_logger.warning(f"{log_prefix_base} Limite de iterações atingido para objetivo '{current_objective[:60]}...'.")
        self._history.append("Observation: Limite de iterações atingido sem resposta final.")
        if is_meta_objective:
            return f"Meta-Objetivo '{current_objective[:60]}...' FALHOU: Limite de iterações."
        else:
            return "Desculpe, não consegui completar a tarefa após várias tentativas."


    # --- _call_llm (Mantido como antes) ---
    def _call_llm(self, messages: list[dict]) -> str:
        # (Código da função _call_llm exatamente como na sua versão anterior)
        agent_logger.info(f"[ReactAgent] Chamando LLM ({len(messages)} mensagens)...")
        headers = {"Content-Type": "application/json"}
        chat_url = self.llm_url
        if not chat_url.endswith("/chat/completions"):
             if chat_url.endswith("/v1") or chat_url.endswith("/v1/"): chat_url = chat_url.rstrip('/') + "/chat/completions"
             else: agent_logger.warning(f"[ReactAgent WARN] URL LLM '{self.llm_url}' pode não ser para /v1/chat/completions...")
        payload = {"messages": messages, "temperature": 0.1, "max_tokens": 512, "stream": False}
        try:
             agent_logger.debug(f"[ReactAgent DEBUG] Enviando para URL: {chat_url}")
             response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
             response.raise_for_status()
             response_data = response.json()
             content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
             if not content: agent_logger.warning("[ReactAgent WARN] LLM retornou conteúdo vazio.")
             return content
        except requests.exceptions.Timeout:
             agent_logger.error(f"[ReactAgent LLM ERROR] Timeout ao chamar LLM em {chat_url}")
             return "Thought: Timeout na comunicação com LLM. Action: final_answer Action Input: {\"answer\": \"Desculpe, demorei muito para pensar.\"}"
        except requests.exceptions.RequestException as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro na requisição LLM para {chat_url}: {e}")
             if e.response is not None and e.response.status_code == 404:
                  return f"Thought: Endpoint LLM não encontrado. Action: final_answer Action Input: {{\"answer\": \"Erro: Endpoint LLM não encontrado ({chat_url}).\"}}"
             return f"Thought: Erro de comunicação com LLM. Action: final_answer Action Input: {{\"answer\": \"Desculpe, erro ao conectar ao LLM ({e}).\"}}"
        except Exception as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro inesperado na chamada LLM: {e}", exc_info=True)
             return "Thought: Erro inesperado na chamada LLM. Action: final_answer Action Input: {\"answer\": \"Desculpe, ocorreu um erro interno inesperado.\"}"


    # --- _execute_tool (Mantido como antes, com formatação de Obs para recall_memory) ---
    def _execute_tool(self, tool_name: str, action_input: dict, current_objective: str, current_history: list) -> str:
        # (Código da função _execute_tool exatamente como na sua versão anterior,
        # incluindo a lógica que adicionamos para formatar a observação de recall_memory)
        agent_logger.info(f"[ReactAgent] Executando ferramenta: {tool_name} com input: {action_input}")
        if tool_name not in TOOLS:
            return f"Erro: A ferramenta '{tool_name}' não existe. Ferramentas disponíveis: {', '.join(TOOLS.keys())}"

        tool_info = TOOLS[tool_name]
        tool_function = tool_info.get("function")
        if not tool_function: return f"Erro: Ferramenta '{tool_name}' configurada incorretamente (sem função)."

        required_params = tool_info.get("parameters", {}).get("required", [])
        missing_params = [p for p in required_params if p not in action_input]
        if missing_params: return f"Erro: Parâmetros obrigatórios ausentes para {tool_name}: {', '.join(missing_params)}. Input: {action_input}"

        observation = ""
        try:
            result = tool_function(
                action_input=action_input,
                agent_memory=self._memory,
                agent_history=current_history
            )
            agent_logger.info(f"[ReactAgent] Resultado da Ferramenta ({tool_name}): {result}")

            status = result.get("status", "error")
            result_data = result.get("data", {})
            message = result_data.get("message", f"Ferramenta {tool_name} executada.")
            skill_action = result.get("action") # Ação específica realizada pela skill

            if status == "success":
                # Atualiza memória se for ação de código
                updated_memory = False
                if skill_action in ["code_generated", "code_modified"]:
                    new_code = result_data.get("modified_code") or result_data.get("code")
                    new_lang = result_data.get("language")
                    if new_code is not None and self._memory.get('last_code') != new_code:
                        self._memory['last_code'] = new_code
                        self._memory['last_lang'] = new_lang if new_lang else self._memory.get('last_lang')
                        agent_logger.info("[ReactAgent MEM] Memória do agente atualizada (código).")
                        updated_memory = True
                    elif new_lang is not None and self._memory.get('last_lang') != new_lang:
                         self._memory['last_lang'] = new_lang
                         agent_logger.info(f"[ReactAgent MEM] Memória do agente atualizada (linguagem: {new_lang}).")
                         updated_memory = True
                if updated_memory:
                    try: save_agent_state(AGENT_STATE_ID, self._memory)
                    except Exception as db_save_err: agent_logger.error(f"[ReactAgent ERROR] Falha ao salvar estado no DB: {db_save_err}")

                # Formata Observação
                observation_parts = [message]
                if skill_action == "code_executed":
                    out = result_data.get("output", "").strip()
                    err = result_data.get("stderr", "").strip()
                    if out: observation_parts.append(f"Saída (stdout):\n```\n{out}\n```")
                    if err: observation_parts.append(f"Erro/Aviso (stderr):\n```\n{err}\n```")
                    if not out and not err: observation_parts.append("(Execução sem saída)")
                elif skill_action in ["code_generated", "code_modified"]:
                    code = result_data.get("modified_code") or result_data.get("code")
                    lang = result_data.get("language", self._memory.get('last_lang') or "text")
                    if code: observation_parts.append(f"Código {'Modificado' if skill_action=='code_modified' else 'Gerado'}:\n```{lang}\n{code}\n```")
                elif skill_action == "web_search_completed" and isinstance(result_data.get("results"), list):
                     snippets = [f"- {r.get('title', 'N/T')}: {r.get('snippet', 'N/A')[:100]}..." for r in result_data["results"]]
                     if snippets: observation_parts.append("Resultados (snippets):\n" + "\n".join(snippets))
                elif skill_action == "memory_recalled": # <<< FORMATAÇÃO PARA RECALL >>>
                    recalled_results = result_data.get("results", [])
                    if recalled_results:
                        formatted = [f"  - Memória {i+1} (ID: {item.get('rowid', 'N/A')}, Dist: {item.get('distance', -1):.4f}): {item.get('content', '?')}" for i, item in enumerate(recalled_results)]
                        observation_parts = [f"A ferramenta 'recall_memory' recuperou {len(recalled_results)} item(ns) relevante(s) da memória:"] + formatted
                    # else: a 'message' já diz que não achou nada
                elif skill_action == "file_read": # <<< FORMATAÇÃO PARA READ_FILE >>>
                     filename = result_data.get("file_name", "?")
                     content_preview = result_data.get("content", "")[:500] # Pega prévia do conteúdo lido
                     if len(result_data.get("content", "")) > 500: content_preview += "..."
                     observation_parts = [f"Conteúdo do arquivo '{filename}' lido com sucesso. Prévia:\n```\n{content_preview}\n```"]
                # Adicione outros elif para skill_action se precisar de formatação específica

                observation = "\n".join(observation_parts)

            elif status == "error":
                error_message = result_data.get("message", f"Erro desconhecido ({tool_name}).")
                observation = f"Erro ao executar a ferramenta {tool_name}: {error_message}"
                # Adiciona stderr se for erro de execução de código
                if skill_action == "execute_code_failed":
                     stderr_content = result_data.get("stderr", "")
                     if stderr_content: observation += f"\nSaída de Erro (stderr):\n```\n{stderr_content.strip()}\n```"
            else:
                observation = f"Aviso/Info da ferramenta {tool_name}: {message}"

        except Exception as e:
            agent_logger.error(f"[ReactAgent EXEC ERROR] Exceção inesperada ao executar {tool_name}: {e}", exc_info=True)
            observation = f"Erro inesperado (exceção) ao executar a ferramenta {tool_name}: {e}"

        MAX_OBSERVATION_LEN = 1500 # Mantém limite
        if len(observation) > MAX_OBSERVATION_LEN:
            observation = observation[:MAX_OBSERVATION_LEN] + "... (Observação truncada)"

        return observation

    # --- _trim_history (Mantido como antes) ---
    def _trim_history(self):
        # (Código da função _trim_history exatamente como na sua versão anterior)
        max_keep = 1 + (MAX_HISTORY_TURNS * 2)
        if len(self._history) > max_keep:
            agent_logger.debug(f"Trimming history from {len(self._history)} entries.")
            self._history = [self._history[0]] + self._history[-(max_keep-1):]
            agent_logger.debug(f"History trimmed to {len(self._history)} entries.")