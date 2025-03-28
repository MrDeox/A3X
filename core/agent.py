import logging
import re
import json
import datetime
import os
import sys
import traceback
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
        self.tools = TOOLS  # Carrega as ferramentas
        self._history = [] # Histórico de Thought, Action, Observation
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega estado/memória inicial
        self.max_iterations = MAX_REACT_ITERATIONS
        self._last_error_type = None
        self._last_skill_file = None
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
                         action_input = {"final_answer": raw_action_input_str}
                         agent_logger.info("[Agent Parse INFO] Treated non-JSON Action Input as final answer text.")
                    else:
                         agent_logger.warning("[Agent Parse WARN] Action Input found, but not in JSON format. Treating as None for non-final_answer.")
            else:
                agent_logger.info("[Agent Parse INFO] Action found, but no Action Input provided.")
                # Se for final_answer sem input, é um erro de formato do LLM
                if action == "final_answer":
                     agent_logger.warning("[Agent Parse WARN] 'final_answer' action received without Action Input.")
                     action_input = {"final_answer": "Erro: Recebi instrução para finalizar, mas sem resposta."}
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
                  action_input = {"final_answer": final_answer}
             else:
                  agent_logger.warning("[Agent Parse WARN] Could not extract Action or Final Answer, and response seems structured.")

        return thought, action, action_input, final_answer

    # --- run (Versão Híbrida com Injeção de Meta-Objetivo) ---
    def run(self, objective: str, is_meta_objective: bool = False, meta_depth: int = 0) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""

        # --- Logging & Setup ---
        if is_meta_objective:
            log_prefix_base = f"[ReactAgent META-{meta_depth}]"
            current_objective = objective # O meta-objetivo é o objetivo atual
        else:
            log_prefix_base = "[ReactAgent]"
            # Limpa o histórico para um novo objetivo principal
            self._history = []
            self._history.append(f"Human: {objective}")
            current_objective = objective

        agent_logger.info(f"\n{log_prefix_base} Iniciando ciclo ReAct (Objetivo: '{current_objective[:100]}...' Profundidade: {meta_depth})")
        agent_logger.debug(f"{log_prefix_base} Histórico Inicial: {self._history}")

        # Variáveis para override no meta-ciclo
        meta_read_filepath: str | None = None
        meta_read_content: str | None = None

        # --- Ciclo ReAct ---
        for i in range(self.max_iterations):
            cycle_num = i + 1
            log_prefix = f"{log_prefix_base} Cycle {cycle_num}/{self.max_iterations}"
            agent_logger.info(f"\n{log_prefix} (Objetivo: '{current_objective[:60]}...' Inicio: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

            start_cycle_time = datetime.datetime.now()

            # 1. Construir Prompt para LLM
            prompt = self._build_react_messages(current_objective)
            # agent_logger.debug(f"{log_prefix} Prompt para LLM:\\n---\\n{prompt}\\n---\") # Debug verboso

            # 2. Chamar LLM
            llm_response_raw = self._call_llm(prompt)
            agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
            # Limita log da resposta bruta
            log_llm_response = llm_response_raw[:1000] + ('...' if len(llm_response_raw) > 1000 else '')
            agent_logger.debug(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{log_llm_response}\n---")

            if not llm_response_raw:
                agent_logger.error(f"{log_prefix} Erro Fatal: LLM retornou resposta vazia.")
                self._history.append("Observation: Erro crítico - LLM retornou resposta vazia.")
                return "Desculpe, ocorreu um erro interno (LLM retornou vazio)." # Retorna erro

            self._history.append(llm_response_raw) # Adiciona resposta LLM ao histórico

            # 3. Parsear Resposta LLM (Thought, Action, Action Input)
            try:
                # Correção: Desempacotar todos os valores retornados
                thought, action_name, action_input, final_answer_parsed = self._parse_llm_response(llm_response_raw)
                # Se final_answer_parsed for encontrado (mesmo com action), ele tem prioridade?
                # Por ora, vamos priorizar action, mas logar se ambos aparecerem.
                if action_name and final_answer_parsed:
                    agent_logger.warning(f"{log_prefix} LLM retornou tanto Action ({action_name}) quanto Final Answer. Priorizando Action.")
                # Lógica para usar final_answer_parsed se action_name for None
                if not action_name and final_answer_parsed:
                    action_name = "final_answer"
                    action_input = {"final_answer": final_answer_parsed}
                    agent_logger.info(f"{log_prefix} Usando Final Answer parseado pois Action estava ausente.")
                elif not action_name and not final_answer_parsed:
                    raise ValueError("Não foi possível extrair Action nem Final Answer da resposta do LLM.")

                # action_input já é parseado como dict por _parse_llm_response
                # action_input = self._parse_action_input(action_input_str, action_name, log_prefix) # Linha removida
            except ValueError as parse_error:
                agent_logger.error(f"{log_prefix} Falha ao parsear resposta do LLM: {parse_error}")
                self._history.append(f"Observation: Erro ao parsear sua resposta. Verifique o formato 'Action: ... Action Input: {{...}}'. Detalhe: {parse_error}")
                continue # Pula para próximo ciclo

            agent_logger.info(f"{log_prefix} Ação Decidida: {action_name}, Input: {action_input}")

            # <<< INÍCIO: Lógica de Override no Meta-Ciclo >>>
            if is_meta_objective and action_name == "modify_code":
                 # Condição: Já lemos um arquivo (`meta_read_filepath` está setado)
                 # E o LLM está pedindo modify_code.
                 if meta_read_filepath and meta_read_content:
                     agent_logger.info(f"[ReactAgent META] Tentando injetar overrides de modify_code com base na leitura anterior de '{meta_read_filepath}'.")
                     # Força os parâmetros corretos se o LLM não os forneceu como esperado
                     if action_input is None: action_input = {} # Garante que é um dict

                     action_input["target_filepath"] = meta_read_filepath
                     action_input["code_to_modify"] = meta_read_content
                     # A descrição pode vir do LLM ou usamos uma genérica
                     if not action_input.get("target_code_description"):
                          action_input["target_code_description"] = "Code provided via override from preceding read_file"
                     # A modificação DEVE vir do LLM, senão a correção não funciona
                     if "modification" not in action_input:
                          agent_logger.warning("[ReactAgent META] Override: LLM pediu modify_code mas não forneceu 'modification' no Action Input. A skill provavelmente falhará.")
                     agent_logger.debug(f"[ReactAgent META] Action Input para modify_code APÓS override: {json.dumps(action_input, default=lambda o: '<not serializable>')}")

                 # Importante: Resetar após a tentativa de injeção (bem-sucedida ou não)
                 # para não usar o mesmo override no próximo ciclo, a menos que leia de novo.
                 # Esta lógica foi movida para DEPOIS de _execute_tool.
            # <<< FIM: Lógica de Override no Meta-Ciclo >>>

            # 4. Executar Ação ou Finalizar
            if action_name == "final_answer":
                final_answer = action_input.get("final_answer", "Finalizado sem resposta específica.")
                agent_logger.info(f"{log_prefix} Ação Final: {final_answer}")
                self._history.append(f"Final Answer: {final_answer}")
                save_agent_state(AGENT_STATE_ID, self._memory) # Salva estado ao finalizar
                return final_answer # Retorna a resposta final
            elif action_name in self.tools:
                # >> MODIFICATION START: Pass meta_depth to _execute_tool <<
                observation = self._execute_tool(
                    tool_name=action_name,
                    action_input=action_input,
                    current_objective=current_objective, # Pass current objective for context
                    current_history=self._history,
                    meta_depth=meta_depth # Pass current meta-depth
                )
                # >> MODIFICATION END <<
            else:
                observation = f"Erro: A ferramenta '{action_name}' não existe. Ferramentas disponíveis: {', '.join(self.tools.keys())}"

            agent_logger.info(f"{log_prefix} Observação recebida: {observation[:200]}{'...' if len(observation) > 200 else ''}")
            self._history.append(f"Observation: {observation}")

            # --- Fim do Ciclo ---
            self._trim_history() # Trim history at the end of the cycle
            end_cycle_time = datetime.datetime.now()
            agent_logger.info(f"{log_prefix} --- Fim {log_prefix_base} Cycle {cycle_num}/{self.max_iterations} (Duração Total: {(end_cycle_time - start_cycle_time).total_seconds():.3f}s) ---")


        agent_logger.warning(f"{log_prefix} Máximo de iterações ({self.max_iterations}) atingido.")
        save_agent_state(AGENT_STATE_ID, self._memory) # Salva estado ao atingir limite
        return "Desculpe, não consegui concluir o objetivo dentro do limite de iterações."


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
             return "Thought: Timeout na comunicação com LLM. Action: final_answer Action Input: {\"final_answer\": \"Desculpe, demorei muito para pensar.\"}"
        except requests.exceptions.RequestException as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro na requisição LLM para {chat_url}: {e}")
             if e.response is not None and e.response.status_code == 404:
                  return f"Thought: Endpoint LLM não encontrado. Action: final_answer Action Input: {{\"final_answer\": \"Erro: Endpoint LLM não encontrado ({chat_url}).\"}}"
             return f"Thought: Erro de comunicação com LLM. Action: final_answer Action Input: {{\"final_answer\": \"Desculpe, erro ao conectar ao LLM ({e}).\"}}"
        except Exception as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro inesperado na chamada LLM: {e}", exc_info=True)
             return "Thought: Erro inesperado na chamada LLM. Action: final_answer Action Input: {\"final_answer\": \"Desculpe, ocorreu um erro interno inesperado.\"}"


    # --- _execute_tool (Refatorado para Tratamento Dinâmico de Erro) ---
    # >> MODIFICATION START: Add meta_depth parameter <<
    def _execute_tool(self, tool_name: str, action_input: dict, current_objective: str, current_history: list, meta_depth: int) -> str:
    # >> MODIFICATION END <<
        # (Initial checks for tool existence, function, parameters remain the same)
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
                    # Se o status não for explicitamente 'error', mas também não for 'success',
                    # usamos o formato 'Aviso/Info'.
                    pass # A linha 524 já define a base da observação de erro

        except Exception as e:
            agent_logger.error(f"[ReactAgent EXEC ERROR] Exceção inesperada ao executar {tool_name}: {e}", exc_info=True)

            # --- Reworked Dynamic Error Handling ---
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_frames = traceback.extract_tb(exc_traceback)
            traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)) # Get full traceback string

            skill_file = None
            error_type = exc_type.__name__ if exc_type else "UnknownError"
            line_no = None

            # Find the last frame originating from a skill file
            for frame in reversed(tb_frames):
                if 'skills/' in frame.filename and frame.filename.endswith('.py'):
                    skill_file = frame.filename
                    line_no = frame.lineno
                    break # Found the most recent skill frame

            if skill_file:
                agent_logger.warning(f"[ReactAgent INTERNAL ERROR DETECTED] Error '{error_type}' in skill '{skill_file}' at line {line_no}. Input: {action_input}. Triggering dynamic recovery meta-objective.")

                # --- DYNAMIC META-OBJECTIVE PROMPT ---
                meta_objective_prompt = (
                    f"URGENTE: META-OBJETIVO DE DIAGNÓSTICO E RECUPERAÇÃO ATIVADO!\n\n"
                    f"**Contexto do Erro:**\n"
                    f"- **Objetivo Original:** {current_objective}\n"
                    f"- **Ferramenta Falhou:** {tool_name}\n"
                    f"- **Input da Ferramenta:** {action_input}\n"
                    f"- **Arquivo da Skill:** {skill_file}\n"
                    f"- **Linha do Erro:** {line_no}\n"
                    f"- **Tipo do Erro:** {error_type}\n"
                    f"- **Mensagem de Erro:** {e}\n\n"
                    f"**Traceback Completo:**\n"
                    f"```python\n{traceback_str}\n```\n\n"
                    f"**Seu Novo Objetivo (Diagnóstico e Recuperação):**\n"
                    f"1.  **Diagnostique:** Analise cuidadosamente o traceback e o contexto para entender a causa raiz do erro.\n"
                    f"2.  **Planeje:** Formule um plano de ação passo a passo para corrigir o problema. Use as ferramentas disponíveis (`read_file`, `modify_code`, `grep_search`, `web_search`, etc.) de forma estratégica.\n"
                    f"3.  **Execute:** Realize as ações do seu plano.\n"
                    f"4.  **Conclua:** Se a correção for bem-sucedida, use `final_answer` para reportar o sucesso e o que foi feito. Se não for possível corrigir ou se precisar de ajuda, use `final_answer` para explicar a situação.\n\n"
                    f"**Instruções:** Pense passo a passo (Thought). Use as ferramentas necessárias (Action, Action Input). Seja metódico e claro no seu raciocínio."
                )
                # --- END DYNAMIC META-OBJECTIVE PROMPT ---

                agent_logger.info(f"Iniciando meta-objetivo de recuperação (Profundidade: {meta_depth + 1}) para erro em '{skill_file}'.")
                # Call self.run recursively with the new dynamic objective
                # Pass the INCREASED meta_depth
                return self.run(objective=meta_objective_prompt, is_meta_objective=True, meta_depth=meta_depth + 1)

            else:
                # Generic exception handling if error didn't originate in a skill file
                agent_logger.warning(f"Erro não originado em arquivo de skill rastreável. Retornando observação de erro genérico.")
                observation = f"Erro inesperado (exceção fora de skill rastreável) ao executar {tool_name}: {error_type} - {e}\nTraceback:\n```python\n{traceback_str[:500]}...\n```"
            # --- End of Reworked Dynamic Error Handling ---

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