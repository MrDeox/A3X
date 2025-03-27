import requests
import json
import re
import os
import sys
import datetime
import traceback # Keep for debugging
import logging

# Ajuste para importar do diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.config import LLAMA_SERVER_URL, MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS
from core.tools import get_tool, get_tool_descriptions, TOOLS # <-- MODIFIED IMPORT
from core.db_utils import load_agent_state, save_agent_state

# Configure logging for the agent
agent_logger = logging.getLogger("ReactAgent")
agent_logger.setLevel(logging.INFO)
# Add handlers if necessary, e.g., logging.StreamHandler()

AGENT_STATE_ID = "main_agent_react" # ID fixo para nosso agente único

class ReactAgent:
    def __init__(self, llm_url=LLAMA_SERVER_URL):
        self.llm_url = llm_url
        self._history = []
        self._memory = {
            'last_code': None,
            'last_lang': None,
        }

        # <<< CARREGAR ESTADO DO DB >>>
        agent_logger.info(f"Loading state for agent '{AGENT_STATE_ID}' from database...")
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega o estado persistente
        # Garante que as chaves esperadas existam, mesmo se o load falhar ou for vazio
        self._memory.setdefault('last_code', None)
        self._memory.setdefault('last_lang', None)
        # Log resumido do estado carregado
        loaded_summary = {k: (v[:30]+'...' if isinstance(v, str) and v and len(v) > 30 else v)
                          for k, v in self._memory.items()}
        agent_logger.info(f"Initial agent memory state: {loaded_summary}")
        # <<< FIM CARREGAR ESTADO >>>

        self.tools = get_tool_descriptions() # Carrega descrições das ferramentas
        agent_logger.info("Agent initialized.")

    def _build_react_messages(self, objective: str) -> list[dict]:
        """Constrói a lista de mensagens para o endpoint /v1/chat/completions."""

        tool_desc = self.tools

        # System Message defining the agent's role and instructions
        system_message = f"""Você é A³X, um agente de IA autônomo e prestativo. Use as ferramentas disponíveis para atingir o objetivo do usuário.

**Ferramentas Disponíveis:**
{tool_desc}

**Instruções Importantes:**
1.  **Analise o Histórico:** Preste atenção às 'Observations' anteriores.
2.  **Decomponha o Problema:** Divida objetivos complexos.
3.  **Use a Observação:** Use o resultado da ferramenta anterior no próximo passo.
4.  **Código Sem Saída:** Se 'execute_code' rodou sem erro mas sem output, avalie se o objetivo foi atingido ou se precisa de um passo extra (ex: usar 'generate_code' para adicionar um 'print' ou 'modify_code').
5.  **Seja Explícito:** Explique seu raciocínio no 'Thought'.
6.  **Finalize Corretamente:** Use 'final_answer' APENAS quando o objetivo original for completamente atingido.
    *   **Se a última ação produziu um resultado direto (ex: saída de 'execute_code', código de 'generate_code'/'modify_code', resposta de 'search_web'),** formule a 'answer' para apresentar esse resultado de forma clara ao usuário (ex: "A execução produziu: 30", "Gerei o seguinte código: ...", "A busca encontrou: Paris é a capital...").
    *   **Se a última ação foi bem-sucedida mas sem um resultado direto para mostrar (ex: 'execute_code' rodou uma definição, arquivo foi criado),** a 'answer' deve apenas confirmar a conclusão da tarefa (ex: "A função foi definida com sucesso.", "Arquivo criado.").
    *   **Não** diga apenas "a informação está na observação anterior". Use a informação da observação para criar a resposta final.
7.  **Ação Inválida/Erro:** Se uma ação falhar (indicado na 'Observation'), analise o erro. Se for um erro de parâmetro (ex: faltando 'file_name'), corrija o 'Action Input' e tente novamente. Se for um erro irrecuperável (ex: arquivo não existe para 'append', código com erro de sintaxe grave), use 'final_answer' para informar o usuário sobre o problema.
8.  **Erro de Módulo (ModuleNotFoundError):** Se a 'Observation' de 'execute_code' indicar claramente um 'ModuleNotFoundError' (ex: "Erro Crítico na Execução: Módulo não encontrado ('ModuleNotFoundError: No module named 'pandas')"), **NÃO** tente instalar o pacote (você não tem essa capacidade). Em vez disso:
    *   Use 'final_answer'.
    *   Informe ao usuário que o código falhou porque o módulo necessário ('XYZ') não está disponível no ambiente de execução seguro.
    *   (Opcional) Se parecer viável, sugira brevemente ao usuário gerar um código alternativo usando apenas bibliotecas padrão do Python.

**Formato OBRIGATÓRIO de Resposta:**
Thought: [Seu processo de raciocínio claro e passo-a-passo.]
Action: [O nome EXATO de uma das ferramentas ou 'final_answer'.]
Action Input: [JSON VÁLIDO com os parâmetros EXATOS da ferramenta ou '{{"answer": "Sua resposta final aqui"}}'.]"""

        messages = [{"role": "system", "content": system_message}]

        # Initial user objective
        messages.append({"role": "user", "content": f"Meu objetivo é: {objective}"})

        # Process ReAct history into assistant/user messages
        if self._history:
             assistant_turn_parts = []
             for entry in self._history:
                 if entry.startswith("Thought:") or entry.startswith("Action:") or entry.startswith("Action Input:"):
                     assistant_turn_parts.append(entry)
                 elif entry.startswith("Observation:"):
                     # Finalize the previous assistant turn
                     if assistant_turn_parts:
                          # Combine parts into a single assistant message
                          messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})
                          assistant_turn_parts = []
                     # Add the observation as if it's input from the environment/user
                     # Prepend "Observation:" to make it clear in the chat history
                     messages.append({"role": "user", "content": entry})

             # Add any remaining assistant parts if history doesn't end with Observation
             if assistant_turn_parts:
                  messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})

        # print("[DEBUG] Mensagens construídas:", json.dumps(messages, indent=2, ensure_ascii=False)) # Optional debug
        return messages

    def _parse_llm_response(self, llm_output: str) -> tuple[str | None, str | None, dict | None]:
        """
        Analisa a resposta do LLM para extrair Pensamento e Ação, com logs detalhados.
        Retorna (thought, action_name, action_input_dict).
        """
        agent_logger.debug(f"[Agent Parse DEBUG] Raw LLM Output:\n{llm_output}") # Log raw output

        # Initialize return values
        thought_content = None
        action_name = None
        action_input = None # IMPORTANT: Initialize to None

        # <<< REGEX and Initial Extraction (Keep Previous Version's Logic) >>>
        flags = re.DOTALL | re.IGNORECASE

        thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", llm_output, flags)
        if thought_match:
            thought_content = thought_match.group(1).strip()
            # agent_logger.info(f"[Agent Parse INFO] Thought extraído: '{thought_content[:100]}...'") # Reduced noise
        else:
            agent_logger.warning("[Agent Parse WARN] Bloco 'Thought:' não encontrado ou mal formatado.")
            # Tenta pegar qualquer coisa antes de 'Action:' como pensamento se o match acima falhar
            fallback_thought_match = re.search(r"^(.*?)Action:", llm_output, flags)
            if fallback_thought_match:
                thought_content = fallback_thought_match.group(1).strip()
                agent_logger.info(f"[Agent Parse INFO] Thought extraído (fallback): '{thought_content[:100]}...'")


        action_match = re.search(r"Action:\s*([\w_]+)(?:\s*Action Input:|$)", llm_output, flags)
        if action_match:
            action_name = action_match.group(1).strip()
            # agent_logger.info(f"[Agent Parse INFO] Action Name extraído: '{action_name}'") # Reduced noise
        else:
             agent_logger.warning("[Agent Parse WARN] Bloco 'Action:' não encontrado ou nome da ação não extraído.")
             # Check if maybe it's just a final answer without the Action keyword
             if "final answer" in llm_output.lower() and not thought_content and not action_name:
                 # Se não achou Thought nem Action, mas "final answer" está no texto
                 action_name = "final_answer"
                 agent_logger.warning("[Agent Parse WARN] Nem Thought nem Action encontrados, mas 'final answer' no texto. Assumindo action 'final_answer'.")


        action_input_match = re.search(r"Action Input:\s*(.*)", llm_output, flags)
        input_str = None # Initialize input_str

        if action_input_match:
            input_str = action_input_match.group(1).strip()
            # <<< LOG 1 >>>
            print(f"[Agent Parse DEBUG] Raw Action Input String: '{input_str}'")
            try:
                # Tenta encontrar JSON dentro de ```json ou direto
                # Prioritize finding JSON block, allows garbage around it
                json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", input_str, re.DOTALL | re.MULTILINE)
                if json_match:
                    json_str = next((group for group in json_match.groups() if group is not None), None)

                    if json_str:
                        # <<< LOG 2 >>>
                        print(f"[Agent Parse DEBUG] JSON String encontrada por regex: '{json_str}'")

                        # Limpeza de Aspas
                        cleaned_json_str = json_str.replace('"', '"').replace("'", "'").replace('\\"', '"')
                        cleaned_json_str = cleaned_json_str.replace('"', '"').replace("'", "'").replace('\\"', '"')
                        cleaned_json_str = cleaned_json_str.replace('\\"', '"')
                        json_str_to_parse = cleaned_json_str.strip() # Strip before parsing

                        if cleaned_json_str != json_str:
                             # <<< LOG 3 >>>
                             print(f"[Agent Parse DEBUG] JSON String após limpeza de aspas/escape: '{json_str_to_parse}'")
                        else:
                             # <<< LOG 4 >>>
                             print("[Agent Parse DEBUG] Nenhuma aspa curva/escape encontrada/substituída.")


                        try:
                             action_input = json.loads(json_str_to_parse)
                             # <<< LOG 5 >>>
                             print(f"[Agent Parse INFO] JSON parseado com sucesso na primeira tentativa. Tipo: {type(action_input)}")

                        except json.JSONDecodeError as json_e:
                             # <<< LOG 6 >>>
                             print(f"[Agent Parse ERROR] Falha ao decodificar JSON inicial: {json_e}\nJSON String Tentada: {json_str_to_parse}")
                             # Segunda tentativa: Remover vírgula extra...
                             json_str_fixed = re.sub(r",\s*(\})$", r"\1", json_str_to_parse) # No strip needed here
                             if json_str_fixed != json_str_to_parse:
                                  # <<< LOG 7 >>>
                                  print(f"[Agent Parse INFO] Tentando remover vírgula final. String corrigida: '{json_str_fixed}'")
                                  try:
                                       action_input = json.loads(json_str_fixed)
                                       # <<< LOG 8 >>>
                                       print(f"[Agent Parse INFO] JSON parseado com sucesso após remover vírgula. Tipo: {type(action_input)}")
                                  except json.JSONDecodeError as json_e2:
                                       # <<< LOG 9 >>>
                                       print(f"[Agent Parse ERROR] Falha ao decodificar JSON mesmo após remover vírgula: {json_e2}\nJSON String Tentada: {json_str_fixed}")
                                       action_input = None # Set to None on failure
                                  except Exception as parse_err2:
                                       print(f"[Agent Parse ERROR] Erro inesperado durante SEGUNDO json.loads: {parse_err2}\nJSON String Tentada: {json_str_fixed}")
                                       action_input = None
                             else:
                                  print("[Agent Parse DEBUG] Correção de vírgula final não alterou a string.")
                                  action_input = None # Set to None if comma fix didn't help
                        except Exception as parse_err: # Captura outros erros de json.loads
                             # <<< LOG 10 >>>
                             print(f"[Agent Parse ERROR] Erro inesperado durante json.loads principal: {parse_err}\nJSON String Tentada: {json_str_to_parse}")
                             action_input = None # Set to None on failure
                    else: # json_str was None after regex match
                         print("[Agent Parse ERROR] Regex encontrou match JSON no Input, mas não conseguiu extrair grupo JSON.")
                         action_input = None

                else: # Se json_match falhou (não encontrou '{...}' ou ```json{...}```)
                     # <<< LOG 11 >>>
                     print(f"[Agent Parse WARN] Bloco JSON não encontrado via regex no Action Input: '{input_str}'")
                     # Lógica de fallback para final_answer como string simples
                     if action_name == "final_answer" and input_str:
                          # Heuristic: If it doesn't look like JSON, wrap it
                          if not input_str.strip().startswith('{') and len(input_str.strip()) > 0:
                               action_input = {"answer": input_str.strip()} # Use the raw string as answer
                               # <<< LOG 12 >>>
                               print(f"[Agent Parse WARN] Action Input não era JSON, mas Action é final_answer. Envolvendo em dict: {action_input}")
                          else:
                               # Começa com '{' mas falhou no regex match - provavelmente inválido
                               # <<< LOG 13 >>>
                               print("[Agent Parse WARN] Input começa com '{' ou é vazio, mas falhou no match regex JSON - considerado inválido.")
                               action_input = None # Set to None if invalid format
                     else:
                          # Se não é final_answer e não achou JSON, é None
                          action_input = None

            except Exception as e: # Captura erro no processamento GERAL do input_str
                 # <<< LOG 14 >>>
                 print(f"[Agent Parse ERROR] Erro inesperado ao processar Action Input String: {e}\nInput String: {input_str}")
                 action_input = None # Ensure None on general error

        else: # Se action_input_match falhou (não encontrou 'Action Input:')
             # <<< LOG 15 >>>
             print("[Agent Parse WARN] Bloco 'Action Input:' não encontrado na resposta do LLM.")
             # Handle final_answer case specifically when Action Input block is missing
             if action_name == "final_answer":
                 agent_logger.info("[Agent Parse INFO] Action é 'final_answer' sem 'Action Input:' bloco. Tentando extrair 'answer' do texto.")
                 answer_match = re.search(r"(?:Final Answer:|Answer:)\s*(.*)", llm_output, flags)
                 if answer_match:
                     action_input = {"answer": answer_match.group(1).strip()}
                     print("[Agent Parse INFO] 'answer' extraído de 'Final Answer:' fallback.")
                 elif thought_content and len(thought_content) > 5: # Fallback to thought if reasonable length
                      action_input = {"answer": thought_content}
                      print("[Agent Parse WARN] 'answer' não encontrado, usando 'Thought' como fallback para 'final_answer'.")
                 else:
                      action_input = None # Set to None if no answer found
                      print("[Agent Parse WARN] Não foi possível extrair 'answer' para 'final_answer' sem Action Input.")
             else:
                action_input = None # Garante None se o bloco inteiro faltar E não for final_answer fallback

        # --- Final Logging (Adjusted) ---
        if action_input is None:
             if action_name == 'final_answer':
                  print("[Agent Parse WARN - FINAL] Final action_input é None para Action 'final_answer'.")
             elif action_name:
                  print(f"[Agent Parse WARN - FINAL] Final action_input é None para Action '{action_name}'.")
             else:
                  print("[Agent Parse WARN - FINAL] Final action_input é None e Action Name também é None.")
        elif action_name:
             print(f"[Agent Parse INFO - FINAL] Parse finalizado. Action: '{action_name}', Input Type: {type(action_input)}")
        # else: action_input is not None but action_name is None (should be rare)

        # Ensure final_answer always has a dict, even if empty/default, if action_name is final_answer
        if action_name == "final_answer" and action_input is None:
            action_input = {"answer": "Não foi possível determinar a resposta final (erro de parse)."}
            print("[Agent Parse WARN] Definindo action_input padrão para final_answer após falha no parse.")


        return thought_content, action_name, action_input

    def run(self, objective: str) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""
        agent_logger.info(f"\n[ReactAgent] Iniciando ciclo ReAct para objetivo: '{objective}'")
        self._history = [] # Limpa histórico ReAct para novo objetivo

        for i in range(MAX_REACT_ITERATIONS):
            start_cycle_time = datetime.datetime.now()
            agent_logger.info(f"\n--- Ciclo ReAct {i+1}/{MAX_REACT_ITERATIONS} (Início: {start_cycle_time.strftime('%H:%M:%S.%f')[:-3]}) ---")

            # 1. Construir Prompt
            messages = self._build_react_messages(objective)

            # 2. Chamar LLM
            response_text = ""
            try:
                 start_llm_time = datetime.datetime.now()
                 response_text = self._call_llm(messages)
                 end_llm_time = datetime.datetime.now()
                 llm_duration = end_llm_time - start_llm_time
                 agent_logger.info(f"[ReactAgent] Resposta LLM Recebida (Duração: {llm_duration.total_seconds():.3f}s)")
                 # Log the raw content received for easier debugging
                 agent_logger.info(f"[ReactAgent] Resposta LLM (Raw Content):\n---\n{response_text}\n---")
            except requests.exceptions.RequestException as req_err:
                 agent_logger.error(f"[ReactAgent ERROR] Erro de conexão ao chamar LLM: {req_err}")
                 return f"Erro: Não foi possível conectar ao servidor LLM em {self.llm_url}. Verifique se ele está rodando."
            except Exception as llm_err:
                 agent_logger.error(f"[ReactAgent ERROR] Erro inesperado ao chamar LLM: {llm_err}")
                 return f"Erro inesperado ao comunicar com o LLM: {llm_err}"

            # 3. Parsear Resposta
            thought, action_name, action_input = self._parse_llm_response(response_text)

            if not action_name or action_input is None:
                agent_logger.error("[ReactAgent ERROR] Falha ao parsear Ação ou Input JSON da resposta do LLM. Verifique a 'Resposta LLM (Raw Content)' acima.")
                observation = "Erro: Não consegui entender a resposta do LLM (formato inválido). Verifique a resposta raw acima e o formato esperado."
                # Add raw response to history for context if parsing fails
                self._history.append(f"Thought: {thought if thought else 'N/A - Falha no Parse'}")
                self._history.append(f"Action: {action_name if action_name else 'N/A - Falha no Parse'}")
                self._history.append(f"Action Input: {action_input if action_input is not None else 'N/A - Falha no Parse'}")
                self._history.append(f"LLM Raw Response (unparseable): {response_text}")
                self._history.append(f"Observation: {observation}")
                continue

            # Adiciona Thought/Action/Input ao histórico *antes* da execução
            self._history.append(f"Thought: {thought}")
            self._history.append(f"Action: {action_name}")
            try:
                 action_input_json = json.dumps(action_input, ensure_ascii=False)
            except Exception:
                 action_input_json = str(action_input)
            self._history.append(f"Action Input: {action_input_json}")

            # 4. Executar Ação ou Finalizar
            if action_name == "final_answer":
                final_response = action_input.get("answer", "Não foi possível gerar uma resposta final.")
                agent_logger.info(f"\n[ReactAgent] Resposta Final Decidida pelo LLM: {final_response}")
                # Add final answer action to history before returning
                self._history.append(f"Observation: Resposta final fornecida.")
                return final_response # Encerra o loop

            # --- Refactored Tool Execution ---
            # Call the centralized _execute_tool method
            observation = self._execute_tool(
                tool_name=action_name,
                action_input=action_input,
                current_objective=objective, # Pass current objective
                current_history=self._history # Pass current history
            )
            # --- End Refactored Tool Execution ---


            # Adiciona Observação ao histórico
            agent_logger.info(f"[ReactAgent] Observação: {observation}")
            self._history.append(f"Observation: {observation}")

            end_cycle_time = datetime.datetime.now()
            cycle_duration = end_cycle_time - start_cycle_time
            agent_logger.info(f"--- Fim Ciclo ReAct {i+1} (Duração Total: {cycle_duration.total_seconds():.3f}s) ---")

        # Se sair do loop por limite de iterações
        agent_logger.warning("[ReactAgent WARN] Limite de iterações atingido.")
        # Add final warning to history
        self._history.append("Observation: Limite de iterações atingido sem resposta final.")
        return "Desculpe, não consegui completar a tarefa após várias tentativas."

    # --- Método _call_llm (sem alterações) ---
    def _call_llm(self, messages: list[dict]) -> str:
        agent_logger.info(f"[ReactAgent] Chamando LLM ({len(messages)} mensagens)...")
        headers = {"Content-Type": "application/json"}

        # Ensure the URL points to the chat completions endpoint
        chat_url = self.llm_url
        if not chat_url.endswith("/chat/completions"):
             if chat_url.endswith("/v1") or chat_url.endswith("/v1/"):
                  chat_url = chat_url.rstrip('/') + "/chat/completions"
             else:
                  agent_logger.warning(f"[ReactAgent WARN] URL LLM '{self.llm_url}' pode não ser para /v1/chat/completions. Tentando mesmo assim.")
                  # Optionally force it if you know the base URL is correct:
                  # chat_url = self.llm_url.rstrip('/') + "/v1/chat/completions"

        payload = {
            "messages": messages,
            "temperature": 0.1, # Keep low for format consistency
            "max_tokens": 512,  # Limit for the agent's response (Thought/Action/Input)
            # "stop": None, # Stop sequences usually not needed for chat models
             "stream": False
         }
        try:
             agent_logger.debug(f"[ReactAgent DEBUG] Enviando para URL: {chat_url}") # Debug URL
             # print(f"[ReactAgent DEBUG] Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}") # Debug Payload
             response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
             response.raise_for_status()
             response_data = response.json()

             # Extract content from the assistant's message in the response
             content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

             # print(f"[ReactAgent DEBUG] Resposta LLM Recebida (raw dict): {response_data}") # Optional debug
             if not content:
                  agent_logger.warning("[ReactAgent WARN] LLM retornou conteúdo vazio na resposta de chat.")
             return content
        except requests.exceptions.Timeout:
             agent_logger.error(f"[ReactAgent LLM ERROR] Timeout ao chamar LLM em {chat_url}")
             # Return an error message in the expected ReAct format for the parser
             return "Thought: Ocorreu um timeout ao tentar comunicar com o modelo de linguagem. Action: final_answer Action Input: {\"answer\": \"Desculpe, demorei muito para pensar e a conexão expirou.\"}"
        except requests.exceptions.RequestException as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro na requisição LLM para {chat_url}: {e}")
             # Check if it's a 404 specifically
             if e.response is not None and e.response.status_code == 404:
                  return f"Thought: Erro de conexão. O endpoint {chat_url} não foi encontrado. Verifique a URL do LLM. Action: final_answer Action Input: {{\"answer\": \"Desculpe, não consigo conectar ao meu cérebro principal (endpoint não encontrado).\"}}"
             return f"Thought: Ocorreu um erro ao tentar comunicar com o modelo de linguagem principal. Action: final_answer Action Input: {{\"answer\": \"Desculpe, estou com problemas para conectar ao meu cérebro principal ({e}).\"}}"
        except Exception as e:
             agent_logger.error(f"[ReactAgent LLM ERROR] Erro inesperado na chamada LLM: {e}")
             traceback.print_exc() # Print traceback for unexpected errors
             return "Thought: Ocorreu um erro inesperado durante a chamada LLM. Action: final_answer Action Input: {\"answer\": \"Desculpe, ocorreu um erro interno inesperado.\"}"


    # --- Refactored _execute_tool method ---
    def _execute_tool(self, tool_name: str, action_input: dict, current_objective: str, current_history: list) -> str:
        """
        Executes the chosen tool with the provided input and context using the standard signature.
        Returns the observation string.
        """
        agent_logger.info(f"[ReactAgent] Executando ferramenta: {tool_name} com input: {action_input}") # Debug
        if tool_name not in TOOLS:
            agent_logger.error(f"[ReactAgent ERROR] Ferramenta '{tool_name}' desconhecida.")
            return f"Erro: A ferramenta '{tool_name}' não existe. Ferramentas disponíveis: {', '.join(TOOLS.keys())}"

        tool_info = TOOLS[tool_name]
        tool_function = tool_info.get("function")

        if not tool_function:
            agent_logger.error(f"[ReactAgent ERROR] Ferramenta '{tool_name}' não tem função associada.")
            return f"Erro: A ferramenta '{tool_name}' está configurada incorretamente (sem função)."

        # Validate parameters (basic check)
        required_params = tool_info.get("parameters", {}).get("required", [])
        missing_params = [p for p in required_params if p not in action_input]
        if missing_params:
            agent_logger.warning(f"[ReactAgent EXEC WARN] Parâmetros faltando para {tool_name}: {missing_params}")
            # Provide more context in the error message
            return f"Erro: Parâmetros obrigatórios ausentes para a ferramenta {tool_name}: {', '.join(missing_params)}. Input recebido: {action_input}"

        observation = ""
        try:
            # --- Call skill using the NEW STANDARD SIGNATURE ---
            result = tool_function(
                action_input=action_input,      # Pass the action parameters
                agent_memory=self._memory,      # Pass the memory dictionary
                agent_history=current_history   # Pass the ReAct history
                # Note: current_objective is not passed directly anymore,
                # skills should rely on action_input if they need specific instructions
            )
            # --- End of new call ---

            agent_logger.info(f"[ReactAgent] Resultado da Ferramenta ({tool_name}): {result}") # Debug

            # --- Handle Skill Results ---
            status = result.get("status", "error") # Default to error if status missing
            result_data = result.get("data", {})
            message = result_data.get("message", f"Ferramenta {tool_name} executada.")
            skill_action = result.get("action") # Get the specific action performed by the skill

            if status == "success":
                # --- ATUALIZA E SALVA MEMÓRIA ---
                updated_memory = False # Flag to track if memory changed

                if skill_action in ["code_generated", "code_modified"]:
                    new_code = result_data.get("modified_code") or result_data.get("code")
                    new_lang = result_data.get("language")
                    # Check if the code actually changed before updating and saving
                    if new_code is not None and self._memory.get('last_code') != new_code:
                        self._memory['last_code'] = new_code # Update dictionary
                        self._memory['last_lang'] = new_lang if new_lang else self._memory.get('last_lang') # Update dictionary
                        agent_logger.info("[ReactAgent MEM] Memória do agente atualizada com novo código.")
                        updated_memory = True
                    elif new_lang is not None and self._memory.get('last_lang') != new_lang:
                        # Also update if only the language changed (less likely but possible)
                         self._memory['last_lang'] = new_lang
                         agent_logger.info(f"[ReactAgent MEM] Memória do agente atualizada com nova linguagem: {new_lang}.")
                         updated_memory = True

                # <<< GARANTIR QUE O SAVE ESTÁ AQUI >>>
                if updated_memory:
                    try:
                        save_agent_state(AGENT_STATE_ID, self._memory)
                        # Log message is inside save_agent_state now
                    except Exception as db_save_err:
                         # Use agent_logger for consistency
                         agent_logger.error(f"[ReactAgent ERROR] Falha ao salvar estado no DB após atualização: {db_save_err}")
                # <<< FIM DA GARANTIA >>>
                # --- Fim da atualização e salvamento ---


                # Format observation message based on action/data
                observation_parts = [message] # Start with the base message

                # Specific formatting for different tools/actions
                if skill_action == "code_executed":
                    code_output = result_data.get("output", "").strip()
                    code_stderr = result_data.get("stderr", "").strip()
                    if code_output:
                        observation_parts.append(f"Saída da Execução (stdout):\n```\n{code_output}\n```")
                    elif code_stderr:
                         observation_parts.append(f"Saída de Erro/Warning (stderr):\n```\n{code_stderr}\n```")
                    else:
                         observation_parts.append("(Execução sem saída stdout/stderr visível)")
                elif skill_action in ["code_generated", "code_modified"]:
                    code_to_show = result_data.get("modified_code") or result_data.get("code")
                    # Use memory for language fallback if not in result_data
                    lang_to_show = result_data.get("language", self._memory.get('last_lang') or "text")
                    if code_to_show:
                        code_type = "Modificado" if skill_action == "code_modified" else "Gerado"
                        observation_parts.append(f"Código {code_type}:\n```{lang_to_show}\n{code_to_show}\n```")
                elif skill_action == "web_search_completed" and isinstance(result_data.get("results"), list):
                     snippets = [f"- {r.get('title', 'N/T')}: {r.get('snippet', 'N/A')[:100]}..." for r in result_data["results"]]
                     if snippets:
                          observation_parts.append("Resultados (snippets):\n" + "\n".join(snippets))
                elif skill_action == "file_operation_success": # Example for manage_files
                    # Message already contains details, maybe add filename if useful
                    pass

                # Join parts for final observation
                observation = "\n".join(observation_parts)

            elif status == "error":
                error_message = result_data.get("message", f"Erro desconhecido ao executar {tool_name}.")
                agent_logger.error(f"[ReactAgent EXEC ERROR] Erro na ferramenta {tool_name}: {error_message}")

                # <<< ADICIONAR LÓGICA PARA ModuleNotFoundError >>>
                is_module_not_found = False
                stderr_content = result_data.get("stderr", "") # Get stderr safely

                # Check specifically for execute_code failures
                if skill_action == "execute_code_failed":
                     if "ModuleNotFoundError" in stderr_content:
                          is_module_not_found = True
                          # Try to extract the module name
                          module_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr_content)
                          module_name = module_match.group(1) if module_match else "desconhecido"
                          # Format observation specifically for this error
                          observation = f"Erro Crítico na Execução: Módulo não encontrado ('ModuleNotFoundError: No module named \\'{module_name}\\''). O ambiente de execução seguro não possui este módulo."
                          agent_logger.error(f"[ReactAgent EXEC ERROR] ModuleNotFoundError detectado para módulo: {module_name}") # Specific log
                     else:
                          # Standard error format for execute_code failure (not ModuleNotFound)
                          observation = f"Erro ao executar a ferramenta {tool_name}: {error_message}"
                          if stderr_content:
                               observation += f"\nSaída de Erro (stderr):\n```\n{stderr_content.strip()}\n```"
                else:
                     # Standard error format for failures in other tools
                     observation = f"Erro ao executar a ferramenta {tool_name}: {error_message}"
                # <<< FIM DA LÓGICA >>>

            else: # Handle other statuses like 'warning'
                agent_logger.warning(f"[ReactAgent EXEC WARN] Status não tratado '{status}' da ferramenta {tool_name}: {message}")
                observation = f"Aviso/Info da ferramenta {tool_name}: {message}"

        except TypeError as e:
            # Catch signature mismatches during the call
            agent_logger.error(f"[ReactAgent EXEC ERROR] Erro de tipo ao chamar {tool_name}: {e}")
            traceback.print_exc() # Print detailed traceback for debugging
            observation = f"Erro interno ao chamar a ferramenta {tool_name}. Assinatura incompatível? Detalhe: {e}"
        except Exception as e:
            agent_logger.error(f"[ReactAgent EXEC ERROR] Exceção inesperada ao executar {tool_name}: {e}")
            traceback.print_exc() # Print detailed traceback for debugging
            observation = f"Erro inesperado ao executar a ferramenta {tool_name}: {e}"

        # Limit observation length
        MAX_OBSERVATION_LEN = 1500
        if len(observation) > MAX_OBSERVATION_LEN:
            observation = observation[:MAX_OBSERVATION_LEN] + "... (Observação truncada)"

        return observation 

    def _trim_history(self):
        """Mantém o histórico da sessão dentro dos limites."""
        # Mantém a primeira mensagem (objetivo) e as últimas N*2 interações (ação+obs ou assistente+tool)
        max_keep = 1 + (MAX_HISTORY_TURNS * 2) # +1 for initial user objective
        if len(self._history) > max_keep:
            agent_logger.debug(f"Trimming session history from {len(self._history)} entries.")
            # Keep the first user message and the last max_keep-1 messages
            self._history = [self._history[0]] + self._history[-(max_keep-1):]
            agent_logger.debug(f"History trimmed to {len(self._history)} entries.")

        # <<< REMOVED SAVE FROM HERE >>>
        # save_agent_state(AGENT_STATE_ID, self._memory) # No longer needed here
        # <<< FIM DA REMOÇÃO >>> 