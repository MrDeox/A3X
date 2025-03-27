import requests
import json
import re
import os
import sys
import datetime
import traceback # Keep for debugging

# Ajuste para importar do diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.config import LLAMA_SERVER_URL, MAX_REACT_ITERATIONS
from core.tools import get_tool, get_tool_descriptions, TOOLS # <-- MODIFIED IMPORT

class ReactAgent:
    def __init__(self, llm_url=LLAMA_SERVER_URL):
        self.llm_url = llm_url
        self._history = []
        self._memory = {
            'last_code': None,
            'last_lang': None,
        }

    def _build_react_messages(self, objective: str) -> list[dict]:
        """Constrói a lista de mensagens para o endpoint /v1/chat/completions."""

        tool_desc = get_tool_descriptions()

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

    def _parse_llm_response(self, response_text: str) -> tuple[str | None, str | None, dict | None]:
        """Extrai Thought, Action e Action Input da resposta do LLM."""
        # Adiciona 'Thought:' ao início se não estiver presente (LLM pode começar direto)
        if not response_text.strip().startswith("Thought:"):
             response_text = "Thought: " + response_text

        thought = re.search(r"Thought:(.*?)Action:", response_text, re.DOTALL)
        action = re.search(r"Action:(.*?)Action Input:", response_text, re.DOTALL)
        action_input_str = re.search(r"Action Input:(.*)", response_text, re.DOTALL)

        thought_content = thought.group(1).strip() if thought else None
        action_name = action.group(1).strip() if action else None

        action_input = None
        if action_input_str:
            input_str = action_input_str.group(1).strip()
            try:
                # Tenta encontrar e extrair um JSON válido, lidando com ```json ``` opcional
                json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", input_str, re.DOTALL | re.MULTILINE)
                if json_match:
                    # Pega o primeiro grupo que não for None (prioriza o JSON dentro de ```json)
                    json_str = next(group for group in json_match.groups() if group is not None)
                    # Attempt to fix common JSON errors before parsing
                    json_str = json_str.replace(r'\\"', r'"').replace(r"\'", r"'") # Handle escaped quotes if needed
                    try:
                        action_input = json.loads(json_str)
                    except json.JSONDecodeError as json_e:
                        print(f"[Agent Parse ERROR] Falha ao decodificar JSON: {json_e}\nJSON String Tentada: {json_str}")
                        # Try a more lenient parse? Or return error? For now, return None.
                        action_input = None
                else:
                     # If no JSON block found, maybe it's a simple string for final_answer?
                     if action_name == "final_answer" and input_str:
                          # Heuristic: If it doesn't look like JSON, wrap it
                          if not input_str.strip().startswith('{'):
                               action_input = {"answer": input_str}
                               print(f"[Agent Parse WARN] Action Input não era JSON, mas Action é final_answer. Envolvendo: {action_input}")
                          else:
                               print(f"[Agent Parse WARN] Não foi possível encontrar/parsear JSON no Action Input: {input_str}")
                     else:
                          print(f"[Agent Parse WARN] Não foi possível encontrar/parsear JSON no Action Input: {input_str}")
            except Exception as e:
                 print(f"[Agent Parse ERROR] Erro inesperado ao parsear Action Input: {e}\nInput String: {input_str}")


        # Validações básicas
        if not thought_content: print("[Agent Parse WARN] Não encontrou 'Thought:' na resposta.")
        if not action_name: print("[Agent Parse WARN] Não encontrou 'Action:' na resposta.")
        if not action_input: print("[Agent Parse WARN] Não encontrou 'Action Input:' ou falhou ao parsear JSON.")

        return thought_content, action_name, action_input


    def run(self, objective: str) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""
        print(f"\n[ReactAgent] Iniciando ciclo ReAct para objetivo: '{objective}'")
        self._history = [] # Limpa histórico ReAct para novo objetivo

        for i in range(MAX_REACT_ITERATIONS):
            start_cycle_time = datetime.datetime.now()
            print(f"\n--- Ciclo ReAct {i+1}/{MAX_REACT_ITERATIONS} (Início: {start_cycle_time.strftime('%H:%M:%S.%f')[:-3]}) ---")

            # 1. Construir Prompt
            messages = self._build_react_messages(objective)

            # 2. Chamar LLM
            response_text = ""
            try:
                 start_llm_time = datetime.datetime.now()
                 response_text = self._call_llm(messages)
                 end_llm_time = datetime.datetime.now()
                 llm_duration = end_llm_time - start_llm_time
                 print(f"[ReactAgent] Resposta LLM Recebida (Duração: {llm_duration.total_seconds():.3f}s)")
                 # Log the raw content received for easier debugging
                 print(f"[ReactAgent] Resposta LLM (Raw Content):\n---\n{response_text}\n---")
            except requests.exceptions.RequestException as req_err:
                 print(f"[ReactAgent ERROR] Erro de conexão ao chamar LLM: {req_err}")
                 return f"Erro: Não foi possível conectar ao servidor LLM em {self.llm_url}. Verifique se ele está rodando."
            except Exception as llm_err:
                 print(f"[ReactAgent ERROR] Erro inesperado ao chamar LLM: {llm_err}")
                 return f"Erro inesperado ao comunicar com o LLM: {llm_err}"

            # 3. Parsear Resposta
            thought, action_name, action_input = self._parse_llm_response(response_text)

            if not action_name or action_input is None:
                print("[ReactAgent ERROR] Falha ao parsear Ação ou Input JSON da resposta do LLM. Verifique a 'Resposta LLM (Raw Content)' acima.")
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
                print(f"\n[ReactAgent] Resposta Final Decidida pelo LLM: {final_response}")
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
            print(f"[ReactAgent] Observação: {observation}")
            self._history.append(f"Observation: {observation}")

            end_cycle_time = datetime.datetime.now()
            cycle_duration = end_cycle_time - start_cycle_time
            print(f"--- Fim Ciclo ReAct {i+1} (Duração Total: {cycle_duration.total_seconds():.3f}s) ---")

        # Se sair do loop por limite de iterações
        print("[ReactAgent WARN] Limite de iterações atingido.")
        # Add final warning to history
        self._history.append("Observation: Limite de iterações atingido sem resposta final.")
        return "Desculpe, não consegui completar a tarefa após várias tentativas."

    # --- Método _call_llm (sem alterações) ---
    def _call_llm(self, messages: list[dict]) -> str:
        print(f"[ReactAgent] Chamando LLM ({len(messages)} mensagens)...")
        headers = {"Content-Type": "application/json"}

        # Ensure the URL points to the chat completions endpoint
        chat_url = self.llm_url
        if not chat_url.endswith("/chat/completions"):
             if chat_url.endswith("/v1") or chat_url.endswith("/v1/"):
                  chat_url = chat_url.rstrip('/') + "/chat/completions"
             else:
                  print(f"[ReactAgent WARN] URL LLM '{self.llm_url}' pode não ser para /v1/chat/completions. Tentando mesmo assim.")
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
             print(f"[ReactAgent DEBUG] Enviando para URL: {chat_url}") # Debug URL
             # print(f"[ReactAgent DEBUG] Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}") # Debug Payload
             response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
             response.raise_for_status()
             response_data = response.json()

             # Extract content from the assistant's message in the response
             content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

             # print(f"[ReactAgent DEBUG] Resposta LLM Recebida (raw dict): {response_data}") # Optional debug
             if not content:
                  print("[ReactAgent WARN] LLM retornou conteúdo vazio na resposta de chat.")
             return content
        except requests.exceptions.Timeout:
             print(f"[ReactAgent LLM ERROR] Timeout ao chamar LLM em {chat_url}")
             # Return an error message in the expected ReAct format for the parser
             return "Thought: Ocorreu um timeout ao tentar comunicar com o modelo de linguagem. Action: final_answer Action Input: {\"answer\": \"Desculpe, demorei muito para pensar e a conexão expirou.\"}"
        except requests.exceptions.RequestException as e:
             print(f"[ReactAgent LLM ERROR] Erro na requisição LLM para {chat_url}: {e}")
             # Check if it's a 404 specifically
             if e.response is not None and e.response.status_code == 404:
                  return f"Thought: Erro de conexão. O endpoint {chat_url} não foi encontrado. Verifique a URL do LLM. Action: final_answer Action Input: {{\"answer\": \"Desculpe, não consigo conectar ao meu cérebro principal (endpoint não encontrado).\"}}"
             return f"Thought: Ocorreu um erro ao tentar comunicar com o modelo de linguagem principal. Action: final_answer Action Input: {{\"answer\": \"Desculpe, estou com problemas para conectar ao meu cérebro principal ({e}).\"}}"
        except Exception as e:
             print(f"[ReactAgent LLM ERROR] Erro inesperado na chamada LLM: {e}")
             traceback.print_exc() # Print traceback for unexpected errors
             return "Thought: Ocorreu um erro inesperado durante a chamada LLM. Action: final_answer Action Input: {\"answer\": \"Desculpe, ocorreu um erro interno inesperado.\"}"


    # --- Refactored _execute_tool method ---
    def _execute_tool(self, tool_name: str, action_input: dict, current_objective: str, current_history: list) -> str:
        """
        Executes the chosen tool with the provided input and context using the standard signature.
        Returns the observation string.
        """
        print(f"[ReactAgent] Executando ferramenta: {tool_name} com input: {action_input}") # Debug
        if tool_name not in TOOLS:
            print(f"[ReactAgent ERROR] Ferramenta '{tool_name}' desconhecida.")
            return f"Erro: A ferramenta '{tool_name}' não existe. Ferramentas disponíveis: {', '.join(TOOLS.keys())}"

        tool_info = TOOLS[tool_name]
        tool_function = tool_info.get("function")

        if not tool_function:
            print(f"[ReactAgent ERROR] Ferramenta '{tool_name}' não tem função associada.")
            return f"Erro: A ferramenta '{tool_name}' está configurada incorretamente (sem função)."

        # Validate parameters (basic check)
        required_params = tool_info.get("parameters", {}).get("required", [])
        missing_params = [p for p in required_params if p not in action_input]
        if missing_params:
            print(f"[ReactAgent EXEC WARN] Parâmetros faltando para {tool_name}: {missing_params}")
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

            print(f"[ReactAgent] Resultado da Ferramenta ({tool_name}): {result}") # Debug

            # --- Handle Skill Results ---
            status = result.get("status", "error") # Default to error if status missing
            result_data = result.get("data", {})
            message = result_data.get("message", f"Ferramenta {tool_name} executada.")

            if status == "success":
                # --- Update memory using the dictionary ---
                action = result.get("action")
                if action in ["code_generated", "code_modified"]:
                    new_code = result_data.get("modified_code") or result_data.get("code")
                    new_lang = result_data.get("language")
                    if new_code:
                        self._memory['last_code'] = new_code # Update dictionary
                        self._memory['last_lang'] = new_lang if new_lang else self._memory.get('last_lang') # Update dictionary
                        print("[ReactAgent MEM] Memória do agente atualizada com novo código.")
                # --- End of memory update ---

                # Format observation message based on action/data
                observation_parts = [message] # Start with the base message

                # Specific formatting for different tools/actions
                if action == "code_executed":
                    code_output = result_data.get("output", "").strip()
                    code_stderr = result_data.get("stderr", "").strip()
                    if code_output:
                        observation_parts.append(f"Saída da Execução (stdout):\n```\n{code_output}\n```")
                    elif code_stderr:
                         observation_parts.append(f"Saída de Erro/Warning (stderr):\n```\n{code_stderr}\n```")
                    else:
                         observation_parts.append("(Execução sem saída stdout/stderr visível)")
                elif action in ["code_generated", "code_modified"]:
                    code_to_show = result_data.get("modified_code") or result_data.get("code")
                    # Use memory for language fallback if not in result_data
                    lang_to_show = result_data.get("language", self._memory.get('last_lang') or "text")
                    if code_to_show:
                        code_type = "Modificado" if action == "code_modified" else "Gerado"
                        observation_parts.append(f"Código {code_type}:\n```{lang_to_show}\n{code_to_show}\n```")
                elif action == "web_search_completed" and isinstance(result_data.get("results"), list):
                     snippets = [f"- {r.get('title', 'N/T')}: {r.get('snippet', 'N/A')[:100]}..." for r in result_data["results"]]
                     if snippets:
                          observation_parts.append("Resultados (snippets):\n" + "\n".join(snippets))
                elif action == "file_operation_success": # Example for manage_files
                    # Message already contains details, maybe add filename if useful
                    pass

                # Join parts for final observation
                observation = "\n".join(observation_parts)

            elif status == "error":
                error_message = result_data.get("message", f"Erro desconhecido ao executar {tool_name}.")
                print(f"[ReactAgent EXEC ERROR] Erro na ferramenta {tool_name}: {error_message}")
                observation = f"Erro ao executar a ferramenta {tool_name}: {error_message}"
                if result.get("action") == "execute_code_failed" and result_data.get("stderr"):
                     observation += f"\nSaída de Erro (stderr):\n```\n{result_data['stderr']}\n```"

            else: # Handle other statuses like 'warning'
                print(f"[ReactAgent EXEC WARN] Status não tratado '{status}' da ferramenta {tool_name}: {message}")
                observation = f"Aviso/Info da ferramenta {tool_name}: {message}"

        except TypeError as e:
            # Catch signature mismatches during the call
            print(f"[ReactAgent EXEC ERROR] Erro de tipo ao chamar {tool_name}: {e}")
            traceback.print_exc() # Print detailed traceback for debugging
            observation = f"Erro interno ao chamar a ferramenta {tool_name}. Assinatura incompatível? Detalhe: {e}"
        except Exception as e:
            print(f"[ReactAgent EXEC ERROR] Exceção inesperada ao executar {tool_name}: {e}")
            traceback.print_exc() # Print detailed traceback for debugging
            observation = f"Erro inesperado ao executar a ferramenta {tool_name}: {e}"

        # Limit observation length
        MAX_OBSERVATION_LEN = 1500
        if len(observation) > MAX_OBSERVATION_LEN:
            observation = observation[:MAX_OBSERVATION_LEN] + "... (Observação truncada)"

        return observation 