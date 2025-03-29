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
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS, MAX_META_DEPTH
from core.tools import TOOLS, get_tool_descriptions
# from core.llm_client import call_llm # Comentei pois _call_llm está definido na classe
from skills.memory import skill_recall_memory
from core.db_utils import save_agent_state, load_agent_state # AGENT_STATE_ID é carregado depois

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente (pode vir de config ou DB utils)
AGENT_STATE_ID = 1 # Ou carregue de outra forma

# <<< CARREGAR JSON SCHEMA >>>
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
        self._last_error_type = None
        self._last_skill_file = None
        self._last_executed_code = None
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

    # --- _parse_llm_response (Simplificado para usar json.loads) ---
    def _parse_llm_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Parses the LLM's raw response string, expecting it to be a JSON object.
        Raises: json.JSONDecodeError if the response is not valid JSON.
        Returns: thought, action, action_input
        """
        agent_logger.debug(f"[Agent Parse DEBUG] Raw LLM Response (expecting JSON):\n{response}")
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                 agent_logger.error(f"[Agent Parse ERROR] LLM Response is valid JSON but not an object (dict): {type(data)}")
                 raise json.JSONDecodeError("Parsed JSON is not an object", response, 0) # Re-raise as decode error

            thought = data.get("Thought")
            action = data.get("Action")
            action_input = data.get("Action Input") # This should already be a dict if schema is respected

            # Validação básica
            if not action:
                 agent_logger.error(f"[Agent Parse ERROR] Required key 'Action' missing in parsed JSON: {data}")
                 # Considerar levantar um erro aqui também ou retornar None para acionar fallback?
                 # Por enquanto, vamos logar e retornar None para action, o que deve ser tratado no loop run
                 return thought, None, action_input
            if action == "final_answer" and not action_input:
                 agent_logger.warning(f"[Agent Parse WARN] 'final_answer' action received without 'Action Input'. Creating default. JSON: {data}")
                 action_input = {"answer": "Erro: Ação final solicitada sem fornecer a resposta."}
            elif action != "final_answer" and action_input is None:
                 agent_logger.info(f"[Agent Parse INFO] Action '{action}' received without 'Action Input'. Assuming empty dict. JSON: {data}")
                 action_input = {} # Assume dict vazio se não houver input para outras ações

            # Garante que action_input é um dict se não for None
            if action_input is not None and not isinstance(action_input, dict):
                 agent_logger.error(f"[Agent Parse ERROR] 'Action Input' in JSON is not a dictionary: {type(action_input)}. Content: {action_input}. Treating as empty.")
                 action_input = {} # Fallback para dict vazio se o tipo estiver errado

            agent_logger.info(f"[Agent Parse INFO] JSON parsed successfully. Action: '{action}'")
            return thought, action, action_input

        except json.JSONDecodeError as e:
            agent_logger.error(f"[Agent Parse ERROR] Failed to decode LLM response as JSON: {e}")
            agent_logger.debug(f"[Agent Parse DEBUG] Failed JSON content:\n{response}")
            raise e # Re-raise the exception to be caught by the caller (run loop)
        except Exception as e:
             # Captura outros erros inesperados durante o parse/validação
             agent_logger.exception(f"[Agent Parse ERROR] Unexpected error during JSON parsing/validation:")
             # Decide se re-levanta ou retorna None. Re-levantar como JSONDecodeError pode ser consistente.
             raise json.JSONDecodeError(f"Unexpected parsing error: {e}", response, 0) from e


    # --- run (Ajustado para tratar JSONDecodeError) ---
    def run(self, objective: str, is_meta_objective: bool = False, meta_depth: int = 0) -> str:
        """Executa o ciclo ReAct para atingir o objetivo."""

        # --- Limite de Profundidade Meta ---
        if meta_depth > MAX_META_DEPTH:
            agent_logger.warning(f"[ReactAgent META-{meta_depth}] Max meta depth ({MAX_META_DEPTH}) reached. Aborting meta-cycle for objective: '{objective[:100]}...'")
            return f"Erro: Profundidade máxima de auto-correção ({MAX_META_DEPTH}) atingida."

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
            try:
                llm_response_raw = self._call_llm(prompt)
                agent_logger.info(f"{log_prefix} Resposta LLM Recebida (Duração: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
            except Exception as e:
                agent_logger.exception(f"{log_prefix} Exceção durante a chamada LLM: {e}")
                llm_response_raw = f"Erro: Falha na chamada LLM: {e}" # Define a mensagem de erro padrão

            # Limita log da resposta bruta
            log_llm_response = llm_response_raw[:1000] + ('...' if len(llm_response_raw) > 1000 else '')
            agent_logger.debug(f"{log_prefix} Resposta LLM (Raw Content):\n---\n{log_llm_response}\n---")

            if not llm_response_raw or llm_response_raw.startswith("Erro:"): # Verifica se _call_llm retornou um erro interno
                agent_logger.error(f"{log_prefix} Erro Fatal: _call_llm retornou erro ou resposta vazia: '{llm_response_raw}'")
                # Adiciona a observação do erro para o LLM tentar corrigir
                self._history.append(f"Observation: Erro crítico na comunicação com o LLM: {llm_response_raw}")
                # Podemos tentar continuar ou retornar erro. Continuar dá chance de recuperação.
                if i == self.max_iterations - 1: # Se for a última iteração, retorna erro
                    return f"Desculpe, ocorreu um erro na comunicação com o LLM: {llm_response_raw}"
                else:
                    continue # Tenta o próximo ciclo

            # Adiciona a resposta *bruta* ao histórico ANTES de tentar parsear
            # Isso garante que o LLM veja o que ele enviou, mesmo que falhe o parse
            self._history.append(llm_response_raw)

            # 3. Parsear Resposta LLM (esperando JSON)
            try:
                thought, action_name, action_input = self._parse_llm_response(llm_response_raw)

                # Verifica se o parse retornou action_name (essencial)
                if not action_name:
                     # Isso pode acontecer se o JSON for válido mas faltar a chave 'Action'
                     agent_logger.error(f"{log_prefix} Parsing ok, mas 'Action' não encontrada no JSON.")
                     raise ValueError("JSON válido, mas chave 'Action' obrigatória está ausente.") # Tratar como erro de formato

            except (json.JSONDecodeError, ValueError) as parse_error:
                # --- PONTO DE TRIGGER PARA META-REFLEXÃO (Futuro) ---
                agent_logger.error(f"{log_prefix} Falha ao parsear resposta JSON do LLM: {parse_error}")
                # Adiciona observação sobre o erro de formato para o LLM
                observation_msg = f"Observation: Erro crítico - sua resposta anterior não estava no formato JSON esperado ou faltava a chave 'Action'. Verifique o formato. Detalhe: {parse_error}"
                self._history.append(observation_msg)
                agent_logger.debug(f"{log_prefix} Added parse error observation: {observation_msg}")

                # TODO: No futuro, chamar meta-reflexão aqui em vez de só continuar
                if i == self.max_iterations - 1: # Se for a última iteração, retorna erro
                    agent_logger.warning(f"{log_prefix} Última iteração falhou no parsing. Finalizando com erro.")
                    save_agent_state(AGENT_STATE_ID, self._memory) # Salva estado ANTES de retornar erro
                    return f"Desculpe, falha ao processar a resposta do LLM após {self.max_iterations} tentativas."
                else:
                    agent_logger.info(f"{log_prefix} Tentando continuar após erro de parsing.")
                    continue # Pula para próximo ciclo para o LLM tentar corrigir

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
                # Usa a chave "answer" que DEVE estar em action_input segundo o schema
                final_answer_text = action_input.get("answer", "Finalizado sem resposta específica.")
                agent_logger.info(f"{log_prefix} Ação Final: {final_answer_text}")
                self._history.append(f"Final Answer: {final_answer_text}") # Adiciona APENAS a resposta final textual ao histórico
                save_agent_state(AGENT_STATE_ID, self._memory) # Salva estado ao finalizar
                return final_answer_text # Retorna a resposta final
            elif action_name in self.tools:
                 tool_result = self._execute_tool(
                     tool_name=action_name,
                     action_input=action_input,
                     current_history=self._history, # Mantido, embora _execute_tool não use mais
                     meta_depth=meta_depth
                 )

                 # <<< ADDED: Store last executed code >>>
                 if action_name == "execute_code":
                     self._last_executed_code = action_input.get("code")

                 # Converte o dicionário de resultado em uma string de Observação
                 # Mantém a estrutura status/action/data para clareza
                 observation = json.dumps(tool_result, ensure_ascii=False, default=lambda o: '<not serializable>')

                 # <<< ADDED: Auto-Correction Logic >>>
                 is_execution_error = (
                     action_name == "execute_code" and
                     tool_result.get("status") == "error" and
                     tool_result.get("action") == "execution_failed"
                 )
                 if is_execution_error and self._last_executed_code and meta_depth < MAX_META_DEPTH:
                     agent_logger.warning(f"{log_prefix} Erro detectado na execução do código. Iniciando ciclo de auto-correção (Profundidade: {meta_depth + 1}).")
                     error_message = tool_result.get("data", {}).get("message", "Erro desconhecido")
                     stderr_output = tool_result.get("data", {}).get("stderr", "")
                     full_error_details = f"Error Message: {error_message}\nStderr:\n{stderr_output}".strip()

                     meta_objective = f"""A tentativa anterior de executar código falhou.
Erro reportado:
---
{full_error_details}
---

Código que falhou:
```python
{self._last_executed_code}
```

Sua tarefa é:
1. Analisar o erro e o código.
2. Usar a ferramenta 'modify_code' para propor uma correção para o código. Passe o código original completo em 'code_to_modify' e a instrução de correção em 'modification'.
3. Após obter o código modificado da ferramenta 'modify_code', use a ferramenta 'execute_code' para testar a versão corrigida.
4. Se a execução do código corrigido for bem-sucedida (sem erros), use 'final_answer' para reportar 'Correção aplicada e testada com sucesso.'.
5. Se a execução do código corrigido ainda falhar, use 'final_answer' para reportar 'Falha ao corrigir o erro após tentativa.'"""

                     # Chamada Recursiva
                     meta_result = self.run(
                         objective=meta_objective,
                         is_meta_objective=True,
                         meta_depth=meta_depth + 1
                     )

                     # Gerar Observation para o ciclo ORIGINAL
                     if meta_result == "Correção aplicada e testada com sucesso.":
                         observation_msg = "Ocorreu um erro na execução anterior, mas um ciclo de auto-correção foi iniciado e concluído com sucesso. O código corrigido foi executado."
                         agent_logger.info(f"{log_prefix} Ciclo de auto-correção bem-sucedido.")
                     elif meta_result.startswith("Erro: Profundidade máxima"):
                         observation_msg = f"Ocorreu um erro na execução anterior. Uma tentativa de auto-correção foi feita, mas atingiu a profundidade máxima ({MAX_META_DEPTH}). {meta_result}"
                         agent_logger.warning(f"{log_prefix} Ciclo de auto-correção falhou (profundidade máxima).")
                     else:
                         observation_msg = f"Ocorreu um erro na execução anterior. Uma tentativa de auto-correção foi feita, mas falhou em corrigir o erro. Resultado da tentativa: {meta_result}"
                         agent_logger.warning(f"{log_prefix} Ciclo de auto-correção falhou.")

                     # Sobrescreve a observation original do erro com a da meta-correção
                     observation = observation_msg # Agora SEM o prefixo duplicado
                     # Limpa o código executado para não tentar corrigir de novo no mesmo ciclo
                     self._last_executed_code = None
                 # <<< END: Auto-Correction Logic >>>

            else:
                 observation_dict = {"status": "error", "action": "tool_not_found", "data": {"message": f"A ferramenta '{action_name}' não existe. Ferramentas disponíveis: {', '.join(self.tools.keys())}"}}
                 observation = json.dumps(observation_dict)

            agent_logger.info(f"{log_prefix} Observação recebida (JSON): {observation[:200]}{'...' if len(observation) > 200 else ''}")
            # Adiciona a observação formatada como string JSON ao histórico
            self._history.append(f"Observation: {observation}")

            # --- Fim do Ciclo ---
            self._trim_history() # Trim history at the end of the cycle
            end_cycle_time = datetime.datetime.now()
            agent_logger.info(f"{log_prefix} --- Fim {log_prefix_base} Cycle {cycle_num}/{self.max_iterations} (Duração Total: {(end_cycle_time - start_cycle_time).total_seconds():.3f}s) ---")


        agent_logger.warning(f"{log_prefix} Máximo de iterações ({self.max_iterations}) atingido.")
        save_agent_state(AGENT_STATE_ID, self._memory) # Salva estado ao atingir limite
        return "Desculpe, não consegui concluir o objetivo dentro do limite de iterações."


    # --- _call_llm (Modificado para usar JSON Schema) ---
    def _call_llm(self, messages: list[dict]) -> str:
        """Chama o LLM local com a lista de mensagens e força a saída JSON usando o schema."""
        # Substituir o placeholder de descrição de ferramentas
        tool_desc = get_tool_descriptions()
        processed_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = content.replace("[TOOL_DESCRIPTIONS]", tool_desc)
            processed_messages.append({"role": msg["role"], "content": content})

        payload = {
            "messages": processed_messages,
            "temperature": 0.5, # Ajuste conforme necessário
            "max_tokens": 1500, # Ajuste conforme necessário
            # "stop": ["Observation:"] # Parar antes da observação pode ajudar?
            "stream": False
        }

        # Adicionar JSON Schema ao payload se ele foi carregado com sucesso
        if LLM_RESPONSE_SCHEMA:
            payload["response_format"] = {
                "type": "json_object",
                "schema": LLM_RESPONSE_SCHEMA
            }
            agent_logger.info("[ReactAgent LLM Call] Usando JSON Schema para forçar output.")
        else:
            agent_logger.warning("[ReactAgent LLM Call] JSON Schema não carregado. A saída do LLM não será forçada.")

        headers = LLAMA_DEFAULT_HEADERS
        # agent_logger.debug(f"[ReactAgent LLM Call DEBUG] Enviando payload: {json.dumps(payload, indent=2)}") # Debug verboso

        try:
            response = requests.post(self.llm_url, headers=headers, json=payload, timeout=180) # Aumentado timeout
            response.raise_for_status()
            response_data = response.json()

            # Extrair conteúdo da resposta - o formato pode variar ligeiramente
            if 'choices' in response_data and response_data['choices']:
                 message = response_data['choices'][0].get('message', {})
                 content = message.get('content', '').strip()
                 if content:
                     return content
                 else:
                     agent_logger.error(f"[ReactAgent LLM Call ERROR] Resposta LLM OK, mas 'content' está vazio. Resposta: {response_data}")
                     return "Erro: LLM retornou resposta sem conteúdo." # Retorna erro específico
            else:
                 agent_logger.error(f"[ReactAgent LLM Call ERROR] Resposta LLM OK, mas formato inesperado (sem 'choices' ou 'choices' vazio). Resposta: {response_data}")
                 return "Erro: LLM retornou resposta em formato inesperado." # Retorna erro específico

        except requests.exceptions.RequestException as e:
            agent_logger.error(f"[ReactAgent LLM Call ERROR] Falha ao conectar/comunicar com LLM em {self.llm_url}: {e}")
            return f"Erro: Falha ao conectar com o servidor LLM ({e})." # Retorna erro específico
        except Exception as e:
            agent_logger.exception(f"[ReactAgent LLM Call ERROR] Erro inesperado ao chamar LLM:")
            return f"Erro: Erro inesperado durante a chamada do LLM ({e})." # Retorna erro específico


    # --- _execute_tool (Refatorado para Tratamento Dinâmico de Erro) ---
    # >> MODIFICATION START: Add meta_depth parameter <<
    def _execute_tool(self, tool_name: str, action_input: Dict[str, Any], current_history: list, meta_depth: int) -> Dict[str, Any]:
    # >> MODIFICATION END <<
        """Executa a ferramenta/skill selecionada."""
        log_prefix = f"[ReactAgent Tool Execution]"
        agent_logger.info(f"{log_prefix} Executando ferramenta: '{tool_name}', Input: {action_input}")

        if tool_name not in self.tools:
            agent_logger.error(f"{log_prefix} Ferramenta desconhecida: '{tool_name}'")
            return {"status": "error", "action": "tool_not_found", "data": {"message": f"Ferramenta '{tool_name}' não encontrada."}}

        tool = self.tools[tool_name]
        tool_function = tool["function"]

        try:
            # Simplificado: Todas as ferramentas agora só recebem action_input
            agent_logger.debug(f"{log_prefix} Calling skill '{tool_name}' with action_input only.")
            result = tool_function(action_input=action_input)

            if not isinstance(result, dict):
                agent_logger.error(f"{log_prefix} Skill '{tool_name}' retornou um tipo inesperado: {type(result)}. Esperado: dict.")
                return {"status": "error", "action": "invalid_skill_return", "data": {"message": f"Skill '{tool_name}' retornou um tipo inválido."}}

            agent_logger.info(f"{log_prefix} Skill '{tool_name}' executada. Status: {result.get('status', 'N/A')}")
            agent_logger.debug(f"{log_prefix} Resultado da Skill '{tool_name}': {result}")

            # Atualiza a memória se a skill teve sucesso e é relevante (ex: código gerado)
            # --- LÓGICA DE ATUALIZAÇÃO DE MEMÓRIA FOI MOVIDA PARA O LOOP run() PRINCIPAL ---

            return result # Retorna o dicionário completo da skill

        except Exception as e:
            agent_logger.exception(f"{log_prefix} Erro ao executar a skill '{tool_name}':")
            # Retornar uma estrutura de erro padronizada
            return {
                "status": "error",
                "action": f"{tool_name}_failed",
                "data": {"message": f"Erro interno ao executar a skill '{tool_name}': {str(e)}"}
            }

    # --- _trim_history (Mantido como antes) ---
    def _trim_history(self):
        # (Código da função _trim_history exatamente como na sua versão anterior)
        max_keep = 1 + (MAX_HISTORY_TURNS * 2)
        if len(self._history) > max_keep:
            agent_logger.debug(f"Trimming history from {len(self._history)} entries.")
            self._history = [self._history[0]] + self._history[-(max_keep-1):]
            agent_logger.debug(f"History trimmed to {len(self._history)} entries.")