# core/cerebrumx.py
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple
import json # Removed comment
import os
import datetime
import re
from datetime import timezone
from pathlib import Path
from collections import namedtuple

# Import base agent and other necessary core components
from a3x.core.agent import ReactAgent, is_introspective_query

# from core.tools import get_tool_descriptions  # <<< REMOVED import
from a3x.core.skills import get_skill_descriptions # <<< Simplified import

# from core.tool_executor import execute_tool  # <<< REMOVED import
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext  # <<< KEPT needed import

# <<< REMOVED Import for old execution logic >>>
# from a3x.core.execution_logic import execute_plan_with_reflection

# Import DB functions for memory access
from a3x.core.db_utils import retrieve_relevant_context, add_episodic_record, retrieve_recent_episodes # <<< Keep episodic record for now

# Potentially import memory, reflection components later

# Import skills needed for reflect_and_learn
from ..skills.core.reflect_on_success import reflect_on_success
from ..skills.core.learn_from_failure_log import learn_from_failure_log
from ..skills.core.reflect_on_failure import reflect_on_failure # <<< ADDED import
# Generalization/Consolidation skills will be called within _reflect_and_learn

# Initialize logger for this module
cerebrumx_logger = logging.getLogger(__name__) # <<< Rename logger? agent_logger already exists in ReactAgent

# <<< REMOVED LOG_FILE_PATH Constants, might be better in config or learning module >>>
# LEARNING_LOG_DIR = "memory/learning_logs"
# HEURISTIC_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "learned_heuristics.jsonl")

# <<< ADDED Import >>>
from .agent import _is_simple_list_files_task, ReactAgent # <<< ADD ReactAgent import

# Helper to create context for direct execution calls
_ToolExecutionContext = namedtuple("_ToolExecutionContext", ["logger", "workspace_root", "llm_url", "tools_dict"])

class CerebrumXAgent(ReactAgent): # Inheriting from ReactAgent
    """
    Agente Autônomo Adaptável Experimental com ciclo cognitivo unificado.
    Incorpora percepção, planejamento, execução ReAct, reflexão e aprendizado em um único fluxo.
    """

    def __init__(self, system_prompt: str, llm_url: Optional[str] = None, tools_dict: Optional[Dict[str, Dict[str, Any]]] = None):
        """Inicializa o Agente CerebrumX."""
        super().__init__(system_prompt, llm_url, tools_dict=tools_dict)
        # Use self.agent_logger inherited from ReactAgent
        self.agent_logger.info("[CerebrumX INIT] Agente CerebrumX inicializado (Ciclo Unificado).")
        # No initial_perception needed here if run takes objective

    # <<< REMOVED run_cerebrumx_cycle >>>
    # async def run_cerebrumx_cycle(...) -> AsyncGenerator[Dict[str, Any], None]: ...

    # --- Novo Ciclo Cognitivo Unificado ---
    async def run(self, objective: str) -> Dict[str, Any]: # Now returns final result dict
        """
        Executa o ciclo cognitivo completo unificado do A³X.
        Perceber -> Planejar -> Executar -> Refletir & Aprender.
        Retorna um dicionário com o resultado final ou o status da execução.
        """
        import time
        from a3x.core.auto_evaluation import auto_evaluate_task

        self.agent_logger.info(f"--- Iniciando Ciclo Cognitivo Unificado --- Objetivo: {objective[:100]}..." )
        start_time = time.time()

        # 1. Percepção (Simplificado)
        perception = self._perceive(objective)
        self.agent_logger.info(f"Percepção processada: {perception}")

        # 2. Recuperação de Contexto
        context = await self._retrieve_context(perception)
        self.agent_logger.info(f"Contexto recuperado: {str(context)[:100]}...")

        # 3. Planejamento
        plan = await self._plan(perception, context)
        self.agent_logger.info(f"Plano gerado: {plan}")
        if not plan:
             self.agent_logger.error("Falha ao gerar plano. Abortando ciclo.")
             return {"status": "error", "message": "Falha crítica no planejamento."}

        # 4. Execução do Plano
        # Returns tuple: (final_status: str, final_message: str, execution_results: list)
        # final_status can be 'completed', 'failed', 'error'
        final_status, final_message, execution_results = await self._execute_plan(plan, context, objective)
        self.agent_logger.info(f"Execução do plano finalizada. Status: {final_status}")

        # 5. Reflexão e Aprendizado Pós-Execução
        await self._reflect_and_learn(perception, plan, execution_results, final_status)

        # 6. Autoavaliação Cognitiva
        end_time = time.time()
        # Coleta heurísticas aplicadas dos resultados (pode ser aprimorado)
        heuristics_used = []
        for res in execution_results:
            if "heuristic" in res.get("data", {}):
                heuristics_used.append(res["data"]["heuristic"])
        auto_evaluate_task(
            objective=objective,
            plan=plan,
            execution_results=execution_results,
            heuristics_used=heuristics_used,
            start_time=start_time,
            end_time=end_time
        )

        # 7. Validação de heurísticas aplicada ao ciclo real
        try:
            from a3x.core.heuristics_validator import validate_heuristics
            # Simula impacto das heurísticas aplicadas nesta execução
            task_info = {
                "objective": objective,
                "plan": plan,
                "execution_results": execution_results
            }
            validate_heuristics([task_info])
            self.agent_logger.info("[HeuristicsValidator] Validação de heurísticas executada ao final do ciclo.")
        except Exception as e:
            self.agent_logger.warning(f"[HeuristicsValidator] Falha ao validar heurísticas: {e}")

        # 8. Retornar Resultado Final
        self.agent_logger.info("--- Ciclo Cognitivo Unificado Concluído --- ")
        return {"status": final_status, "message": final_message, "results": execution_results} # Return consolidated result

    # --- Métodos Internos do Ciclo ---

    def _perceive(self, objective: str) -> Dict[str, Any]:
        """Processa a percepção inicial (objetivo)."""
        # TODO: Expandir lógica de percepção se necessário
        self.agent_logger.info("Processando percepção...")
        return {"processed": objective}

    async def _retrieve_context(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Recupera contexto relevante da memória semântica (FAISS) e episódica (Recente)."""
        self.agent_logger.info("Recuperando contexto da memória (Semântica e Episódica Recente)...")
        query = perception.get("processed", "")
        if not query:
            self.agent_logger.warning("Query de percepção vazia, não é possível buscar contexto semântico.")
            # Ainda assim, buscar episódios recentes
            # return {"semantic_summary": "N/A", "semantic_results": [], "episodic": [], "query": query}

        # --- Busca Semântica (FAISS) --- #
        semantic_matches = []
        semantic_summary = "Nenhuma memória semântica relevante encontrada."
        if query: # Só buscar se houver query
            try:
                # Importar localmente ou mover para o topo do arquivo se preferir
                from a3x.core.embeddings import get_embedding
                from a3x.core.semantic_memory_backend import search_index
                from a3x.core.config import PROJECT_ROOT, SEMANTIC_SEARCH_TOP_K # <<< Importar configs
                import os

                # <<< Definir caminho base do índice FAISS (Idealmente em config.py) >>>
                index_path_base = os.path.join(PROJECT_ROOT, "a3x", "memory", "indexes", "semantic_memory")

                self.agent_logger.info(f"Gerando embedding para a busca semântica: '{query[:50]}...'" )
                query_embedding_list = get_embedding(query)

                if query_embedding_list is not None:
                    self.agent_logger.info(f"Buscando no índice FAISS: {index_path_base} (top_k={SEMANTIC_SEARCH_TOP_K})" )
                    search_results = search_index(
                        index_path_base=index_path_base,
                        query_embedding=query_embedding_list,
                        top_k=SEMANTIC_SEARCH_TOP_K
                    )

                    if search_results:
                        self.agent_logger.info(f"Encontrados {len(search_results)} resultados na memória semântica.")
                        semantic_matches = search_results # Mantém a lista completa de resultados
                        # Criar sumário para o planner
                        semantic_summary = "\nContexto Semântico Relevante (FAISS):\n"
                        for i, res in enumerate(search_results):
                            # Acessa o conteúdo original dentro dos metadados
                            content = res.get("metadata", {}).get("content", "<Conteúdo indisponível>")
                            distance = res.get("distance", -1.0)
                            semantic_summary += f"- [Dist: {distance:.3f}] {content}\n"
                        semantic_summary = semantic_summary.strip()

                        # <<< Opcional: Registrar a consulta e os resultados na memória episódica >>>
                        try:
                            record_metadata = {
                                "query": query,
                                "top_k": SEMANTIC_SEARCH_TOP_K,
                                "num_results": len(search_results),
                                "results_preview": [{ "dist": r.get("distance",-1), "content_preview": r.get("metadata",{}).get("content","")[:50]} for r in search_results]
                            }
                            add_episodic_record(
                                context="context_retrieval",
                                action="semantic_search",
                                outcome="results_found",
                                metadata=record_metadata
                            )
                            self.agent_logger.info("Consulta de memória semântica registrada na memória episódica.")
                        except Exception as db_err:
                            self.agent_logger.error(f"Erro ao registrar consulta semântica na memória episódica: {db_err}")

                    else:
                        self.agent_logger.info("Nenhum resultado encontrado na memória semântica.")
                else:
                    self.agent_logger.error("Falha ao gerar embedding para a busca semântica.")
                    semantic_summary = "Erro ao gerar embedding para busca semântica."

            except ImportError as imp_err:
                self.agent_logger.error(f"Erro de importação necessário para busca semântica: {imp_err}")
                semantic_summary = "Erro: Dependência de busca semântica não encontrada."
            except Exception as e:
                self.agent_logger.exception("Erro inesperado durante a busca de contexto semântico:")
                semantic_summary = f"Erro inesperado na busca semântica: {e}"
        else:
             semantic_summary = "N/A (Query vazia)"

        # --- Busca Episódica (Recente) --- #
        episodic_matches = []
        episodic_summary = "Nenhuma memória episódica recente encontrada."
        try:
            # Importar a nova função
            from a3x.core.db_utils import retrieve_recent_episodes
            from a3x.core.config import EPISODIC_RETRIEVAL_LIMIT # <<< Usar config

            self.agent_logger.info(f"Buscando os {EPISODIC_RETRIEVAL_LIMIT} episódios mais recentes...")
            recent_episodes = retrieve_recent_episodes(limit=EPISODIC_RETRIEVAL_LIMIT)

            if recent_episodes:
                 self.agent_logger.info(f"Encontrados {len(recent_episodes)} episódios recentes.")
                 episodic_matches = [dict(row) for row in recent_episodes] # Converter Rows para Dicts
                 # Criar sumário para o planner
                 episodic_summary = "\nContexto Episódico Recente:\n"
                 for i, episode in enumerate(episodic_matches):
                      ctx = episode.get("context", "?")
                      act = episode.get("action", "?")
                      out = episode.get("outcome", "?")
                      # Mostrar trechos para evitar sobrecarga
                      episodic_summary += f"- Ep.{i+1}: Ctx: '{ctx[:30]}...' Act: '{act[:40]}...' Out: '{out[:30]}...'\n"
                 episodic_summary = episodic_summary.strip()
            else:
                 self.agent_logger.info("Nenhum episódio recente encontrado na memória.")

        except ImportError as imp_err:
            self.agent_logger.error(f"Erro de importação necessário para busca episódica: {imp_err}")
            episodic_summary = "Erro: Dependência de busca episódica não encontrada."
        except Exception as e:
            self.agent_logger.exception("Erro inesperado durante a busca de contexto episódico:")
            episodic_summary = f"Erro inesperado na busca episódica: {e}"

        # --- Montar Contexto Final --- #
        # Combinar sumários para o planner
        combined_summary = f"{semantic_summary}\n\n{episodic_summary}".strip()

        final_context = {
            "combined_summary": combined_summary, # Sumário combinado para o planner
            "semantic_results": semantic_matches, # Resultados completos da busca semântica
            "episodic_results": episodic_matches, # Resultados completos da busca episódica
            "query": query
        }
        return final_context

    # Renamed from _plan_hierarchically, unified planning logic
    async def _plan(self, perception: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Gera um plano de execução para o objetivo, usando o contexto."""
        self.agent_logger.info("--- Gerando Plano de Execução ---")
        objective = perception.get("processed", "")

        # Incorporate logic from ReactAgent._generate_plan if needed (e.g., simple tasks)
        if _is_simple_list_files_task(objective): # Function from agent.py
             self.agent_logger.info("[Planner] Tarefa simples detectada (list_files). Gerando plano simples.")
             plan_to_execute = [
                 f"Use the list_directory tool for the objective: '{objective}'",
                 "Use the final_answer tool to provide the list of files.",
             ]
        else:
            # Call the external planner for complex tasks
            self.agent_logger.info("--- Gerando Plano de Execução ---")
            tool_desc = get_skill_descriptions() # Get available tools

            # <<< START: Consultar Heurísticas Aprendidas >>>
            heuristics_context = None # Initialize as None
            found_heuristics = []
            try:
                self.agent_logger.info(f"Consultando heurísticas aprendidas relevantes para o objetivo: '{objective[:100]}...'")
                # Criar contexto para a chamada da skill de consulta
                consult_exec_context = _ToolExecutionContext(
                    logger=self.agent_logger,
                    workspace_root=self.workspace_root,
                    llm_url=self.llm_url,
                    tools_dict=self.tools
                )
                # Chamar a skill consult_learned_heuristics
                heuristics_result = await execute_tool(
                    tool_name="consult_learned_heuristics",
                    action_input={"objective": objective, "top_k": 3}, # Corrigido: usar 'objective' conforme schema
                    tools_dict=self.tools,
                    context=consult_exec_context
                )

                if heuristics_result.get("status") == "success":
                    # Suporte tanto para lista de heurísticas quanto para listas separadas (success/failure)
                    if "heuristics" in heuristics_result.get("data", {}):
                        found_heuristics = heuristics_result.get("data", {}).get("heuristics", [])
                    else:
                        # Suporte para formato antigo: {"success": [...], "failure": [...]}
                        found_heuristics = []
                        for h in heuristics_result.get("data", {}).get("success", []):
                            found_heuristics.append({"heuristic": h, "type": "success"})
                        for h in heuristics_result.get("data", {}).get("failure", []):
                            found_heuristics.append({"heuristic": h, "type": "failure"})
                    if found_heuristics:
                        formatted_heuristics = "\n\nHeurísticas Relevantes Aprendidas:\n"
                        for i, h in enumerate(found_heuristics):
                            heuristic_text = h.get('heuristic', 'N/A')
                            formatted_heuristics += f"- {heuristic_text}\n"
                        heuristics_context = formatted_heuristics.strip()
                        self.agent_logger.info(f"[Planner] Usando {len(found_heuristics)} heurísticas relevantes para o planejamento.")
                    else:
                        self.agent_logger.info("[Planner] Nenhuma heurística relevante encontrada para este objetivo.")
                else:
                    error_msg = heuristics_result.get('data', {}).get('message', 'Erro desconhecido')
                    self.agent_logger.warning(f"[Planner] Falha ao consultar heurísticas: {error_msg}")

            except Exception as consult_err:
                self.agent_logger.exception("[Planner] Erro inesperado ao consultar heurísticas:")
            # <<< END: Consultar Heurísticas Aprendidas >>>

            try:
                # Assuming planner.generate_plan exists and works
                from a3x.core.planner import generate_plan # Local import ok here?
                # <<< MODIFIED: Passar heuristics_context para generate_plan >>>
                plan_to_execute = await generate_plan(
                    objective=objective,
                    tool_descriptions=tool_desc,
                    agent_logger=self.agent_logger,
                    llm_url=self.llm_url,
                    heuristics_context=heuristics_context # Passar o novo contexto (pode ser None)
                )
                if not plan_to_execute: # Checks for None or empty list
                     self.agent_logger.warning(f"[Planner] Plano não retornado ou vazio. Usando objetivo como passo único: {objective}")
                     plan_to_execute = [f"Attempt to directly address: {objective}"]
                else:
                     self.agent_logger.info(f"[Planner] Plano gerado com sucesso ({len(plan_to_execute)} passos).")
            except Exception as plan_err:
                self.agent_logger.exception("Erro durante a geração do plano:")
                plan_to_execute = [] # Indicate planning failure with an empty list

            # <<< INTEGRAÇÃO DO MIDDLEWARE DE HEURÍSTICAS >>>
            try:
                from a3x.core.heuristic_planner_middleware import inject_heuristics_into_plan
                if found_heuristics:
                    plan_to_execute = inject_heuristics_into_plan(plan_to_execute, found_heuristics)
                    self.agent_logger.info(f"[Planner] Middleware de heurísticas injetou {len(found_heuristics)} heurísticas no plano.")
            except Exception as mw_err:
                self.agent_logger.warning(f"[Planner] Falha ao injetar heurísticas via middleware: {mw_err}")

        # --- Validação pós-plano: filtra passos irrelevantes ---
        from a3x.core.skills import get_skill_registry
        skills_registry = get_skill_registry()
        valid_skills = set(skills_registry.keys())

        def extract_skill_name(step: str) -> Optional[str]:
            # Extrai o nome da skill do padrão "Use the <skill> tool" ou "Use the <skill> skill"
            match = re.search(r"Use the (\w+) (?:tool|skill)", step)
            if match:
                return match.group(1)
            # Alternativamente, tenta extrair "<skill>:" no início do passo
            match2 = re.match(r"(\w+):", step)
            if match2:
                return match2.group(1)
            return None

        filtered_plan = []
        invalid_found = False
        for step in plan_to_execute:
            skill_name = extract_skill_name(step)
            if skill_name and skill_name in valid_skills:
                filtered_plan.append(step)
            elif "final_answer" in step:
                filtered_plan.append(step)
            elif "write_file" in step:
                filtered_plan.append(step)
            else:
                invalid_found = True
                self.agent_logger.info(f"[Planner] Passo removido por não corresponder a skill válida: {step}")

        # Se houver qualquer passo inválido, substitui por plano padrão
        if invalid_found or not filtered_plan or not any("write_file" in step for step in filtered_plan):
            self.agent_logger.warning("[Planner] Passos inválidos detectados ou nenhum passo válido de escrita. Gerando plano padrão para escrita em arquivo.")
            filtered_plan = [
                "Use the write_file tool para salvar o texto solicitado no arquivo.",
                "Use the final_answer tool para confirmar a operação."
            ]

        plan_str = json.dumps(filtered_plan, indent=2, ensure_ascii=False)
        self.agent_logger.info(f"Plano Gerado (filtrado):\n{plan_str}")
        return filtered_plan


    # Merged from execution_logic.execute_plan_with_reflection
    async def _execute_plan(self, plan: List[str], context: Dict[str, Any], original_objective: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Executa um plano passo-a-passo.
        Para planos simples detectados por _is_simple_list_files_task, executa diretamente.
        Para outros planos, usa o ciclo ReAct interno (_perform_react_iteration).
        Realiza reflexão sobre falhas e chama a skill de aprendizado de falhas.
        Retorna (status_final, mensagem_final, lista_resultados_passos).
        """
        log_prefix = "[ExecutePlan]"
        self.agent_logger.info(f"{log_prefix} Iniciando execução do plano com {len(plan)} passos.")
        execution_results = []
        step_result = None  # Initialize step_result here for broader scope
        final_status = "unknown" # Initialize final_status
        final_message = "" # Initialize final_message

        # Check if it's a simple list_files -> final_answer plan
        if _is_simple_list_files_task(original_objective): # Corrected: Pass original_objective, not plan
            self.agent_logger.info("Executando plano simples diretamente (list_directory -> final_answer).")
            # <<< START: Define constant within method scope >>>
            HEURISTIC_LOG_FILE = Path("memory/learning_logs/learned_heuristics.jsonl")
            # <<< END: Define constant within method scope >>>
            try:
                # Step 1: list_directory
                # We need to parse the action/input from the plan step string
                # This is brittle, ideally the planner returns structured steps
                match = re.match(r"Use the (\w+) tool for the objective: '(.*)'", plan[0])
                if not match or match.group(1) != "list_directory":
                    raise ValueError("Plano simples inválido (Passo 1 não é list_directory)")
                list_action = match.group(1)
                # Extract directory from objective - again, brittle
                dir_match = re.search(r"(?:diretório|pasta) (\S+)", original_objective, re.IGNORECASE)
                if not dir_match:
                    dir_match = re.search(r"em (\S+)", original_objective, re.IGNORECASE)

                list_input = {"directory": dir_match.group(1) if dir_match else "."} # Default to current dir if not found
                self.agent_logger.info(f"Passo 1 (Direto): Executando {list_action} com input {list_input}")

                exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)
                list_result = await execute_tool(
                    tool_name=list_action,
                    action_input=list_input,
                    tools_dict=self.tools,
                    context=exec_context
                )
                execution_results.append(list_result)

                if list_result.get("status") != "success":
                    final_status = "failed"
                    final_message = f"Falha no passo 1 (list_directory): {list_result.get('data', {}).get('message', 'Erro desconhecido')}"
                    self.agent_logger.error(final_message)
                    # <<< START: Adicionar Reflexão sobre Falha para Planos Simples >>>
                    try: # Outer try for reflection + learning
                        self.agent_logger.warning("Execução direta falhou. Tentando refletir e aprender com a falha...")
                        failure_context = {
                            "objective": original_objective,
                    "plan": plan,
                            "failed_step_index": 0, # Falha no primeiro passo
                            "failed_step": plan[0],
                            "last_thought": "N/A (Execução Direta)", # Não há thought explícito
                            "last_action": list_action,
                            "last_observation": str(list_result.get("data")) # Usar dados do erro como observação
                        }
                        # Usar o mesmo contexto de execução criado anteriormente
                        reflection_result = await execute_tool(
                            tool_name="reflect_on_failure",
                            action_input=failure_context,
                            tools_dict=self.tools,
                            context=exec_context
                        )
                        self.agent_logger.info(f"Resultado de reflect_on_failure (exceção crítica): {reflection_result}")

                        if reflection_result.get("status") == "success":
                            failure_analysis = reflection_result.get("data", {}).get("explanation") # Adjusted key based on reflect_on_failure output
                            if failure_analysis:
                                self.agent_logger.info(f"Análise da falha (exceção crítica) obtida: {failure_analysis[:100]}...")
                                # Call learn_from_failure_log immediately
                                learn_input = {
                                    "objective": original_objective,
                                    "error_message": str(list_result.get("data", {}).get("message", "Erro desconhecido")), # Use the exception string
                                    "failure_analysis": failure_analysis
                                }
                                try: # Inner try for learning call
                                    self.agent_logger.info(f"Chamando learn_from_failure_log para exceção crítica: {learn_input}")
                                    learn_result = await execute_tool(
                                        tool_name="learn_from_failure_log",
                                        action_input=learn_input,
                                        tools_dict=self.tools,
                                        context=exec_context # Re-use context
                                    )
                                    self.agent_logger.info(f"Resultado de learn_from_failure_log (exceção crítica): {learn_result}")
                                    if learn_result.get("status") == "success":
                                        heuristic = learn_result.get("data", {}).get("heuristic", "N/A")
                                        # <<< START: Log Failure Heuristic to File >>>
                                        if heuristic != "N/A":
                                            try: # Innermost try for logging
                                                # Correct call: datetime.datetime.now()
                                                timestamp = datetime.datetime.now(timezone.utc).isoformat(timespec='microseconds')
                                                log_entry = {
                                                    "timestamp": timestamp,
                                                    "objective": original_objective, # Use original objective
                                                    "plan": plan, # Use the plan variable from this scope
                                                    "results": execution_results, # Contains the error result
                                                    "heuristic": heuristic,
                                                    "type": "failure"
                                                }
                                                log_file = HEURISTIC_LOG_FILE
                                                log_file.parent.mkdir(parents=True, exist_ok=True)

                                                with open(log_file, 'a', encoding='utf-8') as f:
                                                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                                                # This log message is now accurate
                                                self.agent_logger.info(f"Heurística de falha (exceção crítica) registrada em {log_file}: {heuristic[:100]}...")
                                            except Exception as log_write_err: # Catch for logging
                                                self.agent_logger.exception(f"Falha ao escrever heurística de falha no log {log_file}: {log_write_err}")
                                        else:
                                             self.agent_logger.warning("Heurística de falha retornada como 'N/A', não registrada no arquivo.")
                                        # <<< END: Log Failure Heuristic to File >>>
                                    else:
                                        learn_error = learn_result.get("data", {}).get("message", "Erro desconhecido no aprendizado")
                                        self.agent_logger.error(f"Falha ao chamar learn_from_failure_log (exceção crítica): {learn_error}")
                                except Exception as learn_err_crit: # Catch for learning call
                                    self.agent_logger.exception(f"Exceção ao chamar learn_from_failure_log (exceção crítica): {learn_err_crit}")
                            else:
                                self.agent_logger.warning("Reflexão sobre falha (exceção crítica) não retornou análise.")
                        else:
                            reflect_error = reflection_result.get("data", {}).get("message", "Erro desconhecido na reflexão")
                            self.agent_logger.error(f"Falha ao chamar reflect_on_failure (exceção crítica): {reflect_error}")
                    except Exception as reflect_err_crit: # Catch for reflection call
                         self.agent_logger.exception(f"Erro durante a reflexão/aprendizado da falha crítica na execução direta: {reflect_err_crit}")
                    # <<< END: Adicionar Reflexão sobre Falha para Planos Simples >>>
                    # This return should be OUTSIDE the try/except for reflection/learning
                    # because we want to return the 'failed' status regardless of whether reflection succeeded
                    return final_status, final_message, execution_results

                # Step 2: final_answer
                if not plan[1].startswith("Use the final_answer tool"):
                     raise ValueError("Plano simples inválido (Passo 2 não é final_answer)")

                list_data = list_result.get("data", {})
                answer_content = f"Arquivos encontrados em '{list_data.get('directory_requested')}':\n" + "\n".join(list_data.get('items', []))
                answer_input = {"answer": answer_content}
                self.agent_logger.info(f"Passo 2 (Direto): Executando final_answer")

                answer_result = await execute_tool(
                    tool_name="final_answer",
                    action_input=answer_input,
                    tools_dict=self.tools,
                    context=exec_context
                )
                execution_results.append(answer_result)

                if answer_result.get("status") != "success":
                    final_status = "failed"
                    final_message = f"Falha no passo 2 (final_answer): {answer_result.get('data', {}).get('message', 'Erro desconhecido')}"
                else:
                    final_status = "completed"
                    final_message = answer_content # Use the generated answer as the final message
                self.agent_logger.info(f"Execução direta do plano simples finalizada. Status: {final_status}")

            except Exception as direct_exec_err:
                self.agent_logger.exception("Erro durante a execução direta do plano simples:")
                final_status = "error"
                final_message = f"Erro crítico na execução direta: {direct_exec_err}"
                # <<< START: Adicionar Reflexão sobre Falha Crítica (Opcional, mas bom) >>>
                try:
                    self.agent_logger.warning("Execução direta falhou com exceção. Tentando refletir...")
                    failure_context = {
                        "objective": original_objective,
                        "plan": plan,
                        "failed_step_index": 0, # Assumir falha no início
                        "failed_step": plan[0] if plan else "N/A",
                        "last_thought": "N/A (Execução Direta)",
                        "last_action": "N/A (Exceção antes da ação)",
                        "last_observation": f"Exceção Crítica: {direct_exec_err}" # Usar exceção como observação
                    }
                    exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)
                    reflection_result = await execute_tool(
                        tool_name="reflect_on_failure",
                        action_input=failure_context,
                        tools_dict=self.tools,
                        context=exec_context
                    )
                    self.agent_logger.info(f"Resultado de reflect_on_failure (exceção crítica): {reflection_result}")

                    if reflection_result.get("status") == "success":
                        failure_analysis = reflection_result.get("data", {}).get("explanation") # Adjusted key based on reflect_on_failure output
                        if failure_analysis:
                            self.agent_logger.info(f"Análise da falha (exceção crítica) obtida: {failure_analysis[:100]}...")
                            # Call learn_from_failure_log immediately
                            learn_input = {
                                "objective": original_objective,
                                "error_message": str(direct_exec_err), # Use the exception string
                                "failure_analysis": failure_analysis
                            }
                            try:
                                self.agent_logger.info(f"Chamando learn_from_failure_log para exceção crítica: {learn_input}")
                                learn_result = await execute_tool(
                                    tool_name="learn_from_failure_log",
                                    action_input=learn_input,
                                    tools_dict=self.tools,
                                    context=exec_context # Re-use context
                                )
                                self.agent_logger.info(f"Resultado de learn_from_failure_log (exceção crítica): {learn_result}")
                                if learn_result.get("status") == "success":
                                    heuristic = learn_result.get("data", {}).get("heuristic", "N/A")
                                    # <<< START: Log Failure Heuristic to File >>>
                                    if heuristic != "N/A":
                                        try:
                                            # Correct call: datetime.datetime.now()
                                            timestamp = datetime.datetime.now(timezone.utc).isoformat(timespec='microseconds')
                                            log_entry = {
                                                "timestamp": timestamp,
                                                "objective": original_objective, # Use original objective
                                                "plan": plan, # Use the plan variable from this scope
                                                "results": execution_results, # Contains the error result
                                                "heuristic": heuristic,
                                                "type": "failure"
                                            }
                                            log_file = HEURISTIC_LOG_FILE
                                            log_file.parent.mkdir(parents=True, exist_ok=True)
                                            
                                            with open(log_file, 'a', encoding='utf-8') as f:
                                                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                                            # This log message is now accurate
                                            self.agent_logger.info(f"Heurística de falha (exceção crítica) registrada em {log_file}: {heuristic[:100]}...")
                                        except Exception as log_write_err:
                                            self.agent_logger.exception(f"Falha ao escrever heurística de falha no log {log_file}: {log_write_err}")
                                    else:
                                         self.agent_logger.warning("Heurística de falha retornada como 'N/A', não registrada no arquivo.")
                                    # <<< END: Log Failure Heuristic to File >>>
                                else:
                                    learn_error = learn_result.get("data", {}).get("message", "Erro desconhecido no aprendizado")
                                    self.agent_logger.error(f"Falha ao chamar learn_from_failure_log (exceção crítica): {learn_error}")
                            except Exception as learn_err_crit:
                                self.agent_logger.exception(f"Exceção ao chamar skill learn_from_failure_log:")
                        else:
                            self.agent_logger.warning("Reflexão sobre falha (exceção crítica) não retornou análise.")
                    else:
                        reflect_error = reflection_result.get("data", {}).get("message", "Erro desconhecido na reflexão")
                        self.agent_logger.error(f"Falha ao chamar skill reflect_on_failure: {reflect_error}")
                        failure_analysis = f"Erro na reflexão: {reflect_error}" # Update final message

                except Exception as reflect_err: # Corresponds to outer try (line ~458) - ALIGNED
                     self.agent_logger.exception(f"Exceção ao chamar skill reflect_on_failure:")
                     failure_analysis = f"Erro crítico na chamada de reflexão: {reflect_err}" # Update final message

                # Update final message and stop execution (outside the outer try...except) - ALIGNED
                final_message = f"Falha no passo {i+1}: {step_result.get('message', 'Erro Desconhecido')}\n**Análise:** {failure_analysis}"
                final_status = "failed" # Ensure status reflects failure
                execution_results.append(step_result) # Add failed step result
                self.agent_logger.warning(f"{log_prefix} Encerrando execução do plano devido à falha no passo.")
                return final_status, final_message, execution_results # Stop plan execution

            # Append successful step result IF it exists (handles complex plans)
            if step_result:
                 execution_results.append(step_result)

        # <<< START: ELSE BLOCK FOR COMPLEX PLANS (ReAct Loop) >>>
        else:
            # Handle complex plans using the ReAct loop
            self.agent_logger.info(f"{log_prefix} Executing complex plan via ReAct loop.")
            i = 0 # Initialize step counter
            try:
                i = 0
                max_steps = len(plan)
                while i < max_steps:
                    step = plan[i]
                    step_log_prefix = f"{log_prefix} [Step {i+1}/{len(plan)}]"
                    self.agent_logger.info(f"{step_log_prefix} Executing: '{step}'")

                    step_result = None # Reset for each step
                    error_occurred = False
                    failure_analysis = "N/A"
                    
                    # Call the ReAct iteration generator
                    async for react_output in self._perform_react_iteration(step, step_log_prefix):
                        event_type = react_output.get("type")
                        event_content = react_output.get("content")
                        
                        # Store the last meaningful result (observation or final answer for the step)
                        if event_type == "observation":
                            step_result = event_content # Store the full observation dict
                        elif event_type == "step_final_answer":
                            # If the step finishes with an answer, wrap it like a tool result
                            step_result = {"status": "success", "action": "step_completed_with_answer", "data": {"final_answer": event_content}}
                            final_status = "intermediate_success" # Mark step as success
                            break # Move to next step in the plan
                        elif event_type == "error":
                            error_occurred = True
                            error_message = event_content or "Unknown error during ReAct iteration."
                            self.agent_logger.error(f"{step_log_prefix} Error occurred: {error_message}")
                            # Store error info as step result
                            step_result = {"status": "error", "action": "react_error", "data": {"message": error_message}}
                            final_status = "failed"
                            break # Stop processing this step
                    
                    # --- Reflection and Learning on Step Failure (within ReAct loop) ---
                    if error_occurred:
                        self.agent_logger.warning(f"{step_log_prefix} Step failed. Diagnosing and attempting adaptation...")

                        # --- LLM-Based Error Diagnosis ---
                        error_message = step_result.get('data', {}).get('message', 'Unknown error message')
                        # TODO: Capture traceback if possible from execute_tool result
                        traceback_str = step_result.get('data', {}).get('traceback')
                        
                        # Avoid repeating the same error diagnosis excessively
                        if hasattr(self, "_recent_failures"):
                            self._recent_failures.append(error_message)
                            if self._recent_failures.count(error_message) > 2:
                                self.agent_logger.error(f"{log_prefix} Erro repetido detectado ('{error_message[:50]}...'). Interrompendo para evitar repetição cega.")
                                final_message = f"Erro repetido: {error_message}\nO agente interrompeu para evitar repetição cega."
                                execution_results.append(step_result)
                                return final_status, final_message, execution_results
                        else:
                            self._recent_failures = [error_message]

                        # Prepare context for diagnosis skill
                        last_thought_diag = next((h.split("Thought: ", 1)[1] for h in reversed(self._history) if h.startswith("Thought:")), "N/A")
                        last_action_diag = next((h.split("Action: ", 1)[1] for h in reversed(self._history) if h.startswith("Action:")), "N/A")
                        last_observation_diag = next((h.split("Observation: ", 1)[1] for h in reversed(self._history) if h.startswith("Observation:")), "N/A")

                        diagnosis_context = {
                            "objective": original_objective,
                            "plan_step": step,
                            "failed_action": last_action_diag,
                            "last_thought": last_thought_diag,
                            "last_observation": last_observation_diag, # The observation before the error occurred
                            "error_step_result": step_result # The actual error result dict
                        }
                        
                        diagnosis_input = {
                            "error_message": error_message,
                            "traceback": traceback_str,
                            "execution_context": diagnosis_context,
                            "llm_url": self.llm_url # Pass agent's LLM URL
                        }

                        llm_diagnosis = "Diagnosis failed or was not attempted."
                        llm_suggestions = []
                        try:
                            self.agent_logger.info(f"{step_log_prefix} Calling llm_error_diagnosis skill...")
                            exec_context_diag = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)
                            diagnosis_result = await execute_tool(
                                tool_name="llm_error_diagnosis",
                                action_input=diagnosis_input,
                                tools_dict=self.tools,
                                context=exec_context_diag
                            )
                            if diagnosis_result.get("status") == "success":
                                llm_diagnosis = diagnosis_result.get("data", {}).get("diagnosis", "LLM provided no diagnosis.")
                                llm_suggestions = diagnosis_result.get("data", {}).get("suggested_actions", [])
                                self.agent_logger.info(f"{step_log_prefix} LLM Diagnosis: {llm_diagnosis}")
                                if llm_suggestions:
                                    self.agent_logger.info(f"{step_log_prefix} LLM Suggestions: {llm_suggestions}")
                            else:
                                error_msg_diag = diagnosis_result.get("data", {}).get("message", "Unknown error from diagnosis skill.")
                                self.agent_logger.error(f"{step_log_prefix} llm_error_diagnosis skill failed: {error_msg_diag}")
                                llm_diagnosis = f"Diagnosis skill failed: {error_msg_diag}"

                        except Exception as diag_err:
                            self.agent_logger.exception(f"{step_log_prefix} Exception calling llm_error_diagnosis skill:")
                            llm_diagnosis = f"Exception during diagnosis: {diag_err}"
                        
                        # Use the LLM diagnosis as the failure analysis
                        failure_analysis = llm_diagnosis # Keep the diagnosis for reflection if recovery fails

                        # --- Attempt Recovery based on LLM Suggestions ---
                        recovery_attempted = False
                        recovery_succeeded = False
                        if llm_suggestions:
                            self.agent_logger.info(f"{step_log_prefix} Attempting recovery based on LLM suggestions...")
                            recovery_succeeded = await self._attempt_recovery(llm_suggestions, step_log_prefix)
                            recovery_attempted = True # Mark that we tried

                        if recovery_succeeded:
                            self.agent_logger.info(f"{step_log_prefix} Recovery successful. Retrying step {i+1}.")
                            # Optional: Add a small delay before retry?
                            # await asyncio.sleep(1)
                            continue # Retry the current step
                        else:
                            if recovery_attempted:
                                self.agent_logger.warning(f"{step_log_prefix} Recovery attempt failed or no actionable suggestion found.")
                            # Proceed with standard reflection/learning as recovery didn't work
                        # --- End Recovery Attempt ---

                        # --- End LLM-Based Error Diagnosis --- (Moved comment)

                        # Proceed with standard reflection and learning using the LLM diagnosis
                        # Reflection e aprendizado padrão
                        try:
                            last_thought = next((h.split("Thought: ", 1)[1] for h in reversed(self._history) if h.startswith("Thought:")), "N/A")
                            last_action = next((h.split("Action: ", 1)[1] for h in reversed(self._history) if h.startswith("Action:")), "N/A")
                            last_observation_str = next((h.split("Observation: ", 1)[1] for h in reversed(self._history) if h.startswith("Observation:")), "N/A")

                            failure_context = {
                                "objective": original_objective,
                                "plan": plan,
                                "failed_step_index": i,
                                "failed_step": step,
                                "last_thought": last_thought,
                                "last_action": last_action,
                                "last_observation": last_observation_str
                            }
                            exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)
                            
                            reflection_result = await execute_tool(
                                tool_name="reflect_on_failure",
                                action_input=failure_context,
                                tools_dict=self.tools,
                                context=exec_context
                            )
                            self.agent_logger.info(f"Resultado de reflect_on_failure: {reflection_result}")

                            if reflection_result.get("status") == "success":
                                # Use the diagnosis from the LLM skill
                                # failure_analysis = reflection_result.get("data", {}).get("explanation", "Falha na análise.")
                                # failure_analysis is already set from llm_diagnosis
                                if failure_analysis != "Falha na análise.":
                                    self.agent_logger.info(f"Análise da falha obtida: {failure_analysis[:100]}...")
                                    # Call learn_from_failure_log immediately
                                    learn_input = {
                                        "objective": original_objective,
                                        "error_message": step_result.get('data',{}).get('message', 'Erro Desconhecido'), # Use error from step_result
                                        "failure_analysis": failure_analysis
                                    }
                                    try:
                                        self.agent_logger.info(f"Chamando learn_from_failure_log: {learn_input}")
                                        learn_result = await execute_tool(
                                            tool_name="learn_from_failure_log",
                                            action_input=learn_input,
                                            tools_dict=self.tools,
                                            context=exec_context # Re-use context
                                        )
                                        self.agent_logger.info(f"Resultado de learn_from_failure_log: {learn_result}")
                                        # Log heuristic (if successful) - Adapting logging from simple plan failure
                                        if learn_result.get("status") == "success":
                                            heuristic = learn_result.get("data", {}).get("heuristic", "N/A")
                                            if heuristic != "N/A":
                                                try:
                                                    timestamp = datetime.datetime.now(timezone.utc).isoformat(timespec='microseconds')
                                                    log_entry = {
                                                        "timestamp": timestamp,
                                                        "objective": original_objective,
                                                        "plan": plan,
                                                        "results": execution_results + [step_result], # Include current failed step
                                                        "heuristic": heuristic,
                                                        "type": "failure"
                                                    }
                                                    HEURISTIC_LOG_FILE = Path("memory/learning_logs/learned_heuristics.jsonl")
                                                    log_file = HEURISTIC_LOG_FILE
                                                    log_file.parent.mkdir(parents=True, exist_ok=True)
                                                    with open(log_file, 'a', encoding='utf-8') as f:
                                                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                                                    self.agent_logger.info(f"Heurística de falha registrada em {log_file}: {heuristic[:100]}...")
                                                except Exception as log_write_err:
                                                    self.agent_logger.exception(f"Falha ao escrever heurística de falha (ReAct) no log {log_file}: {log_write_err}")
                                        else:
                                            self.agent_logger.warning("learn_from_failure_log retornou heurística 'N/A', não registrada.")
                                    except Exception as learn_err:
                                        self.agent_logger.exception(f"Exceção ao chamar learn_from_failure_log (ReAct): {learn_err}")
                            else:
                                self.agent_logger.warning("Reflexão sobre falha (ReAct) não retornou análise.")
                        except Exception as reflect_err:
                            self.agent_logger.exception(f"Erro durante a reflexão/aprendizado da falha (ReAct): {reflect_err}")
                            failure_analysis = f"Erro na reflexão pós-falha: {reflect_err}" # Update analysis message

                            # Construct final message and stop plan execution
                            # Use the LLM diagnosis in the final message
                            final_message = f"Falha no passo {i+1}: {step_result.get('data',{}).get('message', 'Erro Desconhecido')}\n**LLM Diagnosis:** {failure_analysis}"
                            execution_results.append(step_result) # Add failed step result
                            self.agent_logger.warning(f"{log_prefix} Encerrando execução do plano devido à falha no passo {i+1}.")
                            return final_status, final_message, execution_results # Stop plan execution
                    
                    # Append successful step result (if any)
                    if step_result:
                        execution_results.append(step_result)
                    else:
                        # Should not happen if react loop yielded something, but as a safeguard:
                        self.agent_logger.warning(f"{step_log_prefix} Step completed without a final observation or answer.")
                        execution_results.append({"status": "warning", "action": "step_empty_result", "data": {"message": "Step finished without output."}})
                        final_status = "warning" # Mark overall status potentially

            except Exception as react_loop_err:
                self.agent_logger.exception(f"{log_prefix} Erro inesperado durante o loop ReAct no passo {i+1}:")
                final_status = "error"
                final_message = f"Erro crítico no loop ReAct no passo {i+1}: {react_loop_err}"
                # Append partial results if any exist before the crash
                # --- Replanejamento dinâmico se plano travar por repetição cega ---
                from a3x.core.dynamic_replanner import dynamic_replan
                if final_status == "failed" and dynamic_replan is not None:
                    self.agent_logger.info("[DynamicReplanner] Tentando replanejamento dinâmico após falha do plano.")
                    # Tenta gerar novo subplano e reexecutar
                    new_plan = await dynamic_replan(plan, execution_results)
                    if new_plan != plan:
                        self.agent_logger.info(f"[DynamicReplanner] Novo subplano gerado: {new_plan}")
                        # Reinicia execução do novo plano
                        final_status, final_message, execution_results = await self._execute_plan(new_plan, context, original_objective)
                return final_status, final_message, execution_results

        # <<< END: ELSE BLOCK FOR COMPLEX PLANS (ReAct Loop) >>>

        # If loop completes without critical errors/failures stopping it
        if final_status == "intermediate_success": # Check if last step was successful
             final_status = "completed"
             # Try to get the final answer from the last step_result if it exists
             final_message = "Plano executado com sucesso."
             if execution_results and isinstance(execution_results[-1].get("data"), dict):
                 final_answer = execution_results[-1].get("data", {}).get("final_answer")
                 if final_answer:
                     final_message = final_answer # Use the actual final answer if provided
             self.agent_logger.info("--- Execução do Plano Concluída com Sucesso ---")
        elif final_status == "unknown": # Should not happen ideally
             final_status = "error"
             final_message = "Execução do plano terminou em estado desconhecido."
        elif final_status == "warning": # If a step completed with warning
             final_message = "Plano executado, mas com avisos em alguns passos."

        return final_status, final_message, execution_results


    # New consolidated reflection and learning method
    async def _reflect_and_learn(self, perception: Dict[str, Any], plan: List[str], execution_results: List[Dict[str, Any]], final_status: str):
        """
        Encapsula o ciclo pós-execução chamando a skill unificada 'learning_cycle'.
        """
        self.agent_logger.info("--- Iniciando Fase de Reflexão e Aprendizado (via Skill Learning Cycle) --- ")
        original_objective = perception.get("processed", "Objetivo Desconhecido")

        # Preparar input para a skill learning_cycle
        learning_cycle_input = {
            "objective": original_objective,
                    "plan": plan,
            "execution_results": execution_results,
            "final_status": final_status,
            "agent_tools": self.tools, # Passar as tools do agente
            "agent_workspace": str(self.workspace_root), # Passar workspace do agente
            "agent_llm_url": self.llm_url, # Passar LLM URL do agente
        }

        # Criar contexto para a chamada da skill principal
        # Usando logger do agente e workspace
        exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)
        # Corrige agent_llm_url para nunca ser None
        if learning_cycle_input.get("agent_llm_url") is None:
            learning_cycle_input["agent_llm_url"] = ""

        try:
            # Chamar a skill unificada
            learning_result = await execute_tool(
                tool_name="learning_cycle",
                action_input=learning_cycle_input,
                tools_dict=self.tools, # Passar o registro de tools
                context=exec_context
            )

            if learning_result.get("status") == "success" or learning_result.get("status") == "warning":
                # Log sucesso ou warning da skill
                msg = learning_result.get("data", {}).get("message", "Ciclo de aprendizado concluído sem mensagem específica.")
                self.agent_logger.info(f"Resultado do Learning Cycle: {learning_result.get('status')} - {msg}")
            else:
                # Log erro da skill
                error_msg = learning_result.get("data", {}).get("message", "Erro desconhecido no ciclo de aprendizado.")
                self.agent_logger.error(f"Skill learning_cycle falhou: {error_msg}")

        except Exception as e:
            self.agent_logger.exception("Erro crítico ao chamar a skill learning_cycle:")

        self.agent_logger.info("--- Fase de Reflexão e Aprendizado Concluída --- ")

        # Integração: autogeração de skills a partir de heurísticas após ciclo de aprendizado
        try:
            from a3x.core.skill_autogen import autogen_skills_from_heuristics
            # Coleta heurísticas do ciclo atual
            from a3x.core.learning_logs import read_heuristics_log
            heuristics = read_heuristics_log()
            generated_skills = await autogen_skills_from_heuristics(heuristics, llm_url=self.llm_url or "")
            if generated_skills:
                self.agent_logger.info(f"[SkillAutogen] Skills auto-geradas: {generated_skills}")
        except Exception as autogen_err:
            self.agent_logger.warning(f"[SkillAutogen] Falha ao rodar autogeração de skills: {autogen_err}")

        # Integração: autoavaliação por simulação após ciclo de aprendizado
        try:
            from a3x.core.simulation import auto_evaluate_agent
            # Exemplo de benchmark: simula o último plano executado
            benchmark_plans = [plan] if plan else []
            sim_result = await auto_evaluate_agent(benchmark_plans, context=None, heuristics=None, llm_url=self.llm_url or "")
            self.agent_logger.info(f"[Simulation] Autoavaliação por simulação concluída: {sim_result}")
        except Exception as sim_err:
            self.agent_logger.warning(f"[Simulation] Falha ao rodar autoavaliação por simulação: {sim_err}")

    # --- Helper Method for Runtime Adaptation ---
    async def _attempt_recovery(self, suggestions: List[str], step_log_prefix: str) -> bool:
        """
        Attempts simple recovery actions based on LLM suggestions.
        Returns True if recovery was successful and step should be retried, False otherwise.
        """
        import os # Import needed for os.makedirs
        import re  # Import needed for regex
        import datetime # For heuristic timestamp
        from datetime import timezone # For heuristic timestamp
        from pathlib import Path # For heuristic path handling
        from a3x.core.learning_logs import log_heuristic_with_traceability # For logging

        self.agent_logger.info(f"{step_log_prefix} Analyzing {len(suggestions)} suggestions for recovery actions.")
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            self.agent_logger.debug(f"{step_log_prefix} Evaluating suggestion: '{suggestion}'")

            # 1. Simple Retry Suggestion
            if "retry" in suggestion_lower:
                self.agent_logger.info(f"{step_log_prefix} Recovery action: Retrying step based on suggestion '{suggestion}'.")
                # Log heuristic? Maybe later.
                return True # Indicate success, triggering a retry

            # 2. Instalar Dependência (apenas log/sugestão por segurança)
            install_match = re.search(r"(?:install|instalar) (?:dependency|dependência|package|pacote) ['\"]?([\w\-\.]+)['\"]?", suggestion, re.IGNORECASE)
            if install_match:
                dep_name = install_match.group(1).strip()
                self.agent_logger.warning(f"{step_log_prefix} Sugestão de instalar dependência '{dep_name}' detectada. Por segurança, apenas logando a sugestão.")
                # Log heuristic de sugestão de instalação
                try:
                    plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    recovery_heuristic = {
                        "type": "recovery_suggestion",
                        "trigger_suggestion": suggestion,
                        "action_taken": "Suggested install dependency",
                        "details": {"dependency": dep_name},
                        "context": {"step_log_prefix": step_log_prefix}
                    }
                    log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="pending_manual")
                    self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (suggest install dependency).")
                except Exception as log_err:
                    self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (install dependency): {log_err}")
                # Não executa automaticamente por segurança
                continue

            # 3. Corrigir Permissões (apenas log/sugestão)
            if "permission" in suggestion_lower or "permissão" in suggestion_lower:
                self.agent_logger.warning(f"{step_log_prefix} Sugestão de correção de permissões detectada. Apenas logando a sugestão.")
                try:
                    plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    recovery_heuristic = {
                        "type": "recovery_suggestion",
                        "trigger_suggestion": suggestion,
                        "action_taken": "Suggested fix permissions",
                        "details": {},
                        "context": {"step_log_prefix": step_log_prefix}
                    }
                    log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="pending_manual")
                    self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (suggest fix permissions).")
                except Exception as log_err:
                    self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (fix permissions): {log_err}")
                # Não executa automaticamente por segurança
                continue

            # 4. (Futuro) Chamar skill alternativa sugerida pelo LLM
            # Exemplo: "Use skill 'criar_arquivo_template' para resolver"
            alt_skill_match = re.search(r"(?:use|utilize|chame|call) skill ['\"]?(\w+)['\"]?", suggestion, re.IGNORECASE)
            if alt_skill_match:
                alt_skill = alt_skill_match.group(1)
                self.agent_logger.info(f"{step_log_prefix} Sugestão de chamar skill alternativa '{alt_skill}' detectada. (Implementação futura)")
                # Aqui pode-se implementar chamada automática de skills alternativas, se desejado.
                # Não executa automaticamente por enquanto
                continue
            continue
            if create_match:
                dir_to_create = create_match.group(1).strip()
                # Resolve relative paths against workspace root
                if not os.path.isabs(dir_to_create):
                     dir_to_create_abs = self.workspace_root / dir_to_create
                else:
                     # Be cautious with absolute paths outside workspace? For now, allow.
                     dir_to_create_abs = Path(dir_to_create)
                
                self.agent_logger.info(f"{step_log_prefix} Recovery action: Attempting to create directory '{dir_to_create_abs}' based on suggestion '{suggestion}'.")
                try:
                    # Check if it already exists (idempotency)
                    if dir_to_create_abs.exists():
                         self.agent_logger.info(f"{step_log_prefix} Directory '{dir_to_create_abs}' already exists. Recovery considered successful.")
                         # Log heuristic for confirmed existence
                         try:
                             plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                             execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                             recovery_heuristic = {
                                 "type": "recovery_success",
                                 "trigger_suggestion": suggestion,
                                 "action_taken": "Confirmed directory exists",
                                 "details": {"directory": str(dir_to_create_abs)},
                                 "context": {"step_log_prefix": step_log_prefix}
                             }
                             log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="confirmed_effective")
                             self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (directory exists).")
                         except Exception as log_err:
                             self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (directory exists): {log_err}")
                         return True # Directory exists, step can likely proceed

                    dir_to_create_abs.mkdir(parents=True, exist_ok=True)
                    self.agent_logger.info(f"{step_log_prefix} Successfully created directory '{dir_to_create_abs}'.")
                    # Log adaptation heuristic for successful creation
                    try:
                        plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                        execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                        recovery_heuristic = {
                            "type": "recovery_success",
                            "trigger_suggestion": suggestion,
                            "action_taken": "Created directory",
                            "details": {"directory": str(dir_to_create_abs)},
                            "context": {"step_log_prefix": step_log_prefix}
                        }
                        log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="confirmed_effective")
                        self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (directory created).")
                    except Exception as log_err:
                        self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (directory created): {log_err}")
                    return True # Indicate success, triggering a retry
                except Exception as e:
                    self.agent_logger.error(f"{step_log_prefix} Failed to create directory '{dir_to_create_abs}': {e}")
                    # Don't return True, let it proceed to failure reflection
                    return False # Explicitly indicate recovery failed here

            # 3. (Future) Call Auxiliary Skill Suggestion
            # call_match = re.search(r"call skill ['\"]?(\w+)['\"]? with parameters? (\{.*\})", suggestion, re.IGNORECASE)
            # if call_match:
            #     skill_to_call = call_match.group(1)
            #     try:
            #         params = json.loads(call_match.group(2))
            #         # await execute_tool(...)
            #         # return True if successful
            #     except json.JSONDecodeError:
            #         self.agent_logger.warning(f"{step_log_prefix} Could not parse parameters for suggested skill call: {call_match.group(2)}")
            #     except Exception as e:
            #          self.agent_logger.error(f"{step_log_prefix} Error executing suggested skill '{skill_to_call}': {e}")


            # Add more suggestion parsing logic here (e.g., install dependency, fix permissions - though these might need user interaction)

        self.agent_logger.info(f"{step_log_prefix} No actionable recovery suggestion found or recovery failed.")
        return False # No actionable suggestion found or recovery failed

    # --- Métodos Auxiliares Mantidos/Adaptados ---

    # Removed _learn method

    # <<< REMOVING Zombie method _reflect >>>
    # async def _reflect(
    #     self,
    #     perception: Dict[str, Any],
    #     plan: List[str],
    #     execution_results: List[Dict[str, Any]],
    # ) -> Dict[str, Any]: ...

    # <<< REMOVING Zombie method _simulate_step >>>
    # async def _simulate_step(
    #     self, plan_step: str, context: Dict[str, Any]
    # ) -> Dict[str, Any]: ...

    # ... (Métodos _perceive, _retrieve_context já adaptados acima) ...

    # Helper removed - _log_heuristic (logic should be inside learning skills)
    # async def _log_heuristic(self, log_entry: Dict[str, Any]): ...
