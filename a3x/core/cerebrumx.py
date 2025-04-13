# core/cerebrumx.py
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple
import json
import os
import datetime
import re
from datetime import timezone
from pathlib import Path
from collections import namedtuple
import asyncio # Added import

# Import base agent and other necessary core components
from a3x.core.agent import ReactAgent, is_introspective_query
from a3x.core.skills import get_skill_descriptions
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext

# Centralized config and memory manager
from a3x.core import config as a3x_config
from a3x.core.memory.memory_manager import MemoryManager
from a3x.core.config import HEURISTIC_LOG_FILE # Import the centralized path

# Potentially import memory, reflection components later

# Import skills needed for reflect_and_learn
# Corrected relative imports
from ..skills.core.reflect_on_success import reflect_on_success
from ..skills.core.learn_from_failure_log import learn_from_failure_log
from ..skills.core.reflect_on_failure import reflect_on_failure # <<< ADDED import
# Generalization/Consolidation skills will be called within _reflect_and_learn

# Initialize logger for this module
cerebrumx_logger = logging.getLogger(__name__) # <<< Rename logger? agent_logger already exists in ReactAgent

# <<< ADDED Import >>>
from .agent import _is_simple_list_files_task, ReactAgent # <<< ADD ReactAgent import

# Helper to create context for direct execution calls
_ToolExecutionContext = namedtuple("_ToolExecutionContext", ["logger", "workspace_root", "llm_url", "tools_dict"])

class CerebrumXAgent(ReactAgent): # Inheriting from ReactAgent
    """
    Agente Autônomo Adaptável Experimental com ciclo cognitivo unificado.
    Incorpora percepção, planejamento, execução ReAct, reflexão e aprendizado em um único fluxo.
    """

    def __init__(self, system_prompt: str, llm_url: Optional[str] = None, tools_dict: Optional[Dict[str, Dict[str, Any]]] = None, exception_policy=None):
        """Inicializa o Agente CerebrumX."""
        super().__init__(system_prompt, llm_url, tools_dict=tools_dict)
        self.agent_logger.info("[CerebrumX INIT] Agente CerebrumX inicializado (Ciclo Unificado).")
        # Configuração centralizada para MemoryManager
        memory_config = {
            "SEMANTIC_INDEX_PATH": a3x_config.SEMANTIC_INDEX_PATH,
            "SEMANTIC_SEARCH_TOP_K": a3x_config.SEMANTIC_SEARCH_TOP_K,
            "EPISODIC_RETRIEVAL_LIMIT": a3x_config.EPISODIC_RETRIEVAL_LIMIT,
        }
        self.memory_manager = MemoryManager(memory_config)
        # ExceptionPolicy configurável
        if exception_policy is None:
            from a3x.core.exception_policy import ExceptionPolicy
            self.exception_policy = ExceptionPolicy()
        else:
            self.exception_policy = exception_policy
        # No initial_perception needed here if run takes objective

    # --- Novo Ciclo Cognitivo Unificado ---
    async def run(self, objective: str) -> Dict[str, Any]: # Now returns final result dict
        """
        Executa o ciclo cognitivo completo unificado do A³X.
        Perceber -> Planejar -> Executar -> Refletir & Aprender.
        Retorna um dicionário com o resultado final ou o status da execução.
        """
        import time
        from a3x.core.auto_evaluation import auto_evaluate_task
        # Import moved to top level to avoid potential issues inside methods
        from a3x.core.db_utils import add_episodic_record

        self.agent_logger.info(f"--- Iniciando Ciclo Cognitivo Unificado --- Objetivo: {objective[:100]}..." )
        start_time = time.time()

        # 1. Percepção (Simplificado)
        perception = self._perceive(objective)
        self.agent_logger.info(f"Percepção processada: {perception}")

        # 2. Recuperação de Contexto
        context = None
        try:
            context = await self._retrieve_context(perception)
            self.agent_logger.info(f"Contexto recuperado: {str(context)[:100]}...")
        except Exception as e:
            self.exception_policy.handle(e, context="Erro durante recuperação de contexto")
            context = {"combined_summary": "Erro ao recuperar contexto.", "semantic_results": [], "episodic_results": [], "query": perception.get("processed", "")}

        # 3. Planejamento
        plan = await self._plan(perception, context)
        self.agent_logger.info(f"Plano gerado: {plan}")
        if not plan:
             self.agent_logger.error("Falha ao gerar plano. Abortando ciclo.")
             return {"status": "error", "message": "Falha crítica no planejamento."}

        # 4. Execução do Plano
        final_status, final_message, execution_results = await self._execute_plan(plan, context, objective)
        self.agent_logger.info(f"Execução do plano finalizada. Status: {final_status}")

        # 5. Reflexão e Aprendizado Pós-Execução
        await self._reflect_and_learn(perception, plan, execution_results, final_status)

        # 6. Autoavaliação Cognitiva
        end_time = time.time()
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
        return {"status": final_status, "message": final_message, "results": execution_results}

    # --- Métodos Internos do Ciclo ---

    def _perceive(self, objective: str) -> Dict[str, Any]:
        """Processa a percepção inicial (objetivo)."""
        self.agent_logger.info("Processando percepção...")
        return {"processed": objective}

    async def _retrieve_context(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Recupera contexto relevante da memória semântica (FAISS) e episódica (Recente)."""
        self.agent_logger.info("Recuperando contexto da memória (Semântica e Episódica Recente)...")
        query = perception.get("processed", "")
        if not query:
            self.agent_logger.warning("Query de percepção vazia, não é possível buscar contexto semântico.")

        # --- Busca Semântica (FAISS) --- #
        semantic_matches = []
        semantic_summary = "Nenhuma memória semântica relevante encontrada."
        if query:
            try:
                from a3x.core.embeddings import get_embedding
                from a3x.core.semantic_memory_backend import search_index
                from a3x.core.config import SEMANTIC_SEARCH_TOP_K
                from a3x.core.db_utils import add_episodic_record # Moved import here

                index_path_base = a3x_config.SEMANTIC_INDEX_PATH
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
                        semantic_matches = search_results
                        semantic_summary = "\nContexto Semântico Relevante (FAISS):\n"
                        for i, res in enumerate(search_results):
                            content = res.get("metadata", {}).get("content", "<Conteúdo indisponível>")
                            distance = res.get("distance", -1.0)
                            semantic_summary += f"- [Dist: {distance:.3f}] {content}\n"
                        semantic_summary = semantic_summary.strip()

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
            from a3x.core.db_utils import retrieve_recent_episodes
            from a3x.core.config import EPISODIC_RETRIEVAL_LIMIT

            self.agent_logger.info(f"Buscando os {EPISODIC_RETRIEVAL_LIMIT} episódios mais recentes...")
            recent_episodes = retrieve_recent_episodes(limit=EPISODIC_RETRIEVAL_LIMIT)

            if recent_episodes:
                 self.agent_logger.info(f"Encontrados {len(recent_episodes)} episódios recentes.")
                 episodic_matches = [dict(row) for row in recent_episodes]
                 episodic_summary = "\nContexto Episódico Recente:\n"
                 for i, episode in enumerate(episodic_matches):
                      ctx = episode.get("context", "?")
                      act = episode.get("action", "?")
                      out = episode.get("outcome", "?")
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
        combined_summary = f"{semantic_summary}\n\n{episodic_summary}".strip()
        final_context = {
            "combined_summary": combined_summary,
            "semantic_results": semantic_matches,
            "episodic_results": episodic_matches,
            "query": query
        }
        return final_context

    async def _plan(self, perception: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Gera um plano de execução baseado na percepção e contexto."""
        self.agent_logger.info("--- Gerando Plano de Execução ---")
        objective = perception.get("processed", "")
        context_summary = context.get("combined_summary", "")
        tool_descs = get_skill_descriptions()

        # Consultar heurísticas relevantes
        heuristics_context = None
        # Exemplo placeholder:
        # try:
        #     consult_result = await execute_tool(...) # Chamar consult_learned_heuristics
        #     if consult_result["status"] == "success":
        #         heuristics_context = ... # Formatar resultado
        # except Exception as e:
        #     self.agent_logger.warning(f"Erro ao consultar heurísticas: {e}")

        # Gerar plano usando o LLM
        try:
            from a3x.core.planner import generate_plan
            plan_to_execute = await generate_plan(
                objective=objective,
                tool_descriptions=tool_descs,
                agent_logger=self.agent_logger,
                llm_url=self.llm_url,
                heuristics_context=heuristics_context
            )
            return plan_to_execute if plan_to_execute is not None else []
        except Exception as e:
            self.agent_logger.exception(f"Erro fatal durante a geração do plano: {e}")
            # Tentar gerar um plano de fallback simples (apenas Final Answer com erro)
            return [f"Use the final_answer tool to report planning failure: {e}"]

    async def _execute_plan(self, plan: List[str], context: Dict[str, Any], original_objective: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Executa um plano iterando sobre cada passo usando o ciclo ReAct."""
        self.agent_logger.info(f"[ExecutePlan] Iniciando execução do plano com {len(plan)} passos via ReAct.")
        execution_results: List[Dict[str, Any]] = []
        final_status = "unknown"
        final_message = "Execução iniciada, mas nenhum passo concluído."
        error_occurred = False
        critical_error_message = None # Para erros fatais fora do loop ReAct

        # Criar contexto para execução de skills (usado por _perform_react_iteration)
        exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)

        # --- Execução via ReAct para TODOS os planos --- #
        # Initialize history for this plan execution
        self._history = []

        try:
            for i, step in enumerate(plan):
                step_log_prefix = f"[ExecutePlan Step {i+1}/{len(plan)}]"
                self.agent_logger.info(f"{step_log_prefix} Executando: '{step}'")

                step_results_agg = [] # Agregador para resultados deste passo
                last_observation = None
                last_thought = "N/A"
                last_action = "N/A"
                step_final_answer = None

                try:
                    async for react_output in self._perform_react_iteration(step, step_log_prefix):
                        step_results_agg.append(react_output) # Log raw output

                        if react_output.get("type") == "thought":
                            last_thought = react_output.get("content", "N/A")
                        elif react_output.get("type") == "action":
                            last_action = react_output.get("name", "N/A")
                        elif react_output.get("type") == "observation":
                            last_observation = react_output.get("content")
                        elif react_output.get("type") == "final_answer":
                            step_final_answer = react_output.get("content", "No final answer provided for step.")
                            self.agent_logger.info(f"{step_log_prefix} Passo considerado concluído pelo LLM (Final Answer). Resposta: {step_final_answer[:100]}...")
                            break # Sair do loop ReAct para este passo
                        elif react_output.get("type") == "error":
                            error_occurred = True
                            final_status = "failed"
                            critical_error_message = react_output.get("content", "Erro desconhecido na iteração ReAct.")
                            self.agent_logger.error(f"{step_log_prefix} Erro durante iteração ReAct: {critical_error_message}")

                            # --- Aprendizado Imediato de Falha --- #
                            failure_context = {
                                "objective": original_objective,
                                "failed_step": step,
                                "failed_step_index": i,
                                "plan": plan,
                                "last_thought": last_thought,
                                "last_action": last_action,
                                "last_observation": str(last_observation) if last_observation else "N/A",
                                "error_message": critical_error_message
                            }
                            try:
                                self.agent_logger.warning(f"{step_log_prefix} Tentando refletir e aprender com a falha ReAct...")
                                reflection_result = await execute_tool(tool_name="reflect_on_failure", action_input=failure_context, tools_dict=self.tools, context=exec_context)
                                failure_analysis = reflection_result.get("data", {}).get("explanation", "Falha na análise.")
                                self.agent_logger.info(f"{step_log_prefix} Análise da falha ReAct obtida: {failure_analysis[:100]}...")

                                learn_input = {"failure_analysis": failure_analysis, "objective": original_objective, "error_message": critical_error_message}
                                learn_result = await execute_tool(tool_name="learn_from_failure_log", action_input=learn_input, tools_dict=self.tools, context=exec_context)
                                learned_heuristic = learn_result.get("data", {}).get("heuristic")
                                if learned_heuristic:
                                    self.agent_logger.info(f"{step_log_prefix} Heurística de falha ReAct registrada: {learned_heuristic[:100]}...")
                                    # O log no arquivo JSONL é feito dentro da skill agora
                            except Exception as learn_err:
                                self.agent_logger.error(f"{step_log_prefix} Erro durante o aprendizado da falha ReAct: {learn_err}", exc_info=True)
                            break # Sair do loop ReAct para este passo

                    # Fim do loop ReAct para o passo atual
                    # Adicionar resultado agregado do passo à lista geral
                    execution_results.append({
                        "step": step,
                        "react_trace": step_results_agg,
                        "step_status": "error" if error_occurred else "completed",
                        "step_final_answer": step_final_answer # Pode ser None se não houve Final Answer explícito
                    })

                    if error_occurred:
                        self.agent_logger.warning(f"{step_log_prefix} Erro detectado, interrompendo execução do plano.")
                        break # Interromper execução do plano completo

                except Exception as step_react_err:
                    # Erro inesperado DENTRO do loop _perform_react_iteration
                    self.agent_logger.exception(f"{step_log_prefix} Erro não capturado dentro do loop ReAct para o passo: '{step}'")
                    error_occurred = True
                    final_status = "error"
                    critical_error_message = f"Erro crítico inesperado no loop ReAct: {step_react_err}"
                    # Adicionar um resultado de erro para este passo
                    execution_results.append({
                        "step": step,
                        "react_trace": step_results_agg, # Pode ter resultados parciais
                        "step_status": "error",
                        "error_message": critical_error_message
                    })
                    break # Interromper execução do plano

            # Fim do loop principal do plano
            if not error_occurred:
                final_status = "completed"
                # Usar a resposta final do último passo ReAct ou uma mensagem padrão
                last_step_result = execution_results[-1] if execution_results else {}
                final_message = last_step_result.get("step_final_answer", "Plano concluído com sucesso (sem resposta final explícita do último passo).")
            else:
                 # Status já 'failed' ou 'error', a mensagem de erro foi definida no loop
                 final_message = critical_error_message or "Falha na execução do plano (causa específica não registrada)."

        except Exception as outer_err:
            # Erro inesperado FORA do loop principal do plano (e.g., erro no setup do loop)
            self.agent_logger.exception("Erro crítico inesperado durante a execução geral do plano:")
            final_status = "error"
            final_message = f"Erro crítico na execução do plano: {outer_err}"
            # Adicionar um registro de erro geral se execution_results estiver vazio
            if not execution_results:
                execution_results.append({"step": "N/A", "status": "error", "message": final_message})

        # Limpar histórico após execução do plano
        self._history = []

        return final_status, final_message, execution_results

    # --- Reflexão e Aprendizado Pós-Execução --- #
    async def _reflect_and_learn(self, perception: Dict[str, Any], plan: List[str], execution_results: List[Dict[str, Any]], final_status: str):
        """Chama a skill 'learning_cycle' para análise pós-execução."""
        self.agent_logger.info("--- Iniciando Fase de Reflexão e Aprendizado (via Skill Learning Cycle) --- ")
        try:
            # Criar contexto para a skill learning_cycle
            exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root, llm_url=self.llm_url, tools_dict=self.tools)

            learning_cycle_input = {
                "objective": perception.get("processed", "N/A"),
                "plan": plan,
                "execution_results": execution_results,
                "final_status": final_status,
                "agent_tools": self.tools, # Passar o dicionário de ferramentas
                "agent_workspace": str(self.workspace_root), # Passar como string
                "agent_llm_url": self.llm_url
            }
            learning_result = await execute_tool(tool_name="learning_cycle", action_input=learning_cycle_input, tools_dict=self.tools, context=exec_context)
            # Logar resultado do ciclo de aprendizado
            result_status = learning_result.get("status", "error")
            result_message = learning_result.get("data", {}).get("message", "Falha desconhecida no ciclo de aprendizado.")
            self.agent_logger.info(f"Resultado do Learning Cycle: {result_status} - {result_message}")

        except Exception as e:
            self.agent_logger.exception("Erro fatal durante a fase de reflexão e aprendizado:")
        finally:
             self.agent_logger.info("--- Fase de Reflexão e Aprendizado Concluída --- ")

    async def _attempt_recovery(self, suggestions: List[str], step_log_prefix: str) -> bool:
        """
        Attempts simple recovery actions based on LLM suggestions.
        """
        import os
        import re
        import datetime
        from datetime import timezone
        from pathlib import Path
        from a3x.core.learning_logs import log_heuristic_with_traceability

        self.agent_logger.info(f"{step_log_prefix} Analyzing {len(suggestions)} suggestions for recovery actions.")
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            self.agent_logger.debug(f"{step_log_prefix} Evaluating suggestion: '{suggestion}'")

            if "retry" in suggestion_lower:
                self.agent_logger.info(f"{step_log_prefix} Recovery action: Retrying step based on suggestion '{suggestion}'.")
                return True

            install_match = re.search(r"(?:install|instalar) (?:dependency|dependência|package|pacote) ['\"]?([\w\-\.]+)['\"]?", suggestion, re.IGNORECASE)
            if install_match:
                dep_name = install_match.group(1).strip()
                self.agent_logger.warning(f"{step_log_prefix} Sugestão de instalar dependência '{dep_name}' detectada. Por segurança, apenas logando a sugestão.")
                try: # Try block for logging
                    plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    recovery_heuristic = {"type": "recovery_suggestion", "trigger_suggestion": suggestion, "action_taken": "Suggested install dependency", "details": {"dependency": dep_name}, "context": {"step_log_prefix": step_log_prefix}}
                    log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="pending_manual")
                    self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (suggest install dependency).")
                except Exception as log_err: # Corresponding except for logging try
                    self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (install dependency): {log_err}")
                continue

            if "permission" in suggestion_lower or "permissão" in suggestion_lower:
                self.agent_logger.warning(f"{step_log_prefix} Sugestão de correção de permissões detectada. Apenas logando a sugestão.")
                try: # Try block for logging
                    plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    recovery_heuristic = {"type": "recovery_suggestion", "trigger_suggestion": suggestion, "action_taken": "Suggested fix permissions", "details": {}, "context": {"step_log_prefix": step_log_prefix}}
                    log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="pending_manual")
                    self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (suggest fix permissions).")
                except Exception as log_err: # Corresponding except for logging try
                    self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (fix permissions): {log_err}")
                continue

            alt_skill_match = re.search(r"(?:use|utilize|chame|call) skill ['\"]?(\w+)['\"]?", suggestion, re.IGNORECASE)
            if alt_skill_match:
                alt_skill = alt_skill_match.group(1)
                self.agent_logger.info(f"{step_log_prefix} Sugestão de chamar skill alternativa '{alt_skill}' detectada. (Implementação futura)")
                continue

            # Corrected: Added create_match definition
            create_match = re.search(r"(?:create|criar) (?:directory|diretório) ['\"]?([^'\"]+)['\"]?", suggestion, re.IGNORECASE)
            if create_match:
                dir_to_create = create_match.group(1).strip()
                if not os.path.isabs(dir_to_create):
                     dir_to_create_abs = self.workspace_root / dir_to_create
                else:
                     dir_to_create_abs = Path(dir_to_create)
                self.agent_logger.info(f"{step_log_prefix} Recovery action: Attempting to create directory '{dir_to_create_abs}' based on suggestion '{suggestion}'.")
                try:
                    if dir_to_create_abs.exists():
                         self.agent_logger.info(f"{step_log_prefix} Directory '{dir_to_create_abs}' already exists. Recovery considered successful.")
                         try: # Try block for logging
                             plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                             execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                             recovery_heuristic = {"type": "recovery_success", "trigger_suggestion": suggestion, "action_taken": "Confirmed directory exists", "details": {"directory": str(dir_to_create_abs)}, "context": {"step_log_prefix": step_log_prefix}}
                             log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="confirmed_effective")
                             self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (directory exists).")
                         except Exception as log_err: # Corresponding except for logging try
                             self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (directory exists): {log_err}")
                         return True

                    dir_to_create_abs.mkdir(parents=True, exist_ok=True)
                    self.agent_logger.info(f"{step_log_prefix} Successfully created directory '{dir_to_create_abs}'.")
                    try: # Try block for logging
                        plan_id = f"plan-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                        execution_id = f"exec-recovery-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                        recovery_heuristic = {"type": "recovery_success", "trigger_suggestion": suggestion, "action_taken": "Created directory", "details": {"directory": str(dir_to_create_abs)}, "context": {"step_log_prefix": step_log_prefix}}
                        log_heuristic_with_traceability(recovery_heuristic, plan_id, execution_id, validation_status="confirmed_effective")
                        self.agent_logger.info(f"{step_log_prefix} Logged recovery heuristic (directory created).")
                    except Exception as log_err: # Corresponding except for logging try
                        self.agent_logger.exception(f"{step_log_prefix} Failed to log recovery heuristic (directory created): {log_err}")
                    return True
                except Exception as e:
                    self.agent_logger.error(f"{step_log_prefix} Failed to create directory '{dir_to_create_abs}': {e}")
                    return False # Explicitly indicate recovery failed here

        self.agent_logger.info(f"{step_log_prefix} No actionable recovery suggestion found or recovery failed.")
        return False
