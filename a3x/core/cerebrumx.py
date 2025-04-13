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
import time # <<< ADDED for timing execution

# Import base agent and other necessary core components
from a3x.core.agent import ReactAgent, is_introspective_query
from a3x.core.skills import get_skill_descriptions, get_skill_registry
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

# <<< ADDED Fragment Registry Import >>>
from a3x.fragments.registry import FragmentRegistry
from a3x.fragments.base import BaseFragment # Import base class for type hinting

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

    def __init__(self, system_prompt: str, llm_url: Optional[str] = None, tools_dict: Optional[Dict[str, Dict[str, Any]]] = None, exception_policy=None, agent_config: Optional[Dict] = None):
        """Inicializa o Agente CerebrumX."""
        # Initialize ReactAgent first
        super().__init__(system_prompt, llm_url, tools_dict=tools_dict)
        self.agent_logger.info("[CerebrumX INIT] Agente CerebrumX inicializado (Ciclo Unificado).")
        self.config = agent_config or {}

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

        # <<< ADDED Fragment Registry Initialization >>>
        # Pass shared dependencies (LLM, skills) and config to the registry
        # Using self.tools as a simplified skill registry for now
        # TODO: Implement a proper SkillRegistry class if needed
        self.fragment_registry = FragmentRegistry(
            llm_interface=self.llm, # Pass the LLM interface from ReactAgent
            skill_registry=get_skill_registry(), # Pass the actual skill registry
            config=self.config # Pass the main agent config
        )
        self.agent_logger.info(f"[CerebrumX INIT] Fragment Registry inicializado. {len(self.fragment_registry.list_available_fragments())} fragments carregados.")
        # <<< END ADDED Section >>>

    # --- Novo Ciclo Cognitivo Unificado ---
    async def run(self, objective: str) -> Dict[str, Any]: # Now returns final result dict
        """
        Executa o ciclo cognitivo completo unificado do A³X.
        Perceber -> Planejar -> Executar -> Refletir & Aprender.
        Retorna um dicionário com o resultado final ou o status da execução.
        """
        start_time = time.time()
        from a3x.core.auto_evaluation import auto_evaluate_task
        from a3x.core.db_utils import add_episodic_record

        self.agent_logger.info(f"--- Iniciando Ciclo Cognitivo Unificado --- Objetivo: {objective[:100]}..." )

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

        # 3. Seleção de Fragment (Roteamento da Tarefa)
        selected_fragment = await self._select_fragment(perception, context)

        # 4. Execução via Fragment Selecionado
        final_status = "error"
        final_message = "Nenhum fragment apropriado foi selecionado para a tarefa."
        execution_trace = [] # Trace da execução do fragment
        selected_fragment_name = "N/A"

        if selected_fragment:
            selected_fragment_name = selected_fragment.get_name()
            self.agent_logger.info(f"Fragment '{selected_fragment_name}' selecionado. Iniciando execução...")
            final_fragment_result, execution_trace = await self._execute_fragment(selected_fragment, perception["processed"], context)
            final_status = final_fragment_result.get("status", "error")
            # Get message from final_answer or observation or default message
            final_message = final_fragment_result.get("final_answer") or final_fragment_result.get("observation") or final_fragment_result.get("message", "Fragment execution finished with no message.")
            self.agent_logger.info(f"Execução do Fragment '{selected_fragment_name}' finalizada. Status: {final_status}")
        else:
            self.agent_logger.error(f"Falha ao selecionar fragment para o objetivo: {objective[:100]}...")
            # Optionally, try a default fragment or report error
            # Update metrics for routing failure?

        # 5. Reflexão e Aprendizado Pós-Execução
        await self._reflect_and_learn(perception, selected_fragment_name, execution_trace, final_status)

        # 6. Autoavaliação Cognitiva
        end_time = time.time()
        heuristics_used = []
        # for res in execution_trace:
        #     if "heuristic" in res.get("data", {}):
        #         heuristics_used.append(res["data"]["heuristic"])

        # The concept of a linear "plan" changes. We use the selected fragment name.
        auto_evaluate_task(
            objective=objective,
            plan=[f"Route to: {selected_fragment_name}"], # Represent plan as routing decision
            execution_results=execution_trace,
            heuristics_used=heuristics_used,
            start_time=start_time,
            end_time=end_time
        )

        # 7. Validação de heurísticas (Se aplicável ao nível do orquestrador)
        # This might be more relevant within fragments or based on consolidated learning
        # try:
        #     from a3x.core.heuristics_validator import validate_heuristics
        #     task_info = {
        #         "objective": objective,
        #         "selected_fragment": selected_fragment_name,
        #         "execution_results": execution_trace
        #     }
        #     # validate_heuristics([task_info]) # Needs adaptation for fragment context
        #     self.agent_logger.info("[HeuristicsValidator] Validação de heurísticas executada ao final do ciclo (adaptar para fragments).")
        # except Exception as e:
        #     self.agent_logger.warning(f"[HeuristicsValidator] Falha ao validar heurísticas: {e}")

        # 8. Retornar Resultado Final
        self.agent_logger.info("--- Ciclo Cognitivo Unificado Concluído --- Fragment: {selected_fragment_name}")
        return {"status": final_status, "message": final_message, "fragment_used": selected_fragment_name, "results": execution_trace} # Return trace

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

    async def _select_fragment(self, perception: Dict[str, Any], context: Dict[str, Any]) -> Optional[BaseFragment]:
        """
        Seleciona o Fragment mais apropriado para a tarefa usando o FragmentRegistry.
        """
        objective = perception.get("processed", "")
        self.agent_logger.info(f"Roteando tarefa: '{objective[:100]}...' para um Fragment adequado.")

        if not objective:
             self.agent_logger.error("Objetivo vazio, impossível selecionar fragment.")
             return None

        try:
            selected_fragment = await self.fragment_registry.select_fragment_for_task(objective, context)
            if selected_fragment:
                 self.agent_logger.info(f"Fragment '{selected_fragment.get_name()}' selecionado pelo registry.")
                 return selected_fragment
            else:
                 self.agent_logger.warning("Nenhum fragment específico foi selecionado pelo registry.")
                 # TODO: Implementar fallback ou estratégia de erro
                 # Could try a 'GeneralPurposeFragment' or return None
                 return None
        except Exception as e:
             self.agent_logger.error(f"Erro durante a seleção de fragment: {e}", exc_info=True)
             return None

    async def _execute_fragment(self, fragment: BaseFragment, objective: str, context: Optional[Dict]) -> Tuple[Dict, List[Dict]]:
        """
        Executa o método run_and_optimize do Fragment selecionado.
        """
        self.agent_logger.info(f"Executando Fragment '{fragment.get_name()}' para objetivo: {objective[:100]}...")
        final_result = {"status": "error", "message": "Fragment execution failed to start.", "type": "error"}
        execution_trace = []

        try:
            # Chama o método wrapper do fragment que lida com execução, métricas e otimização
            final_result, execution_trace = await fragment.run_and_optimize(objective, context)
            self.agent_logger.info(f"Fragment '{fragment.get_name()}' terminou. Status final: {final_result.get('status')}")

        except Exception as e:
            self.agent_logger.exception(f"Erro não capturado durante a execução do Fragment '{fragment.get_name()}':")
            error_message = f"Unhandled exception during fragment execution: {e}"
            final_result = {"status": "error", "type": "error", "message": error_message}
            # Append error to trace if possible
            execution_trace.append(final_result)

        return final_result, execution_trace

    # --- Reflexão e Aprendizado Pós-Execução --- #
    async def _reflect_and_learn(self, perception: Dict[str, Any], fragment_name: str, execution_trace: List[Dict[str, Any]], final_status: str):
        """
        Analisa os resultados da execução do Fragment e aplica estratégias de aprendizado.
        """
        self.agent_logger.info(f"Iniciando Reflexão e Aprendizado para Fragment '{fragment_name}' (Status: {final_status})...")

        # Create execution context for reflection skills
        # Note: Reflection skills might need access to the *Orchestrator's* tools
        # or potentially tools specific to reflection/analysis.
        exec_context = _ToolExecutionContext(
            logger=self.agent_logger,
            workspace_root=self.workspace_root,
            llm_url=self.llm_url,
            tools_dict=self.tools # Or a specific subset for reflection
        )

        try:
            objective = perception.get("processed", "")
            if final_status == "success":
                # --- Learn from Success ---
                self.agent_logger.info(f"Refletindo sobre o sucesso do Fragment '{fragment_name}'...")
                reflection_params = {
                    "objective": objective,
                    "plan": [f"Executed by: {fragment_name}"], # Simplified plan
                    "execution_trace": execution_trace,
                    "ctx": exec_context # Pass context
                }
                # Call success reflection skill (ensure it exists and is appropriate)
                # success_reflection = await reflect_on_success(**reflection_params)
                # ... process reflection results ...
                self.agent_logger.info("(Placeholder) Success reflection completed.")

            else: # Handle 'error' or other non-success statuses
                # --- Learn from Failure ---
                self.agent_logger.warning(f"Refletindo sobre a falha do Fragment '{fragment_name}'...")
                # Find the last error message in the trace
                last_error_msg = "Unknown error during fragment execution"
                for step in reversed(execution_trace):
                    if step.get("status") == "error" or step.get("type") == "error":
                        last_error_msg = step.get("message", last_error_msg)
                        break
                    # Check nested data if applicable (depends on trace structure)
                    if isinstance(step.get("data"), dict) and step["data"].get("status") == "error":
                         last_error_msg = step["data"].get("message", last_error_msg)
                         break

                failure_params = {
                    "objective": objective,
                    "plan": [f"Executed by: {fragment_name}"], # Simplified plan
                    "execution_trace": execution_trace,
                    "error_message": last_error_msg,
                    "ctx": exec_context
                }
                # Call failure reflection skill
                # failure_reflection = await reflect_on_failure(**failure_params)
                # ... process reflection results ...
                self.agent_logger.warning(f"(Placeholder) Failure reflection completed. Error: {last_error_msg[:100]}...")

        except Exception as reflect_err:
            self.agent_logger.error(f"Erro durante a fase de Reflexão e Aprendizado: {reflect_err}", exc_info=True)

        self.agent_logger.info("Fase de Reflexão e Aprendizado concluída.")
