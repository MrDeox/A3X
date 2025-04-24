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
from a3x.core.agent import is_introspective_query
from a3x.core.skills import get_skill_descriptions, get_skill_registry
from a3x.core.tool_executor import ToolExecutor

# Centralized config and memory manager
from a3x.core import config as a3x_config
from a3x.core.config import HEURISTIC_LOG_FILE, MAX_REACT_ITERATIONS # <<< Import MAX_REACT_ITERATIONS
from a3x.core.memory.memory_manager import MemoryManager

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
cerebrumx_logger = logging.getLogger(__name__) # <<< Rename logger? agent_logger already exists in 

# <<< ADDED Import >>>
from .agent import _is_simple_list_files_task

# Helper to create context for direct execution calls
# _ToolExecutionContext = namedtuple("_ToolExecutionContext", ["logger", "workspace_root", "llm_url", "tools_dict"])

# <<< ADDED Imports for Path and Registries/Manager >>>
from a3x.core.tool_registry import ToolRegistry
# from a3x.fragments.registry import FragmentRegistry # Duplicate import, remove?
from a3x.core.memory.memory_manager import MemoryManager
# <<< END ADDED Imports >>>

from a3x.core.auto_evaluation import auto_evaluate_task
from a3x.core.db_utils import add_episodic_record # <<< EXISTING - Check if this needs to be moved higher or duplicated

# <<< Import config and interface >>>
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
from a3x.core.config import LLAMA_SERVER_URL
# <<< REMOVED unused Config import >>>
# from a3x.core.config import Config
from a3x.core.context import SharedTaskContext

class CerebrumXAgent:
    """
    Agente Autônomo Adaptável Experimental com ciclo cognitivo unificado.
    Incorpora percepção, planejamento, execução ReAct, reflexão e aprendizado em um único fluxo.
    AGORA DELEGA A EXECUÇÃO PRINCIPAL AO TASKORCHESTRATOR.
    """

    def __init__(
        self,
        agent_id: str, # <<< ADDED
        system_prompt: str,
        tool_registry: ToolRegistry, # <<< ADDED
        fragment_registry: FragmentRegistry, # <<< ADDED
        memory_manager: MemoryManager, # <<< ADDED
        workspace_root: Path, # <<< ADDED
        llm_url: Optional[str] = None,
        # tools_dict is now passed via tool_registry
        # tools_dict: Optional[Dict[str, Any]]] = None, # Removed, use tool_registry
        exception_policy=None,
        agent_config: Optional[Dict] = None,
        max_iterations: int = MAX_REACT_ITERATIONS, # <<< ADDED max_iterations
    ):
        """Inicializa o Agente CerebrumX."""
        # Initialize  first
        llm_interface_instance = None
        if llm_url: # If URL is explicitly passed to CerebrumX
            llm_interface_instance = LLMInterface(llm_url=llm_url)
        else: # Create default using correct priority: config -> default
            effective_llm_url = LLAMA_SERVER_URL or DEFAULT_LLM_URL 
            llm_interface_instance = LLMInterface(llm_url=effective_llm_url)

        self.agent_id = agent_id
        self.agent_logger = cerebrumx_logger
        self.system_prompt = system_prompt
        self.config = agent_config or {}
        self.memory_manager = memory_manager
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry
        self.workspace_root = workspace_root
        self.llm_url = llm_url or LLAMA_SERVER_URL or DEFAULT_LLM_URL
        self._history = []
        self._memory: Dict[str, Any] = {}
        self.orchestrator = None
        # Initialize LLM interface
        self.llm_interface = LLMInterface(llm_url=self.llm_url)
        self.orchestrator = TaskOrchestrator(
            fragment_registry=self.fragment_registry,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            llm_interface=self.llm_interface,
            workspace_root=self.workspace_root,
            agent_logger=self.agent_logger,
            config=self.config,
        )
        # Load agent state if exists
        from a3x.core.db_utils import load_agent_state
        loaded_state = load_agent_state(agent_id=self.agent_id)
        if loaded_state:
            self._history = loaded_state.get("history", [])
            self._memory = loaded_state.get("memory", {})
            self.agent_logger.info(
                f"[CerebrumX INIT] Estado do agente carregado para ID '{self.agent_id}'. Memória: {list(self._memory.keys())}"
            )
        self.agent_logger.info(
            f"[CerebrumX INIT] Agent initialized. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}"
        )
        # ExceptionPolicy configurável
        if exception_policy is None:
            from a3x.core.exception_policy import ExceptionPolicy
            self.exception_policy = ExceptionPolicy()
        else:
            self.exception_policy = exception_policy


    def add_history_entry(self, role: str, content: str):
        """Adiciona uma entrada de dicionário {"role": ..., "content": ...} ao histórico."""
        role_lower = role.lower()
        if role_lower not in ["user", "assistant", "system", "tool", "human"]:
            self.agent_logger.warning(f"[History] Received unexpected role: '{role}'. Using 'assistant'.")
            role_lower = "assistant"
        if role_lower == "human":
            role_lower = "user"
        history_entry = {"role": role_lower, "content": content}
        self._history.append(history_entry)
        self.agent_logger.debug(f"[History] Added entry: {history_entry}")

    def get_history(self):
        """Retorna o histórico interno para possível inspeção ou passagem para outros componentes."""
        return self._history

    def _save_state(self):
        from a3x.core.db_utils import save_agent_state
        state = {"history": self._history, "memory": self._memory}
        save_agent_state(agent_id=self.agent_id, state=state)
        self.agent_logger.info("[CerebrumXAgent] Agent state saved for ID '{}'.".format(self.agent_id))

    # --- Novo Ciclo Cognitivo Unificado (Delegado ao Orchestrator) ---
    async def run(self, objective: str, max_steps: Optional[int] = None) -> Dict[str, Any]: # Pass max_steps
        """
        Inicia a execução de uma tarefa delegando ao TaskOrchestrator.
        Retorna o resultado final fornecido pelo orquestrador.
        """
        start_time = time.time()
        self.agent_logger.info(f"--- Iniciando Tarefa (via CerebrumX) --- Objetivo: {objective[:100]}..." )
        add_episodic_record(f"Iniciando tarefa: {objective}", "task_start", "iniciada", {"objective": objective})

        # 1. Percepção (Simplificado - Adiciona ao histórico do agente base)
        perception = self._perceive(objective)
        self.add_history_entry(role="user", content=perception["processed"]) # Usa método local
        self.agent_logger.info(f"Percepção processada e adicionada ao histórico: {perception}")

        # 2. Recuperação de Contexto (Opcional - Pode ser feito pelo Orchestrator/Fragment se necessário)
        # context = await self._retrieve_context(perception)
        # self.agent_logger.info(f"Contexto recuperado: {str(context)[:100]}...")

        # --- REMOVED Fragment Selection --- 
        # selected_fragment = await self._select_fragment(perception, context)

        # 3. Delegação para o TaskOrchestrator
        final_result = {}
        try:
            self.agent_logger.info(f"Delegando execução para TaskOrchestrator...")
            # self.orchestrator foi inicializado no __init__ do 
            final_result = await self.orchestrator.orchestrate(objective, max_steps=max_steps)
            self.agent_logger.info(f"TaskOrchestrator finalizou. Status: {final_result.get('status')}")

        except Exception as e:
            self.exception_policy.handle(e, context="Erro crítico durante a orquestração")
            self.agent_logger.exception("Erro crítico capturado pelo CerebrumX durante a orquestração:")
            final_result = {
                "status": "error",
                "message": f"Erro crítico na orquestração: {e}",
                "fragment_used": "N/A",
                "results": []
            }

        # --- REMOVED Direct Fragment Execution --- 
        # if selected_fragment: ...
        # else: ...

        # 4. Reflexão e Aprendizado (COMENTADO - A ser movido para TaskOrchestrator ou chamado por ele)
        # final_status = final_result.get("status", "error")
        # execution_trace = final_result.get("history", []) # Use history from orchestrator result
        # selected_fragment_name = final_result.get("fragment_used", "N/A") # Requires orchestrator to maybe return this
        # await self._reflect_and_learn(perception, selected_fragment_name, execution_trace, final_status)

        # 5. Autoavaliação Cognitiva (COMENTADO - Depende de como o histórico é estruturado no orchestrator)
        end_time = time.time()
        # from a3x.core.auto_evaluation import auto_evaluate_task
        # auto_evaluate_task(...)

        # 6. Salvar estado final do agente (incluindo histórico acumulado)
        # O histórico é gerenciado pela classe base  (self._history)
        # O orchestrator pode adicionar ao contexto, mas o histórico final deve estar em self._history?
        # Precisamos garantir que o histórico do orchestrator seja integrado em self._history
        # TODO: Revisar como o histórico é passado entre orchestrator e agente.
        # Por enquanto, o orchestrator retorna o histórico, mas não o integramos aqui.
        
        # Tenta adicionar a resposta final ao histórico do agente, se houver
        final_content = final_result.get("final_answer") or final_result.get("message")
        if final_content:
            self.add_history_entry(role="assistant", content=str(final_content))

        self._save_state() # Salva estado incluindo histórico

        # 7. Retornar Resultado Final do Orchestrator
        final_status_str = str(final_result.get('status', 'unknown')) # Ensure outcome is a string
        self.agent_logger.info(f"--- Tarefa Finalizada (via CerebrumX) --- Status: {final_status_str}")
        add_episodic_record(
            f"Tarefa finalizada: {objective}",
            "task_end",
            final_result.get("status", "unknown"), # Pass status string as outcome
            {"status": final_result.get("status"), "final_message": final_content} # Pass dict as metadata
        )
        return final_result

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
                            semantic_summary += f"- [Dist: {distance:.3f}] {content[:150]}...\n"

                        # Adiciona registro episódico da consulta semântica
                        add_episodic_record(
                            f"Consulta à memória semântica para: '{query[:50]}...'",
                            "semantic_memory_query",
                            {"query": query, "results_summary": [r.get("metadata",{}).get("content","")[:50]+"..." for r in search_results]},
                            metadata={"num_results": len(search_results)}
                        )
                    else:
                         self.agent_logger.info("Nenhum resultado encontrado na busca semântica.")
            except ImportError:
                self.agent_logger.warning("Módulo de embeddings ou FAISS não encontrado. Pulando busca semântica.")
                semantic_summary = "Busca semântica desabilitada (dependência ausente)."
            except Exception as e:
                self.exception_policy.handle(e, context="Erro durante busca semântica")
                semantic_summary = f"Erro ao consultar memória semântica: {e}"

        # --- Memória Episódica (Recente) --- #
        episodic_matches = []
        episodic_summary = "Nenhuma memória episódica recente encontrada."
        try:
            from a3x.core.db_utils import get_recent_episodes
            recent_episodes = get_recent_episodes(limit=10) # Pega os 10 últimos episódios
            if recent_episodes:
                self.agent_logger.info(f"Encontrados {len(recent_episodes)} episódios recentes.")
                episodic_matches = recent_episodes
                episodic_summary = "\nMemória Episódica Recente:\n"
                for ep in recent_episodes:
                    timestamp_str = ep['timestamp'].split('.')[0] # Remove microseconds
                    episodic_summary += f"- [{timestamp_str}] {ep['event_type']}: {ep['description'][:100]}...\n"
            else:
                 self.agent_logger.info("Nenhum episódio recente encontrado.")

        except Exception as e:
            self.exception_policy.handle(e, context="Erro durante busca episódica")
            episodic_summary = f"Erro ao consultar memória episódica: {e}"

        # --- Combina Contextos --- #
        combined_summary = semantic_summary + "\n" + episodic_summary
        return {
            "combined_summary": combined_summary,
            "semantic_results": semantic_matches,
            "episodic_results": episodic_matches,
            "query": query
        }

    # --- REMOVED _select_fragment --- 
    # async def _select_fragment(self, perception: Dict[str, Any], context: Dict[str, Any]) -> Optional[BaseFragment]:
    #    ...

    # --- REMOVED _execute_fragment --- 
    # async def _execute_fragment(self, fragment: BaseFragment, objective: str, context: Optional[Dict]) -> Tuple[Dict, List[Dict]]:
    #    ...

    # --- COMENTADO _reflect_and_learn (a ser movido/chamado pelo Orchestrator) ---
    async def _reflect_and_learn(self, perception: Dict[str, Any], fragment_name: str, execution_trace: List[Dict[str, Any]], final_status: str):
        """Aciona mecanismos de reflexão e aprendizado com base no resultado da execução."""
        self.agent_logger.info(f"Iniciando Reflexão e Aprendizado para Fragment '{fragment_name}' (Status: {final_status})...")

        # TODO: Passar mais contexto para as skills de reflexão (objetivo, contexto inicial, etc.)
        log_entry = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "objective": perception.get("processed", "N/A"),
            "fragment_used": fragment_name,
            "final_status": final_status,
            "execution_trace": execution_trace # Lista de dicionários (passos ReAct ou resultado do fragment.execute)
        }

        if final_status == "success":
            self.agent_logger.info(f"Refletindo sobre o sucesso do Fragment '{fragment_name}'...")
            try:
                await reflect_on_success(context={}, log_entry=log_entry) # Contexto pode ser necessário
            except Exception as e:
                 self.agent_logger.error(f"Erro durante reflect_on_success: {e}", exc_info=True)

        else: # error, max_iterations, timeout, etc.
            self.agent_logger.warning(f"Refletindo sobre a falha do Fragment '{fragment_name}'...")
            try:
                 # Chama reflect_on_failure primeiro
                 await reflect_on_failure(context={}, log_entry=log_entry)
                 # Depois tenta aprender com o log de falha
                 await learn_from_failure_log(context={}, log_entry=log_entry)
            except Exception as e:
                 self.agent_logger.error(f"Erro durante reflect_on_failure ou learn_from_failure_log: {e}", exc_info=True)

        # TODO: Adicionar chamadas periódicas ou condicionais para generalização/consolidação?
        # Ex: await consolidate_heuristics(context={})
        # Ex: await generalize_heuristics(context={})

        self.agent_logger.info("Fase de Reflexão e Aprendizado concluída.")
