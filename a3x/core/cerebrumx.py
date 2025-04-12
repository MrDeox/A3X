# core/cerebrumx.py
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple
import json # Removed comment
import os
import datetime

# Import base agent and other necessary core components
from a3x.core.agent import ReactAgent, is_introspective_query

# from core.tools import get_tool_descriptions  # <<< REMOVED import
from a3x.core.skills import get_skill_descriptions # <<< Simplified import

# from core.tool_executor import execute_tool  # <<< REMOVED import
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext  # <<< KEPT needed import

# <<< REMOVED Import for old execution logic >>>
# from a3x.core.execution_logic import execute_plan_with_reflection

# Import DB functions for memory access
from a3x.core.db_utils import retrieve_relevant_context, add_episodic_record # <<< Keep episodic record for now

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
        self.agent_logger.info(f"--- Iniciando Ciclo Cognitivo Unificado --- Objetivo: {objective[:100]}..." )

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
        final_status, final_message, execution_results = await self._execute_plan(plan, context, perception.get("processed", objective))
        self.agent_logger.info(f"Execução do plano finalizada. Status: {final_status}")

        # 5. Reflexão e Aprendizado Pós-Execução
        await self._reflect_and_learn(perception, plan, execution_results, final_status)

        # 6. Retornar Resultado Final
        self.agent_logger.info("--- Ciclo Cognitivo Unificado Concluído --- ")
        return {"status": final_status, "message": final_message, "results": execution_results} # Return consolidated result

    # --- Métodos Internos do Ciclo ---

    def _perceive(self, objective: str) -> Dict[str, Any]:
        """Processa a percepção inicial (objetivo)."""
        # TODO: Expandir lógica de percepção se necessário
        self.agent_logger.info("Processando percepção...")
        return {"processed": objective}

    async def _retrieve_context(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Recupera contexto relevante da memória semântica e episódica."""
        self.agent_logger.info("Recuperando contexto da memória...")
        query = perception.get("processed", "")
        if not query:
            self.agent_logger.warning("Query de percepção vazia, não é possível buscar contexto semântico.")
            return {"semantic_summary": "N/A", "semantic_results": [], "episodic": [], "query": query}

        # --- Busca Semântica (FAISS) --- #
        semantic_matches = []
        semantic_summary = "Nenhuma memória semântica relevante encontrada."
        try:
            # Importar localmente ou mover para o topo do arquivo se preferir
            from a3x.core.embeddings import get_embedding
            from a3x.core.semantic_memory_backend import search_index
            from a3x.core.config import PROJECT_ROOT, SEMANTIC_SEARCH_TOP_K # <<< Importar configs
            import os

            # <<< Definir caminho base do índice FAISS (Idealmente em config.py) >>>
            index_path_base = os.path.join(PROJECT_ROOT, "a3x", "memory", "indexes", "semantic_memory")

            self.agent_logger.info(f"Gerando embedding para a busca: '{query[:50]}...'" )
            query_embedding_np = get_embedding(query)

            if query_embedding_np is not None:
                query_embedding_list = query_embedding_np.tolist()
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
                    semantic_summary = "\nContexto Semântico Relevante:\n"
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
                semantic_summary = "Erro ao gerar embedding para busca."

        except ImportError as imp_err:
            self.agent_logger.error(f"Erro de importação necessário para busca semântica: {imp_err}")
            semantic_summary = "Erro: Dependência de busca semântica não encontrada."
        except Exception as e:
            self.agent_logger.exception("Erro inesperado durante a busca de contexto semântico:")
            semantic_summary = f"Erro inesperado na busca semântica: {e}"

        # --- Busca Episódica (Placeholder) --- #
        # TODO: Implementar recuperação de memória episódica relevante, se necessário.
        episodic_matches = []
        episodic_summary = "N/A"

        # --- Montar Contexto Final --- #
        final_context = {
            "semantic_summary": semantic_summary,
            "semantic_results": semantic_matches, # Inclui a lista completa de resultados
            "episodic_summary": episodic_summary,
            "episodic_results": episodic_matches,
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
                 f"Use the list_files tool for the objective: '{objective}'",
                 "Use the final_answer tool to provide the list of files.",
             ]
        else:
            # Use the planner logic (previously in planner.py, called by ReactAgent)
            # Assumes self.tools and self.llm_url are set by __init__
            tool_desc = get_skill_descriptions()
            # Context might need formatting for the planner prompt
            formatted_context = f"Consulta Original: {context.get('query')}\nMemória Semântica Relevante: {context.get('semantic')}\nMemória Episódica Relevante: {context.get('episodic')}"

            try:
                # Assuming planner.generate_plan exists and works
                from a3x.core.planner import generate_plan # Local import ok here?
                plan_to_execute = await generate_plan(
                    objective, tool_desc, self.agent_logger, self.llm_url, context_str=formatted_context
                )
                if not plan_to_execute:
                     self.agent_logger.warning("Planejador não retornou um plano. Tentando objetivo como passo único.")
                     plan_to_execute = [objective] # Fallback
            except Exception as plan_err:
                 self.agent_logger.exception("Erro durante a geração do plano:")
                 plan_to_execute = [] # Indicate planning failure

        plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
        self.agent_logger.info(f"Plano Gerado:\n{plan_str}")
        return plan_to_execute


    # Merged from execution_logic.execute_plan_with_reflection
    async def _execute_plan(self, plan: List[str], context: Dict[str, Any], original_objective: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Executa um plano passo-a-passo usando o ciclo ReAct interno (_perform_react_iteration).
        Realiza reflexão sobre falhas e chama a skill de aprendizado de falhas.
        Retorna (status_final, mensagem_final, lista_resultados_passos).
        """
        self.agent_logger.info("--- Iniciando Execução do Plano --- ")
        execution_results = []
        last_successful_thought = ""
        last_attempted_action = ""
        last_observation_content = None
        final_status = "unknown"
        final_message = "Execução não iniciada ou falha indeterminada."

        for i, step in enumerate(plan):
            log_prefix = f"[Exec Step {i+1}/{len(plan)}]"
            self.agent_logger.info(f"{log_prefix} Processando Passo: {step[:60]}...")

            # --- Simulation (Optional - Add back if needed) ---
            # simulated_outcome = await self._simulate_step(step, context) # _simulate_step needs implementation
            # self.agent_logger.debug(f"{log_prefix} Simulação: {simulated_outcome}")
            # yield {"type": "simulation", "step_index": i, "content": simulated_outcome}

            # --- Pre-Reflection (Optional - Add back if needed) ---
            # reflection_decision = await self._reflect_step(step, simulated_outcome, context)
            # if reflection_decision['decision'] == 'skip': continue
            # if reflection_decision['decision'] == 'modify':
            #    yield {"type": "modification_trigger", ...} # Signal replanning if needed
            #    # Handle replanning logic here or signal back to run()
            #    continue

            # --- Execute Step using ReAct Iteration ---
            self.agent_logger.info(f"{log_prefix} Executando passo via ReAct.")
            step_result = None
            error_occurred = False
            try:
                # _perform_react_iteration is inherited from ReactAgent
                async for react_event in self._perform_react_iteration(step, log_prefix):
                    # Log intermediate events if desired
                    self.agent_logger.debug(f"{log_prefix} ReAct Event: {react_event.get('type')}")

                    # Capture last thought/action/observation for potential failure reflection
                    if react_event.get("type") == "thought":
                         last_successful_thought = react_event.get("content", last_successful_thought)
                    elif react_event.get("type") == "action":
                         last_attempted_action = react_event.get("tool_name", last_attempted_action)
                    elif react_event.get("type") == "observation":
                         last_observation_content = react_event.get("content", last_observation_content)

                    if react_event.get("type") == "step_final_answer":
                        step_result = {
                            "status": "success",
                            "message": react_event.get("content")
                        }
                        final_status = "intermediate_success" # Step succeeded, plan continues
                        final_message = step_result["message"]
                        self.agent_logger.info(f"{log_prefix} Passo concluído com sucesso.")
                        break # Step finished successfully
                    elif react_event.get("type") == "error":
                        step_result = {
                            "status": "error",
                            "message": react_event.get("content")
                        }
                        last_observation_content = step_result['message'] # Error message is the observation
                        error_occurred = True
                        final_status = "failed" # Step failed
                        final_message = f"Falha no passo {i+1}: {step_result['message']}"
                        self.agent_logger.error(f"{log_prefix} Passo falhou: {step_result['message']}")
                        break # Step errored out

                if step_result is None: # Should not happen if loop finishes properly
                     step_result = {"status": "unknown", "message": "Iteração ReAct finalizada sem resposta ou erro."}
                     last_observation_content = step_result['message']
                     error_occurred = True
                     final_status = "error" # Treat unknown state as error
                     final_message = f"Erro indeterminado no passo {i+1}."
                     self.agent_logger.error(f"{log_prefix} Estado desconhecido ao final da iteração ReAct.")


            except Exception as e:
                 self.agent_logger.exception(f"{log_prefix} Exceção não tratada durante execução do passo '{step}':")
                 step_result = {
                     "status": "error",
                     "message": f"Exceção não tratada: {e}"
                 }
                 last_observation_content = step_result['message']
                 error_occurred = True
                 final_status = "error" # Treat exception as error
                 final_message = f"Erro crítico no passo {i+1}: {e}"

            # --- Failure Reflection & Learning Integration ---
            if error_occurred:
                self.agent_logger.error(f"{log_prefix} Passo falhou. Iniciando reflexão/aprendizado sobre falha.")
                # Collect context for reflection
                failure_context = {
                    "objective": original_objective,
                    "plan": plan, # Whole plan
                    "failed_step_index": i,
                    "failed_step": step,
                    "last_thought": last_successful_thought or "N/A",
                    "last_action": last_attempted_action or "N/A",
                    "last_observation": str(last_observation_content) or "N/A" # Ensure it's a string
                }
                exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root)
                setattr(exec_context, 'llm_url', self.llm_url) # Pass LLM URL if needed by skills

                try:
                    # Call reflect_on_failure skill
                    reflection_result = await execute_tool(
                        tool_name="reflect_on_failure",
                        action_input=failure_context,
                        tools_dict=self.tools, # Use agent's tools
                        context=exec_context
                    )
                    failure_analysis = "Análise da falha não disponível."
                    if reflection_result.get("status") == "success":
                        failure_analysis = reflection_result.get("data", {}).get("explanation", failure_analysis)
                        self.agent_logger.info(f"{log_prefix} Reflexão sobre falha gerada: {failure_analysis[:100]}...")

                        # Call learn_from_failure_log skill
                        learn_input = {
                            "objective": original_objective,
                            "error_message": step_result.get('message', 'Erro Desconhecido'),
                            "failure_analysis": failure_analysis
                        }
                        try:
                            learn_result = await execute_tool(
                                tool_name="learn_from_failure_log",
                                action_input=learn_input,
                                tools_dict=self.tools,
                                context=exec_context # Re-use context
                            )
                            if learn_result.get("status") == "success":
                                heuristic = learn_result.get("data", {}).get("heuristic", "N/A")
                                self.agent_logger.info(f"{log_prefix} Heurística de falha registrada: {heuristic[:100]}...")
                            else:
                                learn_error = learn_result.get("data", {}).get("message", "Erro desconhecido no aprendizado")
                                self.agent_logger.error(f"{log_prefix} Falha ao chamar skill learn_from_failure_log: {learn_error}")
                        except Exception as learn_err:
                            self.agent_logger.exception(f"{log_prefix} Exceção ao chamar skill learn_from_failure_log:")
                    else:
                        reflect_error = reflection_result.get("data", {}).get("message", "Erro desconhecido na reflexão")
                        self.agent_logger.error(f"{log_prefix} Falha ao chamar skill reflect_on_failure: {reflect_error}")
                        failure_analysis = f"Erro na reflexão: {reflect_error}" # Update final message

                except Exception as reflect_err:
                     self.agent_logger.exception(f"{log_prefix} Exceção ao chamar skill reflect_on_failure:")
                     failure_analysis = f"Erro crítico na chamada de reflexão: {reflect_err}" # Update final message

                # Update final message with failure analysis and stop execution
                final_message = f"Falha no passo {i+1}: {step_result.get('message', 'Erro Desconhecido')}\n**Análise:** {failure_analysis}"
                final_status = "failed" # Ensure status reflects failure
                execution_results.append(step_result) # Add failed step result
                self.agent_logger.warning(f"{log_prefix} Encerrando execução do plano devido à falha no passo.")
                return final_status, final_message, execution_results # Stop plan execution

            # Append successful step result
            execution_results.append(step_result)

        # If loop completes without critical errors/failures stopping it
        if final_status == "intermediate_success": # Check if last step was successful
             final_status = "completed"
             final_message = "Plano executado com sucesso." # Overall success message
             self.agent_logger.info("--- Execução do Plano Concluída com Sucesso ---")
        elif final_status == "unknown": # Should not happen ideally
             final_status = "error"
             final_message = "Execução do plano terminou em estado desconhecido."

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
        exec_context = _ToolExecutionContext(logger=self.agent_logger, workspace_root=self.workspace_root)
        # Não precisa setar llm_url aqui, a skill recebe como arg e passa internamente se necessário

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
