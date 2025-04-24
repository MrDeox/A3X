import torch.nn as nn
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re # For task categorization heuristic

# Import necessary components for registration
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.fragments.registry import fragment
from a3x.core.context import SharedTaskContext, FragmentContext
from a3x.core.constants import STATUS_SUCCESS, STATUS_ERROR, REASON_DELEGATION_FAILED
# Assume MemoryManager might be needed
try:
    from a3x.core.memory.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None # Handle gracefully if import fails
    logging.getLogger(__name__).warning("Could not import MemoryManager in a3net_orchestrator")

logger = logging.getLogger(__name__)

# Constants for adaptive logic
HISTORY_ANALYSIS_EPISODE_COUNT = 50
MIN_EXECUTIONS_FOR_STATS = 3
DEFAULT_FALLBACK_FRAGMENT = "PlannerFragment" # Safer fallback

# Decorator for automatic registration
@fragment(
    name="a3net_orchestrator",
    description="Decide o próximo fragmento a ser executado usando lógica adaptativa baseada em histórico e sugestões.",
    category="Management",
    skills=[],
    managed_skills=[],
    capabilities=["task_delegation", "orchestration_decision"]
)
class OrchestratorFragment(BaseFragment):
    """
    Fragmento A³Net responsável por decidir qual fragmento deve
    executar a próxima sub-tarefa, baseado no objetivo, contexto, histórico,
    e sugestões do MetaLearner.
    Utiliza lógica adaptativa e heurísticas de fallback.
    """
    def __init__(self, ctx: FragmentContext):
        """Inicializa o OrchestratorFragment."""
        super().__init__(ctx=ctx)
        # Get MemoryManager from context
        if MemoryManager and hasattr(ctx, 'memory_manager') and isinstance(ctx.memory_manager, MemoryManager):
            self.memory_manager = ctx.memory_manager
        else:
            self._logger.error("MemoryManager not available in context for OrchestratorFragment. Adaptive logic disabled.")
            self.memory_manager = None
            
        # Placeholder for potential future NN components
        # self.decision_network = ... 
        self._logger.info(f"OrchestratorFragment '{self.get_name()}' initialized. Adaptive: {self.memory_manager is not None}")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        return "Decides the next fragment or skill to delegate a task to based on objective, history, available components, and meta-learning suggestions."

    def _categorize_objective(self, objective: str) -> str:
        """Simple heuristic to categorize objectives for history analysis."""
        objective_lower = objective.lower()
        # Prioritize more specific keywords
        if re.search(r'(código|codigo|code|arquivo|file).*(gerar|criar|escrever|write|generate|create)', objective_lower): return "code_generation"
        if re.search(r'(código|codigo|code|arquivo|file).*(analisar|revisar|debug|corrigir|refatorar|analyze|review|fix|refactor)', objective_lower): return "code_analysis"
        if re.search(r'(mutar|evoluir|melhorar|adaptar|mutate|evolve|improve|adapt)', objective_lower): return "evolution"
        if re.search(r'(avaliar|desempenho|performance|evaluate)', objective_lower): return "evaluation"
        if re.search(r'(planejar|plano|plan)', objective_lower): return "planning"
        if re.search(r'(executar|rodar|execute|run)', objective_lower): return "execution"
        if re.search(r'(sumarizar|resumir|summarize)', objective_lower): return "summarization"
        if re.search(r'(traduzir|translate)', objective_lower): return "translation"
        # Generic fallback category
        return "general"

    async def _get_historical_performance(self, task_category: str, available_fragments: List[str]) -> Optional[str]:
        """Analyzes recent history to find the best performing fragment for a task category."""
        if not self.memory_manager:
            return None
            
        self._logger.info(f"Analyzing historical performance for task category: '{task_category}'...")
        try:
            episodes = await self.memory_manager.get_recent_episodes(limit=HISTORY_ANALYSIS_EPISODE_COUNT)
            if not episodes:
                self._logger.info("No recent episodes found for historical analysis.")
                return None

            # Calculate success rates per fragment for the target category
            fragment_perf = defaultdict(lambda: {"success": 0, "total": 0})
            
            for episode in episodes:
                try:
                    # Extract fragment name and objective/context
                    metadata_str = episode.get('metadata', '{}')
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else (metadata_str if isinstance(metadata_str, dict) else {})
                    fragment_name = metadata.get("fragment_name") or episode.get('action')
                    episode_objective = metadata.get("objective") or episode.get('context') # Check both
                    
                    if not fragment_name or not episode_objective or fragment_name not in available_fragments:
                        continue # Skip if data missing or fragment unavailable
                        
                    # Check if episode matches target category
                    episode_category = self._categorize_objective(str(episode_objective))
                    if episode_category == task_category:
                        stats = fragment_perf[fragment_name]
                        stats["total"] += 1
                        if episode.get('outcome', 'unknown').lower() == 'success':
                            stats["success"] += 1
                except Exception as e:
                    self._logger.warning(f"Error processing episode ID {episode.get('id')} during historical analysis: {e}")
                    
            # Find the best fragment based on success rate (if enough data)
            best_fragment = None
            best_rate = -1.0
            for name, stats in fragment_perf.items():
                if stats["total"] >= MIN_EXECUTIONS_FOR_STATS:
                    rate = stats["success"] / stats["total"]
                    if rate > best_rate:
                        best_rate = rate
                        best_fragment = name
                        
            if best_fragment:
                self._logger.info(f"Historical analysis suggests '{best_fragment}' (rate: {best_rate:.2f}) for category '{task_category}'.")
                return best_fragment
            else:
                self._logger.info(f"No fragment found with sufficient historical data for category '{task_category}'.")
                return None
                 
        except Exception as e:
            self._logger.error(f"Error during historical performance analysis: {e}", exc_info=True)
            return None

    async def execute(self, *, 
                objective: str,
                history: List[Dict[str, str]],
                available_fragments: List[str],
                available_skills: List[str],
                task_context: Dict[str, Any]
                ) -> Dict[str, Any]:
        fragment_id = self.get_name()
        self._logger.info(f"[{fragment_id}] Executing ADAPTIVE decision logic for objective: '{objective[:100]}...'")
        self._logger.debug(f"[{fragment_id}] Available Fragments: {available_fragments}")
        self._logger.debug(f"[{fragment_id}] Available Skills: {available_skills}")
        self._logger.debug(f"[{fragment_id}] History length: {len(history)}")
        self._logger.debug(f"[{fragment_id}] Task Context Keys: {list(task_context.keys())}")

        component_name = None
        sub_task = "Process objective: " + objective # Default subtask
        decision_reason = "No specific strategy applied."

        # Ensure available_fragments is a set for faster lookups
        available_fragments_set = set(available_fragments)

        # --- Decision Logic --- 
        task_category = self._categorize_objective(objective)
        self._logger.info(f"[{fragment_id}] Objective categorized as: '{task_category}'")

        # 1. Check Meta-Learner Suggestions
        meta_suggestions = task_context.get('orchestration_suggestions', [])
        if isinstance(meta_suggestions, list):
            for suggestion in meta_suggestions:
                if isinstance(suggestion, dict) and \
                   suggestion.get('task_category') == task_category and \
                   suggestion.get('preferred_fragment') in available_fragments_set:
                    component_name = suggestion['preferred_fragment']
                    decision_reason = f"Meta-Learner suggestion for category '{task_category}'"
                    self._logger.info(f"[{fragment_id}] Applying Meta-Learner suggestion: Use '{component_name}'.")
                    break # Use the first matching suggestion

        # 2. Check Historical Performance (if no Meta-Learner suggestion applied)
        if not component_name:
            historical_best = await self._get_historical_performance(task_category, available_fragments)
            if historical_best: # historical_best is already validated against available_fragments
                component_name = historical_best
                decision_reason = f"Historical performance for category '{task_category}'"
                self._logger.info(f"[{fragment_id}] Using historically best fragment: '{component_name}'.")

        # 3. Fallback Heuristics (if no suggestion or history)
        if not component_name:
            decision_reason = "Fallback heuristic"
            objective_lower = objective.lower()
            planner_id = "PlannerFragment"
            executor_id = "tool_executor" # This likely refers to a skill executor, not a fragment
            evolution_planner_id = "evolution_planner"

            if "planejar" in objective_lower or "plan" in objective_lower:
                component_name = planner_id if planner_id in available_fragments_set else DEFAULT_FALLBACK_FRAGMENT
                sub_task = "Gerar um plano detalhado para alcançar o objetivo."
            elif "executar" in objective_lower or "execute" in objective_lower or "run" in objective_lower:
                if task_context.get("current_plan") and isinstance(task_context.get("current_plan"), list) and len(task_context["current_plan"]) > 0:
                    # Plan exists: Needs an executor. tool_executor isn't a fragment.
                    # Which fragment *uses* tool_executor? Maybe a dedicated 'PlanExecutorFragment'?
                    # Or perhaps the logic resides directly in the main TaskOrchestrator?
                    # For now, let's assume a hypothetical executor fragment exists or use Planner as fallback.
                    plan_executor_id = "PlanExecutorFragment" # Hypothetical
                    component_name = plan_executor_id if plan_executor_id in available_fragments_set else (executor_id if executor_id in available_fragments_set else planner_id) # Trying executor_id just in case
                    
                    # Generate specific sub_task from plan
                    next_step_index = task_context.get("current_step_index", 0)
                    current_plan = task_context["current_plan"]
                    if isinstance(next_step_index, int) and 0 <= next_step_index < len(current_plan):
                        plan_step = current_plan[next_step_index]
                        step_desc = plan_step.get("description") if isinstance(plan_step, dict) else str(plan_step)
                        sub_task = f"Executar etapa {next_step_index + 1} do plano: {step_desc}"
                    else:
                        sub_task = "Executar próxima etapa do plano (índice inválido?)."
                else:
                    # No plan: Delegate to a planner
                    chosen_planner = evolution_planner_id if evolution_planner_id in available_fragments_set and "evolucao" in objective_lower else planner_id
                    component_name = chosen_planner if chosen_planner in available_fragments_set else DEFAULT_FALLBACK_FRAGMENT
                    sub_task = f"Gerar um plano para o objetivo: '{objective}'"
            else:
                # Default fallback if no keywords match
                component_name = DEFAULT_FALLBACK_FRAGMENT if DEFAULT_FALLBACK_FRAGMENT in available_fragments_set else (planner_id if planner_id in available_fragments_set else None)
                sub_task = f"Analisar objetivo e contexto: '{objective}'"
                decision_reason = "Default fallback strategy"
            
            self._logger.info(f"[{fragment_id}] Applied fallback heuristic ({objective_lower[:20]}...): Delegate to '{component_name}'.")

        # Final Validation and Default
        if not component_name or component_name not in available_fragments_set:
            self._logger.warning(f"[{fragment_id}] Chosen component '{component_name}' is invalid or unavailable. Using default fallback '{DEFAULT_FALLBACK_FRAGMENT}'.")
            component_name = DEFAULT_FALLBACK_FRAGMENT
            sub_task = f"Analisar objetivo e contexto (fallback): '{objective}'"
            decision_reason = "Final fallback (component unavailable)"
            # Ensure the default fallback is actually available
            if DEFAULT_FALLBACK_FRAGMENT not in available_fragments_set:
                self._logger.error(f"[{fragment_id}] CRITICAL: Default fallback fragment '{DEFAULT_FALLBACK_FRAGMENT}' is not in available_fragments! Cannot make decision.")
                return {"status": STATUS_ERROR, "reason": REASON_DELEGATION_FAILED, "message": "Fallback fragment indisponível."}

        decision = {
            "component_name": component_name,
            "sub_task": sub_task,
            "status": STATUS_SUCCESS,
            "decision_reason": decision_reason # Add reason for transparency
        }
        self._logger.info(f"[{fragment_id}] Adaptive decision made: Delegate to '{component_name}'. Reason: {decision_reason}")
        return decision

# Example of how it might be registered (actual registration happens elsewhere)
# from a3x.fragments.registry import FragmentDefinition, register_fragment
# register_fragment(FragmentDefinition(
#     fragment_id="a3net_orchestrator",
#     description="Decide o próximo fragmento a ser executado.",
#     fragment_class=OrchestratorFragment,
#     category="Management", # Example category
#     managed_skills=[] # This fragment doesn't directly manage skills
# )) 