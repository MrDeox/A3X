import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import necessary components for registration
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.fragments.registry import fragment
from a3x.core.context import SharedTaskContext
from a3x.core.constants import STATUS_SUCCESS, STATUS_ERROR, REASON_DELEGATION_FAILED

logger = logging.getLogger(__name__)

# Decorator for automatic registration
@fragment(
    name="a3net_orchestrator",
    description="Decide o próximo fragmento a ser executado usando lógica simbólica.",
    category="Management", # Orquestração é uma tarefa de gerenciamento
    skills=[], # Não executa skills diretamente
    managed_skills=[] # Não gerencia skills diretamente
)
class OrchestratorFragment(BaseFragment): # Use BaseFragment
    """
    Fragmento A³Net responsável por decidir qual fragmento deve
    executar a próxima sub-tarefa, baseado no objetivo e contexto.
    Utiliza lógica simbólica/heurística.
    """
    def __init__(self, fragment_def: FragmentDef): # <<< ADD fragment_def parameter
        """Inicializa o OrchestratorFragment."""
        # TODO: Define specific skills/metadata if needed, or rely on BaseFragment defaults
        # fragment_def = FragmentDef(name="OrchestratorFragment", fragment_class=OrchestratorFragment, description=self.get_purpose(), category="Orchestration")
        super().__init__(fragment_def=fragment_def) # <<< Pass fragment_def to super
        # Initialize neural network components if this fragment uses them
        # Example: self.decision_network = DecisionNetwork(...)
        self._logger.info(f"OrchestratorFragment '{self.get_name()}' initialized.") # Use get_name() from BaseFragment

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        return "Decides the next fragment or skill to delegate a task to based on objective, history, and available components."

    async def execute(self, *, # Use keyword-only arguments for clarity
                objective: str,
                history: List[Dict[str, str]],
                available_fragments: List[str],
                available_skills: List[str],
                task_context: Dict[str, Any]
                ) -> Dict[str, Any]:
        """
        Decide o próximo passo de forma simbólica.

        Args:
            objective: O objetivo geral da tarefa.
            history: O histórico da orquestração até o momento.
            available_fragments: Lista de IDs dos fragmentos disponíveis.
            available_skills: Lista de IDs das skills disponíveis.
            task_context: Dicionário com o contexto compartilhado da tarefa.

        Returns:
            Um dicionário com 'component_name' e 'sub_task'.
        """
        fragment_id = self.get_name() # Get assigned name
        self._logger.info(f"[{fragment_id}] Executing decision logic for objective: '{objective[:100]}...'")
        self._logger.debug(f"[{fragment_id}] Available Fragments: {available_fragments}")
        self._logger.debug(f"[{fragment_id}] Available Skills: {available_skills}")
        self._logger.debug(f"[{fragment_id}] History length: {len(history)}")
        self._logger.debug(f"[{fragment_id}] Task Context Keys: {list(task_context.keys())}")

        # Lógica heurística simples
        objective_lower = objective.lower()
        component_name = "default_fragment" # Fallback
        sub_task = "analisar objetivo e contexto" # Fallback task

        # TODO: Use available_fragments list to validate choices
        planner_id = "PlannerFragment" # Standard ID?
        executor_id = "ToolExecutorFragment" # Standard ID?
        # default_id = "CodeExecutionFragment" # Example default

        if "planejar" in objective_lower or "plan" in objective_lower:
            component_name = planner_id
            sub_task = "Gerar um plano detalhado para alcançar o objetivo."
            self._logger.info(f"[{fragment_id}] Heuristic: Objective mentions planning. Delegating to {component_name}.")

        elif "executar" in objective_lower or "execute" in objective_lower or "run" in objective_lower:
            # Se existe um plano, o ToolExecutor provavelmente deve ser chamado.
            # Se não, talvez o Planner devesse ter sido chamado antes?
            # Por enquanto, vamos delegar ao ToolExecutor se a palavra estiver presente.
            component_name = executor_id
            sub_task = "Executar a próxima etapa da tarefa ou plano."
            self._logger.info(f"[{fragment_id}] Heuristic: Objective mentions execution. Delegating to {component_name}.")

        # Add more heuristics based on history, context, available fragments/skills later
        # Example: Check last step in history for errors -> delegate to DebuggerFragment

        else:
            self._logger.info(f"[{fragment_id}] Heuristic: No specific keyword matched. Using fallback delegation to {component_name}.")
            # Maybe delegate to Planner if no other rule matches?
            # component_name = planner_id
            # sub_task = "Analisar o objetivo e criar um plano inicial."


        decision = {
            "component_name": component_name,
            "sub_task": sub_task,
            "status": "success" # Indicate successful decision making
        }
        self._logger.info(f"[{fragment_id}] Decision made: {decision}")
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