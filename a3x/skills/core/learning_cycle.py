# a3x/skills/core/learning_cycle.py
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from pathlib import Path
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE
import json
import datetime
from a3x.core.skills import skill
from a3x.core.context import SharedTaskContext
from a3x.core.tool_executor import ToolExecutor
from a3x.core.db_utils import add_episodic_record # Para log final
from a3x.core.context import _ToolExecutionContext
from a3x.fragments.base import BaseFragment

logger = logging.getLogger(__name__)

def register_missing_skill_heuristic(skill_name: str, context: dict = None):
    """
    Registra uma heurística de necessidade quando o agente tenta usar uma skill inexistente.
    Salva no arquivo de heurísticas como uma linha JSONL.
    """
    from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE
    import os

    heuristics_path = os.path.join(LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE)
    os.makedirs(LEARNING_LOGS_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds'),
        "type": "missing_skill_attempt",
        "skill_name": skill_name,
        "context": context or {},
        "message": f"O agente tentou usar a ferramenta '{skill_name}', que ainda não existe — considere implementá-la."
    }
    with open(heuristics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

@skill(
    name="learning_cycle",
    description="Orquestra a fase pós-execução completa: reflexão, aprendizado baseado no resultado (sucesso/falha), generalização e consolidação de heurísticas.",
    parameters={
        "objective": {"type": "str", "description": "Objetivo original que iniciou o ciclo"},
        "plan": {"type": "array", "description": "Plano que foi executado"},
        "execution_results": {"type": "array", "description": "Lista de dicionários com resultados de cada passo"},
        "final_status": {"type": "str", "description": "Status final da execução do plano ('completed', 'failed', 'error')"},
        "shared_task_context": {"type": "Optional[a3x.core.context.SharedTaskContext]", "description": "The final shared context from the task execution.", "optional": True} 
    },
)
async def learning_cycle_skill(
    context: _ToolExecutionContext,
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]],
    final_status: str,
    shared_task_context: Optional[SharedTaskContext] = None 
) -> Dict[str, Any]:
    """Skill principal para o ciclo de aprendizado pós-execução."""
    logger.info(f"--- Iniciando Skill Learning Cycle --- Status Final: {final_status}")
    
    tool_registry = getattr(context, 'tool_registry', None)
    tool_executor = ToolExecutor(
        tool_registry=tool_registry,
        llm_interface=getattr(context, 'llm_interface', None),
        fragment_registry=getattr(context, 'fragment_registry', None),
        memory_manager=getattr(context, 'memory_manager', None),
        workspace_root=getattr(context, 'workspace_root', None),
        default_logger=getattr(context, 'logger', None)
    )
    
    exec_context = context

    learned_info = []
    errors = []

    # --- 1. Reflexão e Aprendizado baseado no Status --- 
    if final_status == "completed":
        logger.info("Execução completada. Iniciando reflexão/aprendizado sobre sucesso.")
        try:
            last_successful_step = execution_results[-1] if execution_results else {}
            success_context_input = {
                 "objective": objective,
                 "executed_plan": plan,
                 "final_successful_step": last_successful_step.get('action', 'N/A'),
                 "final_observation": last_successful_step.get('observation', 'N/A'),
                 "final_task_context": shared_task_context 
             }
            reflect_success_result = await tool_executor.execute_tool(
                tool_name="reflect_on_success",
                tool_input=success_context_input,
                context=exec_context
            )
            if reflect_success_result.get("status") == "success":
                 learned_heuristic = reflect_success_result.get("data", {}).get("learned_heuristic", "N/A")
                 logger.info(f"Heurística de sucesso aprendida: {learned_heuristic[:100]}...")
                 learned_info.append(f"Success Heuristic: {learned_heuristic}")
                 try:
                     timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
                     log_entry = {
                         "timestamp": timestamp,
                         "objective": objective,
                         "plan": plan,
                         "results": execution_results,
                         "heuristic": learned_heuristic,
                         "type": "success"
                     }
                     log_file = Path(LEARNING_LOGS_DIR) / HEURISTIC_LOG_FILE
                     log_file.parent.mkdir(parents=True, exist_ok=True)
                     with open(log_file, 'a', encoding='utf-8') as f:
                         f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                     logger.info(f"Heurística de sucesso registrada em {log_file}: {learned_heuristic[:100]}...")
                 except Exception as log_write_err:
                     logger.exception(f"Falha ao escrever heurística de sucesso no log {HEURISTIC_LOG_FILE}: {log_write_err}")
            else:
                 error_msg = f"Skill reflect_on_success falhou: {reflect_success_result.get('data',{}).get('message')}"
                 logger.error(error_msg)
                 errors.append(error_msg)
        except Exception as reflect_err:
            error_msg = f"Erro ao chamar skill reflect_on_success: {reflect_err}"
            logger.exception(error_msg)
            errors.append(error_msg)
            
    elif final_status == "failed" or final_status == "error":
        logger.info("Execução falhou. Iniciando reflexão/aprendizado sobre falha.")
        try:
            failed_step_info = next((step for step in reversed(execution_results) if step.get('status') != 'success'), None)
            if failed_step_info is None and execution_results:
                failed_step_info = execution_results[-1]
            elif failed_step_info is None:
                failed_step_info = {}
                
            failure_context_input = {
                "objective": objective,
                "plan_executed": plan,
                "failed_step": failed_step_info.get('action', 'Unknown Action'),
                "error_message": failed_step_info.get('observation', 'Unknown Error'),
                "final_task_context": shared_task_context 
            }
            reflect_failure_result = await tool_executor.execute_tool(
                tool_name="reflect_on_failure",
                tool_input=failure_context_input,
                context=exec_context 
            )
            if reflect_failure_result.get("status") == "success":
                learned_heuristic = reflect_failure_result.get("data", {}).get("heuristic", "N/A")
                logger.info(f"Heurística de falha aprendida: {learned_heuristic[:100]}...")
                learned_info.append(f"Failure Heuristic: {learned_heuristic}")
            else:
                error_msg = f"Skill reflect_on_failure falhou: {reflect_failure_result.get('data',{}).get('message')}"
                logger.error(error_msg)
                errors.append(error_msg)
        except Exception as reflect_err:
            error_msg = f"Erro ao chamar skill reflect_on_failure: {reflect_err}"
            logger.exception(error_msg)
            errors.append(error_msg)

    # --- 2. Generalização e Consolidação --- 
    logger.info("Iniciando generalização e consolidação de heurísticas...")
    learning_skills = ["auto_generalize_heuristics", "consolidate_heuristics"]
    for skill_name in learning_skills:
        logger.info(f"Executando skill de aprendizado: {skill_name}...")
        try:
            result = await tool_executor.execute_tool(
                tool_name=skill_name,
                tool_input={},
                context=exec_context
            )
            if result.get("status") == "success":
                msg = result.get("data", {}).get("message", "Executada com sucesso.")
                logger.info(f"Skill {skill_name} completada: {msg}")
                learned_info.append(f"{skill_name}: {msg}")
            else:
                msg = result.get("data", {}).get("message", "Falha desconhecida.")
                logger.error(f"Skill {skill_name} falhou: {msg}")
                errors.append(f"{skill_name} Error: {msg}")
        except Exception as skill_err:
            error_msg = f"Erro ao executar skill {skill_name}: {skill_err}"
            logger.exception(error_msg)
            errors.append(error_msg)

    # --- 4. Retorno --- 
    logger.info("--- Skill Learning Cycle Concluída --- ")
    final_message = f"Ciclo de aprendizado concluído. Info: {learned_info}. Erros: {errors}"
    if errors:
         return {"status": "warning", "action": "learning_cycle_completed_with_errors", "data": {"message": final_message, "learned_info": learned_info, "errors": errors}}
    else:
         return {"status": "success", "action": "learning_cycle_completed", "data": {"message": final_message, "learned_info": learned_info}} 

class LearningCycle:
    """Manages the learning cycle for a specific skill or the agent."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.successful_executions: List[dict] = []
        self.failed_executions: List[dict] = []
        # Add more state as needed, e.g., performance metrics, feedback scores

    def record_success(self, context: _ToolExecutionContext, result: str, parameters: dict):
        """Records a successful skill execution."""
        self.successful_executions.append({
            "context": context.serialize() if hasattr(context, 'serialize') else str(context),
            "result": result,
            "parameters": parameters,
            # Potentially add timestamp, resource usage, etc.
        })

    def record_failure(self, context: _ToolExecutionContext, error: Exception, parameters: dict):
        """Records a failed skill execution."""
        self.failed_executions.append({
            "context": context.serialize() if hasattr(context, 'serialize') else str(context),
            "error": str(error),
            "parameters": parameters,
            # Potentially add timestamp, stack trace, etc.
        })

    def analyze_performance(self) -> dict:
        """Analyzes past executions to derive insights."""
        success_rate = len(self.successful_executions) / (len(self.successful_executions) + len(self.failed_executions))
        # Add more sophisticated analysis: common failure modes, optimal parameters, etc.
        return {"success_rate": success_rate}

    def suggest_improvements(self) -> Optional[str]:
        """Suggests improvements based on performance analysis."""
        # Placeholder: Implement logic to generate suggestions
        # e.g., if success rate is low, suggest parameter tuning or code review.
        if self.analyze_performance().get("success_rate", 1.0) < 0.5:
            return f"Skill '{self.skill_name}' has a low success rate. Consider reviewing failures or adjusting parameters."
        return None 