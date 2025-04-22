# a3x/skills/core/learning_cycle.py
import logging
from typing import Dict, Any, List, Optional

from pathlib import Path
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE
import json
import datetime
from a3x.core.skills import skill
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext
from a3x.core.db_utils import add_episodic_record # Para log final
from a3x.core.context import SharedTaskContext

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
        "agent_tools": {"type": "object", "description": "Dicionário de tools/skills disponíveis para o agente"},
        "agent_workspace": {"type": "str", "description": "Caminho do workspace"},
        "agent_llm_url": {"type": "str", "description": "URL do LLM, se necessário para sub-skills"},
        "shared_task_context": {"type": "Optional[a3x.core.context.SharedTaskContext]", "description": "The final shared context from the task execution.", "optional": True} 
    },
)
async def learning_cycle_skill(
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]],
    final_status: str,
    agent_tools: dict,
    agent_workspace: str,
    agent_llm_url: Optional[str] = None,
    shared_task_context: Optional[SharedTaskContext] = None 
) -> Dict[str, Any]:
    """Skill principal para o ciclo de aprendizado pós-execução."""
    logger.info(f"--- Iniciando Skill Learning Cycle --- Status Final: {final_status}")
    
    # Criar contexto para as sub-skills
    # Passando logger, workspace, llm_url e shared_context herdados
    exec_context = _ToolExecutionContext(
        logger=logger, 
        workspace_root=agent_workspace, 
        llm_url=agent_llm_url, 
        tools_dict=agent_tools,
        # Placeholder - assumindo que podem ser None ou obtidos de outra forma se as sub-skills não precisarem deles diretamente
        llm_interface=None, 
        fragment_registry=None, 
        shared_task_context=shared_task_context # Passa o contexto recebido
    )

    learned_info = [] # Para coletar informações aprendidas
    errors = [] # Para coletar erros durante o ciclo

    # --- 1. Reflexão e Aprendizado baseado no Status --- 
    if final_status == "completed":
        logger.info("Execução completada. Iniciando reflexão/aprendizado sobre sucesso.")
        try:
            success_context = {
                 "objective": objective,
                 "plan": plan,
                 "execution_results": execution_results,
                 "final_task_context": shared_task_context 
             }
            reflect_success_result = await execute_tool(
                tool_name="reflect_on_success",
                action_input=success_context,
                tools_dict=agent_tools,
                context=exec_context
            )
            if reflect_success_result.get("status") == "success":
                 learned_heuristic = reflect_success_result.get("data", {}).get("learned_heuristic", "N/A")
                 logger.info(f"Heurística de sucesso aprendida: {learned_heuristic[:100]}...")
                 learned_info.append(f"Success Heuristic: {learned_heuristic}")
                 # Registro da heurística de sucesso em arquivo
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
        # Idealmente, chamaríamos reflect_on_failure aqui também
        # Isso requer refatorar como o final_status e results são tratados
        # Exemplo de como seria:
        try:
            failure_context = {
                "objective": objective,
                "plan": plan,
                "execution_results": execution_results,
                "final_task_context": shared_task_context # Passa o contexto
            }
            reflect_failure_result = await execute_tool(
                tool_name="reflect_on_failure",
                action_input=failure_context,
                tools_dict=agent_tools,
                context=exec_context # Passa o contexto de execução geral
            )
            # Processar resultado de reflect_on_failure similarmente ao de success
            if reflect_failure_result.get("status") == "success":
                learned_heuristic = reflect_failure_result.get("data", {}).get("heuristic", "N/A")
                logger.info(f"Heurística de falha aprendida: {learned_heuristic[:100]}...")
                learned_info.append(f"Failure Heuristic: {learned_heuristic}")
                # Logar heurística de falha (código omitido para brevidade, similar ao de sucesso)
            else:
                error_msg = f"Skill reflect_on_failure falhou: {reflect_failure_result.get('data',{}).get('message')}"
                logger.error(error_msg)
                errors.append(error_msg)
        except Exception as reflect_err:
            error_msg = f"Erro ao chamar skill reflect_on_failure: {reflect_err}"
            logger.exception(error_msg)
            errors.append(error_msg)
        # FIM DO BLOCO DE EXEMPLO PARA REFLECT_ON_FAILURE

        # A lógica de aprendizado de falha pode precisar ser mais sofisticada
        # do que apenas logar a informação que já estava no execution_results.
        # failed_step_info = "Detalhes da falha não disponíveis aqui (processado em _execute_plan)."
        # learned_info.append(f"Failure processed during execution: {failed_step_info}")
        # logger.warning("Reflexão/Aprendizado de falha atualmente tratado durante a execução (_execute_plan).")

    # --- 2. Generalização e Consolidação --- 
    logger.info("Iniciando generalização e consolidação de heurísticas...")
    learning_skills = ["auto_generalize_heuristics", "consolidate_heuristics"]
    for skill_name in learning_skills:
        logger.info(f"Executando skill de aprendizado: {skill_name}...")
        try:
            # Usar o mesmo contexto
            result = await execute_tool(
                tool_name=skill_name,
                action_input={}, # Use default parameters
                tools_dict=agent_tools,
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

    # --- 3. Log Episódico Final (Opcional) --- 
    # A versão em _reflect_and_learn já fazia isso. Manter ou remover?
    # try:
    #     summary_metadata = {...}
    #     add_episodic_record(...)
    # except Exception as db_err:
    #     logger.error(f"Erro ao registrar experiência episódica final: {db_err}")

    # --- 4. Retorno --- 
    logger.info("--- Skill Learning Cycle Concluída --- ")
    final_message = f"Ciclo de aprendizado concluído. Info: {learned_info}. Erros: {errors}"
    if errors:
         return {"status": "warning", "action": "learning_cycle_completed_with_errors", "data": {"message": final_message, "learned_info": learned_info, "errors": errors}}
    else:
         return {"status": "success", "action": "learning_cycle_completed", "data": {"message": final_message, "learned_info": learned_info}} 

def learning_cycle(objective, plan, execution_results, final_status, agent_tools, agent_workspace, agent_llm_url=None):
    """
    Orquestra a fase pós-execução completa: reflexão, aprendizado baseado no resultado (sucesso/falha), generalização e consolidação de heurísticas.
    """
    logger.info("[Learning Cycle] Iniciando ciclo de aprendizado pós-execução...")
    
    # Reflexão baseada no resultado
    if final_status == "success":
        reflection_result = reflect_on_success(objective, plan, execution_results)
        logger.info("[Learning Cycle] Reflexão sobre sucesso concluída.")
    else:
        reflection_result = reflect_on_failure(objective, plan, execution_results[-1]["thought"] if execution_results else "", execution_results[-1]["action"] if execution_results else "", execution_results[-1]["observation"] if execution_results else "")
        logger.info("[Learning Cycle] Reflexão sobre falha concluída.")
    
    # Aprendizado a partir da reflexão
    learn_result = learn_from_failure_log(reflection_result) if final_status != "success" else ""
    if learn_result:
        logger.info("[Learning Cycle] Heurística aprendida a partir da reflexão.")
    
    # Análise de logs de reflexão para ajustes de prompt e parâmetros do LLM
    log_analysis = learn_from_reflection_logs(n_logs=10)
    if log_analysis:
        logger.info("[Learning Cycle] Análise de logs de reflexão concluída.")
        prompt_refinement = refine_decision_prompt()
        if prompt_refinement:
            logger.info("[Learning Cycle] Refinamento de prompt proposto.")
            apply_prompt_refinement_from_logs(prompt_refinement, "learning_cycle_analysis")
            logger.info("[Learning Cycle] Refinamento de prompt aplicado.")
    
    # Generalização e consolidação de heurísticas
    auto_generalize_heuristics(threshold=5)
    consolidate_heuristics(similarity_threshold=0.7)
    logger.info("[Learning Cycle] Ciclo de aprendizado concluído.")
    
    return "Ciclo de aprendizado concluído. Heurísticas atualizadas." 