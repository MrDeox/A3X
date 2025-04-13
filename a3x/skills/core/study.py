import logging
from typing import Dict, Any, List, Optional

from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name="study",
    description=(
        "Entra em modo de aprendizado ativo, buscando e assimilando conhecimento necessário para executar uma tarefa. "
        "Utiliza visão (ex: enxergar tela, ler arquivos, analisar contexto) e faz conexões inteligentes, absorvendo informações, "
        "fazendo perguntas, resumindo, relacionando conceitos e aplicando o que aprendeu de forma adaptativa."
    ),
    parameters={
        "ctx": {"type": "Context", "description": "The execution context."},
        "task": {"type": "str", "description": "The specific learning task to perform."},
        "context": {"type": "dict", "description": "Optional dictionary providing surrounding context for the task."},
        "vision": {"type": "str", "description": "Optional visual input (e.g., screenshot text, file content)."},
        "resources": {"type": "list", "description": "Optional list of resource identifiers (files, links) to study."}
    },
)
async def study_skill(
    task: str,
    ctx: _ToolExecutionContext,
    context: Optional[Dict] = None,
    vision: Optional[str] = None,
    resources: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Skill de estudo ativo: busca, lê, conecta, pergunta, resume e aprende para executar uma tarefa com compreensão profunda.
    Uses the LLMInterface from the execution context.
    """
    logger = ctx.logger
    llm_interface = ctx.llm_interface
    
    if not llm_interface:
        logger.error("LLMInterface not found in execution context for study skill.")
        return {"status": "error", "data": {"message": "Internal error: LLMInterface missing."}}
        
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous study agent. Given a task, context, vision (e.g., screen, files) and resources, "
                "study actively like a human: read, connect ideas, ask questions, search for answers, summarize, relate concepts, "
                "and explain what you learned clearly and in an applicable way. If needed, propose next learning or action steps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Context: {context}\n"
                f"Vision: {vision}\n"
                f"Resources: {resources}\n"
            ),
        },
    ]
    try:
        logger.info(f"[StudySkill] Starting study for task: {task}")
        answer = ""
        async for chunk in llm_interface.call_llm(
            messages=prompt_messages, 
            stream=False
        ):
            answer += chunk
            
        if answer.startswith("[LLM Error:"):
             logger.error(f"[StudySkill] LLM call failed: {answer}")
             return {"status": "error", "data": {"message": answer}}
             
        if not answer.strip():
             logger.warning(f"[StudySkill] LLM returned empty response for task: {task}")
             answer = "(LLM returned empty response)"

        logger.info(f"[StudySkill] LLM response:\n{answer}")
        return {"status": "success", "data": {"study": answer.strip()}}
    except Exception as e:
        logger.error(f"[StudySkill] Error during study: {e}", exc_info=True)
        return {"status": "error", "data": {"message": str(e)}}