import logging
from typing import List, Dict, Any
import json

from a3x.core.skills import skill
from a3x.core.llm_interface import call_llm # Assuming standard call_llm

logger = logging.getLogger(__name__)

FAILURE_REFLECTION_PROMPT_TEMPLATE = """
# Análise de Falha de Execução do Agente A³X

Ocorreu uma falha durante a execução de um plano. Analise os detalhes abaixo e forneça uma explicação estruturada.

## Contexto da Falha

*   **Objetivo Geral:** {objective}
*   **Plano Sendo Executado:**
{plan_steps}
*   **Último Raciocínio (Thought):** {last_thought}
*   **Última Ação Tentada:** {last_action}
*   **Observação/Erro Recebido:** {last_observation}

## Tarefa

Gere uma análise concisa da falha, seguindo ESTRITAMENTE a estrutura abaixo:

**1. O que o Executor tentou fazer:**
[Descreva em 1-2 frases a intenção por trás da última ação tentada, com base no objetivo, plano e raciocínio.]

**2. Por que deu errado:**
[Explique a causa mais provável da falha com base na observação/erro recebida. Seja direto e técnico se necessário (ex: ferramenta não encontrada, parâmetro inválido, erro de API, etc.).]

**3. Como corrigir ou depurar:**
[Forneça instruções CLARAS e ACIONÁVEIS para o Executor (ou um desenvolvedor humano) sobre os próximos passos para resolver o problema. Exemplos: "Verificar se a skill X está registrada", "Corrigir o parâmetro Y na chamada da ferramenta Z", "Analisar o log de erro completo da API externa", "Tentar a ação novamente com o parâmetro W modificado para V"].
"""

@skill(
    name="reflect_on_failure",
    description="Analisa uma falha ocorrida durante a execução de um plano e gera uma explicação estruturada sobre a causa e possíveis correções.",
    parameters={
        "objective": (str, ...), # O objetivo original do plano.
        "plan": (List[str], ...), # Os passos do plano.
        "last_thought": (str, ...), # O último "Thought" executado antes da falha.
        "last_action": (str, ...), # O último "Action" executado.
        "last_observation": (str, ...) # O conteúdo da "Observation" que gerou o erro.
    }
)
async def reflect_on_failure(ctx: Any, objective: str, plan: List[str], last_thought: str, last_action: str, last_observation: str) -> Dict[str, Any]:
    """Analisa uma falha de execução e gera um relatório estruturado via LLM."""

    log_prefix = "[ReflectOnFailure Skill]"
    logger.info(f"{log_prefix} Iniciando reflexão sobre falha na ação '{last_action}' para o objetivo '{objective[:50]}...'")

    # Format plan steps for the prompt
    plan_steps_formatted = "\n".join([f"    - Passo {i+1}: {step}" for i, step in enumerate(plan)])

    # Construct the prompt
    prompt_content = FAILURE_REFLECTION_PROMPT_TEMPLATE.format(
        objective=objective,
        plan_steps=plan_steps_formatted,
        last_thought=last_thought,
        last_action=last_action,
        last_observation=last_observation
    )

    # Prepare prompt for LLM call (assuming a simple user message structure)
    prompt_messages = [
        {"role": "system", "content": "Você é um assistente de análise de logs especializado em depurar falhas de agentes autônomos."},
        {"role": "user", "content": prompt_content}
    ]

    logger.debug(f"{log_prefix} Enviando prompt para LLM para análise de falha...")

    llm_response_raw = ""
    try:
        # Call the LLM (use context's llm_url if available, otherwise default)
        llm_url = getattr(ctx, 'llm_url', None)
        
        # Assuming call_llm returns an async generator
        async for chunk in call_llm(prompt_messages, llm_url=llm_url, stream=False):
             llm_response_raw += chunk
        
        # Removed the complex else block as call_llm is expected to be async generator based on other usage
        if not llm_response_raw:
             logger.warning(f"{log_prefix} LLM call returned empty response for failure analysis.")
             llm_response_raw = "(LLM did not provide an analysis)" # Provide fallback

        logger.info(f"{log_prefix} Resposta da análise de falha recebida do LLM.")
        logger.debug(f"{log_prefix} Resposta bruta LLM:\n{llm_response_raw}")

        # Return the structured explanation from the LLM
        return {
            "status": "success",
            "action": "failure_analysis_generated",
            "data": {
                "explanation": llm_response_raw.strip()
            }
        }

    except Exception as e:
        logger.exception(f"{log_prefix} Erro ao chamar LLM para análise de falha:")
        return {
            "status": "error",
            "action": "llm_call_failed",
            "data": {"message": f"Erro ao gerar análise de falha via LLM: {e}"}
        } 