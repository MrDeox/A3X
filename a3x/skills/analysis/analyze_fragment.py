import logging
from typing import Optional, Dict, Any
from a3x.core.context import Context
from a3x.core.skills import skill

logger = logging.getLogger(__name__)

@skill(
    name="analyze_fragment",
    description="Analisa o desempenho simbólico de um fragmento A³X com base em suas reflexões, falhas ou logs.",
    parameters={
        "fragment_name": {"type": "string", "description": "Nome do fragmento a ser analisado"},
        "reflection": {"type": "string", "description": "Texto simbólico com reflexões ou observações sobre o fragmento"},
        "logs": {"type": "string", "description": "Logs de execução ou mensagens relevantes", "default": ""},
    }
)
async def analyze_fragment(
    context: Context,
    fragment_name: str,
    reflection: str,
    logs: Optional[str] = ""
) -> str:
    """
    Usa o LLM para analisar simbolicamente o comportamento do fragmento.
    """
    logger.info(f"Analisando fragmento: {fragment_name}")

    prompt = (
        f"Fragmento analisado: {fragment_name}\n\n"
        f"Reflexões recentes:\n{reflection}\n\n"
        f"Logs disponíveis:\n{logs if logs else 'Nenhum log fornecido'}\n\n"
        f"Com base nessas informações, forneça uma análise simbólica sobre o desempenho, "
        f"comportamento e possíveis melhorias, especializações ou correções para esse fragmento."
    )

    logger.debug(f"Prompt para análise de fragmento: {prompt}")
    
    try:
        # Tentativa com método 'complete' (se existir)
        result = await context.llm_interface.complete(prompt, max_tokens=600) 
        return result.strip()
    except AttributeError:
        logger.warning("Método 'complete' não encontrado em llm_interface. Usando 'call_llm' como fallback.")
        # Fallback para método 'call_llm' (mais comum)
        messages = [{"role": "user", "content": prompt}]
        response_text = ""
        try:
            async for chunk in context.llm_interface.call_llm(messages=messages, stream=False, max_tokens=600):
                response_text += chunk
            return response_text.strip()
        except Exception as e:
            logger.exception(f"Erro ao chamar LLM via call_llm para analyze_fragment ({fragment_name})")
            return f"Erro ao analisar o fragmento {fragment_name}: {e}"
    except Exception as e:
        logger.exception(f"Erro inesperado ao chamar LLM para analyze_fragment ({fragment_name})")
        return f"Erro inesperado ao analisar o fragmento {fragment_name}: {e}" 