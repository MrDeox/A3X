import logging
from a3x.core.context import Context
from a3x.core.skills import skill
from typing import Optional

logger = logging.getLogger(__name__)

@skill(
    name="analyze_code_quality",
    description="Analisa um trecho de código quanto à clareza, legibilidade, eficiência e boas práticas.",
    parameters={
        "code": {"type": "string", "description": "O código a ser analisado"},
        "language": {"type": "string", "description": "Linguagem de programação do código (ex: python, javascript)", "default": "python"}
    }
)
async def analyze_code_quality(
    context: Context,
    code: str,
    language: Optional[str] = "python"
) -> str:
    """
    Analisa o código recebido utilizando o modelo LLM para identificar pontos fortes e sugestões de melhoria.
    """
    prompt = (
        f"Você é um analista de código especializado em {language}.\n"
        f"Analise o seguinte trecho de código quanto a clareza, legibilidade, organização e boas práticas:\n\n"
        f"{code}\n\n"
        f"Por favor, identifique pontos positivos e negativos e sugira melhorias caso necessário."
    )
    logger.info(f"Enviando código para análise de qualidade. Linguagem: {language}")
    
    # Assuming llm_interface has a method like 'complete' or similar for direct completion
    # If it uses call_llm with messages, adjust accordingly:
    # messages = [{"role": "user", "content": prompt}]
    # response_text = ""
    # async for chunk in context.llm_interface.call_llm(messages=messages, stream=False, max_tokens=500):
    #     response_text += chunk
    # return response_text.strip()
    
    # Using a hypothetical 'complete' method for simplicity as suggested by user prompt
    # Ensure this method exists and matches the expected signature in llm_interface
    try:
        result = await context.llm_interface.complete(prompt, max_tokens=500) # Placeholder - Adjust if needed
        return result.strip()
    except AttributeError:
        logger.error("Método 'complete' não encontrado em llm_interface. Tentando com 'call_llm'.")
        messages = [{"role": "user", "content": prompt}]
        response_text = ""
        try:
             async for chunk in context.llm_interface.call_llm(messages=messages, stream=False, max_tokens=500):
                 response_text += chunk
             return response_text.strip()
        except Exception as e:
            logger.exception("Erro ao chamar LLM via call_llm para analyze_code_quality")
            return f"Erro ao analisar o código: {e}"
    except Exception as e:
        logger.exception("Erro inesperado ao chamar LLM para analyze_code_quality")
        return f"Erro inesperado ao analisar o código: {e}" 