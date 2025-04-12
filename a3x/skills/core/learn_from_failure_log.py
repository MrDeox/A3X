import logging
import json
import os
from typing import Dict, Any, Optional

# Ensure skill decorator and utils are imported correctly
try:
    from a3x.core.skills import skill
    from a3x.core.llm_interface import call_llm # Use interface for consistency
    from a3x.core.context import Context
except ImportError:
    # Fallback for standalone testing
    def skill(**kwargs):
        def decorator(func):
            return func
        return decorator
    async def call_llm(*args, **kwargs):
        yield "Heurística de teste (sem LLM real)"
    Context = Any

logger = logging.getLogger(__name__)

HEURISTIC_EXTRACTION_PROMPT_TEMPLATE = """
# Extração de Heurística de Análise de Falha

Abaixo está a análise de uma falha ocorrida durante a execução de um plano por um agente autônomo (A³X).
Seu objetivo é extrair UMA ÚNICA heurística (regra prática) concisa e acionável em linguagem natural, baseada principalmente na seção "Como corrigir ou depurar".

A heurística deve seguir um dos seguintes formatos:
- "Se [condição/sintoma do erro], então [ação corretiva recomendada]."
- "Ao encontrar [descrição do erro], verifique/tente [ação corretiva]."
- "Para evitar o erro [tipo de erro], certifique-se de [pré-condição ou verificação]."

**Análise da Falha:**
```
{failure_analysis}
```

**Tarefa:**
Gere APENAS a heurística extraída, sem nenhum texto adicional antes ou depois. Mantenha-a curta e direta.

**Heurística:**
"""

# <<< REMOVED DEFAULT_FAILURE_LOG_PATH >>>
# DEFAULT_FAILURE_LOG_PATH = "memory/failure_logs/learned_failures.jsonl"

@skill(
    name="learn_from_failure_log",
    # <<< UPDATED Description and Parameters >>>
    description="Gera uma heurística concisa a partir de uma análise de falha detalhada.",
    parameters=[
        {"name": "failure_analysis", "type": "string", "description": "Texto detalhado analisando a causa raiz e a correção da falha."},
        {"name": "ctx", "type": "Context", "description": "Objeto de contexto (para LLM URL).", "optional": True}
    ]
)
async def learn_from_failure_log(failure_analysis: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Gera uma heurística a partir de uma análise de falha usando LLM."""
    log_prefix = "[LearnFromFailure Skill]"
    logger.info(f"{log_prefix} Gerando heurística a partir da análise de falha...")

    if not failure_analysis:
        logger.warning(f"{log_prefix} Análise de falha vazia fornecida. Nenhuma heurística pode ser gerada.")
        return {"status": "error", "data": {"message": "Análise de falha vazia."}}

    # <<< SIMPLIFIED: Only generate heuristic via LLM >>>
    prompt = HEURISTIC_EXTRACTION_PROMPT_TEMPLATE.format(failure_analysis=failure_analysis)
    prompt_messages = [
         # Simple user prompt might be enough if template is clear
         {"role": "user", "content": prompt}
    ]

    llm_url = getattr(ctx, 'llm_url', None) if ctx else None
    heuristic_text = ""
    try:
        logger.debug(f"{log_prefix} Chamando LLM para extrair heurística...")
        # Expecting a short, direct response, not streaming
        async for chunk in call_llm(prompt_messages, llm_url=llm_url, stream=False, max_tokens=100, temperature=0.2):
             heuristic_text += chunk
        
        heuristic_text = heuristic_text.strip().strip('"') # Corrected quotes
        logger.debug(f"{log_prefix} LLM raw response: {heuristic_text}")

        if heuristic_text and len(heuristic_text) > 5: # Basic validation
            logger.info(f"{log_prefix} Heurística gerada: {heuristic_text}")
            # <<< UPDATED: Return only the heuristic >>>
            return {
                "status": "success",
                "data": {"heuristic": heuristic_text}
            }
        else:
             logger.warning(f"{log_prefix} LLM retornou uma heurística vazia ou muito curta: '{heuristic_text}'")
             return {"status": "error", "data": {"message": "LLM não gerou uma heurística válida."}}

    except Exception as e:
        logger.exception(f"{log_prefix} Erro durante a chamada LLM para extração de heurística:")
        return {"status": "error", "data": {"message": f"Erro na chamada LLM: {e}"}}

    # <<< REMOVED file writing logic >>>

# Example Test Block (Simplified)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    async def run_main_test():
        print("\n--- Running Learn From Failure Log Test --- ")
        dummy_analysis = """
        Erro encontrado: FileNotFoundError ao tentar ler 'data/input.csv'.
        Causa Provável: O arquivo não existe no caminho esperado.
        Como corrigir ou depurar: Antes de tentar ler um arquivo, use a skill 'list_dir' para verificar se o arquivo e o diretório existem.
        """
        result = await learn_from_failure_log(failure_analysis=dummy_analysis)
        print("\n--- Learn From Failure Result --- ")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        result_empty = await learn_from_failure_log(failure_analysis="")
        print("\n--- Learn From Failure Result (Empty Analysis) --- ")
        print(json.dumps(result_empty, indent=2, ensure_ascii=False))

    import asyncio
    asyncio.run(run_main_test()) 