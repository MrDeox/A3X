"""
Skill para analisar individualmente e depois sintetizar insights de múltiplas
análises de logs de reflexão.
"""

import logging
import time
import asyncio # Needed for potential concurrent individual analyses
from typing import List, Dict, Any, AsyncGenerator

# Import the skill decorator
from a3x.core.tools import skill

# Context stub for type hinting (can be replaced by actual SkillContext if available globally)
class SkillContext:
    logger: logging.Logger
    llm_call: Any # Function or async generator supporting streaming

@skill(
    name="synthesize_learning_insights",
    description="Analisa individualmente e depois sintetiza insights de múltiplas análises de logs de reflexão (geradas por learn_from_reflection_logs).",
    parameters={
        "analyses": (List[str], ...), # Ellipsis indicates required parameter
    }
)
async def synthesize_learning_insights(ctx: SkillContext, analyses: List[str]) -> Dict[str, Any]:
    """
    Realiza uma análise em duas etapas:
    1. Gera um resumo estruturado para cada análise de log individualmente.
    2. Sintetiza esses resumos individuais em um relatório consolidado final.

    Args:
        ctx: O contexto da skill, contendo logger e llm_call.
        analyses: Uma lista de strings, onde cada string é uma análise
                  previamente gerada por learn_from_reflection_logs.

    Returns:
        Um dicionário contendo:
        - 'individual_insights': Uma lista com os resumos estruturados de cada análise.
        - 'synthesized_insights': O texto da síntese consolidada final.
        Ou um dicionário de erro se algo falhar.
    """
    start_time = time.time()
    num_analyses = len(analyses)
    ctx.logger.info(f"Starting 2-step synthesis for {num_analyses} analyses...")

    if not analyses:
        ctx.logger.warning("No analyses provided for synthesis.")
        return {"error": "Nenhuma análise fornecida para síntese."}

    individual_summaries: List[str] = []
    # tasks = [] # Removed task list for sequential execution

    # --- Etapa 1: Análise Individual (SEQUENTIAL) ---
    ctx.logger.info("Step 1: Generating individual summaries (sequentially)...")
    # for i, analysis_text in enumerate(analyses):
    #     tasks.append(_analyze_individual_log(ctx, analysis_text, i + 1))
    # individual_results = await asyncio.gather(*tasks)

    # Execute sequentially instead of concurrently
    for i, analysis_text in enumerate(analyses):
        ctx.logger.debug(f"Processing analysis #{i+1} sequentially...")
        result = await _analyze_individual_log(ctx, analysis_text, i + 1)
        if isinstance(result, str): # Success
            individual_summaries.append(result)
        else: # Error occurred
             ctx.logger.error(f"Failed to generate summary for analysis #{i+1} sequentially. Skipping synthesis.")
             return {"error": f"Falha ao gerar resumo para análise individual #{i+1}.", "failed_result": result}

    # # Check for errors and collect successful summaries (Now handled inside the loop)
    # for result in individual_results:
    #     if isinstance(result, str):
    #         individual_summaries.append(result)
    #     else:
    #          ctx.logger.error(f"Failed to generate summary for one of the analyses. Skipping synthesis.")
    #          return {"error": "Falha ao gerar resumo para uma ou mais análises individuais.", "partial_results": individual_results}


    if not individual_summaries:
         ctx.logger.error("No individual summaries were successfully generated (sequentially). Cannot proceed to synthesis.")
         return {"error": "Nenhum resumo individual gerado com sucesso (execução sequencial)."}

    ctx.logger.info(f"Step 1 finished. Generated {len(individual_summaries)} individual summaries.")

    # --- Etapa 2: Síntese Final ---
    ctx.logger.info("Step 2: Generating final synthesis from individual summaries...")
    summary_lines = []
    for i, summary in enumerate(individual_summaries):
        summary_lines.append(f"Resumo #{i+1}:\n{summary}")
    combined_summaries = "\n\n---\n\n".join(summary_lines)

    synthesis_prompt = f"""
Você é um especialista em síntese de conhecimento, encarregado de consolidar aprendizados de múltiplas análises sobre um sistema de IA (Arthur). Abaixo estão resumos estruturados derivados de análises individuais de logs de reflexão:

--- RESUMOS INDIVIDUAIS ---
{combined_summaries}
--- FIM DOS RESUMOS INDIVIDUAIS ---

Sua tarefa é analisar TODOS os resumos individuais acima e gerar uma **SÍNTESE GERAL CONSOLIDADA**. Seu objetivo é:
1. Identificar padrões e temas recorrentes em todos os resumos.
2. Agrupar informações semelhantes (heurísticas, feedbacks, sugestões de prompt, ideias de LoRA).
3. Eliminar redundâncias e apresentar as informações de forma unificada e concisa.

O resultado final deve ser um texto único, claro e bem estruturado, contendo EXCLUSIVAMENTE as seguintes seções (use exatamente estes títulos em negrito):

- **Heurísticas Unificadas:** (Liste as heurísticas de comportamento/pensamento mais significativas e recorrentes identificadas nos resumos. Agrupe temas.)
- **Feedbacks Frequentes:** (Resuma os pontos de feedback mais comuns ou importantes mencionados nos resumos.)
- **Sugestões de Prompt Refatorado:** (Consolide as sugestões de melhoria de prompt. Combine ideias parecidas e priorize.)
- **Categorias de LoRAs Potenciais:** (Agrupe as ideias de LoRAs temáticas em categorias mais amplas. Evite nomes redundantes.)

**Importante:** Seja preciso e conciso. O resultado será usado para guiar o refinamento do sistema. Não adicione introduções, conclusões ou comentários fora das seções solicitadas.

**Síntese Geral Consolidada:**
"""

    final_synthesis = ""
    try:
        ctx.logger.debug(f"Final synthesis prompt constructed (length: {len(synthesis_prompt)} chars). Calling LLM...")
        llm_stream = ctx.llm_call(synthesis_prompt)

        if isinstance(llm_stream, AsyncGenerator):
            async for chunk in llm_stream:
                final_synthesis += chunk
        else:
            ctx.logger.warning("LLM call for final synthesis did not return an async generator.")
            final_synthesis = str(llm_stream)

        if not final_synthesis.strip():
            ctx.logger.warning("LLM final synthesis resulted in an empty response.")
            final_synthesis = "Síntese final falhou ou retornou vazia." # Indicate failure

    except Exception as e:
        ctx.logger.exception("Error during LLM call for final synthesis:")
        final_synthesis = f"Erro ao gerar síntese final: {e}" # Indicate failure

    end_time = time.time()
    duration = end_time - start_time
    ctx.logger.info(f"Full 2-step synthesis finished in {duration:.2f} seconds.")

    return {
        "individual_insights": individual_summaries,
        "synthesized_insights": final_synthesis.strip()
    }


async def _analyze_individual_log(ctx: SkillContext, analysis_text: str, index: int) -> str | Dict[str, str]:
    """
    Helper function to analyze a single log analysis text using LLM.
    Returns the summarized string on success, or an error dict on failure.
    """
    prompt = f"""
Você é um analista de aprendizado focado em extrair a essência de textos. Abaixo está uma análise gerada a partir de um log de reflexão de um sistema de IA chamado Arthur:

--- ANÁLISE INDIVIDUAL ---
{analysis_text}
--- FIM DA ANÁLISE INDIVIDUAL ---

Sua tarefa é extrair os pontos mais importantes desta análise específica e apresentá-los de forma estruturada e concisa. Use EXATAMENTE os seguintes marcadores em negrito:

- **Heurísticas Mais Relevantes:** (Resuma as 1-3 heurísticas chave mencionadas)
- **Feedbacks Importantes:** (Resuma os 1-2 pontos de feedback principais, se houver)
- **Sugestões de Prompt Notáveis:** (Resuma as 1-2 sugestões mais impactantes, se houver)
- **LoRAs Interessantes:** (Liste as 1-2 ideias de LoRA mais promissoras, se houver)

Seja direto, use frases curtas e evite jargões desnecessários. Foque na essência da análise fornecida. Não adicione introduções ou conclusões.

**Resumo Estruturado:**
"""
    summary = ""
    try:
        ctx.logger.debug(f"Analyzing individual log #{index} (length: {len(analysis_text)} chars). Calling LLM...")
        llm_stream = ctx.llm_call(prompt)

        if isinstance(llm_stream, AsyncGenerator):
            async for chunk in llm_stream:
                summary += chunk
        else:
             ctx.logger.warning(f"LLM call for individual analysis #{index} did not return an async generator.")
             summary = str(llm_stream)

        if not summary.strip():
            ctx.logger.warning(f"LLM individual analysis #{index} resulted in an empty response.")
            return {"error": f"Análise individual #{index} retornou vazia."}

        ctx.logger.debug(f"Successfully generated summary for analysis #{index}.")
        return summary.strip()

    except Exception as e:
        ctx.logger.exception(f"Error during LLM call for individual analysis #{index}:")
        return {"error": f"Erro ao analisar log individual #{index}: {e}"}


# Remover ou comentar o código de teste antigo/mock, pois a estrutura mudou.
# async def _mock_llm_call...
# async def _test_skill()...
# if __name__ == '__main__': ... 