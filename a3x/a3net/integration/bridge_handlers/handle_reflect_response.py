import logging
from typing import Dict, Optional, Any, Callable, Awaitable

# Assuming these are available in the environment
from ..core.professor_llm_fragment import ProfessorLLMFragment
from ..core.context_store import ContextStore

logger = logging.getLogger(__name__)

async def handle_reflect_response(
    directive: Dict[str, Any],
    fragment_instances: Optional[Dict[str, Any]],
    context_store: Optional[ContextStore],
    post_message_handler: Optional[Callable[..., Awaitable[None]]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'refletir_resposta' directive logic."""

    fragment_id = directive.get("fragment_id")

    if not fragment_id:
        logger.error("[A3X Bridge Handler - Reflect Resp] 'fragment_id' missing.")
        return { "status": "error", "message": "'fragment_id' is required for refletir_resposta" }
    
    if not context_store:
        logger.error("[A3X Bridge Handler - Reflect Resp] ContextStore not available.")
        return { "status": "error", "message": "ContextStore is required for refletir_resposta" }

    try:
        last_ask_key = f"last_ask_result:{fragment_id}"
        last_ask_data = await context_store.get(last_ask_key)

        if not last_ask_data:
             logger.warning(f"[A3X Bridge Handler - Reflect Resp] No last ask result found for key '{last_ask_key}'")
             return { "status": "error", "message": f"No previous answer found for fragment '{fragment_id}' to reflect upon." }

        input_val = last_ask_data.get('input_value', '[Input not recorded]')
        output_val = last_ask_data.get('output', '[Output not recorded]')
        confidence_val = last_ask_data.get('confidence', -1.0)
        explanation_val = last_ask_data.get('explanation')

        input_str = f'{repr(input_val)}' if isinstance(input_val, list) else f'"{input_val}"'
        
        prompt = (
            f"O fragmento '{fragment_id}' respondeu '{output_val}' "
            f"(confiança: {confidence_val:.4f}) para a entrada {input_str}."
        )
        if explanation_val:
             prompt += f"\nExplicação fornecida: \"{explanation_val}\""
        prompt += f"\n\nO que o fragmento poderia aprender ou como ele poderia melhorar com base nessa interação? "
        prompt += f"Responda APENAS com comandos A3L (ex: treinar, criar, etc.) para aplicar o aprendizado."
        
        logger.info(f"[A3X Bridge Handler - Reflect Resp] Prompting Professor for reflection on '{fragment_id}'.")
        logger.debug(f"[A3X Bridge Handler - Reflect Resp] Professor Prompt: {prompt}")

        professor_fragment = fragment_instances.get("prof_geral") if fragment_instances else None
        if not isinstance(professor_fragment, ProfessorLLMFragment):
             logger.error("[A3X Bridge Handler - Reflect Resp] Professor fragment 'prof_geral' not found or not a ProfessorLLMFragment.")
             return { "status": "error", "message": "Professor LLM not available for reflection." }

        llm_response_a3l = await professor_fragment.ask_llm(prompt)

        if not llm_response_a3l:
             logger.warning(f"[A3X Bridge Handler - Reflect Resp] Professor returned empty response for reflection on '{fragment_id}'.")
             return { "status": "success", "message": "Reflection requested, but Professor provided no suggestions." }

        logger.info(f"[A3X Bridge Handler - Reflect Resp] Professor suggested A3L: {llm_response_a3l}")

        if post_message_handler:
            reflection_directive = {
                "type": "interpret_text",
                "text": llm_response_a3l,
                "origin": f"Reflection on {fragment_id}"
            }
            await post_message_handler(
                message_type="a3l_directive", 
                content=reflection_directive, 
                target_fragment="ki_main_instance"
            )
            logger.info(f"[A3X Bridge Handler - Reflect Resp] Sent Professor's suggestions to KI.")
            return { "status": "success", "message": "Reflection suggestions sent to KI." }
        else:
             logger.error("[A3X Bridge Handler - Reflect Resp] post_message_handler not available.")
             return { "status": "error", "message": "Cannot process reflection suggestions: message handler missing." }

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Reflect Resp] Error handling reflection for '{fragment_id}': {e}", exc_info=True)
        return { "status": "error", "message": f"Reflection failed: {e}" } 