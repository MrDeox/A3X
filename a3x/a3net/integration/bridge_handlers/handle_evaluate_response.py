import logging
from typing import Dict, Optional, Any, Callable, Awaitable

# Assuming these are available in the environment
from ..core.professor_llm_fragment import ProfessorLLMFragment
from ..core.context_store import ContextStore

logger = logging.getLogger(__name__)

async def handle_evaluate_response(
    directive: Dict[str, Any],
    fragment_instances: Optional[Dict[str, Any]],
    context_store: Optional[ContextStore],
    post_message_handler: Optional[Callable[..., Awaitable[None]]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'avaliar_resposta' directive logic."""

    fragment_id = directive.get("fragment_id")
    evaluation = directive.get("evaluation") # Already lowercased by interpreter

    if not fragment_id or not evaluation:
        logger.error("[A3X Bridge Handler - Eval Resp] 'fragment_id' or 'evaluation' missing.")
        return { "status": "error", "message": "'fragment_id' and 'evaluation' are required for avaliar_resposta" }

    if evaluation not in ["correta", "incorreta"]:
         logger.error(f"[A3X Bridge Handler - Eval Resp] Invalid evaluation value: '{evaluation}'")
         return { "status": "error", "message": "Evaluation must be 'correta' or 'incorreta'" }

    if not context_store:
        logger.error("[A3X Bridge Handler - Eval Resp] ContextStore not available.")
        return { "status": "error", "message": "ContextStore is required for avaliar_resposta" }

    try:
        feedback_key = f"feedback_pendente:{fragment_id}"
        pending_task_id = await context_store.get(feedback_key)

        if not pending_task_id:
            logger.warning(f"[A3X Bridge Handler - Eval Resp] No pending feedback for '{fragment_id}' (key: {feedback_key}). Ignoring.")
            return { "status": "success", "message": f"No pending feedback to evaluate for fragment '{fragment_id}'." }

        await context_store.delete(feedback_key)
        logger.info(f"[A3X Bridge Handler - Eval Resp] Cleared pending feedback flag for '{fragment_id}' (task: {pending_task_id}).")
        
        last_ask_key = f"last_ask_result:{fragment_id}"
        last_ask_data = await context_store.get(last_ask_key)
        
        if not last_ask_data:
            logger.error(f"[A3X Bridge Handler - Eval Resp] Missing last ask context for '{fragment_id}' (key: {last_ask_key}).")
        
        if evaluation == "correta":
            logger.info(f"[A3X Bridge Handler - Eval Resp] Response for '{fragment_id}' (task: {pending_task_id}) marked CORRECT.")
            return { "status": "success", "message": f"Feedback '{evaluation}' registered for fragment '{fragment_id}'." }
        
        elif evaluation == "incorreta":
            logger.info(f"[A3X Bridge Handler - Eval Resp] Response for '{fragment_id}' (task: {pending_task_id}) marked INCORRECT. Asking Professor.")
            
            if not last_ask_data:
                 logger.warning(f"[A3X Bridge Handler - Eval Resp] Cannot ask Professor; original context for task {pending_task_id} missing.")
                 return { "status": "success", "message": f"Feedback '{evaluation}' registered, but cannot request correction (missing context)." }
                 
            input_val = last_ask_data.get('input_value', '[Input not recorded]')
            output_val = last_ask_data.get('output', '[Output not recorded]')
            input_str = f'{repr(input_val)}' if isinstance(input_val, list) else f'"{input_val}"'

            prompt = (
                f"A resposta '{output_val}' do fragmento '{fragment_id}' para a entrada {input_str} "
                f"foi avaliada como INCORRETA.\n\n"
                f"Sugira comandos A3L (como 'treinar' ou 'criar') para corrigir o comportamento do fragmento '{fragment_id}' "
                f"ou para ensinar o conceito correto relacionado à entrada. "
                f"Responda APENAS com os comandos A3L."
            )
            logger.debug(f"[A3X Bridge Handler - Eval Resp] Professor Prompt for correction: {prompt}")

            professor_fragment = fragment_instances.get("prof_geral") if fragment_instances else None
            if not isinstance(professor_fragment, ProfessorLLMFragment):
                logger.error("[A3X Bridge Handler - Eval Resp] Professor fragment 'prof_geral' not found or not a ProfessorLLMFragment.")
                return { "status": "error", "message": "Professor LLM not available for correction suggestions." }

            llm_response_a3l = await professor_fragment.ask_llm(prompt)

            if not llm_response_a3l:
                logger.warning(f"[A3X Bridge Handler - Eval Resp] Professor returned empty response for correction on '{fragment_id}'.")
                return { "status": "success", "message": f"Feedback '{evaluation}' registered, Professor provided no correction suggestions." }
            
            logger.info(f"[A3X Bridge Handler - Eval Resp] Professor suggested corrections (A3L): {llm_response_a3l}")

            if post_message_handler:
                correction_directive = {
                    "type": "interpret_text",
                    "text": llm_response_a3l,
                    "origin": f"Correction for incorrect answer by {fragment_id} (task: {pending_task_id})"
                }
                await post_message_handler(
                    message_type="a3l_directive",
                    content=correction_directive,
                    target_fragment="ki_main_instance"
                )
                logger.info(f"[A3X Bridge Handler - Eval Resp] Sent Professor's correction suggestions to KI.")
                return { "status": "success", "message": "Feedback 'incorreta' registered, correction suggestions sent to KI." }
            else:
                logger.error("[A3X Bridge Handler - Eval Resp] post_message_handler not available.")
                return { "status": "error", "message": "Cannot process correction suggestions: message handler missing." }

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Eval Resp] Error handling evaluation for '{fragment_id}': {e}", exc_info=True)
        if context_store and fragment_id:
            try: await context_store.delete(f"feedback_pendente:{fragment_id}")
            except: pass
        return { "status": "error", "message": f"Evaluation processing failed: {e}" } 