import logging
from typing import Dict, Optional, Any, Callable, Awaitable

# Assuming these are available in the environment
from ..core.professor_llm_fragment import ProfessorLLMFragment
from ..core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from ..a3lang_interpreter import interpret_a3l_line # Assuming accessible

logger = logging.getLogger(__name__)

async def handle_resolve_question(
    directive: Dict[str, Any],
    fragment_instances: Optional[Dict[str, Any]],
    post_message_handler: Optional[Callable[..., Awaitable[None]]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'resolver_pergunta' directive logic."""

    question_text = directive.get("question")
    origin = directive.get("_origin", "Unknown Resolver Origin")

    if not question_text:
        logger.error("[A3X Bridge Handler - Resolver] 'question' missing.")
        return {"status": "error", "message": "'question' missing for resolver_pergunta"}

    if not post_message_handler:
         logger.error("[A3X Bridge Handler - Resolver] post_message_handler not available.")
         return {"status": "error", "message": "post_message_handler missing"}

    if fragment_instances is None:
        logger.error("[A3X Bridge Handler - Resolver] fragment_instances not available.")
        return {"status": "error", "message": "fragment_instances unavailable"}
    
    professor_fragment = fragment_instances.get("prof_geral") 
    ki_fragment = fragment_instances.get("ki_main_instance") # Use the known KI instance ID
    
    if not isinstance(professor_fragment, ProfessorLLMFragment):
        logger.error("[A3X Bridge Handler - Resolver] ProfessorLLMFragment ('prof_geral') not found or invalid.")
        return {"status": "error", "message": "ProfessorLLMFragment not available"}
    
    if not isinstance(ki_fragment, KnowledgeInterpreterFragment):
         logger.error("[A3X Bridge Handler - Resolver] KnowledgeInterpreterFragment ('ki_main_instance') not found or invalid.")
         return {"status": "error", "message": "KnowledgeInterpreterFragment not available"}

    logger.info(f"[A3X Bridge Handler - Resolver] Attempting to resolve question from {origin}: '{question_text[:100]}...'")

    try:
        prompt = (
            f"Preciso de ajuda com a seguinte questão/dúvida que surgiu internamente: '{question_text}'\n\n"
            f"Por favor, forneça uma explicação clara OU, preferencialmente, uma sequência de comandos `.a3l` que abordem esta questão ou me ensinem a lidar com ela.\n"
            f"Sua resposta será processada por um interpretador para extrair comandos `.a3l`.\n"
            f"Se fornecer comandos `.a3l`, siga as convenções:\n"
            f"- Ações no infinitivo: `criar fragmento`, `treinar fragmento`, etc.\n"
            f"- Nomes entre aspas simples: `'nome_fragmento'`\n"
            f"- Parâmetros explícitos: `tipo 'neural'`, `por 10 épocas`\n"
            f"- Sem explicações adicionais se fornecer apenas A3L.\n\n"
            f"Resposta:"
        )
        
        professor_response = await professor_fragment.ask_llm(prompt)
        logger.info(f"[A3X Bridge Handler - Resolver] Professor response: {professor_response[:200]}...")

        if not professor_response or not isinstance(professor_response, str):
             logger.warning("[A3X Bridge Handler - Resolver] Professor returned empty or invalid response.")
             return {"status": "error", "message": "Professor returned empty/invalid response"}

        extracted_commands, ki_metadata = await ki_fragment.interpret_knowledge(professor_response)
        ki_source = ki_metadata.get("source", "KI Resolver")

        if "pergunta_pendente" in ki_metadata:
             logger.warning(f"[A3X Bridge Handler - Resolver] KI generated a NEW pending question. Discarding to prevent loops. Original: '{question_text[:100]}...'")

        if extracted_commands:
            logger.info(f"[A3X Bridge Handler - Resolver] KI ({ki_source}) extracted {len(extracted_commands)} commands. Enqueuing...")
            for cmd_index, cmd_str in enumerate(extracted_commands):
                try:
                    reinterpreted_directive = interpret_a3l_line(cmd_str)
                    if reinterpreted_directive:
                        reinterpreted_directive["_origin"] = f"Resolved by KI ({ki_source}) from Prof via {origin}, step {cmd_index+1}"
                        await post_message_handler(
                            message_type="a3l_command",
                            content=reinterpreted_directive,
                            target_fragment="Executor"
                        )
                        logger.info(f"[A3X Bridge Handler - Resolver] Command enqueued: {cmd_str}")
                    else:
                        logger.warning(f"[A3X Bridge Handler - Resolver] Failed to re-interpret resolved command: {cmd_str}")
                except Exception as reinterpr_err:
                    logger.error(f"[A3X Bridge Handler - Resolver] Error re-interpreting/enqueuing command '{cmd_str}': {reinterpr_err}", exc_info=True)
            
            return {"status": "success", "message": f"Resolved question, {len(extracted_commands)} commands enqueued."}
        else:
             logger.warning(f"[A3X Bridge Handler - Resolver] KI ({ki_source}) extracted no commands from Professor's response for question: '{question_text[:100]}...'")
             return {"status": "warning", "message": "KI extracted no commands from Professor's response."}

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Resolver] Exception during question resolution: {e}", exc_info=True)
        return {"status": "error", "message": f"Resolver loop failed: {e}"} 