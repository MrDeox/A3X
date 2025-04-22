#!/usr/bin/env python
# coding: utf-8

import asyncio
import logging
from typing import Dict, Any, Optional, List

# Import necessary functions/classes from other modules
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.context_store import ContextStore
from .utils import append_to_log, _log_result

logger = logging.getLogger(__name__)

class KnowledgeInterpreterExecutor:
    """ 
    Orchestrates interpretation and execution based on KI results.
    This is a conceptual example; the actual logic lives within run.py's 
    message_processor and the handle_directive function now.
    """
    def __init__(self, 
                 ki_fragment: KnowledgeInterpreterFragment, 
                 memory_bank: MemoryBank,
                 context_store: ContextStore,
                 directive_handler: Optional[callable] = None,
                 post_message_handler: Optional[callable] = None):
        self.ki = ki_fragment
        self.memory_bank = memory_bank
        self.context_store = context_store
        self.directive_handler = directive_handler
        self.post_message_handler = post_message_handler
        logger.info("KnowledgeInterpreterExecutor initialized.")

    async def process_text_and_execute(self, text: str, context_id: Optional[str] = None):
        """Interprets text via KI and executes resulting commands."""
        logger.info(f"KI Executor: Processing text: {text[:100]}...")
        
        commands, metadata = await self.ki.interpret_knowledge(text, context_fragment_id=context_id)
        
        logger.info(f"KI Executor: Interpretation result - Commands: {commands}, Metadata: {metadata}")
        
        if commands:
            logger.info(f"KI Executor: Executing {len(commands)} extracted commands...")
            results = []
            for cmd_str in commands:
                directive_dict = interpret_a3l_line(cmd_str)
                if directive_dict:
                    if self.directive_handler:
                        logger.debug(f"  Executing via passed handler: {directive_dict}")
                        try:
                            result = await self.directive_handler(
                                directive_dict,
                                memory_bank=self.memory_bank,
                                context_store=self.context_store,
                                post_message_handler=self.post_message_handler
                            )
                            results.append(result)
                        except Exception as e:
                            logger.error(f"  Error executing directive '{cmd_str}' via handler: {e}", exc_info=True)
                            results.append({"status": "error", "message": str(e), "command": cmd_str})
                    elif self.post_message_handler:
                        logger.debug(f"  Posting to queue: {directive_dict}")
                        await self.post_message_handler("a3l_command", directive_dict, "Executor")
                        results.append({"status": "posted", "message": "Command posted to queue.", "command": cmd_str})
                    else:
                        logger.error(f"  Cannot execute '{cmd_str}': No directive_handler or post_message_handler available.")
                        results.append({"status": "error", "message": "No execution mechanism.", "command": cmd_str})
                else:
                    logger.warning(f"  Could not interpret extracted command: {cmd_str}")
                    results.append({"status": "error", "message": "Failed to interpret command.", "command": cmd_str})
            return results
        else:
            logger.info("KI Executor: No commands extracted by KI.")
            if metadata.get("prediction"):
                logger.info(f"  Prediction found: {metadata['prediction']}")
            return []

def interpret_and_execute_reflection(
    reflection_text: str,
    knowledge_interpreter: Optional[KnowledgeInterpreterFragment],
    results_summary: Dict[str, int],
    line_num: int, # For context in logging
    fragment_id: str, # For context in logging
    log_prefix_tag: str = "Auto-Interpretado" # E.g., "Auto-Interpretado Direto", "Auto-Interpretado Pós-Sugestão"
):
    """Interprets A3L commands from reflection text and executes them."""

    if not knowledge_interpreter:
        logging.warning(f"{log_prefix_tag}: KnowledgeInterpreterFragment not available for {fragment_id}.")
        append_to_log(f"# [AVISO {log_prefix_tag}] Interpretador indisponível para reflexão de {fragment_id}.")
        return

    frag_id_display = fragment_id if fragment_id else "?"
    print(f"[Line {line_num} {log_prefix_tag}] Interpretando texto da reflexão de {frag_id_display}...")

    try:
        extracted_a3l_commands, _ = knowledge_interpreter.interpret_knowledge(reflection_text)
        num_extracted = len(extracted_a3l_commands)
        logging.info(f"Comandos A3L extraídos ({log_prefix_tag}): {num_extracted} para {frag_id_display}")
        append_to_log(f"# [{log_prefix_tag}] {num_extracted} comandos extraídos da reflexão de {frag_id_display}.")

        if extracted_a3l_commands:
            logging.info(f"--- Iniciando execução de {num_extracted} comandos A3L ({log_prefix_tag}) --- ")
            for idx, cmd_str in enumerate(extracted_a3l_commands):
                current_cmd_num = idx + 1
                exec_log_prefix = f"[Line {line_num} {log_prefix_tag} {current_cmd_num}/{num_extracted}]"
                file_log_prefix = f"# [{log_prefix_tag} {current_cmd_num}/{num_extracted}]"
                print(f"{exec_log_prefix} Executando: {cmd_str}")
                try:
                    interpreted_cmd_directive = interpret_a3l_line(cmd_str)
                    # TODO: Implementar lógica de dependência aqui, se necessário.
                    # Ex: Analisar `interpreted_cmd_directive` e `cmd_result` anterior 
                    # para decidir se o comando atual deve ser executado, modificado ou pulado.
                    if interpreted_cmd_directive:
                        cmd_result = handle_directive(interpreted_cmd_directive)
                        print(f"{exec_log_prefix} Resultado: {cmd_result}")
                        # Processar resultado
                        cmd_status = cmd_result.get("status") if cmd_result else None
                        if cmd_status == "success":
                            results_summary["success"] += 1
                            _log_result(interpreted_cmd_directive, cmd_result, log_prefix=file_log_prefix + " ")
                        elif cmd_status == "skipped":
                            results_summary["skipped"] += 1
                            reason = cmd_result.get('reason', '')
                            append_to_log(f"{file_log_prefix} [Ignorado] {reason}")
                        else:
                            results_summary["failed"] += 1
                            error_msg = cmd_result.get('error', 'Erro desconhecido') if cmd_result else 'Resultado Nulo'
                            append_to_log(f"{file_log_prefix} [FALHA] {error_msg}")
                    else:
                        # Falha na interpretação ou comentário/vazio
                        if cmd_str and not cmd_str.strip().startswith("#"):
                            results_summary["failed"] += 1
                            print(f"{exec_log_prefix} Falha ao interpretar comando extraído: {cmd_str}")
                            append_to_log(f"{file_log_prefix} [FALHA Interpretação] {cmd_str}")
                        else:
                             # Log comments/empty lines minimally
                             append_to_log(f"{file_log_prefix} [Comentário/Vazio]") 

                except Exception as exec_err:
                    logging.error(f"Erro EXCEPCIONAL ao executar comando ({log_prefix_tag}) '{cmd_str}': {exec_err}", exc_info=True)
                    results_summary["failed"] += 1
                    append_to_log(f"{file_log_prefix} [FALHA EXECUÇÃO] {exec_err}")
            logging.info(f"--- Fim da execução de comandos A3L ({log_prefix_tag}) ---")
        else:
            print(f"[Line {line_num} {log_prefix_tag}] Nenhum comando A3L extraído da reflexão de {frag_id_display}.")
            # Log already happened above
    except Exception as interpret_err:
        logging.error(f"Erro ao usar KnowledgeInterpreterFragment para processar reflexão ({log_prefix_tag}): {interpret_err}", exc_info=True)
        append_to_log(f"# [FALHA KI {log_prefix_tag}] Erro ao interpretar reflexão de {frag_id_display}: {interpret_err}") 