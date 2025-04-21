#!/usr/bin/env python
# coding: utf-8

import logging
from typing import Dict, Any, Optional

# Import necessary functions/classes from other modules
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
from a3x.a3net.integration.a3x_bridge import handle_directive
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from .utils import append_to_log, _log_result

logger = logging.getLogger(__name__)

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