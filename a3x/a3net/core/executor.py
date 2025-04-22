# Main execution logic for A3L files

import sys
from pathlib import Path
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import asyncio

# Updated import paths relative to project structure
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.context_store import ContextStore
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment # Import ProfessorLLMFragment

# Import from sibling modules
from .utils import (
    get_cumulative_epochs,
    generate_symbolic_suggestion,
    analisar_reflexao_e_sugerir_criacao,
    append_to_log,
    _log_result,
    avaliar_fragmento_criado, # Although avaliar isn't directly called in run_a3l, specialization_loop might use it indirectly via suggestions.
    OUTPUT_LOG_FILE
)
from .specialization_loop import iniciar_ciclo_especializacao
# Import the new function for handling reflection interpretation
from .knowledge_interpreter_executor import interpret_and_execute_reflection
# Import the refactored reflect_fragment
from .specialization_loop import reflect_fragment

# --- Adicionar import de ContextStore --- (Ajustar caminho se necessário)
try:
    from a3x.core.context.context_store import ContextStore
except ImportError:
    ContextStore = None 

logger = logging.getLogger(__name__)

# Globals
LAST_A3L_LOG_FILE = "last_run.a3l.log"
knowledge_interpreter = None # Global KI instance for the run
last_problematic_input = None # Store problematic input for specialization

def _execute_extracted_commands(
    extracted_commands: List[str],
    results_summary: Dict[str, int],
    line_num: int,
    origin_id: str, # ID of fragment that produced the text (professor or reflected fragment)
    log_prefix_tag: str
):
    """Helper function to interpret and execute a list of extracted command strings."""
    if not extracted_commands:
        print(f"[Line {line_num} {log_prefix_tag}] Nenhum comando A3L extraído para {origin_id}.")
        return

    num_extracted = len(extracted_commands)
    logging.info(f"--- Iniciando execução de {num_extracted} comandos A3L ({log_prefix_tag} para {origin_id}) --- ")
    append_to_log(f"# [{log_prefix_tag}] Iniciando execução de {num_extracted} comandos extraídos de {origin_id}.")

    for idx, cmd_str in enumerate(extracted_commands):
        current_cmd_num = idx + 1
        exec_log_prefix = f"[Line {line_num} {log_prefix_tag} {current_cmd_num}/{num_extracted}]"
        file_log_prefix = f"# [{log_prefix_tag} {current_cmd_num}/{num_extracted}]"
        print(f"{exec_log_prefix} Executando: {cmd_str}")
        try:
            interpreted_cmd_directive = interpret_a3l_line(cmd_str)
            
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
                    results_summary["failed"] += 1 # Count as failure if not comment/empty
                    print(f"{exec_log_prefix} Falha ao interpretar comando extraído: {cmd_str}")
                    append_to_log(f"{file_log_prefix} [FALHA Interpretação] {cmd_str}")
                # else: # Don't log comments/empty lines extracted from text? Or maybe:
                #     append_to_log(f"{file_log_prefix} [Comentário/Vazio]") 

        except Exception as exec_err:
            logging.error(f"Erro EXCEPCIONAL ao executar comando ({log_prefix_tag}) '{cmd_str}': {exec_err}", exc_info=True)
            results_summary["failed"] += 1
            append_to_log(f"{file_log_prefix} [FALHA EXECUÇÃO] {exec_err}")
            
    logging.info(f"--- Fim da execução de comandos A3L ({log_prefix_tag} para {origin_id}) ---")
    append_to_log(f"# [{log_prefix_tag}] Fim da execução de comandos extraídos de {origin_id}.")

# --- NEW: Helper function for interpreting and executing reflection/professor text ---
def interpret_and_execute_reflection(
    reflection_text: str,
    knowledge_interpreter: Optional[KnowledgeInterpreterFragment],
    results_summary: Dict[str, int],
    line_num: int,
    fragment_id: Optional[str], # ID of the fragment that produced the text (professor or reflected fragment)
    log_prefix_tag: str # e.g., "Auto-Interpretado Direto", "Interpretado do Professor"
):
    """
    Interprets text (from reflection or professor) using the KI and executes extracted commands.
    Updates results_summary and logs actions.
    """
    if not knowledge_interpreter:
        logging.warning(f"{log_prefix_tag}: KnowledgeInterpreterFragment not available for fragment ID {fragment_id or 'N/A'}.")
        append_to_log(f"# [AVISO {log_prefix_tag}] Interpretador indisponível para texto de {fragment_id or 'N/A'}.")
        # Optionally increment skipped or add a specific counter if needed
        # results_summary["skipped"] += 1 # Or maybe not, as the primary action (reflection/ask) succeeded
        return # Cannot proceed without KI

    frag_id_display = fragment_id if fragment_id else "?"
    print(f"[Line {line_num} {log_prefix_tag}] Interpretando texto de {frag_id_display}...")
    try:
        extracted_commands, _ = knowledge_interpreter.interpret_knowledge(
            reflection_text,
            context_fragment_id=fragment_id # Pass context ID
        )
        if not extracted_commands:
             append_to_log(f"# [{log_prefix_tag}] Nenhum comando A3L extraído do texto de {frag_id_display}.")
             return # Nothing to execute

        # Call the command execution helper
        _execute_extracted_commands(
            extracted_commands=extracted_commands,
            results_summary=results_summary,
            line_num=line_num, # Use original line number for context
            origin_id=frag_id_display,
            log_prefix_tag=log_prefix_tag
        )
    except Exception as interpret_err:
        logging.error(f"Erro ao usar KnowledgeInterpreterFragment para processar texto ({log_prefix_tag}): {interpret_err}", exc_info=True)
        append_to_log(f"# [FALHA KI {log_prefix_tag}] Erro ao interpretar texto de {frag_id_display}: {interpret_err}")
        results_summary["failed"] += 1 # Count interpretation failure
# --- END NEW FUNCTION ---

async def run_a3l_file(a3l_filepath: str, context_store: Optional[ContextStore] = None):
    """Reads, interprets, and executes commands from an A3L file."""
    a3l_path = Path(a3l_filepath)
    start_time = datetime.now()
    if not a3l_path.exists() or not a3l_path.is_file():
        print(f"Error: Cannot find A3L file: {a3l_filepath}")
        logger.error(f"Cannot find A3L file: {a3l_filepath}")
        return

    # --- Clear/Initialize Log File --- (Uses OUTPUT_LOG_FILE from utils)
    try:
        with open(OUTPUT_LOG_FILE, "w") as f:
            f.write(f"# Symbolic Log for A3L Execution: {a3l_filepath}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
        print(f"[Session Log] Initialized log file: {OUTPUT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize output log file {OUTPUT_LOG_FILE}: {e}")
        return # Stop if log cannot be initialized

    print(f"--- Running A3L Script: {a3l_filepath} ---")

    results_summary = {"success": 0, "failed": 0, "skipped": 0, "unrecognized": 0}
    # fragment_cache is handled internally by the bridge now

    # --- Instanciar o Interpretador de Conhecimento --- FIX:
    try:
        # Provide default ID and description
        knowledge_interpreter = KnowledgeInterpreterFragment(
            fragment_id="ki_default", 
            description="Default Knowledge Interpreter Fragment"
        )
        logging.info("Instância de KnowledgeInterpreterFragment criada com sucesso.")
    except Exception as e:
        logging.error(f"Falha ao instanciar KnowledgeInterpreterFragment: {e}")
        knowledge_interpreter = None # Continuar sem interpretação de reflexão se falhar
    # --- Fim da Instanciação ---

    # Armazena a última entrada que causou baixa confiança
    last_problematic_input: Optional[List[float]] = None

    # State for multi-line command handling
    current_command_buffer = ""
    in_multiline_ask = False

    with open(a3l_filepath, 'r') as f:
        lines = f.readlines() # Read all lines at once

    line_num = 0
    # --- Armazenar resultados por comando ---
    command_results_log: List[Dict[str, Any]] = []

    while line_num < len(lines):
        i = line_num # Use 'i' for the raw line index
        line = lines[i]
        line_num += 1 # Increment effective line number for user feedback

        original_line = line.strip() # Keep original for logging/potential errors
        result = None # Inicializar result para evitar UnboundLocalError

        print(f"\n[Line {line_num}] Processing: {original_line}")

        line_to_interpret = None
        # --- Multi-line 'ask' handling ---
        if in_multiline_ask:
            current_command_buffer += line # Concatenate raw lines
            if line.strip().endswith("]"):
                in_multiline_ask = False # Found the end
                line_to_interpret = current_command_buffer
                current_command_buffer = "" # Reset buffer
                print(f"[Line {line_num}] Completed multi-line command.")
            else:
                print(f"[Line {line_num}] Buffering multi-line command...")
                continue # Read next line to continue buffering
        else:
            # Check if this line starts an 'ask' command that might be multi-line
            stripped_line = original_line # Already stripped
            if re.match(r"^perguntar\\s+ao\\s+fragmento\\s+\'([^\']+)\'\\s+com\\s+\[", stripped_line, re.IGNORECASE) and not stripped_line.endswith("]"):
                in_multiline_ask = True
                current_command_buffer = line # Start buffer with raw line
                print(f"[Line {line_num}] Starting multi-line command buffer...")
                continue # Start buffering
            else:
                 line_to_interpret = line # Process single raw line
        # --- End multi-line handling ---

        command_start_time = datetime.now()
        command_status = "unrecognized" # Default status
        command_error = None
        executed_directive = None # Store the interpreted directive

        try:
            # Interpret the potentially multi-line command
            directive = interpret_a3l_line(line_to_interpret)
            executed_directive = directive # Store for logging

            # --- Process ONLY if directive is valid ---
            if directive:
                print(f"[Line {line_num}] Interpreted Directive: {directive}")
                directive_type = directive.get("type") # Get type here
                
                # FIX: Initialize original_frag_id to handle UnboundLocalError
                original_frag_id = directive.get("fragment_id") # Get it from the main directive if available

                # --- Handle Cumulative Epoch Conditional ---
                if directive_type == "cumulative_epochs_conditional":
                    condition = directive["condition"]
                    action_directive = directive["action"]
                    cond_fragment_id = condition["fragment_id"]
                    min_epochs = condition["min_epochs"]

                    print(f"[Line {line_num}] Evaluating epoch condition for '{cond_fragment_id}' > {min_epochs} epochs...")
                    # Get cumulative epochs just-in-time (using function from utils)
                    cumulative_epochs_val = get_cumulative_epochs(cond_fragment_id, OUTPUT_LOG_FILE)

                    if cumulative_epochs_val > min_epochs:
                        print(f"[Line {line_num}] Epoch condition MET ({cumulative_epochs_val} > {min_epochs}). Executing nested action: {action_directive['type']}")
                        # Execute the nested action directive using the bridge
                        result = handle_directive(action_directive)
                        print(f"[Line {line_num}] Nested Action Result: {result}")

                        # Process nested result
                        if result and result.get("status") == "success":
                            results_summary["success"] += 1
                            _log_result(action_directive, result) # Log success of nested action

                            # --- Suggestion Cycle for NESTED ASK --- (Uses function from utils)
                            if action_directive.get("type") == "ask":
                                # Use the fragment_id from the nested action or condition
                                nested_frag_id = action_directive.get("fragment_id") or cond_fragment_id 
                                confidence = result.get("confidence")
                                if nested_frag_id and confidence is not None:
                                    suggestion = generate_symbolic_suggestion(nested_frag_id, confidence)
                                    if suggestion:
                                        # (Only log nested suggestion for now, no auto-execution)
                                        print(f"[Line {line_num} Nested Suggestion] Generated: {suggestion}")
                                        append_to_log(f"# Sugestão (Aninhada): {suggestion}")
                                    # --- End Nested Suggestion Handling ---
                            # --- End Nested Suggestion Cycle ---

                        elif result and result.get("status") == "skipped":
                            results_summary["skipped"] += 1
                            reason = result.get('reason', 'Unknown reason')
                            append_to_log(f"# [Ação Aninhada Ignorada] {reason}")
                        else:
                            results_summary["failed"] += 1
                            error_msg = result.get('error', 'Unknown error') if result else 'Null result'
                            append_to_log(f"# [FALHA Aninhada] Erro ao executar: {action_directive.get('type', 'unknown')} -> {error_msg}")
                    else:
                        print(f"[Line {line_num}] Epoch condition FALSE ({cumulative_epochs_val} <= {min_epochs}). Skipping action: {action_directive['type']}")
                        results_summary["skipped"] += 1
                        append_to_log(f"# [Condição de Época FALSA] Ação ignorada: {action_directive.get('type', 'unknown')}")

                # --- Handle 'ask_professor' type --- 
                elif directive_type == "ask_professor":
                    professor_id = directive.get("professor_id")
                    question = directive.get("question")
                    knowledge_interpreter_id = "ki_default" # Assume default KI for now

                    if not professor_id or not question:
                        print(f"[Line {line_num}] Error (ask_professor): Missing professor_id or question.")
                        results_summary["failed"] += 1
                        append_to_log(f"# [FALHA Ask Professor] Diretiva inválida: {directive}")
                        continue # Skip to next line

                    print(f"[Line {line_num}] Asking Professor '{professor_id}' question: '{question[:60]}...'")
                    
                    # 1. Load Professor Fragment
                    professor_fragment = MEMORY_BANK.load(professor_id)
                    if not isinstance(professor_fragment, ProfessorLLMFragment):
                        print(f"[Line {line_num}] Error (ask_professor): Fragment '{professor_id}' is not a ProfessorLLMFragment.")
                        results_summary["failed"] += 1
                        append_to_log(f"# [FALHA Ask Professor] Fragmento '{professor_id}' não é do tipo ProfessorLLMFragment.")
                        continue
                        
                    # 2. Load Knowledge Interpreter Fragment
                    # Use the instance created at the beginning of run_a3l_file
                    ki_fragment = knowledge_interpreter # Use the pre-initialized instance
                    if not isinstance(ki_fragment, KnowledgeInterpreterFragment):
                        print(f"[Line {line_num}] Error (ask_professor): Knowledge Interpreter '{knowledge_interpreter_id}' not available or invalid.")
                        results_summary["failed"] += 1
                        append_to_log(f"# [FALHA Ask Professor] Interpretador de Conhecimento '{knowledge_interpreter_id}' indisponível.")
                        continue

                    try:
                        # 3. Ask the Professor (Get Text Response)
                        llm_response_text = professor_fragment.ask_llm(question)
                        print(f"[Line {line_num}] Professor '{professor_id}' responded (textual). Interpreting...")
                        append_to_log(f"# [Resposta Professor {professor_id}] " + llm_response_text.replace('\\n', ' \\\\n ')) # Log raw response
                        
                        # 4. Interpret the Response
                        # Use the centralized function
                        interpret_and_execute_reflection(
                            reflection_text=llm_response_text,
                            knowledge_interpreter=ki_fragment, # Pass the instance
                            results_summary=results_summary,
                            line_num=line_num,
                            fragment_id=professor_id, # Log originates from professor
                            log_prefix_tag="Interpretado do Professor"
                        )
                        # Log the original ask_professor as successful *if* interpretation cycle didn't error out immediately
                        # (interpret_and_execute_reflection handles its own sub-task logging)
                        results_summary["success"] += 1 # Count the ask_professor itself as a success
                        append_to_log(f"# [Sucesso Ask Professor] Pergunta enviada a '{professor_id}' e resposta interpretada.")

                    except Exception as e:
                        print(f"[Line {line_num}] Error during ask_professor cycle: {e}")
                        logger.exception(f"Error during ask_professor for ID {professor_id}")
                        results_summary["failed"] += 1
                        append_to_log(f"# [FALHA Ask Professor] Erro no ciclo para '{professor_id}': {e}")
                    
                    # End of ask_professor specific logic

                # --- Handle 'learn_from_professor' type (NEW) ---
                elif directive_type == "learn_from_professor":
                    professor_id = directive.get("professor_id")
                    question = directive.get("question")
                    context_id = directive.get("context_fragment_id") # Optional

                    if not professor_id or not question:
                         print(f"[Line {line_num}] Error (learn): Directive is missing professor_id or question.")
                         results_summary["failed"] += 1
                         append_to_log(f"# [FALHA Learn] Diretiva inválida: {directive}")
                    else:
                        # Call the dedicated function
                        learn_result = await learn_from_professor(
                            professor_id=professor_id,
                            question=question,
                            ki_fragment=knowledge_interpreter, # Pass the KI instance
                            results_summary=results_summary, # Pass the main summary dict to be updated
                            context_store=context_store, # Pass the context store
                            memory_bank=MEMORY_BANK, # Pass the memory_bank
                            fragmento_referido=context_id
                        )
                        # learn_from_professor handles its own logging and summary updates
                        # We might just log the overall outcome here
                        if learn_result.get("status") == "success":
                             # Success count is handled internally by _execute_extracted_commands
                             print(f"[Line {line_num}] Learn cycle completed for professor '{professor_id}'.")
                             append_to_log(f"# [Sucesso Learn] Ciclo de aprendizado com '{professor_id}' concluído.")
                             # Note: results_summary["success"] is incremented *inside* _execute_extracted_commands
                             # and potentially by the learn_from_professor function wrapper if desired.
                             # Avoid double counting here unless learn_from_professor doesn't update.
                             pass 
                        else:
                             # Error/failure logging is handled within learn_from_professor
                             print(f"[Line {line_num}] Learn cycle for professor '{professor_id}' encountered errors.")
                             # results_summary["failed"] is incremented inside learn_from_professor on error
                             pass

                # --- Handle 'interpret_text' type (NOVO) ---
                elif directive_type == "interpret_text":
                    text_to_interpret = directive.get("text")
                    original_a3l_line = directive.get("original_line")
                    command_status = "failed" # Default
                    command_error = None
                    extracted_commands_result = []
                    
                    if knowledge_interpreter and text_to_interpret is not None:
                        print(f"[Line {line_num}] Interpreting text via KnowledgeInterpreter: '{text_to_interpret[:80]}...'")
                        try:
                            # Chama o interpretador
                            extracted_commands, _ = knowledge_interpreter.interpret_knowledge(text_to_interpret)
                            extracted_commands_result = extracted_commands # Guarda para o log
                            
                            if extracted_commands:
                                print(f"[Line {line_num}] KI Extracted: {extracted_commands}")
                                append_to_log(f"# [KI Resultado - Linha {line_num}] Extraído(s): {extracted_commands}")
                                command_status = "success" # A interpretação foi um sucesso
                            else:
                                print(f"[Line {line_num}] KI: Nenhum comando A3L extraído do texto.")
                                append_to_log(f"# [KI Resultado - Linha {line_num}] Nenhum comando extraído.")
                                command_status = "success" # A interpretação foi um sucesso, mas não achou nada
                                
                        except Exception as ki_err:
                            logger.error(f"Erro ao executar KnowledgeInterpreter [Linha {line_num}]: {ki_err}", exc_info=True)
                            command_error = f"KI Error: {ki_err}"
                            append_to_log(f"# [FALHA KI - Linha {line_num}] {command_error}")
                    else:
                        error_msg = "KnowledgeInterpreter não disponível" if not knowledge_interpreter else "Texto para interpretar não encontrado na diretiva"
                        logger.error(f"Falha ao processar interpret_text [Linha {line_num}]: {error_msg}")
                        command_error = error_msg
                        append_to_log(f"# [FALHA interpret_text - Linha {line_num}] {command_error}")
                        
                    # Atualiza o sumário geral (contamos 'interpret_text' como uma ação)
                    if command_status == "success":
                        results_summary["success"] += 1
                    else:
                        results_summary["failed"] += 1
                    
                    # (O log individual do comando será feito no bloco 'finally')
                    # NÃO EXECUTAR os extracted_commands aqui.

                # --- Handle Other Directive Types (via Bridge) ---
                else:
                    # Execute the regular directive using the bridge
                    print(f"[Line {line_num}] Executing directive via bridge...")
                    result = handle_directive(directive)
                    print(f"[Line {line_num}] Execution Result: {result}")

                    # --- Process Result --- 
                    if result:
                        status = result.get("status")
                        if status == "success":
                            results_summary["success"] += 1
                            _log_result(directive, result) # Log the successful action

                            # --- Auto-Interpret DIRECT A3L from Reflection --- 
                            # --- REMOVED THIS BLOCK ---
                            # if directive_type == "reflect_fragment" and directive.get("format") == "a3l" and "reflection_a3l" in result:
                            #     reflection_text = result["reflection_a3l"]
                            #     # Call the centralized function
                            #     interpret_and_execute_reflection(
                            #         reflection_text=reflection_text,
                            #         knowledge_interpreter=knowledge_interpreter, # Use global KI
                            #         results_summary=results_summary,
                            #         line_num=line_num,
                            #         fragment_id=original_frag_id, # Pass the ID of the fragment that reflected
                            #         log_prefix_tag="Auto-Interpretado Direto"
                            #     )
                            # --- END REMOVED BLOCK ---

                            # --- Suggestion Cycle AFTER Direct Reflection Analysis (or just after reflection success) ---
                            # Check if it was a reflection, regardless of format, to potentially trigger creation suggestion
                            if directive_type == "reflect_fragment" and "reflection" in result and original_frag_id:
                                # Analyze the reflection text (might be generic text now, not necessarily A3L)
                                reflection_text_for_analysis = result["reflection"] # Use the general reflection field
                                # --- Suggestion Cycle based on the reflection text --- 
                                creation_suggestion = analisar_reflexao_e_sugerir_criacao(original_frag_id, reflection_text_for_analysis)
                                if creation_suggestion:
                                    print(f"[Line {line_num} Sugestão Criação Pós-Reflexão-Direta] Gerado: {creation_suggestion}")
                                    append_to_log(f"# [Sugestão Criação Pós-Reflexão Direta] {creation_suggestion}")
                                    # --- Execute Suggestion --- 
                                    # print(f"[Line {{line_num}} Sugestão Criação] Interpretando e executando...")
                                    # creation_directive = interpret_a3l_line(creation_suggestion)
                                    # if creation_directive:
                                    #     creation_result = handle_directive(creation_directive)
                                    #     print(f"[Line {{line_num}} Sugestão Criação] Resultado: {{creation_result}}")
                                    #     if creation_result and creation_result.get("status") == "success":
                                    #         results_summary["success"] += 1 # Count creation as success
                                    #         _log_result(creation_directive, creation_result, log_prefix="# [Sugestão Criação Executada] ") # Log creation success
                                    # else:
                                    #     results_summary["failed"] += 1 # Count failed interpretation
                                    #     append_to_log(f"# [FALHA Sugestão Criação] {{creation_result.get('message', 'Erro desconhecido.') if creation_result else 'Resultado Nulo'}}")
                                # --- Fim Execução Sugestão Criação ---
                            # --- Handle non-success results (error, null result) --- 
                        elif status == "skipped":
                            results_summary["skipped"] += 1
                            reason = result.get('reason', 'Unknown reason')
                            append_to_log(f"# [Ação Ignorada] {reason}")
                        else: # Includes 'error' or missing status
                            results_summary["failed"] += 1
                            error_msg = result.get('error', 'Unknown error') if result else 'Null result'
                            append_to_log(f"# [FALHA] Erro ao executar: {original_line} -> {error_msg}")
                    else:
                        results_summary["failed"] += 1 # Null result is a failure
                        append_to_log(f"# [FALHA] Resultado nulo ao executar: {original_line}")

            # --- Handle lines that didn't produce a valid directive --- 
            else:
                # Only log/report if the original line wasn't just whitespace or a comment
                if original_line and not original_line.strip().startswith("#"):
                    print(f"[Line {line_num}] Line not recognized or produced no valid directive: {original_line}")
                    results_summary["unrecognized"] += 1
                    append_to_log(f"# [Não Reconhecido/Inválido] {original_line}")
                # else: # It was whitespace or a comment, do nothing

            # Dentro dos blocos de processamento de diretiva (após a execução):
            # Atualizar command_status e command_error com base no 'result' ou 'learn_result'
            # Exemplo para o bloco 'else' (outras diretivas via bridge):
            if result:
                status = result.get("status")
                if status == "success":
                    command_status = "success"
                    # ... (código existente)
                elif status == "skipped":
                    command_status = "skipped"
                    # ... (código existente)
                else: # Includes 'error' or missing status
                    command_status = "failed"
                    command_error = result.get('error', 'Unknown error')
                    # ... (código existente)
            else:
                command_status = "failed"
                command_error = "Null result from handle_directive"
                # ... (código existente)
                
            # Similar updates needed inside the handlers for cumulative_epochs, ask_professor, learn_from_professor
            # For simplicity, we assume those functions update results_summary correctly, 
            # but we might need to capture their final status here too for the command log.
            # Let's assume for now the main status reflects the overall attempt for the line.

        except Exception as loop_err:
            logger.error(f"Erro no loop principal do Executor [Linha {line_num}]: {loop_err}", exc_info=True)
            command_status = "executor_exception"
            command_error = str(loop_err)
            results_summary["failed"] += 1 # Count as failure
            append_to_log(f"# [FALHA Executor Loop] Linha {line_num}: {loop_err}")

        finally:
            # --- Log do Comando Individual ---
            if command_status not in ["comment_or_empty"]:
                command_end_time = datetime.now()
                command_log_entry = {
                    "a3l_file": a3l_filepath,
                    "line_number": line_num,
                    "line_content": original_line,
                    "interpreted_directive": executed_directive,
                    "status": command_status,
                    "error": command_error,
                    "start_time": command_start_time.isoformat(),
                    "end_time": command_end_time.isoformat(),
                    "duration_ms": (command_end_time - command_start_time).total_seconds() * 1000
                }
                command_results_log.append(command_log_entry)
            # --- Fim Log Comando ---

    # End of loop
    end_time = datetime.now()
    print(f"\n--- A3L Script Finished: {a3l_filepath} --- ")
    print(f"Summary: Success={results_summary['success']}, Failed={results_summary['failed']}, Skipped={results_summary['skipped']}, Unrecognized={results_summary['unrecognized']}")
    
    # --- Salvar Histórico de Comandos na ContextStore --- 
    if context_store:
        try:
            # Usar o nome do arquivo A3L como chave (ou parte dela)
            log_key = f"a3l_execution_history:{a3l_path.stem}_{start_time.strftime('%Y%m%d%H%M%S')}"
            await context_store.set(log_key, {
                "a3l_file": a3l_filepath,
                "overall_summary": results_summary,
                "script_start_time": start_time.isoformat(),
                "script_end_time": end_time.isoformat(),
                "script_duration_ms": (end_time - start_time).total_seconds() * 1000,
                "commands": command_results_log
            })
            logger.info(f"Histórico de execução A3L salvo na ContextStore com chave: {log_key}")
        except Exception as store_err:
            logger.error(f"Falha ao salvar histórico de execução A3L na ContextStore: {store_err}", exc_info=True)
    else:
        logger.warning("ContextStore não fornecida, histórico de execução A3L não será salvo.")
    # --- Fim Salvar Histórico ---

# Note: The main execution block (if __name__ == '__main__') should remain in the original script 
# (e.g., examples/interpret_a3l.py) that calls this function. 

# <<< NEW FUNCTION >>>
async def learn_from_professor(
    professor_id: str,
    question: str,
    ki_fragment: Optional[KnowledgeInterpreterFragment],
    results_summary: Dict[str, int],
    context_store: Optional[ContextStore], # <<< ADDED context_store parameter
    memory_bank: MemoryBank,
    fragmento_referido: Optional[str] = None 
) -> Dict[str, Any]:
    """Handles the full cycle of asking a professor, interpreting the response, and logging suggestions."""
    
    cycle_summary = {"asked_professor": False, "got_response": False, "commands_extracted": 0, "commands_attempted": 0}
    
    print(f"-- Iniciando Ciclo Learn from Professor '{professor_id}' ---")
    append_to_log(f"# [Learn Cycle] Iniciado para Professor '{professor_id}', Contexto='{fragmento_referido or 'Nenhum'}'. Pergunta: {question}")

    # --- 1. Load Professor --- 
    professor_fragment = MEMORY_BANK.load(professor_id)
    if not isinstance(professor_fragment, ProfessorLLMFragment):
        print(f"[Learn Cycle] Erro: Fragmento '{professor_id}' não é um ProfessorLLMFragment.")
        append_to_log(f"# [Learn Cycle FALHA] Fragmento '{professor_id}' não é ProfessorLLMFragment.")
        results_summary["failed"] += 1 # Count as failure
        return {**cycle_summary, "status": "error", "message": f"Fragment '{professor_id}' is not ProfessorLLMFragment."}

    # --- 2. Check Knowledge Interpreter --- 
    if not ki_fragment:
        print(f"[Learn Cycle] Erro: Knowledge Interpreter não está disponível.")
        append_to_log(f"# [Learn Cycle FALHA] Knowledge Interpreter indisponível.")
        results_summary["failed"] += 1 # Count as failure
        return {**cycle_summary, "status": "error", "message": "Knowledge Interpreter not available."}

    # --- 3. Ask Professor --- 
    try:
        cycle_summary["asked_professor"] = True
        llm_response_text = professor_fragment.ask_llm(question)
        cycle_summary["got_response"] = True
        print(f"[Learn Cycle] Resposta recebida do Professor '{professor_id}'. Interpretando...")
        append_to_log(f"# [Learn Cycle Resposta Prof {professor_id}] " + llm_response_text.replace('\n', ' \\n '))
    except Exception as e:
        print(f"[Learn Cycle] Erro ao consultar Professor '{professor_id}': {e}")
        logger.exception(f"Erro ao consultar Professor {professor_id}")
        append_to_log(f"# [Learn Cycle FALHA] Erro ao consultar Professor '{professor_id}': {e}")
        results_summary["failed"] += 1 # Count as failure
        return {**cycle_summary, "status": "error", "message": f"Error asking professor: {e}"}

    # --- 4. Interpret Response --- 
    try:
        extracted_commands, _ = ki_fragment.interpret_knowledge(
            llm_response_text, 
            context_fragment_id=fragmento_referido # Pass the context
        )
        cycle_summary["commands_extracted"] = len(extracted_commands)
    except Exception as e:
        print(f"[Learn Cycle] Erro ao interpretar resposta com KI '{ki_fragment.fragment_id}': {e}")
        logger.exception(f"Erro ao interpretar resposta do Professor {professor_id} com KI {ki_fragment.fragment_id}")
        append_to_log(f"# [Learn Cycle FALHA KI] Erro ao interpretar resposta de {professor_id}: {e}")
        results_summary["failed"] += 1 # Count as failure
        return {**cycle_summary, "status": "error", "message": f"Error interpreting response: {e}"}
        
    # --- 5. Log Extracted Commands as Suggestions --- (NOVO)
    if extracted_commands:
        print(f"[Learn Cycle] Comandos A3L extraídos da resposta do Professor '{professor_id}': {extracted_commands}")
        append_to_log(f"# [Sugestão KI via Professor {professor_id}] Extraído(s): {extracted_commands}. Aguardando comando A3L.")
        # Salvar cada comando extraído como uma sugestão individual na store
        if context_store: # <<< Now we can safely check and use context_store
            for cmd in extracted_commands:
                # Check if context_store has the 'push' method before calling
                if hasattr(context_store, 'push') and callable(getattr(context_store, 'push')):
                    try:
                        await context_store.push("pending_suggestions", {
                            "command": cmd,
                            "reason": f"Extraído da resposta do Professor '{professor_id}' para a pergunta: '{question[:50]}...'",
                            "source": f"KI via {professor_id}"
                        })
                    except Exception as push_err:
                         logger.error(f"[Learn Cycle] Falha ao enviar sugestão para ContextStore: {push_err}", exc_info=True)
                         append_to_log(f"# [Learn Cycle FALHA Store] Falha ao salvar sugestão: {push_err}")
                else:
                    logger.warning("[Learn Cycle] ContextStore fornecida não possui método 'push' para salvar sugestão.")
                    append_to_log("# [Learn Cycle AVISO Store] ContextStore sem método 'push'.")

    else:
        print(f"[Learn Cycle] Nenhum comando A3L extraído da resposta do Professor '{professor_id}'.")
        append_to_log(f"# [Sugestão KI via Professor {professor_id}] Nenhum comando extraído.")

    print(f"--- Fim do Ciclo Learn from Professor '{professor_id}' --- ")
    append_to_log(f"# [Learn Cycle] Fim para Professor '{professor_id}'.")
    
    # Return summary specific to this cycle - success means consultation and interpretation worked
    return {**cycle_summary, "status": "success", "extracted_commands": extracted_commands} # Inclui comandos para possível uso externo 