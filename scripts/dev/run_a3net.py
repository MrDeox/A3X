import asyncio
import logging
import argparse
from pathlib import Path
import signal # Para lidar com interrupções (Ctrl+C)
import time
from typing import Dict, Any, Callable, Awaitable, Optional, List # <<< Added Optional, List
import json
import os

# --- Importações do Projeto A³X/A³Net ---
# (Ajuste os caminhos exatos conforme a estrutura do seu projeto)
try:
    # Contexto e Memória
    from a3x.a3net.core.context_store import SQLiteContextStore, ContextStore
    from a3x.a3net.core.memory_bank import MemoryBank # <<< CORRECT IMPORT
    # Executor
    from a3x.a3net.modules import executor
    # Fragmentos Principais
    from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
    from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment, LLM_ENABLED # Importa flag
    # Removido: Importação não utilizada que causava erro
    # from a3x.a3net.core.planner_fragment import PlannerFragment 
    # Fragmentos Supervisores/Autônomos
    from a3x.a3net.core.supervisor_fragment import SupervisorFragment
    from a3x.a3net.core.self_critic_fragment import SelfCriticFragment
    from a3x.a3net.core.executor_supervisor_fragment import ExecutorSupervisorFragment
    from a3x.a3net.core.meta_generator_fragment import MetaGeneratorFragment
    # Importar o novo fragmento autônomo
    from a3x.a3net.fragments.autonomous_self_starter import AutonomousSelfStarterFragment
    # Utilitários (se necessário)
    from a3x.a3net.modules.utils import append_to_log, OUTPUT_LOG_FILE 
    # Importar interpretador A3L
    from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
    # Importar o gerenciador do servidor LLM
    from a3x.a3net.core.llm_server_manager import LLMServerManager
    # Executor - Import specific function
    from a3x.a3net.modules.executor import learn_from_professor # Import learn_from_professor
    # Import the label router
    from a3x.a3net.core.label_router import route_label # <<< Added label router
    # --- IMPORT HANDLERS FOR MOVED handle_directive --- 
    from a3x.a3net.integration.bridge_handlers.handle_train import handle_train_fragment
    from a3x.a3net.integration.bridge_handlers.handle_create import handle_create_fragment
    from a3x.a3net.integration.bridge_handlers.handle_ask import handle_ask 
    from a3x.a3net.integration.bridge_handlers.handle_run_graph import handle_run_graph 
    from a3x.a3net.integration.bridge_handlers.handle_reflect import handle_reflect_fragment 
    from a3x.a3net.integration.bridge_handlers.handle_export import handle_export_fragment 
    from a3x.a3net.integration.bridge_handlers.handle_import import handle_import_fragment 
    from a3x.a3net.integration.bridge_handlers.handle_evaluate import handle_evaluate_fragment
    from a3x.a3net.integration.bridge_handlers.handle_request_examples import handle_request_examples
    from a3x.a3net.integration.bridge_handlers.handle_compare_performance import handle_compare_performance
    from a3x.a3net.integration.bridge_handlers.handle_verify_knowledge import handle_verify_knowledge
    from a3x.a3net.integration.bridge_handlers.handle_resolve_question import handle_resolve_question
    from a3x.a3net.integration.bridge_handlers.handle_reflect_response import handle_reflect_response
    from a3x.a3net.integration.bridge_handlers.handle_evaluate_response import handle_evaluate_response
    from a3x.a3net.integration.bridge_handlers.handle_learn_from_professor import handle_learn_from_professor
    # -----------------------------------------------------
except ImportError as e:
    print(f"Erro de Importação: {e}. Verifique os caminhos e a estrutura do projeto.")
    print("Certifique-se de executar este script a partir do diretório raiz do projeto A3X ou que o PYTHONPATH esteja configurado.")
    exit(1)

# --- Configuração de Logging ---
# (Pode ser configurado de forma mais robusta externamente)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Log para console
        logging.FileHandler(OUTPUT_LOG_FILE, mode='w') # Log para arquivo (sobrescreve)
    ]
)
# Silenciar logs muito verbosos de bibliotecas, se necessário
# logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# <<< FORCE DEBUG LEVEL FOR INTERPRETER LOGGER >>>
logging.getLogger('a3x.a3net.integration.a3lang_interpreter').setLevel(logging.DEBUG)
# <<< END FORCE >>>

# --- Fila/Mecanismo de Mensagens ---
# Fila simples para desacoplar post_chat_message do Executor/Handlers
message_queue = asyncio.Queue(maxsize=100) # Adiciona um tamanho máximo

# --- O handler precisa ser acessível globalmente ou passado --- 
# Definindo fora para ser acessível, ou passar como argumento
async def post_message_handler(message_type, content, target_fragment):
    if message_queue.full():
         logger.warning(f"Message queue cheia! Descartando mensagem para {target_fragment}")
         return
    await message_queue.put({
        "message_type": message_type,
        "content": content,
        "target_fragment": target_fragment
    })
    logger.debug(f"Mensagem postada para {target_fragment} (Type: {message_type})")
# -----------------------------------------------------------

# --- Module-Level State for Last Ask (Needed by handle_directive) ---
LAST_ASK_RESULT: Optional[Dict[str, Any]] = None 
# -------------------------------------------------------------------

# <<< PASTE handle_directive FUNCTION HERE >>>
async def handle_directive(
    directive: Dict, 
    memory_bank: MemoryBank, 
    fragment_instances: Optional[Dict[str, Any]] = None,
    context_store: Optional[ContextStore] = None, 
    post_message_handler: Optional[Callable[..., Awaitable[None]]] = None
) -> Optional[Dict[str, Any]]:
    """Handles directives received from the symbolic A³X system.
    (Moved from a3x_bridge.py to run.py to avoid circular imports)
    
    Supports 'train_fragment', 'run_graph', 'ask', 'export_fragment', 'import_fragment', 
    'reflect_fragment', 'conditional_directive', 'avaliar_fragmento', 'comparar_desempenho' types.
    
    Args:
        directive: The dictionary representing the A3L command.
        memory_bank: The MemoryBank instance.
        fragment_instances: Optional dictionary of currently active fragment instances.
        context_store: Optional instance of the ContextStore for saving evaluation results.
        post_message_handler: Optional async function to post messages back to the queue.

    Returns:
        An optional dictionary containing the status and results of the directive execution.
    """
    # Access and potentially modify the global state
    global LAST_ASK_RESULT # <<< Ensure global state is accessible here
    
    directive_type = directive.get("type")
    goal = directive.get('goal', None) 
    # Use logger from run.py scope
    logger.info(f"[Executor via handle_directive] <<< Handling directive type: '{directive_type}', goal: {goal or 'Not specified'}")
    
    result = None # Initialize result

    # ========================================
    # === Dispatch to Specific Handlers ====
    # ========================================
    if directive_type == "train_fragment":
        result = await handle_train_fragment(directive, memory_bank, context_store)
    elif directive_type == "create_fragment":
        result = await handle_create_fragment(directive, memory_bank)
    elif directive_type == "run_graph":
        result = await handle_run_graph(directive, memory_bank)
    elif directive_type == "ask":
        # Pass LAST_ASK_RESULT state to handle_ask if needed, or let handle_ask manage it
        result = await handle_ask(directive, memory_bank, context_store)
        # Potentially update LAST_ASK_RESULT here based on the result
        if result and result.get('status') == 'success':
            LAST_ASK_RESULT = result # Update global state on successful ask
    elif directive_type == "export_fragment":
        result = await handle_export_fragment(directive, memory_bank)
    elif directive_type == "import_fragment":
        result = await handle_import_fragment(directive, memory_bank)
    elif directive_type == "reflect_fragment":
        result = await handle_reflect_fragment(directive, memory_bank)
    # ... (other direct handler calls: solicitar_exemplos, avaliar_fragmento, etc.) ...
    elif directive_type == "solicitar_exemplos":
        result = await handle_request_examples(directive, fragment_instances)
    elif directive_type == "avaliar_fragmento":
        result = await handle_evaluate_fragment(directive, memory_bank, context_store)
    elif directive_type == "comparar_desempenho":
        result = await handle_compare_performance(directive, context_store, post_message_handler)
    elif directive_type == "verificar_conhecimento":
        result = await handle_verify_knowledge(directive, memory_bank, fragment_instances, context_store)
    elif directive_type == "resolver_pergunta":
        result = await handle_resolve_question(directive, fragment_instances, post_message_handler)
    elif directive_type == "refletir_resposta":
        result = await handle_reflect_response(directive, fragment_instances, context_store, post_message_handler)
    elif directive_type == "avaliar_resposta":
        result = await handle_evaluate_response(directive, fragment_instances, context_store, post_message_handler)
    elif directive_type == "learn_from_professor":
        result = await handle_learn_from_professor(directive, memory_bank, post_message_handler)
        
    # =======================================
    # === Handle 'conditional_directive' ===
    # =======================================
    elif directive_type == "conditional_directive":
        condition = directive.get("condition")
        action_directive = directive.get("action")
        logger.info(f"[Executor - Conditional] Evaluating condition: {condition}")

        condition_met = False
        condition_type = condition.get("condition_type", "attribute_check")
        
        # --- Evaluate Conditions --- 
        if condition_type == "attribute_check":
            # ... (attribute check logic as before, using memory_bank) ...
            cond_fragment_id = condition.get("fragment_id")
            cond_attribute = condition.get("attribute")
            cond_expected_value = condition.get("expected_value")
            if not all([cond_fragment_id, cond_attribute]) or cond_expected_value is None:
                 logger.error(f"Incomplete condition details in conditional directive: {condition}")
                 return { "status": "error", "error": "Incomplete condition in attribute check." }
            try:
                fragment = memory_bank.load(cond_fragment_id)
                if fragment and hasattr(fragment, cond_attribute):
                    actual_value = getattr(fragment, cond_attribute)
                    # Basic comparison for numbers
                    if isinstance(actual_value, (int, float)) and isinstance(cond_expected_value, (int, float)):
                        # Add comparison operator support if needed
                        if actual_value == cond_expected_value: condition_met = True
                    # Add other type comparisons if needed
            except Exception as e:
                 logger.exception(f"Error evaluating attribute condition for {cond_fragment_id}")
        elif condition_type == "confidence_check":
            # ... (confidence check logic as before, using LAST_ASK_RESULT) ...
             threshold = condition.get("threshold")
             comparison = condition.get("comparison", "less_than").lower()
             if threshold is None:
                  logger.error("Missing threshold for confidence_check conditional.")
                  return { "status": "error", "error": "Missing threshold in confidence condition." }
             if LAST_ASK_RESULT and isinstance(LAST_ASK_RESULT.get('confidence'), (int, float)):
                 last_confidence = LAST_ASK_RESULT['confidence']
                 if comparison == "less_than" and last_confidence < threshold: condition_met = True
                 elif comparison == "greater_than" and last_confidence > threshold: condition_met = True
                 # Add other comparisons (<=, >=, ==) if needed
             else:
                 logger.warning(f"Cannot evaluate confidence check: LAST_ASK_RESULT invalid: {LAST_ASK_RESULT}")
        else:
             logger.error(f"Unknown condition type in conditional directive: {condition_type}")
             return { "status": "error", "error": f"Unknown condition type: {condition_type}." }

        # --- Execute Action if Condition Met --- 
        if condition_met:
            logger.info(f"[Executor - Conditional] Condition MET. Executing action: {action_directive.get('type')}")
            # <<< RECURSIVE CALL (now within the same module) >>>
            result = await handle_directive(action_directive, memory_bank, fragment_instances, context_store, post_message_handler)
        else:
            logger.info("[Executor - Conditional] Condition NOT MET. Skipping action.")
            result = {"status": "skipped", "message": "Condition not met"}
            
    # --- Default Handler --- 
    else:
        logger.warning(f"[Executor - handle_directive] Unhandled directive type: {directive_type}")
        result = { "status": "skipped", "message": f"Unhandled directive type: {directive_type}" }

    # --- Log and Return Result --- 
    if result:
        logger.info(f"[Executor - handle_directive] <<< Directive '{directive_type}' completed with status: {result.get('status')}")
    else:
        logger.warning(f"[Executor - handle_directive] <<< Directive '{directive_type}' produced no result.")

    return result
# <<< END OF PASTED handle_directive FUNCTION >>>

# <<< message_processor definition remains here >>>
async def message_processor(
    context_store: ContextStore, 
    memory_bank: MemoryBank, 
    ki_fragment: KnowledgeInterpreterFragment, 
    fragment_instances: Dict[str, Any]
):
    """Processa mensagens da fila, agora chamando handle_directive defined in run.py."""
    logger.info("Message Processor iniciado.")
    while True:
        try:
            message = await message_queue.get()
            logger.debug(f"[MessageProc] Recebido: {message}")
            
            target_fragment_id = message.get("target_fragment")
            msg_type = message.get("message_type")
            content = message.get("content")
            
            logger.debug(f"[MessageProc] Roteando: Target='{target_fragment_id}', Type='{msg_type}'")

            # --- Roteamento Principal de Mensagens ---
            # <<< UPDATED to check for Executor target and a3l_command type >>>
            if target_fragment_id == "Executor" and msg_type == "a3l_command" and isinstance(content, dict):
                directive_dict = content 
                origin_source = content.get("_origin", "Unknown Origin") # Use _origin if available
                directive_type = directive_dict.get("type")
                
                # Log before calling handle_directive
                logger.info(f"[MessageProc -> Executor] Executing directive from '{origin_source}': Type='{directive_type}'")
                
                try:
                    # <<< CALL handle_directive (defined in this run.py file) >>>
                    result = await handle_directive(
                        directive=directive_dict,
                        memory_bank=memory_bank, 
                        fragment_instances=fragment_instances,
                        context_store=context_store,
                        post_message_handler=post_message_handler
                    )
                    
                    status = result.get('status', 'unknown') if result else 'no_result'
                    message_log = result.get('message', 'No message') if result else ''
                    logger.info(f"[MessageProc <- Executor] Directive '{directive_type}' from '{origin_source}' completed. Status: {status}. {message_log}")
                    append_to_log(f"# [EXECUTOR] {origin_source} -> '{directive_type}' = {status}. {message_log}")
                    
                    # --- Handle Error for Self-Correction --- 
                    if result and status == "error":
                        error_message = result.get("message", "Unknown error during directive execution.")
                        # Try to reconstruct the original line if possible, otherwise use the dict
                        failed_directive_info = directive_dict.get("_raw_line", json.dumps(directive_dict))
                        logger.warning(f"[MessageProc] Error detected executing directive from {origin_source}: {error_message}. Triggering self-correction...")
                        
                        # Find Autonomous Starter
                        starter = fragment_instances.get("autonomous_starter") 
                        if isinstance(starter, AutonomousSelfStarterFragment):
                            logger.info(f"[MessageProc] Sending error to AutonomousSelfStarterFragment...")
                            # Simplify error? Assume handle_execution_error does it or pass raw?
                            # Passing raw error message for now.
                            await starter.handle_execution_error(error_message, failed_directive_info)
                        else:
                            logger.error(f"[MessageProc] AutonomousSelfStarterFragment not found to handle execution error.")
                    # ------------------------------------------
                            
                except Exception as exec_err:
                    logger.error(f"[MessageProc] Unhandled exception during handle_directive call for '{directive_type}' from '{origin_source}': {exec_err}", exc_info=True)
                    append_to_log(f"# [FALHA EXECUTOR] Erro inesperado em '{directive_type}': {exec_err}")
                    # Optionally trigger self-correction here too?
            
            # --- Roteamento para outros fragmentos (se necessário) ---
            elif target_fragment_id and target_fragment_id != "Executor":
                target_instance = fragment_instances.get(target_fragment_id)
                if target_instance and hasattr(target_instance, 'handle_message'):
                    logger.debug(f"[MessageProc] Forwarding message (type: {msg_type}) to fragment '{target_fragment_id}'...")
                    try:
                        # Assume handle_message is async
                        await target_instance.handle_message(message_type=msg_type, content=content)
                    except Exception as frag_handle_err:
                        logger.error(f"[MessageProc] Error calling handle_message on fragment '{target_fragment_id}': {frag_handle_err}", exc_info=True)
                else:
                    logger.warning(f"[MessageProc] Target fragment '{target_fragment_id}' not found or does not have handle_message method.")

            # --- Mensagens não roteadas --- 
            else:
                logger.warning(f"[MessageProc] Could not route message: Target='{target_fragment_id}', Type='{msg_type}', Content Type: {type(content)}")

            message_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("Message Processor task cancelled.")
            break # Sai do loop while
        except Exception as e:
            logger.error(f"[MessageProc] Unhandled error in message processing loop: {e}", exc_info=True)
            # Evita que o loop pare por um erro inesperado, mas loga
            await asyncio.sleep(1) # Pequena pausa antes de tentar pegar a próxima mensagem

# --- Define the new background task for checking pending questions ---
async def check_pending_questions_task(
    context_store: ContextStore, 
    post_message_handler: Callable[..., Awaitable[None]],
    interval_seconds: int = 15 # Check every 15 seconds
):
    """Periodically checks ContextStore for pending questions and triggers resolution."""
    logger.info("Starting Pending Question Checker Task...")
    while True:
        try:
            logger.debug(f"Checking for pending questions in ContextStore...")
            # Find the first key tagged 'pergunta_pendente'
            pending_keys = await context_store.find_keys_by_tag("pergunta_pendente", limit=1)
            
            if pending_keys:
                key_to_process = pending_keys[0]
                question_text = await context_store.get(key_to_process)
                
                if question_text:
                    logger.info(f"Found pending question (Key: {key_to_process}): '{question_text[:100]}...'. Triggering resolution.")
                    append_to_log(f"# [AUTO-RESOLUÇÃO] Encontrada pergunta pendente. Tentando resolver: {question_text[:100]}...") # Use append_to_log if available globally or pass it

                    # --- Create the 'resolver_pergunta' directive ---
                    # We need to ensure the question text is properly formatted/escaped if needed for A3L
                    # For now, assume it's a simple string argument. A3L parser might need adjustments.
                    # Using simple string interpolation for now. Might need shlex.quote later.
                    resolver_directive_content = f"resolver pergunta '{question_text}'" 
                    
                    # Re-interpret the generated string back into a directive dictionary
                    # We need access to interpret_a3l_line or similar logic here
                    # For now, create the dictionary directly assuming a simple structure
                    # This might need adjustment based on how A3LangInterpreter handles it
                    resolver_directive_dict = {
                        "type": "resolver_pergunta",
                        "question": question_text,
                        "_origin": f"PendingQuestionChecker (Key: {key_to_process})"
                    }

                    # --- Post the directive to the message queue ---
                    await post_message_handler(
                        message_type="a3l_command", 
                        content=resolver_directive_dict,
                        target_fragment="Executor" # Target the main executor flow
                    )
                    logger.info(f"Posted 'resolver_pergunta' directive for key '{key_to_process}'.")

                    # --- IMPORTANT: Remove the processed question from ContextStore ---
                    try:
                        await context_store.delete(key_to_process)
                        logger.info(f"Removed processed pending question key '{key_to_process}' from ContextStore.")
                        append_to_log(f"# [AUTO-RESOLUÇÃO] Pergunta {key_to_process} enviada para resolução e removida.")
                    except Exception as del_err:
                        logger.error(f"Failed to delete processed pending question key '{key_to_process}': {del_err}", exc_info=True)
                        # If deletion fails, we might re-process it later, which isn't ideal but avoids losing the question.
                        append_to_log(f"# [FALHA ContextStore] Erro ao remover pergunta pendente {key_to_process}: {del_err}")

                else:
                    logger.warning(f"Found pending question key '{key_to_process}' but value was empty. Removing.")
                    try:
                        await context_store.delete(key_to_process)
                    except Exception: 
                        pass # Ignore deletion error for empty value

            else:
                logger.debug("No pending questions found.")

            # Wait for the next interval
            await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            logger.info("Pending Question Checker Task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in Pending Question Checker Task: {e}", exc_info=True)
            # Avoid tight loop on persistent errors
            await asyncio.sleep(interval_seconds * 2) 
# ----------------------------------------------------------------------

async def main_loop(a3l_script_path: str):
    """Loop principal que carrega fragmentos, processa A3L e gerencia o ciclo de vida."""
    logger.info("Iniciando main_loop...")
    
    # --- Inicialização Centralizada ---
    context_store = None # Define outside try block
    MEMORY_BANK = None   # Define outside try block
    llm_manager = None   # Define outside try block
    tasks = []           # Define tasks list here to be accessible in finally
    fragment_instances: Dict[str, Any] = {} # Define here for finally block access
    main_task_completed_normally = False # Flag to check if main logic finished
    
    try:
        # --- Initializations (ContextStore, MemoryBank, LLM Manager) ---
        context_store = SQLiteContextStore(db_path="data/databases/a3net/a3net_context.sqlite")
        await context_store.initialize()
        logger.info(f"ContextStore inicializado em {context_store.db_path}")
        
        MEMORY_BANK = MemoryBank(save_dir="models/a3net", export_dir="models/a3net/exported")
        logger.info(f"MemoryBank inicializado (Save: {MEMORY_BANK.save_dir}, Export: {MEMORY_BANK.export_dir})")

        if LLM_ENABLED:
            try:
                 llamafile_path = os.environ.get("LLAMAFILE_PATH")
                 model_path = os.environ.get("LLAMA_MODEL_PATH")
                 if llamafile_path and model_path:
                      logger.info(f"Configurando LLMServerManager com Llamafile: {llamafile_path}, Model: {model_path}")
                      llm_manager = LLMServerManager(
                          llamafile_path=llamafile_path, model_path=model_path,
                          host="127.0.0.1", port=8080
                      )
                 else:
                      logger.warning("LLAMAFILE_PATH ou LLAMA_MODEL_PATH não definidos. LLMServerManager não configurado.")
            except Exception as llm_init_err:
                 logger.error(f"Falha ao configurar LLMServerManager: {llm_init_err}", exc_info=True)
    
    # <<< Separate exception for critical init errors >>>
    except Exception as init_err:
        logger.critical(f"Erro fatal durante inicialização (ContextStore/MemoryBank/LLM): {init_err}", exc_info=True)
        # Ensure cleanup runs even if init fails partially
        if context_store: await context_store.close() # Attempt close
        # No tasks to cancel yet
        return # Stop execution here

    # <<< Start the main try block for fragment loading and execution loop >>>
    try: 
        # Check if essential components initialized
        if not context_store or not MEMORY_BANK:
            logger.critical("Falha ao inicializar ContextStore ou MemoryBank. Encerrando.")
            raise RuntimeError("Essential components (ContextStore or MemoryBank) failed to initialize.") # Raise error to trigger finally

        # --- Carregar/Instanciar Fragmentos Essenciais ---
        # (Professor, KI, Critic, ExecutorSupervisor, AutonomousStarter)
        professor_id = "professor_orientador"
        professor_fragment = ProfessorLLMFragment(
            fragment_id=professor_id,
            llm_url=os.environ.get("LLAMA_API_URL", "http://127.0.0.1:8080/completion"), 
            llm_server_manager=llm_manager,
        )
        fragment_instances[professor_id] = professor_fragment
        logger.info(f"Fragmento ProfessorLLMFragment '{professor_id}' instanciado.")

        ki_id = "knowledge_interpreter_main"
        ki_fragment = KnowledgeInterpreterFragment(
            fragment_id=ki_id, professor_fragment=professor_fragment,
            context_store=context_store, memory_bank=MEMORY_BANK,
            post_chat_message_callback=post_message_handler
        )
        fragment_instances[ki_id] = ki_fragment
        logger.info(f"Fragmento KnowledgeInterpreterFragment '{ki_id}' instanciado.")

        critic_id = "self_critic_evaluator"
        critic_fragment = SelfCriticFragment(
            fragment_id=critic_id,
            professor_fragment=professor_fragment if professor_fragment.is_active else None,
            context_store=context_store, post_chat_message_callback=post_message_handler
        )
        fragment_instances[critic_id] = critic_fragment
        logger.info(f"Fragmento SelfCriticFragment '{critic_id}' instanciado.")

        exec_supervisor_id = "executor_supervisor_main"
        exec_supervisor_fragment = ExecutorSupervisorFragment(
            fragment_id=exec_supervisor_id, knowledge_interpreter=ki_fragment,
            self_critic=critic_fragment, context_store=context_store,
            memory_bank=MEMORY_BANK, post_chat_message_callback=post_message_handler
        )
        fragment_instances[exec_supervisor_id] = exec_supervisor_fragment
        logger.info(f"Fragmento ExecutorSupervisorFragment '{exec_supervisor_id}' instanciado.")

        starter_id = "autonomous_starter"
        autonomous_starter = AutonomousSelfStarterFragment(
            fragment_id=starter_id,
            professor_fragment=professor_fragment,
            knowledge_interpreter=ki_fragment,
            context_store=context_store,
            post_message_handler=post_message_handler
        )
        fragment_instances[starter_id] = autonomous_starter
        logger.info(f"Fragmento AutonomousSelfStarterFragment '{starter_id}' instanciado.")

    # <<< Exception for fragment initialization errors >>>
    except Exception as frag_init_err:
        logger.critical(f"Erro fatal ao instanciar fragmentos essenciais: {frag_init_err}", exc_info=True)
        # Let finally block handle cleanup
        raise # Reraise to ensure finally block runs and loop terminates

    # <<< Start the try block for task creation and execution >>>
    try:
        # --- Configurar Tarefas Assíncronas ---
        message_processor_task = asyncio.create_task(
            message_processor(context_store, MEMORY_BANK, ki_fragment, fragment_instances),
            name="MessageProcessor"
        )
        tasks.append(message_processor_task)
        logger.info("Tarefa Message Processor iniciada.")

        pending_question_task = asyncio.create_task(
            check_pending_questions_task(context_store, post_message_handler),
            name="PendingQuestionChecker"
        )
        tasks.append(pending_question_task)
        logger.info("Tarefa Check Pending Questions iniciada.")

        autonomous_starter_task = asyncio.create_task(
            autonomous_starter.run_autonomous_cycle(),
            name="AutonomousStarterCycle"
        )
        tasks.append(autonomous_starter_task)
        logger.info("Tarefa Autonomous Starter Cycle iniciada.")

        # --- Processamento do Script A3L Inicial (se fornecido) ---
        if a3l_script_path: # Ensure path is not None or empty
             logger.info(f"Processando script A3L inicial: {a3l_script_path}")
             # (Existing script processing logic - seems okay)
             log_file = Path("data/logs/a3net/session_output.a3l")
             log_file.parent.mkdir(parents=True, exist_ok=True)
             with open(log_file, "w", encoding="utf-8") as f:
                 f.write(f"# --- Session Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---" + "\n")
                 f.write(f"# --- Executing A3L Script: {a3l_script_path} ---" + "\n")
             print(f"[Session Log] Initialized log file: {log_file}")
             print(f"--- Running A3L Script: {a3l_script_path} ---")

             script_summary = {"Success": 0, "Failed": 0, "Skipped": 0, "Unrecognized": 0}
             line_num = 0
             try:
                 with open(a3l_script_path, 'r', encoding='utf-8') as script_file:
                     for line in script_file:
                         line_num += 1
                         line_content = line.strip()
                         print(f"\n[Line {line_num}] Processing: {line_content}")
                         with open(log_file, "a", encoding="utf-8") as f: f.write(line)

                         if not line_content or line_content.startswith('#'):
                             logger.debug(f"Skipping empty line or comment: {line_num}")
                             script_summary["Skipped"] += 1
                             continue
                         
                         processed_line_content = line_content.replace('\\"', '"')
                         try:
                             logger.info(f"[Runner] Passing to interpreter: {repr(processed_line_content)}")
                             directive = interpret_a3l_line(processed_line_content)
                         except Exception as interp_err:
                             logger.error(f"Erro INESPERADO ao interpretar linha {line_num} ('{processed_line_content}'): {interp_err}", exc_info=True)
                             directive = None
                             script_summary["Failed"] += 1

                         if directive:
                             logger.info(f"[Line {line_num}] Interpreted directive: {directive}")
                             await post_message_handler(
                                 message_type="a3l_command", content=directive,
                                 target_fragment="Executor"
                             )
                             script_summary["Success"] += 1
                         else:
                             logger.warning(f"[Line {line_num}] Line not recognized by A3L interpreter. Attempting KI interpretation...") 
                             print(f"[Line {line_num}] Line not recognized by A3L interpreter. Attempting KI interpretation...") 
                             interpret_directive = {
                                 "type": "interpret_text", "text": processed_line_content,
                                 "_origin": "Unrecognized A3L Script Line"
                             }
                             await post_message_handler(
                                 message_type="a3l_command", content=interpret_directive,
                                 target_fragment="Executor"
                             )
                             script_summary["Success"] += 1 
                             with open(log_file, "a", encoding="utf-8") as f:
                                 f.write(f"# INFO: Line {line_num} sent to KI for interpretation.\n")
             except FileNotFoundError:
                 logger.error(f"Erro crítico: Arquivo de script A3L não encontrado em '{a3l_script_path}'")
                 raise
             except Exception as file_err:
                 logger.error(f"Erro crítico ao ler o arquivo A3L '{a3l_script_path}': {file_err}", exc_info=True)
                 raise
             
             print(f"\n--- A3L Script Finished: {a3l_script_path} --- ")
             print(f"Summary: Success={script_summary['Success']}, Failed={script_summary['Failed']}, Skipped={script_summary['Skipped']}, Unrecognized={script_summary['Unrecognized']}")
             with open(log_file, "a", encoding="utf-8") as f:
                 f.write(f"# --- A3L Script Finished ---" + "\n")
                 f.write(f"# Summary: Success={script_summary['Success']}, Failed={script_summary['Failed']}, Skipped={script_summary['Skipped']}, Unrecognized={script_summary['Unrecognized']}\n")
                 f.write(f"# --- Session End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---" + "\n")
        else:
            logger.info("Nenhum script A3L inicial fornecido. Iniciando em modo autônomo.")
        
        # APÓS o script terminar, esperamos pelas tarefas de background
        logger.info("Script A3L principal concluído (se houver). Mantendo tarefas de background ativas indefinidamente (aguardando Ctrl+C ou erro)...")
        
        # --- Wait for background tasks (normal operation waits indefinitely) ---
        # Wait for the first task to complete (could be error or normal completion)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Check if any critical task failed
        critical_tasks_set = {message_processor_task, autonomous_starter_task} # Add others if needed
        for task in done:
            if task in critical_tasks_set:
                 try:
                     task.result() # Raise exception if task failed
                     logger.warning(f"Tarefa crítica {task.get_name()} finalizou inesperadamente sem erro.")
                 except asyncio.CancelledError:
                      logger.info(f"Tarefa crítica {task.get_name()} foi cancelada.")
                 except Exception as task_exc:
                     logger.error(f"Tarefa crítica {task.get_name()} falhou: {task_exc}", exc_info=True)
                     raise # Reraise the exception to trigger the finally block immediately
        
        # If no critical task failed, but some task finished, it's unexpected. Log and continue waiting.
        if done and not any(t in critical_tasks_set for t in done):
            for task in done:
                 logger.warning(f"Tarefa não crítica {task.get_name()} finalizou inesperadamente. Verifique a lógica.")
                 # Relaunch or handle if necessary
        
        # If we reach here, no critical task failed yet. Wait indefinitely until cancelled.
        logger.info("Tarefas de background em execução. Aguardando interrupção (Ctrl+C)...")
        await asyncio.gather(*tasks) # This will wait forever or until cancelled/error
        main_task_completed_normally = True # Set flag if gather finishes without error (unlikely in normal run)
        logger.info("Todas as tarefas de background terminaram normalmente (inesperado!).")

    # <<< Moved exception handling outside the main execution block >>>
    except asyncio.CancelledError:
         logger.info("Loop principal cancelado (provavelmente via Ctrl+C). Iniciando finalização...")
    except Exception as e:
        logger.error(f"Erro fatal no loop principal ou em tarefa de background: {e}", exc_info=True)
        # A exceção já causou a saída do await asyncio.gather/wait
        logger.info("Erro detectado. Iniciando finalização...")
        
    # <<< FINALLY block is now correctly aligned and outside the inner try >>>
    finally:
        # --- 7. Garantir Finalização Limpa --- 
        logger.info("Iniciando finalização limpa...")
        
        # --- Stop Fragments with stop() method FIRST ---
        logger.info("Tentando parar fragmentos com método stop()...")
        for frag_id, instance in fragment_instances.items():
             if hasattr(instance, 'stop'):
                 logger.debug(f"Chamando stop() para {frag_id}...")
                 try:
                     # Ensure stop is awaitable if it needs to be
                     stop_method = getattr(instance, 'stop')
                     if asyncio.iscoroutinefunction(stop_method):
                         await stop_method()
                     else:
                         stop_method() # Call synchronous stop
                 except Exception as stop_err:
                      logger.error(f"Erro ao parar {frag_id}: {stop_err}", exc_info=True)

        # --- Cancel all other background tasks --- 
        logger.info(f"Cancelando {len(tasks)} tarefas de background restantes...")
        cancelled_count = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        logger.info(f"{cancelled_count} tarefas canceladas.")
        
        # --- Wait for cancellation --- 
        logger.info("Aguardando finalização das tarefas canceladas...")
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                 task = tasks[i]
                 task_name = task.get_name() if hasattr(task, 'get_name') else f"Task-{i}" # Get name if possible
                 if isinstance(result, asyncio.CancelledError):
                     logger.info(f"Tarefa {task_name} cancelada com sucesso.")
                 elif isinstance(result, Exception):
                      logger.error(f"Tarefa {task_name} finalizou com exceção inesperada durante cleanup: {result}", exc_info=result)
                 elif not task.done():
                     logger.warning(f"Tarefa {task_name} ainda não concluída após gather no cleanup.")
                 else:
                     logger.debug(f"Tarefa {task_name} finalizada durante cleanup.")
            logger.info("Tarefas de background finalizadas ou canceladas.")

        # --- Stop the LLM server AFTER tasks are finished/cancelled --- 
        if llm_manager:
            logger.info("Parando o servidor LLM gerenciado...")
            await llm_manager.stop_server() # Assumes stop_server is async
            logger.info("Servidor LLM parado.")

        # --- Close ContextStore LAST --- 
        if context_store and hasattr(context_store, 'close'):
             try:
                 await context_store.close()
                 logger.info("ContextStore fechada.")
             except Exception as cs_close_err:
                  logger.error(f"Erro ao fechar ContextStore: {cs_close_err}", exc_info=True)
             
        logger.info("Processo A³Net finalizado.")

# --- Ponto de Entrada --- 
# <<< Ensure this is at the top level (zero indentation) >>>
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa um script A3L com o ciclo autônomo A³Net.")
    parser.add_argument("a3l_script", help="Caminho para o arquivo .a3l a ser executado.")
    args = parser.parse_args()

    script_path_obj = Path(args.a3l_script)
    if not script_path_obj.is_file():
        print(f"Erro: Arquivo A3L não encontrado em '{script_path_obj}'")
        exit(1)

    # --- Lidar com Interrupção (Ctrl+C) --- 
    loop = asyncio.get_event_loop()
    main_task = None

    def signal_handler():
        print("\nCtrl+C detectado! Iniciando finalização graciosa...")
        if main_task and not main_task.done(): # Check if task exists and not done
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
         try:
              loop.add_signal_handler(sig, signal_handler)
         except NotImplementedError:
             logger.warning(f"add_signal_handler não suportado para {sig} nesta plataforma.")

    try:
        # Pass the script path directly to main_loop
        main_task = loop.create_task(main_loop(str(script_path_obj)))
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        logger.info("Tarefa principal cancelada pelo signal handler.")
    finally:
        # --- Final cleanup of the event loop --- 
        logger.info("Iniciando cleanup final do loop de eventos...")
        # Remove handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try: loop.remove_signal_handler(sig)
            except (NotImplementedError, ValueError): pass
        
        # Gather any remaining tasks created outside main_loop (shouldn't be any ideally)
        # Cancel tasks that might still be running if main_loop exited prematurely
        tasks = asyncio.all_tasks(loop=loop)
        current_task = asyncio.current_task(loop=loop)
        tasks_to_cancel = [t for t in tasks if t is not current_task and not t.done()]
        if tasks_to_cancel:
            logger.info(f"Cancelando {len(tasks_to_cancel)} tarefas restantes no loop...")
            for task in tasks_to_cancel:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
            logger.info("Tarefas restantes canceladas.")
        else:
            logger.info("Nenhuma tarefa extra para cancelar no loop.")

        # Shutdown async generators
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            logger.info("Async generators desligados.")
        except Exception as shut_err:
            logger.error(f"Erro durante shutdown_asyncgens: {shut_err}", exc_info=True)
        
        # Close the loop
        # loop.close() # Closing the loop can sometimes cause issues if resources are still held
        logger.info("Loop de eventos finalizado.") 