import asyncio
import logging
import argparse
from pathlib import Path
import signal # Para lidar com interrupções (Ctrl+C)
import time
from typing import Dict, Any # Import necessary types

# --- Importações do Projeto A³X/A³Net ---
# (Ajuste os caminhos exatos conforme a estrutura do seu projeto)
try:
    # Contexto e Memória
    from a3x.a3net.core.context_store import SQLiteContextStore, ContextStore
    from a3x.a3net.integration.a3x_bridge import MEMORY_BANK, handle_directive # Assumindo singleton ou inicialização global
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
except ImportError as e:
    print(f"Erro de Importação: {e}. Verifique os caminhos e a estrutura do projeto.")
    print("Certifique-se de executar este script a partir do diretório raiz do projeto A3X ou que o PYTHONPATH esteja configurado.")
    exit(1)

# --- Configuração de Logging ---
# (Pode ser configurado de forma mais robusta externamente)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Log para console
        logging.FileHandler(OUTPUT_LOG_FILE, mode='w') # Log para arquivo (sobrescreve)
    ]
)
# Silenciar logs muito verbosos de bibliotecas, se necessário
# logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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

async def message_processor(context_store: ContextStore, ki_fragment: KnowledgeInterpreterFragment, fragment_instances: Dict[str, Any]):
    """Processa mensagens da fila, principalmente roteando comandos A3L para o Executor."""
    logger.info("Message Processor iniciado.")
    while True:
        try:
            message = await message_queue.get()
            logger.debug(f"[MessageProc] Recebido: {message}")
            
            target = message.get("target_fragment")
            msg_type = message.get("message_type")
            content = message.get("content") # content é o dicionário da diretiva

            if target == "Executor" and msg_type == "a3l_command" and isinstance(content, dict):
                directive_dict = content 
                origin_source = content.get("origin_suggestion", {}).get("source", "A3L_Script")
                directive_type = directive_dict.get("type")

                # --- Tratar 'interpret_text' separadamente --- 
                if directive_type == "interpret_text":
                    logger.info(f"[MessageProc] Recebido comando 'interpret_text' da origem '{origin_source}'. Chamando KI...")
                    text_to_interpret = directive_dict.get("text")
                    
                    if ki_fragment and text_to_interpret:
                        try:
                            # Chama o interpretador para obter comandos A3L (agora async)
                            extracted_commands, metadata = await ki_fragment.interpret_knowledge(text_to_interpret, context_fragment_id=None) # Passar contexto se relevante?
                            source_info = metadata.get("source", "Unknown KI Source") # Usar metadados
                            
                            if extracted_commands:
                                logger.info(f"[MessageProc] KI '{ki_fragment.fragment_id}' (Source: {source_info}) extraiu comandos: {extracted_commands}")
                                append_to_log(f"# [KI Resultado - MsgProc via {source_info}] Extraído(s) de '{origin_source}': {extracted_commands}")
                                
                                # --- Re-enfileirar comandos extraídos para execução --- 
                                logger.info(f"[MessageProc] Re-enfileirando {len(extracted_commands)} comandos extraídos para execução...")
                                for cmd_str in extracted_commands:
                                    try:
                                        # Re-interpretar a string do comando para obter o dicionário da diretiva
                                        reinterpreted_directive = interpret_a3l_line(cmd_str)
                                        if reinterpreted_directive:
                                             # Adicionar informação da origem para rastreamento
                                             reinterpreted_directive["_origin"] = f"Extracted by KI ({source_info}) from {origin_source}"
                                             # Enviar de volta para a fila como um comando A3L normal
                                             await post_message_handler(
                                                 message_type="a3l_command",
                                                 content=reinterpreted_directive,
                                                 target_fragment="Executor"
                                             )
                                             logger.info(f"[MessageProc] Comando reenfileirado: {cmd_str}")
                                        else:
                                            logger.warning(f"[MessageProc] Falha ao re-interpretar comando extraído pelo KI: {cmd_str}")
                                            append_to_log(f"# [AVISO KI - MsgProc] Falha ao re-interpretar comando extraído: {cmd_str}")
                                    except Exception as reinterpr_err:
                                         logger.error(f"[MessageProc] Erro ao re-interpretar/reenfileirar comando extraído '{cmd_str}': {reinterpr_err}", exc_info=True)
                                         append_to_log(f"# [FALHA KI - MsgProc] Erro ao reenfileirar comando '{cmd_str}': {reinterpr_err}")
                                # ----------------------------------------------------
                                
                                # Comentado/Removido: Lógica anterior de enviar para ExecutorSupervisor
                                # if hasattr(ki_fragment, 'post_chat_message') and callable(getattr(ki_fragment, 'post_chat_message')):
                                #      await ki_fragment.post_chat_message(
                                #          message_type="suggestion", 
                                #          content={"commands": extracted_commands, "source": f"KI via {origin_source}"},
                                #          target_fragment="ExecutorSupervisorFragment" # Ou um ID específico
                                #      )
                                #      logger.info(f"[MessageProc] Comandos extraídos enviados como sugestão para ExecutorSupervisor.")
                                # else:
                                #      logger.warning("[MessageProc] KI não possui post_chat_message para enviar comandos extraídos.")
                                
                            else:
                                logger.info(f"[MessageProc] KI '{ki_fragment.fragment_id}' (Source: {source_info}) não extraiu comandos do texto.")
                                append_to_log(f"# [KI Resultado - MsgProc via {source_info}] Nenhum comando extraído de '{origin_source}'.")
                                
                        except Exception as ki_err:
                            logger.error(f"[MessageProc] Erro ao executar KI '{ki_fragment.fragment_id}' para interpret_text: {ki_err}", exc_info=True)
                            append_to_log(f"# [FALHA KI - MsgProc] Erro ao interpretar texto de '{origin_source}': {ki_err}")
                            # Considerar isso como falha da diretiva interpret_text?
                    else:
                        error_msg = "KI não disponível" if not ki_fragment else "Texto para interpretar não encontrado"
                        logger.error(f"[MessageProc] Falha ao processar interpret_text: {error_msg}")
                        append_to_log(f"# [FALHA interpret_text - MsgProc] {error_msg}")

                # --- Tratar 'ask_professor' diretamente ---
                elif directive_type == "ask_professor":
                    professor_id = directive_dict.get("professor_id")
                    question = directive_dict.get("question")
                    logger.info(f"[MessageProc] Processando 'ask_professor' para '{professor_id}'...")
                    professor_fragment = fragment_instances.get(professor_id)
                    if isinstance(professor_fragment, ProfessorLLMFragment) and question:
                        try:
                            response = await professor_fragment.ask_llm(question)
                            logger.info(f"[MessageProc] Resposta de ask_professor '{professor_id}': {response[:100]}...")
                            # TODO: O que fazer com a resposta? Por ora, apenas logamos.
                            # Poderia ser reenfileirada para 'interpret_text' se necessário?
                            append_to_log(f"# [ask_professor - MsgProc] '{professor_id}' respondeu: {response}")
                        except Exception as ask_err:
                            logger.error(f"[MessageProc] Erro ao executar ask_professor para '{professor_id}': {ask_err}", exc_info=True)
                            append_to_log(f"# [FALHA ask_professor - MsgProc] Erro: {ask_err}")
                    else:
                        error_msg = f"Professor '{professor_id}' não encontrado ou inválido" if not professor_fragment else "Questão não fornecida"
                        logger.error(f"[MessageProc] Falha ao processar ask_professor: {error_msg}")
                        append_to_log(f"# [FALHA ask_professor - MsgProc] {error_msg}")
                        \
                # --- Tratar 'learn_from_professor' diretamente ---
                elif directive_type == "learn_from_professor":
                    professor_id = directive_dict.get("professor_id")
                    question = directive_dict.get("question")
                    context_id = directive_dict.get("context_fragment_id") # Opcional
                    logger.info(f"[MessageProc] Processando 'learn_from_professor' para '{professor_id}'...")
                    
                    if professor_id and question and ki_fragment and context_store:
                        try:
                            # Chamar a função importada do executor
                            # Passa um dicionário PRE-INICIALIZADO para results_summary.
                            # A função learn_from_professor deve lidar com seu próprio logging/estado.
                            initial_summary = {'success': 0, 'failed': 0, 'skipped': 0} # <<< Initialize dict
                            learn_result = await learn_from_professor(\
                                professor_id=professor_id,\
                                question=question,\
                                ki_fragment=ki_fragment, \
                                results_summary=initial_summary, # Passa dict inicializado
                                context_store=context_store,\
                                fragmento_referido=context_id\
                            )
                            logger.info(f"[MessageProc] Resultado de learn_from_professor '{professor_id}': {learn_result}")
                            # O log A3L é feito dentro da função learn_from_professor
                        except Exception as learn_err:
                             logger.error(f"[MessageProc] Erro ao executar learn_from_professor para '{professor_id}': {learn_err}", exc_info=True)
                             append_to_log(f"# [FALHA learn_from_professor - MsgProc] Erro: {learn_err}")
                    else:
                         missing = []
                         if not professor_id: missing.append("professor_id")
                         if not question: missing.append("question")
                         if not ki_fragment: missing.append("ki_fragment")
                         if not context_store: missing.append("context_store")
                         error_msg = f"Dependências ausentes para learn_from_professor: {', '.join(missing)}"
                         logger.error(f"[MessageProc] Falha ao processar learn_from_professor: {error_msg}")
                         append_to_log(f"# [FALHA learn_from_professor - MsgProc] {error_msg}")
                         \
                # --- Tratar 'estudar habilidade' (Macro Comando) ---
                elif directive_type == "estudar_habilidade":
                    task_name = directive_dict.get("task_name")
                    if not task_name:
                        logger.error(f"[MessageProc] Erro: 'task_name' ausente na diretiva 'estudar_habilidade': {directive_dict}")
                        append_to_log(f"# [FALHA Macro - estudar_habilidade] task_name ausente.")
                    else:
                        fragment_id = task_name 
                        logger.info(f"[MessageProc] Decompondo macro 'estudar habilidade' para task='{task_name}', fragment='{fragment_id}'.")
                        append_to_log(f"# --- Decompondo 'estudar habilidade \"{task_name}\"' ---")
                        
                        # Sequência de sub-comandos
                        epochs = 10 # Default epochs for training step
                        sub_commands = [
                            f'planejar dados para tarefa "{task_name}"',                 # Etapa 1: Planejar formato
                            f'solicitar exemplos para tarefa "{task_name}"',             # Etapa 2: Solicitar exemplos (usará o formato planejado)
                            f'treinar fragmento \'{fragment_id}\' por {epochs} épocas',        # Etapa 3: Treinar
                            f'avaliar fragmento \'{fragment_id}\' com dados de teste \'{task_name}\' ', # Etapa 4: Avaliar
                            f'comparar desempenho do fragmento \'{fragment_id}\' após treino em \'{task_name}\'' # Etapa 5: Comparar
                        ]
                        
                        # Enqueue each sub-command
                        for cmd_index, cmd_str in enumerate(sub_commands):
                            try:
                                # Interpret the sub-command string back into a directive dict
                                sub_directive = interpret_a3l_line(cmd_str)
                                if sub_directive:
                                    # Add origin information
                                    sub_directive["_origin"] = f"Decomposed from 'estudar habilidade \"{task_name}\"', step {cmd_index+1}"
                                    # Post to the queue for normal processing
                                    await post_message_handler(
                                        message_type="a3l_command",
                                        content=sub_directive,
                                        target_fragment="Executor" # Target Executor
                                    )
                                    logger.info(f"[MessageProc] Sub-comando {cmd_index+1} enfileirado: {cmd_str}")
                                    append_to_log(f"# [MACRO - estudar_habilidade] Passo {cmd_index+1} enfileirado: {cmd_str}")
                                    # Optional: Add a small delay between posting commands if needed for sequencing
                                    # await asyncio.sleep(0.1) 
                                else:
                                    logger.error(f"[MessageProc] Falha ao re-interpretar sub-comando gerado {cmd_index+1}: {cmd_str}")
                                    append_to_log(f"# [FALHA Macro - estudar_habilidade] Falha ao interpretar sub-comando: {cmd_str}")
                                    # Decide whether to stop decomposition on error? For now, continue.
                            except Exception as decomp_err:
                                logger.error(f"[MessageProc] Erro ao processar/enfileirar sub-comando {cmd_index+1} '{cmd_str}': {decomp_err}", exc_info=True)
                                append_to_log(f"# [FALHA Macro - estudar_habilidade] Erro ao enfileirar sub-comando: {cmd_str} - {decomp_err}")
                                # Decide whether to stop decomposition on error? For now, continue.
                        logger.info(f"[MessageProc] Decomposição de 'estudar habilidade' para '{task_name}' concluída.")
                
                # --- Tratar outros tipos de diretiva via handle_directive --- 
                else:
                    logger.info(f"[MessageProc] Roteando diretiva '{directive_type}' da origem '{origin_source}' para execução via handle_directive: {directive_dict}")
                    try:
                        # <<< Pass context_store, fragment_instances, and post_message_handler >>>
                        result = await handle_directive(
                            directive=directive_dict, 
                            fragment_instances=fragment_instances, 
                            context_store=context_store,
                            post_message_handler=post_message_handler # <<< Pass the handler
                        )
                        
                        # --- Handle Corrective Commands Returned by Bridge --- 
                        if result:
                            logger.info(f"[MessageProc] Resultado de handle_directive: {result}")
                            status = result.get("status", "unknown")
                            if status == "error":
                                logger.error(f"[MessageProc] Erro retornado por handle_directive: {result.get('message', 'Detalhe não especificado')}")
                                # Log the error in A3L format
                                append_to_log(f"# [FALHA Bridge - {directive_type}] Erro: {result.get('message', 'Detalhe não especificado')}")
                                
                            # Check for corrective commands (even if status wasn't explicitly 'error')
                            corrective_commands = result.get("corrective_commands")
                            if corrective_commands and isinstance(corrective_commands, list):
                                logger.info(f"[MessageProc] Recebido {len(corrective_commands)} comandos corretivos de handle_directive.")
                                append_to_log(f"# [AUTO-CORRECAO - MsgProc] Recebido(s): {corrective_commands}")
                                for cmd_str in corrective_commands:
                                    if isinstance(cmd_str, str) and cmd_str:
                                        try:
                                            reinterpreted_directive = interpret_a3l_line(cmd_str)
                                            if reinterpreted_directive:
                                                reinterpreted_directive["_origin"] = f"Corrective action from {directive_type}"
                                                await post_message_handler(
                                                    message_type="a3l_command",
                                                    content=reinterpreted_directive,
                                                    target_fragment="Executor"
                                                )
                                                logger.info(f"[MessageProc] Comando corretivo reenfileirado: {cmd_str}")
                                            else:
                                                logger.warning(f"[MessageProc] Falha ao re-interpretar comando corretivo: {cmd_str}")
                                                append_to_log(f"# [AVISO AUTO-CORRECAO - MsgProc] Falha ao interpretar: {cmd_str}")
                                        except Exception as corr_err:
                                            logger.error(f"[MessageProc] Erro ao processar comando corretivo '{cmd_str}': {corr_err}", exc_info=True)
                                            append_to_log(f"# [FALHA AUTO-CORRECAO - MsgProc] Erro: {corr_err}")
                                    else:
                                        logger.warning(f"[MessageProc] Ignorando item inválido em corrective_commands: {cmd_str}")
                        # ------------------------------------------------------
                        else:
                            logger.warning(f"[MessageProc] handle_directive retornou None para a diretiva: {directive_dict}")
                            append_to_log(f"# [AVISO Bridge - {directive_type}] Handler retornou None.")
                            
                    except Exception as bridge_err:
                        logger.error(f"[MessageProc] Erro ao executar handle_directive para '{directive_type}': {bridge_err}", exc_info=True)
                        append_to_log(f"# [FALHA Bridge - {directive_type}] Erro: {bridge_err}")

            else:
                # TODO: Implementar roteamento para outros fragmentos/tipos se necessário
                logger.debug(f"[MessageProc] Mensagem não roteada para Executor: Target={target}, Type={msg_type}")

            message_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Message Processor cancelado.")
            break
        except Exception as e:
            logger.error(f"[MessageProc] Erro ao processar mensagem: {e}", exc_info=True)
            await asyncio.sleep(5) 

async def main_loop(a3l_script_path: str):
    """Função principal que inicializa, executa e gerencia o A³Net."""
    context_store = None
    background_tasks = []
    fragment_instances = {} # Armazenar instâncias para referência e parada
    server_manager = None # Inicializa como None

    try:
        # --- 0. Inicializar Gerenciador do Servidor LLM --- 
        # !!! AJUSTE O COMANDO ABAIXO PARA O SEU AMBIENTE !!!
        # Exemplo: server_cmd = ["/home/user/llama.cpp/server", "-m", "/home/user/models/llama-2-7b-chat.Q4_K_M.gguf", "-c", "4096", "--port", "8080", "--host", "0.0.0.0"]
        server_cmd = [
             "/home/arthur/projects/A3X/llama.cpp/build/bin/llama-server", # Updated path
             "-m", "/home/arthur/projects/A3X/models/google_gemma-3-4b-it-Q4_K_S(1).gguf", # Updated path
             "-c", "4096", # Context size
             "--port", "8080",
             "--host", "0.0.0.0" # Escutar em todas as interfaces (ou 127.0.0.1 para local)
             # Adicione outros flags necessários (ex: -ngl 32 para GPU layers)
        ]
        server_manager = LLMServerManager(server_command=server_cmd, host="127.0.0.1", port=8080)
        # Tentar iniciar o servidor ANTES de qualquer coisa que dependa dele
        await server_manager.start_server()
        # Se start_server falhar com exceção, o try/except principal pegará
        # ------------------------------------------------------
        
        # --- 1. Inicializar ContextStore ---
        # TODO: Tornar o nome do arquivo configurável
        db_path = "a3net_context.sqlite" 
        context_store = SQLiteContextStore(db_path) 
        await context_store.initialize() 
        logger.info(f"ContextStore inicializada em '{db_path}'.")

        # --- 2. Inicializar Interpretador de Conhecimento ---
        # Necessário para Executor
        ki_fragment = KnowledgeInterpreterFragment(
             fragment_id="ki_main_instance", 
             description="Instância global do Interpretador de Conhecimento"
        )
        fragment_instances[ki_fragment.fragment_id] = ki_fragment
        logger.info("KnowledgeInterpreterFragment inicializado.")

        # --- 3. Instanciar e Configurar Fragmentos ---
        logger.info("Instanciando e configurando fragmentos...")
        
        # Supervisores e Geradores
        supervisor = SupervisorFragment("supervisor_1", "Periodic Reflector")
        supervisor.set_message_handler(post_message_handler)
        fragment_instances[supervisor.fragment_id] = supervisor

        self_critic = SelfCriticFragment("critic_1", "Periodic Critic")
        self_critic.set_context_and_handler(context_store, post_message_handler)
        fragment_instances[self_critic.fragment_id] = self_critic

        executor_supervisor = ExecutorSupervisorFragment("exec_supervisor_1", "Suggestion Executor")
        executor_supervisor.set_context_and_handler(context_store, post_message_handler)
        fragment_instances[executor_supervisor.fragment_id] = executor_supervisor
        
        meta_generator = MetaGeneratorFragment("meta_gen_1", "Internal Goal Generator")
        meta_generator.set_message_handler(post_message_handler)
        fragment_instances[meta_generator.fragment_id] = meta_generator

        # --- Novo Fragmento Autônomo --- 
        # Usar as instâncias já criadas de KI e Professor
        # A instância do professor pode ser None se LLM_ENABLED=False
        professor_instance = fragment_instances.get("prof_geral", None) 
        
        self_starter = AutonomousSelfStarterFragment(
            fragment_id="autonomous_starter_1",
            # Passar as dependências diretamente no construtor
            ki_fragment=ki_fragment, 
            professor_fragment=professor_instance,
            context_store=context_store # Injetar o context_store
        )
        
        # Usa self_starter.metadata.name como a chave do dicionário
        fragment_instances[self_starter.metadata.name] = self_starter 
        logger.info("AutonomousSelfStarterFragment instanciado e dependências (KI, Professor, ContextStore) passadas.")
        # ---------------------------------

        # Professor (Opcional)
        # Movido Bloco de instanciação do professor para ANTES do SelfStarter
        professor = None # Definir professor como None inicialmente
        if LLM_ENABLED:
            # TODO: Tornar URL configurável (via env var, config file, etc.)
            llm_url = "http://localhost:8080/completion" 
            # --- MOCK REMOVED ---
            # mock_a3l_response = "# Mock response: Nenhuma ação aplicável"
            # logger.warning(f"!!! USANDO MOCK RESPONSE PARA ProfessorLLMFragment: {mock_a3l_response} !!!")
            professor = ProfessorLLMFragment(
                 fragment_id="prof_geral", 
                 description="Oráculo LLM geral", # Reverted description 
                 llm_url=llm_url
                 # mock_response=mock_a3l_response # <<< MOCK REMOVED
            )
            # ------------------
            fragment_instances[professor.fragment_id] = professor # Adicionar ao dicionário
            
            # --- Injetar Professor no KI --- 
            if ki_fragment and hasattr(ki_fragment, 'set_professor'):
                 ki_fragment.set_professor(professor)
                 logger.info(f"Injetado Professor '{professor.fragment_id}' no KI '{ki_fragment.fragment_id}'.")
            # --------------------------------- 
            
            # Adicionar ao MEMORY_BANK para ser encontrado por 'aprender com'
            if MEMORY_BANK:
                 try:
                     MEMORY_BANK.save(professor.fragment_id, professor)
                     logger.info(f"ProfessorLLMFragment '{professor.fragment_id}' SALVO no MEMORY_BANK.")
                 except Exception as save_err:
                     logger.error(f"Falha ao SALVAR ProfessorLLM '{professor.fragment_id}' no MEMORY_BANK: {save_err}", exc_info=True)
                     if hasattr(MEMORY_BANK, 'add'):
                          try:
                             MEMORY_BANK.add(professor.fragment_id, professor)
                             logger.info(f"ProfessorLLMFragment '{professor.fragment_id}' adicionado (via add) ao MEMORY_BANK.")
                          except Exception as add_err:
                             logger.error(f"Falha ao ADICIONAR (add) ProfessorLLM '{professor.fragment_id}' no MEMORY_BANK: {add_err}", exc_info=True)
                     else:
                         logger.warning("MEMORY_BANK não possui método 'add' para fallback.")
            else:
                 logger.warning("MEMORY_BANK não está disponível. ProfessorLLM pode não ser encontrável por 'aprender com'.")
        else:
            logger.warning("LLM_ENABLED=False. ProfessorLLMFragment não será instanciado.")
            if ki_fragment and hasattr(ki_fragment, 'set_professor'):
                 ki_fragment.set_professor(None)
                 logger.info(f"Professor não habilitado. KI '{ki_fragment.fragment_id}' configurado sem professor.")

        logger.info("Fragmentos instanciados e configurados.")

        # --- 5. Iniciar Tarefas em Background ---
        logger.info("Iniciando tarefas em background...")
        processor_task = asyncio.create_task(message_processor(context_store, ki_fragment, fragment_instances))
        background_tasks.append(processor_task)
        
        supervisor_task = asyncio.create_task(supervisor.start())
        background_tasks.append(supervisor_task)
        
        executor_supervisor_task = asyncio.create_task(executor_supervisor.start())
        background_tasks.append(executor_supervisor_task)
        
        meta_generator_task = asyncio.create_task(meta_generator.start())
        background_tasks.append(meta_generator_task)
        
        await asyncio.sleep(1) # Pequeno delay para garantir que os loops iniciem

        # --- 6. Executar Script A3L Principal (CORRIGIDO) ---
        logger.info(f"Iniciando execução do script A3L principal: {a3l_script_path}")
        
        # Log para indicar início da execução do arquivo A3L
        log_file = Path("a3x/a3net/examples/session_output.a3l") # TODO: Make configurable?
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# --- Session Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"# --- Executing A3L Script: {a3l_script_path} ---\n")
        print(f"[Session Log] Initialized log file: {log_file}")
        print(f"--- Running A3L Script: {a3l_script_path} ---")

        # Contador para sumário
        script_summary = {"Success": 0, "Failed": 0, "Skipped": 0, "Unrecognized": 0}
        line_num = 0

        try:
            with open(a3l_script_path, 'r', encoding='utf-8') as script_file:
                for line in script_file:
                    line_num += 1
                    line_content = line.strip()
                    print(f"\n[Line {line_num}] Processing: {line_content}") # Log original
                    with open(log_file, "a", encoding="utf-8") as f: # Append original line to log
                        f.write(line) # Write original line with newline

                    if not line_content or line_content.startswith('#'):
                        logger.debug(f"Skipping empty line or comment: {line_num}")
                        script_summary["Skipped"] += 1
                        continue
                    
                    # REMOVER escapes de aspas duplas antes de interpretar
                    processed_line_content = line_content.replace('\\"', '"')

                    # Interpretar a linha usando a função importada
                    try:
                        # LOG ADICIONAL PARA VER A STRING EXATA ANTES DA CHAMADA
                        logger.info(f"[Runner] Passing to interpreter: {repr(processed_line_content)}") # Logar a linha processada 
                        directive = interpret_a3l_line(processed_line_content) # Passa a linha processada
                    except Exception as interp_err:
                        logger.error(f"Erro INESPERADO ao interpretar linha {line_num} ('{processed_line_content}'): {interp_err}", exc_info=True) # Usar processed_line_content no log de erro
                        directive = None # Tratar como não reconhecido
                        script_summary["Failed"] += 1 # Contar como falha

                    if directive:
                        logger.info(f"[Line {line_num}] Interpreted directive: {directive}")
                        # Enviar para a fila de mensagens para o Executor processar
                        await post_message_handler(
                            message_type="a3l_command", 
                            content=directive, # Enviar o dicionário resultante
                            target_fragment="Executor" # Alvo padrão para comandos A3L
                        )
                        script_summary["Success"] += 1
                        # TODO: Poderia esperar uma confirmação aqui? Ou deixar assíncrono?
                        # Por simplicidade, vamos deixar assíncrono por enquanto.
                        # await asyncio.sleep(0.1) # Pequeno delay entre comandos?
                    else:
                        # Log já é feito dentro de interpret_a3l_line
                        print(f"[Line {line_num}] Line not recognized or produced no valid directive: {processed_line_content}") # Logar a linha processada aqui também
                        script_summary["Unrecognized"] += 1
                        with open(log_file, "a", encoding="utf-8") as f:
                             f.write(f"# WARNING: Line {line_num} not recognized by interpreter.\n")
        except FileNotFoundError:
            logger.error(f"Erro crítico: Arquivo de script A3L não encontrado em '{a3l_script_path}'")
            # Não pode continuar, poderia lançar exceção ou finalizar
            raise # Re-lançar a exceção para finalizar o main_loop
        except Exception as file_err:
            logger.error(f"Erro crítico ao ler o arquivo A3L '{a3l_script_path}': {file_err}", exc_info=True)
            raise # Re-lançar para finalizar
        
        print(f"\n--- A3L Script Finished: {a3l_script_path} --- ")
        print(f"Summary: Success={script_summary['Success']}, Failed={script_summary['Failed']}, Skipped={script_summary['Skipped']}, Unrecognized={script_summary['Unrecognized']}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"# --- A3L Script Finished ---\n")
            f.write(f"# Summary: Success={script_summary['Success']}, Failed={script_summary['Failed']}, Skipped={script_summary['Skipped']}, Unrecognized={script_summary['Unrecognized']}\n")
            f.write(f"# --- Session End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # APÓS o script terminar, esperamos pelas tarefas de background
        # Mantemos as tarefas de background rodando indefinidamente até Ctrl+C
        # ou até que uma delas termine com um erro.
        logger.info("Script A3L principal concluído. Mantendo tarefas de background ativas indefinidamente (aguardando Ctrl+C ou erro)...")
        
        # --- Executar o Fragmento Autônomo (Após o script principal) ---
        logger.info(f"Iniciando execução autônoma do fragmento '{self_starter.fragment_id}'...")
        try:
            await self_starter.execute() # Executa a missão autônoma
            logger.info(f"Execução autônoma de '{self_starter.fragment_id}' concluída.")
        except Exception as starter_err:
            logger.error(f"Erro durante a execução autônoma de '{self_starter.fragment_id}': {starter_err}", exc_info=True)
        # -------------------------------------------------------------

        # Esperar por TODAS as tarefas. Se alguma falhar (exceto CancelledError),
        # o gather() levantará a exceção, que será pega pelo try/except externo.
        await asyncio.gather(*background_tasks)
        
        # Esta linha só será alcançada se TODAS as tarefas terminarem SEM exceções 
        # (o que não deve acontecer em operação normal, pois elas rodam em loop)
        logger.info("Todas as tarefas de background terminaram normalmente (inesperado!).")

    except asyncio.CancelledError:
         logger.info("Loop principal cancelado (provavelmente via Ctrl+C). Iniciando finalização...")
    except Exception as e:
        logger.error(f"Erro fatal no loop principal ou em tarefa de background: {e}", exc_info=True)
        # A exceção já causou a saída do await asyncio.gather
        logger.info("Erro detectado. Iniciando finalização...")
    finally:
        # --- 7. Garantir Finalização Limpa ---
        logger.info("Iniciando finalização limpa...")
        
        # --- Parar o servidor LLM primeiro --- 
        if server_manager:
            logger.info("Parando o servidor LLM gerenciado...")
            await server_manager.stop_server()
        # -------------------------------------
        
        # Parar Supervisores que possuem método stop()
        # (O ideal é que start() retorne a task para cancelamento seguro)
        for frag_id, instance in fragment_instances.items():
             if hasattr(instance, 'stop'):
                 logger.info(f"Parando {frag_id}...")
                 await instance.stop() # stop() deve cancelar a task interna

        # Cancelar todas as tarefas restantes (incluindo message_processor e script_task se ainda rodando)
        for task in background_tasks:
            if not task.done():
                task.cancel()
        
        # Aguardar cancelamento das tarefas
        if background_tasks:
            await asyncio.wait(background_tasks, timeout=5.0) # Timeout para evitar bloqueio

        # Verificar tarefas pendentes após cancelamento
        for task in background_tasks:
            if not task.done():
                logger.warning(f"Task {task.get_name()} não finalizou após cancelamento.")
            elif task.exception():
                logger.error(f"Task {task.get_name()} finalizou com exceção: {task.exception()}")


        # Fechar ContextStore
        if context_store and hasattr(context_store, 'close'):
            await context_store.close()
            logger.info("ContextStore fechada.")
            
        logger.info("Processo A³Net finalizado.")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa um script A3L com o ciclo autônomo A³Net.")
    parser.add_argument("a3l_script", help="Caminho para o arquivo .a3l a ser executado.")
    # Adicionar outros argumentos se necessário (ex: --db-path, --llm-url)
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
        if main_task:
            main_task.cancel()

    # Adicionar handler para SIGINT e SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
         try:
              loop.add_signal_handler(sig, signal_handler)
         except NotImplementedError:
             # Windows pode não suportar add_signal_handler
             logger.warning(f"add_signal_handler não suportado para {sig} nesta plataforma.")

    try:
        main_task = loop.create_task(main_loop(str(script_path_obj)))
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        logger.info("Tarefa principal cancelada.")
    finally:
        # Remover handlers para evitar chamadas durante shutdown do loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except NotImplementedError:
                pass # Ignora se não suportado
        
        # Garante que o loop feche corretamente
        # (Importante se houver tarefas pendentes que precisam ser canceladas)
        # loop.run_until_complete(loop.shutdown_asyncgens())
        # loop.close()
        logger.info("Loop de eventos finalizado.") 