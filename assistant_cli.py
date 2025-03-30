import requests
import json
import os
from dotenv import load_dotenv
import argparse
import time
import traceback # Para debug de erros
import logging # Usar logging
import sys
import subprocess # Necessário para executar código via firejail
import asyncio # <<< Adicionar import >>>

# Configurar logging básico para o CLI
logging.basicConfig(level=logging.INFO, format='[%(levelname)s CLI] %(message)s')
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# Imports dos módulos core
# Removidos NLU, Planner, Dispatcher por enquanto
# from core.nlu import interpret_command
# from core.nlg import generate_natural_response, generate_simplified_response # Mantém NLG por enquanto
from core.config import MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LOG_LEVEL, DB_FILE, AGENT_STATE_ID # <<< Import LLAMA_SERVER_URL >>>
# from core.dispatcher import get_skill, SKILL_DISPATCHER
# from core.planner import generate_plan
from core.db_utils import initialize_database, save_agent_state, load_agent_state
from core.agent import ReactAgent # <-- NOVO IMPORT

# Adiciona o diretório pai ao sys.path para encontrar a pasta 'core'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# <<< DEFINIR PROMPT PADRÃO AQUI >>>
DEFAULT_SYSTEM_PROMPT = """Você é um agente autônomo chamado A³X que segue o framework ReAct para atingir objetivos complexos.

Seu ciclo é: pensar, agir, observar.

⚠️ **FORMATO OBRIGATÓRIO** DE RESPOSTA:

Sempre responda neste exato formato, e somente nele:

Thought: <raciocínio sobre o próximo passo>
Action: <nome_da_ferramenta_disponível>
Action Input: <objeto JSON com os parâmetros da ferramenta>

✅ Exemplo:

Thought: Para ler o conteúdo do arquivo solicitado, devo usar a ferramenta 'read_file'.
Action: read_file
Action Input: {"file_name": "caminho/do/arquivo.txt"}

Nunca explique o que está fazendo fora do bloco "Thought:". Nunca adicione justificativas ou mensagens fora do formato.

Se não for possível agir, retorne uma Action chamada 'final_answer' com a resposta final no campo 'answer'.

Esse formato será interpretado por outro sistema e precisa estar 100% correto.""" # Placeholder for tool descriptions - will be replaced by agent

# --- Funções Auxiliares ---

# Função para garantir que estamos no diretório correto
def change_to_project_root():
    """Garante que o script seja executado a partir do diretório raiz do projeto."""
    # Determina o diretório raiz do projeto (diretório pai do diretório do script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    try:
        # Tenta ir para a raiz apenas se não estiver lá
        if os.getcwd() != project_root:
            os.chdir(project_root)
        logger.info(f"Running in directory: {os.getcwd()}")
    except FileNotFoundError:
        logger.error(f"Project root directory not found: {project_root}. Exiting.")
        print(f"[Fatal Error] Project directory not found: {project_root}")
        sys.exit(1)
    except Exception as cd_err:
        logger.error(f"Error changing directory to {project_root}: {cd_err}. Exiting.")
        print(f"[Fatal Error] Could not change to project directory: {cd_err}")
        sys.exit(1)

# Função process_command agora recebe a instância do agente
async def process_command(agent: ReactAgent, command: str, conversation_history: list) -> None:
    """Processa um único comando usando a instância fornecida do Agente ReAct."""
    logger.info(f"Processing command: '{command}'")
    print(f"\n> {command}")
    conversation_history.append({"role": "user", "content": command})

    # NÃO instancia mais o agente aqui dentro

    final_response = ""
    agent_outcome = None

    try:
        # Usa a instância do agente passada como argumento
        final_response = await agent.run(objective=command)
        agent_outcome = {"status": "success", "action": "react_cycle_completed", "data": {"message": final_response}}
        logger.info(f"Agent completed. Final response: {final_response}")

    except Exception as e:
        logger.exception(f"Fatal Agent Error processing command '{command}':")
        final_response = f"Sorry, a critical internal error occurred while processing your command."
        agent_outcome = {"status": "error", "action": "react_cycle_failed", "data": {"message": str(e)}}

    # --- LÓGICA DE RESPOSTA (NLG) ---
    # Usamos a resposta final direta do agente
    response_text = final_response
    print(f"[A³X]: {response_text}")
    conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "agent_outcome": agent_outcome # Guarda o resultado do agente para referência
    })
    # ... (limitar histórico se necessário) ...


def main():
    # Garante execução no diretório raiz do projeto
    change_to_project_root()

    parser = argparse.ArgumentParser(description='Assistente CLI A³X (ReAct)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')
    parser.add_argument('--task', help='Tarefa única a ser executada pelo agente (prioridade sobre -c e -i)')
    parser.add_argument("--stream-direct", help="Prompt direto para o LLM em modo streaming")

    args = parser.parse_args()

    conversation_history = [] # Histórico da conversa geral

    # <<< INICIALIZA DB AQUI >>>
    logger.info("Initializing database...")
    initialize_database()
    # <<< FIM INICIALIZAÇÃO DB >>>

    # <<< INSTANCIAÇÃO DO AGENTE (Only if not streaming direct) >>>
    agent = None
    if not args.stream_direct:
        logger.info("Initializing ReactAgent...")
        try:
            # <<< PASSAR ARGUMENTOS >>>
            llm_api_url = LLAMA_SERVER_URL # Get URL from config
            if not llm_api_url or not llm_api_url.startswith("http"):
                raise ValueError(f"Invalid LLAMA_SERVER_URL found in config/environment: {llm_api_url}")

            agent = ReactAgent(llm_url=llm_api_url, system_prompt=DEFAULT_SYSTEM_PROMPT)
            logger.info("Agent ready.")
        except Exception as agent_init_err:
             logger.exception("Fatal error initializing ReactAgent:")
             print(f"[Fatal Error] Could not initialize the agent: {agent_init_err}")
             exit(1)
    # <<< FIM DA INSTANCIAÇÃO >>>

    # <<< LÓGICA DE EXECUÇÃO MODIFICADA >>>
    if args.stream_direct:
        # <<< ADDED: Direct Streaming Mode >>>
        logger.info(f"Streaming direct prompt: '{args.stream_direct[:50]}...'")
        print("--- Streaming LLM Response --- ")
        
        async def stream_direct_call():
            messages = [{"role": "user", "content": args.stream_direct}]
            try:
                # Import call_llm here to avoid potential issues if agent init fails
                from core.llm_interface import call_llm 
                async for chunk in call_llm(messages, stream=True):
                    print(chunk, end="", flush=True)
                print() # Final newline
            except Exception as stream_err:
                logger.exception("Error during direct LLM stream:")
                print(f"\n[Error Streaming] {stream_err}")
                
        asyncio.run(stream_direct_call())
        # <<< END ADDED >>>
        
    elif args.task:
        # Modo Tarefa Única (nova prioridade)
        logger.info(f"Executing single task: {args.task}")
        if agent:
            asyncio.run(process_command(agent, args.task, conversation_history))
        else:
            logger.error("Agent not initialized, cannot run task.")

    elif args.command:
        # Modo comando único
        logger.info(f"Executing single command: {args.command}")
        asyncio.run(process_command(agent, args.command, conversation_history))

    elif args.input_file:
        # Modo arquivo de entrada
        try:
            logger.info(f"Reading commands from: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"Found {len(commands)} commands to process.")

                for line_num, command in enumerate(commands, 1):
                    logger.info(f"--- Processing command from line {line_num} ---")
                    print(f"\n--- Command from Line {line_num} ---")
                    asyncio.run(process_command(agent, command, conversation_history))
                    time.sleep(1) # Pausa opcional
            logger.info("Finished processing input file.")
            print("\n[Info] End of input file.")
        except FileNotFoundError:
            logger.error(f"Input file not found: {args.input_file}")
            print(f"[Fatal Error] Input file not found: {args.input_file}")
        except Exception as file_proc_err:
            logger.exception(f"Error processing input file '{args.input_file}':")
            print(f"\n[Fatal Error] An error occurred while processing the file '{args.input_file}': {file_proc_err}")

    else:
        # Modo interativo
        print("Assistente CLI A³X (ReAct) iniciado. Digite 'sair' para encerrar.")
        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() in ['sair', 'exit', 'quit']:
                    logger.info("Exit command received. Shutting down.")
                    print("Exiting assistant...")
                    break
                if not command:
                    continue
                asyncio.run(process_command(agent, command, conversation_history))
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down.")
                print("\nExiting assistant...")
                break
            except Exception as loop_err:
                logger.exception("Unexpected error in main interactive loop:")
                print(f"\n[Unexpected Error] {loop_err}")
                # Decide se continua ou sai em caso de erro no loop principal
                # Por enquanto, continua

if __name__ == "__main__":
    main() 