# /home/arthur/Projects/A3X/cli/interface.py
import os
import sys
import asyncio
import argparse
import time
import logging
from dotenv import load_dotenv
from typing import Optional
import json

from rich.console import Console
from rich.panel import Panel

# Adiciona o diretório raiz ao sys.path para encontrar 'core' e 'skills'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Imports do Core (após adicionar ao path)
try:
    from core.config import MAX_HISTORY_TURNS, LOG_LEVEL, MEMORY_DB_PATH # Assuming DB_FILE and AGENT_STATE_ID are not needed directly here
    from core.db_utils import initialize_database
    from core.agent import ReactAgent
    from core.llm_interface import call_llm # Import para stream_direct
except ImportError as e:
    print(f"[CLI Interface FATAL] Failed to import core modules: {e}. Ensure PYTHONPATH is correct or run from project root.")
    sys.exit(1)

# <<< ADDED Import from new display module >>>
try:
    # Use relative import within the same package
    from .display import handle_agent_interaction, stream_direct_llm
except ImportError as e:
    print(f"[CLI Interface FATAL] Failed to import display module: {e}. Ensure cli/display.py exists.")
    sys.exit(1)

# <<< ADDED Import for logging setup >>>
try:
    from core.logging_config import setup_logging
except ImportError as e:
    print(f"[CLI Interface FATAL] Failed to import logging config: {e}.")
    sys.exit(1)

# <<< REMOVED Local Logging Setup >>>
# logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s CLI] %(message)s')
logger = logging.getLogger(__name__) # Keep getting the logger for this module

# Instância do Console Rich
console = Console()

# <<< REMOVED DEFAULT_SYSTEM_PROMPT DEFINITION >>>
# DEFAULT_SYSTEM_PROMPT = """..."""

# --- Helper Functions ---

def _load_system_prompt(file_path: str = "prompts/react_system_prompt.md") -> str:
    """Carrega o prompt do sistema de um arquivo."""
    full_path = os.path.join(project_root, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {full_path}")
        console.print(f"[bold red][Error][/] System prompt file not found at '{full_path}'. Using a minimal fallback.")
        return "You are a helpful assistant."
    except Exception as e:
        logger.exception(f"Error reading system prompt file {full_path}:")
        console.print(f"[bold red][Error][/] Could not read system prompt file: {e}. Using minimal fallback.")
        return "You are a helpful assistant."

def _parse_arguments():
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Assistente CLI A³X (ReAct)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')
    # Added interactive mode argument
    group.add_argument('--interactive', action='store_true', help='Inicia o modo interativo')
    parser.add_argument('--task', help='Tarefa única a ser executada pelo agente (prioridade sobre -c, -i, --interactive)')
    parser.add_argument("--stream-direct", help="Prompt direto para o LLM em modo streaming")
    return parser.parse_args()

def _initialize_agent(system_prompt: str) -> Optional[ReactAgent]:
    """Inicializa e retorna uma instância do ReactAgent."""
    logger.info("Initializing ReactAgent...")
    try:
        # Validação da URL do LLM removida
        # Assume que ReactAgent não precisa mais de llm_url se for interagir diretamente
        agent = ReactAgent(system_prompt=system_prompt)
        logger.info("Agent ready.")
        return agent
    except Exception as agent_init_err:
        logger.exception("Fatal error initializing ReactAgent:")
        console.print(f"[bold red][Fatal Error][/] Could not initialize the agent: {agent_init_err}")
        return None

def change_to_project_root():
    """Garante que o script seja executado a partir do diretório raiz do projeto."""
    try:
        if os.getcwd() != project_root:
            os.chdir(project_root)
        logger.info(f"Running in directory: {os.getcwd()}")
    except FileNotFoundError:
        logger.error(f"Project root directory not found: {project_root}. Exiting.")
        console.print(f"[bold red][Fatal Error][/] Project directory not found: {project_root}")
        sys.exit(1)
    except Exception as cd_err:
        logger.error(f"Error changing directory to {project_root}: {cd_err}. Exiting.")
        console.print(f"[bold red][Fatal Error][/] Could not change to project directory: {cd_err}")
        sys.exit(1)

async def _run_interactive_mode(agent: ReactAgent):
    """Executa o loop de interação com o usuário."""
    console.print("[bold green]Modo Interativo A³X.[/] Digite 'sair', 'exit' ou 'quit' para terminar.")
    conversation_history = [] # Reset history for interactive session
    while True:
        try:
            command = console.input("[bold magenta]>[/bold magenta] ")
            if command.lower() in ['sair', 'exit', 'quit']:
                break
            if not command:
                continue
            await handle_agent_interaction(agent, command, conversation_history)
        except KeyboardInterrupt:
            console.print("\nSaindo...")
            break
        except EOFError: # Handle Ctrl+D
            console.print("\nSaindo...")
            break

async def _process_input_file(agent: ReactAgent, file_path: str):
    """Processa comandos de um arquivo de entrada."""
    logger.info(f"Reading commands from: {file_path}")
    conversation_history = [] # Reset history
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                command = line.strip()
                if command and not command.startswith('#'): # Ignora linhas vazias e comentários
                    await handle_agent_interaction(agent, command, conversation_history)
                    console.rule() # Separador entre comandos do arquivo
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        console.print(f"[bold red]Erro:[/bold red] Arquivo de entrada não encontrado: '{file_path}'")
    except Exception as file_err:
        logger.exception(f"Error processing input file {file_path}:")
        console.print(f"[bold red]Erro:[/bold red] Falha ao processar arquivo de entrada: {file_err}")

def run_cli():
    """Função principal que configura e executa a interface de linha de comando."""
    load_dotenv() # Carrega .env
    setup_logging() # <<< CALL Centralized Logging Setup >>>
    change_to_project_root() # Garante que estamos no diretório certo

    args = _parse_arguments()

    # conversation_history = [] # History is managed within interaction loops now

    # Inicializa DB
    logger.info("Initializing database...")
    initialize_database()

    # Stream direto não precisa do agente
    if args.stream_direct:
        asyncio.run(stream_direct_llm(args.stream_direct))
        return

    # Carrega prompt do sistema e inicializa o agente para os outros modos
    system_prompt = _load_system_prompt()
    agent = _initialize_agent(system_prompt)

    if not agent:
        logger.error("Agent initialization failed. Exiting.")
        sys.exit(1) # Sai se o agente não puder ser inicializado

    # Lógica de Execução Principal refatorada
    if args.task:
        if args.task.strip().endswith(".json") and os.path.exists(args.task):
            logger.info(f"Loading structured task from JSON: {args.task}")
            try:
                with open(args.task, "r", encoding="utf-8") as f:
                    task = json.load(f)

                # Execução direta de skills
                skill_name = task.get("skill")
                action_name = task.get("action")
                params = task.get("params", {})

                if skill_name == "auto_publisher":
                    try:
                        from skills.auto_publisher import AutoPublisherSkill
                        skill = AutoPublisherSkill()

                        if action_name == "generate_and_publish":
                            topic = params.get("topic", "Untitled")
                            fmt = params.get("format", "markdown")
                            target = params.get("publish_target", "github")

                            try:
                                logger.info(f"Generating content for topic: {topic}")
                                content = skill.generate_content(topic)
                                logger.info(f"Content generated (length: {len(content)}). Exporting...")
                                filepath = skill.export_content(content, filename_prefix=topic.replace(" ", "_").lower(), format=fmt)
                                logger.info(f"Content exported to: {filepath}")

                                if target == "github":
                                    link = skill.publish_to_github(filepath)
                                elif target == "gumroad":
                                     link = skill.publish_to_gumroad(filepath)
                                else:
                                     link = "[ERROR] Unknown publish target specified."
                                     logger.error(f"Unknown publish target: {target}")


                                print(f"--- AutoPublisher Task Complete ---")
                                print(f"Topic: {topic}")
                                print(f"Output File: {filepath}")
                                print(f"Simulated Publish Target: {target}")
                                print(f"Simulated Link: {link}")
                                print(f"---------------------------------")

                            except Exception as skill_exec_err:
                                logger.exception(f"Error during AutoPublisherSkill execution:")
                                print(f"[ERROR] Failed to execute auto_publisher skill: {skill_exec_err}")

                        else:
                            logger.error(f"Action '{action_name}' not supported for skill '{skill_name}'.")
                            print(f"[ERROR] Action '{action_name}' not supported for skill '{skill_name}'.")

                    except ImportError:
                         logger.error("Could not import AutoPublisherSkill. Make sure skills/auto_publisher.py exists.")
                         print("[ERROR] Could not find the AutoPublisher skill module.")
                    except Exception as skill_init_err:
                         logger.exception("Error initializing AutoPublisherSkill:")
                         print(f"[ERROR] Failed to initialize AutoPublisher skill: {skill_init_err}")

                else:
                    logger.error(f"Direct execution for skill '{skill_name}' specified in {args.task} is not implemented.")
                    print(f"[ERROR] Skill '{skill_name}' not recognized for direct execution.")

            except json.JSONDecodeError:
                 logger.error(f"Invalid JSON file: {args.task}")
                 print(f"[ERROR] Could not parse JSON file: {args.task}")
            except FileNotFoundError:
                 # This case should ideally be caught by os.path.exists, but added for safety
                 logger.error(f"Task file not found (after initial check): {args.task}")
                 print(f"[ERROR] Task file not found: {args.task}")
            except Exception as task_load_err:
                 logger.exception(f"Error processing task file {args.task}:")
                 print(f"[ERROR] Failed to load or process task file: {task_load_err}")

        else:
            # Comportamento antigo: passa como texto pro ReAct
            logger.info(f"Executing freeform task via ReAct loop: {args.task}")
            asyncio.run(handle_agent_interaction(agent, args.task, [])) # Pass empty history

    elif args.command:
        logger.info(f"Executing single command via ReAct loop: {args.command}")
        asyncio.run(handle_agent_interaction(agent, args.command, [])) # Pass empty history

    elif args.input_file:
        asyncio.run(_process_input_file(agent, args.input_file))
    elif args.interactive:
        asyncio.run(_run_interactive_mode(agent))
    else:
        # Default to interactive mode if no other mode is specified
        asyncio.run(_run_interactive_mode(agent))

# Ponto de entrada do script
if __name__ == "__main__":
    run_cli()
