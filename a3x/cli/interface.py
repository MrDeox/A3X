# /home/arthur/Projects/A3X/cli/interface.py
import os
import sys
import asyncio
import argparse
import time
import logging
import subprocess
import atexit
from dotenv import load_dotenv
from typing import Optional, Tuple
import json
import importlib

from rich.console import Console
from rich.panel import Panel

# Adiciona o diretório raiz ao sys.path para encontrar 'core' e 'skills'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Imports do Core (após adicionar ao path)
try:
    # from core.config import (
    from a3x.core.config import (
        # MAX_HISTORY_TURNS, # F401
        # LOG_LEVEL, # F401
        # MEMORY_DB_PATH, # F401
        # LLM_PROVIDER, # F401
        LLAMA_MODEL_PATH as DEFAULT_MODEL_PATH,
        LLAMA_SERVER_URL as DEFAULT_SERVER_URL,
        CONTEXT_SIZE as DEFAULT_CONTEXT_SIZE,
    )
    # from core.db_utils import initialize_database
    from a3x.core.db_utils import initialize_database
    # from core.cerebrumx import CerebrumXAgent
    from a3x.core.cerebrumx import CerebrumXAgent
    # from core.llm_interface import call_llm # F401
    # from core.tools import load_skills # F401
    # from core.logging_config import setup_logging
    from a3x.core.logging_config import setup_logging
except ImportError as e:
    print(
        f"[CLI Interface FATAL] Failed to import core modules: {e}. Ensure PYTHONPATH is correct or run from project root."
    )
    sys.exit(1)

# <<< ADDED Import from new display module >>>
try:
    # Use relative import within the same package
    # from .display import handle_agent_interaction, stream_direct_llm
    from a3x.cli.display import handle_agent_interaction, stream_direct_llm
except ImportError as e:
    print(
        f"[CLI Interface FATAL] Failed to import display module: {e}. Ensure cli/display.py exists."
    )
    sys.exit(1)

# <<< ADDED Import for logging setup >>>
try:
    # from core.logging_config import setup_logging
    from a3x.core.logging_config import setup_logging
except ImportError as e:
    print(f"[CLI Interface FATAL] Failed to import logging config: {e}.")
    sys.exit(1)

# <<< REMOVED Local Logging Setup >>>
# logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s CLI] %(message)s')
logger = logging.getLogger(__name__)  # Keep getting the logger for this module

# Instância do Console Rich
console = Console()

# <<< REMOVED DEFAULT_SYSTEM_PROMPT DEFINITION >>>
# DEFAULT_SYSTEM_PROMPT = """..."""

# Load configuration and setup logging at the module level
# config = load_config()
# logger = setup_logging()

# <<< ADDED: Global variable for server process >>>
_llama_server_process: Optional[subprocess.Popen] = None

# --- Helper Functions ---


def _load_system_prompt(file_path: str = "prompts/react_system_prompt.md") -> str:
    """Carrega o prompt do sistema de um arquivo."""
    full_path = os.path.join(project_root, file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {full_path}")
        console.print(
            f"[bold red][Error][/] System prompt file not found at '{full_path}'. Using a minimal fallback."
        )
        return "You are a helpful assistant."
    except Exception as e:
        logger.exception(f"Error reading system prompt file {full_path}:")
        console.print(
            f"[bold red][Error][/] Could not read system prompt file: {e}. Using minimal fallback."
        )
        return "You are a helpful assistant."


def _parse_arguments():
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Assistente CLI A³X (ReAct)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--command", help="Comando único para executar")
    group.add_argument(
        "-i",
        "--input-file",
        help="Arquivo para ler comandos sequencialmente (um por linha)",
    )
    # Added interactive mode argument
    group.add_argument(
        "--interactive", action="store_true", help="Inicia o modo interativo"
    )
    parser.add_argument(
        "--task",
        help="Tarefa única a ser executada pelo agente (prioridade sobre -c, -i, --interactive)",
    )
    parser.add_argument(
        "--stream-direct", help="Prompt direto para o LLM em modo streaming"
    )
    # <<< ADDED model and gpu_layers arguments >>>
    parser.add_argument(
        "--model",
        help="Path para o arquivo do modelo GGUF a ser usado (sobrescreve config/default)",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Número de camadas a serem descarregadas na GPU (padrão: -1 = auto/todas)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Porta para iniciar o servidor LLM interno (padrão: 8080)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Não iniciar um servidor LLM interno, usar a URL configurada (LLAMA_SERVER_URL)",
    )
    return parser.parse_args()


def _initialize_agent(
    system_prompt: str, llm_url_override: Optional[str] = None
) -> Optional[CerebrumXAgent]:
    """Inicializa e retorna uma instância do CerebrumXAgent. Aceita override de URL."""
    logger.info("Initializing CerebrumXAgent...")
    try:
        # Pass the potentially overridden URL to the agent constructor
        agent = CerebrumXAgent(system_prompt=system_prompt, llm_url=llm_url_override)
        logger.info("CerebrumXAgent ready.")
        return agent
    except Exception as agent_init_err:
        logger.exception("Fatal error initializing CerebrumXAgent:")
        console.print(
            f"[bold red][Fatal Error][/] Could not initialize the agent: {agent_init_err}"
        )
        return None


def change_to_project_root():
    """Garante que o script seja executado a partir do diretório raiz do projeto."""
    try:
        if os.getcwd() != project_root:
            os.chdir(project_root)
        logger.info(f"Running in directory: {os.getcwd()}")
    except FileNotFoundError:
        logger.error(f"Project root directory not found: {project_root}. Exiting.")
        console.print(
            f"[bold red][Fatal Error][/] Project directory not found: {project_root}"
        )
        sys.exit(1)
    except Exception as cd_err:
        logger.error(f"Error changing directory to {project_root}: {cd_err}. Exiting.")
        console.print(
            f"[bold red][Fatal Error][/] Could not change to project directory: {cd_err}"
        )
        sys.exit(1)


async def _run_interactive_mode(agent: CerebrumXAgent):
    """Executa o loop de interação com o usuário."""
    console.print(
        "[bold green]Modo Interativo CerebrumX.[/] Digite 'sair', 'exit' ou 'quit' para terminar."
    )
    conversation_history = []  # Reset history for interactive session
    while True:
        try:
            command = console.input("[bold magenta]>[/bold magenta] ")
            if command.lower() in ["sair", "exit", "quit"]:
                break
            if not command:
                continue
            await handle_agent_interaction(agent, command, conversation_history)
        except KeyboardInterrupt:
            console.print("\nSaindo...")
            break
        except EOFError:  # Handle Ctrl+D
            console.print("\nSaindo...")
            break


async def _process_input_file(agent: CerebrumXAgent, file_path: str):
    """Processa comandos de um arquivo de entrada."""
    logger.info(f"Reading commands from: {file_path}")
    conversation_history = []  # Reset history
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                command = line.strip()
                if command and not command.startswith(
                    "#"
                ):  # Ignora linhas vazias e comentários
                    await handle_agent_interaction(agent, command, conversation_history)
                    console.rule()  # Separador entre comandos do arquivo
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        console.print(
            f"[bold red]Erro:[/bold red] Arquivo de entrada não encontrado: '{file_path}'"
        )
    except Exception as file_err:
        logger.exception(f"Error processing input file {file_path}:")
        console.print(
            f"[bold red]Erro:[/bold red] Falha ao processar arquivo de entrada: {file_err}"
        )


# <<< NEW: Argument Handler Functions >>>
async def _handle_task_argument(agent: CerebrumXAgent, task_arg: str):
    """Handles the --task argument logic."""
    if task_arg.strip().endswith(".json") and os.path.exists(task_arg):
        logger.info(f"Loading structured task(s) from JSON: {task_arg}")
        try:
            with open(task_arg, "r", encoding="utf-8") as f:
                tasks_input = json.load(f)

            tasks = []
            if isinstance(tasks_input, dict):  # Single task
                tasks.append(tasks_input)
            elif isinstance(tasks_input, list):  # List of tasks
                tasks = tasks_input
            else:
                raise TypeError(
                    "Task JSON must contain a single object or a list of objects."
                )

            # Process each task in the list
            for task_index, task in enumerate(tasks):
                logger.info(
                    f"Processing task {task_index + 1}/{len(tasks)}: ID '{task.get('id', 'N/A')}'"
                )

                skill_name = task.get("skill")
                # action_name = task.get("action") # Not used in this dynamic logic path
                params = task.get("parameters", {})

                # --- Dynamic Skill Execution Logic ---
                # Determine module name (simple heuristic)
                # This logic might need better refinement or a central registry lookup
                if skill_name in [
                    "open_url",
                    "click_element",
                    "fill_form_field",
                    "get_page_content",
                    "close_browser",
                ]:
                    module_name = "browser_skill"
                elif skill_name == "auto_publisher":
                    module_name = "auto_publisher"
                # Add more mappings if needed
                else:
                    module_name = skill_name  # Assume module name matches skill name

                module_path = f"skills.{module_name}"
                function_name = skill_name  # Assume function name matches skill name

                result = {
                    "status": "error",
                    "message": f"Skill '{skill_name}' could not be executed.",
                }
                try:
                    skill_module = importlib.import_module(module_path)
                    if hasattr(skill_module, function_name):
                        skill_function = getattr(skill_module, function_name)

                        if asyncio.iscoroutinefunction(skill_function):
                            logger.info(
                                f"Executing async skill '{skill_name}' with params: {params}"
                            )
                            result = await skill_function(**params)  # Use await
                        else:
                            logger.info(
                                f"Executing sync skill '{skill_name}' with params: {params}"
                            )
                            # Run sync function in executor to avoid blocking asyncio loop
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, skill_function, **params
                            )

                        # Ensure result is a dict for consistent processing
                        if not isinstance(result, dict):
                            logger.warning(
                                f"Skill '{skill_name}' did not return a dict. Wrapping output."
                            )
                            result = {"status": "success", "message": str(result)}

                    else:
                        # Fallback logic (e.g., for class-based skills like AutoPublisher) might go here
                        # or raise an error.
                        logger.error(
                            f"Function '{function_name}' not found in module '{module_path}'. Skill execution failed."
                        )
                        raise AttributeError(
                            f"Function '{function_name}' not found in '{module_path}'."
                        )

                except ModuleNotFoundError:
                    logger.error(f"Skill module not found: {module_path}")
                    result = {
                        "status": "error",
                        "message": f"Skill module '{module_path}' not found.",
                    }
                except AttributeError as e:
                    logger.error(str(e))
                    result = {"status": "error", "message": str(e)}
                except Exception as e:
                    logger.exception(
                        f"Error executing skill '{skill_name}' dynamically:"
                    )
                    result = {
                        "status": "error",
                        "message": f"Failed to execute skill: {e}",
                    }

                # Display result
                logger.info(
                    f"Task {task_index + 1} result: {result.get('status', 'N/A')}"
                )
                console.print(
                    Panel(
                        json.dumps(result, indent=2, ensure_ascii=False),
                        title=f"Task {task_index + 1} ({skill_name}) Result",
                        border_style="green"
                        if result.get("status") == "success"
                        else "red",
                    )
                )
                console.rule()  # Separator between tasks

        except FileNotFoundError:
            logger.error(f"Task JSON file not found: {task_arg}")
            console.print(
                f"[bold red]Erro:[/bold red] Arquivo JSON da tarefa não encontrado: '{task_arg}'"
            )
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in task file: {task_arg}")
            console.print(
                f"[bold red]Erro:[/bold red] JSON inválido no arquivo da tarefa: '{task_arg}'"
            )
        except TypeError as type_err:
            logger.error(f"Invalid task format in JSON file: {type_err}")
            console.print(f"[bold red]Erro:[/bold red] {type_err}")
        except Exception as task_err:
            logger.exception(f"Error processing task JSON {task_arg}:")
            console.print(
                f"[bold red]Erro:[/bold red] Falha ao processar tarefas do JSON: {task_err}"
            )

    else:  # Direct task string (treat as a command for the agent)
        logger.info(f"Running single task from command line: {task_arg}")
        await handle_agent_interaction(agent, task_arg, [])  # Pass empty history


async def _handle_command_argument(agent: CerebrumXAgent, command_arg: str):
    """Handles the -c or --command argument."""
    logger.info(f"Executing single command: {command_arg}")
    await handle_agent_interaction(agent, command_arg, [])


async def _handle_file_argument(agent: CerebrumXAgent, file_arg: str):
    """Handles the -i or --input-file argument."""
    await _process_input_file(agent, file_arg)


async def _handle_interactive_argument(agent: CerebrumXAgent):
    """Handles the --interactive argument or default behavior."""
    await _run_interactive_mode(agent)


# <<< ADDED: Function to start llama.cpp server >>>
def _start_llama_server(
    model_path: str,
    gpu_layers: int,
    port: int,
    context_size: int = DEFAULT_CONTEXT_SIZE,  # Use from config
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Starts the llama.cpp server as a background process."""
    global _llama_server_process

    # Find the llama-server executable relative to the project root
    # Assumes it's at project_root/llama.cpp/build/bin/llama-server
    server_executable = os.path.join(
        project_root, "llama.cpp", "build", "bin", "llama-server"
    )

    if not os.path.exists(server_executable):
        logger.error(f"llama-server executable not found at: {server_executable}")
        console.print(
            f"[bold red][Error][/] llama-server not found at expected location: {server_executable}. Please build llama.cpp."
        )
        return None, None

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        console.print(f"[bold red][Error][/] Model file not found: {model_path}")
        return None, None

    command = [
        server_executable,
        "--model",
        model_path,
        "--ctx-size",
        str(context_size),
        "--n-gpu-layers",
        str(gpu_layers),
        "--port",
        str(port),
        # Add other necessary/default arguments for llama-server
        "--host",
        "127.0.0.1",  # Bind to localhost
        # "--embedding", # Add if embedding endpoint is needed
        # "--verbose", # Add for more server logs if needed
    ]

    logger.info(
        f"Starting internal llama-server on port {port} with model '{os.path.basename(model_path)}' ({gpu_layers} GPU layers...)"
    )
    logger.debug(f"Server command: {' '.join(command)}")

    try:
        # Use Popen for background process, redirect stdout/stderr to PIPE or DEVNULL
        # Redirecting to DEVNULL to keep CLI clean, check server logs if needed
        _llama_server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,  # Or subprocess.PIPE to capture logs
            stderr=subprocess.PIPE,  # Capture errors to check for startup issues
        )

        # Short pause to allow server to start and potentially fail
        time.sleep(3)

        # Check if the process terminated unexpectedly
        if _llama_server_process.poll() is not None:
            stderr_output = (
                _llama_server_process.stderr.read().decode()
                if _llama_server_process.stderr
                else "No stderr output."
            )
            logger.error(
                f"Internal llama-server failed to start. Exit code: {_llama_server_process.returncode}"
            )
            logger.error(f"Server stderr:\n{stderr_output}")
            console.print(
                "[bold red][Fatal Error][/] Internal llama-server failed to start. Check logs/stderr."
            )
            _llama_server_process = None
            return None, None

        # Register cleanup function
        atexit.register(_stop_llama_server)

        server_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        logger.info(
            f"Internal llama-server started successfully. PID: {_llama_server_process.pid}. URL: {server_url}"
        )
        return _llama_server_process, server_url

    except Exception as e:
        logger.exception("Failed to start internal llama-server:")
        console.print(
            f"[bold red][Fatal Error][/] Could not start internal llama-server: {e}"
        )
        return None, None


# <<< ADDED: Function to stop llama.cpp server >>>
def _stop_llama_server():
    """Stops the background llama.cpp server process if it's running."""
    global _llama_server_process
    if _llama_server_process and _llama_server_process.poll() is None:
        logger.info(
            f"Stopping internal llama-server (PID: {_llama_server_process.pid})..."
        )
        try:
            # Try terminating gracefully first
            _llama_server_process.terminate()
            try:
                # Wait a bit for termination
                _llama_server_process.wait(timeout=5)
                logger.info("Internal llama-server terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Internal llama-server did not terminate gracefully, killing..."
                )
                _llama_server_process.kill()
                logger.info("Internal llama-server killed.")
        except Exception as e:
            logger.error(f"Error stopping internal llama-server: {e}", exc_info=True)
        finally:
            _llama_server_process = None


# <<< REFACTORED run_cli Function >>>
def run_cli():
    """Função principal que configura e executa a interface de linha de comando."""
    load_dotenv()
    setup_logging()
    change_to_project_root()

    args = _parse_arguments()

    # --- Determine LLM Server URL ---
    llm_server_url_to_use = None
    server_process = None

    if not args.no_server:
        # Determine model path (CLI arg > config > default)
        model_path_to_use = args.model or DEFAULT_MODEL_PATH
        # Determine GPU layers (CLI arg > default)
        gpu_layers_to_use = args.gpu_layers  # Default is -1 in parser

        server_process, internal_server_url = _start_llama_server(
            model_path=model_path_to_use,
            gpu_layers=gpu_layers_to_use,
            port=args.port,
            context_size=DEFAULT_CONTEXT_SIZE,
        )
        if not server_process:
            logger.critical("Failed to start internal LLM server. Exiting.")
            sys.exit(1)
        llm_server_url_to_use = internal_server_url
    else:
        # Use external server URL (config > default)
        llm_server_url_to_use = DEFAULT_SERVER_URL  # Rely on config/env var or default
        logger.info(f"Using external LLM server configured at: {llm_server_url_to_use}")

    # --- Initialize DB ---
    logger.info("Initializing database...")
    initialize_database()

    # --- Handle stream-direct (might need URL override if using internal server) ---
    if args.stream_direct:
        logger.info(f"Streaming direct from LLM at {llm_server_url_to_use}...")
        # TODO: Modify stream_direct_llm if it needs the URL passed explicitly
        try:
            asyncio.run(
                stream_direct_llm(
                    args.stream_direct, llm_url_override=llm_server_url_to_use
                )
            )  # Pass URL
        except Exception as stream_err:
            logger.exception("Error during direct streaming:")
            console.print(f"[bold red]Error during direct streaming: {stream_err}[/]")
        finally:
            # Ensure server stops if started internally
            if server_process:
                _stop_llama_server()
        return

    # --- Agent Initialization (Common for other modes) ---
    system_prompt = _load_system_prompt()
    # Pass the determined URL to the agent initializer
    agent = _initialize_agent(system_prompt, llm_url_override=llm_server_url_to_use)

    if not agent:
        logger.error("Agent initialization failed. Exiting.")
        if server_process:
            _stop_llama_server()  # Stop server if agent fails
        sys.exit(1)

    # --- Dispatch based on arguments ---
    try:
        if args.task:
            asyncio.run(_handle_task_argument(agent, args.task))
        elif args.command:
            asyncio.run(_handle_command_argument(agent, args.command))
        elif args.input_file:
            asyncio.run(_handle_file_argument(agent, args.input_file))
        elif args.interactive or not (args.command or args.input_file or args.task):
            asyncio.run(_handle_interactive_argument(agent))
        else:
            logger.error(
                "Inconsistent argument state. Cannot determine execution mode. Exiting."
            )
            sys.exit(1)
    except Exception as main_run_err:
        logger.exception("An error occurred during the main execution loop:")
        console.print(f"[bold red][Runtime Error][/] {main_run_err}")
    finally:
        # Ensure server stops if started internally, regardless of execution outcome
        if server_process:
            _stop_llama_server()


if __name__ == "__main__":
    # Ensure clean exit on Ctrl+C
    try:
        run_cli()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping server if running...")
        _stop_llama_server()  # Attempt graceful shutdown
        print("\nExiting cleanly.")
        sys.exit(0)
