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
from typing import Optional, Tuple, AsyncGenerator, Set, Dict, Any
import json
import importlib
import ast
import requests

from rich.console import Console
from rich.panel import Panel

# Adiciona o diretório raiz ao sys.path para encontrar 'core' e 'skills'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
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

    # <<< Replace commented out/incorrect tool import >>>
    from a3x.core.tools import (
        load_skills,
        get_skill_registry,
        get_tool_descriptions,
        SKILL_REGISTRY, # Import the registry itself if needed directly
    )
    # from core.logging_config import setup_logging
    from a3x.core.logging_config import setup_logging
    from a3x.core.tool_executor import execute_tool
    from a3x.core.llm_interface import call_llm
except ImportError as e:
    # <<< ADDED: Initialize logger here BEFORE using it >>>
    # Need a basic logger config if setup_logging hasn't run yet
    import logging
    logging.basicConfig(level=logging.INFO) # Basic config
    logger = logging.getLogger(__name__) # Get the logger instance
    # <<< END ADDED >>>

    print(
        f"[CLI Interface FATAL] Failed to import core modules: {e}. Ensure PYTHONPATH is correct or run from project root."
    )
    # Attempt to log the warning AFTER logger is defined
    logger.warning(f"Could not import training module: {e}. --train command will be unavailable.")
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

# <<< ADDED: Import trainer function >>>
try:
    from a3x.training.trainer import run_qlora_finetuning
except ImportError as e:
    run_qlora_finetuning = None # Define as None if import fails
    # <<< ADDED: Initialize logger here BEFORE using it in this block >>>
    import logging
    # Use a basic config temporarily if full setup hasn't happened
    logging.basicConfig(level=logging.WARNING)
    _temp_logger = logging.getLogger(__name__) # Use temp name to avoid conflict if main logger exists later
    _temp_logger.warning(f"Could not import training module: {e}. --train command will be unavailable.")
    # <<< END ADDED >>>

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
_llava_server_process: Optional[subprocess.Popen] = None # Added for LLaVA

# <<< ADDED: LLaVA Server Configuration >>>
LLAVA_PORT = 8081
LLAVA_SERVER_URL = f"http://127.0.0.1:{LLAVA_PORT}"
# <<< NEW LLaVA Stack Config >>>
LLAVA_CONTROLLER_PORT = 10000 # Default LLaVA controller port
LLAVA_WORKER_PORT = 21002   # Default LLaVA worker port
LLAVA_API_PORT = 9999       # Target OpenAI API port
LLAVA_API_URL = f"http://localhost:{LLAVA_API_PORT}/v1"
LLAVA_MODEL_DIR_RELATIVE = "llava-1.5-7b" # Directory name for downloaded model
LLAVA_REPO_DIR_RELATIVE = "LLaVA" # Directory name for cloned LLaVA repo
_llava_controller_process: Optional[subprocess.Popen] = None
_llava_worker_process: Optional[subprocess.Popen] = None
_llava_api_server_process: Optional[subprocess.Popen] = None
# <<< END NEW LLaVA Stack Config >>>

LLAVA_MODEL_DIR_RELATIVE = "LLaVA/llava-v1.5-7b" # Relative to project root

# <<< ADDED: Minimal Context for Direct Skill Call >>>
from collections import namedtuple
SkillContext = namedtuple("SkillContext", ["logger", "llm_call"])

# Modified wrapper to be an async generator and pass stream=True
async def _direct_llm_call_wrapper(prompt: str) -> AsyncGenerator[str, None]:
    """Wrapper for call_llm to format prompt and enable streaming.
       Yields response chunks.
    """
    messages = [{"role": "user", "content": prompt}]
    logger.debug("Direct LLM Wrapper: Calling call_llm with stream=True")
    try:
        async for chunk in call_llm(messages, stream=True):
            yield chunk
    except Exception as e:
        logger.error(f"Error in _direct_llm_call_wrapper (streaming): {e}", exc_info=True)
        yield f"[LLM Call Error: {e}]"

# --- Helper Functions ---


def _load_system_prompt(file_path: str = "prompts/react_system_prompt.md") -> str:
    """Carrega o prompt do sistema de um arquivo."""
    full_path = os.path.join(project_root, "a3x", file_path)
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
    group.add_argument(
        "--interactive", action="store_true", help="Inicia o modo interativo"
    )
    group.add_argument(
        "--task",
        help="Tarefa única a ser executada pelo agente (substitui os modos command/input-file/interactive)",
    )
    group.add_argument(
        "--stream-direct", help="Prompt direto para o LLM em modo streaming"
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Executa um ciclo de fine-tuning QLoRA usando dados do buffer de experiências."
    )
    # <<< ADDED: Temporary direct skill execution arguments >>>
    group.add_argument(
        "--run-skill",
        help="(TESTING) Nome da skill a ser executada diretamente."
    )
    parser.add_argument(
        "--skill-args",
        default="{}",
        help="(TESTING) Argumentos para a skill em formato JSON string (usado com --run-skill). Ex: '{\"param\":\"value\"}'"
    )
    # <<< END ADDED >>>

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
    system_prompt: str, llm_url_override: Optional[str] = None, tools_dict: Optional[Dict[str, Any]] = None
) -> Optional[CerebrumXAgent]:
    """Inicializa e retorna uma instância do CerebrumXAgent. Aceita override de URL."""
    logger.info("Initializing CerebrumXAgent...")
    try:
        # Pass the potentially overridden URL to the agent constructor
        agent = CerebrumXAgent(system_prompt=system_prompt, llm_url=llm_url_override, tools_dict=tools_dict)
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
                        border_style=(
                            "green" if result.get("status") == "success" else "red"
                        ),
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
    mmproj_path: Optional[str] = None # <<< ADDED: Multimodal projector path
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Starts the llama.cpp server as a background process, optionally with multimodal support."""
    global _llama_server_process

    # Correct the path to the NEWLY COMPILED llama-server executable
    server_executable = os.path.join(project_root, "llama.cpp", "build", "bin", "llama-server") # Updated path

    if not os.path.exists(server_executable):
        logger.error(f"llama-server executable not found at: {server_executable}")
        console.print(
            f"[bold red][Error][/] llama-server not found at expected location: {server_executable}. Please build llama.cpp."
        )
        return None, None

    # ADDED: Log the exact model path being checked
    logger.info(f"Checking existence of model file at absolute path: {os.path.abspath(model_path)}")

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

    # <<< ADDED: Conditionally add multimodal projector argument >>>
    if mmproj_path:
        if os.path.exists(mmproj_path):
            logger.info(f"Adding multimodal projector: {mmproj_path}")
            command.extend(["--mmproj", mmproj_path])
        else:
            logger.error(f"Multimodal projector file not found at: {mmproj_path}")
            console.print(f"[bold red][Error][/] MMPROJ file not found: {mmproj_path}")
            # Continue without mmproj? Or fail? Let's fail for now if specified but missing.
            return None, None
    # <<< END ADDED >>>

    logger.info(
        f"Starting internal llama-server on port {port} with model '{os.path.basename(model_path)}' ({gpu_layers} GPU layers...)"
    )
    logger.debug(f"Server command: {' '.join(command)}")

    try:
        # Get current environment and add LD_LIBRARY_PATH for the new build location
        env = os.environ.copy()
        lib_path = os.path.join(project_root, "llama.cpp", "build", "bin") # Updated path to bin dir
        # Handle existing LD_LIBRARY_PATH
        env['LD_LIBRARY_PATH'] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(':')
        logger.info(f"Setting LD_LIBRARY_PATH for subprocess: {env['LD_LIBRARY_PATH']}")

        # Open log files for server output
        stdout_log_path = os.path.join(project_root, "llama_server_stdout.log")
        stderr_log_path = os.path.join(project_root, "llama_server_stderr.log")
        stdout_log = open(stdout_log_path, 'wb')
        stderr_log = open(stderr_log_path, 'wb')
        logger.info(f"Redirecting llama-server stdout to: {stdout_log_path}")
        logger.info(f"Redirecting llama-server stderr to: {stderr_log_path}")

        # Use Popen for background process, redirecting to log files
        _llama_server_process = subprocess.Popen(
            command,
            env=env, # Pass the modified environment
            stdout=stdout_log,  # Redirect stdout to file
            stderr=stderr_log,  # Redirect stderr to file
        )

        # Short pause to allow server to start and potentially fail - INCREASED DELAY
        time.sleep(15) # Increased from 3 to 15 seconds

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


# <<< ADDED: Function to check LLM server health >>>
def _check_llm_server_health(url: str = DEFAULT_SERVER_URL, timeout: float = 2.0) -> bool:
    """Checks if the default LLM (llama.cpp) server is responding at the health endpoint."""
    # Construct health check URL (assuming /health endpoint)
    health_url = url.replace("/v1/chat/completions", "/health")
    logger.debug(f"Checking LLM server health at: {health_url}")
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"LLM server is healthy (responded from {health_url}).")
            return True
        else:
            logger.warning(f"LLM server health check failed at {health_url}. Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.info(f"LLM server connection refused at {health_url}. Server likely not running.")
        return False
    except requests.exceptions.Timeout:
        logger.warning(f"LLM server health check timed out at {health_url} (timeout={timeout}s).")
        return False
    except Exception as e:
        logger.error(f"Error checking LLM server health at {health_url}: {e}", exc_info=True)
        return False


# --- LLaVA Server Management (Stack Version) ---

def _start_llava_stack(
    model_repo_path: str, # e.g., project_root/llava-1.5-7b
    llava_repo_dir: str, # e.g., project_root/LLaVA
    controller_port: int = LLAVA_CONTROLLER_PORT,
    worker_port: int = LLAVA_WORKER_PORT,
    api_port: int = LLAVA_API_PORT,
    num_workers: int = 1 # Default to one worker
) -> bool:
    """Starts the LLaVA controller, model worker(s), and OpenAI API server."""
    global _llava_controller_process, _llava_worker_process, _llava_api_server_process

    controller_ready = False
    worker_ready = False
    api_server_ready = False

    controller_log_path = os.path.join(project_root, "llava_controller.log")
    worker_log_path = os.path.join(project_root, "llava_worker.log")
    api_server_log_path = os.path.join(project_root, "llava_api_server.log")

    try:
        # 1. Start Controller
        logger.info(f"[LLaVA Stack] Starting Controller on port {controller_port}...")
        controller_cmd = [
            sys.executable, "-m", "llava.serve.controller",
            "--host", "0.0.0.0", "--port", str(controller_port)
        ]
        _llava_controller_process = subprocess.Popen(
            controller_cmd, cwd=llava_repo_dir,
            stdout=open(controller_log_path, 'wb'), stderr=subprocess.STDOUT
        )
        time.sleep(5) # Allow controller to initialize
        if _llava_controller_process.poll() is not None:
            logger.error("[LLaVA Stack] Controller failed to start. Check llava_controller.log.")
            raise RuntimeError("LLaVA Controller failed")
        logger.info(f"[LLaVA Stack] Controller started (PID: {_llava_controller_process.pid}).")
        controller_ready = True

        # 2. Start Model Worker
        logger.info(f"[LLaVA Stack] Starting Model Worker (model: {model_repo_path})...")
        worker_cmd = [
            sys.executable, "-m", "llava.serve.model_worker",
            "--host", "0.0.0.0",
            "--controller-address", f"http://localhost:{controller_port}",
            "--port", str(worker_port),
            "--worker-address", f"http://localhost:{worker_port}",
            "--model-path", model_repo_path,
            "--num-gpus", str(num_workers) # Assuming 1 worker = 1 GPU for simplicity, adjust if needed
            # Add --device vulkan if worker supports it explicitly?
        ]
        _llava_worker_process = subprocess.Popen(
            worker_cmd, cwd=llava_repo_dir,
            stdout=open(worker_log_path, 'wb'), stderr=subprocess.STDOUT
        )
        time.sleep(20) # Allow worker time to load model
        if _llava_worker_process.poll() is not None:
            logger.error("[LLaVA Stack] Model Worker failed to start. Check llava_worker.log.")
            raise RuntimeError("LLaVA Model Worker failed")
        logger.info(f"[LLaVA Stack] Model Worker started (PID: {_llava_worker_process.pid}).")
        worker_ready = True

        # 3. Start OpenAI API Server
        logger.info(f"[LLaVA Stack] Starting OpenAI API Server on port {api_port}...")
        api_server_cmd = [
            sys.executable, "-m", "llava.serve.openai_api_server",
            "--host", "0.0.0.0",
            "--controller-address", f"http://localhost:{controller_port}",
            "--port", str(api_port)
        ]
        _llava_api_server_process = subprocess.Popen(
            api_server_cmd, cwd=llava_repo_dir,
            stdout=open(api_server_log_path, 'wb'), stderr=subprocess.STDOUT
        )
        time.sleep(5) # Allow API server to initialize
        if _llava_api_server_process.poll() is not None:
            logger.error("[LLaVA Stack] API Server failed to start. Check llava_api_server.log.")
            raise RuntimeError("LLaVA API Server failed")
        logger.info(f"[LLaVA Stack] OpenAI API Server started (PID: {_llava_api_server_process.pid}). Endpoint: {LLAVA_API_URL}")
        api_server_ready = True

        # Register cleanup function only if all parts started
        atexit.register(_stop_llava_stack)
        logger.info("[LLaVA Stack] Successfully started Controller, Worker, and API Server.")
        return True

    except Exception as e:
        logger.exception("[LLaVA Stack] Failed to start the full LLaVA stack:")
        console.print(f"[bold red][Error][/] Could not start LLaVA server stack: {e}")
        # Attempt to clean up any parts that might have started
        _stop_llava_stack()
        return False

def _stop_llava_stack():
    """Stops the LLaVA controller, worker, and API server processes."""
    global _llava_controller_process, _llava_worker_process, _llava_api_server_process
    processes = {
        "API Server": _llava_api_server_process,
        "Model Worker": _llava_worker_process,
        "Controller": _llava_controller_process,
    }
    # Stop in reverse order of start
    for name, process in processes.items():
        if process and process.poll() is None:
            logger.info(f"[LLaVA Stack] Stopping {name} (PID: {process.pid})...")
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"[LLaVA Stack] {name} terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(f"[LLaVA Stack] {name} did not terminate gracefully, killing...")
                process.kill()
                logger.info(f"[LLaVA Stack] {name} killed.")
            except Exception as e:
                logger.error(f"[LLaVA Stack] Error stopping {name}: {e}", exc_info=True)
        # Clear the global variable regardless
        if name == "API Server": _llava_api_server_process = None
        elif name == "Model Worker": _llava_worker_process = None
        elif name == "Controller": _llava_controller_process = None

def _check_llava_api_health(url: str = LLAVA_API_URL, timeout: float = 5.0) -> bool:
    """Checks if the LLaVA OpenAI API server endpoint is responding."""
    health_check_endpoint = f"{url}/models" # Standard OpenAI endpoint
    logger.debug(f"Checking LLaVA API health at: {health_check_endpoint}")
    try:
        response = requests.get(health_check_endpoint, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"LLaVA API Server is healthy (responded from {health_check_endpoint}).")
            return True
    except requests.exceptions.ConnectionError:
        logger.info(f"LLaVA API server connection refused at {health_check_endpoint}. Server likely not running.")
        return False
    except requests.exceptions.Timeout:
        logger.warning(f"LLaVA API health check timed out at {health_check_endpoint} (timeout={timeout}s).")
        return False
    except Exception as e:
        logger.error(f"Error checking LLaVA API health at {health_check_endpoint}: {e}", exc_info=True)
        return False

# --- End LLaVA Server Management ---

def run_cli():
    """Função principal da CLI."""
    setup_logging() # Ensure logging is setup
    args = _parse_arguments()
    change_to_project_root()
    initialize_database() # Ensure DB is ready

    # <<< MOVED: Load Skills Explicitly BEFORE Agent Init >>>
    # Determine skills directory path (still potentially useful for discovery)
    skills_dir_path = os.path.join(project_root, "a3x", "skills")
    # Pass the package name string, not the directory path
    load_skills("a3x.skills") # Explicit call here
    # <<< Get the registry AFTER loading skills >>>
    loaded_skill_registry = get_skill_registry()

    # --- Initialize Agent FIRST ---
    logger.info("Initializing Agent and loading skills BEFORE server start...")
    system_prompt = _load_system_prompt()
    # Initialize agent with the default URL for now. The actual URL for calls might
    # be determined later if the internal server starts.
    # <<< Pass the loaded registry to the agent constructor >>>
    agent = _initialize_agent(system_prompt, llm_url_override=DEFAULT_SERVER_URL, tools_dict=loaded_skill_registry)

    if agent is None:
        logger.error("Agent initialization returned None. Critical failure.")
        sys.exit(1) # Agent initialization failed

    # <<< Validate skills registration AFTER agent init >>>
    expected_skills_set = discover_expected_skills(skills_dir_path)
    validate_registered_skills(expected_skills_set, agent.tools) # Pass agent.tools directly
    # <<< END Validation >>>

    # --- Server Management and Execution Modes --- 
    server_process = None
    llm_url = DEFAULT_SERVER_URL # Start with default

    # >>> LLM Server Management <<<
    # Only start server if needed and not running a skill directly or training
    # (Direct skill runs might use a minimal context or mock, training doesn't need agent server)
    should_start_server = not args.no_server and not args.run_skill and not args.train and not args.stream_direct

    if should_start_server:
        model_path = args.model or DEFAULT_MODEL_PATH
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        logger.info("Attempting to start internal LLM server...") # Added log
        server_process, llm_url_from_server = _start_llama_server(
            model_path=model_path,
            gpu_layers=args.gpu_layers,
            port=args.port
        )
        if server_process and llm_url_from_server:
            llm_url = llm_url_from_server # Update URL if server started
            logger.info(f"Internal server started. LLM URL set to: {llm_url}")
            # Ensure server is stopped on exit - register only if started
            atexit.register(_stop_llama_server)
        else:
            logger.error("Failed to start internal LLM server. Continuing with default URL if possible, but agent may fail.")
            # Don't exit here, allow modes that might not need server (though agent likely will)
            # console.print("[bold red][Error][/] Failed to start internal LLM server. Exiting.")
            # sys.exit(1)
    elif args.no_server:
        logger.info(f"Skipping internal server start, using configured URL: {llm_url}")
    # >>> End LLM Server Management <<<

    # --- Mode Execution ---
    # Agent is already initialized, but we might need to pass the potentially updated llm_url?
    # For now, assume agent reads URL dynamically or was configured sufficiently at init.

    if args.run_skill:
        # <<< MODIFIED: Server Check/Start Logic for --run-skill >>>
        server_type = "LLaMA" # Default server type
        server_check_func = _check_llm_server_health
        server_start_func = _start_llama_server
        server_stop_func = _stop_llama_server
        server_port = args.port # Default port from args
        server_model_path = args.model or DEFAULT_MODEL_PATH
        server_gpu_layers = args.gpu_layers # Default GPU layers
        server_mmproj_path = None # Default to no mmproj
        server_start_success = False # Flag to track if start was successful

        # Determine model and potentially mmproj based on skill
        if args.run_skill == "visual_perception":
            logger.info(f"Skill is '{args.run_skill}', configuring for Obsidian multimodal model.")
            # Override model and add mmproj path for visual_perception
            # Ideally, these would come from config, but hardcoding for now
            obsidian_model_rel = "models/obsidian-3b/obsidian-q6.gguf"
            obsidian_mmproj_rel = "models/obsidian-3b/mmproj-obsidian-f16.gguf"
            # Correct paths based on user-attached directory structure
            obsidian_model_rel = "models/obsidian-q6.gguf"
            obsidian_mmproj_rel = "models/mmproj-obsidian-f16.gguf"
            server_model_path = os.path.join(project_root, obsidian_model_rel)
            server_mmproj_path = os.path.join(project_root, obsidian_mmproj_rel)
            server_port = args.port # Use the potentially user-specified port for unified server
            logger.info(f"Using model: {server_model_path}")
            logger.info(f"Using mmproj: {server_mmproj_path}")
            # <<< Override GPU layers specifically for Obsidian >>>
            server_gpu_layers = 27 # Hardcoded override based on user input
            logger.info(f"Setting GPU layers specifically to {server_gpu_layers} for Obsidian.")
        else:
            logger.info(f"Skill '{args.run_skill}' requires default text LLM server (llama.cpp).")
            # Ensure default model path is absolute
            if not os.path.isabs(server_model_path):
                server_model_path = os.path.join(project_root, server_model_path)
            # No mmproj for standard skills
            server_mmproj_path = None
            # Use GPU layers from command line args for other skills
            server_gpu_layers = args.gpu_layers

        # Prepare arguments for the unified server start function
        server_specific_start_args = {
            "model_path": server_model_path,
            "gpu_layers": server_gpu_layers,
            "port": server_port,
            "mmproj_path": server_mmproj_path # Pass mmproj path (will be None if not needed)
        }

        # <<< SERVER CHECK/START LOGIC (Unified) >>>
        server_needed_and_started = False
        # Check health on the correct port for the potentially multimodal server
        if not _check_llm_server_health(url=f"http://127.0.0.1:{server_port}/v1/chat/completions"):
            logger.info(f"--run-skill mode: {server_type} server (port {server_port}) not detected. Attempting to start...")
            # Attempt to start the server
            local_server_process, server_url_from_start = server_start_func(**server_specific_start_args)

            if local_server_process and server_url_from_start:
                logger.info(f"Internal {server_type} server started successfully for --run-skill.")
                server_needed_and_started = True # Mark that we started it
            else:
                # This else belongs to the inner if (start attempt check)
                logger.error(f"Failed to start internal {server_type} server for --run-skill. Skill execution will likely fail.")
                console.print(f"[bold red]Warning:[/bold red] Failed to start {server_type} server. The skill might fail.")
        # <<< END SERVER CHECK/START LOGIC >>>

        # <<< SKILL EXECUTION BLOCK >>>
        try:
            skill_args_dict = json.loads(args.skill_args)
            if not isinstance(skill_args_dict, dict):
                 raise ValueError("Skill arguments must be a JSON object (dictionary).")

            logger.info(f"Attempting direct execution of skill: '{args.run_skill}' with args: {skill_args_dict}")

            # Get skill function directly
            skill_info = agent.tools.get(args.run_skill)
            if not skill_info or not callable(skill_info.get("function")):
                logger.error(f"Skill '{args.run_skill}' not found or function is not callable.")
                console.print(f"[bold red]Error:[/bold red] Skill '{args.run_skill}' not found or invalid.")
                sys.exit(1) # Correctly indented

            skill_function = skill_info["function"]

            # Create minimal context using the streaming wrapper
            skill_ctx = SkillContext(logger=agent.agent_logger, llm_call=_direct_llm_call_wrapper)

            # Call the skill function directly
            async def run_the_skill_directly():
                if asyncio.iscoroutinefunction(skill_function):
                    # Skill is async, call it.
                    result = await skill_function(ctx=skill_ctx, **skill_args_dict)
                else:
                    # Handle sync skill case if necessary, though visual_perception is async
                    logger.warning(f"Attempting to run synchronous skill '{args.run_skill}' in async context.")
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, skill_function, ctx=skill_ctx, **skill_args_dict)

                console.print("[bold green]Skill Result:[/]")
                console.print_json(data=result)

            asyncio.run(run_the_skill_directly())

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON string provided for --skill-args: {args.skill_args}")
            console.print(f"[bold red]Error:[/bold red] Invalid JSON provided for --skill-args.")
        sys.exit(1)
        except ValueError as ve:
             logger.error(f"Error in skill arguments: {ve}")
             console.print(f"[bold red]Error:[/bold red] {ve}")
             sys.exit(1)
        except TypeError as te:
            logger.exception(f"TypeError during direct skill execution ('{args.run_skill}'): Likely context issue or wrong args.")
            console.print(f"[bold red]Error:[/bold red] TypeError executing skill: {te}")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Error during direct skill execution ('{args.run_skill}'):")
            console.print(f"[bold red]Error:[/bold red] Failed to execute skill directly: {e}")
            sys.exit(1)
        finally:
             # Stop server if we started it for this run
             if server_needed_and_started:
                 logger.info(f"Stopping {server_type} server that was started for --run-skill...")
                 _stop_llama_server() # Always stop the llama_server process
        # <<< END SKILL EXECUTION BLOCK >>>

    elif args.train:
        # <<< ADDED: Handle Train Argument >>>
        if run_qlora_finetuning:
            logger.info("Starting QLoRA Fine-tuning process...")
            try:
                # Ensure necessary args are passed if needed, for now assume defaults are handled
                run_qlora_finetuning()
                logger.info("QLoRA Fine-tuning process completed.")
            except Exception as train_err:
                logger.exception("Error during QLoRA fine-tuning:")
                console.print(f"[bold red][Error][/] Fine-tuning failed: {train_err}")
        else:
            logger.error("Training module not available. Cannot execute --train.")
            console.print("[bold red][Error][/] Training functionality is not available.")
        # <<< END ADDED >>>
    elif args.stream_direct:
        # Handle direct streaming mode
        asyncio.run(stream_direct_llm(args.stream_direct, llm_url))
    elif args.task:
            asyncio.run(_handle_task_argument(agent, args.task))
        elif args.command:
            asyncio.run(_handle_command_argument(agent, args.command))
        elif args.input_file:
            asyncio.run(_handle_file_argument(agent, args.input_file))
    else: # Default to interactive if no other mode specified
            asyncio.run(_handle_interactive_argument(agent))

    # This logging and server stop should be outside the specific mode blocks
        logger.info("CLI run finished.")
    # Explicitly stop server if it was started by this run (only applies to non --run-skill modes)
    if server_process:
        _stop_llama_server()


if __name__ == "__main__":
    # Ensure clean exit on Ctrl+C
    try:
        run_cli()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping server(s) if running...")
        _stop_llama_server()  # Attempt graceful shutdown for llama.cpp
        print("\nExiting cleanly.")
        sys.exit(0)

# <<< START: Skill Validation Helpers >>>

def discover_expected_skills(skills_dir: str) -> Set[str]:
    """Scans the skills directory, parses .py files, and extracts skill names from @skill decorators."""
    expected_skills = set()
    if not os.path.isdir(skills_dir):
        logger.error(f"Skills directory not found for discovery: {skills_dir}")
        return expected_skills

    logger.debug(f"Discovering expected skills in: {skills_dir}")
    for root, _, files in os.walk(skills_dir):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__init__"):
                file_path = os.path.join(root, filename)
                logger.debug(f"Scanning file for skills: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=filename)
                    for node in ast.walk(tree):
                        # Check async and regular function definitions
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for decorator in node.decorator_list:
                                # Check if decorator is a Call (e.g., @skill(...))
                                if isinstance(decorator, ast.Call):
                                    decorator_name = ""
                                    if isinstance(decorator.func, ast.Name):
                                        decorator_name = decorator.func.id
                                    elif isinstance(decorator.func, ast.Attribute):
                                        # Handle cases like @core_tools.skill
                                        decorator_name = decorator.func.attr

                                    # Check if the decorator name is 'skill'
                                    if decorator_name == 'skill':
                                        # Find the 'name' keyword argument
                                        for keyword in decorator.keywords:
                                            if keyword.arg == 'name':
                                                if isinstance(keyword.value, ast.Constant):
                                                    skill_name = keyword.value.value
                                                    if isinstance(skill_name, str):
                                                        logger.debug(f"  Found expected skill: {skill_name}")
                                                        expected_skills.add(skill_name)
                                                break # Found name keyword
                                        break # Found skill decorator
                except FileNotFoundError:
                    logger.warning(f"File listed by os.walk not found: {file_path}")
                    continue # Skip this file
                except SyntaxError as e:
                    logger.error(f"Syntax error parsing {file_path}: {e}")
                    continue # Skip files with syntax errors
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
                    continue # Skip files with other errors

    logger.debug(f"Discovered {len(expected_skills)} expected skills: {expected_skills}")
    return expected_skills

def validate_registered_skills(expected_skills: Set[str], registered_skills_dict: Dict[str, Any]):
    """Compares expected skills (from files) with actually registered skills.

    Args:
        expected_skills (Set[str]): Set of skill names expected from file discovery.
        registered_skills_dict (Dict[str, Any]): The dictionary of registered skills (agent.tools).
    """
    logger.info("Validating registered skills...")
    # expected_skills = discover_expected_skills(skills_dir) # No longer needed here

    try:
        # Use the provided dictionary keys
        registered_skills = set(registered_skills_dict.keys())
        logger.debug(f"Found {len(registered_skills)} registered skills: {registered_skills}")
    except AttributeError as ae:
        logger.error(f"Could not get keys from registered_skills_dict: {ae}. Skipping validation.")
        return
    except Exception as e:
        logger.error(f"Error processing registered_skills_dict: {e}. Skipping validation.", exc_info=True)
        return

    missing_skills = expected_skills - registered_skills

    if missing_skills:
        error_msg = (
            f"🛑 FALHA CRÍTICA NO BOOT: Uma ou mais skills não foram registradas corretamente.\n"
            f"🔍 Causa provável: erro silencioso durante a importação de skills (ex: problema de sistema, dependência ausente, erro no código).\n"
            f"🧩 Skills ausentes detectadas: {sorted(list(missing_skills))}\n"
            f"✅ Sugestão: valide se os arquivos estão corretos e se não há erro de importação no módulo correspondente."
        )
        logger.critical(error_msg)
        console.print(f"[bold red]{error_msg}[/]")
        # Attempt graceful server shutdown if possible before exiting
        _stop_llama_server()
        sys.exit(1)
    else:
        logger.info(f"✅ Skill Registration Validation Passed: All {len(expected_skills)} expected skills are registered.")

# <<< END: Skill Validation Helpers >>>
