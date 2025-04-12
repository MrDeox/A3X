# /home/arthur/Projects/A3X/cli/interface.py
import os
import sys
import asyncio
import argparse
import time
import logging
import subprocess
import atexit
import signal
import platform
from dotenv import load_dotenv
from typing import Optional, Tuple, AsyncGenerator, Set, Dict, Any, Callable
import json
import importlib
import ast
import requests
import inspect
from pathlib import Path
from collections import namedtuple
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# <<< ADDED: Log Python environment at script start >>>
print(f"DEBUG: Running with Python Executable: {sys.executable}")
print(f"DEBUG: Initial sys.path: {sys.path}")
# --- End Added Log ---

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
        PROJECT_ROOT,
        # DEFAULT_MMPROJ_PATH, # Removed unused import
        # ENABLE_LLAMA_SERVER_AUTOSTART, # Removed unused import
    )

    # from core.db_utils import initialize_database
    from a3x.core.db_utils import initialize_database

    # from core.cerebrumx import CerebrumXAgent
    from a3x.core.cerebrumx import CerebrumXAgent

    # <<< Replace commented out/incorrect tool import >>>
    from a3x.core.skills import (
        load_skills,
        get_skill,
        get_skill_descriptions,
        SKILL_REGISTRY, # Import the registry itself
        # SkillContext, # Removed import, using local definition
    )
    # from core.logging_config import setup_logging
    from a3x.core.logging_config import setup_logging
    from a3x.core.tool_executor import execute_tool
    from a3x.core.llm_interface import call_llm
    # <<< ADDED: Import server manager functions >>>
    from a3x.core.server_manager import start_llama_server, start_sd_server, stop_all_servers, managed_processes
    from a3x.core.agent import DEFAULT_REACT_SYSTEM_PROMPT # <<< Import the new prompt
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
LLAVA_MODEL_DIR_RELATIVE = "LLaVA/llava-v1.5-7b" # Relative to project root
LLAVA_REPO_DIR_RELATIVE = "LLaVA" # Directory name for cloned LLaVA repo
_llava_controller_process: Optional[subprocess.Popen] = None
_llava_worker_process: Optional[subprocess.Popen] = None
_llava_api_server_process: Optional[subprocess.Popen] = None
# <<< END NEW LLaVA Stack Config >>>

# <<< REMOVED: Old, simpler SkillContext definition >>>
# from collections import namedtuple
# SkillContext = namedtuple("SkillContext", ["logger", "llm_call", "is_test"])

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
    # <<< ADDED: Argument for skill args file >>>
    parser.add_argument(
        "--skill-args-file",
        help="(TESTING) Path para um arquivo JSON contendo argumentos para a skill (usado com --run-skill, tem precedência sobre --skill-args).",
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
    """Handles the --task argument (either a JSON file or a direct objective string)."""
    if task_arg.lower().endswith('.json') or task_arg.lower().endswith('.jsonl'):
        logger.info(f"Loading structured task(s) from JSON: {task_arg}")
        try:
            with open(task_arg, 'r', encoding='utf-8') as f:
                # <<< START MODIFICATION: Handle objective or list >>>
                tasks_data = json.load(f)

            if isinstance(tasks_data, dict) and 'objective' in tasks_data:
                # Handle JSON containing a single objective for the agent
                objective = tasks_data['objective']
                if not isinstance(objective, str) or not objective:
                    logger.error(f"Invalid 'objective' format in task file {task_arg}. Must be a non-empty string.")
                    console.print(f"[bold red]Erro:[/bold red] Formato inválido para 'objective' no arquivo JSON.")
                    return

                logger.info(f"Processing single objective from JSON: '{objective[:100]}...'")
                # Pass the objective to the agent's main interaction handler
                await handle_agent_interaction(agent, objective, []) # Pass empty history

            elif isinstance(tasks_data, list):
                # Handle the original list-of-skills format
                logger.info(f"Processing list of {len(tasks_data)} predefined skill calls from JSON.")
                for task_index, task in enumerate(tasks_data):
                    if not isinstance(task, dict) or not all(k in task for k in ['skill_name', 'function_name', 'params']):
                        logger.error(f"Invalid task format at index {task_index} in {task_arg}. Required keys: skill_name, function_name, params.")
                        console.print(f"[bold red]Erro:[/bold red] Formato inválido para tarefa no índice {task_index}.")
                        continue # Skip this invalid task

                    skill_name = task.get('skill_name')
                    function_name = task.get('function_name')
                    params = task.get('params', {})

                    logger.info(f"--- Executing Task {task_index + 1}/{len(tasks_data)}: Skill '{skill_name}', Function '{function_name}' ---")

                    # WARNING: This direct execution path bypasses agent planning/context.
                    # It might be prone to errors if skills rely on agent state or complex context.
                    # Consider removing or refactoring this path if agent execution is preferred.
                    module_path = f"skills.{skill_name}"
                    result = None
                    try:
                        # Ensure the skill module is loaded
                        if module_path not in sys.modules:
                            importlib.import_module(module_path)
                        skill_module = sys.modules[module_path]

                        # Check if function exists
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
                                loop = asyncio.get_event_loop()
                                result = await loop.run_in_executor(
                                    None, skill_function, **params
                                )

                            if not isinstance(result, dict):
                                logger.warning(
                                    f"Skill '{skill_name}' did not return a dict. Wrapping output."
                                )
                                result = {"status": "success", "message": str(result)}

                        else:
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
            else:
                # Invalid top-level JSON structure
                logger.error(f"Invalid JSON structure in task file {task_arg}. Expected a list of skill calls or a dictionary with an 'objective' key.")
                console.print(f"[bold red]Erro:[/bold red] Estrutura JSON inválida no arquivo de tarefa.")
                return
            # <<< END MODIFICATION >>>

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


# <<< REPLACED Function >>>
async def _handle_run_skill_argument(args, llm_url: str):
    """Handles the --run-skill argument, detecting and executing functions or class methods."""
    logger.debug(f"Starting _handle_run_skill_argument for skill: {args.run_skill}")
    skill_name = args.run_skill
    skill_args_dict = {}
    error_occurred = False

    # <<< MOVED: Define SkillContext locally >>>
    # Define SkillContext named tuple robustly within this function's scope
    _ConcreteSkillContext = namedtuple("SkillContext", ["logger", "llm_call", "is_test", "workspace_root", "task"])
    # <<< END MOVED >>>

    # 1. Load Skill Arguments (same as before)
    if args.skill_args_file:
        logger.info(f"Attempting to load skill args from file: {args.skill_args_file}")
        try:
            with open(args.skill_args_file, 'r') as f:
                skill_args_dict = json.load(f)
            if not isinstance(skill_args_dict, dict):
                raise ValueError("JSON file content must be a dictionary.")
            logger.info(f"Successfully loaded skill args from {args.skill_args_file}.")
            logger.debug(f"Loaded args: {skill_args_dict}")
        except FileNotFoundError:
            logger.error(f"Skill arguments file not found: {args.skill_args_file}")
            console.print(f"[bold red]Error:[/bold red] Skill arguments file not found: {args.skill_args_file}")
            error_occurred = True
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid JSON or format in --skill-args-file ({args.skill_args_file}): {e}")
            console.print(f"[bold red]Error:[/bold red] Invalid JSON/format in --skill-args-file ({args.skill_args_file}): {e}")
            error_occurred = True
        except Exception as e:
            logger.error(f"Unexpected error reading skill args file {args.skill_args_file}: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] Unexpected error reading skill args file: {e}")
            error_occurred = True
    elif args.skill_args:
        logger.info("Attempting to load skill args from --skill-args string.")
        try:
            skill_args_dict = json.loads(args.skill_args)
            logger.debug(f"Parsed skill args: {skill_args_dict}")
            if not isinstance(skill_args_dict, dict):
                raise ValueError("Skill arguments must be a JSON object (dictionary).")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid JSON or format in --skill-args: {e}")
            console.print(f"[bold red]Error:[/bold red] Invalid JSON/format for --skill-args: {e}")
            error_occurred = True
        except Exception as e:
            logger.error(f"Unexpected error parsing --skill-args: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] Unexpected error parsing --skill-args: {e}")
            error_occurred = True
    else:
        logger.info("No skill arguments provided via file or string, using empty dict.")
        skill_args_dict = {}

    if error_occurred:
        return # Exit if loading args failed

    # 2. Get Skill Function/Method from Registry
    logger.debug("Ensuring SkillRegistry is populated...")
    # Assuming load_skills was called earlier in run_cli or upon import

    logger.debug(f"Attempting to retrieve skill '{skill_name}' from registry.")
    skill_info = SKILL_REGISTRY.get(skill_name)

    if not skill_info:
        logger.error(f"Skill '{skill_name}' not found in registry.")
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        registered_skills = list(SKILL_REGISTRY.keys())
        console.print(f"Available skills: {', '.join(registered_skills) if registered_skills else 'None'}")
        return

    func_obj = skill_info.get("function")
    if not func_obj or not callable(func_obj):
        logger.error(f"Skill '{skill_name}' found but has no callable function object.")
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' is improperly registered.")
        return

    logger.debug(f"Retrieved skill function object '{func_obj.__qualname__}' for skill '{skill_name}'.")

    # 3. Prepare Context and LLM Wrapper
    # Define a non-streaming LLM call wrapper
    async def non_streaming_llm_call_wrapper(prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        logger.debug(f"Executing non-streaming LLM call for skill '{skill_name}'")
        messages = [{"role": "user", "content": prompt}]
        collected_response = ""
        try:
            async for chunk in call_llm(messages, stream=False): # Assuming call_llm is available
                collected_response += chunk
            logger.debug(f"Non-streaming LLM call for skill '{skill_name}' completed. Length: {len(collected_response)}")
            return collected_response
        except Exception as e:
            logger.error(f"Error in non_streaming_llm_call_wrapper: {e}", exc_info=True)
            return f"[LLM Call Error: {e}]"

    # Create the skill context using the locally defined _ConcreteSkillContext
    skill_context = _ConcreteSkillContext(
        logger=logger,
        llm_call=non_streaming_llm_call_wrapper,
        is_test=True,
        workspace_root=Path(PROJECT_ROOT).resolve(), # Pass the resolved project root
        task=args.task
    )
    logger.debug(f"Created SkillContext for '{skill_name}' with workspace_root: {Path(PROJECT_ROOT).resolve()}")


    # 4. Detect Type and Execute
    result = None
    executable_callable: Callable = None
    instance: Optional[Any] = None

    try:
        # Check if it's a method within a class (using __qualname__)
        is_method = inspect.isfunction(func_obj) and '.' in func_obj.__qualname__

        if is_method:
            logger.info(f"Skill '{skill_name}' detected as a method: {func_obj.__qualname__}")
            qname = func_obj.__qualname__
            class_name = qname.split('.')[-2]
            method_name = func_obj.__name__ # Use actual method name

            # Get the module
            module = inspect.getmodule(func_obj)
            if not module:
                raise RuntimeError(f"Could not determine the module for {qname}")

            logger.debug(f"Attempting to get class '{class_name}' from module '{module.__name__}'")
            SkillClass = getattr(module, class_name)

            logger.info(f"Instantiating class '{class_name}' with workspace_root='{Path(PROJECT_ROOT).resolve()}'")
            # Instantiate the class, passing workspace_root if the constructor accepts it
            # We assume constructors accept 'workspace_root' or handle its absence gracefully.
            try:
                # Check if constructor accepts workspace_root
                sig = inspect.signature(SkillClass.__init__)
                if 'workspace_root' in sig.parameters:
                     instance = SkillClass(workspace_root=Path(PROJECT_ROOT).resolve())
                else:
                     instance = SkillClass() # Instantiate without workspace_root
            except Exception as e:
                 logger.error(f"Failed to instantiate {class_name}: {e}", exc_info=True)
                 raise RuntimeError(f"Failed to instantiate class {class_name}: {e}")

            logger.debug(f"Successfully instantiated {class_name}.")

            # Get the bound method from the instance
            executable_callable = getattr(instance, method_name)
            logger.debug(f"Retrieved bound method '{executable_callable.__qualname__}' for execution.")

        else:
            logger.info(f"Skill '{skill_name}' detected as a regular function: {func_obj.__qualname__}")
            executable_callable = func_obj # Use the original function object

        # Execute the callable (function or bound method)
        logger.info(f"Executing {'method' if is_method else 'function'} '{executable_callable.__qualname__}' with args: {skill_args_dict}")

        # Check if the final callable is async
        is_async = inspect.iscoroutinefunction(executable_callable)

        if is_async:
            logger.debug(f"Executing async callable '{executable_callable.__qualname__}'")
            # Pass skill_context as the first arg if it's a standalone function
            # For bound methods, 'self' is already part of executable_callable
            if not is_method:
                 result = await executable_callable(skill_context, **skill_args_dict)
            else:
                 # For methods, we might still need context if designed that way,
                 # but typical @skill might inject it or method takes **kwargs
                 # Let's assume the method signature accepts context + kwargs OR just kwargs
                 # Check signature to be more robust (Optional)
                 try:
                     result = await executable_callable(skill_context, **skill_args_dict)
                 except TypeError as e:
                     logger.warning(f"TypeError calling method {executable_callable.__qualname__} with context. Trying without context. Error: {e}")
                     # Try calling without context if the method doesn't expect it explicitly
                     result = await executable_callable(**skill_args_dict)

        else:
            logger.debug(f"Executing sync callable '{executable_callable.__qualname__}' in executor.")
            loop = asyncio.get_event_loop()
            if not is_method:
                result = await loop.run_in_executor(None, lambda: executable_callable(skill_context, **skill_args_dict))
            else:
                # Similar logic for sync methods - try with context, then without
                try:
                    result = await loop.run_in_executor(None, lambda: executable_callable(skill_context, **skill_args_dict))
                except TypeError as e:
                    logger.warning(f"TypeError calling method {executable_callable.__qualname__} with context. Trying without context. Error: {e}")
                    result = await loop.run_in_executor(None, lambda: executable_callable(**skill_args_dict))


        logger.info(f"Skill '{skill_name}' execution finished.")
        console.print("[bold green]Skill Result:[/]")
        try:
            console.print(json.dumps(result, indent=2))
        except (TypeError, OverflowError):
            console.print(result)

    except TypeError as te:
        # More specific error for argument mismatches
        logger.exception(f"TypeError executing skill '{skill_name}': Likely incorrect arguments provided. {te}")
        console.print(f"[bold red]Error:[/bold red] TypeError executing skill '{skill_name}'. Check arguments provided ({skill_args_dict}). Error: {te}")
        # Optionally print expected signature
        if executable_callable:
            try:
                sig = inspect.signature(executable_callable)
                console.print(f"Expected signature: {skill_name}{sig}")
            except ValueError: # Can happen for built-ins etc.
                 pass

    except Exception as e:
        logger.exception(f"Error during execution preparation or running of skill '{skill_name}':")
        console.print(f"[bold red]Error executing skill '{skill_name}':[/] {e}")
    finally:
        # Stop server if we started it (same as before)
        # Note: The logic for starting/stopping servers might need revisiting
        # if run_skill is meant to be truly standalone without server interaction.
        if _llama_server_process and _llama_server_process.poll() is None:
            logger.info(f"Stopping internal llama-server (PID: {_llama_server_process.pid})...")
            #_stop_llama_server() # Assuming this is synchronous and safe here
            pass # Let main_async handle server shutdown

# <<< END REPLACED Function >>>


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

# <<< START: Skill Validation Helpers (MOVED UP) >>>

# <<< ADDED: Custom Exception for Critical Skill Registration Errors >>>
class CriticalSkillRegistrationError(Exception):
    """Custom exception raised when essential skills fail to register."""
    def __init__(self, message, missing_skills=None):
        super().__init__(message)
        self.missing_skills = missing_skills if missing_skills is not None else set()
# <<< END ADDED >>>

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
    """Compares expected skills (discovered from files) with actually registered skills."""
    registered_skill_names = set(registered_skills_dict.keys())

    missing_skills = expected_skills - registered_skill_names
    extra_skills = registered_skill_names - expected_skills

    if missing_skills:
        # Critical Error: Skills defined in code but not registered via import
        logger.critical("🧩 Skills missing registration detected!")
        for skill_name in missing_skills:
            logger.critical(f"  - Skill '{skill_name}' defined but NOT imported/registered.")
            logger.critical(f"    Check the __init__.py file in the skill's directory (e.g., a3x/skills/category/__init__.py)")
            logger.critical(f"    Ensure 'from . import {skill_name}' (or the module name) exists.")
        # Raise critical exception to prevent startup
        raise CriticalSkillRegistrationError(f"Skills missing registration: {missing_skills}")
    else:
        logger.debug("✅ All discovered skills appear to be registered.")

    if extra_skills:
        # Warning: Skills registered but not found by discovery (e.g., old file deleted but import remains)
        logger.warning("🧩 Potentially stale skill registrations detected!")
        for skill_name in extra_skills:
            logger.warning(f"  - Skill '{skill_name}' registered but NOT found by file discovery.")
            logger.warning(f"    This might indicate an old skill file was deleted but the import in __init__.py remains.")
            logger.warning(f"    Consider removing the import for '{skill_name}' if it's no longer used.")
        # This is just a warning, allow startup
    else:
        logger.debug("✅ No unexpected/stale skill registrations found.")

# <<< END: Skill Validation Helpers >>>


def run_cli():
    """Função principal da CLI."""
    # setup_logging() # Chamado dentro de main_async agora

    # --- Funções de Limpeza e Sinais ---
    def cleanup_servers():
        """Função de limpeza registrada com atexit para garantir parada dos servidores.
           Deve conter apenas lógica síncrona ou chamadas que funcionem fora de um loop asyncio ativo.
        """
        logger.info("Executing synchronous cleanup via atexit...")
        # A antiga função _stop_llama_server era síncrona e pode ser chamada aqui,
        # mas a lógica moderna está em stop_all_servers (assíncrona).
        # _stop_llama_server() # Manter ou remover dependendo se ainda é útil

        # <<< REMOVED: Chamada direta a stop_all_servers (assíncrona) >>>
        # logger.info("Ensuring all managed servers are stopped via atexit...")
        # try:
        #     # Tentar executar a limpeza assíncrona aqui pode ser problemático
        #     # devido ao estado do loop de eventos no atexit.
        #     # asyncio.run(stop_all_servers())
        #     pass # Melhor chamar stop_all_servers explicitamente no final de main_async
        # except RuntimeError as e:
        #      # Comum se o loop já estiver fechado
        #      logger.warning(f"RuntimeError during atexit cleanup (likely loop closed): {e}")
        # except Exception as e:
        #      logger.exception("Unexpected error during atexit server cleanup:")
        logger.info("Synchronous atexit cleanup finished.")

    atexit.register(cleanup_servers)

    # ... (signal handler) ...

    # --- Função Principal Assíncrona ---
    async def main_async():
        """Função principal que executa a lógica assíncrona."""
        global _llama_server_process # Permitir modificação
        args = _parse_arguments()

        # Setup logging ASAP
        setup_logging()
        logger.info("Starting A³X CLI...")
        logger.info(f"Arguments: {args}")

        # Mudar para o diretório raiz do projeto
        change_to_project_root()
        logger.info(f"Running in directory: {os.getcwd()}")

        # Inicializar DB
        initialize_database() # Ensure DB is ready

        # Determinar URL do LLM
        llm_url = args.model or DEFAULT_SERVER_URL # Usa --model se fornecido, senão config

        servers_started_by_cli = False
        try:
            # Iniciar servidor LLaMA se necessário
            if not args.no_server:
                logger.info("Attempting to start and manage LLM and SD servers...")
                # Start LLaMA
                llama_success = await start_llama_server() # Usa a função do server_manager
                if llama_success:
                    logger.info("LLaMA server managed successfully.")
                    servers_started_by_cli = True
                else:
                    logger.warning("LLM server failed to start correctly or was already running.")

                # Start SD (Add error handling if needed)
                # sd_success = await start_sd_server()
                # if sd_success:
                #    logger.info("SD server managed successfully.")
                # else:
                #    logger.warning("SD server failed to start correctly or was already running.")
            else:
                logger.info("Skipping internal server start (--no-server specified).")
                # Verify connection to existing server?
                # if not _check_llm_server_health(llm_url):
                #    logger.warning(f"Warning: Could not connect to LLM server at {llm_url}. Agent may fail.")

            logger.info("Servers ready (or not managed). Proceeding with CLI operation.")

            # Carregar skills DEPOIS que os servidores (se gerenciados) estão prontos
            logger.info("Loading skills...")
            load_skills() # Use the core function
            tools_dict = SKILL_REGISTRY # <<< Corrected: Pass the actual registry
            if not tools_dict:
                 logger.warning("No skills were loaded. The agent will have no tools.")
                 # raise CriticalSkillRegistrationError("CRITICAL: No skills registered after loading.")
            else:
                logger.info(f"Loaded {len(tools_dict)} skills.")

            # Inicializar Agente (passando tools)
            logger.info("Initializing Agent...")
            agent = _initialize_agent(system_prompt=DEFAULT_REACT_SYSTEM_PROMPT, llm_url_override=llm_url, tools_dict=tools_dict) # Pass tools here
            if not agent:
                logger.critical("Failed to initialize agent. Exiting.")
                return # Sai da função main_async
            logger.info("Agent ready.")

            # Validar skills registradas vs arquivos encontrados
            logger.info("Validating skills registration...")
            try:
                # Convert list to set for comparison
                expected_skill_names = set([
                    "generate_code", "write_file", "read_file", "list_directory",
                    "append_to_file", "delete_path", "hierarchical_planner",
                    "simulate_step", "final_answer", "web_search",
                    # Add other essential skills here for validation
                ])
                # Validate available skills against expected
                registered_skills = set(SKILL_REGISTRY.keys())
                expected_skills_set = set(expected_skill_names) # <<< CONVERT TO SET >>>

                missing_skills = expected_skills_set - registered_skills
                extra_skills = registered_skills - expected_skills_set

                if missing_skills:
                    logger.warning(
                        f"Missing expected skills in registry: {sorted(list(missing_skills))}"
                    )
                if extra_skills:
                    logger.warning(
                        f"Extra skills in registry: {sorted(list(extra_skills))}"
                    )
            except Exception as skill_val_err:
                logger.exception(f"Error during skill validation placeholder:")

            # Executar lógica baseada nos argumentos
            if args.task:
                await _handle_task_argument(agent, args.task)
            elif args.command:
                await _handle_command_argument(agent, args.command)
            elif args.input_file:
                await _handle_file_argument(agent, args.input_file)
            elif args.interactive:
                await _handle_interactive_argument(agent)
            elif args.stream_direct:
                 await stream_direct_llm(args.stream_direct, _direct_llm_call_wrapper)
            elif args.train:
                if run_qlora_finetuning:
                    logger.info("Starting QLoRA fine-tuning process...")
                    # Assumindo que a função pode rodar sincronamente ou você a adapta
                    run_qlora_finetuning()
                    logger.info("Fine-tuning process completed.")
                else:
                    logger.error("--train specified, but training module failed to import.")
            # <<< ADDED: Handle direct skill execution >>>
            elif args.run_skill:
                await _handle_run_skill_argument(args, llm_url)
            # <<< END ADDED >>>
            else:
                # Comportamento padrão: modo interativo se nenhum outro modo for especificado
                logger.info("No specific mode selected, entering interactive mode.")
                await _handle_interactive_argument(agent)

        except Exception as e:
            logger.critical(f"An unexpected error occurred in main_async: {e}", exc_info=True)
        finally:
            # <<< ADDED: Explicitly stop servers managed by this CLI run >>>
            logger.info("Main CLI function finished or errored, ensuring server cleanup...")
            await stop_all_servers()
            logger.info("Server cleanup process completed via stop_all_servers().")

    # --- Ponto de Entrada Principal ---
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("CLI interrupted by user (KeyboardInterrupt).")
        # A limpeza de atexit deve ser chamada aqui automaticamente
        # Mas podemos tentar uma limpeza explícita adicional se necessário,
        # embora possa causar problemas de loop se main_async não terminou.
    except Exception as e:
        logger.critical(f"A critical error occurred outside the main async loop: {e}", exc_info=True)
    finally:
        logger.info("A³X CLI finished.")


if __name__ == "__main__":
    run_cli()
