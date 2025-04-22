import argparse
import json
import yaml
import logging
from pathlib import Path
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

def parse_arguments():
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Assistente CLI A³X")

    # Execution Mode Group (Mutually Exclusive Options)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-c", "--command", help="Comando único para executar")
    mode_group.add_argument(
        "-i",
        "--input-file",
        help="Arquivo para ler comandos sequencialmente (um por linha)",
    )
    mode_group.add_argument(
        "--interactive", action="store_true", help="Inicia o modo interativo"
    )
    mode_group.add_argument(
        "--task",
        help="Tarefa única a ser executada pelo agente (pode ser usada com outros argumentos não exclusivos)",
    )
    mode_group.add_argument(
        "--stream-direct", help="Prompt direto para o LLM em modo streaming"
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Executa um ciclo de fine-tuning QLoRA usando dados do buffer de experiências."
    )
    mode_group.add_argument(
        "--run-skill",
        help="Run a specific skill directly (provide skill name and arguments like: 'write_file filename=\"test.txt\" content=\"Hello\" --run-skill-args-json')",
    )

    # Arguments for --run-skill
    parser.add_argument(
        "--run-skill-args",
        nargs="*",
        help="Arguments for --run-skill as key=value pairs.",
        default=[],
    )
    parser.add_argument(
        "--run-skill-args-json",
        action="store_true",
        help="If set, treats the arguments for --run-skill as a single JSON string.",
    )
    parser.add_argument(
        "--run-skill-args-file",
        help="Path to a YAML or JSON file containing arguments for --run-skill.",
        default=None,
    )

    # General Options
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15, # Default from ReactAgent
        help="Maximum number of steps the agent can take for a task."
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help="Override the default LLM server URL for this run.",
    )
    parser.add_argument(
        "--no-server-autostart",
        action="store_true",
        help="Disable automatic starting of required backend servers (LLM, LLaVA, etc.)."
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to the system prompt file relative to the project root.",
        default="data/prompts/react_system_prompt.md", # UPDATED Default path
    )
    # Add log level and file args here for clarity, even if setup_logging reads config
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Override the logging level.'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None, # Let setup_logging handle default path
        help='Override the file to write logs to.'
    )


    return parser.parse_args()

def parse_skill_arguments(args) -> dict | None:
    """Parses skill arguments from command line args, file, or JSON string."""
    skill_args = {}
    try:
        if args.run_skill_args_file:
            file_path = Path(args.run_skill_args_file)
            logger.info(f"Attempting to load skill args from file: {file_path}")
            if not file_path.is_file():
                console.print(f"[bold red]Error:[/bold red] Skill args file not found: {file_path}")
                return None
            with file_path.open("r", encoding="utf-8") as f:
                if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
                    try:
                        skill_args = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                         console.print(f"[bold red]Error parsing YAML file:[/bold red] {e}")
                         return None
                    except ImportError:
                         console.print(f"[bold red]Error:[/bold red] PyYAML library is required to parse YAML files. Please install it.")
                         return None
                elif file_path.suffix.lower() == ".json":
                    skill_args = json.load(f)
                else:
                    console.print(f"[bold red]Error:[/bold red] Unsupported file type for skill args: {file_path.suffix}")
                    return None
            if not isinstance(skill_args, dict):
                 console.print(f"[bold red]Error:[/bold red] Content of skill args file must be a dictionary (key-value pairs).")
                 return None
            logger.info(f"Successfully loaded skill args from {file_path}.")

        elif args.run_skill_args_json:
            if args.run_skill_args:
                logger.info("Attempting to load skill args from JSON string.")
                json_string = " ".join(args.run_skill_args)
                skill_args = json.loads(json_string)
                if not isinstance(skill_args, dict):
                    console.print(f"[bold red]Error:[/bold red] JSON string must represent a dictionary.")
                    return None
            else:
                 console.print("[bold yellow]Warning:[/bold yellow] --run-skill-args-json provided but no arguments given.")
                 skill_args = {}

        else:
            logger.info("Attempting to parse skill args from key=value pairs.")
            for arg in args.run_skill_args:
                if "=" not in arg:
                    console.print(
                        f"[bold red]Error:[/bold red] Invalid argument format '{arg}'. Use key=value."
                    )
                    return None
                key, value = arg.split("=", 1)
                # Basic type guessing (can be improved)
                if value.lower() == "true":
                    skill_args[key] = True
                elif value.lower() == "false":
                    skill_args[key] = False
                elif value.isdigit():
                    skill_args[key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    try:
                        skill_args[key] = float(value)
                    except ValueError:
                        skill_args[key] = value # Fallback to string
                else:
                     # Handle quoted strings if necessary
                     if (value.startswith('"') and value.endswith('"')) or \
                        (value.startswith("'") and value.endswith("'")):
                         skill_args[key] = value[1:-1]
                     else:
                         skill_args[key] = value

        logger.debug(f"Parsed skill args: {skill_args}")
        return skill_args

    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error parsing JSON arguments:[/bold red] {e}")
        return None
    except FileNotFoundError: # Should be caught above, but as safeguard
         console.print(f"[bold red]Error:[/bold red] Skill args file not found: {args.run_skill_args_file}")
         return None
    except ImportError as e:
        # Redundant due to check above, but kept for safety
        console.print(f"[bold red]Import Error:[/bold red] A required library might be missing for file type {Path(args.run_skill_args_file).suffix}. Details: {e}")
        return None
    except Exception as e:
        logger.exception("Error parsing skill arguments:")
        console.print(f"[bold red]An unexpected error occurred parsing skill arguments:[/bold red] {e}")
        return None 