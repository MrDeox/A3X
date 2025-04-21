# a3x/cli/main.py
import asyncio
import logging
import sys
import os
import atexit
import signal
from typing import Optional, Dict, Any, Set

from rich.console import Console
import pydantic # Add import for error handling

# Use relative imports for modules within the same package (cli)
from .parsing import parse_arguments
from .commands import (
    run_interactive_mode,
    run_from_file,
    run_task,
    run_single_command,
    run_stream_direct,
    run_training_cycle,
    run_skill_directly
)
# Import helpers from new utility modules
from .agent_utils import initialize_agent
from .fs_utils import load_system_prompt, change_to_project_root
from .server_utils import check_llm_server_health, check_sd_server_health, check_llava_api_health
from .skill_utils import discover_expected_skills, validate_registered_skills, CriticalSkillRegistrationError
from a3x.core.tool_registry import ToolRegistry

# Import core components directly
# Assuming execution context allows finding 'a3x' package
try:
    from a3x.core.config import PROJECT_ROOT, LLAMA_SERVER_URL, LLAVA_API_URL
    from a3x.core.logging_config import setup_logging
    from a3x.core.server_manager import start_all_servers, stop_all_servers
    from a3x.core.cerebrumx import CerebrumXAgent
    from a3x.core.skills import load_all_skills, SKILL_REGISTRY
    from a3x.fragments.registry import FragmentRegistry
    from a3x.core.db_utils import initialize_database, close_db_connection
except ImportError as e:
    print(f"[CLI Main Error] Failed to import core modules: {e}")
    # Define fallbacks or exit
    PROJECT_ROOT = "."
    LLAMA_SERVER_URL = ""
    LLAVA_API_URL = ""
    setup_logging = lambda log_level_str=None, log_file_path=None: None
    start_all_servers = lambda _: None
    stop_all_servers = lambda: None
    load_all_skills = lambda _: ({}, None)
    CerebrumXAgent = None # Define fallback type
    SKILL_REGISTRY = {}
    FragmentRegistry = None
    initialize_database = lambda: None
    close_db_connection = lambda: None
    # Fallback imports for CLI utils (will cause errors if core failed)
    initialize_agent = lambda *args, **kwargs: None
    load_system_prompt = lambda *args, **kwargs: ""
    change_to_project_root = lambda: None
    check_llm_server_health = lambda *args, **kwargs: False
    check_sd_server_health = lambda *args, **kwargs: False
    check_llava_api_health = lambda *args, **kwargs: False
    discover_expected_skills = lambda *args, **kwargs: set()
    validate_registered_skills = lambda *args, **kwargs: None
    class CriticalSkillRegistrationError(Exception): pass
    sys.exit(1)


logger = logging.getLogger(__name__)
console = Console()

# --- Main Async Execution --- #

async def main_async():
    """Asynchronous main logic for the CLI."""
    args = parse_arguments()

    # --- Logging Setup (using config from core, with overrides) ---
    setup_logging(log_level_str=args.log_level, log_file_path=args.log_file)
    logger = logging.getLogger(__name__) # Get logger for this module AFTER setup
    logger.info("A³X CLI starting...")
    logger.debug(f"Parsed arguments: {args}")

    # --- Ensure Running from Project Root --- #
    change_to_project_root()
    logger.info(f"Changed working directory to project root: {os.getcwd()}")

    # --- Initialize Database EARLY --- #
    try:
        logger.info("Initializing database structure...")
        initialize_database() # Uses default path from config
        logger.info("Database initialized successfully.")
    except Exception as db_init_err:
        logger.exception("Critical error initializing the database. Exiting.")
        console.print(f"[bold red]Database Error:[/bold red] {db_init_err}")
        sys.exit(1)

    # --- Server Auto-Start --- #
    autostart_enabled = not args.no_server_autostart
    servers_started_by_cli = False
    if autostart_enabled:
        logger.info("Attempting to auto-start required servers...")
        try:
            # Pass health check functions from server_utils
            servers_started_by_cli = await start_all_servers()
            if servers_started_by_cli:
                logger.info("ServerManager returned, indicating one or more servers might have been started.")
                # Register cleanup ONLY if the manager instance was created
                atexit.register(stop_all_servers)
                signal.signal(signal.SIGTERM, lambda sig, frame: stop_all_servers())
                signal.signal(signal.SIGINT, lambda sig, frame: stop_all_servers())
            else:
                logger.info("All required servers seem to be running already.")
        except Exception as e:
            logger.exception("Error during server auto-start process:")
            console.print(f"[bold yellow]Warning:[/bold yellow] Failed to start/check servers: {e}")
    else:
        logger.info("Server auto-start disabled by --no-server-autostart argument.")

    # --- Load System Prompt --- #
    # Use function from fs_utils
    system_prompt = load_system_prompt(args.system_prompt_file)

    # --- Load Skills and Register Tools --- #
    logger.info("Initializing ToolRegistry...")
    tool_registry = ToolRegistry()

    logger.info("Loading skills...")
    packages = ["a3x.skills.core", "a3x.skills.web"]
    load_all_skills(skill_package_list=packages)

    logger.info("Registering loaded skills into ToolRegistry...")
    registered_count = 0
    if SKILL_REGISTRY:
        for skill_name, skill_info in SKILL_REGISTRY.items():
            if not isinstance(skill_info, dict):
                logger.warning(f"Invalid skill info format for '{skill_name}'. Skipping.")
                continue

            tool_callable = skill_info.get("function")
            pydantic_schema_model = skill_info.get("schema")

            if not tool_callable or not pydantic_schema_model:
                logger.warning(f"Skipping incomplete skill info for '{skill_name}': Missing function or schema model.")
                continue

            try:
                # Generate JSON schema from Pydantic model
                base_json_schema = pydantic_schema_model.model_json_schema(by_alias=False) 
                
                # Ensure base schema is a dict, default to empty if not
                if not isinstance(base_json_schema, dict):
                    logger.warning(f"Generated base schema for '{skill_name}' is not a dict: {type(base_json_schema)}. Using empty schema.")
                    base_json_schema = {}

                # Construct the final schema, guaranteeing name and description
                final_tool_schema = {
                    "name": skill_name, 
                    "description": skill_info.get("description", f"Skill: {skill_name}"), # Use skill_name as fallback desc
                    "parameters": { 
                        "type": base_json_schema.get("type", "object"), 
                        "properties": base_json_schema.get("properties", {}),
                        "required": base_json_schema.get("required", [])
                    }
                }
                # Add other top-level keys from base_json_schema if they exist and aren't the ones we set
                for key, value in base_json_schema.items():
                    if key not in ["name", "description", "parameters", "properties", "required", "type"]:
                         final_tool_schema[key] = value

                tool_instance = None # Still assuming None for instance

                # Register with the ToolRegistry instance
                tool_registry.register_tool(
                    name=skill_name,
                    instance=tool_instance, 
                    tool=tool_callable,
                    schema=final_tool_schema # Pass guaranteed structure
                )
                registered_count += 1
            except pydantic.errors.PydanticInvalidForJsonSchema as schema_err:
                 # Catch the specific Pydantic schema generation error
                 logger.error(f"Failed to generate JSON schema for skill '{skill_name}': {schema_err}", exc_info=False) # Log without full traceback
            except Exception as reg_err:
                logger.error(f"Failed to register skill '{skill_name}' in ToolRegistry: {reg_err}", exc_info=True) # Log other errors with traceback
        logger.info(f"Registered {registered_count} skills into ToolRegistry.")
        logger.debug(f"ToolRegistry contents: {list(tool_registry.list_tools().keys())}")
    else:
        logger.warning("SKILL_REGISTRY is empty after loading. ToolRegistry will be empty.")

    # --- Validate Skills --- #
    try:
        skills_dir = os.path.join(PROJECT_ROOT, "a3x", "skills")
        expected_skills = discover_expected_skills(skills_dir)
        validate_registered_skills(expected_skills, SKILL_REGISTRY)
        logger.info("Core skill validation passed.")
    except CriticalSkillRegistrationError as skill_val_err:
         logger.error(f"Critical skill validation failed: {skill_val_err.args[0]}. Missing: {skill_val_err.missing_skills}", exc_info=False)
         console.print(f"[bold red]Skill Validation Error:[/bold red] {skill_val_err.args[0]}")
         console.print(f"Missing essential skills: {skill_val_err.missing_skills}")
         sys.exit(1)
    except Exception as skill_val_err:
         logger.error(f"Skill validation failed: {skill_val_err}", exc_info=True)
         console.print(f"[bold red]Skill Validation Error:[/bold red] {skill_val_err}")

    # --- Initialize Agent (only if needed by command) --- #
    agent: Optional[CerebrumXAgent] = None
    if args.interactive or args.command or args.input_file or args.task:
        logger.info("Initializing agent...")
        # Use function from agent_utils
        agent = initialize_agent(
            system_prompt,
            tool_registry=tool_registry,
            llm_url_override=args.llm_url,
            max_steps=args.max_steps
        )
        if not agent:
            logger.critical("Failed to initialize the agent. Exiting.")
            console.print("[bold red]Error:[/bold red] Could not initialize the agent.")
            sys.exit(1)
        logger.info("Agent initialized successfully.")

    # --- Execute Command --- #
    try:
        if args.interactive:
            await run_interactive_mode(agent)
        elif args.command:
            await run_single_command(agent, args.command)
        elif args.input_file:
            await run_from_file(agent, args.input_file)
        elif args.task:
            await run_task(agent, args.task)
        elif args.stream_direct:
            # Need to import direct_llm_call_wrapper from llm_utils if run_stream_direct doesn't handle it
            # Assuming run_stream_direct handles the wrapper call itself for now
            await run_stream_direct(args.stream_direct, args.llm_url)
        elif args.train:
            await run_training_cycle()
        elif args.run_skill:
            await run_skill_directly(args, args.llm_url)
        else:
            console.print(
                "No execution mode specified. Use --interactive, --task, --command, --input-file, etc. Use -h for help."
            )

    except Exception as main_err:
        logger.exception("An unexpected error occurred during CLI execution:")
        console.print(f"[bold red]Unexpected Error:[/bold red] {main_err}")
    finally:
        logger.info("CLI execution finished. Cleaning up...")
        close_db_connection()
        logger.info("Cleanup complete. Exiting.")

# --- Synchronous Entry Point --- #

def run_cli():
    """Synchronous entry point that runs the async main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]CLI interrupted by user. Exiting.[/bold yellow]")
        close_db_connection()
        sys.exit(0)
    except Exception as e:
        try:
            logger = logging.getLogger(__name__)
            logger.critical(f"Critical error during CLI setup or teardown: {e}", exc_info=True)
        except Exception:
            pass
        print(f"Critical Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_cli() 