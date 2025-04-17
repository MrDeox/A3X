#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import json
import asyncio
import time
from urllib.parse import urlparse, urljoin
from pathlib import Path
from rich.console import Console  # Import Console from rich
from typing import Callable, Dict, Any, get_type_hints

# <<< START VENV WORKAROUND >>>
# Manually add the venv site-packages directory to sys.path
# This is needed because the venv creation linked to Cursor, not system Python.
_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_site_packages = os.path.join(
    _project_root, "venv", "lib", "python3.13", "site-packages"
)
if _venv_site_packages not in sys.path:
    sys.path.insert(0, _venv_site_packages)
# Cleanup temporary variables
del _project_root
del _venv_site_packages
# <<< END VENV WORKAROUND >>>

# <<< ADDED: Log Python environment at script start >>>
print(f"DEBUG: Running with Python Executable: {sys.executable}")
print(f"DEBUG: Initial sys.path: {sys.path}")
# --- End Added Log ---

# Adiciona o diretório raiz ao sys.path para encontrar 'core' e 'skills'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# Garante que a raiz do projeto A3X esteja no path
# Isso assume que assistant_cli.py está em a3x/assistant_cli.py
root_a3x = os.path.dirname(project_root) # /home/arthur/projects
if root_a3x not in sys.path:
    sys.path.insert(0, root_a3x) 

# Imports do Core (após adicionar ao path)
try:
    from a3x.core.config import (
        LLAMA_SERVER_MODEL_PATH,
        PROJECT_ROOT,
        DATABASE_PATH,
        SEMANTIC_INDEX_PATH,
        SEMANTIC_SEARCH_TOP_K,
        EPISODIC_RETRIEVAL_LIMIT
    )
    from a3x.core.db_utils import initialize_database, _db_connections, close_db_connection
    from a3x.core.agent import ReactAgent # Usando ReactAgent diretamente por enquanto
    # Import LLMInterface and its DEFAULT_LLM_URL
    from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
    # Import ServerManager ONLY
    from a3x.core.server_manager import ServerManager
    # Update the import path for MemoryManager
    from a3x.core.memory.memory_manager import MemoryManager
    from a3x.core.logging_config import setup_logging
    from a3x.core.context import Context # Import Context
    # SKILL_REGISTRY e get_skill_descriptions são carregados dinamicamente agora
    # from a3x.core.skills import SKILL_REGISTRY, get_skill_descriptions
    from a3x.core.tool_registry import ToolRegistry
    from a3x.fragments.registry import FragmentRegistry
    from a3x.core.skills import get_skill_registry
    # >>> ADD skill imports for registration <<<
    from a3x.skills.file_manager import FileManagerSkill
    from a3x.skills.planning import hierarchical_planner
    from a3x.skills.final_answer import final_answer
    from a3x.skills.core.learning_cycle import learning_cycle_skill as learning_cycle
    import inspect
    # <<< ADDED IMPORTS for Web/Visual Skills >>>
    from a3x.skills.web_search import web_search
    from a3x.skills.browser_skill import (
        open_url,
        click_element,
        fill_form_field,
        get_page_content,
        # close_browser # Optionally add later if needed
    )
    from a3x.skills.perception.describe_image_blip import describe_image_blip
    # --- End Added Imports ---
    # <<< ADDED: Import execute_code >>>
    from a3x.skills.execute_code import execute_code
    # <<< END ADDED >>>
    # <<< ADDED: Import propose_skill_from_gap >>>
    from a3x.skills.propose_skill_from_gap import propose_skill_from_gap
    # <<< END ADDED >>>
    # <<< ADDED: Import reload_generated_skills >>>
    from a3x.skills.reload_generated_skills import reload_generated_skills
    # <<< END ADDED >>>
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"[CLI Interface FATAL] Failed to import core modules: {e}. Ensure PYTHONPATH is correct or run from project root.")
    logger.warning(f"Could not import training module: {e}. --train command will be unavailable.")
    sys.exit(1)

# Imports de CLI (após adicionar ao path)
try:
    from a3x.cli.interface import run_cli
except ImportError as e:
    print(f"[CLI Interface FATAL] Could not import run_cli: {e}")
    sys.exit(1)

# Setup logging ASAP
logger = setup_logging() # Assuming setup_logging doesn't need config dict anymore

# Initialize Rich Console
console = Console()

# DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Your goal is to assist the user with their tasks, providing accurate and relevant information."

def parse_arguments():
    """Parses command line arguments for the A³X Assistant CLI."""
    parser = argparse.ArgumentParser(description="A³X Assistant CLI")
    
    # Removed the 'run' subcommand, making --task the trigger
    parser.add_argument(
        "--task",
        type=str,
        required=True, # Task is now required to run
        help="The task for the agent to perform."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistral",
        help="Name of the LLM model to use."
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size for the LLM."
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=0,
        help="Number of GPU layers to offload."
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        help='Set the logging level.'
    )
    parser.add_argument(
        '--log-file', 
        type=str, 
        default=None, 
        help='Specify a file to write logs to.'
    )
    parser.add_argument(
        '--max-steps', 
        type=int, 
        default=10, 
        help='Maximum number of steps the agent can take.'
    )

    return parser.parse_args()

# <<< Context Definition (Placeholder/Example) >>>
class CLISkillContext: # Define a context class specific to CLI usage
    def __init__(self, logger_instance, llm_interface_instance):
        self.logger = logger_instance
        self.llm_interface = llm_interface_instance
        self.workspace_root = Path(PROJECT_ROOT).resolve()

# <<< Helper function to generate basic schema >>>
def generate_schema(func: Callable, name: str, description: str) -> Dict[str, Any]:
    """Generates a basic JSON-like schema from function signature and docstring."""
    schema = {
        "name": name,
        "description": description or f"Skill: {name}",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
    try:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        for param_name, param in sig.parameters.items():
            # Skip context-like parameters injected by system? (e.g., 'ctx', 'context')
            # For now, include all parameters found.
            #if param_name in ['ctx', 'context']: 
            #    continue
            
            param_info = {}
            param_type = type_hints.get(param_name)
            
            # Basic type mapping (extend as needed)
            if param_type == str:
                param_info["type"] = "string"
            elif param_type == int:
                param_info["type"] = "integer"
            elif param_type == float:
                param_info["type"] = "number"
            elif param_type == bool:
                param_info["type"] = "boolean"
            elif param_type == list or getattr(param_type, '__origin__', None) == list:
                 param_info["type"] = "array"
                 # Try to infer item type (basic)
                 item_type_args = getattr(param_type, '__args__', [])
                 if item_type_args and item_type_args[0] == str:
                    param_info["items"] = {"type": "string"}
                 else:
                    param_info["items"] = {} # Generic item
            elif param_type == dict or getattr(param_type, '__origin__', None) == dict:
                 param_info["type"] = "object"
            else:
                param_info["type"] = "string" # Default to string if type is unknown/complex

            # Add description from docstring if possible (simple parsing)
            # This requires a specific docstring format (e.g., Google style)
            # For now, just add the type.
            # param_info["description"] = f"Parameter {param_name}" 
            
            schema["parameters"]["properties"][param_name] = param_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)
                
    except Exception as e:
        logger.warning(f"Could not introspect signature for {name}: {e}")
        
    return schema

async def main():
    """Main function to run the A³X Assistant CLI."""
    args = parse_arguments()
    
    # Setup logging using arguments FIRST
    setup_logging(log_level_str=args.log_level, log_file_path=args.log_file)
    logger = logging.getLogger(__name__) # Now get the configured logger
    
    # --- Initialize Database EARLY --- #
    try:
        logger.info("Initializing database structure...")
        initialize_database() # Call using default DATABASE_PATH from config
        logger.info("Database initialized successfully.")
    except Exception as db_init_err:
        logger.fatal(f"FATAL: Failed to initialize database: {db_init_err}", exc_info=True)
        return # Exit if DB initialization fails
    # --- End DB Initialization --- #

    # Log the received arguments
    logger.info(f"Received task: {args.task}")
    logger.info(f"Using model name: {args.model_name}")
    logger.info(f"Using context size: {args.context_size}")
    logger.info(f"Using NGL: {args.ngl}")
    logger.info(f"Log Level set to: {args.log_level}")
    logger.info(f"Log File path: {args.log_file or 'Default path used'}")
    logger.info(f"Maximum steps set to: {args.max_steps}")
    
    # Use imported constants directly
    db_path = DATABASE_PATH
    logger.info(f"Using database path from config: {db_path}")
    semantic_index_path = SEMANTIC_INDEX_PATH
    semantic_top_k = SEMANTIC_SEARCH_TOP_K
    episodic_limit = EPISODIC_RETRIEVAL_LIMIT

    # Create config dict for MemoryManager
    memory_config = {
        "DATABASE_PATH": db_path,
        "SEMANTIC_INDEX_PATH": semantic_index_path,
        "SEMANTIC_SEARCH_TOP_K": semantic_top_k,
        "EPISODIC_RETRIEVAL_LIMIT": episodic_limit
    }
    logger.info(f"Initializing MemoryManager with config: {memory_config}")
    memory_manager = MemoryManager(config=memory_config)

    # Initialize ServerManager
    # Use LLAMA_SERVER_URL from env or default from llm_interface
    llm_url = os.getenv("LLAMA_SERVER_URL", DEFAULT_LLM_URL)
    if not llm_url:
        logger.error("LLM Server URL is not configured. Please set LLAMA_SERVER_URL in your env or config.")
        return

    server_host = '127.0.0.1'
    server_port = 8080
    try:
        parsed_url = urlparse(llm_url)
        server_host = parsed_url.hostname or server_host
        server_port = parsed_url.port or server_port
    except Exception as e:
        logger.warning(f"Could not parse LLM URL {llm_url} for host/port. Using defaults {server_host}:{server_port}. Error: {e}")

    server_manager = ServerManager()

    # Instantiate LLMInterface
    completion_url = llm_url
    if not completion_url.endswith('/completion'):
        completion_url = urljoin(completion_url, 'completion')

    llm_interface = LLMInterface(
        llm_url=completion_url,
        model_name=args.model_name,
        context_size=args.context_size
    )

    # Load skills (keep this for now, maybe used elsewhere)
    skill_registry_dict = get_skill_registry() 
    logger.info(f"Loaded {len(skill_registry_dict)} skills (from dynamic load).")
    
    # --- Initialize ToolRegistry WITH tools ---
    tool_registry = ToolRegistry()
    
    # Initialize skills that need instances (like FileManager)
    # Use PROJECT_ROOT as the default workspace for the CLI run
    file_manager = FileManagerSkill(workspace_root=PROJECT_ROOT)
    
    # Register file manager skills
    for name, method in inspect.getmembers(FileManagerSkill, predicate=inspect.isfunction):
        if hasattr(method, '_skill_name'): # Check for the decorator attribute
            skill_name = getattr(method, '_skill_name')
            bound_method = getattr(file_manager, name) # Get bound method
            description = getattr(method, '_skill_description', method.__doc__ or f"Skill: {skill_name}")
            # Generate schema
            schema = generate_schema(method, skill_name, description)
            tool_registry.register_tool(
                name=skill_name,
                instance=file_manager, # Pass the instance
                tool=bound_method,
                schema=schema # Pass the generated schema
            )
            logger.debug(f"Registered tool: {skill_name} from FileManagerSkill")

    # Register standalone skills
    planner_description = hierarchical_planner.__doc__ or "Generates a plan."
    planner_schema = generate_schema(hierarchical_planner, "hierarchical_planner", planner_description)
    tool_registry.register_tool(
        name="hierarchical_planner",
        instance=None,
        tool=hierarchical_planner,
        schema=planner_schema
    )
    logger.debug("Registered tool: hierarchical_planner")
    
    final_answer_description = final_answer.__doc__ or "Provides the final answer."
    final_answer_schema = generate_schema(final_answer, "final_answer", final_answer_description)
    tool_registry.register_tool(
        name="final_answer",
        instance=None,
        tool=final_answer,
        schema=final_answer_schema
    )
    logger.debug("Registered tool: final_answer")
    
    learning_cycle_description = learning_cycle.__doc__ or "Learning cycle skill."
    learning_cycle_schema = generate_schema(learning_cycle, "learning_cycle", learning_cycle_description)
    tool_registry.register_tool(
        name="learning_cycle",
        instance=None,
        tool=learning_cycle,
        schema=learning_cycle_schema
    )
    logger.debug("Registered tool: learning_cycle")
    
    # <<< ADDED: Register Web and Visual Skills >>>
    # --- Web Search ---
    try:
        web_search_description = web_search.__doc__ or "Performs a web search."
        web_search_schema = generate_schema(web_search, "web_search", web_search_description)
        tool_registry.register_tool(
            name="web_search",
            instance=None,
            tool=web_search,
            schema=web_search_schema
        )
        logger.debug("Registered tool: web_search")
    except Exception as e:
        logger.error(f"Failed to register web_search: {e}")

    # --- Browser Skills ---
    # Note: These might ideally belong to a BrowserSkill instance if they share state,
    # but registering as standalone for now based on initial observation.
    browser_skills_to_register = {
        "open_url": open_url,
        "click_element": click_element,
        "fill_form_field": fill_form_field,
        "get_page_content": get_page_content,
    }
    for skill_name, skill_func in browser_skills_to_register.items():
        try:
            description = skill_func.__doc__ or f"Browser skill: {skill_name}"
            schema = generate_schema(skill_func, skill_name, description)
            tool_registry.register_tool(
                name=skill_name,
                instance=None, # Assuming standalone for now
                tool=skill_func,
                schema=schema
            )
            logger.debug(f"Registered tool: {skill_name}")
        except Exception as e:
            logger.error(f"Failed to register {skill_name}: {e}")
            
    # --- Visual Perception ---
    try:
        describe_image_blip_description = describe_image_blip.__doc__ or "Describes an image using BLIP."
        describe_image_blip_schema = generate_schema(describe_image_blip, "describe_image_blip", describe_image_blip_description)
        tool_registry.register_tool(
            name="describe_image_blip",
            instance=None,
            tool=describe_image_blip,
            schema=describe_image_blip_schema
        )
        logger.debug("Registered tool: describe_image_blip")
    except Exception as e:
        logger.error(f"Failed to register describe_image_blip: {e}")
        
    # <<< ADDED: Register execute_code skill >>>
    try:
        execute_code_description = execute_code.__doc__ or "Executes a code block."
        execute_code_schema = generate_schema(execute_code, "execute_code", execute_code_description)
        tool_registry.register_tool(
            name="execute_code",
            instance=None, # Standalone function
            tool=execute_code,
            schema=execute_code_schema
        )
        logger.debug("Registered tool: execute_code")
    except Exception as e:
        logger.error(f"Failed to register execute_code: {e}")
    # <<< END ADDED >>>
    
    # <<< ADDED: Register propose_skill_from_gap >>>
    try:
        propose_skill_description = propose_skill_from_gap.__doc__ or "Generates Python code for a new skill."
        propose_skill_schema = generate_schema(propose_skill_from_gap, "propose_skill_from_gap", propose_skill_description)
        tool_registry.register_tool(
            name="propose_skill_from_gap",
            instance=None, # Standalone function
            tool=propose_skill_from_gap,
            schema=propose_skill_schema
        )
        logger.debug("Registered tool: propose_skill_from_gap")
    except Exception as e:
        logger.error(f"Failed to register propose_skill_from_gap: {e}")
    # <<< END ADDED >>>
    
    # <<< ADDED: Register reload_generated_skills >>>
    try:
        reload_skills_description = reload_generated_skills.__doc__ or "Reloads and registers skills from the generated directory."
        reload_skills_schema = generate_schema(reload_generated_skills, "reload_generated_skills", reload_skills_description)
        tool_registry.register_tool(
            name="reload_generated_skills",
            instance=None, # Standalone function
            tool=reload_generated_skills,
            schema=reload_skills_schema
        )
        logger.debug("Registered tool: reload_generated_skills")
    except Exception as e:
        logger.error(f"Failed to register reload_generated_skills: {e}")
    # <<< END ADDED >>>
        
    logger.info(f"Initialized ToolRegistry with {len(tool_registry.list_tools())} tools.")

    # --- Initialize FragmentRegistry ---
    fragment_registry = FragmentRegistry() 
    logger.info(f"Initialized FragmentRegistry. Found {len(fragment_registry.get_all_definitions())} fragment definitions.")

    try:
        logger.info("Starting llama-server...")
        await server_manager.start_server(server_name="llama")
        await asyncio.sleep(2)

        agent = ReactAgent(
            agent_id="1",
            llm_interface=llm_interface,
            skill_registry=skill_registry_dict,
            tool_registry=tool_registry,
            fragment_registry=fragment_registry,
            memory_manager=memory_manager,
            workspace_root=str(PROJECT_ROOT),
            logger=logger
        )

        # --- Execute Task using run_task --- 
        logger.info(f"[CLI] Starting agent task: {args.task}")
        result = await agent.run_task(objective=args.task, max_steps=args.max_steps)

        # --- Handle Result --- 
        if result.get("status") == "success":
            print("\n\033[92m✅ Final Answer:\033[0m") # Green color for success
            # Print final answer content in cyan
            console.print(result.get("final_answer", "(No final answer content)"), style="cyan")
        else:
            print("\n\033[91m❌ Task Failed:\033[0m") # Red color for failure
            # Print failure message in red
            console.print(result.get("message", "Unknown error"), style="red")

        logger.info("[CLI] Task execution finished.")

    except Exception as e:
        logger.fatal(f"❌ FATAL: Unexpected error during agent execution: {e}", exc_info=True)
    finally:
        logger.info("Stopping llama-server...")
        await server_manager.stop_server(server_name="llama")
        logger.info("Server stop initiated.")

        # <<< ADDED: Close all open DB connections >>>
        logger.info("Closing database connections...")
        closed_count = 0
        # Iterate over a copy of the keys to avoid issues if dict changes during iteration
        db_paths = list(_db_connections.keys())
        for db_path in db_paths:
            conn = _db_connections.pop(db_path, None)
            if conn:
                try:
                    await close_db_connection(conn)
                    logger.info(f"Closed DB connection for: {db_path}")
                    closed_count += 1
                except Exception as db_close_err:
                    logger.error(f"Error closing DB connection for {db_path}: {db_close_err}")
        logger.info(f"Database cleanup complete. Closed {closed_count} connection(s).")
        # <<< END ADDED >>>

if __name__ == "__main__":
    # Chama a função principal da interface CLI using asyncio
    asyncio.run(main())

def cli_entry():
    """Synchronous entry point for the CLI script."""
    asyncio.run(main())
