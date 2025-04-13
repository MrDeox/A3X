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
    from a3x.core.db_utils import initialize_database
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

# <<< Imports Globais >>>
from a3x.core.skills import get_skill_registry

# Setup logging ASAP
logger = setup_logging() # Assuming setup_logging doesn't need config dict anymore

# DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Your goal is to assist the user with their tasks, providing accurate and relevant information."

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='A³X Assistant CLI')
    parser.add_argument('--task', type=str, help='The task for the agent to perform')
    # Use a hardcoded default for model name
    parser.add_argument('--model-name', type=str, default='default_model', help='Name of the model to use (default: default_model)')
    parser.add_argument('--context-size', type=int, default=4096, help='Context size for the LLM (default: 4096)')
    parser.add_argument('--ngl', type=int, default=20, help='Number of GPU layers to offload (default: 20)')
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Nível de logging para a console e arquivo (padrão: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None, # Default to None, setup_logging handles default path
        help="Arquivo para salvar os logs (padrão: logs/a3x_cli.log)"
    )
    return parser.parse_args()

# <<< Context Definition (Placeholder/Example) >>>
class CLISkillContext: # Define a context class specific to CLI usage
    def __init__(self, logger_instance, llm_interface_instance):
        self.logger = logger_instance
        self.llm_interface = llm_interface_instance
        self.workspace_root = Path(PROJECT_ROOT).resolve()

async def main():
    """Main function to run the A³X Assistant CLI."""
    args = parse_arguments()
    
    # Setup logging using arguments FIRST
    setup_logging(log_level_str=args.log_level, log_file_path=args.log_file)
    logger = logging.getLogger(__name__) # Now get the configured logger
    
    # Log the received arguments
    logger.info(f"Received task: {args.task}")
    logger.info(f"Using model name: {args.model_name}")
    logger.info(f"Using context size: {args.context_size}")
    logger.info(f"Using NGL: {args.ngl}")
    logger.info(f"Log Level set to: {args.log_level}")
    logger.info(f"Log File path: {args.log_file or 'Default path used'}")
    
    # Use imported constants directly
    db_path = DATABASE_PATH
    logger.info(f"Using database path from config: {db_path}")

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

    server_manager = ServerManager(host=server_host, port=server_port, ngl=args.ngl)

    # Instantiate LLMInterface
    completion_url = llm_url
    if not completion_url.endswith('/completion'):
        completion_url = urljoin(completion_url, 'completion')

    llm_interface = LLMInterface(
        llm_url=completion_url,
        model_name=args.model_name,
        context_size=args.context_size
    )

    # Load skills and get the registry dictionary
    skill_registry_dict = get_skill_registry() 
    logger.info(f"Loaded {len(skill_registry_dict)} skills.")
    
    # Create config dict for MemoryManager
    memory_config = {
        "DATABASE_PATH": DATABASE_PATH,
        "SEMANTIC_INDEX_PATH": SEMANTIC_INDEX_PATH,
        "SEMANTIC_SEARCH_TOP_K": SEMANTIC_SEARCH_TOP_K,
        "EPISODIC_RETRIEVAL_LIMIT": EPISODIC_RETRIEVAL_LIMIT
    }
    # Initialize MemoryManager with the config dict
    memory = MemoryManager(config=memory_config)

    try:
        logger.info("Starting llama-server...")
        server_manager.start_server() # Synchronous call
        await asyncio.sleep(2)

        agent = ReactAgent(
            agent_id="1",
            llm_interface=llm_interface,
            skill_registry=skill_registry_dict,
            workspace_root=str(PROJECT_ROOT)
        )

        # --- Execute Task using run_task --- 
        logger.info(f"[CLI] Starting agent task: {args.task}")
        result = await agent.run_task(objective=args.task)

        # --- Handle Result --- 
        if result.get("status") == "success":
            print("\n\033[92m✅ Final Answer:\033[0m") # Green color for success
            print(result.get("final_answer", "(No final answer content)"))
        else:
            print("\n\033[91m❌ Task Failed:\033[0m") # Red color for failure
            print(result.get("message", "Unknown error"))

        logger.info("[CLI] Task execution finished.")

    except Exception as e:
        logger.fatal(f"❌ FATAL: Unexpected error during agent execution: {e}", exc_info=True)
    finally:
        logger.info("Stopping llama-server...")
        server_manager.stop_server() # Synchronous call
        logger.info("Synchronous cleanup finished.")

if __name__ == "__main__":
    # Chama a função principal da interface CLI using asyncio
    asyncio.run(main())
