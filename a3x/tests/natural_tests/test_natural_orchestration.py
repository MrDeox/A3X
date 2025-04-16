# a3x/tests/natural_tests/test_natural_orchestration.py

import logging
import asyncio
import sys
import os
from pathlib import Path
import json

# Add project root to sys.path if needed (adjust path as necessary)
project_root = Path(__file__).resolve().parents[3] # Adjust based on actual structure
sys.path.insert(0, str(project_root))

from a3x.core.agent import ReactAgent
from a3x.core.llm_interface import LLMInterface
from a3x.core.config import LLAMA_SERVER_URL # Import config URL
from a3x.core.server_manager import ServerManager # <<< ADDED Import

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger() 
root_logger.setLevel(logging.DEBUG) # Capture all logs
# Clear existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# Add console handler
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Show INFO and higher on console
root_logger.addHandler(console_handler)
# Optional: Add file handler for detailed debug logs
try:
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "natural_test.log", mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # Log everything to file
    root_logger.addHandler(file_handler)
except Exception as e:
    root_logger.warning(f"Could not configure file logger: {e}")

logger = logging.getLogger(__name__) # Logger for this script

async def run_natural_test():
    logger.info("--- Starting Natural Orchestration Test ---")

    # --- Initialize Server Manager ---
    server_manager = ServerManager()
    final_result = None # Initialize final_result

    # --- Use Context Manager for LLM Server --- 
    try:
        async with server_manager.managed_server("llama"):
            logger.info("LLM Server started successfully via ServerManager.")
            
            # --- Configuration ---
            llm_url = LLAMA_SERVER_URL or os.getenv("LLM_API_URL") 
            if not llm_url:
                logger.error("LLM Server URL variable not found. Cannot proceed.")
                return
            
            logger.info(f"Using LLM URL for Agent: {llm_url}")
            
            # Objective for the test
            natural_objective = ("Leia o conteúdo do arquivo 'exemplo.txt', "
                                 "adicione a linha 'Teste natural concluído.' no final "
                                 "e salve o arquivo como 'exemplo_modificado.txt'.")
            logger.info(f"Test Objective: {natural_objective}")

            # Ensure input file exists (created by previous step)
            input_file = project_root / "exemplo.txt"
            output_file = project_root / "exemplo_modificado.txt"
            if not input_file.exists():
                logger.error(f"Input file {input_file} not found.")
                # Consider creating it if it doesn't exist
                try:
                    input_file.write_text("Este é um arquivo de exemplo inicial.\n", encoding='utf-8')
                    logger.info(f"Created dummy input file: {input_file}")
                except Exception as create_err:
                    logger.error(f"Failed to create dummy input file: {create_err}")
                    return # Abort if we can't ensure input file
            
            # Delete old output file if it exists
            if output_file.exists():
                logger.info(f"Deleting existing output file: {output_file}")
                try:
                    output_file.unlink()
                except OSError as e:
                    logger.error(f"Failed to delete existing output file: {e}")
                    # Decide whether to continue or abort
                    return

            # --- Agent Initialization ---
            try:
                logger.info("Initializing LLMInterface for Agent...")
                # LLMInterface determines final URL internally based on passed url
                llm_interface = LLMInterface(llm_url=llm_url) 
                
                logger.info("Initializing ReactAgent...")
                agent = ReactAgent(llm_interface=llm_interface)
                logger.info("Agent initialized successfully.")
            except Exception as e:
                logger.exception("Failed to initialize agent components:")
                return # Exit if agent setup fails

            # --- Run Task ---
            try:
                logger.info("Running agent task...")
                final_result = await agent.run_task(objective=natural_objective)
                logger.info(f"--- Task Execution Finished. Final Result: ---")
                logger.info(json.dumps(final_result, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.exception("An error occurred during agent.run_task:")
                final_result = {"status": "error", "message": f"Test script error during run_task: {e}"}

            # --- Verification (Inside the context manager is fine) ---
            logger.info("--- Verifying Output File (Optional) ---")
            if output_file.exists():
                try:
                    content = output_file.read_text(encoding='utf-8')
                    logger.info(f"Content of {output_file}:\n-------\n{content}\n-------")
                    if "Teste natural concluído." in content:
                        logger.info("Verification successful: Expected line found in output file.")
                    else:
                        logger.warning("Verification failed: Expected line NOT found in output file.")
                except Exception as e:
                    logger.error(f"Error reading output file for verification: {e}")
            else:
                logger.warning(f"Output file {output_file} was not created.")

    except RuntimeError as e:
        logger.error(f"Failed to manage server: {e}")
        # Handle the case where the server couldn't be started
        final_result = {"status": "error", "message": f"Server management failed: {e}"}
    except Exception as e:
        logger.exception("An unexpected error occurred during the test run:")
        final_result = {"status": "error", "message": f"Unexpected test error: {e}"}
    finally:
        # Ensure ServerManager stops any remaining servers if context manager failed somehow
        # Although __aexit__ should handle it, this is an extra safety net.
        # logger.info("Ensuring all managed servers are stopped...")
        # await server_manager.stop_all_servers() # Might be redundant due to context manager
        logger.info(f"--- Natural Orchestration Test Finished (Final Result Logged Above) ---")
        # Optional: return final_result if needed by a test runner

if __name__ == "__main__":
    asyncio.run(run_natural_test()) 