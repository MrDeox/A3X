# a3x/run/self_reflection_loop.py

import asyncio
import logging
import sys
import json
import subprocess
import time
import os
import httpx
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Adjust path to import from a3x package if running as script
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core components
from a3x.core.llm_interface import LLMInterface
from a3x.core.config import (
    LLAMA_SERVER_URL, PROJECT_ROOT, LLAMA_HEALTH_ENDPOINT, 
    LLAMA_SERVER_BINARY, LLAMA_CPP_DIR, LLAMA_SERVER_ARGS, 
    LLAMA_SERVER_STARTUP_TIMEOUT
)
from a3x.core.context import Context
from a3x.core.skills import SkillContext
from a3x.core.logging_config import setup_logging

# Reflection and Refactoring components
from a3x.reflection.structure_reflector import StructureReflector # Needed indirectly for skill context
from a3x.skills.evaluate_architecture import evaluate_architecture # Import the skill function
from a3x.fragments.architect_advisor import ArchitectAdvisorFragment
from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
from a3x.fragments.base import BaseFragment, FragmentDef # To check inheritance if needed

# --- Configuration --- #
LOOP_SLEEP_DURATION_SECONDS = 300 # 5 minutes
TARGET_MODULE_FOR_EVALUATION = "a3x" # Initial module to evaluate

# --- Setup Logging --- #
# Use the existing setup function
setup_logging()
logger = logging.getLogger("SelfReflectionLoop")


# --- Helper Functions --- #

async def check_llm_server_health(url: str, timeout: int = 5) -> bool:
    """Checks if the LLM server is responding at the given health URL."""
    if not url:
        logger.error("LLM health check URL is not configured.")
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            # Consider status 200 OK as healthy
            if response.status_code == 200:
                logger.debug(f"LLM server health check successful at {url}")
                return True
            else:
                logger.warning(f"LLM server health check at {url} failed with status: {response.status_code}")
                return False
    except httpx.RequestError as e:
        logger.warning(f"LLM server health check connection error at {url}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during LLM server health check at {url}: {e}")
        return False

def start_llm_server_process() -> Optional[subprocess.Popen]:
    """Attempts to start the LLM server process in the background."""
    server_binary_path = Path(LLAMA_SERVER_BINARY)
    cpp_dir_path = Path(LLAMA_CPP_DIR)

    if not server_binary_path.exists():
        logger.error(f"LLM server binary not found at: {server_binary_path}")
        logger.error("Please ensure llama.cpp is built correctly.")
        # Optionally try to build here, but it's complex. For now, just error out.
        # build_script = cpp_dir_path / "build.sh" or similar
        # if build_script.exists(): run build ... else error
        return None

    if not cpp_dir_path.is_dir():
        logger.error(f"llama.cpp directory not found at: {cpp_dir_path}")
        return None

    command = [str(server_binary_path)] + LLAMA_SERVER_ARGS
    logger.info(f"Attempting to start LLM server with command: {' '.join(command)}")
    logger.info(f"Working directory: {cpp_dir_path}")

    try:
        # Start in background, redirect output to avoid blocking
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True # Detach from parent
        )
        logger.info(f"LLM server process started with PID: {process.pid}")
        return process
    except FileNotFoundError:
        logger.error(f"Failed to start LLM server: Command not found ({server_binary_path}). Is it executable?")
        return None
    except Exception as e:
        logger.exception(f"Failed to start LLM server process: {e}")
        return None

# --- Dummy Components for Fragment Instantiation --- #

class DummyCommunicator:
    """A dummy communicator that logs broadcasts and doesn't listen."""
    async def broadcast(self, message: Any, msg_type: str):
        log_msg = message
        if isinstance(message, dict): # Pretty print dicts
             try:
                 log_msg = json.dumps(message, indent=2)
             except TypeError:
                 log_msg = str(message) # Fallback
        logger.debug(f"[DummyCommunicator] BROADCAST ({msg_type}):\n{log_msg}")

    async def listen(self, msg_type: str):
        """Dummy async generator that never yields."""
        logger.debug(f"[DummyCommunicator] LISTEN called for {msg_type}, but will not yield.")
        if False:
            yield {}

class DummySkillExecutor:
    """A dummy skill executor that logs calls but doesn't execute."""
    def __init__(self, context: SkillContext):
        # Store context if BaseFragment needs it via executor
        self.context = context

    async def execute(self, skill_name: str, args: dict) -> Any:
        logger.warning(f"[DummySkillExecutor] Attempted to execute skill '{skill_name}' with args {args}. Returning None.")
        # We call evaluate_architecture directly, so this shouldn't be strictly needed
        # unless BaseFragment or ArchitectAdvisorFragment requires a functional executor during init.
        return None

# --- Main Reflection Loop --- #

async def run_self_reflection_loop():
    """Runs the continuous loop of architectural evaluation and refactoring simulation."""
    logger.info("Starting Self-Reflection Loop...")

    # --- Check/Start LLM Server --- #
    logger.info(f"Checking LLM server health at {LLAMA_HEALTH_ENDPOINT}...")
    server_process = None # Keep track of the process if we start it
    is_healthy = await check_llm_server_health(LLAMA_HEALTH_ENDPOINT)

    if not is_healthy:
        logger.warning("LLM server is not responding. Attempting to start it...")
        server_process = start_llm_server_process()
        if server_process is None:
            logger.critical("Failed to start the LLM server process. Exiting.")
            return # Exit if server couldn't be started
        
        logger.info(f"Waiting up to {LLAMA_SERVER_STARTUP_TIMEOUT} seconds for server to initialize...")
        await asyncio.sleep(LLAMA_SERVER_STARTUP_TIMEOUT)
        
        logger.info("Re-checking LLM server health after startup attempt...")
        is_healthy = await check_llm_server_health(LLAMA_HEALTH_ENDPOINT)
        if not is_healthy:
            logger.critical("LLM server failed to become healthy after startup attempt. Exiting.")
            # Consider terminating server_process here if needed, though it runs detached.
            return # Exit if server didn't start correctly
        logger.info("LLM server started successfully.")
    else:
        logger.info("LLM server is already running.")

    # --- Initialize Core Components --- #
    logger.info(f"Initializing LLMInterface with URL: {LLAMA_SERVER_URL}")
    try:
        # Corrected parameter name from base_url to llm_url
        llm_interface = LLMInterface(llm_url=LLAMA_SERVER_URL)
        # Add health check if needed, though it might be implicitly handled by the first call
        # if not await check_llm_server_health(llm_interface.llm_url):
        #     logger.error("LLM server is not healthy. Exiting.")
        #     return
    except Exception as e:
        logger.exception(f"Failed to initialize LLMInterface: {e}")
        return

    # Create SkillContext (or Context if that's the correct class name)
    # Pass only the expected parameters
    skill_context = Context(
        logger=logger, 
        # llm_url=llm_interface.llm_url, # llm_url can be derived from llm_interface
        llm_interface=llm_interface, # <<< Pass the interface object >>>
        # mem={}, # Initialize memory if needed by skills
        # tools={} # Pass tools if needed by skills
    )
    logger.info("Context created.")

    # --- Instantiate Fragments (with dummies) --- #
    logger.info("Instantiating reflection/refactoring fragments...")
    dummy_communicator = DummyCommunicator()
    dummy_executor = DummySkillExecutor(context=skill_context)

    try:
        # <<< Create FragmentDef instances >>>
        architect_advisor_def = FragmentDef(
            name="architect_advisor_standalone", 
            fragment_class=ArchitectAdvisorFragment,
            description="Advisor for architecture in standalone mode."
            # Add skills if needed by base class or logic
        )
        structure_refactorer_def = FragmentDef(
            name="structure_refactor_standalone",
            fragment_class=StructureAutoRefactorFragment,
            description="Refactorer for structure in standalone mode."
            # Add skills if needed by base class or logic
        )
        
        # <<< Pass FragmentDef instances using the correct argument name >>>
        architect_advisor = ArchitectAdvisorFragment(
            fragment_def=architect_advisor_def,
            # Pass tool_registry if needed by the fragment or base class
            # tool_registry=... 
        )
        # StructureAutoRefactor needs communicator for BaseFragment init and broadcast_result
        structure_refactorer = StructureAutoRefactorFragment(
             fragment_def=structure_refactorer_def,
            # Pass tool_registry if needed by the fragment or base class
            # tool_registry=... 
         )
        logger.info("Fragments instantiated.")
    except Exception as e:
        logger.exception(f"Failed to instantiate fragments: {e}")
        return

    # --- Main Loop --- #
    while True:
        logger.info("--- Starting new reflection cycle --- stimulating evaluate_architecture ")
        try:
            # 1. Call evaluate_architecture skill directly
            logger.info(f"Running architecture evaluation for: {TARGET_MODULE_FOR_EVALUATION}")
            evaluation_report = await evaluate_architecture(
                context=skill_context,
                module_path=TARGET_MODULE_FOR_EVALUATION
            )

            if not evaluation_report or evaluation_report.startswith("Error:"):
                logger.error(f"Architecture evaluation failed or returned error: {evaluation_report}")
                # Wait before retrying if evaluation failed
                await asyncio.sleep(LOOP_SLEEP_DURATION_SECONDS)
                continue

            logger.info(f"Architecture evaluation complete. Report length: {len(evaluation_report)}")
            logger.debug(f"Evaluation Report (first 500 chars):\n{evaluation_report[:500]}...")

            # 2. Extract directives from the report
            logger.info("Extracting directives from report...")
            directives = architect_advisor.extract_directives_from_report(evaluation_report)

            if not directives:
                logger.info("No actionable directives found in the report.")
            else:
                logger.info(f"Found {len(directives)} directives. Handling them...")
                # 3. Handle each directive
                for directive in directives:
                    logger.info(f"Handling directive -> Action: {directive.get('action')}, Target: {directive.get('target')}")
                    await structure_refactorer.handle_directive(directive)
                    # Small pause between handling directives if needed
                    await asyncio.sleep(1)

        except Exception as cycle_err:
            logger.exception(f"Error occurred during reflection cycle: {cycle_err}")

        # 6. Wait and repeat
        logger.info(f"Reflection cycle finished. Sleeping for {LOOP_SLEEP_DURATION_SECONDS} seconds...")
        await asyncio.sleep(LOOP_SLEEP_DURATION_SECONDS)

# --- Entry Point --- #
if __name__ == "__main__":
    print("Starting A³X Self-Reflection Loop...")
    print(f"Target Module: {TARGET_MODULE_FOR_EVALUATION}")
    print(f"Loop Delay: {LOOP_SLEEP_DURATION_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    # Ensure LLM Server is running at the configured URL!
    server_process_ref = None # To potentially hold the process reference
    try:
        # The loop function now handles server start implicitly
        asyncio.run(run_self_reflection_loop()) 
    except KeyboardInterrupt:
        logger.info("Self-reflection loop stopped by user.")
        print("\nSelf-reflection loop stopped.")
    except Exception as main_err:
        logger.critical(f"Critical error running self-reflection loop: {main_err}", exc_info=True)
        print(f"\nCritical Error: {main_err}") 