import pytest
import subprocess
import time
import os
import signal
import requests # Para verificar se o servidor está pronto
import logging
import sys
import asyncio # Added import
import httpx # Added import
from unittest.mock import AsyncMock, MagicMock # Ensured imports

# --- Core component imports for specs/instantiation ---
from core.agent import ReactAgent # Corrected import: ReactAgent
# from core.llm_interface import llm_interface # Removed problematic import
# from core.db_utils import DatabaseManager # Removed import
# from core.config import Config # Removed import (Wasn't used)

# Configure logging for the fixture
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Log to stdout for pytest -s capture
log_handler.setFormatter(log_formatter)

fixture_logger = logging.getLogger("ManagedLlamaServerFixture")
fixture_logger.addHandler(log_handler)
fixture_logger.setLevel(logging.INFO) # Set desired log level (INFO, DEBUG, etc.)

# Define the base URL directly for now
LLAMA_SERVER_URL_BASE = "http://127.0.0.1:8080"

# --- Session-scoped Event Loop --- 
@pytest.fixture(scope="session")
def event_loop():
    """Overrides pytest default function scope event loop"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

# --- Mock Fixtures --- 

@pytest.fixture
def mock_db():
    """Provides a mock DatabaseManager."""
    # Use AsyncMock for async methods used by the agent
    mock = AsyncMock() # Removed spec=DatabaseManager
    # Configure mock methods that might be called during agent initialization or run
    mock.initialize_db.return_value = None
    mock.save_memory.return_value = None
    mock.load_memory.return_value = [] # Example: Assume loading returns an empty list
    mock.recall_memories.return_value = [] # Example: Assume recall returns an empty list
    return mock

# Placeholders for other dependencies of the new agent_instance
@pytest.fixture
def mock_llm_interface():
    return MagicMock() # Removed spec=llm_interface

@pytest.fixture
def mock_planner():
    return AsyncMock() # Removed spec

@pytest.fixture
def mock_reflector():
    return AsyncMock() # Removed spec

@pytest.fixture
def mock_parser():
    return MagicMock() # Removed spec

@pytest.fixture
def mock_tool_executor():
    return AsyncMock() # Removed spec


# Updated agent_instance fixture using the new dependencies
@pytest.fixture
def agent_instance(mock_llm_interface, mock_planner, mock_reflector, mock_parser, mock_tool_executor, mock_db):
    """Provides a fully mocked Agent instance for testing basic execution flow."""
    # Mock load_agent_state which might be called during init
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("core.agent.load_agent_state", lambda _: {})
        # Instantiate ReactAgent with only llm_url and system_prompt
        # The other mock fixtures are used by tests to patch module-level functions
        agent_obj = ReactAgent(llm_url="mock_llm_url", system_prompt="mock_system_prompt")
    return agent_obj # Return the agent object


# --- Server Management Fixture ---
@pytest.fixture(scope="function") # Escopo "function": inicia/para para CADA teste que usar
def managed_llama_server(request):
    """Fixture to start/stop llama-server for a test."""
    server_process = None
    pgid = None
    server_log_path = "/home/arthur/Projects/A3X/llama_server_test.log" # Log file in workspace root

    # --- Definir Caminhos e Modelo (Hardcoded por enquanto) ---
    # TODO: Parametrizar o modelo e caminhos via request.param ou markers
    # Assuming llama.cpp was built in the project root directory structure
    llama_cpp_dir = "/home/arthur/Projects/A3X/llama.cpp"
    llama_server_executable = os.path.join(llama_cpp_dir, "build/bin/llama-server")
    # Use a smaller, faster-loading model for testing if available, otherwise use the specified one
    # Example: model_path = "/path/to/your/smaller_test_model.gguf"
    model_path = "/home/arthur/Projects/A3X/models/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf" # Default model

    # Check if executable and model exist
    if not os.path.isfile(llama_server_executable):
        pytest.fail(f"llama-server executable not found at: {llama_server_executable}. Please build llama.cpp.")
    if not os.path.isfile(model_path):
         pytest.fail(f"LLM model file not found at: {model_path}. Please download the model.")

    readiness_check_url = f"{LLAMA_SERVER_URL_BASE}/v1/models" # Use /v1/models as readiness check

    fixture_logger.info(f"Fixture Setup: Starting llama-server with model {os.path.basename(model_path)}...")
    fixture_logger.info(f"Server executable: {llama_server_executable}")
    fixture_logger.info(f"Model path: {model_path}")
    fixture_logger.info(f"Log file: {server_log_path}")
    fixture_logger.info(f"Readiness check URL: {readiness_check_url}")


    # --- Parar Servidor Anterior (Garantia) ---
    try:
        # Using pkill with the executable name's basename for robustness
        pkill_command = ["pkill", "-f", os.path.basename(llama_server_executable)]
        fixture_logger.info(f"Attempting to stop any previous server process using: {' '.join(pkill_command)}")
        # Use run instead of check_output, capture output, don't check=True as it might fail if no process exists
        result = subprocess.run(pkill_command, capture_output=True, text=True)
        if result.returncode == 0:
             fixture_logger.info("pkill found and stopped a previous server process.")
        else:
            fixture_logger.info("No previous server process found by pkill or error occurred (this is often ok).")
            if result.stderr:
                 fixture_logger.debug(f"pkill stderr: {result.stderr}")
        time.sleep(2) # Increased pause after pkill
    except FileNotFoundError:
         fixture_logger.warning("'pkill' command not found. Skipping pre-stop step.")
    except Exception as e:
        fixture_logger.warning(f"Could not pre-stop server (maybe not running or other issue): {e}")

    # --- Iniciar Novo Servidor ---
    try:
        fixture_logger.info(f"Starting server process: {llama_server_executable} -m {model_path}")
        with open(server_log_path, "w") as log_file:
            server_process = subprocess.Popen(
                [llama_server_executable, "-m", model_path, "--port", "8080"], # Explicitly set port
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Create a new process group
            )
        # Wait a fraction of a second for the process group ID to become available
        time.sleep(0.5)
        try:
             pgid = os.getpgid(server_process.pid)
             fixture_logger.info(f"llama-server started with PID {server_process.pid} (Process Group ID: {pgid}). Log: {server_log_path}")
        except ProcessLookupError:
             fixture_logger.error("Server process terminated immediately after starting. Check logs.")
             pytest.fail("Server process failed to start correctly. Check llama_server_test.log")


        # --- Esperar Servidor Ficar Pronto ---
        max_wait = 120 # Increased max wait time (model loading can be slow)
        wait_interval = 3 # Increased interval
        start_wait = time.time()
        server_ready = False
        fixture_logger.info(f"Waiting up to {max_wait}s for server to become ready at {readiness_check_url}...")

        while time.time() - start_wait < max_wait:
            # Check if the process terminated prematurely
            if server_process.poll() is not None:
                 fixture_logger.error(f"llama-server process PID {server_process.pid} terminated unexpectedly during startup! Exit code: {server_process.returncode}")
                 # Attempt to read the tail of the log file for clues
                 try:
                     with open(server_log_path, "r") as f:
                         log_tail = f.readlines()[-20:] # Get last 20 lines
                     fixture_logger.error(f"Last lines from llama_server_test.log:\n{''.join(log_tail)}")
                 except Exception as log_e:
                     fixture_logger.error(f"Could not read log file {server_log_path}: {log_e}")
                 pytest.fail("llama-server process terminated unexpectedly during startup.") # Fail the test setup

            # Attempt to connect to the readiness endpoint
            try:
                response = requests.get(readiness_check_url, timeout=2) # Increased timeout for request
                # llama.cpp returns 200 for /v1/models when ready
                if response.status_code == 200:
                    fixture_logger.info(f"Server ready after ~{int(time.time() - start_wait)} seconds.")
                    server_ready = True
                    break
                else:
                    # Log unexpected status codes
                     fixture_logger.debug(f"Server responded with status {response.status_code}, not ready yet.")

            except requests.exceptions.ConnectionError:
                fixture_logger.debug("Server not ready yet (connection refused).")
            except requests.exceptions.Timeout:
                 fixture_logger.debug("Server not ready yet (connection timeout).")
            except Exception as e:
                 fixture_logger.warning(f"An unexpected error occurred while checking server readiness: {e}")

            fixture_logger.info(f"Still waiting... ({int(time.time() - start_wait)}s / {max_wait}s)")
            time.sleep(wait_interval)

        if not server_ready:
            fixture_logger.error(f"llama-server failed to become ready within {max_wait} seconds.")
            # Attempt to read log tail on timeout as well
            try:
                 with open(server_log_path, "r") as f:
                     log_tail = f.readlines()[-20:]
                 fixture_logger.error(f"Last lines from llama_server_test.log:\n{''.join(log_tail)}")
            except Exception as log_e:
                 fixture_logger.error(f"Could not read log file {server_log_path}: {log_e}")
            # Try to kill the potentially stuck process before failing
            if pgid and server_process and server_process.poll() is None:
                 try:
                    fixture_logger.warning(f"Attempting to kill potentially hung server process group {pgid} before failing test.")
                    os.killpg(pgid, signal.SIGKILL)
                 except Exception as kill_e:
                    fixture_logger.error(f"Failed to SIGKILL server process group {pgid}: {kill_e}")
            pytest.fail(f"llama-server failed to become ready within {max_wait} seconds. Check logs.") # Fail the test setup

        # --- Passar Controle para o Teste ---
        fixture_logger.info("Server is ready. Yielding to test function.")
        yield server_process # O teste executa aqui (pode usar o objeto process se precisar)

    finally:
        # --- Parar Servidor (Teardown) ---
        fixture_logger.info("Fixture Teardown: Cleaning up llama-server process...")
        if pgid and server_process and server_process.poll() is None: # Check if we have a pgid, process object, and it's running
            fixture_logger.info(f"Attempting to stop llama-server process group {pgid} with SIGTERM...")
            try:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    # Wait for the process to terminate
                    server_process.wait(timeout=15) # Increased timeout for graceful shutdown
                    fixture_logger.info(f"Server process group {pgid} stopped gracefully.")
                except subprocess.TimeoutExpired:
                    fixture_logger.warning(f"Server process group {pgid} did not terminate after SIGTERM. Sending SIGKILL.")
                    os.killpg(pgid, signal.SIGKILL)
                    # Short wait after SIGKILL
                    time.sleep(1)
                    if server_process.poll() is None:
                         fixture_logger.error(f"Failed to stop server process group {pgid} even with SIGKILL.")
                    else:
                         fixture_logger.info(f"Server process group {pgid} stopped with SIGKILL.")
                except Exception as wait_e: # Catch potential errors during wait
                     fixture_logger.error(f"Error waiting for server process {pgid} after SIGTERM: {wait_e}. Attempting SIGKILL.")
                     os.killpg(pgid, signal.SIGKILL) # Ensure kill attempt even if wait fails

            except ProcessLookupError:
                 fixture_logger.warning(f"Server process group {pgid} not found during teardown (already terminated?).")
            except Exception as e:
                fixture_logger.error(f"Error stopping server process group {pgid} with SIGTERM: {e}. Attempting SIGKILL.")
                try:
                    # Ensure pgid exists before trying SIGKILL again
                    if pgid:
                         os.killpg(pgid, signal.SIGKILL)
                         fixture_logger.info(f"Server process group {pgid} stopped with SIGKILL after error.")
                except Exception as kill_e:
                     fixture_logger.error(f"Failed to SIGKILL server process group {pgid} after initial error: {kill_e}")
        elif server_process and server_process.poll() is not None:
             fixture_logger.info(f"Server process PID {server_process.pid} was already terminated before teardown.")
        else:
             fixture_logger.info("No running llama-server process was found or managed by this fixture instance to stop.")

        # Optional: Clean up the log file? Or leave it for debugging?
        # try:
        #     if os.path.exists(server_log_path):
        #         os.remove(server_log_path)
        #         fixture_logger.info(f"Removed log file: {server_log_path}")
        # except Exception as e:
        #     fixture_logger.warning(f"Could not remove log file {server_log_path}: {e}")

# Example of how a test would use the fixture:
#
# def test_something_requiring_server(managed_llama_server):
#     # Your test code that interacts with http://127.0.0.1:8080
#     response = requests.get("http://127.0.0.1:8080/v1/models")
#     assert response.status_code == 200
#     # ... more test logic
#

# Removed old agent_instance and LLM_JSON_RESPONSE_HELLO_FINAL fixtures
