import pytest
import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock
import logging
import subprocess
import signal
import time
import json
import requests
# --- Core component imports for specs/instantiation ---
from core.agent import ReactAgent  # Corrected import: ReactAgent
from core.config import (  # Assuming these exist in your config
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,  # Use consistently named root
    LLAMA_MODEL_PATH as DEFAULT_MODEL_PATH,
    CONTEXT_SIZE as DEFAULT_CONTEXT_SIZE,
)
# from core.llm_interface import llm_interface # Removed problematic import
# from core.db_utils import DatabaseManager # Removed import
# from core.execution_logic import execute_tool # Removed import

# Add project root to sys path for imports
# Determine the project root directory based on the location of conftest.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[conftest.py] Added project root to sys.path: {project_root}")

# Configure logging for the fixture
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_handler = logging.StreamHandler(sys.stdout)  # Log to stdout for pytest -s capture
log_handler.setFormatter(log_formatter)

fixture_logger = logging.getLogger("ManagedLlamaServerFixture")
fixture_logger.addHandler(log_handler)
fixture_logger.setLevel(logging.INFO)  # Set desired log level (INFO, DEBUG, etc.)

# Define the base URL directly for now
# LLAMA_SERVER_URL_BASE = "http://127.0.0.1:8080" # Original default

# --- Test Server Configuration ---
TEST_SERVER_PORT = 8081  # Use a different port for testing
TEST_SERVER_HOST = "127.0.0.1"
TEST_SERVER_BASE_URL = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"
TEST_SERVER_COMPLETIONS_URL = f"{TEST_SERVER_BASE_URL}/v1/chat/completions"
# Use /health endpoint for readiness check (introduced in later llama.cpp versions)
READINESS_CHECK_URL = f"{TEST_SERVER_BASE_URL}/health"
SERVER_LOG_PATH = os.path.join(CONFIG_PROJECT_ROOT, "llama_server_session_test.log")
LLAMA_EXECUTABLE_PATH = os.path.join(
    CONFIG_PROJECT_ROOT, "llama.cpp", "build", "bin", "llama-server"
)
# Default GPU layers for tests (can be overridden by env var)
DEFAULT_TEST_GPU_LAYERS = int(os.getenv("PYTEST_LLAMA_GPU_LAYERS", "-1"))
# Default Model Path for tests (can be overridden by env var)
DEFAULT_TEST_MODEL_PATH = os.getenv("PYTEST_LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH)
DEFAULT_TEST_CONTEXT_SIZE = int(
    os.getenv("PYTEST_LLAMA_CONTEXT_SIZE", str(DEFAULT_CONTEXT_SIZE))
)

# Define the actual model path
# Use absolute path for clarity in fixture
LLAMA_CPP_SERVER_PATH = "/home/arthur/projects/A3X/llama.cpp/build_rocm/bin/llama-server"  # Corrected path based on user-provided folder content
# REAL_MODEL_PATH = os.path.join(project_root, "models", "test-model.gguf") # Old dummy model
REAL_MODEL_PATH = "/home/arthur/projects/A3X/models/gemma-3-4b-it-Q4_K_M.gguf"  # Using Gemma model as requested


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
    mock = AsyncMock()  # Removed spec=DatabaseManager
    # Configure mock methods that might be called during agent initialization or run
    mock.initialize_db.return_value = None
    mock.save_memory.return_value = None
    mock.load_memory.return_value = []  # Example: Assume loading returns an empty list
    mock.recall_memories.return_value = []  # Example: Assume recall returns an empty list
    return mock


# Placeholders for other dependencies of the new agent_instance
@pytest.fixture
def mock_llm_interface():
    return MagicMock()  # Removed spec=llm_interface


@pytest.fixture
def mock_planner():
    return AsyncMock()  # Removed spec


@pytest.fixture
def mock_reflector():
    return AsyncMock()  # Removed spec


@pytest.fixture
def mock_parser():
    return MagicMock()  # Removed spec


@pytest.fixture
def mock_tool_executor():
    return AsyncMock()  # Removed spec


# --- NEW Fixture for Temporary Test Files ---


@pytest.fixture
def temp_workspace_files(tmp_path_factory):
    """Provides a temporary directory for test files simulating the workspace.

    Yields:
        pathlib.Path: The path to the temporary directory.
    """
    # Create a base temporary directory managed by pytest
    # tmp_path_factory manages cleanup automatically.
    # We create a subdirectory to better simulate a workspace structure if needed.
    temp_dir = tmp_path_factory.mktemp("test_workspace_")
    yield temp_dir
    # Cleanup is handled by tmp_path_factory


# Updated agent_instance fixture using the new dependencies
@pytest.fixture
def agent_instance(
    mock_llm_interface,
    mock_planner,
    mock_reflector,
    mock_parser,
    mock_tool_executor,
    mock_db,
    mock_llm_url,
):
    """Provides a fully mocked Agent instance for testing basic execution flow."""
    # Mock load_agent_state which might be called during init
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("core.agent.load_agent_state", lambda _: {})
        # Instantiate ReactAgent with the provided mock_llm_url and a mock system_prompt
        # The other mock fixtures are used by tests to patch module-level functions
        agent_obj = ReactAgent(llm_url=mock_llm_url, system_prompt="mock_system_prompt")
    return agent_obj  # Return the agent object


# --- NEW SESSION-SCOPED SERVER FIXTURE ---
@pytest.fixture(scope="session")
def managed_llama_server_session(request):
    """Starts and stops the REAL llama.cpp server for integration tests in session scope."""
    server_process = None
    llama_server_path = LLAMA_CPP_SERVER_PATH
    model_path = REAL_MODEL_PATH
    host = TEST_SERVER_HOST
    port = TEST_SERVER_PORT
    # Adjust context size and GPU layers as needed for your model/hardware
    ctx_size = DEFAULT_CONTEXT_SIZE
    n_gpu_layers = 33  # Using 33 GPU layers as requested

    # Check if server binary exists
    if not os.path.exists(llama_server_path):
        pytest.fail(
            f"llama.cpp server binary not found at: {llama_server_path}. Please adjust LLAMA_CPP_SERVER_PATH.",
            pytrace=False,
        )
    # Check if model exists
    if not os.path.exists(model_path):
        pytest.fail(
            f"LLM model file not found at: {model_path}. Please adjust REAL_MODEL_PATH or download the model.",
            pytrace=False,
        )

    # Command to start the server
    cmd = [
        llama_server_path,
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(ctx_size),
        # Add other relevant parameters
        "-ngl",
        str(n_gpu_layers),
        "--log-disable",  # Optional: disable server logs for cleaner test output
    ]

    fixture_logger.info(
        f"Starting REAL llama.cpp server (Gemma) for session: {' '.join(cmd)}"
    )

    # --- Best effort to kill previous instances ---
    try:
        pkill_cmd = ["pkill", "-f", os.path.basename(llama_server_path)]
        fixture_logger.info(f"Running pre-kill command: {' '.join(pkill_cmd)}")
        subprocess.run(pkill_cmd, timeout=5, check=False, capture_output=True)
        fixture_logger.info("Pre-kill command finished (exit code ignored).")
        time.sleep(2)  # Give processes time to terminate if killed
    except FileNotFoundError:
        fixture_logger.warning("'pkill' command not found. Skipping pre-kill step.")
    except Exception as pre_kill_err:
        fixture_logger.warning(f"Error during pre-kill: {pre_kill_err}")
    # --- End pre-kill ---

    try:
        # Start the server process
        with open(SERVER_LOG_PATH, "w") as log_file:
            server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Crucial for creating a process group
            )
        time.sleep(0.5)  # Give time for pid/pgid to be available
        try:
            pgid = os.getpgid(server_process.pid)
            fixture_logger.info(
                f"REAL server started successfully (PID: {server_process.pid}) on port {port}"
            )
        except ProcessLookupError:
            fixture_logger.error(
                "REAL server process terminated immediately after starting! Check log file."
            )
            # Try reading the log file for more details
            log_content = "Could not read log file."
            try:
                with open(SERVER_LOG_PATH, "r") as f:
                    log_content = f.read()
            except Exception:
                pass  # Ignore read errors
            pytest.fail(
                f"REAL server failed to start.\nCmd: {' '.join(cmd)}\nLog ({SERVER_LOG_PATH}):\n{log_content}",
                pytrace=False,
            )

    except Exception as start_err:
        fixture_logger.exception("Failed to start REAL llama-server process:")
        pytest.fail(
            f"Failed to start REAL server: {start_err}. Check logs/permissions.",
            pytrace=False,
        )

    # --- Wait for Server Readiness ---
    max_wait = 180  # Generous timeout for model loading
    wait_interval = 5
    start_wait = time.time()
    server_ready = False
    last_error = None

    fixture_logger.info(
        f"Waiting up to {max_wait}s for REAL server (Gemma) at {READINESS_CHECK_URL} ..."
    )
    while time.time() - start_wait < max_wait:
        # Check if process died
        if server_process.poll() is not None:
            fixture_logger.error(
                f"REAL server process PID {server_process.pid} terminated unexpectedly! Exit code: {server_process.returncode}"
            )
            pytest.fail(
                f"REAL server terminated unexpectedly. Check {SERVER_LOG_PATH}",
                pytrace=False,
            )

        # Check readiness endpoint
        try:
            response = requests.get(READINESS_CHECK_URL, timeout=3)
            # /health returns 200 OK with body {"status": "ok"} when ready
            if response.status_code == 200:
                try:
                    health_status = response.json()
                    if health_status.get("status") == "ok":
                        fixture_logger.info(
                            f"REAL server ready after ~{int(time.time() - start_wait)}s."
                        )
                        server_ready = True
                        break
                    else:
                        last_error = f"Health endpoint returned status {response.status_code} but unexpected body: {response.text[:100]}"
                        fixture_logger.debug(last_error)
                except json.JSONDecodeError:
                    last_error = f"Health endpoint returned status {response.status_code} but non-JSON body: {response.text[:100]}"
                    fixture_logger.debug(last_error)
            else:
                last_error = f"Health check failed with status: {response.status_code}"
                fixture_logger.debug(last_error)

        except requests.exceptions.ConnectionError:
            last_error = "Connection refused"
            fixture_logger.debug(f"Waiting... ({last_error})")
        except requests.exceptions.Timeout:
            last_error = "Connection timeout"
            fixture_logger.debug(f"Waiting... ({last_error})")
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            fixture_logger.warning(f"Error during readiness check: {e}")

        time.sleep(wait_interval)

    if not server_ready:
        fixture_logger.error(
            f"REAL server (Gemma) failed to become ready within {max_wait}s. Last error: {last_error}"
        )
        pytest.fail(
            f"REAL server readiness timeout. Check {SERVER_LOG_PATH}", pytrace=False
        )

    # --- Yield URL to Tests ---
    fixture_logger.info(
        f"REAL server (Gemma) ready. Yielding URL: {TEST_SERVER_BASE_URL}"
    )
    yield TEST_SERVER_BASE_URL  # Yield the base URL, tests can append endpoints

    # --- Cleanup ---
    fixture_logger.info(
        f"[Pytest Fixture] Stopping REAL server (Gemma - PID: {server_process.pid})..."
    )
    # Send SIGTERM to the entire process group
    try:
        if pgid:  # Ensure pgid was obtained
            os.killpg(pgid, signal.SIGTERM)
            fixture_logger.info(f"Sent SIGTERM to process group {pgid}.")
        else:
            fixture_logger.warning(
                f"PGID not found, sending SIGTERM directly to PID {server_process.pid}"
            )
            server_process.terminate()
    except ProcessLookupError:
        fixture_logger.warning(
            f"Process group {pgid} or PID {server_process.pid} not found during SIGTERM (already stopped?)."
        )
    except Exception as term_err:
        fixture_logger.error(f"Error sending SIGTERM: {term_err}")

    # Wait for termination
    try:
        server_process.wait(timeout=15)  # Increased timeout slightly
        fixture_logger.info("REAL server (Gemma) stopped gracefully.")
    except subprocess.TimeoutExpired:
        fixture_logger.warning(
            "[Pytest Fixture] Server did not terminate gracefully after SIGTERM, killing..."
        )
        try:
            if pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                server_process.kill()
            # Short wait to confirm kill
            try:
                server_process.wait(timeout=5)
                fixture_logger.info("REAL server (Gemma) stopped after SIGKILL.")
            except subprocess.TimeoutExpired:
                fixture_logger.error(
                    "REAL server (Gemma) could not be stopped even with SIGKILL."
                )
        except ProcessLookupError:
            fixture_logger.warning(
                "Process group/PID not found during SIGKILL (already stopped?)."
            )
        except Exception as kill_err:
            fixture_logger.error(f"Error sending SIGKILL: {kill_err}")
    except Exception as wait_err:
        fixture_logger.error(f"Error waiting for server process: {wait_err}")

    fixture_logger.info("SESSION FIXTURE: Teardown complete.")


# --- Other Fixtures (Mocked, keep for tests not needing real LLM) ---


@pytest.fixture
def mock_llm_url():
    """Fixture para fornecer uma URL mock para o LLM."""
    # <<< MODIFIED: Use a valid local URL format >>>
    return "http://127.0.0.1:12345/v1/chat/completions"  # Avoids DNS resolution errors
