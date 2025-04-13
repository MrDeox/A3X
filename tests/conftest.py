import pytest
import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock
import logging
import subprocess
import time
import requests

# --- Core component imports for specs/instantiation ---
from a3x.core.agent import ReactAgent  # Corrected import: ReactAgent
from a3x.core.config import (
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    LLAMA_SERVER_MODEL_PATH as DEFAULT_MODEL_PATH,
    # CONTEXT_SIZE as DEFAULT_CONTEXT_SIZE,  # Remover ou ajustar se não existir
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
    os.getenv("PYTEST_LLAMA_CONTEXT_SIZE", "4096")
)

# Define the actual model path
# Use absolute path for clarity in fixture
# LLAMA_CPP_SERVER_PATH = "/home/arthur/projects/A3X/llama.cpp/build_rocm/bin/llama-server"  # Corrected path based on user-provided folder content
LLAMA_CPP_SERVER_PATH = "/home/arthur/projects/A3X/llama.cpp/build_vulkan/bin/llama-server"  # Using Vulkan build
# REAL_MODEL_PATH = "/home/arthur/projects/A3X/models/test-model.gguf") # Old dummy model
# REAL_MODEL_PATH = "/home/arthur/projects/A3X/models/gemma-3-4b-it-q4_0.gguf"  # Using the previous q4_0 model
REAL_MODEL_PATH = "/home/arthur/projects/A3X/models/gemma-3-4b-it-Q4_K_M.gguf"  # <<< UPDATED TO USE Q4_K_M model >>>


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
    mock.recall_memories.return_value = (
        []
    )  # Example: Assume recall returns an empty list
    # ADDED mock for CerebrumX test
    mock.retrieve_relevant_context = AsyncMock(
        return_value={
            "semantic_match": "Mocked semantic context from mock_db",
            "short_term_history": [],  # Keep history simple
        }
    )
    mock.save_agent_state.return_value = None
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
        mp.setattr("a3x.core.agent.load_agent_state", lambda *args, **kwargs: {})
        # Instantiate ReactAgent with the provided mock_llm_url and a mock system_prompt
        # The other mock fixtures are used by tests to patch module-level functions
        agent_obj = ReactAgent(llm_url=mock_llm_url, system_prompt="mock_system_prompt")
        # ADDED: Explicitly assign the mock DB to the agent instance
        agent_obj._memory = mock_db
    return agent_obj  # Return the agent object


# --- NEW Fixture for CerebrumXAgent ---
@pytest.fixture
def cerebrumx_agent_instance(
    mock_llm_interface,
    mock_planner,
    mock_reflector,
    mock_parser,
    mock_tool_executor,
    mock_db,
    mock_llm_url,
):  # Re-use the same mock dependencies
    """Provides a fully mocked CerebrumXAgent instance."""
    # Mock load_agent_state which might be called during init
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            "a3x.core.agent.load_agent_state", lambda *args, **kwargs: {}
        )  # Assuming CerebrumXAgent also uses this
        # Import CerebrumXAgent here to avoid circular dependency issues
        from a3x.core.cerebrumx import CerebrumXAgent  # Correct import path

        agent_obj = CerebrumXAgent(
            llm_url=mock_llm_url, system_prompt="mock_cerebrumx_prompt"
        )
        # ADDED: Explicitly assign the mock DB to the agent instance
        agent_obj._memory = mock_db
    return agent_obj  # Return the CerebrumXAgent object


# --- NEW SESSION-SCOPED SERVER FIXTURE ---
@pytest.fixture(scope="session")
def managed_llama_server_session(request):
    """
    Manages the lifecycle of a llama.cpp server process for a testing session.
    Starts the server, waits for it to be ready, yields the base URL,
    and ensures cleanup.
    """
    server_process = None
    llama_server_path = LLAMA_CPP_SERVER_PATH
    model_path = REAL_MODEL_PATH
    host = TEST_SERVER_HOST
    port = TEST_SERVER_PORT
    # Adjust context size and GPU layers as needed for your model/hardware
    ctx_size = DEFAULT_CONTEXT_SIZE
    # F841: n_gpu_layers = 33  # Using 33 GPU layers as requested

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
        # "--log-disable", # Keep logs enabled for debugging
    ]
    fixture_logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
    # server_log_file = open(SERVER_LOG_PATH, "w") # <<< REMOVING FILE OPEN
    # Use Popen for non-blocking start, DO NOT REDIRECT stdout/stderr
    server_process = subprocess.Popen(cmd)  # Removed stdout and stderr redirection
    fixture_logger.info(
        f"llama-server process started (PID: {server_process.pid}). Waiting for readiness..."
    )

    # --- Wait for server readiness ---
    base_url = f"http://{host}:{port}"
    health_url = f"{base_url}/health"
    max_wait_time = 30  # seconds
    poll_interval = 0.5  # seconds
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait_time:
        try:
            # Check if process exited unexpectedly
            if server_process.poll() is not None:
                fixture_logger.error(
                    f"llama-server process terminated unexpectedly with code {server_process.returncode}. Check pytest output for server logs."
                )
                break  # Exit loop if process died

            response = requests.get(
                health_url, timeout=poll_interval
            )  # Use short timeout for probe
            if response.status_code == 200:
                fixture_logger.info(
                    "llama-server /health returned 200 OK. Waiting 1s extra for stabilization..."
                )
                time.sleep(1.0)  # Add a 1-second sleep for extra stabilization
                fixture_logger.info("llama-server should be stable now.")
                server_ready = True
                break
            else:
                # Optional: Log non-200 status if needed for debugging
                # fixture_logger.debug(f"Server not ready yet, status: {response.status_code}")
                pass
        except requests.exceptions.ConnectionError:
            # fixture_logger.debug("Server not ready yet (connection error).")
            pass  # Expected while server is starting
        except requests.exceptions.Timeout:
            # fixture_logger.debug("Server not ready yet (timeout).")
            pass  # Expected if server is slow
        except Exception as e:
            fixture_logger.warning(
                f"Unexpected error during health check: {e}"
            )  # Log other errors

        time.sleep(poll_interval)
    # --- End wait ---

    if not server_ready:
        # Ensure process is terminated if it never became ready
        if server_process.poll() is None:
            fixture_logger.error(
                f"llama-server did not become ready within {max_wait_time} seconds. Terminating process."
            )
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fixture_logger.warning("Server did not terminate gracefully, killing.")
                server_process.kill()
        # server_log_file.close() # <<< REMOVING FILE CLOSE
        pytest.fail(
            f"llama-server failed to start and become ready within {max_wait_time}s. Check pytest output for server logs."
        )

    yield base_url  # Yield the base URL for tests to use

    # Cleanup: Ensure the server process is terminated
    fixture_logger.info("Tearing down llama-server...")
    if server_process.poll() is None:  # Check if it's still running
        server_process.terminate()
        try:
            server_process.wait(timeout=10)  # Wait for graceful termination
            fixture_logger.info("llama-server terminated gracefully.")
        except subprocess.TimeoutExpired:
            fixture_logger.warning(
                "llama-server did not terminate gracefully after 10s, killing."
            )
            server_process.kill()
            server_process.wait()  # Wait for kill
            fixture_logger.info("llama-server killed.")
    else:
        fixture_logger.info(
            f"llama-server already terminated with code {server_process.returncode}."
        )

    # server_log_file.close() # <<< REMOVING FILE CLOSE
    # fixture_logger.info("llama-server log file closed.") # <<< REMOVING LOG MSG


# --- Other Fixtures (Mocked, keep for tests not needing real LLM) ---


@pytest.fixture
def mock_llm_url():
    """Fixture para fornecer uma URL mock para o LLM."""
    # Use a valid loopback address, potentially the one used by the test server
    # return "http://mock-llm-errors/v1/chat/completions" # Original problematic URL
    return f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/chat/completions"  # Use constants from top


@pytest.fixture
def LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE() -> str:
    # <<< REFORMAT TO ReAct TEXT FORMAT >>>
    return """
Thought: User wants to execute risky code that will likely fail. I need to use the execute_code tool.
Action: execute_code
Action Input: {
  "code": "print(1/0)",
  "language": "python"
}
"""
    # Original JSON string:
    # return '{"thought": "User wants to execute risky code.", "Action": "execute_code", "action_input": {"code": "print(1/0)", "language": "python"}}'
