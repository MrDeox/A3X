import asyncio
import logging
import aiohttp
import os
import signal
import sys
import platform
from typing import Optional, Tuple

from a3x.core.config import (
    LLAMA_SERVER_BINARY, LLAMA_CPP_DIR, LLAMA_SERVER_ARGS, LLAMA_HEALTH_ENDPOINT,
    LLAMA_SERVER_STARTUP_TIMEOUT,
    SD_SERVER_MODULE, SD_API_CHECK_ENDPOINT, SD_SERVER_STARTUP_TIMEOUT,
    SERVER_CHECK_INTERVAL, SERVER_LOG_FILE, SD_WEBUI_DEFAULT_PATH_CONFIG
)

# Use a specific logger for the server manager
logger = logging.getLogger("A3XServerManager")
# Add the file handler configured in logging_config.py
# Note: This assumes setup_logging() in logging_config is called elsewhere first.
# If not, the handler might need to be created/added here.
# For simplicity, let's assume setup_logging handles adding the handler based on the logger name.
try:
    log_handler = logging.FileHandler(SERVER_LOG_FILE, mode='a')
    log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO) # Ensure logger level is set
    logger.propagate = False # Optional: Prevent logging to console via root logger
except Exception as e:
    print(f"[ServerManager WARN] Could not configure file logging for server manager: {e}")
    # Fallback to basic console logging if file setup fails
    if not logger.hasHandlers():
         logging.basicConfig() # Ensure basicConfig is called if no handlers exist
         logger.setLevel(logging.INFO)

# Dictionary to keep track of managed processes
managed_processes = {}

async def _check_server_ready(name: str, url: str, timeout: int) -> bool:
    """Checks if a server endpoint is responsive."""
    logger.info(f"Checking if {name} server is ready at {url} (timeout: {timeout}s)...")
    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                # Use HEAD request for health checks if possible, otherwise GET
                method = session.head if "/health" in url else session.get
                async with method(url, timeout=SERVER_CHECK_INTERVAL - 1) as response:
                    # Allow 200 OK or common redirect/not found for initial checks
                    if response.status in [200, 307, 404]: 
                        logger.info(f"{name} server responded (Status: {response.status}) at {url}. Considering ready.")
                        return True
                    else:
                        logger.debug(f"{name} check failed with status: {response.status}. Retrying...")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
            logger.debug(f"{name} connection refused or timed out at {url}. Retrying...")
        except Exception as e:
            logger.warning(f"{name} check at {url} encountered an error: {e}. Retrying...")

        elapsed_time = asyncio.get_event_loop().time() - start_time
        if elapsed_time > timeout:
            logger.error(f"{name} server did not become ready at {url} within the {timeout}s timeout.")
            return False

        await asyncio.sleep(SERVER_CHECK_INTERVAL)

async def _start_process(name: str, cmd_list: list, cwd: str, ready_url: str, ready_timeout: int) -> Optional[asyncio.subprocess.Process]:
    """Starts a process using asyncio.subprocess and checks if it becomes ready."""
    global managed_processes

    # Check if already running via API check first
    if await _check_server_ready(name, ready_url, timeout=2): # Quick check
        logger.info(f"{name} server appears to be already running at {ready_url}. Skipping start.")
        return None # Indicate already running

    logger.info(f"Starting {name} server...")
    logger.info(f"Command: {' '.join(cmd_list)}")
    logger.info(f"Working Directory: {cwd}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd_list,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Ensure SIGTERM is sent on parent exit on non-Windows
            preexec_fn=os.setsid if platform.system() != "Windows" else None 
        )
        logger.info(f"{name} server process started with PID: {process.pid}")
        managed_processes[name] = process

        # Start background tasks to log stdout/stderr
        asyncio.create_task(_log_stream(process.stdout, f"[{name} STDOUT]"))
        asyncio.create_task(_log_stream(process.stderr, f"[{name} STDERR]"))

        # Wait for the server to become ready
        if not await _check_server_ready(name, ready_url, ready_timeout):
            logger.error(f"Failed to start {name} server: API did not become ready.")
            await stop_server(name) # Attempt to clean up the failed process
            return None
        
        logger.info(f"{name} server started and ready.")
        return process

    except FileNotFoundError:
        logger.error(f"Failed to start {name} server: Command not found ({cmd_list[0]}). Ensure it's installed and in PATH or the path is correct.")
        return None
    except Exception as e:
        logger.exception(f"Failed to start {name} server process:")
        return None

async def _log_stream(stream: Optional[asyncio.StreamReader], prefix: str):
    """Logs lines from a stream asynchronously."""
    if not stream:
        return
    while True:
        try:
            line = await stream.readline()
            if not line:
                break # End of stream
            logger.info(f"{prefix} {line.decode(errors='ignore').strip()}")
        except Exception as e:
            logger.error(f"Error reading stream {prefix}: {e}")
            break

async def start_llama_server() -> Optional[asyncio.subprocess.Process]:
    """Starts the llama.cpp server."""
    if not os.path.exists(LLAMA_SERVER_BINARY):
        logger.error(f"llama-server binary not found at {LLAMA_SERVER_BINARY}. Cannot start server.")
        logger.error("Please ensure llama.cpp is compiled correctly.")
        return None
    if not os.path.exists(LLAMA_SERVER_MODEL_PATH):
        logger.error(f"LLM model file not found at {LLAMA_SERVER_MODEL_PATH}. Cannot start server.")
        logger.error("Please check the LLAMA_SERVER_MODEL_PATH in your configuration or download the model.")
        return None
        
    return await _start_process(
        name="LLaMA",
        cmd_list=[LLAMA_SERVER_BINARY] + LLAMA_SERVER_ARGS,
        cwd=LLAMA_CPP_DIR, # Run from llama.cpp directory
        ready_url=LLAMA_HEALTH_ENDPOINT,
        ready_timeout=LLAMA_SERVER_STARTUP_TIMEOUT
    )

async def start_sd_server() -> Optional[asyncio.subprocess.Process]:
    """Starts the SD API server (sd_api_server.py)."""
    # Command needs to run the module using the python executable from the *current* venv
    python_executable = sys.executable 
    cmd = [python_executable, "-m", SD_SERVER_MODULE, "--sd-webui-path", SD_WEBUI_DEFAULT_PATH_CONFIG]
    
    # Determine CWD (should be project root where config/logs are expected)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    return await _start_process(
        name="SD",
        cmd_list=cmd,
        cwd=project_root, # Run from project root
        ready_url=SD_API_CHECK_ENDPOINT,
        ready_timeout=SD_SERVER_STARTUP_TIMEOUT
    )

async def stop_server(name: str):
    """Stops a specific managed server process."""
    global managed_processes
    process = managed_processes.pop(name, None)
    if process and process.returncode is None:
        logger.info(f"Stopping {name} server (PID: {process.pid})...")
        try:
            # Send SIGTERM for graceful shutdown
            if platform.system() == "Windows":
                 # Sending Ctrl+C might be better for some Python scripts on Windows
                 process.send_signal(signal.CTRL_C_EVENT)
            else:
                 process.terminate() # SIGTERM on Unix-like
            
            # Wait for process to terminate
            await asyncio.wait_for(process.wait(), timeout=15.0)
            logger.info(f"{name} server process stopped gracefully (Return Code: {process.returncode}).")
        except asyncio.TimeoutError:
            logger.warning(f"{name} server did not stop gracefully after 15s. Killing...")
            process.kill()
            await process.wait() # Wait for kill to complete
            logger.info(f"{name} server process killed.")
        except Exception as e:
            logger.exception(f"Error stopping {name} server:")
            # Try killing if still running
            if process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                    logger.info(f"{name} server process force-killed after error.")
                except Exception as kill_e:
                    logger.error(f"Error during force-kill: {kill_e}")
    elif process:
         logger.info(f"{name} server (PID: {process.pid}) already stopped (Return Code: {process.returncode}).")
    else:
        logger.info(f"{name} server not found in managed processes.")

async def stop_all_servers():
    """Stops all managed server processes."""
    logger.info("Stopping all managed servers...")
    # Create a list of tasks to stop servers concurrently
    tasks = [stop_server(name) for name in list(managed_processes.keys())]
    if tasks:
        await asyncio.gather(*tasks)
    logger.info("All managed servers stopped.")

# Example usage (for testing this module directly)
async def _main_test():
    print("Testing server manager...")
    # Requires llama.cpp compiled and model downloaded
    # Requires stable-diffusion-webui cloned
    
    # Setup basic logging for test
    logging.basicConfig(level=logging.INFO)
    
    print("Starting LLaMA server...")
    llama_proc = await start_llama_server()
    print("Starting SD server...")
    sd_proc = await start_sd_server()

    if llama_proc or sd_proc:
        print("Servers started (or were already running). Press Ctrl+C to stop.")
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Main task cancelled.")
        finally:
            print("Stopping servers...")
            await stop_all_servers()
            print("Servers stopped.")
    else:
        print("No servers were started.")

if __name__ == "__main__":
    try:
        asyncio.run(_main_test())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        # Ensure cleanup is attempted even if main_test didn't finish setup
        asyncio.run(stop_all_servers()) 