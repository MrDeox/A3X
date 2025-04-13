import asyncio
import logging
import aiohttp
import os
import signal
import sys
import platform
from typing import Optional, Tuple
from pathlib import Path
import subprocess
import time
import psutil
import requests

import a3x.servers.sd_api_server 

from a3x.core.config import (
    LLAMA_SERVER_BINARY, LLAMA_CPP_DIR, LLAMA_SERVER_ARGS, LLAMA_HEALTH_ENDPOINT,
    LLAMA_SERVER_STARTUP_TIMEOUT, LLAMA_SERVER_MODEL_PATH,
    SD_SERVER_MODULE, SD_API_CHECK_ENDPOINT, SD_SERVER_STARTUP_TIMEOUT,
    SERVER_CHECK_INTERVAL, SERVER_LOG_FILE, SD_WEBUI_DEFAULT_PATH_CONFIG,
    PROJECT_ROOT
)

# Comment out the import to prevent error since LLMInterface class doesn't exist
# from .llm_interface import LLMInterface

# Use a specific logger for the server manager
logger = logging.getLogger("A3XServerManager")
# Add the file handler configured in logging_config.py
# Note: This assumes setup_logging() in logging_config is called elsewhere first.
# If not, the handler might need to be created/added here.
# For simplicity, let's assume setup_logging handles adding the handler based on the logger name.
try:
    # Attempt to remove existing handlers to avoid duplication if run multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    log_handler = logging.FileHandler(SERVER_LOG_FILE, mode='a')
    # Use a more detailed format
    log_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    # Set level to DEBUG for more detailed output
    logger.setLevel(logging.DEBUG) 
    logger.propagate = False # Optional: Prevent logging to console via root logger
    
    # Also add a console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO) # Keep console less verbose maybe?
    logger.addHandler(console_handler)
    
    logger.debug("Server Manager logging initialized (DEBUG to file, INFO to console).")
    
except Exception as e:
    print(f"[ServerManager WARN] Could not configure file logging for server manager: {e}")
    # Fallback to basic console logging if file setup fails
    if not logger.hasHandlers():
         logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s") # Basic console logging
         logger.setLevel(logging.INFO)

# Dictionary to keep track of managed processes
managed_processes = {}

async def _check_server_ready(name: str, url: str, timeout: int) -> bool:
    """Checks if a server endpoint is responsive."""
    logger.info(f"Checking readiness for {name} at {url} (timeout: {timeout}s)...")
    start_time = asyncio.get_event_loop().time()
    attempt = 0
    while True:
        attempt += 1
        elapsed_time = asyncio.get_event_loop().time() - start_time
        if elapsed_time > timeout:
            logger.error(f"{name} server did not become ready at {url} within the {timeout}s timeout after {attempt-1} attempts.")
            return False
            
        logger.debug(f"Attempt {attempt}: Checking {name} at {url}...")
        try:
            async with aiohttp.ClientSession() as session:
                # Use HEAD request for health checks if possible, otherwise GET
                method = session.head if "/health" in url else session.get
                async with method(url, timeout=SERVER_CHECK_INTERVAL - 1) as response:
                    # Allow 200 OK or common redirect/not found for initial checks
                    if response.status in [200]: # Be stricter for readiness
                        logger.info(f"{name} server responded OK (Status: {response.status}) at {url} after {elapsed_time:.1f}s. Ready.")
                        return True
                    # Log other statuses that might indicate it's running but not fully ready
                    elif response.status in [307, 404, 503]: 
                         logger.debug(f"{name} responded with status {response.status}. Not ready yet. Retrying...")
                    else:
                        logger.warning(f"{name} check failed with unexpected status: {response.status}. Retrying...")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
            logger.debug(f"{name} connection refused or timed out at {url}. Retrying...")
        except Exception as e:
            logger.warning(f"{name} check at {url} encountered an error: {e}. Retrying...")

        await asyncio.sleep(SERVER_CHECK_INTERVAL)

async def _start_process(name: str, cmd_list: list, cwd: str, ready_url: str, ready_timeout: int) -> Optional[asyncio.subprocess.Process]:
    """Starts a process using asyncio.subprocess and checks if it becomes ready."""
    global managed_processes

    logger.info(f"Preparing to start {name} server...")
    # Quick check if already running via API check first
    if await _check_server_ready(name, ready_url, timeout=3): # Quick check
        logger.info(f"{name} server appears to be already running at {ready_url}. Skipping start.")
        return None # Indicate already running

    logger.info(f"Executing start command for {name} server.")
    logger.info(f"--> Command: {' '.join(cmd_list)}")
    logger.info(f"--> Working Directory: {cwd}")

    try:
        # Redirect stderr to stdout to capture all output in one stream if desired,
        # or keep separate as before.
        process = await asyncio.create_subprocess_exec(
            *cmd_list,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, # Keep stderr separate
            # Ensure SIGTERM is sent on parent exit on non-Windows
            preexec_fn=os.setsid if platform.system() != "Windows" else None 
        )
        logger.info(f"{name} server process initiated with PID: {process.pid}")
        managed_processes[name] = process

        # Start background tasks to log stdout/stderr
        # Pass the specific logger instance
        asyncio.create_task(_log_stream(process.stdout, f"[{name} PID:{process.pid} STDOUT]", logger))
        asyncio.create_task(_log_stream(process.stderr, f"[{name} PID:{process.pid} STDERR]", logger))

        # Wait for the server to become ready
        is_ready = await _check_server_ready(name, ready_url, ready_timeout)
        
        if not is_ready:
            logger.error(f"Failed to start {name} server: API did not become ready.")
            await stop_server(name) # Attempt to clean up the failed process
            return None
        
        logger.info(f"--- {name} server started successfully and is ready. ---")
        return process

    except FileNotFoundError:
        logger.error(f"Failed to start {name} server: Command not found ({cmd_list[0]}). Ensure it's installed and in PATH or the path is correct.")
        return None
    except Exception as e:
        logger.exception(f"Unexpected exception during {name} server process startup:")
        return None

async def _log_stream(stream: Optional[asyncio.StreamReader], prefix: str, target_logger: logging.Logger):
    """Logs lines from a stream asynchronously using the provided logger."""
    if not stream:
        target_logger.warning(f"Stream for {prefix} is None, cannot log.")
        return
    target_logger.debug(f"Starting log stream reader for {prefix}")
    while True:
        try:
            line = await stream.readline()
            if not line:
                target_logger.debug(f"Stream {prefix} ended.")
                break # End of stream
            # Use INFO level for stream output to ensure visibility
            target_logger.info(f"{prefix} {line.decode(errors='ignore').strip()}") 
        except asyncio.CancelledError:
            target_logger.debug(f"Log stream reader for {prefix} cancelled.")
            break
        except Exception as e:
            target_logger.error(f"Error reading stream {prefix}: {e}")
            break # Exit loop on error
    target_logger.debug(f"Stopping log stream reader for {prefix}")

async def start_llama_server(model_path: str, port: int, host: str, gpu_layers: int, context_size: int, mmproj_path: Optional[str]) -> Optional[asyncio.subprocess.Process]:
    """Starts the llama.cpp server."""
    logger.info("Checking prerequisites for LLaMA server...")
    if not os.path.exists(LLAMA_SERVER_BINARY):
        logger.error(f"LLaMA server binary check FAILED: Not found at {LLAMA_SERVER_BINARY}.")
        logger.error("Please ensure llama.cpp is compiled correctly.")
        return None
    else:
        logger.debug(f"LLaMA server binary check OK: Found at {LLAMA_SERVER_BINARY}.")

    # <<< Resolve model_path to absolute path >>>
    # Assume model_path provided might be relative to project root
    absolute_model_path = os.path.abspath(os.path.join(PROJECT_ROOT, model_path))
    logger.debug(f"Resolved model path to absolute: {absolute_model_path}")
    
    if not os.path.exists(absolute_model_path):
        logger.error(f"LLaMA model file check FAILED: Absolute path not found at {absolute_model_path}.")
        logger.error(f"(Original path provided: {model_path})")
        return None
    else:
        logger.debug(f"LLaMA model file check OK: Absolute path found at {absolute_model_path}.")

    # <<< REVISED: Simplified and Robust Argument Construction >>>
    # Arguments provided to the function take highest priority
    final_args = {
        "-m": absolute_model_path, # <<< Use absolute path >>>
        "-c": str(context_size),
        "--host": host,
        "--port": str(port),
        "-ngl": str(gpu_layers),
    }

    # Add multimodal projector if provided and exists
    # <<< Resolve mmproj_path to absolute path if provided >>>
    absolute_mmproj_path = None
    if mmproj_path:
        absolute_mmproj_path = os.path.abspath(os.path.join(PROJECT_ROOT, mmproj_path))
        if os.path.exists(absolute_mmproj_path):
            final_args["--mmproj"] = absolute_mmproj_path
            logger.info(f"Adding multimodal projector (absolute path): {absolute_mmproj_path}")
        else:
             logger.warning(f"Multimodal projector file not found at absolute path {absolute_mmproj_path} (original: {mmproj_path}), ignoring.")

    # Parse default arguments from LLAMA_SERVER_ARGS
    # This simple parser assumes key-value pairs or flags.
    default_args = {}
    args_iter = iter(LLAMA_SERVER_ARGS)
    for arg in args_iter:
        if arg.startswith("--") or arg.startswith("-"):
            # Check if it looks like a flag or a key needing a value
            try:
                # Peek next element without consuming it
                next_val = next(args_iter)
                if next_val.startswith("--") or next_val.startswith("-"):
                    # Next item is another flag, so current is a flag
                    default_args[arg] = True
                    # Need to put the peeked value back - resetting iterator is easier here
                    args_iter = iter([next_val] + list(args_iter))
                else:
                    # Next item is likely the value for the current key
                    default_args[arg] = next_val
            except StopIteration:
                # Reached end, must be a flag
                default_args[arg] = True
        # else: # Ignore values that don't follow a key (shouldn't happen)
            # logger.warning(f"Ignoring standalone value in LLAMA_SERVER_ARGS: {arg}")
            
    logger.debug(f"Parsed default server args: {default_args}")

    # Add default arguments ONLY if not already specified by function params
    for key, value in default_args.items():
        if key not in final_args: # Prioritize function args
            # <<< Ensure default model/mmproj paths are also made absolute if they are paths >>>
            if key == "-m" or key == "--mmproj":
                abs_default_path = os.path.abspath(os.path.join(PROJECT_ROOT, value))
                if os.path.exists(abs_default_path):
                     final_args[key] = abs_default_path
                     logger.debug(f"Adding default arg (resolved path): {key}={abs_default_path}")
                else:
                    logger.warning(f"Ignoring default arg {key}: Path {value} (resolved to {abs_default_path}) not found.")
            else:
                final_args[key] = value
                logger.debug(f"Adding default arg: {key}={value}")
        else:
            logger.debug(f"Ignoring default arg {key} because it was provided by function call or already set.")

    # Build the final command list
    cmd = [LLAMA_SERVER_BINARY]
    for key, value in final_args.items():
        cmd.append(key)
        if value is not True: # Append value only if it's not a boolean flag
            cmd.append(str(value)) # Ensure value is string

    # <<< REVERTED: Use /health for readiness check >>>
    health_endpoint = f"http://{host}:{port}/health"
    logger.info(f"Will check LLaMA readiness using health endpoint: {health_endpoint}")

    return await _start_process(
        name="LLaMA",
        cmd_list=cmd,
        cwd=LLAMA_CPP_DIR, # Run from llama.cpp directory
        ready_url=health_endpoint, # <<< CHANGED BACK to /health >>>
        ready_timeout=LLAMA_SERVER_STARTUP_TIMEOUT
    )

async def start_sd_server() -> Optional[asyncio.subprocess.Process]:
    """Starts the SD API server (sd_api_server.py)."""
    # Command needs to run the module using the python executable from the *current* venv
    python_executable = sys.executable
    logger.info(f"Checking prerequisites for SD server using Python: {python_executable}") 
    
    # Verify the intermediate script exists
    try:
        sd_api_script_path = Path(a3x.servers.sd_api_server.__file__)
    except AttributeError:
         logger.error("SD API server script check FAILED: Could not determine path from imported module.")
         return None
         
    if not sd_api_script_path.exists():
         logger.error(f"SD API server script check FAILED: Not found at {sd_api_script_path}")
         return None
    else:
         logger.debug(f"SD API server script check OK: Found at {sd_api_script_path}")
         
    # Verify the WebUI path exists
    if not Path(SD_WEBUI_DEFAULT_PATH_CONFIG).is_dir():
         logger.error(f"SD WebUI path check FAILED: Directory not found at {SD_WEBUI_DEFAULT_PATH_CONFIG}")
         return None
    else:
         logger.debug(f"SD WebUI path check OK: Found at {SD_WEBUI_DEFAULT_PATH_CONFIG}")
         
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
            logger.debug(f"Waiting for {name} (PID: {process.pid}) process to terminate...")
            await asyncio.wait_for(process.wait(), timeout=15.0)
            logger.info(f"{name} server process stopped gracefully (Return Code: {process.returncode}).")
        except (asyncio.TimeoutError, RuntimeError) as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning(f"{name} server did not stop gracefully after 15s. Killing...")
            else:
                # Log the specific RuntimeError
                logger.error(f"Error stopping {name} server (likely loop issue): {e}")
                logger.warning(f"Proceeding to kill {name} server due to error.")
            
            # Force kill if timeout or RuntimeError occurs
            try:
                process.kill()
                # Add a small sleep before waiting again after kill
                await asyncio.sleep(0.1)
                await process.wait() # Wait for kill to complete
                logger.info(f"{name} server process killed (Return Code: {process.returncode}).")
            except ProcessLookupError:
                 logger.warning(f"{name} server process (PID: {process.pid}) was already dead before kill.")
            except Exception as kill_e:
                 # Log errors during the kill process itself
                 logger.exception(f"Error during force-kill of {name} (PID: {process.pid}):")
        except ProcessLookupError:
             logger.warning(f"{name} server process (PID: {process.pid}) was already dead before terminate/wait.")
        except Exception as e:
            logger.exception(f"Unexpected error during {name} server stop (PID: {process.pid}):")
    elif process:
        logger.info(f"{name} server process (PID: {process.pid}) already stopped (Return Code: {process.returncode}).")
    else:
        logger.debug(f"{name} server not found in managed processes or already stopped.")

async def stop_all_servers():
    """Stops all managed server processes."""
    logger.info("Stopping all managed servers...")
    # Create a list of tasks to stop servers concurrently
    tasks = [stop_server(name) for name in list(managed_processes.keys())]
    if tasks:
        await asyncio.gather(*tasks)
    logger.info("All managed servers stopped.")

async def start_all_servers():
    """Starts all configured servers (LLaMA and SD)."""
    global managed_processes
    logger.info("Starting LLaMA server...")
    llama_process = await _start_llama_server()
    if llama_process:
        managed_processes['llama'] = llama_process
        logger.info(f"LLaMA server started successfully (PID: {llama_process.pid}).")
    else:
        logger.error("LLaMA server failed to start or was already running.")

    logger.info("Starting SD server...")
    sd_process = await _start_sd_server()
    if sd_process:
        logger.info(f"SD server started successfully (PID: {sd_process.pid}).")
    else:
        logger.error("SD server failed to start or was already running.")

# Example usage (for testing this module directly)
async def _main_test():
    """Main function for testing server startup and shutdown."""
    print("Testing server manager...")
    
    print("Starting LLaMA server...")
    llama_proc = await start_llama_server()
    if llama_proc:
        print("LLaMA server started process.")
    else:
        print("LLaMA server failed to start or was already running.")
        # Decide if we should exit or continue if llama fails?
        # For now, continue to test SD server
    
    print("Starting SD server...")
    sd_proc = await start_sd_server()
    if sd_proc:
        print("SD server started process.")
    else:
        print("SD server failed to start or was already running.")

    # Keep running until interrupted (e.g., Ctrl+C)
    # In a real application, this might be event-driven or have other logic
    if managed_processes: # Check if any process was actually started
        print("Servers are starting/running. Press Ctrl+C to stop all managed servers.")
        try:
            # Wait indefinitely until cancelled
            await asyncio.Future() 
        except asyncio.CancelledError:
            print("\nShutdown signal received.")
        finally:
            print("Stopping all managed servers...")
            await stop_all_servers()
            print("Server manager test finished.")
    else:
        print("No servers were started by the manager.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(_main_test())
    
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in main loop, cancelling tasks...")
        main_task.cancel()
        # Ensure cleanup runs even with KeyboardInterrupt
        # loop.run_until_complete(main_task) # This might re-raise CancelledError
        # Run shutdown explicitly if needed
        if not main_task.done():
             # Give cancellation a moment to propagate
             loop.run_until_complete(asyncio.sleep(0.1))
        
        # Manually run stop_all_servers if main_task didn't finish cleanly
        if managed_processes: 
             print("Running final server cleanup...")
             loop.run_until_complete(stop_all_servers())
        
    finally:
        loop.close()
        print("Event loop closed.") 

class ServerManager:
    def __init__(self, host='127.0.0.1', port=8080, ngl=20):
        self.host = host
        self.port = port
        self.ngl = ngl
        self.server_process = None
        self.llm_url = f"http://{self.host}:{self.port}"
        logger.info(f"Initialized ServerManager with LLM URL: {self.llm_url} and ngl: {self.ngl}")

    def start_server(self):
        """Start the llama-server if it's not already running."""
        if self.is_server_running():
            logger.info("llama-server is already running.")
            return

        logger.info("Starting llama-server...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        llama_server_path = os.path.join(project_root, "llama.cpp", "build", "bin", "llama-server")
        model_path = os.path.join(project_root, "models", "gemma-3-4b-it-Q6_K.gguf")

        if not os.path.exists(llama_server_path):
            logger.error(f"llama-server not found at {llama_server_path}")
            raise FileNotFoundError(f"llama-server binary not found at {llama_server_path}")

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        cmd = [
            llama_server_path,
            "-m", model_path,
            "-ngl", str(self.ngl)
        ]

        logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(3)  # Give the server a moment to start

        if self.server_process.poll() is not None:
            error_output = self.server_process.stderr.read() if self.server_process.stderr else "No error output"
            logger.error(f"Failed to start llama-server: {error_output}")
            raise RuntimeError(f"Failed to start llama-server: {error_output}")

        max_retries = 10
        for attempt in range(max_retries):
            if self.is_server_running():
                logger.info("llama-server started successfully.")
                return
            time.sleep(1)

        logger.error("llama-server failed to start after multiple attempts.")
        raise RuntimeError("llama-server failed to start after multiple attempts.")

    def stop_server(self):
        """Stop the llama-server if it's running."""
        if self.server_process and self.is_server_running():
            logger.info("Stopping llama-server...")
            parent = psutil.Process(self.server_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            self.server_process.wait(timeout=5)
            logger.info("llama-server stopped.")
        else:
            logger.info("llama-server is not running.")
        self.server_process = None

    def is_server_running(self):
        """Check if the llama-server is running by querying its health endpoint."""
        try:
            response = requests.get(f"{self.llm_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            if self.server_process and self.server_process.poll() is None:
                return True
            return False

    def get_llm_interface(self):
        """Return an instance of a minimal LLM interface."""
        # Since LLMInterface class doesn't exist, return a minimal object or None
        return None  # Or define a minimal class or function call here if needed 