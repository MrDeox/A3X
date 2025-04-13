import asyncio
import logging
import aiohttp
import os
import signal
import sys
import platform
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import subprocess
import time
import psutil
import requests
import httpx

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

# Define server configurations in a structured way
SERVER_CONFIGS = {
    "llama": {
        "binary": LLAMA_SERVER_BINARY,
        "args": LLAMA_SERVER_ARGS, # This is already a list from config
        "health_endpoint": LLAMA_HEALTH_ENDPOINT,
        "startup_timeout": LLAMA_SERVER_STARTUP_TIMEOUT,
        "log_file": SERVER_LOG_FILE, # Log server stdout/stderr here
        "cwd": PROJECT_ROOT, # Run from project root usually
    },
    "stable_diffusion": {
        # Example config for SD server (adjust as needed)
        "binary": sys.executable, # Usually run as python module
        "args": ["-m", SD_SERVER_MODULE], # Example args
        "health_endpoint": SD_API_CHECK_ENDPOINT,
        "startup_timeout": SD_SERVER_STARTUP_TIMEOUT,
        "log_file": SERVER_LOG_FILE,
        "cwd": PROJECT_ROOT,
    }
    # Add other servers here
}

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
    """Manages the lifecycle of external server processes (LLM, SD, etc.)."""

    def __init__(self):
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._log_files: Dict[str, object] = {} # Store file handles

    async def start_server(self, server_name: str) -> bool:
        """Starts the specified server process and waits for it to become healthy."""
        if server_name in self._processes and self._processes[server_name].returncode is None:
            logger.info(f"Server '{server_name}' is already running (PID: {self._processes[server_name].pid}).")
            # Optionally check health again even if running
            return await self.check_server_status(server_name)

        if server_name not in SERVER_CONFIGS:
            logger.error(f"Unknown server name: '{server_name}'. Cannot start.")
            return False

        config = SERVER_CONFIGS[server_name]
        binary = config["binary"]
        args = config["args"]
        cwd = config.get("cwd", None)
        health_endpoint = config.get("health_endpoint")
        startup_timeout = config.get("startup_timeout", 60)
        log_file_path = config.get("log_file")

        if not os.path.exists(binary):
             logger.error(f"Server binary not found for '{server_name}' at path: {binary}")
             # Attempt to use default path if config one failed?
             # Example: binary = "/path/to/default/llama-server"
             # Or just return False
             return False

        command = [binary] + args
        logger.info(f"Starting server '{server_name}' with command: {' '.join(command)}")
        log_prefix = f"[{server_name.upper()}_SERVER]"

        log_handle = None
        if log_file_path:
            try:
                # Ensure log directory exists
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                log_handle = open(log_file_path, "a") # Append mode
                self._log_files[server_name] = log_handle
                stdout_redir = log_handle
                stderr_redir = log_handle
                logger.info(f"Redirecting {server_name} stdout/stderr to {log_file_path}")
            except Exception as e:
                logger.error(f"Failed to open log file {log_file_path} for {server_name}: {e}")
                stdout_redir = asyncio.subprocess.PIPE # Fallback to pipe
                stderr_redir = asyncio.subprocess.PIPE
        else:
            stdout_redir = asyncio.subprocess.PIPE
            stderr_redir = asyncio.subprocess.PIPE

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=stdout_redir,
                stderr=stderr_redir,
                cwd=cwd,
                # Potentially set environment variables if needed
                # env=os.environ.copy()
            )
            self._processes[server_name] = process
            logger.info(f"Server '{server_name}' process started with PID: {process.pid}")

        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}. Ensure the server binary path is correct and executable.")
            if log_handle: log_handle.close()
            if server_name in self._log_files: del self._log_files[server_name]
            return False
        except Exception as e:
            logger.exception(f"Failed to start server '{server_name}' process:")
            if log_handle: log_handle.close()
            if server_name in self._log_files: del self._log_files[server_name]
            return False

        # Wait for the server to become healthy
        if health_endpoint:
            start_time = time.monotonic()
            is_healthy = False
            logger.info(f"Waiting for '{server_name}' server to become healthy at {health_endpoint} (timeout: {startup_timeout}s)...")
            while time.monotonic() - start_time < startup_timeout:
                if process.returncode is not None:
                     logger.error(f"Server '{server_name}' process terminated prematurely with code {process.returncode}. Check logs.")
                     break # Exit health check loop

                try:
                    async with httpx.AsyncClient(timeout=5.0) as client: # Short timeout for health check
                        response = await client.get(health_endpoint)
                        if response.is_success:
                            logger.info(f"Server '{server_name}' is healthy (status code: {response.status_code}).")
                            is_healthy = True
                            break
                        else:
                             logger.debug(f"Health check for '{server_name}' failed: {response.status_code}. Retrying...")
                except httpx.RequestError as e:
                    logger.debug(f"Health check connection error for '{server_name}': {e}. Server likely not ready yet. Retrying...")
                
                await asyncio.sleep(2) # Wait before retrying health check

            if not is_healthy:
                logger.error(f"Server '{server_name}' did not become healthy within the {startup_timeout}s timeout.")
                await self.stop_server(server_name) # Attempt to clean up
                return False
        else:
            logger.warning(f"No health endpoint configured for server '{server_name}'. Assuming started successfully after a short delay.")
            await asyncio.sleep(5) # Arbitrary delay if no health check

        return True # Server started (and potentially healthy)

    async def stop_server(self, server_name: str) -> bool:
        """Stops the specified server process gracefully."""
        if server_name not in self._processes:
            logger.info(f"Server '{server_name}' not found in managed processes.")
            return True # Already stopped or never started by this manager

        process = self._processes[server_name]
        if process.returncode is not None:
            logger.info(f"Server '{server_name}' (PID: {process.pid}) already terminated with code {process.returncode}.")
        else:
            logger.info(f"Stopping server '{server_name}' (PID: {process.pid})...")
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10.0) # Wait for graceful termination
                logger.info(f"Server '{server_name}' terminated gracefully.")
            except asyncio.TimeoutError:
                logger.warning(f"Server '{server_name}' did not terminate gracefully after 10s. Sending KILL signal.")
                try:
                    process.kill()
                    await process.wait() # Ensure kill completes
                    logger.info(f"Server '{server_name}' killed.")
                except ProcessLookupError:
                     logger.warning(f"Server '{server_name}' process already gone when trying to kill.")
                except Exception as e:
                    logger.exception(f"Error killing server '{server_name}':")
                    # Process might be orphaned, but report error
            except ProcessLookupError:
                 logger.warning(f"Server '{server_name}' process already gone when trying to terminate.")
            except Exception as e:
                logger.exception(f"Error terminating server '{server_name}':")
                # Return False? Or assume it might be stopped?

        # Clean up
        del self._processes[server_name]
        if server_name in self._log_files:
             try:
                 self._log_files[server_name].close()
             except Exception as e:
                 logger.error(f"Error closing log file for {server_name}: {e}")
             del self._log_files[server_name]
             
        return True

    async def stop_all_servers(self):
        """Stops all managed server processes."""
        server_names = list(self._processes.keys()) # Get keys before iterating
        logger.info(f"Stopping all managed servers: {server_names}")
        for server_name in server_names:
            await self.stop_server(server_name)
        logger.info("All managed servers stopped.")

    async def check_server_status(self, server_name: str) -> bool:
        """Checks if the server process is running and optionally checks health endpoint."""
        if server_name not in self._processes:
             return False # Not managed or already stopped
        process = self._processes[server_name]
        if process.returncode is not None:
             logger.info(f"Server '{server_name}' process has terminated (code: {process.returncode}).")
             return False # Process has exited

        # Process is running, check health endpoint if configured
        config = SERVER_CONFIGS.get(server_name, {})
        health_endpoint = config.get("health_endpoint")
        if health_endpoint:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(health_endpoint)
                    is_healthy = response.is_success
                    logger.debug(f"Health check for running server '{server_name}': {'Healthy' if is_healthy else 'Unhealthy'} (Status: {response.status_code})")
                    return is_healthy
            except httpx.RequestError:
                logger.debug(f"Health check failed for running server '{server_name}'. Assuming unhealthy/not ready.")
                return False # Cannot connect, assume not healthy
        else:
            # No health endpoint, just check if process is running (which it is at this point)
            logger.debug(f"Server '{server_name}' process is running (PID: {process.pid}). No health endpoint to check.")
            return True

    # --- Context Manager for easy use in tests ---
    # <<< Corrected: Use @asynccontextmanager or return an object with __aenter__/__aexit__ >>>
    # Option 1: Using a helper class (as previously attempted, but correcting usage)
    # Define the context manager class *inside* the ServerManager or globally if preferred
    class _ServerContextManager:
        """Helper async context manager class."""
        def __init__(self, manager: 'ServerManager', name: str):
            self._manager = manager
            self._name = name
            self._started = False
            logger.debug(f"[CM Helper '{self._name}'] Initialized.")

        async def __aenter__(self):
            logger.debug(f"[CM Helper '{self._name}'] Entering async context...")
            self._started = await self._manager.start_server(self._name)
            if not self._started:
                raise RuntimeError(f"Failed to start managed server '{self._name}'")
            logger.debug(f"[CM Helper '{self._name}'] Server started, returning self.")
            return self # Return the context manager instance itself

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            logger.debug(f"[CM Helper '{self._name}'] Exiting async context (Exc type: {exc_type})...")
            if self._started: # Only try to stop if __aenter__ succeeded
                logger.debug(f"[CM Helper '{self._name}'] Stopping server...")
                await self._manager.stop_server(self._name)
                logger.debug(f"[CM Helper '{self._name}'] Server stop initiated.")
            else:
                 logger.debug(f"[CM Helper '{self._name}'] Server was not started by __aenter__, skipping stop.")
            # Return False to propagate exceptions, True to suppress
            return False 

    def managed_server(self, server_name: str):
        """Returns an async context manager instance to start/stop a server."""
        logger.debug(f"Creating context manager for server '{server_name}'.")
        # Return an *instance* of the helper class
        return ServerManager._ServerContextManager(self, server_name)
        
    # Option 2: Using @asynccontextmanager (simpler if logic fits)
    # from contextlib import asynccontextmanager
    # @asynccontextmanager
    # async def managed_server_alt(self, server_name: str):
    #     logger.debug(f"[CM Decorator] Entering context for server '{server_name}'")
    #     started = False
    #     try:
    #         started = await self.start_server(server_name)
    #         if not started:
    #             raise RuntimeError(f"Failed to start managed server '{server_name}'")
    #         yield # Yield control to the 'with' block
    #     finally:
    #         logger.debug(f"[CM Decorator] Exiting context for server '{server_name}'")
    #         if started:
    #             await self.stop_server(server_name)

# Example usage (for testing ServerManager itself):
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    manager = ServerManager()
    
    try:
        logger.info("Starting LLAMA server via manager...")
        started = await manager.start_server("llama")
        if started:
            logger.info("LLAMA server started successfully.")
            await asyncio.sleep(5) # Keep it running for a bit
            status = await manager.check_server_status("llama")
            logger.info(f"LLAMA server status check: {'Running/Healthy' if status else 'Stopped/Unhealthy'}")
        else:
            logger.error("Failed to start LLAMA server.")
            
    finally:
        logger.info("Stopping LLAMA server...")
        await manager.stop_server("llama")
        logger.info("LLAMA server stop requested.")

    # Example using context manager
    # try:
    #     logger.info("Testing context manager...")
    #     async with manager.managed_server("llama"):
    #          logger.info("Inside context manager - server should be running.")
    #          await asyncio.sleep(3)
    #          status = await manager.check_server_status("llama")
    #          logger.info(f"Status inside CM: {status}")
    #     logger.info("Exited context manager - server should be stopped.")
    #     await asyncio.sleep(1)
    #     status = await manager.check_server_status("llama")
    #     logger.info(f"Status after CM: {status}")
    # except Exception as e:
    #      logger.error(f"Error during context manager test: {e}")


if __name__ == "__main__":
    # This is mainly for direct testing of the server manager
    # asyncio.run(main())
    pass 