from __future__ import annotations
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

class ServerManager:
    """Manages the lifecycle of external server processes (LLM, SD, etc.)."""

    def __init__(self):
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._log_files: Dict[str, object] = {} # Store file handles

    async def start_server(self, server_name: str) -> bool:
        """Starts the specified server process and waits for it to become healthy."""
        if server_name not in SERVER_CONFIGS:
            logger.error(f"Unknown server name: '{server_name}'. Cannot start.")
            return False
            
        config = SERVER_CONFIGS[server_name]
        health_endpoint = config.get("health_endpoint")
        startup_timeout = config.get("startup_timeout", 60)

        # --- Check health BEFORE attempting to start --- 
        if health_endpoint:
            logger.debug(f"Checking initial health of '{server_name}' at {health_endpoint}...")
            is_already_healthy = await self._check_health_endpoint(health_endpoint)
            if is_already_healthy:
                logger.info(f"Server '{server_name}' is already running and healthy.")
                # Ensure it's not in our managed list if we didn't start it
                if server_name in self._processes and self._processes[server_name].returncode is not None:
                     del self._processes[server_name]
                return True
            else:
                 logger.info(f"Server '{server_name}' not detected or not healthy. Proceeding with start attempt...")
        # --- End health check ---

        # Check if we are already managing a running process for this server
        if server_name in self._processes and self._processes[server_name].returncode is None:
            logger.info(f"Server '{server_name}' was already started by this manager (PID: {self._processes[server_name].pid}). Verifying health again...")
            return await self.check_server_status(server_name) # Re-check status just in case

        # --- Proceed with starting the process --- 
        binary = config["binary"]
        args = config["args"]
        cwd = config.get("cwd", None)
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

    async def _check_health_endpoint(self, url: str, timeout: float = 2.0) -> bool:
        """Helper function to check a single health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                return response.is_success
        except httpx.RequestError:
            return False
        except Exception as e:
            logger.error(f"Unexpected error during health check to {url}: {e}")
            return False

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

async def stop_all_servers(manager: 'ServerManager'):
    """Stops all servers managed by the provided ServerManager instance."""
    logger.info("Stopping all managed servers via ServerManager instance...")
    await manager.stop_all_servers()
    logger.info("All managed servers stopped (via ServerManager instance).")

async def start_all_servers() -> Optional[ServerManager]:
    """Initializes the ServerManager and starts all configured servers.

    Returns:
        The ServerManager instance if all essential servers started successfully,
        otherwise None.
    """
    logger.info("Initializing ServerManager and starting all configured servers...")
    manager = ServerManager()
    all_successful = True
    
    # Start essential servers (e.g., llama) first
    essential_servers = ['llama'] # Define which servers are critical
    for server_name in essential_servers:
        if server_name in SERVER_CONFIGS:
            success = await manager.start_server(server_name)
            if not success:
                logger.critical(f"Essential server '{server_name}' failed to start. Aborting server startup.")
                # Attempt to stop any servers that might have started before the failure
                await manager.stop_all_servers() 
                return None # Indicate critical failure
        else:
             logger.warning(f"Essential server '{server_name}' is not defined in SERVER_CONFIGS. Skipping.")
             # Consider this a failure? Or allow proceeding without it?
             # For now, let's treat missing essential config as failure.
             all_successful = False 
             break # Stop trying to start others

    if not all_successful: # If an essential server config was missing
         await manager.stop_all_servers()
         return None

    # Start other non-essential servers
    for server_name in SERVER_CONFIGS:
        if server_name not in essential_servers:
            success = await manager.start_server(server_name)
            if not success:
                logger.warning(f"Non-essential server '{server_name}' failed to start. Continuing...")
                # Do not set all_successful to False, allow continuing

    logger.info("Server startup process completed.")
    return manager # Return the manager instance

# Example usage (for testing this module directly)
# <<< REFACTORED _main_test >>>
async def _main_test():
    """Main function for testing ServerManager startup and shutdown using context managers."""
    print("--- Testing ServerManager --- ")
    manager_instance = ServerManager()
    
    servers_to_test = ["llama", "sd"] # Add other server names if needed
    started_servers = []
    server_tasks = []

    try:
        print("Attempting to start servers concurrently...")
        # Start servers concurrently and store tasks
        for server_name in servers_to_test:
            print(f"Creating start task for {server_name}...")
            # Use the context manager for each server
            cm = manager_instance.managed_server(server_name)
            server_tasks.append(asyncio.create_task(cm.__aenter__(), name=f"start_{server_name}"))
        
        # Wait for all start tasks to complete
        done, pending = await asyncio.wait(server_tasks, return_when=asyncio.ALL_COMPLETED)
        
        for task in done:
            server_name = task.get_name().split('_')[-1] # Extract name from task name
            try:
                result = task.result() # Raises exception if task failed
                print(f"{server_name} server started successfully via manager.")
                started_servers.append(server_name)
            except Exception as e:
                print(f"Error starting server {server_name}: {type(e).__name__} - {e}")
                # Cleanup for this specific server should be handled by its context manager's __aexit__ if entered

        if not started_servers:
            print("No servers were successfully started.")
            return # Exit test early

        # If at least one server started, check status and wait
        print(f"Servers ({', '.join(started_servers)}) reported as started. Checking status...")
        await asyncio.sleep(2) # Wait a moment
        for server_name in started_servers:
            status = await manager_instance.check_server_status(server_name)
            print(f" -> Status check for {server_name}: {'Running/Healthy' if status else 'Stopped/Unhealthy'}")
                 
        print("Test running. Press Ctrl+C to stop managed servers.")
        # Keep running until interrupted
        await asyncio.Future() 

    except asyncio.CancelledError:
        print("\nShutdown signal received during test.")
        # Cancellation will propagate to tasks; cleanup handled by finally
    finally:
        print("--- Test finished or interrupted. Initiating final cleanup via manager... ---")
        # Context managers (__aexit__) should be called for any successfully started servers 
        # when their corresponding start task is cancelled or finishes.
        # Calling stop_all_servers on the manager instance ensures cleanup even if 
        # a start task failed before __aenter__ completed fully or if CM exit failed.
        await manager_instance.stop_all_servers() 
        print("--- Server manager test finished. ---")

# <<< REFACTORED __main__ block >>>
if __name__ == "__main__":
    # Create the manager instance here so it's accessible in finally block
    manager_instance = ServerManager() 
    loop = asyncio.get_event_loop()
    # Pass manager to the test function if it needs it (currently doesn't directly)
    main_task = loop.create_task(_main_test())
    
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in main loop, cancelling tasks...")
        # Cancel the main task
        if main_task and not main_task.done():
            main_task.cancel()
            # Give cancellation a moment to propagate
            # loop.run_until_complete(asyncio.sleep(0.1))
        # Wait for the task to actually finish cancelling
        # Suppress CancelledError if it propagates here
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            print("Main task cancelled.")
            
    finally:
        # --- Ensure Final Cleanup --- 
        # The _main_test finally block should handle stopping servers managed within it.
        # However, if KeyboardInterrupt happens *before* _main_test completes its 
        # finally block, or if the manager instance needs broader scope cleanup,
        # we might need an explicit stop here. 
        # For now, rely on _main_test's finally block, but keep this structure.
        
        # Retrieve potentially running processes directly from manager if needed,
        # although the _main_test finally block should handle this.
        # if manager and manager._processes:
        #     print("Running final server cleanup from main block...")
        #     loop.run_until_complete(manager.stop_all_servers())
        
        print("Closing event loop...")
        loop.close()
        print("Event loop closed.") 

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