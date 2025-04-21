import asyncio
import logging
import os
import signal
from typing import Optional, List

logger = logging.getLogger(__name__)

class LLMServerManager:
    """Manages the lifecycle of a local LLM server process."""

    def __init__(self, 
                 server_command: List[str], 
                 host: str = "localhost", 
                 port: int = 8080,
                 startup_timeout: int = 15, # Seconds to wait for server startup check
                 shutdown_timeout: int = 10): # Seconds to wait for graceful shutdown
        """
        Initializes the server manager.

        Args:
            server_command: The command and arguments to execute for starting the server 
                            (e.g., ["/path/to/llama.cpp/server", "-m", "model.gguf", "-c", "4096"]).
            host: The hostname the server should listen on.
            port: The port the server should listen on.
            startup_timeout: How long to wait checking if the server started successfully.
            shutdown_timeout: How long to wait for graceful shutdown before killing.
        """
        if not server_command:
            raise ValueError("server_command cannot be empty.")
            
        self.server_command = server_command
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.server_process: Optional[asyncio.subprocess.Process] = None
        self._server_started_by_us = False # Flag to track if *this* instance started it

        logger.info(f"LLMServerManager initialized for {host}:{port}.")
        logger.info(f"Server command: {' '.join(server_command)}") # Log the command for debugging

    async def _is_server_running(self) -> bool:
        """Checks if a server is already listening on the host and port."""
        try:
            # Try to open a connection - success means something is listening
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), 
                timeout=1.0 # Short timeout for check
            )
            writer.close()
            await writer.wait_closed()
            logger.debug(f"Connection check: Successfully connected to {self.host}:{self.port}. Server seems running.")
            return True
        except ConnectionRefusedError:
            logger.debug(f"Connection check: Connection refused on {self.host}:{self.port}. Server seems down.")
            return False
        except asyncio.TimeoutError:
             logger.warning(f"Connection check: Timeout when trying to connect to {self.host}:{self.port}.")
             return False # Treat timeout as potentially down or unresponsive
        except Exception as e:
            logger.error(f"Connection check: Error checking server status on {self.host}:{self.port}: {e}")
            return False # Assume not running on error

    async def start_server(self):
        """Starts the LLM server process if it's not already running."""
        if self.server_process and self.server_process.returncode is None:
            logger.info("Server process already managed by this instance and running.")
            return

        if await self._is_server_running():
            logger.warning(f"Server already running on {self.host}:{self.port} (possibly started externally). Manager will not start a new one.")
            self._server_started_by_us = False
            return

        logger.info(f"Attempting to start LLM server with command: {' '.join(self.server_command)}")
        try:
            # Start the server process
            self.server_process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdout=asyncio.subprocess.PIPE, # Capture stdout
                stderr=asyncio.subprocess.PIPE  # Capture stderr
                # Consider adding cwd=... if the command needs a specific working directory
            )
            self._server_started_by_us = True
            logger.info(f"Server process started with PID: {self.server_process.pid}")

            # Optionally, wait a bit and check if it actually started listening
            await asyncio.sleep(2) # Give it a moment to bind
            
            # Monitor startup by checking the port or reading stderr/stdout
            try:
                 logger.info(f"Waiting up to {self.startup_timeout}s for server to become responsive...")
                 await asyncio.wait_for(self._wait_for_server_responsive(), timeout=self.startup_timeout)
                 logger.info(f"Server on {self.host}:{self.port} is responsive.")
            except asyncio.TimeoutError:
                 logger.error(f"Server did not become responsive on {self.host}:{self.port} within {self.startup_timeout}s.")
                 await self.stop_server() # Attempt to clean up the potentially failed process
                 raise RuntimeError("LLM Server failed to start and become responsive.")
            except Exception as start_err:
                 logger.error(f"Error during server startup monitoring: {start_err}")
                 await self.stop_server() # Attempt cleanup
                 raise
                 
        except FileNotFoundError:
             logger.error(f"Error starting server: Command not found '{self.server_command[0]}'. Is it in PATH or is the path correct?")
             self.server_process = None
             self._server_started_by_us = False
             raise
        except Exception as e:
            logger.error(f"Error starting server process: {e}", exc_info=True)
            self.server_process = None
            self._server_started_by_us = False
            raise
            
    async def _wait_for_server_responsive(self, interval=1.0):
        """Polls the server port until it becomes responsive."""
        while True:
            if await self._is_server_running():
                return
            # Check if process died unexpectedly
            if self.server_process and self.server_process.returncode is not None:
                 # Read stderr to get potential error message
                 stderr_output = await self._read_stream(self.server_process.stderr)
                 logger.error(f"Server process {self.server_process.pid} terminated unexpectedly during startup check. Stderr: {stderr_output}")
                 raise RuntimeError(f"LLM Server process died unexpectedly. Stderr: {stderr_output}")
            await asyncio.sleep(interval)

    async def _read_stream(self, stream: Optional[asyncio.StreamReader], limit=1024):
        """Helper to read from stdout/stderr streams non-blockingly."""
        if not stream:
            return ""
        try:
            data = await stream.read(limit)
            return data.decode(errors='ignore').strip()
        except Exception as e:
            logger.warning(f"Error reading stream: {e}")
            return "(Error reading stream)"

    async def stop_server(self):
        """Stops the managed LLM server process gracefully."""
        if not self.server_process or self.server_process.returncode is not None:
            logger.info("Server process not running or not managed by this instance.")
            return
        
        if not self._server_started_by_us:
             logger.info(f"Server on {self.host}:{self.port} was not started by this manager. Skipping shutdown.")
             return

        logger.info(f"Attempting to stop server process {self.server_process.pid}...")
        try:
            # Try graceful termination first (SIGTERM/SIGINT)
            # SIGINT (Ctrl+C) is often preferred for llama.cpp server
            logger.debug(f"Sending SIGINT to process {self.server_process.pid}")
            self.server_process.send_signal(signal.SIGINT)
            
            # Wait for termination
            try:
                await asyncio.wait_for(self.server_process.wait(), timeout=self.shutdown_timeout)
                logger.info(f"Server process {self.server_process.pid} terminated gracefully with code {self.server_process.returncode}.")
            except asyncio.TimeoutError:
                logger.warning(f"Server process {self.server_process.pid} did not terminate gracefully after {self.shutdown_timeout}s. Sending SIGKILL.")
                self.server_process.kill()
                await self.server_process.wait() # Wait for kill to complete
                logger.info(f"Server process {self.server_process.pid} killed.")
                
            # Read any remaining output
            stdout_output = await self._read_stream(self.server_process.stdout)
            stderr_output = await self._read_stream(self.server_process.stderr)
            if stdout_output: logger.debug(f"Final server stdout: {stdout_output}")
            if stderr_output: logger.debug(f"Final server stderr: {stderr_output}")

        except ProcessLookupError:
             logger.warning(f"Process {self.server_process.pid} not found during shutdown (already terminated?).")
        except Exception as e:
            logger.error(f"Error stopping server process {self.server_process.pid}: {e}", exc_info=True)
        finally:
            self.server_process = None
            self._server_started_by_us = False 