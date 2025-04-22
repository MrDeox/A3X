import logging
import subprocess
import signal
import sys
import os
import asyncio
import aiohttp
import platform
import argparse
from typing import Optional
from pathlib import Path
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration --- 
SD_WEBUI_API_CHECK_URL = "http://127.0.0.1:7860/docs" # Default API endpoint for checks
SD_WEBUI_API_READY_TIMEOUT = 180 # seconds to wait for API to become ready
CHECK_INTERVAL_SECONDS = 5
STARTUP_TIMEOUT_SECONDS = 120 # Give it 2 minutes to start

SD_WEBUI_DEFAULT_PATH = "./stable-diffusion-webui" # Default relative path

# --- Global variable for the process --- 
webui_process = None

async def check_api_ready(url: str, timeout: int = SD_WEBUI_API_READY_TIMEOUT) -> bool:
    """Checks if the SD Web UI API endpoint is responsive."""
    logger.info(f"Checking if SD Web UI API is ready at {url} (timeout: {timeout}s)...")
    start_time = asyncio.get_event_loop().time()
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.error(f"API readiness check timed out after {timeout}s.")
            return False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5) # Short timeout for each check
                if response.status_code == 200:
                    logger.info(f"SD Web UI API is ready! (Status: {response.status_code})")
                    return True
                else:
                    logger.debug(f"API not ready yet (Status: {response.status_code}). Retrying...")
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.debug(f"API check connection error: {e}. Retrying...")
        except Exception as e:
             logger.error(f"Unexpected error during API check: {e}. Retrying...")

        await asyncio.sleep(5) # Wait before next check

def start_webui(sd_webui_path: str, model_arg: Optional[str], cpu_limit_percent: Optional[int] = None) -> Optional[subprocess.Popen]:
    """
    Starts the Stable Diffusion Web UI process using webui.sh.

    Args:
        sd_webui_path: Path to the stable-diffusion-webui directory.
        model_arg: Optional path to a specific model checkpoint file.
        cpu_limit_percent: Optional CPU usage limit (percentage). If set, uses
                           taskset and cpulimit to restrict the process.

    Returns:
        The subprocess.Popen object if successful, None otherwise.
    """
    global webui_process
    script_path = Path(sd_webui_path) / "webui.sh"
    launch_script_path = Path(sd_webui_path) / "launch.py"
    python_executable = sys.executable # Use the same python that runs this script

    if not script_path.exists():
        logger.error(f"webui.sh not found at: {script_path}")
        return None
    if not launch_script_path.exists():
        logger.error(f"launch.py not found at: {launch_script_path}")
        return None

    # Construct the base command arguments for launch.py or webui.sh
    # Note: We'll primarily modify COMMANDLINE_ARGS env var now
    command_base_args = ["--api"] # Always add --api flag
    if model_arg:
        # This might be better handled by setting override_settings in the API call
        # rather than a startup argument, but kept for potential direct use.
        # command_base_args.extend(["--ckpt", model_arg])
        logger.warning(f"Model argument ({model_arg}) provided, but it's recommended to set model via API payload.")
        # Instead, let's ensure the base ckpt-dir is set if models dir exists
        # <<< CHANGED: Hardcode models directory path >>>
        models_dir_absolute = "/home/arthur/projects/A3X/models"
        if not Path(models_dir_absolute).is_dir():
             logger.warning(f"Models directory not found at {models_dir_absolute}, --ckpt-dir will not be set.")
             models_dir_absolute = None # Ensure it's None if not found
    else:
         # <<< CHANGED: Hardcode models directory path >>>
         models_dir_absolute = "/home/arthur/projects/A3X/models"
         if not Path(models_dir_absolute).is_dir():
             logger.warning(f"Models directory not found at {models_dir_absolute}, --ckpt-dir will not be set.")
             models_dir_absolute = None


    logger.info(f"Starting Stable Diffusion Web UI from: {sd_webui_path}")

    try:
        # Make a copy of the current environment
        env = os.environ.copy()

        # Ensure COMMANDLINE_ARGS exists and initialize if not
        if 'COMMANDLINE_ARGS' not in env:
            env['COMMANDLINE_ARGS'] = ""

        # Add essential flags to COMMANDLINE_ARGS for stability/compatibility
        # These are added here to ensure they are present regardless of webui-user.sh defaults
        # Note: webui.sh might add its own args too.

        # Add --skip-torch-cuda-test flag (often needed if GPU check fails spuriously)
        if '--skip-torch-cuda-test' not in env['COMMANDLINE_ARGS']:
            env['COMMANDLINE_ARGS'] = f"{env['COMMANDLINE_ARGS']} --skip-torch-cuda-test".strip()

        # <<< ADDED: Skip loading model at start to prevent default download >>>
        if '--skip-load-model-at-start' not in env['COMMANDLINE_ARGS']:
            env['COMMANDLINE_ARGS'] = f"{env['COMMANDLINE_ARGS']} --skip-load-model-at-start".strip()

        # Add CPU flags like --no-half if needed (example)
        if '--no-half' not in env['COMMANDLINE_ARGS']:
             env['COMMANDLINE_ARGS'] = f"{env['COMMANDLINE_ARGS']} --no-half".strip()

        # Add checkpoint directory if found
        ckpt_dir_arg = None
        if models_dir_absolute:
            ckpt_dir_arg = f"--ckpt-dir \\\"{models_dir_absolute}\\\"".strip() # Quote the path for shell
            if '--ckpt-dir' not in env['COMMANDLINE_ARGS']:
                 env['COMMANDLINE_ARGS'] = f"{env['COMMANDLINE_ARGS']} {ckpt_dir_arg}".strip()

        # <<< ADDED: Explicitly set --ckpt to the desired model >>>
        # This should prevent the default download behavior
        desired_model_name = "anything-v3-fp16-pruned.safetensors"
        desired_model_path = None
        if models_dir_absolute:
            potential_path = Path(models_dir_absolute) / desired_model_name
            if potential_path.exists():
                desired_model_path = str(potential_path.resolve())
                logger.info(f"Found desired model at: {desired_model_path}. Adding --ckpt argument.")
                ckpt_arg = f"--ckpt \\\"{desired_model_path}\\\"".strip() # Quote the path
                if '--ckpt' not in env['COMMANDLINE_ARGS']:
                     env['COMMANDLINE_ARGS'] = f"{env['COMMANDLINE_ARGS']} {ckpt_arg}".strip()
            else:
                logger.warning(f"Desired model '{desired_model_name}' not found in {models_dir_absolute}. Cannot add --ckpt argument.")

        # Remove potential duplicate flags (basic cleanup)
        # Need to improve this logic to handle quoted paths and potentially multiple args
        current_args_str = env['COMMANDLINE_ARGS']
        parts = []
        # Basic split respecting quotes (won't handle nested or escaped quotes perfectly)
        # >>> Placeholder for better arg parsing if needed <<<
        # For now, rely on simple splitting and set logic later if used in limited command
        # logger.warning("Argument cleanup logic is basic and might not handle all duplicates perfectly.")
        # env['COMMANDLINE_ARGS'] = ' '.join(current_args_str.split()) # Simple whitespace normalization


        logger.info(f"Effective COMMANDLINE_ARGS for WebUI: {env['COMMANDLINE_ARGS']}")

        shell_command_str = ""
        if cpu_limit_percent is not None and cpu_limit_percent > 0:
            # --- CPU Limiting Logic ---
            logger.info(f"Applying CPU limit: {cpu_limit_percent}% using taskset (cores 0,1) and cpulimit.")
            # Build the command to run launch.py directly via taskset and cpulimit
            # Combine base args and env args
            all_args = env.get('COMMANDLINE_ARGS', '').split() + command_base_args
            # Remove duplicates (simple approach)
            all_args = sorted(list(set(all_args)))
            args_str = ' '.join(all_args)

            # Construct the full shell command string
            # Use exec to replace the shell process with cpulimit
            shell_command_str = (
                f"exec taskset -c 0,1 cpulimit -l {cpu_limit_percent} -- "
                f"{python_executable} {launch_script_path} {args_str} 2>&1"
            )
            logger.info(f"Executing limited command via bash: {shell_command_str}")

        else:
            # --- Default Logic (no CPU limit) ---
            logger.info("No CPU limit applied. Starting WebUI via webui.sh.")
            # Construct the command to execute webui.sh within bash
            # Use exec to replace the shell process with the webui.sh process
            # webui.sh internally handles COMMANDLINE_ARGS from env
            webui_command_str = f"{script_path}" # Base command is just the script path
            shell_command_str = f"exec {webui_command_str} 2>&1" # Redirect stderr to stdout
            logger.info(f"Executing default command via bash: {shell_command_str}")


        # --- Execute the selected command ---
        webui_process = subprocess.Popen(
            ["/bin/bash", "-c", shell_command_str], # Pass bash -c and the constructed command string
            cwd=sd_webui_path, # Run from the webui directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Still capture Popen errors (e.g., bash not found)
            text=True,
            env=env, # Pass modified environment (contains COMMANDLINE_ARGS)
        )
        logger.info(f"SD Web UI process initiated with PID: {webui_process.pid}")
        return webui_process

    except FileNotFoundError as e:
         logger.error(f"Error starting process: {e}. Ensure bash, taskset, cpulimit (if used), and python are in PATH.")
         return None
    except Exception as e:
        logger.exception("Failed to start SD Web UI process:")
        # Clean up global state if process object exists but failed somehow
        if webui_process:
             webui_process = None
        return None # Return None on failure

def handle_signal(sig, frame):
    """Gracefully shuts down the Web UI process upon receiving a signal."""
    logger.info(f"Received signal {sig}. Shutting down SD Web UI process...")
    global webui_process
    if webui_process and webui_process.poll() is None: # Check if process exists and is running
        logger.info(f"Terminating SD Web UI process (PID: {webui_process.pid})...")
        try:
            # Send SIGTERM first for graceful shutdown
            if platform.system() == "Windows":
                 # SIGTERM might not work well for shell scripts wrapping python on Win
                 # Sending Ctrl+C might be better, but Popen doesn't directly support it easily across platforms
                 # os.kill(webui_process.pid, signal.CTRL_C_EVENT) # This needs process_group=True potentially
                 logger.warning("Attempting SIGTERM on Windows, may require manual process kill if ineffective.")
                 webui_process.terminate() # Try SIGTERM
            else:
                 # On Linux, send SIGTERM to the process group to terminate shell and children
                 os.killpg(os.getpgid(webui_process.pid), signal.SIGTERM)

            # Wait a bit for graceful exit
            webui_process.wait(timeout=15) # Increased timeout
            logger.info("SD Web UI process terminated gracefully (or appeared to).")
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out. Forcing termination (SIGKILL)...")
            if platform.system() != "Windows":
                 try:
                      os.killpg(os.getpgid(webui_process.pid), signal.SIGKILL)
                 except ProcessLookupError:
                      logger.info("Process group already gone.") # Process might have died during timeout
                 except Exception as kill_e:
                      logger.error(f"Error sending SIGKILL to process group: {kill_e}, trying direct kill.")
                      webui_process.kill() # Fallback to direct kill
            else:
                 webui_process.kill() # Force kill if terminate didn't work on Windows

            try:
                 webui_process.wait(timeout=5) # Wait for kill confirmation
                 logger.info("SD Web UI process killed.")
            except subprocess.TimeoutExpired:
                 logger.error("Process did not terminate even after SIGKILL.")
            except Exception as e:
                 logger.error(f"Error waiting for process kill: {e}")

        except Exception as e:
            logger.error(f"Error during SD Web UI process termination: {e}")
            # Still try to kill if termination failed badly
            try:
                 if webui_process.poll() is None:
                      if platform.system() != "Windows":
                           os.killpg(os.getpgid(webui_process.pid), signal.SIGKILL)
                      else:
                           webui_process.kill()
                      webui_process.wait(timeout=5)
                      logger.info("SD Web UI process force-killed after error.")
            except Exception as kill_e:
                 logger.error(f"Error during force-kill: {kill_e}")
    else:
        logger.info("SD Web UI process already terminated or not started.")

    # Exit this script gracefully
    sys.exit(0)

async def monitor_process(proc: subprocess.Popen):
    """Monitors the stdout/stderr of the process and logs it."""
    # Use asyncio streams for non-blocking reads
    stdout_reader = asyncio.StreamReader()
    stderr_reader = asyncio.StreamReader()
    stdout_protocol = asyncio.StreamReaderProtocol(stdout_reader)
    stderr_protocol = asyncio.StreamReaderProtocol(stderr_reader)

    loop = asyncio.get_event_loop()

    try:
        # Connect pipes to readers
        if proc.stdout:
            await loop.connect_read_pipe(lambda: stdout_protocol, proc.stdout)
        if proc.stderr:
            await loop.connect_read_pipe(lambda: stderr_protocol, proc.stderr)

        async def log_stream(stream_reader: asyncio.StreamReader, prefix: str):
            """Reads and logs lines from a stream reader."""
            while not stream_reader.at_eof():
                try:
                    line = await stream_reader.readline()
                    if line:
                         # Log directly using the logger configured at the top
                         logger.info(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
                    else:
                        break # End of stream
                except asyncio.CancelledError:
                    logger.info(f"Log stream reader for {prefix} cancelled.")
                    break
                except Exception as e:
                    logger.exception(f"Error reading stream {prefix}: {e}")
                    break
            logger.info(f"Log stream reader for {prefix} finished.")

        # Run readers concurrently
        log_tasks = []
        if proc.stdout:
            log_tasks.append(asyncio.create_task(log_stream(stdout_reader, "SD WebUI STDOUT")))
        if proc.stderr:
            log_tasks.append(asyncio.create_task(log_stream(stderr_reader, "SD WebUI STDERR")))


        # Wait for the process to exit OR for logging tasks to finish (unexpectedly)
        process_wait_task = loop.run_in_executor(None, proc.wait)

        done, pending = await asyncio.wait(
             log_tasks + [process_wait_task],
             return_when=asyncio.FIRST_COMPLETED
        )

        # If process exited, log the code and trigger shutdown
        if process_wait_task in done:
            return_code = process_wait_task.result()
            logger.info(f"SD Web UI process exited with code {return_code}.")
            # Cancel pending log tasks
            for task in pending:
                task.cancel()
            if log_tasks: await asyncio.wait(log_tasks) # Wait for cancellation/completion
            handle_signal(signal.SIGTERM, None) # Trigger clean shutdown of this script

        # If a log task finished first (error or EOF unexpectedly?), log it
        else:
             logger.warning("A log stream finished unexpectedly. Process might still be running.")
             # Optionally, wait for process exit here or let the main loop handle it
             await process_wait_task # Wait for process to eventually finish
             return_code = process_wait_task.result()
             logger.info(f"SD Web UI process eventually exited with code {return_code}.")
             # Cancel any remaining pending log tasks
             for task in pending:
                 if task != process_wait_task: task.cancel()
             if log_tasks: await asyncio.wait(log_tasks) # Wait for cancellation/completion
             handle_signal(signal.SIGTERM, None)

    except Exception as e:
        logger.exception(f"Error in process monitor setup or execution: {e}")
        # Ensure process is terminated if monitor fails critically
        if proc.poll() is None:
             handle_signal(signal.SIGTERM, None) # Attempt graceful shutdown


async def main(sd_webui_path: str, model_arg: Optional[str], cpu_limit_percent: Optional[int]):
    # Register signal handlers for graceful shutdown
    # Use os.setsid for process group management on Unix-like systems
    preexec_fn = os.setsid if platform.system() != "Windows" else None

    # Start the Web UI process
    # Pass preexec_fn to Popen if needed (modify start_webui if Popen is called there)
    # Note: Popen is now called inside start_webui, so preexec_fn should be passed there if needed.
    # For now, let's assume start_webui handles process creation correctly.
    proc = start_webui(sd_webui_path, model_arg, cpu_limit_percent) # Pass the limit down
    if not proc:
        logger.error("Failed to initiate SD WebUI process.")
        return # Exit if process failed to start

    # Setup signal handlers AFTER the child process is potentially created
    # to avoid the main script catching signals meant for the child immediately
    signal.signal(signal.SIGINT, handle_signal) # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal) # Termination signal
    if platform.system() != "Windows":
        # SIGHUP/SIGQUIT might interfere with shell job control if run interactively,
        # but are useful for daemon-like scenarios. Keep them for robustness.
        try:
            signal.signal(signal.SIGHUP, handle_signal) # Hangup signal
            signal.signal(signal.SIGQUIT, handle_signal) # Quit signal
        except AttributeError:
             logger.warning("SIGHUP/SIGQUIT not available on this platform.")


    # Start monitoring the process output in the background
    monitor_task = asyncio.create_task(monitor_process(proc))

    # Wait for the API to become ready (optional but good practice)
    # If the API check fails, we might still want to keep the monitor running
    # to see *why* it failed from the logs.
    api_ready = await check_api_ready(SD_WEBUI_API_CHECK_URL)

    if api_ready:
        logger.info("SD API Server is running. SD Web UI process is managed. Press Ctrl+C to stop.")
    else:
        logger.error("Failed to confirm SD Web UI API readiness after startup. Check logs.")
        # Don't exit immediately, let the monitor task continue to capture potential errors
        # that prevented readiness. It will trigger shutdown if the process exits.

    # Wait for the monitor task to complete (which happens on process exit or critical error)
    try:
        await monitor_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled, monitor task should have been handled.")
    finally:
        # Ensure cleanup runs if monitor task finishes/cancelled unexpectedly without calling handle_signal
        if webui_process and webui_process.poll() is None:
             logger.warning("Monitor task ended, but process still running. Forcing shutdown.")
             handle_signal(signal.SIGTERM, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A3X Server to manage Stable Diffusion Web UI process.")
    parser.add_argument(
        "--sd-webui-path",
        type=str,
        default=SD_WEBUI_DEFAULT_PATH,
        help=f"Path to the stable-diffusion-webui directory (default: {SD_WEBUI_DEFAULT_PATH})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional: Path to a specific model checkpoint file to load (might be ignored by webui.sh)."
    )
    # <<< ADDED CPU Limit Argument >>>
    parser.add_argument(
        "--cpu-limit-percent",
        type=int,
        default=None,
        help="Optional: Limit SD process CPU usage to this percentage using cpulimit (requires cpulimit and taskset)."
    )

    args = parser.parse_args()

    # Resolve the path
    webui_path = str(Path(args.sd_webui_path).resolve())

    # Run the main async function
    try:
        asyncio.run(main(webui_path, args.model, args.cpu_limit_percent)) # Pass the parsed limit
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating shutdown...")
        # handle_signal should have been called if process was running
        # If we reach here, maybe the process wasn't started or already stopped
        if webui_process and webui_process.poll() is None:
             handle_signal(signal.SIGINT, None) # Manually trigger if needed
    except Exception as e:
         logger.exception("Unhandled exception in main execution loop:")
    finally:
         logger.info("SD API Server script finished.") 