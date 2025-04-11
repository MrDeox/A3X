import logging
import subprocess
import signal
import sys
import os
import asyncio
import aiohttp
import platform
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration --- 
SD_WEBUI_API_CHECK_URL = "http://127.0.0.1:7860/docs" # Default API endpoint for checks
CHECK_INTERVAL_SECONDS = 5
STARTUP_TIMEOUT_SECONDS = 120 # Give it 2 minutes to start

SD_WEBUI_DEFAULT_PATH = "./stable-diffusion-webui" # Default relative path

# --- Global variable for the process --- 
webui_process = None

async def check_api_ready(url: str) -> bool:
    \"\"\"Periodically checks if the SD Web UI API is responsive.\"\"\"
    logger.info(f\"Checking if SD Web UI API is ready at {url}...\")
    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=3) as response:
                    if response.status == 200:
                        logger.info(f\"SD Web UI API is ready! (Status: {response.status})\")
                        return True
                    else:
                        logger.debug(f\"API check failed with status: {response.status}\")
        except aiohttp.ClientConnectorError:
            logger.debug(f\"API connection refused. Retrying in {CHECK_INTERVAL_SECONDS}s...\")
        except asyncio.TimeoutError:
            logger.debug(f\"API check timed out. Retrying in {CHECK_INTERVAL_SECONDS}s...\")
        except Exception as e:
            logger.warning(f\"API check encountered an error: {e}. Retrying...\")

        if asyncio.get_event_loop().time() - start_time > STARTUP_TIMEOUT_SECONDS:
            logger.error(\"SD Web UI API did not become ready within the timeout period.\")
            return False
        
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

def start_webui(sd_webui_path: str, model_arg: Optional[str] = None):
    \"\"\"Starts the Stable Diffusion Web UI process.\"\"\"
    global webui_process
    
    if not os.path.isdir(sd_webui_path):
        logger.error(f\"Stable Diffusion WebUI path not found: {sd_webui_path}\")
        logger.error(\"Please provide the correct path using --sd-webui-path or ensure it exists at the default location.\")
        sys.exit(1)

    # Determine the script name based on OS
    script_name = "webui.bat" if platform.system() == "Windows" else "webui.sh"
    script_path = os.path.join(sd_webui_path, script_name)

    if not os.path.exists(script_path):
        logger.error(f\"Web UI launch script not found: {script_path}\")
        sys.exit(1)

    # Construct the command
    command = [script_path, "--api"] # Always add --api flag
    if model_arg:
        command.extend(["--ckpt", model_arg]) # Use --ckpt for specific model loading
        # Alternative: --ckpt-dir if you want to specify a directory
        logger.info(f\"Adding model argument: --ckpt {model_arg}\")
    
    # Add any other necessary flags here, e.g., --xformers, --lowvram
    # command.append("--xformers") 

    logger.info(f\"Starting Stable Diffusion Web UI from: {sd_webui_path}\")
    logger.info(f\"Executing command: {' '.join(command)}\")
    
    try:
        # Use subprocess.Popen for non-blocking execution
        # On Windows, DETACHED_PROCESS might be needed if shell=True causes issues, but usually not required
        # For Linux/macOS, shell=False is generally safer
        webui_process = subprocess.Popen(
            command,
            cwd=sd_webui_path, # Run from the webui directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            # shell=(platform.system() == "Windows") # Use shell=True cautiously, maybe only on Windows
        )
        logger.info(f\"SD Web UI process started with PID: {webui_process.pid}\")
        return webui_process
    except Exception as e:
        logger.exception(\"Failed to start SD Web UI process:\")
        sys.exit(1)

def handle_signal(sig, frame):
    \"\"\"Gracefully shuts down the Web UI process upon receiving a signal.\"\"\"
    logger.info(f\"Received signal {sig}. Shutting down SD Web UI process...\")
    global webui_process
    if webui_process and webui_process.poll() is None: # Check if process exists and is running
        logger.info(f\"Terminating SD Web UI process (PID: {webui_process.pid})...\")
        try:
            # Send SIGTERM first for graceful shutdown
            if platform.system() == "Windows":
                 webui_process.send_signal(signal.CTRL_C_EVENT) # Try Ctrl+C on Windows
                 # Alternative: webui_process.terminate()
            else:
                 webui_process.terminate() # SIGTERM on Unix-like
            
            # Wait a bit for graceful exit
            webui_process.wait(timeout=10)
            logger.info(\"SD Web UI process terminated gracefully.\")
        except subprocess.TimeoutExpired:
            logger.warning(\"Graceful shutdown timed out. Forcing termination (SIGKILL)...\")
            webui_process.kill() # Force kill if terminate didn't work
            webui_process.wait()
            logger.info(\"SD Web UI process killed.\")
        except Exception as e:
            logger.error(f\"Error during SD Web UI process termination: {e}\")
            # Still try to kill if termination failed badly
            try:
                 if webui_process.poll() is None:
                      webui_process.kill()
                      webui_process.wait()
                      logger.info(\"SD Web UI process force-killed after error.\")
            except Exception as kill_e:
                 logger.error(f\"Error during force-kill: {kill_e}\")
    else:
        logger.info(\"SD Web UI process already terminated or not started.\")
    sys.exit(0) # Exit the server script

async def monitor_process(proc):
    \"\"\"Monitors the stdout/stderr of the process and logs it.\"\"\"
    while True:
        try:
            # Read stdout line by line
            if proc.stdout:
                stdout_line = await asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline)
                if stdout_line:
                    logger.info(f\"[SD WebUI STDOUT]: {stdout_line.strip()}\")
            
            # Read stderr line by line
            if proc.stderr:
                stderr_line = await asyncio.get_event_loop().run_in_executor(None, proc.stderr.readline)
                if stderr_line:
                    logger.error(f\"[SD WebUI STDERR]: {stderr_line.strip()}\")

            # Check if process has exited
            if proc.poll() is not None:
                logger.info(f\"SD Web UI process exited with code {proc.poll()}.\")
                # Optionally trigger a shutdown or restart here
                handle_signal(signal.SIGTERM, None) # Trigger clean shutdown of this script
                break

            await asyncio.sleep(0.1) # Small delay to prevent busy-waiting

        except Exception as e:
            logger.exception(f\"Error in process monitor loop: {e}\")
            # Decide if we should break or continue monitoring after an error
            break

async def main(sd_webui_path: str, model_arg: Optional[str]):
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal) # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal) # Termination signal
    if platform.system() != "Windows":
        signal.signal(signal.SIGHUP, handle_signal) # Hangup signal
        signal.signal(signal.SIGQUIT, handle_signal) # Quit signal

    # Start the Web UI process
    proc = start_webui(sd_webui_path, model_arg)
    if not proc:
        return # Exit if process failed to start

    # Start monitoring the process output in the background
    monitor_task = asyncio.create_task(monitor_process(proc))

    # Wait for the API to become ready
    api_ready = await check_api_ready(SD_WEBUI_API_CHECK_URL)

    if api_ready:
        logger.info(\"SD API Server is running. SD Web UI process is managed. Press Ctrl+C to stop.\")
        # Keep the server running indefinitely until a signal is received
        # The monitor_task will call handle_signal if the process exits unexpectedly
        await monitor_task # Wait for the monitor task to complete (which happens on process exit)
    else:
        logger.error(\"Failed to confirm SD Web UI API readiness. Shutting down...\")
        handle_signal(signal.SIGTERM, None) # Trigger shutdown
        # Ensure monitor task is cancelled if it hasn't finished
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                logger.info(\"Monitor task cancelled.\")

if __name__ == \"__main__\":
    parser = argparse.ArgumentParser(description=\"A3X Server to manage Stable Diffusion Web UI process.\")
    parser.add_argument(
        \"--sd-webui-path\", 
        type=str, 
        default=SD_WEBUI_DEFAULT_PATH, 
        help=f\"Path to the Stable Diffusion Web UI installation directory (default: {SD_WEBUI_DEFAULT_PATH})\"
    )
    parser.add_argument(
        \"--model\", 
        type=str, 
        default=None,
        help=\"Optional: Path or name of the specific checkpoint model file to load (passed via --ckpt).\"
    )
    
    args = parser.parse_args()

    # Resolve the path relative to the script location if it's relative
    # Best practice is usually to run from project root, so default should be fine
    webui_path = args.sd_webui_path
    if not os.path.isabs(webui_path):
         # This assumes the script is run from the project root (where ./stable-diffusion-webui might be)
         webui_path = os.path.abspath(webui_path) 
         logger.info(f\"Resolved relative sd-webui-path to: {webui_path}\")

    try:
        asyncio.run(main(webui_path, args.model))
    except KeyboardInterrupt:
        logger.info(\"Keyboard interrupt received. Exiting cleanly.\")
        handle_signal(signal.SIGINT, None) # Ensure cleanup on manual interrupt 