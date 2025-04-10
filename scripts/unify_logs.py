# scripts/unify_logs.py
import asyncio
import aiofiles
import os
import pathlib
from datetime import datetime, timezone

# --- Configuration ---
LOGS_DIR = pathlib.Path(__file__).parent.parent / "logs"
A3X_LOG_FILE = LOGS_DIR / "a3x.log"
LLM_LOG_FILE = LOGS_DIR / "llama_server.log"
UNIFIED_LOG_FILE = LOGS_DIR / "unified.log"
POLL_INTERVAL = 0.5 # Seconds

async def follow(file_path: pathlib.Path, prefix: str):
    """
    Asynchronously tails a file, handling rotation/recreation,
    and writes prefixed lines to the unified log.
    """
    print(f"Starting to follow {file_path} with prefix [{prefix}]...")
    current_inode = None
    f = None

    while True:
        try:
            if f is None:
                # Ensure file exists before opening
                if not file_path.exists():
                    print(f"[{prefix}] File {file_path} does not exist yet. Waiting...")
                    await asyncio.sleep(POLL_INTERVAL * 5) # Wait longer if file doesn't exist
                    continue

                f = await aiofiles.open(file_path, mode='r', encoding='utf-8', errors='ignore')
                try:
                    stat_res = os.stat(file_path)
                    current_inode = stat_res.st_ino
                except FileNotFoundError:
                    # File might have been deleted between exists() check and stat()
                    print(f"[{prefix}] File {file_path} disappeared before stat. Retrying...")
                    await f.close()
                    f = None
                    continue

                # Decide where to start reading from
                # For simplicity upon (re)start, seek to end to avoid processing old logs.
                await f.seek(0, os.SEEK_END)
                print(f"[{prefix}] Opened {file_path}. Inode: {current_inode}. Seeking to end.")

            # Check for file rotation/deletion
            try:
                stat_res = os.stat(file_path)
                if stat_res.st_ino != current_inode:
                    print(f"[{prefix}] Inode changed for {file_path}. Reopening file.")
                    await f.close()
                    f = None
                    current_inode = None
                    continue # Re-enter loop to reopen
            except FileNotFoundError:
                print(f"[{prefix}] File {file_path} not found (deleted/rotated?). Reopening.")
                await f.close()
                f = None
                current_inode = None
                continue # Re-enter loop to reopen

            # Read new lines
            line = await f.readline()
            if line:
                line_content = line.strip()
                if line_content: # Ignore empty lines
                    now_utc = datetime.now(timezone.utc).isoformat(timespec='microseconds')
                    formatted_line = f"{now_utc} [{prefix}] {line_content}\n"
                    async with aiofiles.open(UNIFIED_LOG_FILE, mode='a', encoding='utf-8') as unified_log:
                        await unified_log.write(formatted_line)
                    # print(formatted_line.strip()) # Optional: print to console too
            else:
                # No new line, wait before polling again
                await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"[{prefix}] Error following {file_path}: {e}. Attempting to recover...")
            if f:
                try:
                    await f.close()
                except Exception:
                    pass # Ignore errors during close on error recovery
            f = None
            current_inode = None
            await asyncio.sleep(POLL_INTERVAL * 2) # Longer sleep on error

async def main():
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {LOGS_DIR}")
    print(f"Unified log target: {UNIFIED_LOG_FILE}")

    # Ensure target files exist initially (avoids immediate error in follow)
    A3X_LOG_FILE.touch(exist_ok=True)
    LLM_LOG_FILE.touch(exist_ok=True)

    task_a3x = asyncio.create_task(follow(A3X_LOG_FILE, "A3X"))
    task_llm = asyncio.create_task(follow(LLM_LOG_FILE, "LLM"))

    print("Log unification started. Press Ctrl+C to stop.")
    try:
        await asyncio.gather(task_a3x, task_llm)
    except asyncio.CancelledError:
        print("Log unification stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.") 