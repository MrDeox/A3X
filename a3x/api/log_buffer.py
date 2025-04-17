import time
import datetime
from collections import deque
from typing import Deque, Dict, Any, Optional
import logging

# Configure a specific logger for the buffer if needed, or use root
log_buffer_logger = logging.getLogger("LogBuffer")

# Define the maximum size of the log buffer
DEFAULT_MAX_LOG_EVENTS = 200

# The deque to hold log events
LOG_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=DEFAULT_MAX_LOG_EVENTS)

def log_event(level: str, message: str, source: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None):
    """Adds a structured event to the circular log buffer."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    event = {
        "timestamp": timestamp,
        "level": level.upper(), # e.g., INFO, DEBUG, ERROR
        "message": message,
        "source": source, # e.g., Orchestrator, FileManagerFragment, LLMInterface
    }
    if extra_data:
        if isinstance(extra_data, dict):
            event.update(extra_data) # Merge if it's a dictionary
        else:
            log_buffer_logger.warning(f"Received non-dict extra_data in log_event (type: {type(extra_data)}). Storing under 'raw_extra_data'.")
            event["raw_extra_data"] = str(extra_data) # Store as string representation

    LOG_EVENTS.appendleft(event)
    # Optional: Also log to standard logger
    # log_buffer_logger.log(logging.getLevelName(level.upper()), f"[{source or 'Unknown'}] {message}")

def get_log_events(limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """Retrieves log events from the buffer, optionally limiting the number."""
    if limit is None:
        return list(LOG_EVENTS)
    else:
        # Since deque appends left, slicing gives the most recent items
        return list(LOG_EVENTS)[:limit]

def configure_buffer_size(max_size: int):
    """Allows runtime configuration of the log buffer size."""
    global LOG_EVENTS
    if max_size > 0:
        log_buffer_logger.info(f"Configuring log buffer size to {max_size}")
        # Create a new deque with the new size and copy existing items
        new_deque = deque(list(LOG_EVENTS), maxlen=max_size)
        LOG_EVENTS = new_deque
    else:
        log_buffer_logger.warning("Invalid log buffer size requested (must be > 0).") 