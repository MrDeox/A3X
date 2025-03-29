import logging
from core.config import MAX_HISTORY_TURNS # Import the constant

def trim_history(history: list, max_turns: int, agent_logger: logging.Logger) -> list:
    """Trims the history list to keep only the most recent max_turns interactions, preserving the first (Human) message."""
    # Calculate max items to keep: 1 (Human) + (max_turns * 2 [LLM response + Observation])
    max_keep = 1 + (max_turns * 2)
    if len(history) > max_keep:
        agent_logger.debug(f"Trimming history from {len(history)} entries to keep max {max_turns} turns ({max_keep} items).")
        # Keep the first message (Human prompt) and the last (max_keep - 1) messages
        trimmed_history = [history[0]] + history[-(max_keep-1):]
        agent_logger.debug(f"History trimmed to {len(trimmed_history)} entries.")
        return trimmed_history
    else:
        return history # No trimming needed
