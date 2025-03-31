# core/skills_utils.py
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def create_skill_response(
    status: str,
    action: str,
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    error_details: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a standardized dictionary response for skill execution results.

    Args:
        status: 'success' or 'error'.
        action: A string code indicating the specific action taken or error type.
        data: A dictionary containing the primary data payload (e.g., results).
        message: A user-friendly message describing the outcome.
        error_details: Specific technical details if status is 'error'.

    Returns:
        A dictionary representing the skill's execution result.
    """
    response = {
        "status": status,
        "action": action,
    }
    if data is not None:
        response["data"] = data
    if message is not None:
        response["message"] = message
    if error_details is not None and status == 'error':
        response["error_details"] = error_details

    # Log the response being created
    if status == 'error':
        logger.error(f"Skill Response (Error): Action='{action}', Message='{message}', Details='{error_details}'")
    else:
        logger.debug(f"Skill Response (Success): Action='{action}', Message='{message}'") # Log success as debug

    return response

# Potential future additions:
# - WORKSPACE_ROOT variable if needed globally by skills
# - Helper functions for common skill tasks (e.g., path validation)
