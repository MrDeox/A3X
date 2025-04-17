import logging
from typing import Dict, Any

from a3x.core.skills import skill
from a3x.core.context import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name="get_public_ip",
    description="Fetch the public IP address using ipify.org",
    parameters={
        "ipify_url": { "type": "string", "description": "The URL of the ipify service" }
    }
)
async def get_public_ip(ctx: _ToolExecutionContext, ipify_url: str) -> Dict[str, Any]:
    """
    Fetch the public IP address using ipify.org.

    Args:
        ctx: Execution context.
        ipify_url: The URL of the ipify service.

    Returns:
        Dict[str, Any]: {'status': 'success'/'error', 'data': {...}}
    """
    logger.info(f"Executing skill: get_public_ip")
    try:
        import requests
        response = requests.get(ipify_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        ip_address = response.text.strip()
        return { "status": "success", "data": { "ip_address": ip_address } }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching IP address: {e}", exc_info=True)
        return { "status": "error", "data": { "message": f"Failed to fetch IP address: {e}" } }
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return { "status": "error", "data": { "message": f"Unexpected error: {e}" } }