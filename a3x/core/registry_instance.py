# a3x/core/registry_instance.py
import logging
from a3x.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

logger.info("Instantiating global SKILL_REGISTRY (ToolRegistry instance)...")
# Define the singleton instance of the ToolRegistry
SKILL_REGISTRY = ToolRegistry()
logger.info(f"Global SKILL_REGISTRY (ID: {id(SKILL_REGISTRY)}) created.") 