from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Central registry for skills
SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {}

logger.debug(f"SKILL_REGISTRY initialized in skill_management.py (ID: {id(SKILL_REGISTRY)})") 