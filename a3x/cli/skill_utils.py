# a3x/cli/skill_utils.py
import os
import ast
import logging
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)

class CriticalSkillRegistrationError(Exception):
    """Custom exception for critical skill registration failures."""
    def __init__(self, message, missing_skills=None):
        super().__init__(message)
        self.missing_skills = missing_skills or []

def discover_expected_skills(skills_dir: str) -> Set[str]:
    """Dynamically discovers expected skill function names from Python files in the skills directory."""
    expected_skills: Set[str] = set()
    logger.debug(f"Discovering expected skills in: {skills_dir}")
    if not os.path.isdir(skills_dir):
        logger.warning(f"Skills directory not found: {skills_dir}")
        return expected_skills

    for filename in os.listdir(skills_dir):
        if filename.endswith(".py") and not filename.startswith("__") and not filename.startswith("."):
            module_name = filename[:-3]
            full_path = os.path.join(skills_dir, filename)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8") as file_content:
                        tree = ast.parse(file_content.read(), filename=filename)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not node.name.startswith("_"):
                                expected_skills.add(node.name)
                                logger.debug(f"Discovered potential skill function: {node.name} in {filename}")
                except Exception as e:
                    logger.warning(f"Could not parse {filename} to discover skills: {e}")

    logger.info(f"Discovered {len(expected_skills)} potential skill functions.")
    return expected_skills

def validate_registered_skills(expected_skills: Set[str], registered_skills_dict: Dict[str, Any]):
    """Validates if expected skills are present in the registered skills dictionary."""
    logger.debug("Validating registered skills against discovered skill functions...")
    registered_skill_names = set(registered_skills_dict.keys())
    missing_skills = expected_skills - registered_skill_names
    potentially_unregistered = registered_skill_names - expected_skills

    if missing_skills:
        essential_skills = {"file_manager", "planning", "execute_code", "final_answer"} # Example set
        critical_missing = missing_skills.intersection(essential_skills)
        warning_message = f"Skill Validation Warning: The following discovered skill functions might not be registered: {missing_skills}"
        logger.warning(warning_message)

        if critical_missing:
            error_message = f"Critical skills missing registration: {critical_missing}"
            logger.error(error_message)
            raise CriticalSkillRegistrationError(error_message, missing_skills=list(critical_missing))

    if potentially_unregistered:
        logger.debug(f"Skills registered but not found by discovery (might be aliases/dynamic): {potentially_unregistered}")

    logger.info("Skill registration validation check complete.") 