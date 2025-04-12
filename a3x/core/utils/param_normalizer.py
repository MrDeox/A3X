# core/utils/param_normalizer.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Dictionary mapping skill names to their parameter aliases
# Format: { "skill_name": { "canonical_param": ["alias1", "alias2", ...], ... } }
PARAM_ALIASES: Dict[str, Dict[str, list[str]]] = {
    "generate_code": {
        "purpose": ["objective", "prompt", "task", "description", "code_description", "code"],
        "language": ["lang"],
        "construct_type": ["type", "structure"],
        "context": ["reference"]
    },
    # Add other skills and their aliases here as needed
    "write_file": {
        "path": ["file_path", "filepath", "target_path"],
        "content": ["text", "data", "file_content", "content_to_write"]
    },
    # "list_directory": {
    #     "path": ["dir", "directory", "folder_path"]
    # },
}

def normalize_action_input(skill_name: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes the keys in the input dictionary for a given skill based on predefined aliases.

    Args:
        skill_name: The name of the skill being called.
        input_dict: The original dictionary of parameters provided by the LLM.

    Returns:
        A new dictionary with parameter keys normalized to their canonical names.
        Keys not found in aliases are passed through unchanged.
    """
    if not isinstance(input_dict, dict):
        logger.warning(f"Invalid input_dict type ({type(input_dict)}) for normalization. Expected dict. Returning original.")
        return input_dict # Or raise error? For now, return original

    if skill_name not in PARAM_ALIASES:
        # logger.debug(f"No parameter aliases defined for skill '{skill_name}'. Returning original input.")
        return input_dict # No aliases defined for this skill

    normalized_dict = {}
    used_aliases = set() # Keep track of aliases used to avoid double mapping if LLM provides both alias and canonical
    aliases_for_skill = PARAM_ALIASES[skill_name]

    # Invert the alias map for easier lookup: { "alias": "canonical" }
    alias_to_canonical_map: Dict[str, str] = {}
    for canonical, alias_list in aliases_for_skill.items():
        for alias in alias_list:
            if alias in alias_to_canonical_map:
                logger.warning(f"Alias '{alias}' is defined for multiple canonical parameters ('{alias_to_canonical_map[alias]}' and '{canonical}') in skill '{skill_name}'. Using the first definition encountered.")
            else:
                alias_to_canonical_map[alias] = canonical

    original_keys = list(input_dict.keys()) # Iterate over a copy of keys

    for key in original_keys:
        value = input_dict[key]
        canonical_key = key # Assume it's already canonical initially

        # Check if the key is an alias
        if key in alias_to_canonical_map:
            canonical_key = alias_to_canonical_map[key]
            logger.info(f"Normalizing parameter for skill '{skill_name}': Alias '{key}' mapped to canonical '{canonical_key}'.")
            # Check if the canonical key is *also* present in the original input
            if canonical_key in input_dict and key != canonical_key:
                 logger.warning(f"Both alias '{key}' and canonical key '{canonical_key}' found in input for skill '{skill_name}'. Prioritizing value from canonical key '{canonical_key}'.")
                 # Skip adding the value from the alias if the canonical key is already present
                 continue
            # Check if this alias has already provided a value for the canonical key
            if canonical_key in used_aliases:
                 logger.warning(f"Multiple aliases resolved to canonical key '{canonical_key}' for skill '{skill_name}'. Using the first value encountered.")
                 continue # Skip subsequent aliases for the same canonical key

        # Add to normalized dict using the canonical key
        if canonical_key in normalized_dict:
             # This case should ideally be handled by the checks above, but as a safety net:
             logger.warning(f"Canonical key '{canonical_key}' already exists in normalized dict for skill '{skill_name}'. Overwriting value (check alias definitions).")

        normalized_dict[canonical_key] = value
        used_aliases.add(canonical_key) # Mark the canonical key as having received a value

    # Check for keys in the original dict that weren't aliases or canonicals (pass them through)
    # This is implicitly handled by the loop logic now. If a key wasn't an alias,
    # canonical_key remains key, and it gets added to normalized_dict.

    logger.debug(f"Original input for '{skill_name}': {input_dict}")
    logger.debug(f"Normalized input for '{skill_name}': {normalized_dict}")
    return normalized_dict 