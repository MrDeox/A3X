# a3x/core/skill_utils.py
import inspect
import logging
from typing import Any, Callable, Dict, get_type_hints

logger = logging.getLogger(__name__)

# Function moved from a3x/assistant_cli.py
def generate_schema(func: Callable, name: str, description: str) -> Dict[str, Any]:
    \"\"\"Generates a basic JSON-like schema from function signature and docstring.\"\"\"
    schema = {
        \"name\": name,
        \"description\": description or f\"Skill: {name}\",
        \"parameters\": {
            \"type\": \"object\",
            \"properties\": {},
            \"required\": [],
        },
    }
    try:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        for param_name, param in sig.parameters.items():
            # Skip context-like parameters injected by system? (e.g., \'ctx\', \'context\')
            # For now, include all parameters found.
            # if param_name in [\'ctx\', \'context\']:\n            #    continue

            param_info = {}
            param_type = type_hints.get(param_name)

            # Basic type mapping (extend as needed)
            if param_type == str:
                param_info[\"type\"] = \"string\"
            elif param_type == int:
                param_info[\"type\"] = \"integer\"
            elif param_type == float:
                param_info[\"type\"] = \"number\"
            elif param_type == bool:
                param_info[\"type\"] = \"boolean\"
            elif param_type == list or getattr(param_type, \"__origin__\", None) == list:
                param_info[\"type\"] = \"array\"
                # Try to infer item type (basic)
                item_type_args = getattr(param_type, \"__args__\", [])
                if item_type_args and item_type_args[0] == str:
                    param_info[\"items\"] = {\"type\": \"string\"}
                else:
                    param_info[\"items\"] = {}  # Generic item
            elif param_type == dict or getattr(param_type, \"__origin__\", None) == dict:
                param_info[\"type\"] = \"object\"
            else:
                param_info[\"type\"] = \"string\"  # Default to string if type is unknown/complex

            # Add description from docstring if possible (simple parsing)
            # This requires a specific docstring format (e.g., Google style)
            # For now, just add the type.
            # param_info[\"description\"] = f\"Parameter {param_name}\"

            schema[\"parameters\"][\"properties\"][param_name] = param_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                schema[\"parameters\"][\"required\"].append(param_name)

    except Exception as e:
        logger.warning(f\"Could not introspect signature for {name}: {e}\")

    return schema 