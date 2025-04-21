# a3x/fragments/architect_advisor.py

import asyncio
import logging
import json
import re
from typing import List, Dict, Any

from .base import BaseFragment

logger = logging.getLogger(__name__)

# Simple keywords to identify potential actionable suggestions
# This can be refined with more sophisticated NLP later
ACTIONABLE_KEYWORDS = [
    "separar", "mover", "modularizar", "refatorar", "quebrar",
    "simplificar", "extrair", "isolar", "reduzir acoplamento",
    "violação", "acoplamento excessivo", "responsabilidade única"
]

# Mapping keywords to potential actions (can be expanded)
KEYWORD_ACTION_MAP = {
    "separar": "split_responsibilities",
    "quebrar": "split_responsibilities",
    "extrair": "extract_logic",
    "isolar": "isolate_component",
    "mover": "move_code",
    "modularizar": "refactor_module",
    "refatorar": "refactor_module",
    "simplificar": "refactor_module",
    "reduzir acoplamento": "address_coupling_violation",
    "violação": "address_coupling_violation",
    "acoplamento excessivo": "address_coupling_violation",
    "responsabilidade única": "split_responsibilities",
}

class ArchitectAdvisorFragment(BaseFragment):
    """A reflective fragment that analyzes architectural reports and suggests refactoring actions."""

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "Analyzes architectural reports and suggests refactoring actions."

    async def run(self):
        """Runs the architectural evaluation and generates refactoring directives."""
        logger.info("ArchitectAdvisorFragment starting analysis cycle...")

        module_to_scan = "a3x" # TODO: Make this configurable via args/context

        try:
            logger.info(f"Executing 'evaluate_architecture' skill for module: {module_to_scan}")
            # Ensure the skill name matches the one defined
            # Assuming execute_skill returns a string (the report)
            report: Any = await self.execute_skill(
                "evaluate_architecture",
                {"module_path": module_to_scan}
            )

            if not isinstance(report, str):
                 logger.error(f"'evaluate_architecture' skill did not return a string report. Type: {type(report)}")
                 return

            logger.debug(f"Received architecture report:\n{report[:500]}...")

            directives = self.extract_directives_from_report(report)

            if not directives:
                logger.info("No actionable directives extracted from the report.")
            else:
                logger.info(f"Extracted {len(directives)} actionable directives. Broadcasting...")
                for directive in directives:
                    try:
                        # Broadcast each directive as a separate message
                        await self.communicator.broadcast(
                             message=directive, # Already a dict/JSON
                             msg_type="architecture_suggestion"
                         )
                        logger.debug(f"Broadcasted directive: {directive}")
                    except Exception as broadcast_err:
                         logger.error(f"Failed to broadcast directive {directive}: {broadcast_err}")

        except Exception as e:
            logger.exception(f"Error during architectural evaluation cycle: {e}")

        logger.info("ArchitectAdvisorFragment analysis cycle finished.")
        # TODO: Implement logic for periodic runs or trigger mechanism if needed
        # For now, it runs once.

    def extract_directives_from_report(self, report: str) -> List[Dict[str, Any]]:
        """Parses the Markdown report and extracts actionable directives."""
        directives = []
        current_file = None

        # Regex to find file sections more reliably
        file_section_regex = re.compile(r"^### Arquivo: `([^`]+)`", re.MULTILINE)
        # Split the report into potential sections based on the file header
        sections = file_section_regex.split(report)

        # sections will be [text_before_first_file, filename1, text_for_file1, filename2, text_for_file2, ...]
        # We only care about pairs of (filename, text_for_file)
        for i in range(1, len(sections), 2):
            filename = sections[i].strip()
            diagnosis_text = sections[i+1].strip()

            logger.debug(f"Analyzing diagnosis for file: {filename}")

            # Look for sentences/lines containing keywords within the diagnosis text
            # Split diagnosis into lines or sentences for better granularity
            lines = diagnosis_text.replace("```", "").strip().split('\n')
            for line in lines:
                normalized_line = line.lower().strip()
                if not normalized_line:
                    continue

                matched_keyword = None
                for keyword in ACTIONABLE_KEYWORDS:
                    if keyword in normalized_line:
                        matched_keyword = keyword
                        break # Take the first keyword found in the line

                if matched_keyword:
                    # Map keyword to a generic action type
                    action = KEYWORD_ACTION_MAP.get(matched_keyword, "refactor_module") # Default action

                    directive = {
                        "type": "directive",
                        "action": action,
                        "target": filename, # File associated with this diagnosis section
                        "message": line.strip() # The original suggestion line
                    }
                    directives.append(directive)
                    logger.debug(f"Generated directive from keyword '{matched_keyword}': {directive}")
                    # Avoid adding multiple directives for the same line if multiple keywords match
                    break # Move to the next line once a keyword is found

        return directives

# Example of how this fragment might be registered (in a registry setup)
# fragments_registry.register("architect_advisor", ArchitectAdvisorFragment) 