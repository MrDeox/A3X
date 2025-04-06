import os
import logging
import re
import json
from datetime import datetime
from typing import Dict, Any

# from core.skill_registry import register_skill # REMOVED
from a3x.core.tools import skill  # ADDED
from a3x.core.skills_utils import create_skill_response
# from core.config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_USERNAME, GUMROAD_API_KEY, GUMROAD_PRODUCT_ID # REMOVED - Variables not defined in config.py

# Setup logging
log_dir = "output/autopublisher/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(
    log_dir, "autopublisher.log"
)  # Optional: file handler for general logs

# Configure logger for the skill
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set desired level

# JSON Log handler (separate file per execution) - Not using standard handler for this
# General file handler (optional)
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Add console handler to see logs during execution
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# --- Constantes --- (Placeholder values)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Load attempt, will be None if not set
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GUMROAD_API_KEY = os.getenv("GUMROAD_API_KEY")
GUMROAD_PRODUCT_ID = os.getenv("GUMROAD_PRODUCT_ID")

# Define the skill name consistently
SKILL_NAME = "auto_publisher"


# Removed decorator from class
class AutoPublisherSkill:
    def __init__(self, output_dir="output/autopublisher"):
        self.output_dir = output_dir
        # Ensure base output dir exists, log dir is handled separately above
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = log_dir  # Store log directory path

    def generate_content(self, topic: str) -> str:
        """
        Generates textual content on the given topic using the LLM (currently simulated).
        """
        prompt = f"Write a short blog post about {topic}."
        logger.info(
            f"[{SKILL_NAME}] Generating content with prompt: '{prompt[:50]}...'"
        )
        try:
            # TEMPORARY: Simulate LLM response for now
            logger.warning(
                f"[{SKILL_NAME}] Using simulated LLM response for content generation."
            )
            generated_text = (
                f"Simulated blog post about {topic}. Lorem ipsum dolor sit amet..."
            )

            # TODO: Implement proper async handling for call_llm if required

            logger.info(f"[{SKILL_NAME}] Content generated successfully.")
            return generated_text
        except Exception as e:
            logger.exception(f"[{SKILL_NAME}] Error during content generation:")
            # Return error within the string, the orchestrator will handle the failure status
            return f"Error generating content: {e}"

    def export_content(
        self, content: str, filename_prefix: str = "content", format: str = "markdown"
    ) -> str | None:
        """
        Exports the content to the specified format (.md or .txt for now). Returns filepath or None on error.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            format_lower = format.lower()
            extension = (
                ".md" if format_lower == "markdown" else ".txt"
            )  # Basic extension mapping
            if format_lower not in ["markdown", "txt"]:
                logger.warning(
                    f"[{SKILL_NAME}] Unsupported export format '{format}'. Defaulting to '.txt'."
                )
                extension = ".txt"

            filename = f"{filename_prefix}_{timestamp}{extension}"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"[{SKILL_NAME}] Content exported to: {filepath}")
            return filepath
        except Exception:
            logger.exception(f"[{SKILL_NAME}] Error exporting content:")
            return None

    def publish_to_github(self, filepath: str) -> str:
        """
        Simulates publishing content to GitHub. Returns a simulated URL.
        """
        logger.info(
            f"[SIMULATION][{SKILL_NAME}] Would publish {filepath} to GitHub repository."
        )
        # Assume GITHUB_USERNAME and GITHUB_REPO are set for a more realistic simulation
        user = GITHUB_USERNAME or "your_username"
        repo = GITHUB_REPO or "your_repo"
        return f"https://github.com/{user}/{repo}/blob/main/{os.path.basename(filepath)}"  # Simplified URL

    def publish_to_gumroad(self, filepath: str, title: str, price: float = 0.99) -> str:
        """
        Simulates publishing a product to Gumroad. Returns a simulated URL.
        Includes title and price in simulation.
        """
        logger.info(
            f"[SIMULATION][{SKILL_NAME}] Would publish {filepath} to Gumroad as product '{title}' for ${price:.2f}."
        )
        # Simulate a unique product slug based on title
        slug = title.lower().replace(" ", "-").replace(r"[^a-z0-9-]", "")[:50]
        return f"https://{GITHUB_USERNAME or 'yourusername'}.gumroad.com/l/{slug}"  # Simulated Gumroad link structure

    # <<< NEW: Private helper for logging >>>
    def _log_execution_result(
        self, log_data: Dict[str, Any], status: str
    ) -> str | None:
        """Saves the execution log data to a JSON file."""
        log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{status}.json"
        log_filepath = os.path.join(self.log_dir, log_filename)
        try:
            with open(log_filepath, "w", encoding="utf-8") as f_log:
                json.dump(log_data, f_log, indent=4)
            logger.info(f"[{SKILL_NAME}] Operation log saved to: {log_filepath}")
            return str(log_filepath)  # Return path on success
        except Exception as log_e:
            logger.error(
                f"[{SKILL_NAME}] Failed to write JSON log file to {log_filepath}: {log_e}",
                exc_info=True,
            )
            return None  # Return None on failure

    # --- Main execute method with @skill decorator ---
    @skill(
        name=SKILL_NAME,  # Use the consistent name
        description="Generates content based on a topic, exports it, and publishes (simulated) to a target platform (GitHub/Gumroad).",
        parameters={
            "topic": (str, "The topic to generate content about."),
            "format": (str, "markdown"),
            "target": (str, "gumroad"),
            "price": (float, 0.99),
        },
    )
    def execute(
        self,
        topic: str,
        format: str = "markdown",
        target: str = "gumroad",
        price: float = 0.99,
    ) -> dict:
        """
        Orchestrates content generation, export, and publication.
        """
        logger.info(
            f"Executing {SKILL_NAME} skill for topic: '{topic}', format: '{format}', target: '{target}'"
        )
        status = "error"  # Default status
        message = "Orchestration failed."
        filepath = None
        final_link = None
        log_filepath_str = None
        error_details = None

        # Prepare log data structure
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "format": format,
            "target": target,
            "price": price if target.lower() == "gumroad" else None,
            "status": status,  # Will be updated
            "content_generated": False,
            "filepath": None,
            "final_link": None,
            "error": None,
        }

        try:
            # 1. Generate Content
            content = self.generate_content(topic)
            if content.startswith("Error generating content:"):
                raise ValueError(f"Content generation failed: {content}")
            log_data["content_generated"] = True

            # 2. Export Content
            filename_prefix = re.sub(
                r"[^a-z0-9_]", "", topic.lower().replace(" ", "_")
            )[:30]
            filepath = self.export_content(
                content, filename_prefix=filename_prefix, format=format
            )
            if filepath is None:
                raise IOError("Content export failed.")
            log_data["filepath"] = filepath  # Log the actual filepath string

            # 3. Publish Content
            if target.lower() == "github":
                final_link = self.publish_to_github(filepath)
            elif target.lower() == "gumroad":
                final_link = self.publish_to_gumroad(filepath, title=topic, price=price)
            else:
                raise ValueError(
                    f"Unsupported publish target: '{target}'. Use 'github' or 'gumroad'."
                )
            log_data["final_link"] = final_link

            status = "success"
            message = f"Content generated, exported to '{os.path.basename(filepath)}', and published (simulated) to {target}."
            log_data["status"] = status

        except Exception as e:
            logger.error(
                f"[{SKILL_NAME}] Error during execute for topic '{topic}': {e}",
                exc_info=True,
            )
            error_details = str(e)
            log_data["status"] = "error"
            log_data["error"] = error_details
            status = "error"
            message = f"Error during auto-publication: {error_details}"

        finally:
            # 4. Log Result using helper
            log_filepath_str = self._log_execution_result(log_data, status)
            if log_filepath_str is None and status == "success":
                message += " (Warning: Failed to write log file)"
            elif log_filepath_str is None and status == "error":
                # Don't overwrite primary error message if logging also fails
                logger.error(
                    f"[{SKILL_NAME}] Primary operation failed AND log writing failed."
                )
                message = f"Primary operation error: {error_details}. Additionally, log writing failed."

        # Return standardized response
        return create_skill_response(
            status=status,
            action=f"{SKILL_NAME}_completed",
            data={
                "topic": topic,
                "format": format,
                "target": target,
                "filepath": filepath,  # Return the path returned by export
                "link": final_link,
                "log_file": log_filepath_str,  # Path to the log file
                "price": price if target.lower() == "gumroad" else None,
            },
            message=message,
            error_details=error_details,  # Pass original error details if any
        )


# Example usage (if run directly)
# if __name__ == '__main__':
#     publisher = AutoPublisherSkill()
#     result = publisher.execute(
#         topic="Test Topic Direct Run",
#         format="markdown",
#         target="gumroad",
#         price=1.99
#     )
#     print(json.dumps(result, indent=2))
#
#     result_gh = publisher.execute(
#         topic="GitHub Test Topic",
#         format="txt",
#         target="github"
#     )
