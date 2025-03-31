import os
import logging
import re
import json
from datetime import datetime
from typing import Dict, Any
# from core.skill_registry import register_skill # REMOVED
from core.tools import skill # ADDED
from core.skills_utils import create_skill_response
from core.llm_interface import call_llm
# from core.config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_USERNAME, GUMROAD_API_KEY, GUMROAD_PRODUCT_ID # REMOVED - Variables not defined in config.py

# Setup logging
log_dir = "output/autopublisher/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "autopublisher.log") # Optional: file handler for general logs

# Configure logger for the skill
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set desired level

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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Load attempt, will be None if not set
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GUMROAD_API_KEY = os.getenv("GUMROAD_API_KEY")
GUMROAD_PRODUCT_ID = os.getenv("GUMROAD_PRODUCT_ID")

# Define the skill name consistently
SKILL_NAME = "auto_publisher"

@skill(
    name=SKILL_NAME,
    description="Generates content based on a topic, exports it, and publishes (simulated) to a target platform (GitHub/Gumroad).",
    parameters={
        "topic": (str, "The topic to generate content about."),
        "format": (str, "The format to export the content (e.g., 'markdown', 'txt'). Default: 'markdown'."),
        "target": (str, "The platform to publish to ('github' or 'gumroad').")
    },
    output_type="dict", # Returns a dictionary with status, message, link, etc.
    category="Monetization"
)
class AutoPublisherSkill:
    def __init__(self, output_dir="output/autopublisher"):
        self.output_dir = output_dir
        # Ensure base output dir exists, log dir is handled separately above
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = log_dir # Store log directory path

    def generate_content(self, topic: str) -> str:
        """
        Generates textual content on the given topic using the LLM (currently simulated).
        """
        prompt = f"Write a short blog post about {topic}."
        logger.info(f"[{SKILL_NAME}] Generating content with prompt: '{prompt[:50]}...'")
        try:
            # TEMPORARY: Simulate LLM response for now
            logger.warning(f"[{SKILL_NAME}] Using simulated LLM response for content generation.")
            generated_text = f"Simulated blog post about {topic}. Lorem ipsum dolor sit amet..."
            
            # TODO: Implement proper async handling for call_llm if required

            logger.info(f"[{SKILL_NAME}] Content generated successfully.")
            return generated_text
        except Exception as e:
            logger.exception(f"[{SKILL_NAME}] Error during content generation:")
            # Return error within the string, the orchestrator will handle the failure status
            return f"Error generating content: {e}"

    def export_content(self, content: str, filename_prefix: str = "content", format: str = "markdown") -> str | None:
        """
        Exports the content to the specified format (.md or .txt for now). Returns filepath or None on error.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            format_lower = format.lower()
            extension = ".md" if format_lower == "markdown" else ".txt" # Basic extension mapping
            if format_lower not in ["markdown", "txt"]:
                 logger.warning(f"[{SKILL_NAME}] Unsupported export format '{format}'. Defaulting to '.txt'.")
                 extension = ".txt"

            filename = f"{filename_prefix}_{timestamp}{extension}"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"[{SKILL_NAME}] Content exported to: {filepath}")
            return filepath
        except Exception as e:
            logger.exception(f"[{SKILL_NAME}] Error exporting content:")
            return None

    def publish_to_github(self, filepath: str) -> str:
        """
        Simulates publishing content to GitHub. Returns a simulated URL.
        """
        logger.info(f"[SIMULATION][{SKILL_NAME}] Would publish {filepath} to GitHub repository.")
        # Assume GITHUB_USERNAME and GITHUB_REPO are set for a more realistic simulation
        user = GITHUB_USERNAME or "your_username"
        repo = GITHUB_REPO or "your_repo"
        return f"https://github.com/{user}/{repo}/blob/main/{os.path.basename(filepath)}" # Simplified URL

    def publish_to_gumroad(self, filepath: str, title: str, price: float = 0.99) -> str:
        """
        Simulates publishing a product to Gumroad. Returns a simulated URL.
        Includes title and price in simulation.
        """
        logger.info(f"[SIMULATION][{SKILL_NAME}] Would publish {filepath} to Gumroad as product '{title}' for ${price:.2f}.")
        # Simulate a unique product slug based on title
        slug = title.lower().replace(" ", "-").replace(r'[^a-z0-9-]', '')[:50]
        return f"https://{GITHUB_USERNAME or 'yourusername'}.gumroad.com/l/{slug}" # Simulated Gumroad link structure

    def generate_and_publish_from_niche(self, topic: str, format: str = "markdown", target: str = "gumroad", price: float = 0.99) -> dict:
        """
        Orchestrates content generation, export, and publication for a given niche/topic.

        Args:
            topic (str): The topic to generate content about.
            format (str, optional): The export format ('markdown' or 'txt'). Defaults to "markdown".
            target (str, optional): The publishing platform ('github' or 'gumroad'). Defaults to "gumroad".
            price (float, optional): The price if publishing to Gumroad. Defaults to 0.99.

        Returns:
            dict: A dictionary containing the execution status, messages, and results.
        """
        logger.info(f"Starting generation and publication for topic: '{topic}', format: '{format}', target: '{target}'")
        status = "error" # Default status
        message = "Orchestration failed."
        content = ""
        filepath = None
        final_link = None
        log_filepath = None # Initialize log_filepath

        # Simplified log data structure, more focused on final outcome
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "format": format,
            "target": target,
            "price": price if target == "gumroad" else None, # Log price only for Gumroad
            "status": status,
            "content_generated": False,
            "filepath": None,
            "final_link": None,
            "error": None
        }

        try:
            # 1. Generate Content
            logger.info(f"[{SKILL_NAME}] Generating content for topic: '{topic}'...")
            content = self.generate_content(topic)
            if content.startswith("Error generating content:"): # Check if generation failed
                raise ValueError(f"Content generation failed: {content}")
            logger.info(f"[{SKILL_NAME}] Content generated successfully.")
            log_data["content_generated"] = True

            # 2. Export Content
            # Basic filename sanitization from topic
            filename_prefix = re.sub(r'[^a-z0-9_]', '', topic.lower().replace(" ", "_"))[:30]
            logger.info(f"[{SKILL_NAME}] Exporting content to format: '{format}' with prefix '{filename_prefix}'...")
            filepath = self.export_content(content, filename_prefix=filename_prefix, format=format)
            if filepath is None:
                 raise IOError("Content export failed.")
            logger.info(f"[{SKILL_NAME}] Content exported successfully to: {filepath}")
            log_data["filepath"] = filepath

            # 3. Publish Content
            logger.info(f"[{SKILL_NAME}] Publishing content to target: '{target}'...")
            if target == "github":
                final_link = self.publish_to_github(filepath)
            elif target == "gumroad":
                # Use the topic as the default title for Gumroad product
                final_link = self.publish_to_gumroad(filepath, title=topic, price=price)
            else:
                raise ValueError(f"Unsupported publish target: '{target}'. Use 'github' or 'gumroad'.")
            logger.info(f"[{SKILL_NAME}] Content published successfully (simulated). Link: {final_link}")
            log_data["final_link"] = final_link
            status = "success" # Changed from 'published' for consistency
            message = f"Content generated, exported to '{os.path.basename(filepath)}', and published (simulated) to {target}."
            log_data["status"] = status

        except Exception as e:
            logger.error(f"[{SKILL_NAME}] Error during generate_and_publish_from_niche for topic '{topic}': {e}", exc_info=True)
            log_data["status"] = "error"
            log_data["error"] = str(e)
            status = "error" # Ensure status reflects failure
            message = f"Error during auto-publication: {e}"

        finally:
            # 4. Log Result to JSON (Simplified logging)
            log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{status}.json"
            log_filepath = os.path.join(self.log_dir, log_filename)
            try:
                with open(log_filepath, 'w', encoding='utf-8') as f_log:
                    json.dump(log_data, f_log, indent=4)
                logger.info(f"[{SKILL_NAME}] Operation log saved to: {log_filepath}")
            except Exception as log_e:
                logger.error(f"[{SKILL_NAME}] Failed to write JSON log file to {log_filepath}: {log_e}", exc_info=True)
                # Append log writing error to message if primary operation succeeded
                if status == "success":
                    message += f" (Warning: Failed to write log file: {log_e})"
                # Don't overwrite primary error if that occurred

        # Use create_skill_response for consistent output
        return create_skill_response(
            status=status,
            action=f"{SKILL_NAME}_completed", # Generic action name
            data={
                "topic": topic,
                "format": format,
                "target": target,
                "filepath": filepath,
                "link": final_link,
                "log_file": log_filepath,
                "price": price if target == "gumroad" else None
            },
            message=message,
            error_details=log_data["error"] # Pass error message back if any
        )

# Example usage (if run directly)
# if __name__ == '__main__':
#     publisher = AutoPublisherSkill()
#     result = publisher.generate_and_publish_from_niche(
#         topic="Test Topic Direct Run",
#         format="markdown",
#         target="gumroad",
#         price=1.99
#     )
#     print(json.dumps(result, indent=2))
#
#     result_gh = publisher.generate_and_publish_from_niche(
#         topic="GitHub Test Topic",
#         format="txt",
#         target="github"
#     )
#     print(json.dumps(result_gh, indent=2))