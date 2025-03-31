# skills/publisher.py
import os
import json
import logging
from datetime import datetime
import re
from pathlib import Path

from core.tools import skill
from core.skills_utils import create_skill_response, WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# Define o nome da skill de forma consistente
SKILL_NAME = "publisher"

@skill(
    name=SKILL_NAME,
    description="Publishes digital products (simulated) to platforms like Gumroad.",
    parameters={
        "filepath": (str, "Path to the file content to be published."),
        "title": (str, "Title of the product."),
        "price": (float, "Price of the product (e.g., 0.99)."),
        "target": (str, "Target platform (currently only 'gumroad' simulation supported).")
    }
)
class PublisherSkill:
    """
    Skill to handle publishing content to various platforms.
    Currently simulates publishing to Gumroad by saving metadata.
    """

    def __init__(self, output_base_dir="output/published"):
        self.output_dir = Path(WORKSPACE_ROOT) / output_base_dir
        # Ensure the base output directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output directory exists: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create or access output directory {self.output_dir}: {e}", exc_info=True)
            # Skill can still proceed, but saving might fail later
            # Consider raising an error during init if directory is critical

    def publish_to_gumroad(self, filepath: str, title: str, price: float) -> dict:
        """
        Simulates publishing a product file to Gumroad.

        Creates a JSON file in `output/published/` containing metadata
        about the simulated publication.

        Args:
            filepath (str): Relative path within the workspace to the product file.
            title (str): The title for the product.
            price (float): The price for the product.

        Returns:
            dict: A dictionary with the status and details of the simulation.
        """
        logger.info(f"Executing {SKILL_NAME} skill: publish_to_gumroad (simulation)")
        logger.info(f"Attempting to publish file: '{filepath}' with title '{title}' for ${price:.2f}")

        # --- Input Validation ---
        # Validate filepath relative to WORKSPACE_ROOT
        try:
            if not isinstance(filepath, str) or not filepath:
                raise ValueError("File path cannot be empty.")
            if os.path.isabs(filepath) or ".." in filepath:
                raise ValueError("Path must be relative within the workspace and cannot contain '..'.")

            abs_filepath = (Path(WORKSPACE_ROOT) / filepath).resolve()

            if not str(abs_filepath).startswith(str(Path(WORKSPACE_ROOT).resolve())):
                raise ValueError("Path resolves outside the workspace.")

            if not abs_filepath.is_file():
                return create_skill_response(
                    status="error",
                    action=f"{SKILL_NAME}_failed_file_not_found",
                    error_details=f"Input file not found at resolved path: {abs_filepath}",
                    message=f"Error: The file specified ('{filepath}') does not exist or is not a file."
                )
            logger.debug(f"Validated input file path: {abs_filepath}")

        except ValueError as e:
             logger.warning(f"Path validation failed for '{filepath}': {e}")
             return create_skill_response(status="error", action=f"{SKILL_NAME}_failed_invalid_path", error_details=str(e), message=f"Invalid file path: {e}")
        except Exception as e:
            logger.error(f"Unexpected error validating path '{filepath}': {e}", exc_info=True)
            return create_skill_response(status="error", action=f"{SKILL_NAME}_failed_internal_error", error_details=str(e), message=f"Internal error validating path: {e}")

        # Validate title and price (basic)
        if not isinstance(title, str) or not title:
            return create_skill_response(status="error", action=f"{SKILL_NAME}_failed_invalid_param", error_details="Title cannot be empty.", message="Invalid product title provided.")
        if not isinstance(price, (int, float)) or price < 0:
            return create_skill_response(status="error", action=f"{SKILL_NAME}_failed_invalid_param", error_details="Price must be a non-negative number.", message="Invalid product price provided.")

        # --- Simulation Logic ---
        timestamp = datetime.now().isoformat()
        # Create a safe filename from the title
        safe_title = re.sub(r'[^a-z0-9_\-]', '', title.lower().replace(' ', '_'))[:50]
        output_filename = f"gumroad_sim_{safe_title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        output_filepath = self.output_dir / output_filename

        # Simulate Gumroad product URL
        # Use a generic username if GITHUB_USERNAME isn't set/relevant
        gumroad_user = os.getenv("GUMROAD_USERNAME", "yourusername") # Or a dedicated Gumroad user env var
        slug = safe_title or f"product-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        simulated_url = f"https://{gumroad_user}.gumroad.com/l/{slug}"

        publication_data = {
            "simulation_timestamp": timestamp,
            "action": "publish_to_gumroad_simulation",
            "source_filepath": str(abs_filepath), # Store absolute path for clarity in log
            "original_relative_path": filepath,
            "product_title": title,
            "product_price": price,
            "target_platform": "gumroad",
            "simulated_product_url": simulated_url,
            "status": "simulated_success"
        }

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(publication_data, f_out, indent=4)
            logger.info(f"Successfully saved Gumroad publication simulation data to: {output_filepath}")

            return create_skill_response(
                status="success",
                action=f"{SKILL_NAME}_gumroad_simulation_saved",
                data={
                    "simulation_log_file": str(output_filepath),
                    "simulated_product_url": simulated_url,
                    "title": title,
                    "price": price
                },
                message=f"Gumroad publication simulated successfully. Details saved to {output_filename}."
            )

        except IOError as e:
            logger.error(f"Failed to write simulation JSON to {output_filepath}: {e}", exc_info=True)
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_write_log",
                error_details=str(e),
                message=f"Error saving simulation details: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during simulation saving for '{title}': {e}", exc_info=True)
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_internal_error",
                error_details=str(e),
                message=f"Internal error during simulation saving: {e}"
            )

    # Placeholder for other potential publishing targets
    # def publish_to_etsy(self, ...):
    #     pass

    # The main execution method could dispatch based on 'target'
    def execute(self, filepath: str, title: str, price: float, target: str = "gumroad", agent_history: list | None = None) -> dict:
        """
        Main entry point for the PublisherSkill.
        Dispatches the publishing task to the appropriate method based on the target.
        """
        logger.info(f"PublisherSkill execute called for target: {target}")
        if target.lower() == "gumroad":
            return self.publish_to_gumroad(filepath, title, price)
        else:
            logger.warning(f"Unsupported publishing target: '{target}'")
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_unsupported_target",
                error_details=f"Target '{target}' is not supported.",
                message=f"Publishing target '{target}' not implemented. Only 'gumroad' simulation is available."
            )

# Example Usage (if run directly)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # Create a dummy file to publish
#     dummy_dir = Path(WORKSPACE_ROOT) / "output" / "publisher_test"
#     dummy_dir.mkdir(parents=True, exist_ok=True)
#     dummy_file = dummy_dir / "my_ebook.txt"
#     with open(dummy_file, "w") as f:
#         f.write("This is the content of the dummy ebook.")
#
#     publisher = PublisherSkill()
#     result = publisher.execute(
#         filepath="output/publisher_test/my_ebook.txt", # Relative path
#         title="My Awesome Ebook",
#         price=4.99,
#         target="gumroad"
#     )
#     print("\n--- Gumroad Simulation Result ---")
#     print(json.dumps(result, indent=2))
#
#     result_other = publisher.execute(
#         filepath="output/publisher_test/my_ebook.txt",
#         title="My Awesome Ebook",
#         price=4.99,
#         target="etsy"
#     )
#     print("\n--- Unsupported Target Result ---")
#     print(json.dumps(result_other, indent=2))
#
#     # Test invalid path
#     result_invalid = publisher.execute(
#         filepath="../invalid/path.txt",
#         title="Invalid Path Test",
#         price=1.00,
#         target="gumroad"
#     )
#     print("\n--- Invalid Path Result ---")
#     print(json.dumps(result_invalid, indent=2))
