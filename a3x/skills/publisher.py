# skills/publisher.py
import os
import json
import logging
from datetime import datetime
import re
from pathlib import Path

from a3x.core.tools import skill
from a3x.core.skills_utils import create_skill_response
from a3x.core.config import PROJECT_ROOT as WORKSPACE_ROOT
from a3x.core.validators import validate_workspace_path

# from a3x.core.config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_USERNAME, GUMROAD_API_KEY, GUMROAD_PRODUCT_ID # COMMENTED OUT

logger = logging.getLogger(__name__)

# Define o nome da skill de forma consistente
SKILL_NAME = "publisher"


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
            logger.error(
                f"Failed to create or access output directory {self.output_dir}: {e}",
                exc_info=True,
            )
            # Skill can still proceed, but saving might fail later
            # Consider raising an error during init if directory is critical

    @validate_workspace_path(
        arg_name="filepath", check_existence=True, target_type="file"
    )
    def publish_to_gumroad(
        self,
        filepath: str,
        title: str,
        price: float,
        resolved_path: Path,
        original_path_str: str,
        **kwargs,
    ) -> dict:
        """
        Simulates publishing a product file to Gumroad.
        Path validation handled by @validate_workspace_path.

        Creates a JSON file in `output/published/` containing metadata
        about the simulated publication.

        Args:
            filepath (str): Original relative path (passed by decorator as original_path_str).
            title (str): The title for the product.
            price (float): The price for the product.
            resolved_path (Path): Injected by decorator - validated absolute Path object.
            original_path_str (str): Injected by decorator - original path string.
            **kwargs: Catches any other args passed by the decorator or caller.

        Returns:
            dict: A dictionary with the status and details of the simulation.
        """
        # Use injected args
        validated_filepath = original_path_str
        abs_filepath = resolved_path

        logger.info(f"Executing {SKILL_NAME} skill: publish_to_gumroad (simulation)")
        logger.info(
            f"Attempting to publish file: '{validated_filepath}' (resolved: {abs_filepath}) with title '{title}' for ${price:.2f}"
        )

        # --- Input Validation (Path validation removed, handled by decorator) ---
        # Validate title and price (basic)
        if not isinstance(title, str) or not title:
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_invalid_param",
                error_details="Title cannot be empty.",
                message="Invalid product title provided.",
            )
        if not isinstance(price, (int, float)) or price < 0:
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_invalid_param",
                error_details="Price must be a non-negative number.",
                message="Invalid product price provided.",
            )

        # --- Simulation Logic ---
        timestamp = datetime.now().isoformat()
        # Create a safe filename from the title
        safe_title = re.sub(r"[^a-z0-9_\-]", "", title.lower().replace(" ", "_"))[:50]
        output_filename = (
            f"gumroad_sim_{safe_title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        output_filepath = self.output_dir / output_filename

        # Simulate Gumroad product URL
        gumroad_user = os.getenv("GUMROAD_USERNAME", "yourusername")
        slug = safe_title or f"product-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        simulated_url = f"https://{gumroad_user}.gumroad.com/l/{slug}"

        publication_data = {
            "simulation_timestamp": timestamp,
            "action": "publish_to_gumroad_simulation",
            "source_filepath": str(
                abs_filepath
            ),  # Store absolute path for clarity in log
            "original_relative_path": validated_filepath,
            "product_title": title,
            "product_price": price,
            "target_platform": "gumroad",
            "simulated_product_url": simulated_url,
            "status": "simulated_success",
        }

        try:
            with open(output_filepath, "w", encoding="utf-8") as f_out:
                json.dump(publication_data, f_out, indent=4)
            logger.info(
                f"Successfully saved Gumroad publication simulation data to: {output_filepath}"
            )

            return create_skill_response(
                status="success",
                action=f"{SKILL_NAME}_gumroad_simulation_saved",
                data={
                    "simulation_log_file": str(output_filepath),
                    "simulated_product_url": simulated_url,
                    "title": title,
                    "price": price,
                },
                message=f"Gumroad publication simulated successfully. Details saved to {output_filename}.",
            )

        except IOError as e:
            logger.error(
                f"Failed to write simulation JSON to {output_filepath}: {e}",
                exc_info=True,
            )
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_write_log",
                error_details=str(e),
                message=f"Error saving simulation details: {e}",
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during simulation saving for '{title}': {e}",
                exc_info=True,
            )
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_internal_error",
                error_details=str(e),
                message=f"Internal error during simulation saving: {e}",
            )

    # Placeholder for other potential publishing targets
    # def publish_to_etsy(self, ...):
    #     pass

    @skill(
        name=SKILL_NAME,
        description="Publishes digital products (simulated) to platforms like Gumroad.",
        parameters={
            # Parameters defined here are for the entry point 'execute' method.
            # The decorator's path validation happens in the dispatched method.
            "filepath": (str, ...),  # User provides this
            "title": (str, "Title of the product."),
            "price": (float, 0.99),  # Example default
            "target": (str, "gumroad"),
        },
    )
    def execute(
        self, filepath: str, title: str, price: float, target: str = "gumroad"
    ) -> dict:
        """
        Main entry point for the PublisherSkill.
        Dispatches the publishing task to the appropriate method based on the target.
        Path validation is handled by the dispatched method's decorator.
        """
        logger.info(f"PublisherSkill execute called for target: {target}")
        if target.lower() == "gumroad":
            # Pass filepath directly; the decorator on publish_to_gumroad will handle it.
            # It expects 'filepath' as the key for the path string.
            # Since publish_to_gumroad accepts **kwargs, injected parameters will pass through.
            # No need to explicitly pass **kwargs from here if execute doesn't receive extra ones.
            # OLD: return self.publish_to_gumroad(filepath=filepath, title=title, price=price)
            # WORKAROUND: Pass None for decorator-injected args, as the decorator will run again.
            return self.publish_to_gumroad(
                filepath=filepath,
                title=title,
                price=price,
                resolved_path=None,  # Dummy value
                original_path_str=None,  # Dummy value
            )
        else:
            logger.warning(f"Unsupported publishing target: '{target}'")
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_unsupported_target",
                error_details=f"Target '{target}' is not supported.",
                message=f"Publishing target '{target}' not implemented. Only 'gumroad' simulation is available.",
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
