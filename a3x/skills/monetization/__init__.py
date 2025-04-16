import logging

logger = logging.getLogger(__name__)

# Import skill modules to trigger registration
# try:
#     from . import persona_generator
#     logger.debug("Imported persona_generator skill.")
# except ImportError as e:
#     logger.warning(f"Could not import persona_generator: {e}")

try:
    from . import nsfw_image_generator
    logger.debug("Imported nsfw_image_generator skill.")
except ImportError as e:
    logger.warning(f"Could not import nsfw_image_generator: {e}")

logger.info("Monetization skills package initialized.") 