# This file makes the 'perception' directory a Python package. 

# Import modules within this sub-package to ensure their skills are registered
from . import visual_perception
from . import describe_image_blip
from . import ocr_extract

"""Perception skills that analyze the environment."""

import logging

logger = logging.getLogger(__name__)

logger.debug("Perception skills package initialized.") 