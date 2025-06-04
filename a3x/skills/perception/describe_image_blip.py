"""Skill to describe an image using the BLIP model."""

from pathlib import Path
from typing import Annotated, Any, Dict, Optional
import logging
import torch

from PIL import Image
# Use specific BLIP classes instead of AutoModel

from a3x.core.skills import skill
from a3x.core.config import PROJECT_ROOT
from a3x.core.context import Context


# <<< REMOVE Class Wrapper >>>
# class DescribeImageBlipSkill:
#     """Skill to generate a caption for an image using Salesforce BLIP."""

@skill(
    name="describe_image_blip",
    description="Analyzes an image using BLIP and provides a textual description.",
    parameters={
        "context": {"type": Context, "description": "Execution context for logger and potentially model path."},
        "image_path": {"type": str, "description": "The file path to the image to analyze."},
        "prompt": {"type": Optional[str], "default": None, "description": "Optional text prompt to guide the description."}
    }
)
async def describe_image_blip(
    context: Context,
    image_path: str,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Describes an image using the BLIP model via transformers."""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError as e:
        logging.getLogger(__name__).error(
            f"Error importing transformers: {e}"
        )
        return {
            "status": "error",
            "message": f"Transformers not available: {e}"
        }
    # <<< Need to access logger from ctx, assuming it has a .log attribute >>>
    logger = context.log if hasattr(context, 'log') else logging.getLogger(__name__)

    if not isinstance(image_path, Path):
         # The executor might pass the raw string from JSON, ensure it's a Path
         try:
             image_path = Path(image_path)
         except Exception as path_err:
             logger.error(f"Invalid image_path input: {image_path}. Error: {path_err}")
             return f"Error: Invalid image_path type provided: {type(image_path)}"

    if not image_path.exists():
        logger.error(f"Image file not found at {image_path}")
        return f"Error: Image file not found at {image_path}"
    if not image_path.is_file():
        logger.error(f"Path {image_path} is not a file.")
        return f"Error: Path {image_path} is not a file."

    model_name = "Salesforce/blip-image-captioning-base"
    caption = "Error: Failed to generate caption." # Default error message

    try:
        # <<< REMOVE ctx.llm_call wrapper for direct skill test >>>
        # if hasattr(ctx, 'llm_call') and callable(ctx.llm_call):
        #     async with ctx.llm_call(
        #         "blip_captioning",
        #         parameters={'image_path': str(image_path), 'model_name': model_name}
        #     ) as call_details:
        #         logger.info(f"Loading BLIP model '{model_name}' and processor...")
        #         # ... [rest of logic inside with block] ...
        #         call_details.result = caption
        #         call_details.prompt = "[BLIP Image Input]"
        # else:
        # Fallback if ctx.llm_call is not available (e.g., during direct testing w/ basic context)
        # <<< ALWAYS run this block directly for now >>>
        # logger.warning("Context object does not have 'llm_call'. Proceeding without LLM call wrapping.")
        logger.info(f"Loading BLIP model '{model_name}' and processor...")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        logger.info("Model and processor loaded.")
        logger.info(f"Opening image: {image_path}")
        raw_image = Image.open(image_path).convert('RGB')
        logger.info("Processing image and generating caption...")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"Generated caption: {caption}")

    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        logger.error(f"Error during BLIP image captioning for {image_path}: {e}", exc_info=True)
        return f"Error generating caption: {e}"

    # Return the caption directly, not a dict (tool executor expects skill to return dict)
    # return caption
    # <<< CHANGE: Return standardized dict response >>>
    return {
        "status": "success",
        "action": "image_described",
        "data": {"description": caption},
        "message": f"Successfully described image: {image_path}"
    } 