"""Skill to describe an image using the BLIP model."""

from pathlib import Path
from typing import Annotated, Any
import logging

from PIL import Image
# Use specific BLIP classes instead of AutoModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# <<< IMPORT skill decorator from core.tools >>>
from a3x.core.tools import skill


# <<< REMOVE Class Wrapper >>>
# class DescribeImageBlipSkill:
#     """Skill to generate a caption for an image using Salesforce BLIP."""

@skill(
    # <<< CHANGE parameters format to Dict[str, tuple] >>>
    name="describe_image_blip", # Need to explicitly add name here
    description="Generates a textual description (caption) for the provided image file using the BLIP model.",
    parameters={
        "image_path": (Path, ...), # Ellipsis indicates required
    }
    # parameters=[
    #     Parameter(name="image_path", type="Path", description="The path to the image file."),
    # ],
    # output_type="str", # Decorator doesn't seem to use output_type
    # examples=[
    #     'await describe_image_blip(image_path=Path("screenshots/latest.png"))'
    # ] # Decorator doesn't seem to use examples
)
async def describe_image_blip(
    # <<< REMOVE self parameter >>>
    # self,
    # <<< CHANGE Context type hint >>>
    # ctx: Context,
    ctx: Any, # Use Any for now
    image_path: Path # Keep the Path type hint
) -> str:
    """Describes the image found at the given path using BLIP."""
    # <<< Need to access logger from ctx, assuming it has a .log attribute >>>
    logger = ctx.log if hasattr(ctx, 'log') else logging.getLogger(__name__)

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

    except ImportError as e:
         logger.error(f"Error importing BlipProcessor/BlipForConditionalGeneration: {e}. Transformers installed?")
         return f"Error: Missing BLIP classes from transformers ({e})."
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