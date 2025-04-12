import asyncio
import base64
import httpx
import logging
import os
import json
from slugify import slugify
from pathlib import Path
from typing import Optional, Dict, Any
from a3x.core.skills import skill
from PIL import Image
from a3x.core.config import SD_API_URL_BASE, PROJECT_ROOT
import io

# Configure logger for the skill
logger = logging.getLogger(__name__)

# SD API Configuration (adjust if your setup differs)
SD_API_URL_BASE = SD_API_URL_BASE
TXT2IMG_ENDPOINT = f"{SD_API_URL_BASE}/sdapi/v1/txt2img"

# Output directory relative to project root
OUTPUT_DIR = Path("assets/covers")

# Default parameters for image generation
DEFAULT_PAYLOAD = {
    "prompt": "", # Will be replaced
    "negative_prompt": "text, letters, words, watermark, signature, low quality, blurry, deformed, bad anatomy, multiple limbs",
    "seed": -1,
    "steps": 30,
    "cfg_scale": 7,
    "width": 512,
    "height": 768, # Common ebook cover aspect ratio
    "sampler_name": "Euler a",
    "n_iter": 1,
    "batch_size": 1,
    # "restore_faces": True, # Optional: Depending on content
    # "tiling": False, # Optional
    "override_settings": {"sd_model_checkpoint": "anything-v3-fp16-pruned.safetensors"}, # Specify the model
}

@skill(
    name="generate_ebook_cover",
    description="Gera uma imagem de capa para um ebook com base no título, descrição e público-alvo.",
    parameters={
        "title": (str, ...),
        "description": (str, ...),
        "target_audience": (str, ...),
        "ctx": (Context, None)
    }
)
async def generate_ebook_cover(title: str, description: str, target_audience: str, ctx: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates an ebook cover image using the Stable Diffusion WebUI API based on the
    provided title, description, and target audience.

    Args:
        title: The title of the ebook.
        description: A brief description of the ebook content.
        target_audience: The intended audience for the ebook.

    Returns:
        The path to the saved cover image file or an error message.
    """
    if not SD_API_URL_BASE:
        return "Error: SD_API_URL_BASE is not configured in core/config.py"

    # Construct a dynamic prompt (simplified for testing)
    # prompt = (
    #     f"Ebook cover for '{title}', described as '{description}'. " 
    #     f"Target audience: {target_audience}. " 
    #     f"Style: professional, eye-catching, high-resolution digital art, " 
    #     f"book cover design elements, clear title text if possible"
    # )
    # negative_prompt = "low quality, blurry, text illegible, multiple books, deformed, watermark, signature, amateur"

    # Minimal safe payload for testing
    payload = {
        "prompt": f"ebook cover, title: {title}, style: professional digital art", # Simplified prompt
        "negative_prompt": "low quality, blurry, watermark, text, signature, words, letters",
        "width": 512,
        "height": 768,
        "steps": 20,
        "cfg_scale": 6, # Lowered CFG scale
        "sampler_name": "Euler a", # Changed from sampler_index
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": "anything-v3-fp16-pruned.safetensors" 
        }
    }

    endpoint = f"{SD_API_URL_BASE}/sdapi/v1/txt2img"
    output_dir = "assets/covers"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = slugify(title)
    output_path = os.path.join(output_dir, f"{base_filename}.jpg")

    logger.info(f"Generating ebook cover for '{title}'")
    logger.info(f"Prompt: {payload['prompt']}")
    logger.info(f"Sending request to: {endpoint}")

    try:
        async with httpx.AsyncClient(timeout=180.0) as client: # Increased timeout
            response = await client.post(endpoint, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            if 'images' in result and result['images']:
                image_data = base64.b64decode(result['images'][0])
                image = Image.open(io.BytesIO(image_data))
                image.save(output_path, "JPEG")
                logger.info(f"Ebook cover saved successfully to: {output_path}")
                return output_path
            else:
                error_msg = "Error: No image data found in the API response."
                logger.error(f"{error_msg} Response: {result}")
                return error_msg

    except httpx.TimeoutException:
        error_msg = "Error: Timeout occurred while contacting the Stable Diffusion API."
        logger.error(error_msg)
        return error_msg
    except httpx.RequestError as exc:
        error_msg = f"Error: An error occurred while requesting {exc.request.url!r}: {exc}"
        logger.error(error_msg)
        return error_msg
    except httpx.HTTPStatusError as exc:
        error_msg = f"Error: HTTP error {exc.response.status_code} while requesting {exc.request.url!r}. Response: {exc.response.text}"
        logger.error(error_msg)
        return error_msg
    except (IOError, OSError) as exc:
        error_msg = f"Error: Failed to save image to {output_path}: {exc}"
        logger.error(error_msg)
        return error_msg
    except Exception as exc:
        error_msg = f"Error: An unexpected error occurred: {exc}"
        logger.exception(error_msg) # Log full traceback for unexpected errors
        return error_msg

# Example for direct testing (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def run_test():
        try:
            image_path = await generate_ebook_cover(
                title="Zero to OnlyFans with AI",
                description="How to make money with AI-generated NSFW content",
                target_audience="indie entrepreneurs"
            )
            print(f"Test successful. Image saved to: {image_path}")
        except Exception as e:
            print(f"Test failed: {e}")
    asyncio.run(run_test()) 