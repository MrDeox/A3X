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
from a3x.core.config import PROJECT_ROOT
import io
from a3x.core.context import Context

# Configure logger for the skill
logger = logging.getLogger(__name__)

# SD API Configuration (adjust if your setup differs)
# SD_API_URL_BASE = SD_API_URL_BASE # <<< COMENTADO - CAUSA NameError
TXT2IMG_ENDPOINT = "/sdapi/v1/txt2img" # Endpoint relativo

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
    description="Gera uma imagem de capa de e-book usando um modelo de difusão estável.",
    parameters={
        "title": {"type": str, "description": "O título do e-book para a capa."},
        "description": {"type": str, "description": "Uma breve descrição do e-book para inspirar a capa."},
        "target_audience": {"type": str, "description": "O público-alvo do e-book."}
    }
)
async def generate_ebook_cover(title: str, description: str, target_audience: str, ctx: Optional[Any] = None) -> str:
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
    # <<< RETORNANDO ERRO TEMPORARIAMENTE >>>
    logger.error("generate_ebook_cover skill is temporarily disabled due to missing SD_API_URL_BASE configuration.")
    return "Error: A configuração da URL base da API do Stable Diffusion (SD_API_URL_BASE) está faltando ou comentada. Skill desabilitada."

    # --- Lógica original comentada ---
    # if not SD_API_URL_BASE:
    #     return "Error: SD_API_URL_BASE is not configured in core/config.py"
    #
    # payload = {
    #     "prompt": f"ebook cover, title: {title}, style: professional digital art",
    #     "negative_prompt": "low quality, blurry, watermark, text, signature, words, letters",
    #     "width": 512,
    #     "height": 768,
    #     "steps": 20,
    #     "cfg_scale": 6,
    #     "sampler_name": "Euler a",
    #     "seed": -1,
    #     "override_settings": {
    #         "sd_model_checkpoint": "anything-v3-fp16-pruned.safetensors"
    #     }
    # }
    #
    # endpoint = f"{SD_API_URL_BASE}{TXT2IMG_ENDPOINT}"
    # output_dir = "assets/covers"
    # os.makedirs(output_dir, exist_ok=True)
    # base_filename = slugify(title)
    # output_path = os.path.join(output_dir, f"{base_filename}.jpg")
    #
    # logger.info(f"Generating ebook cover for '{title}'")
    # logger.info(f"Prompt: {payload['prompt']}")
    # logger.info(f"Sending request to: {endpoint}")
    #
    # try:
    #     async with httpx.AsyncClient(timeout=180.0) as client:
    #         response = await client.post(endpoint, json=payload)
    #         response.raise_for_status()
    #
    #         result = response.json()
    #
    #         if 'images' in result and result['images']:
    #             image_data = base64.b64decode(result['images'][0])
    #             image = Image.open(io.BytesIO(image_data))
    #             image.save(output_path, "JPEG")
    #             logger.info(f"Ebook cover saved successfully to: {output_path}")
    #             return output_path
    #         else:
    #             error_msg = "Error: No image data found in the API response."
    #             logger.error(f"{error_msg} Response: {result}")
    #             return error_msg
    #
    # except httpx.TimeoutException:
    #     error_msg = "Error: Timeout occurred while contacting the Stable Diffusion API."
    #     logger.error(error_msg)
    #     return error_msg
    # except httpx.RequestError as exc:
    #     error_msg = f"Error: An error occurred while requesting {exc.request.url!r}: {exc}"
    #     logger.error(error_msg)
    #     return error_msg
    # except httpx.HTTPStatusError as exc:
    #     error_msg = f"Error: HTTP error {exc.response.status_code} while requesting {exc.request.url!r}. Response: {exc.response.text}"
    #     logger.error(error_msg)
    #     return error_msg
    # except (IOError, OSError) as exc:
    #     error_msg = f"Error: Failed to save image to {output_path}: {exc}"
    #     logger.error(error_msg)
    #     return error_msg
    # except Exception as exc:
    #     error_msg = f"Error: An unexpected error occurred: {exc}"
    #     logger.exception(error_msg)
    #     return error_msg

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