import logging
import json
import os
import aiohttp
import base64
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from a3x.core.skills import skill
from a3x.core.config import PROJECT_ROOT
from a3x.core.context import Context

logger = logging.getLogger(__name__)

SD_API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"
GENERATED_DIR = os.path.join("a3x", "memory", "generated")
PERSONAS_DIR = os.path.join("a3x", "memory", "personas")
DEFAULT_LORA_DIR = os.path.join("models", "Lora")

os.makedirs(GENERATED_DIR, exist_ok=True)

# --- Default SD Parameters (can be overridden by persona or skill args) ---
DEFAULT_NEGATIVE_PROMPT = ("(worst quality, low quality:1.4), bad anatomy, extra fingers, "
                           "fewer fingers, text, error, blurry, deformed, disfigured, "
                           "mutation, ugly, username, signature, watermark, jpeg artifacts") # Example
DEFAULT_STEPS = 25
DEFAULT_CFG_SCALE = 7
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768
DEFAULT_SAMPLER = "DPM++ 2M Karras" # Common high-quality sampler

# --- LoRA Handling ---
def find_loras(lora_dir: str = DEFAULT_LORA_DIR) -> List[str]:
    """Finds .safetensors files in the specified LoRA directory."""
    if not os.path.isdir(lora_dir):
        logger.warning(f"LoRA directory not found: {lora_dir}")
        return []
    try:
        return [f for f in os.listdir(lora_dir) if f.lower().endswith('.safetensors')]
    except OSError as e:
        logger.error(f"Error listing LoRA directory {lora_dir}: {e}")
        return []

def format_lora_string(lora_filename: str, weight: float = 0.7) -> str:
    """Formats a LoRA filename into the string format expected by SD Web UI prompts."""
    lora_name = os.path.splitext(lora_filename)[0]
    return f"<lora:{lora_name}:{weight}>"

@skill(
    name="nsfw_image_generator",
    description="Generates an NSFW image based on a persona file and optional overrides.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM access and file paths."},
        "persona_path": {"type": str, "description": "Path to the persona JSON file."},
        "num_images": {"type": int, "default": 1, "description": "Number of images to generate (default: 1)."},
        "lora_weight": {"type": float, "default": 0.7, "description": "Weight for the LoRA model (default: 0.7)."},
        "override_prompt": {"type": Optional[str], "default": None, "description": "Optional prompt to override the persona's visual prompt."},
        "override_negative_prompt": {"type": Optional[str], "default": None, "description": "Optional negative prompt to override defaults."},
        "override_lora_filename": {"type": Optional[str], "default": None, "description": "Optional specific LoRA filename to use."}
    }
)
async def generate_nsfw_image(
    context: Context,
    persona_path: str,
    num_images: int = 1,
    lora_weight: float = 0.7,
    override_prompt: Optional[str] = None,
    override_negative_prompt: Optional[str] = None,
    override_lora_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates NSFW images via Stable Diffusion Web UI API based on a persona file.

    Args:
        ctx: The skill execution context (not used directly here).
        persona_path: Path to the persona JSON file.
        num_images: Number of images to generate for this persona.
        lora_weight: Weight to apply to the selected LoRA.
        override_prompt: Optional prompt to use instead of the one in the persona file.
        override_negative_prompt: Optional negative prompt to use.
        override_lora_filename: Optional specific LoRA filename (e.g., 'altgirl.safetensors') to use.

    Returns:
        A dictionary containing the status and a list of generated image paths.
    """
    logger.info(f"Starting NSFW image generation for persona: {persona_path}")

    # --- Load Persona --- 
    persona_data = None
    if not os.path.isabs(persona_path):
        # Assume relative to project root or a predefined base like PERSONAS_DIR
        persona_path_abs = os.path.join(PERSONAS_DIR, os.path.basename(persona_path)) 
        # More robust: Check if it exists directly first, then try joining with PERSONAS_DIR
        if not os.path.exists(persona_path) and os.path.exists(persona_path_abs):
            persona_path = persona_path_abs
        elif not os.path.exists(persona_path):
            logger.error(f"Persona file not found at specified path or in personas dir: {persona_path}")
            return {"status": "error", "message": "Persona file not found."}
    
    try:
        with open(persona_path, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Persona file not found: {persona_path}")
        return {"status": "error", "message": "Persona file not found."}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode persona JSON file {persona_path}: {e}")
        return {"status": "error", "message": "Invalid persona JSON file."}
    except Exception as e:
        logger.exception(f"Error loading persona file {persona_path}:")
        return {"status": "error", "message": f"Error loading persona file: {e}"}

    # --- Select LoRA --- 
    available_loras = find_loras()
    selected_lora_string = ""
    if override_lora_filename:
        if override_lora_filename in available_loras:
            selected_lora_string = format_lora_string(override_lora_filename, lora_weight)
            logger.info(f"Using overridden LoRA: {override_lora_filename}")
        else:
            logger.warning(f"Overridden LoRA '{override_lora_filename}' not found in {DEFAULT_LORA_DIR}. Proceeding without LoRA.")
    elif available_loras:
        # Simple strategy: pick one randomly for variety, or prioritize known ones
        # Prioritize known good ones if they exist
        preferred_loras = [l for l in available_loras if l.lower().startswith(('altgirl', 'slugbox'))]
        if preferred_loras:
            chosen_lora = random.choice(preferred_loras)
        else:
            chosen_lora = random.choice(available_loras) # Fallback to random
        selected_lora_string = format_lora_string(chosen_lora, lora_weight)
        logger.info(f"Selected LoRA: {chosen_lora} with weight {lora_weight}")
    else:
        logger.warning("No LoRAs found or specified. Proceeding without LoRA.")

    # --- Prepare Prompt --- 
    base_prompt = override_prompt if override_prompt else persona_data.get("visual_prompt")
    if not base_prompt:
        logger.error("No visual prompt found in persona or override.")
        return {"status": "error", "message": "Visual prompt is missing."}
    
    final_prompt = f"{selected_lora_string} {base_prompt}".strip()
    negative_prompt = override_negative_prompt if override_negative_prompt else DEFAULT_NEGATIVE_PROMPT

    # --- Prepare Payload --- 
    payload = {
        "prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "seed": -1, # Random seed
        "steps": DEFAULT_STEPS,
        "cfg_scale": DEFAULT_CFG_SCALE,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "sampler_name": DEFAULT_SAMPLER,
        "batch_size": num_images,
        # Add other parameters as needed, e.g., hires fix, specific VAE
    }
    logger.debug(f"Sending payload to SD API: {json.dumps(payload, indent=2)}")

    # --- Call SD API --- 
    generated_files = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(SD_API_URL, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'images' in result:
                        persona_name_slug = re.sub(r'\W+', '-', persona_data.get("persona_name", "unknown")).lower()
                        for i, img_b64 in enumerate(result["images"]):
                            try:
                                img_data = base64.b64decode(img_b64)
                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                filename = f"{persona_name_slug}_{timestamp}_{i+1}.png"
                                filepath = os.path.join(GENERATED_DIR, filename)
                                with open(filepath, 'wb') as f:
                                    f.write(img_data)
                                generated_files.append(filepath)
                                logger.info(f"Saved generated image: {filepath}")
                            except (base64.binascii.Error, IOError, Exception) as e:
                                logger.error(f"Error saving image {i}: {e}")
                        
                        if generated_files:
                            return {"status": "success", "generated_paths": generated_files}
                        else:
                            return {"status": "error", "message": "API returned success but failed to save images."}
                    else:
                        logger.error(f"SD API Error: 'images' key missing in response. Response: {result}")
                        return {"status": "error", "message": "SD API response missing 'images' data.", "api_response": result}
                else:
                    error_text = await response.text()
                    logger.error(f"SD API request failed with status {response.status}: {error_text}")
                    # Provide specific hint for connection error
                    if response.status == 503 or 'Connection refused' in error_text:
                         hint = "Could not connect to SD Web UI API. Is it running with the --api flag? Try running: python -m a3x.servers.sd_api_server"
                         logger.warning(hint)
                         return {"status": "error", "message": f"SD API request failed (Status {response.status}). {hint}", "details": error_text}
                    else:
                         return {"status": "error", "message": f"SD API request failed (Status {response.status}).", "details": error_text}
    except aiohttp.ClientConnectorError as e:
        hint = "Could not connect to SD Web UI API. Is it running with the --api flag? Try running: python -m a3x.servers.sd_api_server"
        logger.error(f"{hint} Details: {e}")
        return {"status": "error", "message": hint}
    except Exception as e:
        logger.exception("An unexpected error occurred during image generation:")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

"""
# Example usage (for testing purposes)
async def main_test():
    # Create a dummy persona file for testing
    dummy_persona = {
        "persona_name": "Test Dummy",
        "bio": "Test bio.",
        "tags": ["test", "dummy"],
        "visual_prompt": "ultrarealistic photo of a cute kitten, studio lighting, sharp focus, NSFW",
        "style_reference": "photorealistic"
    }
    dummy_path = os.path.join(PERSONAS_DIR, "test-dummy.json")
    with open(dummy_path, 'w') as f:
        json.dump(dummy_persona, f)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Ensure a LoRA exists for testing this part
    if not find_loras():
         print("WARNING: No LoRAs found in models/Lora/. LoRA selection won't be tested.")
    else:
         print(f"Found LoRAs: {find_loras()}")

    print(f"Attempting to generate image using persona: {dummy_path}")
    # Provide a mock context if needed by the skill internals (not needed here)
    result = await generate_nsfw_image(None, persona_path=dummy_path, num_images=1)
    print("--- RESULT ---")
    print(result)
    print("--------------")
    # Clean up dummy file
    # os.remove(dummy_path)

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main_test()) # Uncomment to run standalone test
    print("NSFW Image Generator Skill Module Loaded. Use agent CLI or import to run.")
""" 