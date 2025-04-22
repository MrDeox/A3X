import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from diffusers import AutoPipelineForText2Image
from PIL import Image

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
HOST = "127.0.0.1"
PORT = 7861
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "generated"
MAX_IMAGE_DIM = 1024

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State (Model Pipeline) ---
pipeline = None
device = None

# --- Pydantic Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int = Field(512, gt=0, le=MAX_IMAGE_DIM)
    height: int = Field(768, gt=0, le=MAX_IMAGE_DIM)
    num_images: int = Field(1, ge=1) # Currently supports only 1 for simplicity
    guidance_scale: float = Field(7.5, ge=0.0)
    num_inference_steps: int = Field(30, ge=1)
    seed: int = Field(42)

class ImageGenerationResponse(BaseModel):
    image_path: str
    seed: int

# --- Helper Functions ---
def setup_device():
    """Determine the appropriate device (ROCm/CUDA or CPU)."""
    global device
    if torch.cuda.is_available() and torch.version.hip:
        device = "cuda"
        logger.info("ROCm (HIP) detected. Using GPU.")
        # Potentially add torch.backends.cudnn optimizations here if needed
    else:
        device = "cpu"
        logger.info("ROCm (HIP) not detected or unavailable. Using CPU.")
    return device

def load_pipeline(device):
    """Load the diffusion pipeline."""
    global pipeline
    try:
        logger.info(f"Loading model {MODEL_ID}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            # revision="fp16" # Uncomment if specifically wanting fp16 weights and available
        )
        pipeline.to(device)
        # Potential future optimization: pipeline.enable_xformers_memory_efficient_attention() if xformers installed
        logger.info(f"Model {MODEL_ID} loaded successfully on device '{device}'.")
    except Exception as e:
        logger.error(f"Failed to load model {MODEL_ID}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load the diffusion model: {e}")

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    global device, pipeline
    device = setup_device()
    load_pipeline(device)
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {OUTPUT_DIR}")
    yield
    # Clean up resources on shutdown (optional)
    logger.info("Shutting down server.")
    pipeline = None
    device = None
    torch.cuda.empty_cache() # Clear VRAM if GPU was used

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan, title="A3X Diffusers Server")

# --- API Endpoints ---
@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    global pipeline, device
    if pipeline is None:
        logger.error("Pipeline not loaded. Cannot generate image.")
        raise HTTPException(status_code=503, detail="Model pipeline is not available.")

    logger.info(f"Received generation request: prompt='{request.prompt[:50]}...', seed={request.seed}")

    try:
        # Prepare generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(request.seed)

        # Prepare pipeline arguments
        pipeline_args = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": request.num_images,
            "generator": generator,
        }

        # TODO: Implement LoRA loading logic here if needed
        # Example placeholder:
        # if request.lora_weights:
        #    pipeline.load_lora_weights(request.lora_weights)
        #    # Adjust prompt/kwargs if needed for LoRA trigger words etc.
        #    pipeline_args["cross_attention_kwargs"] = {"scale": request.lora_scale} # Example

        # Generate image
        with torch.inference_mode(): # Use inference mode for efficiency
             images = pipeline(**pipeline_args).images

        # TODO: Unload LoRA weights if loaded temporarily
        # Example placeholder:
        # if request.lora_weights:
        #    pipeline.unload_lora_weights()

        if not images or len(images) == 0:
             raise RuntimeError("Image generation failed, no images returned.")

        # For now, we only handle the first image as per request model default
        image: Image.Image = images[0]

        # Save image
        unique_filename = f"{uuid.uuid4()}_seed{request.seed}.png"
        output_path = OUTPUT_DIR / unique_filename
        image.save(output_path, format="PNG")
        logger.info(f"Image saved to: {output_path}")

        # Return path
        return ImageGenerationResponse(image_path=str(output_path.relative_to(Path(__file__).parent.parent)), seed=request.seed) # Return relative path from project root potentially

    except Exception as e:
        logger.error(f"Image generation failed: {e}", exc_info=True)
        torch.cuda.empty_cache() # Attempt to clear VRAM on error
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "device": device, "model_loaded": pipeline is not None}

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {HOST}:{PORT}")
    uvicorn.run(__name__ + ":app", host=HOST, port=PORT, reload=False) # Use reload=True for development only 