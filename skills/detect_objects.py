# /home/arthur/Projects/A3X/skills/detect_objects.py
import logging
import pathlib
from typing import Dict, Any, List
from ultralytics import YOLO
import json

# Configure logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (IMPORTANTE PARA VALIDAÇÃO)
WORKSPACE_ROOT = pathlib.Path("/home/arthur/Projects/A3X").resolve()

# Initialize model cache (YOLO handles its own model loading/caching)
yolo_model = None
MODEL_NAME = "yolov8n.pt" # Nano model - small and fast

def load_yolo_model():
    """Loads the YOLO model, downloading weights if necessary."""
    global yolo_model
    if yolo_model is None:
        try:
            logger.info(f"Loading YOLO model: {MODEL_NAME}...")
            # The model weights will be downloaded automatically on first use
            # and cached by the ultralytics library.
            yolo_model = YOLO(MODEL_NAME)
            logger.info(f"YOLO model {MODEL_NAME} loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model {MODEL_NAME}: {e}")
            yolo_model = None # Ensure it stays None on failure
            raise # Re-raise the exception to be caught by the skill
    return yolo_model

# Função auxiliar de validação de caminho
def _is_path_safe(target_path: pathlib.Path) -> bool:
    """Verifica se o caminho está dentro do WORKSPACE_ROOT."""
    try:
        resolved_path = target_path.resolve()
        return resolved_path.is_relative_to(WORKSPACE_ROOT)
    except Exception as e:
        logger.error(f"Erro ao validar caminho '{target_path}': {e}")
        return False

def skill_detect_objects(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detects objects in an image using the YOLOv8n model.

    Expected action_input parameters:
      - image_path (str): Path to the image file (PNG, JPG, etc.) (required).
      - confidence_threshold (float, optional): Minimum confidence score (0.0 to 1.0) for detected objects. Default: 0.25.
    """
    logger.debug(f"Executing skill_detect_objects with input: {action_input}")

    image_path_str = action_input.get("image_path")
    confidence_threshold = action_input.get("confidence_threshold", 0.25)

    # --- Basic Validation ---
    if not image_path_str:
        return {"status": "error", "action": "detection_failed", "data": {"message": "Error: 'image_path' parameter is required."}}

    try:
        image_path = pathlib.Path(image_path_str)
        if not image_path.is_absolute():
            image_path = (WORKSPACE_ROOT / image_path_str).resolve()
        else:
            image_path = image_path.resolve()

        if not _is_path_safe(image_path):
            return {"status": "error", "action": "detection_failed", "data": {"message": f"Error: Accessing image outside allowed directories is not permitted: {image_path_str}"}}
        
        if not image_path.is_file():
            return {"status": "error", "action": "detection_failed", "data": {"message": f"Error: Image file not found at: {image_path}"}}

        # --- Object Detection Logic ---
        model = load_yolo_model() # Load or get cached model
        if model is None:
             return {"status": "error", "action": "detection_failed", "data": {"message": f"Error: Failed to load YOLO model {MODEL_NAME}."}}

        logger.info(f"Performing object detection on image '{image_path}' with confidence >= {confidence_threshold}")
        
        # Perform prediction
        results = model.predict(source=str(image_path), conf=confidence_threshold)

        # Process results
        detected_objects = []
        if results:
            # results[0].boxes contains the bounding boxes, confidences, and class IDs
            # results[0].names contains the mapping from class ID to class name
            names = results[0].names
            for box in results[0].boxes:
                detected_objects.append({
                    "class_id": int(box.cls),
                    "class_name": names[int(box.cls)],
                    "confidence": float(box.conf),
                    # Bounding box in xyxy format (xmin, ymin, xmax, ymax)
                    "box_xyxy": [round(coord) for coord in box.xyxy[0].tolist()]
                })
        
        logger.info(f"Object detection successful. Found {len(detected_objects)} objects.")
        return {
            "status": "success",
            "action": "detection_completed",
            "data": {"detected_objects": detected_objects}
        }

    except FileNotFoundError:
         logger.error(f"File not found during detection process for: {image_path_str}")
         return {"status": "error", "action": "detection_failed", "data": {"message": f"Error: Image file not found (process issue?): {image_path_str}"}}    
    except Exception as e:
        logger.exception("Unexpected error during object detection:")
        return {"status": "error", "action": "detection_failed", "data": {"message": f"Unexpected error during object detection: {e}"}}

