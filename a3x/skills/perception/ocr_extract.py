"""Skill to perform OCR and extract text with bounding boxes from an image."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Core A3X imports
from a3x.core.skills import skill

# Dependencies for OCR - require installation:
# pip install pytesseract Pillow
# System dependency: tesseract-ocr (e.g., sudo apt install tesseract-ocr)
try:
    import pytesseract
    from PIL import Image, UnidentifiedImageError
    from pytesseract import Output
    _ocr_dependencies_installed = True
except ImportError as e:
    _ocr_dependencies_installed = False
    pytesseract = None
    Image = None
    UnidentifiedImageError = Exception
    Output = None
    _import_error_message = str(e)

logger = logging.getLogger(__name__)

# Confidence threshold to filter OCR results
DEFAULT_OCR_CONF_THRESHOLD = 60 # Only include results with confidence >= 60%

@skill(
    name="extract_text_boxes_from_image",
    description="Extracts text segments, their bounding boxes ([x1, y1, x2, y2]), and confidence scores from an image file using Tesseract OCR.",
    parameters={
        "image_path": (Path, ...), # Required Path parameter
        "confidence_threshold": (int, DEFAULT_OCR_CONF_THRESHOLD) # Optional confidence threshold (0-100)
    }
)
async def extract_text_boxes_from_image(ctx: Any, image_path: Path, confidence_threshold: int = DEFAULT_OCR_CONF_THRESHOLD) -> Dict[str, Any]:
    """
    Performs OCR on an image file to extract text blocks with bounding boxes and confidence.

    Args:
        ctx: The skill execution context (provides logger).
        image_path (Path): The path to the image file.
        confidence_threshold (int): Minimum confidence level (0-100) to include in results.

    Returns:
        A dictionary containing the status and a list of text boxes, or an error.
        Success data format: {"status": "success", "action": "ocr_completed", "data": {"text_boxes": [...]}}
        Each item in text_boxes: {"text": str, "box": [x1, y1, x2, y2], "confidence": float}
    """
    logger.info(f"Starting OCR text box extraction for: {image_path}")

    if not _ocr_dependencies_installed:
        logger.error(f"OCR dependencies missing: {_import_error_message}. Install pytesseract, Pillow, and tesseract-ocr.")
        return {
            "status": "error",
            "action": "dependency_missing",
            "message": "OCR dependencies (pytesseract/Pillow/tesseract-ocr) not installed.",
            "error_details": f"Import Error: {_import_error_message}. See installation instructions."
        }

    if not isinstance(image_path, Path):
         try:
             image_path = Path(image_path)
         except Exception as path_err:
             logger.error(f"Invalid image_path input: {image_path}. Error: {path_err}")
             return {"status": "error", "action": "invalid_parameter", "message": f"Invalid image_path type: {type(image_path)}"}

    if not image_path.exists() or not image_path.is_file():
        logger.error(f"Image file not found or is not a file: {image_path}")
        return {"status": "error", "action": "file_not_found", "message": f"Image file not found: {image_path}"}

    extracted_boxes = []
    try:
        img = Image.open(image_path)
        # Use image_to_data to get detailed information including bounding boxes and confidence
        # Set lang='eng' by default, could be parameterized later if needed.
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng')
        img.close()

        n_boxes = len(ocr_data['level'])
        for i in range(n_boxes):
            # Convert confidence string to float, handling potential errors/non-numeric values
            try:
                conf = float(ocr_data['conf'][i])
            except ValueError:
                conf = -1.0 # Assign low confidence if conversion fails

            # Filter by confidence level and ignore boxes with no text
            text = ocr_data['text'][i].strip()
            if conf >= confidence_threshold and text:
                # Extract bounding box coordinates
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                # Calculate x2, y2
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                extracted_boxes.append({
                    "text": text,
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf / 100.0, 4) # Normalize confidence to 0-1 range
                })

        logger.info(f"OCR completed. Extracted {len(extracted_boxes)} text boxes above threshold {confidence_threshold}%.")
        return {
            "status": "success",
            "action": "ocr_completed",
            "data": {"text_boxes": extracted_boxes}
        }

    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file format: {image_path}")
        return {"status": "error", "action": "invalid_image_format", "message": f"Invalid or unsupported image format: {image_path}"}
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract executable not found. Ensure tesseract-ocr is installed and in PATH.")
        return {"status": "error", "action": "dependency_missing", "message": "Tesseract executable not found.", "error_details": "Install tesseract-ocr system package."}
    except Exception as e:
        logger.exception(f"Error during OCR processing for {image_path}:")
        return {"status": "error", "action": "ocr_failed", "message": f"An unexpected error occurred during OCR: {e}"} 