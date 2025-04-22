import logging
import os
from datetime import datetime
from typing import Dict, Optional, Any

# Assuming these are available in the environment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment

logger = logging.getLogger(__name__)

async def handle_reflect_fragment(
    directive: Dict[str, Any],
    memory_bank: MemoryBank
) -> Optional[Dict[str, Any]]:
    """Handles the 'reflect_fragment' directive logic."""

    fragment_id = directive.get("fragment_id")
    output_format = directive.get("format", "a3l") # Default to a3l

    # --- MOCK FOR TESTING KNOWLEDGE INTERPRETER ---
    if fragment_id == 'dummy_reflector':
        logger.info(f"[A3X Bridge Handler - Reflect] --- MOCK REFLECTION for {fragment_id} ---")
        mock_reflection_text = "O fragmento 'dummy_reflector' parece impreciso, talvez devêssemos treiná-lo novamente por 3 épocas. Também, podemos criar um fragmento 'dummy_reflector_v2' com base em 'dummy_reflector'."
        logger.info(f"[A3X Bridge Handler - Reflect] --- Returning mock text: '{mock_reflection_text}' ---")
        return {"status": "success", "result": mock_reflection_text, "format": "text"} 
    # --- END MOCK ---

    if not fragment_id or not isinstance(fragment_id, str):
        logger.error("[A3X Bridge Handler - Reflect] 'fragment_id' (string) missing or invalid.")
        return { "status": "error", "error": "Reflection failed: 'fragment_id' missing or invalid" }
    
    logger.info(f"[A3X Bridge Handler - Reflect] Attempting to reflect on fragment '{fragment_id}' (format: {output_format})...")
    fragment = memory_bank.load(fragment_id)

    if fragment is None:
        logger.error(f"[A3X Bridge Handler - Reflect] Fragment '{fragment_id}' not found.")
        return { "status": "error", "error": f"Fragment '{fragment_id}' not found." }

    # --- Handle A3L Format --- 
    if output_format == "a3l":
         if hasattr(fragment, 'generate_reflection_a3l') and callable(getattr(fragment, 'generate_reflection_a3l')):
             try:
                 a3l_reflection = fragment.generate_reflection_a3l()
                 logger.info(f"[A3X Bridge Handler - Reflect] Generated A3L reflection for '{fragment_id}'.")
                 return {"status": "success", "reflection_a3l": a3l_reflection}
             except Exception as e:
                 logger.error(f"[A3X Bridge Handler - Reflect] Error generating A3L reflection for {fragment_id}: {e}", exc_info=True)
                 return {"status": "error", "error": f"Failed to generate A3L reflection: {e}"}
         else:
              logger.error(f"[A3X Bridge Handler - Reflect] Fragment '{fragment_id}' ({type(fragment).__name__}) does not support generate_reflection_a3l method.")
              return { "status": "error", "error": f"Fragment type {type(fragment).__name__} does not support A3L reflection format." }
    
    # --- Handle Dictionary Format --- 
    elif output_format == "dict":
        reflection_data = {
            "fragment_id": fragment_id,
            "class_name": fragment.__class__.__name__,
            "module": fragment.__class__.__module__
        }
        for attr in ['input_dim', 'hidden_dim', 'output_dim', 'num_classes']:
            if hasattr(fragment, attr):
                reflection_data[attr] = getattr(fragment, attr)
        if hasattr(fragment, 'id_to_label'):
            reflection_data['id_to_label'] = getattr(fragment, 'id_to_label')
        if hasattr(fragment, 'description'):
             reflection_data['description'] = getattr(fragment, 'description', None)
        try:
            # Need Path import for this
            from pathlib import Path 
            state_file_path = memory_bank._get_save_path(fragment_id, "pt")
            if state_file_path.exists():
                mtime_timestamp = os.path.getmtime(state_file_path)
                reflection_data['last_modified_timestamp'] = mtime_timestamp
                reflection_data['last_modified_utc'] = datetime.utcfromtimestamp(mtime_timestamp).isoformat() + 'Z'
            else:
                 reflection_data['last_modified_timestamp'] = None
                 reflection_data['last_modified_utc'] = "State file not found"
        except Exception as e:
            logger.warning(f"[A3X Bridge Handler - Reflect] Could not get modification time for {fragment_id}: {e}")
            reflection_data['last_modified_timestamp'] = None
            reflection_data['last_modified_utc'] = f"Error retrieving: {e}"

        logger.info(f"[A3X Bridge Handler - Reflect] Reflection data gathered for '{fragment_id}'.")
        return {"status": "success", "reflection": reflection_data}
    
    # --- Handle Invalid Format --- 
    else:
        logger.error(f"[A3X Bridge Handler - Reflect] Invalid reflection format specified: {output_format}")
        return { "status": "error", "error": f"Invalid reflection format: {output_format}. Supported formats: dict, a3l." } 