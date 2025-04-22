"""
Routes actions based on predicted labels from neural fragments by posting messages.
"""
import logging
from typing import List, Dict, Any, Callable, Coroutine
import asyncio # For create_task
import time

logger = logging.getLogger(__name__)

async def route_label(
    label: str, 
    prediction_data: Dict[str, Any], 
    post_message_handler: Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]
) -> None:
    """
    Determines messages to send to other fragments based on a predicted label.

    Args:
        label: The predicted label string (e.g., 'intention', 'question').
        prediction_data: The dictionary containing prediction details 
                         (text, label, confidence, timestamp).
        post_message_handler: The async function to call to post a message 
                              (expects message_type, content, target_fragment).

    Returns:
        None. Messages are posted asynchronously.
    """
    original_text = prediction_data.get('text', '')
    confidence = prediction_data.get('confidence', 0.0)
    sender_id = "LabelRouter"

    logger.info(f"[{sender_id}] Routing label: '{label}' (Confidence: {confidence:.2f})")

    # --- Message Routing Rules (Customize Here) ---
    message_content = {
        "original_text": original_text,
        "predicted_label": label,
        "confidence": confidence,
        "timestamp": prediction_data.get('timestamp')
    }

    target_fragment = None
    message_type = "info" # Default message type

    if label == "pergunta" or label == "duvida":
        target_fragment = "prof_geral" # Target the main Professor
        message_type = "ajuda_requerida" # Specific type for help requests
        message_content["question"] = original_text # Add specific field

    elif label == "falha" or label == "erro":
        # Assuming a coordinator or supervisor fragment exists
        target_fragment = "supervisor_1" # Or maybe "critic_1"?
        message_type = "alerta_falha"
        message_content["details"] = f"Detected potential failure/error in text: {original_text[:100]}..."

    elif label == "intencao":
        # Notify a planner or coordinator fragment
        target_fragment = "exec_supervisor_1" # Or a dedicated PlannerFragment if created
        message_type = "nova_intencao"
        message_content["intention_text"] = original_text
        
    elif label == "feedback_positivo" or label == "feedback_negativo":
        target_fragment = "critic_1" # Send feedback to the critic
        message_type = "feedback_usuario" # Generic feedback type
        message_content["feedback_text"] = original_text
        message_content["polarity"] = label # Include original label
        
    # Add more rules based on expected labels...
    # elif label == "saudacao":
    #     target_fragment = "user_interface_manager" # Example
    #     message_type = "interacao_usuario"
        
    else:
        logger.info(f"[{sender_id}] No specific message route defined for label '{label}'.")
        # Optionally, send a generic message to a supervisor?
        # target_fragment = "supervisor_1"
        # message_type = "interpretacao_sem_rota"

    # --- Send the message if a target was determined --- 
    if target_fragment:
        logger.info(f"[{sender_id}] Posting message type '{message_type}' to '{target_fragment}' for label '{label}'.")
        try:
            # Use create_task for non-blocking message posting
            asyncio.create_task(
                post_message_handler(
                    message_type=message_type, 
                    content=message_content, 
                    target_fragment=target_fragment
                )
            )
        except Exception as post_err:
            # Log error but don't block the caller (message_processor)
            logger.error(f"[{sender_id}] Failed to create task for posting message to {target_fragment}: {post_err}", exc_info=True)

    # This function no longer returns A3L commands
    return

    # --- Simple Routing Rules (Customize Here) ---
    if label == "intencao": # Example: If label is 'intention'
        # Extract the core intention text (simplistic example)
        intention_core = original_text # Could use NLP later to refine
        # Generate a command to create a fragment to handle this intention
        # Note: Needs a way to define the new fragment's specifics (type, dims etc.)
        # This is a placeholder command - refinement needed.
        new_fragment_name = f"handler_intent_{abs(hash(intention_core)) % 10000}" 
        a3l_commands.append(f"criar fragmento '{new_fragment_name}' tipo 'reflexivo' description \"Handles intention: {intention_core[:50]}...\"")
        a3l_commands.append(f"interpretar texto para '{new_fragment_name}' texto \"{intention_core}\"") # Maybe send text to new frag?
        logger.info(f"[LabelRouter] Generated A3L commands for label '{label}': {a3l_commands}")

    elif label == "pergunta": # Example: If label is 'question'
        # Maybe trigger a search or ask another fragment?
        a3l_commands.append(f"aprender com 'prof_geral' question \"{original_text}\"")
        logger.info(f"[LabelRouter] Generated A3L commands for label '{label}': {a3l_commands}")
        
    elif label == "feedback_positivo":
        # Just log it for now
        logger.info(f"[LabelRouter] Received positive feedback label for: {original_text[:50]}...")
        # No commands generated

    # Add more rules based on expected labels from your 'language_analyzer' fragment
    # elif label == "saudacao": ...
    # elif label == "despedida": ...

    else:
        logger.info(f"[LabelRouter] No specific route defined for label '{label}'.")

    return a3l_commands 