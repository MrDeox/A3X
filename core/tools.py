# core/tools.py
import json
import os
import sys
import traceback
import logging

# Ajuste para importar skills do diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- Explicit Skill Imports --- 
# Initialize TOOLS dictionary first
TOOLS = {}

logger = logging.getLogger(__name__)

try:
    from skills.manage_files import skill_manage_files
    TOOLS['manage_files'] = {
        "function": skill_manage_files,
        "description": "Creates, overwrites, or appends to files. Args: action ('create' or 'append'), file_name (path relative to workspace), content (string).",
        "parameters": {
            "action": "'create' or 'append'",
            "file_name": "Relative path to the file within the workspace",
            "content": "The text content to write or append"
        }
    }
except ImportError:
    logger.error("Failed to import skill: manage_files", exc_info=True)

try:
    from skills.read_file import skill_read_file
    TOOLS['read_file'] = {
        "function": skill_read_file,
        "description": "Reads the entire content of a specified text file. Args: file_path (relative or absolute)",
        "parameters": {"file_path": "Path to the file"}
    }
except ImportError:
    logger.error("Failed to import skill: read_file", exc_info=True)

try:
    from skills.list_files import skill_list_files
    TOOLS['list_files'] = {
        "function": skill_list_files,
        "description": "Lists files and directories in a specified path relative to the workspace root (default: root). Args: directory (optional, relative path)",
        "parameters": {"directory": "(Optional) Relative path from workspace root"}
    }
except ImportError:
    logger.error("Failed to import skill: list_files", exc_info=True)

try:
    from skills.delete_file import skill_delete_file
    TOOLS['delete_file'] = {
        "function": skill_delete_file,
        "description": "Deletes a specified file. Requires confirmation. Args: file_path (relative or absolute), confirm (boolean, must be true)",
        "parameters": {"file_path": "Path to the file", "confirm": "Must be true to delete"}
    }
except ImportError:
    logger.error("Failed to import skill: delete_file", exc_info=True)

try:
    from skills.generate_code import skill_generate_code
    TOOLS['generate_code'] = {
        "function": skill_generate_code,
        "description": "Generates code based on a description. Args: purpose (description), language (optional, default python), context (optional)",
        "parameters": {"purpose": "Description of the code's function", "language": "(Optional) Programming language", "context": "(Optional) Existing code context"}
    }
except ImportError:
    logger.error("Failed to import skill: generate_code", exc_info=True)

try:
    from skills.execute_code import skill_execute_code
    TOOLS['execute_code'] = {
        "function": skill_execute_code,
        "description": "Executes Python code. Args: code (string)",
        "parameters": {"code": "The Python code to execute"}
    }
except ImportError:
    logger.error("Failed to import skill: execute_code", exc_info=True)

try:
    from skills.modify_code import skill_modify_code
    TOOLS['modify_code'] = {
        "function": skill_modify_code,
        "description": "Modifies existing code based on instructions. Args: modification (instructions), code_to_modify (original code)",
        "parameters": {"modification": "Instructions for change", "code_to_modify": "The original code string"}
    }
except ImportError:
    logger.error("Failed to import skill: modify_code", exc_info=True)

try:
    from skills.ocr_image import skill_ocr_image
    TOOLS['ocr_image'] = {
        "function": skill_ocr_image,
        "description": "Extracts text from an image using Tesseract OCR. Args: image_path, lang (optional, default 'eng')",
        "parameters": {"image_path": "Path to the image file", "lang": "(Optional) Language code(s) for Tesseract (e.g., eng, por, eng+por)"}
    }
except ImportError:
    logger.error("Failed to import skill: ocr_image", exc_info=True)

try:
    from skills.classify_sentiment import skill_classify_sentiment
    TOOLS['classify_sentiment'] = {
        "function": skill_classify_sentiment,
        "description": "Classifies the sentiment of a text (1-5 stars). Args: text",
        "parameters": {"text": "The text to classify"}
    }
except ImportError:
    logger.error("Failed to import skill: classify_sentiment", exc_info=True)

try:
    from skills.final_answer import skill_final_answer
    TOOLS['final_answer'] = {
        "function": skill_final_answer,
        "description": "Provides the final answer to the user's request. Args: answer (string)",
        "parameters": {"answer": "The final answer text"}
    }
except ImportError:
    logger.error("Failed to import skill: final_answer", exc_info=True)

# Example: Add memory skills if needed and handle import errors
# try:
#     from skills.memory import skill_save_memory, skill_recall_memory
#     TOOLS['save_memory'] = {
#         "function": skill_save_memory,
#         "description": "Saves information to long-term memory. Args: content (string), metadata (optional dict)",
#         "parameters": {"content": "Information to save", "metadata": "(Optional) Dictionary of metadata"}
#     }
#     TOOLS['recall_memory'] = {
#         "function": skill_recall_memory,
#         "description": "Recalls relevant information from long-term memory based on a query. Args: query (string), top_k (optional int)",
#         "parameters": {"query": "Search query", "top_k": "(Optional) Number of results to return"}
#     }
# except ImportError:
#     logger.error("Failed to import memory skills", exc_info=True)

# --- End Explicit Skill Imports ---

def get_tool(tool_name: str) -> dict | None:
    """Retorna a descrição e função de uma ferramenta específica."""
    return TOOLS.get(tool_name)

def get_tool_descriptions() -> str:
    """Retorna uma string formatada com as descrições de todas as ferramentas."""
    descriptions = []
    for name, tool_info in TOOLS.items():
        # Formatar parâmetros para melhor clareza no prompt
        params_str_list = []
        if tool_info.get("parameters"):
             for param_name, param_desc in tool_info["parameters"].items():
                 params_str_list.append(f"  - {param_name}: {param_desc}")
             params_str = "\n".join(params_str_list)
             descriptions.append(f"- {name}:\n  Description: {tool_info['description']}\n  Parameters:\n{params_str}")
        else:
            descriptions.append(f"- {name}:\n  Description: {tool_info['description']}\n  Parameters: None")

    return "\n".join(descriptions) 