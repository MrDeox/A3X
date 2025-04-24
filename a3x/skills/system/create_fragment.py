import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from a3x.core.skills import skill
from a3x.core.context import Context
from a3x.fragments.base import BaseFragment # Assuming BaseFragment is here
from a3x.utils.string_utils import snake_to_pascal_case # Helper for class name

logger = logging.getLogger(__name__)

# Define the target directory relative to the workspace root
TARGET_DIR = Path(\"a3x/fragments/auto_generated\")

@skill(
    name=\"criar_fragmento\",
    description=\"Cria um novo arquivo de fragmento básico em a3x/fragments/auto_generated/ com nome e descrição fornecidos.\",
    parameters={
        \"type\": \"object\",
        \"properties\": {
            \"nome\": {\"type\": \"string\", \"description\": \"O nome do fragmento (ex: meu_analisador). Será usado para o nome do arquivo e da classe.\"},
            \"descricao\": {\"type\": \"string\", \"description\": \"Uma breve descrição da funcionalidade do fragmento.\"},
        },
        \"required\": [\"nome\", \"descricao\"]
    }
)
async def criar_fragmento(
    action_input: Dict[str, Any], context: Optional[Any] = None
) -> Dict[str, Any]:
    \"\"\"
    Creates a new basic fragment file in a3x/fragments/auto_generated/.

    Args:
        action_input (Dict[str, Any]): Dictionary containing 'nome' and 'descricao'.
        context (Optional[Any]): The execution context (unused here but good practice).

    Returns:
        Dict[str, Any]: A dictionary with status and the path of the created file, or an error message.
    \"\"\"
    fragment_name = action_input.get(\"nome\")
    fragment_description = action_input.get(\"descricao\")

    if not fragment_name or not fragment_description:
        logger.error(\"\'nome\' e \'descricao\' são obrigatórios para criar_fragmento.\")
        return {\"status\": \"error\", \"message\": \"Parâmetros \'nome\' e \'descricao\' são obrigatórios.\"}

    # Sanitize fragment_name for filename (basic example, might need more robust handling)
    file_name = f\"{fragment_name.lower().replace(' ', '_').replace('-', '_')}.py\"
    
    # Convert snake_case/kebab-case name to PascalCase for the class name
    class_name = snake_to_pascal_case(fragment_name)

    # Construct the full path
    if context and context.workspace_root:
         # Ensure the target directory exists
        target_dir_abs = context.workspace_root / TARGET_DIR
        target_dir_abs.mkdir(parents=True, exist_ok=True) # Create dir if not exists
        file_path = target_dir_abs / file_name
    else:
        # Fallback if context or workspace_root is not available (might not be ideal)
        logger.warning(\"Workspace root not found in context. Using relative path for fragment creation.\")
        target_dir_rel = TARGET_DIR
        target_dir_rel.mkdir(parents=True, exist_ok=True) # Create dir if not exists
        file_path = target_dir_rel / file_name


    # Generate basic fragment content
    fragment_content = f\"\"\"\\
from typing import Dict, Any, Optional
import logging

from a3x.fragments.base import BaseFragment
from a3x.core.context import Context

logger = logging.getLogger(__name__)

class {class_name}(BaseFragment):
    \\\"\\\"\\\"Fragmento que faz: {fragment_description}.\\\"\\\"\\\"

    async def execute(self, context: Optional[Any] = None, **kwargs: Any) -> Any:
        logger.info(f\"Executing {class_name}...\")
        # TODO: Implement fragment logic here
        result = f\\\"Resultado da execução de {class_name}\\\"
        logger.info(f\"{class_name} executed successfully.\")
        return result

    # You can add other methods like setup, teardown, etc.
    # def setup(self):
    #     logger.info(f\"Setting up {class_name}...\")

\"\"\"

    try:
        with open(file_path, \"w\", encoding=\"utf-8\") as f:
            f.write(fragment_content)
        
        relative_path_str = str(file_path.relative_to(context.workspace_root)) if context and context.workspace_root else str(file_path)
        logger.info(f\"Fragmento \'{class_name}\' criado com sucesso em \'{relative_path_str}\'.\")
        return {\"status\": \"success\", \"path\": relative_path_str}

    except IOError as e:
        logger.exception(f\"Erro ao escrever o arquivo do fragmento \'{file_name}\':\")
        return {\"status\": \"error\", \"message\": f\"Erro de I/O ao criar o fragmento: {e}\"}
    except Exception as e:
        logger.exception(f\"Erro inesperado ao criar o fragmento \'{file_name}\':\")
        return {\"status\": \"error\", \"message\": f\"Erro inesperado: {e}\"} 