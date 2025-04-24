import logging
import re
from typing import Dict, List, Optional, Any

from a3x.fragments.base import BaseFragment, FragmentContext
from a3x.fragments.registry import fragment

logger = logging.getLogger(__name__)

@fragment(
    name="knowledge_interpreter",
    description="Converte respostas textuais em comandos A3L estruturados",
    category="Interpretation",
    skills=["parse_commands", "validate_syntax"]
)
class KnowledgeInterpreterFragment(BaseFragment):
    """Fragment responsável por interpretar e converter respostas textuais em comandos A3L."""

    # Padrões de regex para identificar comandos A3L
    COMMAND_PATTERNS = {
        "create_fragment": r"create\s+fragment\s+(\w+)\s+description\s+\"([^\"]+)\"",
        "promote_fragment": r"promote\s+fragment\s+(\w+)",
        "archive_fragment": r"archive\s+fragment\s+(\w+)",
        "train_model": r"train\s+model\s+(\w+)",
        "evaluate_fragment": r"evaluate\s+fragment\s+(\w+)",
        "analyze_performance": r"analyze\s+performance\s+of\s+(\w+)",
        "generate_dataset": r"generate\s+dataset\s+for\s+(\w+)"
    }

    async def execute(self, ctx: FragmentContext, text: str) -> Dict[str, Any]:
        """Interpreta o texto e extrai comandos A3L válidos."""
        try:
            commands = self._extract_commands(text)
            validated_commands = self._validate_commands(commands)
            
            return {
                "status": "success",
                "commands": validated_commands,
                "raw_text": text
            }
            
        except Exception as e:
            logger.exception("Erro durante interpretação de comandos:")
            return {
                "status": "error",
                "message": str(e)
            }

    def _extract_commands(self, text: str) -> List[Dict[str, Any]]:
        """Extrai comandos A3L do texto usando regex."""
        commands = []
        
        for cmd_type, pattern in self.COMMAND_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                command = {
                    "type": cmd_type,
                    "args": match.groups(),
                    "raw_match": match.group(0)
                }
                commands.append(command)
                
        return commands

    def _validate_commands(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Valida os comandos extraídos."""
        validated = []
        
        for cmd in commands:
            if self._is_valid_command(cmd):
                validated.append(cmd)
            else:
                logger.warning(f"Comando inválido ignorado: {cmd['raw_match']}")
                
        return validated

    def _is_valid_command(self, command: Dict[str, Any]) -> bool:
        """Verifica se um comando é válido."""
        # Implementar validações específicas para cada tipo de comando
        if command["type"] == "create_fragment":
            return len(command["args"]) == 2 and all(command["args"])
        elif command["type"] in ["promote_fragment", "archive_fragment"]:
            return len(command["args"]) == 1 and command["args"][0]
        else:
            return True  # Outros comandos são considerados válidos por padrão 