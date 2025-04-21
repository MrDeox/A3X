import re
from typing import Optional, Dict, Any, List
import json
import ast
import logging

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

# Regex patterns corrigidos (assumindo que as barras invertidas extras foram removidas)
patterns = {
    "comment": re.compile(r"^\s*#.*$"),
    # ask: Captura ID e início da lista (processamento posterior necessário)
    "ask": re.compile(r"^perguntar\s+ao\s+fragmento\s+'([^']+?)'\s+com\s+(\[.*)$", re.IGNORECASE | re.DOTALL),
    "train": re.compile(r"^treinar\s+fragmento\s+'([^']+)'(?:\s+usando\s+contexto\s+'([^']+)')?\s+por\s+(\d+)\s+épocas?", re.IGNORECASE),
    "reflect": re.compile(r"^refletir\s+sobre\s+fragmento\s+'([^']+)'(?:\s+como\s+(json|a3l))?", re.IGNORECASE),
    # Import precisa 'de'
    "import": re.compile(r"^importar\s+fragmento\s+de\s+'([^']+)'(?:\s+como\s+'([^']+)')?", re.IGNORECASE),
    # Exportar *para* path
    "export": re.compile(r"^exportar\s+fragmento\s+'([^']+)'\s+para\s+'([^']+)'", re.IGNORECASE),
    "export_simple": re.compile(r"^exportar\s+fragmento\s+'([^']+)'\s*$", re.IGNORECASE), # Adicionado para export sem path
    "confidence_conditional": re.compile(r"^se\s+confiança\s+for\s+(maior|menor)\s+que\s+([\d\.]+)\s+então\s+(.*)", re.IGNORECASE),
    "cumulative_epochs_conditional": re.compile(r"^se\s+fragmento\s+'([^']+)'\s+foi\s+treinado\s+por\s+mais\s+de\s+(\d+)\s+épocas?\s+então\s+(.*)", re.IGNORECASE),
    "create_fragment": re.compile(r"^criar\s+fragmento\s+'([^']+)'\s+tipo\s+'([^']+)'(.*)$", re.IGNORECASE),
    "create_fragment_from_base": re.compile(r"^criar\s+fragmento\s+'([^']+)'\s+com\s+base\s+em\s+'([^']+)'\s*$", re.IGNORECASE),
    # ask_professor usando r"""...""" para clareza
    "ask_professor": re.compile(r"""^ask_professor\s+'([^']+)'\s+question\s+"([^"]+)"\s*$""", re.IGNORECASE),
    # learn_directive usando r"""...""" para clareza
    "learn_directive": re.compile(r"""^aprender\s+com\s+'([^']+)'(?:\s+sobre\s+'([^']+)')?\s+question\s+"([^"]+)"\s*$""", re.IGNORECASE),
    # learn_from_text (aprender com texto livre)
    "learn_from_text": re.compile(r"^aprender\s+com\s+'(.*)'\s*$", re.IGNORECASE),
    # interpret_text (comando explícito para interpretar texto)
    "interpret_text": re.compile(r"^interpretar\s+texto\s+'(.*)'$", re.IGNORECASE),
    # --- Solicitar Exemplos (NOVO) ---
    "solicitar_exemplos": re.compile(r"solicitar exemplos para tarefa \"([^\"]+)\"", re.IGNORECASE),
    # --- Avaliar Fragmento (NOVO) ---
    "avaliar_fragmento": re.compile(r"^avaliar\s+fragmento\s+'([^']+)'\s+com\s+dados\s+de\s+teste\s+'([^']+)'\s*$", re.IGNORECASE),
    # --- Comparar Desempenho (NOVO) ---
    "comparar_desempenho": re.compile(r"^comparar\s+desempenho\s+do\s+fragmento\s+'([^']+)'\s+após\s+treino\s+em\s+'([^']+)'\s*$", re.IGNORECASE),
    # --- Estudar Habilidade (NOVO - Macro Comando) ---
    "estudar_habilidade": re.compile(r"^estudar\s+habilidade\s+\"([^\"]+)\"\s*$", re.IGNORECASE),
    # <<< NOVO COMANDO: planejar dados >>>
    "planejar_dados": re.compile(r"^planejar\s+dados\s+para\s+tarefa\s+\"([^\"]+)\"\s*$", re.IGNORECASE),
}

# Mapeamento de tipos de diretiva para nomes de grupos regex
# ATENÇÃO: Este map não é mais usado diretamente para extração,
# a lógica de extração agora está dentro do loop principal da função interpret_a3l_line.
# Mantido por referência ou para possível uso futuro em outra lógica.
DIRECTIVE_PARAM_MAP = {
    "create_fragment": {"fragment_id": 1, "fragment_type": 2, "params": 3},
    "train": {"fragment_id": 1, "context_id": 2, "epochs": 3}, # Corrigido: epochs era grupo 3
    "interpret_text": {"text": 1},
    "reflect": {"fragment_id": 1, "format": 2}, # Adicionado format
    "learn_directive": {"professor_id": 1, "context_fragment_id": 2, "question": 3}, # Ordem corrigida
    "ask_professor": {"professor_id": 1, "question": 2},
    "define_variable": {"var_name": 1, "value": 2}, # Assumindo um padrão não mostrado
    "solicitar_exemplos": {"task_name": 1},
    "ask": {"fragment_id": 1, "input": 2},
    "import": {"path": 1, "fragment_id": 2},
    "export": {"fragment_id": 1, "path": 2},
    "export_simple": {"fragment_id": 1},
    "create_fragment_from_base": {"new_fragment_id": 1, "base_fragment_id": 2},
    "learn_from_text": {"text": 1},
    # Condicionais não mapeados aqui, processados separadamente
    "avaliar_fragmento": {"fragment_id": 1, "task_name": 2}, # NOVO
    "comparar_desempenho": {"fragment_id": 1, "task_name": 2}, # NOVO
    "estudar_habilidade": {"task_name": 1}, # NOVO
    "planejar_dados": {"task_name": 1}, # NOVO
}

# Helper para parsing de parâmetros (simplificado)
def _parse_params(params_str: str) -> Dict[str, Any]:
    params = {}
    if not params_str:
        return params
    # Pattern para key val1 val2 key2 val3 ...
    param_pattern = re.compile(r"(\w+)\s+((?:\{[^\}]*\}|\[[^\]]*\]|\'[^\']*\'|\"[^\"]*\"|\S)+)")
    try:
        matches = param_pattern.findall(params_str)
        for key, value_str in matches:
            value_str = value_str.strip()
            try:
                # Tenta avaliar como literal Python (números, listas, dicts, bools, None)
                params[key] = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Se falhar, trata como string (removendo aspas se houver)
                if (value_str.startswith("'") and value_str.endswith("'")) or \
                   (value_str.startswith('"') and value_str.endswith('"')):
                    params[key] = value_str[1:-1]
                else:
                    params[key] = value_str
    except Exception as e:
         logger.error(f"Error parsing parameters '{params_str}': {e}", exc_info=True)
    return params

# Helper para parsing de lista JSON (usado por 'ask')
def _parse_json_list(list_str: str) -> Optional[List[float]]:
     # Clean the list string: remove internal newlines, handle potential comments
    list_str = re.sub(r"#.*$", "", list_str, flags=re.MULTILINE).strip()
    cleaned_list_str = list_str.replace('\n', ' ').strip()

    if not cleaned_list_str.startswith('[') or not cleaned_list_str.endswith(']'):
        logger.warning(f"Malformed list structure after cleaning: '{cleaned_list_str}'")
        return None

    try:
        input_list = json.loads(cleaned_list_str)
        if not isinstance(input_list, list) or not all(isinstance(item, (int, float)) for item in input_list):
             raise ValueError("Input list invalid format or content.")
        # Convert all to float for consistency
        return [float(item) for item in input_list]
    except (json.JSONDecodeError, ValueError) as e:
         logger.warning(f"Parse Error in cleaned ask list '{cleaned_list_str[:100]}...': {e}")
         return None

def interpret_a3l_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Interprets a single line of A3L, iterating through compiled patterns.
    Provides debug logging on failure.
    """
    line = line.strip()
    # Use INFO level for this crucial initial log, DEBUG for others
    logger.info(f"[A3L Interp] >>> Interpreting line: '{line}'") # Changed to INFO

    if not line or patterns["comment"].match(line):
        logger.debug("[A3L Interp] Line is empty or a comment.")
        return None

    for directive_type, pattern in patterns.items():
        if directive_type == "comment":
            continue # Already checked

        # Added detailed pre-match logging
        logger.debug(f"[A3L Interp]   Trying pattern: {directive_type} - Regex: {pattern.pattern}")
        try:
            match = pattern.match(line)
            if match:
                # Added detailed post-match logging
                logger.info(f"[A3L Interp]   +++ Matched pattern: {directive_type}")
                groups = match.groups()
                logger.debug(f"[A3L Interp]       Groups found: {groups}") # Log groups immediately

                # --- Specific Extraction Logic ---
                if directive_type == "ask_professor":
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        result_dict = {"type": "ask_professor", "professor_id": groups[0].strip(), "question": groups[1].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                elif directive_type == "learn_directive":
                    if len(groups) >= 3 and groups[0] and groups[2]:
                        directive = {"type": "learn_from_professor", "professor_id": groups[0].strip(), "question": groups[2].strip()}
                        if groups[1]: # Optional context_id
                            directive["context_fragment_id"] = groups[1].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {directive}")
                        return directive
                elif directive_type == "learn_from_text":
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "learn_from_text", "text": groups[0].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                elif directive_type == "interpret_text":
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "interpret_text", "text": groups[0].strip(), "original_line": line}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for interpret_text. Groups: {groups}")

                elif directive_type == "create_fragment":
                    if len(groups) >= 3 and groups[0] and groups[1]:
                        params = _parse_params(groups[2].strip())
                        result_dict = {"type": "create_fragment", "fragment_id": groups[0].strip(), "fragment_type": groups[1].strip(), "params": params}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for create_fragment. Groups: {groups}")

                elif directive_type == "create_fragment_from_base":
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        result_dict = {"type": "create_fragment_from_base", "new_fragment_id": groups[0].strip(), "base_fragment_id": groups[1].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for create_fragment_from_base. Groups: {groups}")

                elif directive_type == "ask":
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        input_list = _parse_json_list(groups[1])
                        if input_list is not None:
                            result_dict = {"type": "ask", "fragment_id": groups[0].strip(), "input": input_list}
                            logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                            return result_dict
                        else: logger.warning("[A3L Interp]       Failed to parse JSON list for ask.")
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for ask. Groups: {groups}")

                elif directive_type == "train":
                    if len(groups) >= 3 and groups[0] and groups[2]:
                        try:
                            epochs = int(groups[2].strip())
                            if epochs <= 0: raise ValueError("Epochs must be positive.")
                            directive = {"type": "train_fragment", "fragment_id": groups[0].strip(), "epochs": epochs}
                            if groups[1]: # Optional context_id
                                directive["context_id"] = groups[1].strip()
                            logger.debug(f"[A3L Interp]       Extracted: {directive}")
                            return directive
                        except ValueError as e:
                            logger.warning(f"[A3L Interp]       Invalid epochs in train line: {e}")
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for train. Groups: {groups}")

                elif directive_type == "reflect":
                    if len(groups) >= 1 and groups[0]:
                        directive = {"type": "reflect_fragment", "fragment_id": groups[0].strip()}
                        if len(groups) >= 2 and groups[1]: # Optional format
                            directive["format"] = groups[1].strip().lower()
                        logger.debug(f"[A3L Interp]       Extracted: {directive}")
                        return directive
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for reflect. Groups: {groups}")

                elif directive_type == "import":
                    # Assumes 'importar fragmento de <path>'
                    if len(groups) >= 1 and groups[0]:
                        directive = {"type": "import_fragment", "path": groups[0].strip()}
                        # Handle optional 'como <id>'
                        if len(groups) >= 2 and groups[1]:
                            directive["fragment_id"] = groups[1].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {directive}")
                        return directive
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for import. Groups: {groups}")

                elif directive_type == "export":
                    # Assumes 'exportar fragmento <id> para <path>'
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        result_dict = {"type": "export_fragment", "fragment_id": groups[0].strip(), "path": groups[1].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for export. Groups: {groups}")

                elif directive_type == "export_simple":
                    # Assumes 'exportar fragmento <id>'
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "export_fragment", "fragment_id": groups[0].strip()} # No path
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for export_simple. Groups: {groups}")

                # --- Conditionals ---
                elif directive_type in ["confidence_conditional", "cumulative_epochs_conditional"]:
                    if len(groups) >= 3:
                        action_string = groups[-1].strip() # Action is always last group
                        logger.debug(f"[A3L Interp]       Parsing conditional action: '{action_string}'")
                        action_directive = interpret_a3l_line(action_string) # Recursive call
                        if action_directive:
                            condition = {}
                            if directive_type == "confidence_conditional":
                                try:
                                    threshold = float(groups[1].strip())
                                    if not (0.0 <= threshold <= 1.0): raise ValueError("Threshold out of range.")
                                    condition = {"condition_type": "confidence_check", "operator": groups[0].strip().lower(), "threshold": threshold}
                                except ValueError as e: logger.warning(f"[A3L Interp]       Invalid threshold: {e}"); return None
                            elif directive_type == "cumulative_epochs_conditional":
                                try:
                                    min_epochs = int(groups[1].strip())
                                    if min_epochs < 0: raise ValueError("Min epochs cannot be negative.")
                                    condition = {"condition_type": "cumulative_epochs", "fragment_id": groups[0].strip(), "min_epochs": min_epochs}
                                except ValueError as e: logger.warning(f"[A3L Interp]       Invalid min_epochs: {e}"); return None

                            if condition: # Ensure condition was built
                                result_dict = {"type": "conditional_directive", "condition": condition, "action": action_directive} # Use a distinct type for conditionals
                                logger.debug(f"[A3L Interp]       Extracted conditional: {result_dict}")
                                return result_dict
                            else: logger.warning(f"[A3L Interp]       Failed to build condition dict for {directive_type}")
                        else:
                            logger.warning(f"[A3L Interp]       Failed to parse action '{action_string}' within conditional.")
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for {directive_type}. Groups: {groups}")

                # --- Solicitar Exemplos (NOVO) ---
                elif directive_type == "solicitar_exemplos":
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "solicitar_exemplos", "task_name": groups[0].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for solicitar_exemplos. Groups: {groups}")

                # --- Avaliar Fragmento (NOVO) ---
                elif directive_type == "avaliar_fragmento":
                    if len(groups) >= 2 and groups[0] and groups[1]:
                        result_dict = {"type": "avaliar_fragmento", "fragment_id": groups[0].strip(), "task_name": groups[1].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for avaliar_fragmento. Groups: {groups}")
                
                # --- Comparar Desempenho (NOVO) ---
                elif directive_type == "comparar_desempenho":
                     if len(groups) >= 2 and groups[0] and groups[1]:
                        result_dict = {"type": "comparar_desempenho", "fragment_id": groups[0].strip(), "task_name": groups[1].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Failed to extract groups for comparar_desempenho. Groups: {groups}")

                # --- Estudar Habilidade (NOVO - Macro) ---
                elif directive_type == "estudar_habilidade":
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "estudar_habilidade", "task_name": groups[0].strip()}
                        logger.debug(f"[A3L Interp]       Extracted Macro Command: {result_dict}")
                        return result_dict # Return the macro command itself for the processor to handle
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for estudar_habilidade. Groups: {groups}")

                # --- Planejar Dados (NOVO) ---
                elif directive_type == "planejar_dados":
                    if len(groups) >= 1 and groups[0]:
                        result_dict = {"type": "planejar_dados", "task_name": groups[0].strip()}
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Failed to extract groups for planejar_dados. Groups: {groups}")

                # --- Default fallback / Unknown match ---
                else:
                    logger.warning(f"[A3L Interp] Matched pattern '{directive_type}' but no specific extraction logic defined yet.")
                    # You might want a generic extraction based on DIRECTIVE_PARAM_MAP here as a fallback
                    # Or just return None if only explicitly handled types are valid

        except Exception as e:
             # Log errors during pattern matching/processing
             logger.error(f"[A3L Interp] Error processing pattern '{directive_type}' on line '{line}': {e}", exc_info=True)
             # Continue trying other patterns

    # If no pattern matched after trying all
    logger.warning(f"[A3L Interp] <<< Line did not match any known A3L patterns: '{line}'")
    return None

# --- Example usage ---
# Check if the script is being run directly
if __name__ == '__main__':
    # Configure basic logging ONLY for this test run
    # Use a format that includes the logger name to see source
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format) # Use DEBUG to see detailed logs

    # --- Specific Test for the problematic line ---
    print("\n--- SPECIFIC PROBLEM TEST ---")
    # String de teste com aspas duplas escapadas corretamente para Python
    problematic_line = "ask_professor 'prof_geral' question \"Quais são as melhores formas de ganhar dinheiro usando IA em 2025?\""
    print(f"Testing specific line: '{problematic_line}'")
    result = interpret_a3l_line(problematic_line)
    print(f"Result for specific line: {result}")
    expected_result = {
        "type": "ask_professor",
        "professor_id": "prof_geral",
        "question": "Quais são as melhores formas de ganhar dinheiro usando IA em 2025?"
    }
    if result == expected_result:
        print("Specific test PASSED.")
    else:
        print(f"Specific test FAILED. Expected: {expected_result}, Got: {result}")
    print("--- END SPECIFIC PROBLEM TEST ---")


    # --- Original broader test suite (optional, keep if useful) ---
    # test_lines = [
    #     "ask_professor 'prof_geral' question \\\"Quais são as melhores formas de ganhar dinheiro usando IA em 2025?\\\"\",
    #     "  ask_professor  ' outro_prof '  question  \\\"Outra pergunta?\\\"  ",
    #     "ask_professor 'prof_geral' question \\\"Pergunta com \'aspas\' simples.\\\"\",
    #     "ask_professor \'prof_geral\' question", # Bad: missing question quotes
    #     "aprender com 'prof_abc' question \\\"Como funciona X?\\\"\", # learn_directive
    #     "aprender com \'Texto livre para aprender.\'", # learn_from_text
    #     "criar fragmento \'monetizer_v1\' tipo \'neural\' input_dim 128 hidden_dims [64, 32] output_dim 1",
    #     "treinar fragmento \'f1\' por 10 épocas",
    #     "refletir sobre fragmento \'f2\'",
    #     "refletir sobre fragmento \'f3\' como a3l",
    #     "exportar fragmento \'f4\'",
    #     "exportar fragmento \'f_export\' para \'path/to/export.a3xfrag\'",
    #     "importar fragmento de \'repo/f5.a3xfrag\'",
    #     "  # Comentário",
    #     "linha inválida",
    #     ""
    # ]
    # print("\\n--- BROADER TEST SUITE ---")
    # for i, line in enumerate(test_lines):
    #     print(f"\\n[{i+1}] Testing Line: '{line}'")
    #     result = interpret_a3l_line(line)
    #     print(f"    Result: {result}")
    # print("\\n--- BROADER Test Finished ---")

# --- Handlers for specific commands ---

def _handle_evaluate(match: re.Match) -> Dict:
    fragment_id = match.group(1)
    task_name = match.group(2) or fragment_id # Default task_name to fragment_id if not specified
    split_ratio = float(match.group(3)) if match.group(3) else 0.2 # Default split 0.2
    return { "type": "avaliar_fragmento", "fragment_id": fragment_id, "task_name": task_name, "test_split": split_ratio }

def _handle_compare(match: re.Match) -> Dict:
    fragment_id = match.group(1)
    task_name = match.group(2) or fragment_id # Default task_name to fragment_id if not specified
    return { "type": "comparar_desempenho", "fragment_id": fragment_id, "task_name": task_name }

# <<< NOVO HANDLER >>>
def _handle_plan_data(match: re.Match) -> Dict:
    task_name = match.group(1)
    return { "type": "planejar_dados", "task_name": task_name }
