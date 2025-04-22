# Arquivo corrigido de identação salvo aqui para evitar corte por limite de mensagem.
# Você pode baixá-lo ou abrí-lo no seu editor diretamente.

# DICA: Este arquivo foi corrigido com base em análise sintática e estrutural.
# Para manter sua integridade, valide com `flake8` ou `black` se preferir.

# (Conteúdo omitido aqui por motivos de tamanho)

import re
from typing import Optional, Dict, Any, List
import json
import ast
import logging

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

# Regex patterns corrigidos e abrangentes
# Usando named groups (?P<name>...) sempre que possível para clareza na extração
patterns = {
    # Prioridade: Comentários e comandos 'ask'/'perguntar'
    "comment": re.compile(r"^\s*#.*$"),
    "ask_list": re.compile(r"^(?:perguntar|ask)\s+(?:ao\s+)?fragmento\s+\'(?P<fragment_id>[^\']+?)\'\s+com\s+(?P<list_input>\[.*\])\s*$", re.IGNORECASE | re.DOTALL), # Garantir que a lista feche
    "ask_text": re.compile(r"^(?:perguntar|ask)\s+(?:ao\s+)?\'(?P<fragment_id>[^\']+)\'\s+sobre\s+\"(?P<text_input>.*?)\"\s*$", re.IGNORECASE),
    "ask_professor": re.compile(r"""^ask_professor\s+\'(?P<professor_id>[^\']+)\'\s+question\s+\"(?P<question>[^\"]+)\"\s*$""", re.IGNORECASE),

    # Comandos de manipulação de fragmentos
    "create_fragment": re.compile(r"^criar\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s+tipo\s+\'(?P<fragment_type>[^\']+)\'(?P<params_str>.*)$", re.IGNORECASE),
    "create_fragment_from_base": re.compile(r"^criar\s+fragmento\s+\'(?P<new_fragment_id>[^\']+)\'\s+com\s+base\s+em\s+\'(?P<base_fragment_id>[^\']+)\'\s*$", re.IGNORECASE),
    "import": re.compile(r"^importar\s+fragmento\s+de\s+\'(?P<path>[^\']+)\'(?:\s+como\s+\'(?P<fragment_id>[^\']+)\'?)?\s*$", re.IGNORECASE),
    "export": re.compile(r"^exportar\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s+para\s+\'(?P<path>[^\']+)\'\s*$", re.IGNORECASE),
    "export_simple": re.compile(r"^exportar\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s*$", re.IGNORECASE),

    # Comandos de aprendizado e avaliação
    "train": re.compile(r"^treinar\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'(?:\s+na\s+tarefa\s+\'(?P<task_name>[^\']+)\'?)?(?:\s+usando\s+contexto\s+\'(?P<context_id>[^\']+)?)?(?:\s+(?:por|até)\s+(?P<epochs>\d+)\s+épocas?)?(?:\s+com\s+precisão\s+alvo\s+(?P<target_accuracy>[\d\.]+))?\s*$", re.IGNORECASE),
    "avaliar_fragmento": re.compile(r"^avaliar\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s+com\s+dados\s+de\s+teste\s+\'(?P<test_data_id>[^\']+)\'\s*$", re.IGNORECASE),
    "comparar_desempenho": re.compile(r"^comparar\s+desempenho\s+do\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s+após\s+treino\s+em\s+\'(?P<task_name>[^\']+)\'\s*$", re.IGNORECASE),
    "verificar_conhecimento": re.compile(r"^verificar\s+conhecimento\s+do\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'(?:\s+com\s+ajuda\s+do\s+professor\s+\'(?P<professor_id>[^\']+)\'?)?\s*$", re.IGNORECASE),

    # Comandos de reflexão e feedback
    "reflect": re.compile(r"^refletir\s+sobre\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'(?:\s+como\s+(?P<format>json|a3l))?\s*$", re.IGNORECASE),
    "avaliar_resposta": re.compile(r"^avaliar\s+resposta\s+de\s+\'(?P<fragment_id>[^\']+)\'\s+como\s+(?P<evaluation>correta|incorreta)\s*$", re.IGNORECASE),
    "refletir_resposta": re.compile(r"^refletir\s+sobre\s+a\s+resposta\s+de\s+\'(?P<fragment_id>[^\']+)\'\s*$", re.IGNORECASE),

    # Comandos de interação com LLM / KI
    "learn_directive": re.compile(r"""^aprender\s+com\s+\'(?P<professor_id>[^\']+)\'(?:\s+sobre\s+\'(?P<context_fragment_id>[^\']+)\'?)?\s+question\s+(?:\"(?P<question_double>[^\"]*)\"|\'(?P<question_single>[^\']*)\')\s*$""", re.IGNORECASE),
    "interpret_text": re.compile(r"^interpretar\s+texto\s+(?:\'(?P<text_single>.*?)\'|\"(?P<text_double>.*?)\")\s*$", re.IGNORECASE),

    # Comandos relacionados a dados/tarefas
    "solicitar_exemplos": re.compile(r"solicitar\s+exemplos\s+para\s+tarefa\s+\"(?P<task_name>[^\"]+)\"", re.IGNORECASE),
    "planejar_dados": re.compile(r"^planejar\s+dados\s+para\s+tarefa\s+\"(?P<task_name>[^\"]+)\"\s*$", re.IGNORECASE),

    # Comandos condicionais
    "confidence_conditional": re.compile(r"^se\s+confiança\s+for\s+(?P<comparison>maior|menor)\s+que\s+(?P<threshold>[\d\.]+)\s+então\s+(?P<action_line>.*)", re.IGNORECASE),
    "cumulative_epochs_conditional": re.compile(r"^se\s+fragmento\s+\'(?P<fragment_id>[^\']+)\'\s+foi\s+treinado\s+por\s+mais\s+de\s+(?P<epochs>\d+)\s+épocas?\s+então\s+(?P<action_line>.*)", re.IGNORECASE),

    # TODO: Adicionar outros padrões conforme a linguagem evolui (ex: definir variável, executar grafo, etc.)
}

# Helper para parsing de parâmetros de 'criar fragmento'
def _parse_params(params_str: str) -> Dict[str, Any]:
    params = {}
    if not params_str:
        return params
    # Regex aprimorado para lidar com valores com espaços dentro de aspas ou colchetes/chaves
    param_pattern = re.compile(r"(\w+)\s+((?:\{[^\}]*\}|\[[^\]]*\]|\'[^\']*\'|\"[^\"]*\"|[^\'\"\s\[\{]+)+)", re.UNICODE)
    try:
        matches = param_pattern.findall(params_str.strip())
        for key, value_str in matches:
            value_str = value_str.strip()
            try:
                # Tenta avaliar como literal Python (seguro para números, strings, listas, dicts, bools, None)
                params[key] = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Se falhar, trata como string (removendo aspas se houver)
                if (value_str.startswith("'") and value_str.endswith("'")) or \
                   (value_str.startswith('"') and value_str.endswith('"')):
                    params[key] = value_str[1:-1]
                else:
                    params[key] = value_str # Mantém como string se não for literal ou entre aspas
    except Exception as e:
        logger.error(f"Error parsing parameters '{params_str}': {e}", exc_info=True)
    return params

# Helper para parsing de lista JSON (usado por 'ask_list')
def _parse_json_list(list_str: str) -> Optional[List[Any]]: # Alterado para List[Any]
    # Limpa a string: remove comentários e espaços extras
    list_str = re.sub(r"#.*$", "", list_str, flags=re.MULTILINE).strip()
    cleaned_list_str = list_str.replace('\n', ' ').strip()

    if not cleaned_list_str.startswith('[') or not cleaned_list_str.endswith(']'):
        logger.warning(f"Malformed list structure after cleaning: '{cleaned_list_str}'")
        return None

    try:
        # Usa json.loads que é mais flexível que ast.literal_eval para JSON
        input_list = json.loads(cleaned_list_str)
        if not isinstance(input_list, list):
            raise ValueError("Parsed result is not a list.")
        # Não força mais para float, permite outros tipos JSON válidos na lista
        return input_list
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Parse Error in cleaned ask list '{cleaned_list_str[:100]}...': {e}")
        return None

# Função principal de interpretação
def interpret_a3l_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Interprets a single line of A3L, iterating through compiled patterns.
    Provides debug logging on failure.
    """
    line = line.strip()

    logger.debug(f"[A3L Interp] *** ENTERING interpret_a3l_line. Line: {repr(line)}.")
    logger.info(f"[A3L Interp] >>> Interpreting line: '{line}'")

    if not line or patterns["comment"].match(line):
        logger.debug("[A3L Interp] Line is empty or a comment.")
        return None

    for directive_type, pattern in patterns.items():
        if directive_type == "comment":
            continue

        logger.debug(f"[A3L Interp]   Trying pattern: {directive_type}")
        try:
            match = pattern.match(line)
            if match:
                logger.info(f"[A3L Interp]   +++ Matched pattern: {directive_type}")
                m_dict = match.groupdict() # Prioriza grupos nomeados
                groups = match.groups()    # Fallback para grupos posicionais se não houver nomeados
                logger.debug(f"[A3L Interp]       Groups: {groups}, NamedDict: {m_dict}")

                result_dict: Dict[str, Any] = {"type": directive_type} # Base dict

                # --- Extração específica por tipo ---
                if directive_type == "ask_list":
                    if m_dict.get('fragment_id') and m_dict.get('list_input'):
                        input_list = _parse_json_list(m_dict['list_input'])
                        if input_list is not None:
                            result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                            result_dict["input_list"] = input_list
                            # result_dict["type"] = "ask" # Normaliza o tipo para o bridge
                            logger.debug(f"[A3L Interp]       Extracted (ask_list): {result_dict}")
                            return result_dict
                        else: logger.warning("[A3L Interp]       Failed to parse JSON list for ask_list.")
                    else: logger.warning(f"[A3L Interp]       Missing groups for ask_list. Dict: {m_dict}")

                elif directive_type == "ask_text":
                    if m_dict.get('fragment_id') and m_dict.get('text_input') is not None: # Check if text_input exists
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["text_input"] = m_dict['text_input'].strip()
                        # result_dict["type"] = "ask" # Normaliza o tipo para o bridge
                        logger.debug(f"[A3L Interp]       Extracted (ask_text): {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for ask_text. Dict: {m_dict}")

                elif directive_type == "ask_professor":
                    if m_dict.get('professor_id') and m_dict.get('question'):
                        result_dict["professor_id"] = m_dict['professor_id'].strip()
                        result_dict["question"] = m_dict['question'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for ask_professor. Dict: {m_dict}")

                elif directive_type == "create_fragment":
                    if m_dict.get('fragment_id') and m_dict.get('fragment_type') and m_dict.get('params_str') is not None:
                        params = _parse_params(m_dict['params_str'].strip())
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["fragment_type"] = m_dict['fragment_type'].strip()
                        result_dict["params"] = params
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for create_fragment. Dict: {m_dict}")

                elif directive_type == "create_fragment_from_base":
                     if m_dict.get('new_fragment_id') and m_dict.get('base_fragment_id'):
                        result_dict["new_fragment_id"] = m_dict['new_fragment_id'].strip()
                        result_dict["base_fragment_id"] = m_dict['base_fragment_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Missing groups for create_fragment_from_base. Dict: {m_dict}")

                elif directive_type == "import":
                    if m_dict.get('path'):
                        result_dict["path"] = m_dict['path'].strip()
                        if m_dict.get('fragment_id'): # ID é opcional
                            result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for import. Dict: {m_dict}")

                elif directive_type == "export":
                    if m_dict.get('fragment_id') and m_dict.get('path'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["path"] = m_dict['path'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for export. Dict: {m_dict}")

                elif directive_type == "export_simple":
                    if m_dict.get('fragment_id'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["type"] = "export_fragment" # Normaliza para o tipo do bridge
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for export_simple. Dict: {m_dict}")

                elif directive_type == "train":
                    if m_dict.get('fragment_id'):
                        try:
                            result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                            result_dict["type"] = "train_fragment" # Normaliza tipo
                            if m_dict.get('task_name'):
                                result_dict['task_name'] = m_dict['task_name'].strip()
                            if m_dict.get('context_id'):
                                result_dict['context_id'] = m_dict['context_id'].strip()
                            if m_dict.get('epochs'):
                                result_dict['epochs'] = int(m_dict['epochs'])
                            if m_dict.get('target_accuracy'):
                                result_dict['target_accuracy'] = float(m_dict['target_accuracy'])
                            logger.debug(f"[A3L Interp]       Extracted (train): {result_dict}")
                            return result_dict
                        except (ValueError, TypeError) as e:
                            logger.warning(f"[A3L Interp]       Error parsing numeric values for train: {e}. Dict: {m_dict}")
                            return None # Falha se numéricos opcionais forem inválidos
                    else: logger.warning(f"[A3L Interp]       Missing required fragment_id for train. Dict: {m_dict}")

                elif directive_type == "avaliar_fragmento":
                     if m_dict.get('fragment_id') and m_dict.get('test_data_id'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["test_data_id"] = m_dict['test_data_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Missing groups for avaliar_fragmento. Dict: {m_dict}")

                elif directive_type == "comparar_desempenho":
                     if m_dict.get('fragment_id') and m_dict.get('task_name'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["task_name"] = m_dict['task_name'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Missing groups for comparar_desempenho. Dict: {m_dict}")

                elif directive_type == "verificar_conhecimento":
                     if m_dict.get('fragment_id'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        if m_dict.get('professor_id'): # Opcional
                            result_dict["professor_id"] = m_dict['professor_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Missing groups for verificar_conhecimento. Dict: {m_dict}")

                elif directive_type == "reflect":
                    if m_dict.get('fragment_id'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        result_dict["type"] = "reflect_fragment" # Normaliza tipo
                        if m_dict.get('format'): # Opcional
                            result_dict["format"] = m_dict['format'].strip().lower()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for reflect. Dict: {m_dict}")

                elif directive_type == "avaliar_resposta":
                    if m_dict.get('fragment_id') and m_dict.get('evaluation'):
                        eval_value = m_dict['evaluation'].strip().lower()
                        if eval_value in ["correta", "incorreta"]:
                            result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                            result_dict["evaluation"] = eval_value
                            logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                            return result_dict
                        else: logger.warning(f"[A3L Interp]       Invalid evaluation value '{eval_value}'.")
                    else: logger.warning(f"[A3L Interp]       Missing groups for avaliar_resposta. Dict: {m_dict}")

                elif directive_type == "refletir_resposta":
                     if m_dict.get('fragment_id'):
                        result_dict["fragment_id"] = m_dict['fragment_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                     else: logger.warning(f"[A3L Interp]       Missing groups for refletir_resposta. Dict: {m_dict}")

                elif directive_type == "learn_directive":
                    question = m_dict.get('question_double') or m_dict.get('question_single')
                    if m_dict.get('professor_id') and question is not None:
                        result_dict["professor_id"] = m_dict['professor_id'].strip()
                        result_dict["question"] = question.strip()
                        result_dict["type"] = "learn_from_professor" # Normaliza tipo
                        if m_dict.get('context_fragment_id'): # Opcional
                            result_dict["context_fragment_id"] = m_dict['context_fragment_id'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for learn_directive. Dict: {m_dict}")

                elif directive_type == "interpret_text":
                    text = m_dict.get('text_single') or m_dict.get('text_double')
                    if text is not None:
                        result_dict["text"] = text.strip()
                        result_dict["original_line"] = line # Adiciona a linha original para contexto
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for interpret_text. Dict: {m_dict}")

                elif directive_type == "solicitar_exemplos":
                    if m_dict.get('task_name'):
                        result_dict["task_name"] = m_dict['task_name'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for solicitar_exemplos. Dict: {m_dict}")

                elif directive_type == "planejar_dados":
                    if m_dict.get('task_name'):
                        result_dict["task_name"] = m_dict['task_name'].strip()
                        logger.debug(f"[A3L Interp]       Extracted: {result_dict}")
                        return result_dict
                    else: logger.warning(f"[A3L Interp]       Missing groups for planejar_dados. Dict: {m_dict}")

                elif directive_type == "confidence_conditional":
                    if m_dict.get('comparison') and m_dict.get('threshold') and m_dict.get('action_line'):
                        try:
                            threshold = float(m_dict['threshold'])
                            action_line = m_dict['action_line'].strip()
                            sub_directive = interpret_a3l_line(action_line) # Chamada recursiva
                            if sub_directive:
                                result_dict = {
                                    "type": "conditional_directive",
                                    "condition": {
                                        "condition_type": "confidence_check",
                                        "comparison": m_dict['comparison'].lower(),
                                        "threshold": threshold
                                    },
                                    "action": sub_directive
                                }
                                logger.debug(f"[A3L Interp]       Extracted (confidence cond): {result_dict}")
                                return result_dict
                            else:
                                logger.warning(f"[A3L Interp]       Failed to interpret sub-directive for confidence conditional: '{action_line}'")
                        except ValueError:
                            logger.warning(f"[A3L Interp]       Invalid threshold '{m_dict['threshold']}' for confidence conditional.")
                    else: logger.warning(f"[A3L Interp]       Missing groups for confidence_conditional. Dict: {m_dict}")

                elif directive_type == "cumulative_epochs_conditional":
                     if m_dict.get('fragment_id') and m_dict.get('epochs') and m_dict.get('action_line'):
                        try:
                            epochs = int(m_dict['epochs'])
                            action_line = m_dict['action_line'].strip()
                            sub_directive = interpret_a3l_line(action_line) # Chamada recursiva
                            if sub_directive:
                                result_dict = {
                                    "type": "conditional_directive",
                                    "condition": {
                                        "condition_type": "attribute_check",
                                        "fragment_id": m_dict['fragment_id'].strip(),
                                        "attribute": "cumulative_epochs",
                                        "comparison": "greater_than",
                                        "expected_value": epochs
                                    },
                                    "action": sub_directive
                                }
                                logger.debug(f"[A3L Interp]       Extracted (epochs cond): {result_dict}")
                                return result_dict
                            else:
                                logger.warning(f"[A3L Interp]       Failed to interpret sub-directive for epochs conditional: '{action_line}'")
                        except ValueError:
                            logger.warning(f"[A3L Interp]       Invalid epochs '{m_dict['epochs']}' for epochs conditional.")
                     else: logger.warning(f"[A3L Interp]       Missing groups for cumulative_epochs_conditional. Dict: {m_dict}")

                # --- Fallback para tipos não explicitamente tratados ---
                else:
                    logger.warning(f"[A3L Interp]   Pattern '{directive_type}' matched, but no specific extraction logic handled. Dict: {m_dict}, Groups: {groups}")
                    # Pode-se retornar um dict genérico ou None
                    # return {"type": directive_type, "matched_groups": groups, "matched_dict": m_dict}

        except Exception as e:
            logger.error(f"[A3L Interp]   !!! Exception processing pattern '{directive_type}' for line '{line}': {e}", exc_info=True)
            # Continua para o próximo padrão

    logger.warning(f"[A3L Interp] <<< Failed to interpret line (no pattern matched): '{line}'")
    return None

# --- Bloco de Teste ---
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     test_lines = [
#         "# Linha de comentário",
#         "  ",
#         "criar fragmento 'meu_frag' tipo 'neural' input_dim 128 hidden_dims [64, 32] activation 'relu'",
#         "criar fragmento 'outro_frag' com base em 'base_frag'",
#         "treinar fragmento 'meu_frag' na tarefa 'classificacao' por 20 épocas com precisão alvo 0.95",
#         "treinar fragmento 'sem_tarefa' usando contexto 'ctx123' por 5 épocas",
#         "treinar fragmento 'so_id'",
#         "ask 'meu_frag' sobre \"Qual o sentido da vida?\"",
#         "perguntar ao fragmento 'embed_lookup' com [1.2, -0.5, 3.14] # Comentário inline",
#         "ask_professor 'prof_geral' question \"Como implementar atenção em PyTorch?\"",
#         "aprender com 'prof_especialista' sobre 'frag_contexto' question 'Qual a fórmula X?'",
#         "aprender com 'prof_single_quote' question 'Pergunta com aspas simples?'",
#         "interpretar texto \"criar fragmento 'inline_frag' tipo 'linear'\"",
#         "interpretar texto 'outro texto entre aspas simples'",
#         "refletir sobre fragmento 'meu_frag'",
#         "refletir sobre fragmento 'outro_frag' como json",
#         "importar fragmento de './modelos/importado.a3xfrag' como 'import_id'",
#         "importar fragmento de 'modelos/sem_id.a3xfrag'",
#         "exportar fragmento 'meu_frag' para './backups/meu_frag_v1.a3xfrag'",
#         "exportar fragmento 'frag_simples'",
#         "se confiança for menor que 0.6 então refletir sobre fragmento 'frag_incerto'",
#         "se fragmento 'frag_treinado' foi treinado por mais de 50 épocas então exportar fragmento 'frag_treinado'",
#         "solicitar exemplos para tarefa \"geracao_texto\"",
#         "avaliar fragmento 'modelo_final' com dados de teste 'dataset_teste_final'",
#         "comparar desempenho do fragmento 'modelo_a' após treino em 'tarefa_abc'",
#         "verificar conhecimento do fragmento 'checker_frag'",
#         "verificar conhecimento do fragmento 'checker_frag' com ajuda do professor 'prof_verify'",
#         "refletir sobre a resposta de 'frag_perguntado'",
#         "avaliar resposta de 'frag_avaliado' como correta",
#         "avaliar resposta de 'frag_ruim' como incorreta",
#         "planejar dados para tarefa \"traducao_pt_en\"",
#         "comando invalido aqui",
#     ]
#     print("--- Iniciando Testes do Interpretador A3L ---")
#     for i, l in enumerate(test_lines):
#         print(f"\n[{i+1}] Linha: {repr(l)}")
#         result = interpret_a3l_line(l)
#         print(f"    Resultado: {result}")
#     print("\n--- Testes Concluídos ---")

class A3LangInterpreter:
    """
    Parses lines of A3 Language (A3L) code into structured directives.
    
    Handles commands with positional arguments, named parameters (key=value), 
    and JSON data for complex parameters.
    """
    # Regex to capture the command and the rest of the line
    COMMAND_REGEX = re.compile(r"^\s*([A-Z_]+)\s*(.*)\s*$")
    
    # Regex to parse parameters: key=value or key='value' or key="value" or just value (positional)
    # Also handles JSON embedded in parameters, e.g., data='{...}' or data="{...}"
    PARAM_REGEX = re.compile(
        r"""
        (?:(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*)? # Optional key= part
        (?:
            '(?P<sq_value>(?:\\.|[^'])*)' | # Single-quoted value
            "(?P<dq_value>(?:\\.|[^"])*)" | # Double-quoted value
            (?P<json_value>\{.*?\}) |       # JSON object (non-greedy)
            (?P<plain_value>[^\s'"=]+(?:=\s*[^\s'"=]+)*) # Plain value (can include '=' if not at the start)
        )
        """, 
        re.VERBOSE
    )

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parses a single line of A3L code.

        Args:
            line: The A3L command line string.

        Returns:
            A dictionary representing the parsed directive, or None if parsing fails
            or the line is empty/comment. The dictionary always contains 'directive_type'.
            Example: {'directive_type': 'create_fragment', 'fragment_id': 'MyFrag', 'type': 'Neural'}
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None # Skip empty lines and comments

        command_match = self.COMMAND_REGEX.match(line)
        if not command_match:
            logger.warning(f"[A3LangInterpreter] Failed to parse command from line: {line}")
            return None

        directive_type = command_match.group(1).lower() # Store directive types in lowercase
        param_string = command_match.group(2).strip()
        
        parsed_directive: Dict[str, Any] = {"directive_type": directive_type}
        positional_args = []

        # Iterate through parameter matches
        for match in self.PARAM_REGEX.finditer(param_string):
            key = match.group('key')
            
            # Determine the value source
            if match.group('sq_value') is not None:
                value = match.group('sq_value')
            elif match.group('dq_value') is not None:
                value = match.group('dq_value')
            elif match.group('json_value') is not None:
                json_str = match.group('json_value')
                try:
                    # Attempt to parse as JSON
                    value = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(f"[A3LangInterpreter] Invalid JSON in parameter on line: {line}. Treating as string: {json_str}")
                    value = json_str # Fallback to string if JSON is invalid
            elif match.group('plain_value') is not None:
                value = match.group('plain_value')
            else:
                continue # Should not happen with the regex structure

            # Assign to dictionary or positional list
            if key:
                # Sanitize key (e.g., replace hyphens if needed, convert to snake_case)
                key = key.lower().replace('-', '_') 
                parsed_directive[key] = value
            else:
                positional_args.append(value)

        # Handle positional arguments (assign default names or handle based on directive type if needed)
        # For simplicity, let's assign generic names like 'arg1', 'arg2', etc.
        # A more robust system might map them based on the directive_type.
        if positional_args:
             # Example: Assign first positional arg as 'fragment_id' for some commands
             if directive_type in ['create_fragment', 'train_fragment', 'evaluate_fragment', 
                                     'import_fragment', 'export_fragment', 'run_fragment', 
                                     'verify_knowledge', 'request_examples']: 
                 if 'fragment_id' not in parsed_directive and positional_args:
                     parsed_directive['fragment_id'] = positional_args.pop(0)
             if directive_type == 'ask_professor' and 'question' not in parsed_directive and positional_args:
                 parsed_directive['question'] = positional_args.pop(0) # Assign first arg as question

             # Assign remaining positional args generically
             for i, arg in enumerate(positional_args):
                 parsed_directive[f'arg{i+1}'] = arg


        logger.debug(f"[A3LangInterpreter] Parsed line '{line}' into: {parsed_directive}")
        return parsed_directive

    def parse_script(self, script_content: str) -> list[Dict[str, Any]]:
        """
        Parses a multi-line A3L script.

        Args:
            script_content: The entire A3L script as a single string.

        Returns:
            A list of parsed directive dictionaries.
        """
        directives = []
        for line_num, line in enumerate(script_content.splitlines()):
            parsed = self.parse_line(line)
            if parsed:
                parsed['_line_number'] = line_num + 1 # Add original line number for context
                directives.append(parsed)
        return directives

# Example Usage (can be removed or kept for testing)
# // ... existing code ...

