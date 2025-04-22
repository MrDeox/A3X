import re
import logging
from typing import List, Tuple, Dict, Optional, Any
# Add necessary imports for execution
from a3x.core.context import Context
from a3x.core.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

DEFAULT_PATTERNS = (
    # === Cognitive State / Reflection Directives ===
    (r"^refletir (?:sobre o fragmento|sobre) ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]?$", "reflect_fragment"),
    (r"^perguntar ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]? ['\"]?(?P<query_text>.+)['\"]?$", "ask"), # Simple ask
    (r"^interpretar texto ['\"]?(?P<raw_text>.+)['\"]?$", "interpret_text"),
    (r"^resolver pergunta (?P<question_id>\\d+)$", "resolve_question"), # Directive to resolve a specific pending question
    # NEW: Verify fragment knowledge qualitatively, professor optional
    (r"^verificar conhecimento (?:do fragmento|de) ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]?(?: com ajuda (?:do professor|de) ['\"]?(?P<professor_id>[a-zA-Z0-9_.-]+)['\"]?)?$", "verificar_conhecimento"),


    # === Meta / Control Flow Directives ===
    (r"^executar grafo ['\"]?(?P<graph_id>[a-zA-Z0-9_.-]+)['\"]?(?: com entrada ['\"]?(?P<input_data>.+?)['\"]?)?$", "run_graph"),
    (r"^comparar desempenho entre ['\"]?(?P<fragment_id1>[a-zA-Z0-9_.-]+)['\"]? e ['\"]?(?P<fragment_id2>[a-zA-Z0-9_.-]+)['\"]? na tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?$", "compare_performance"),

    # === Training / Adaptation Directives ===
    (r"^treinar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]? na tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?(?: por (?P<epochs>\\d+) epocas)?(?: com contexto (?P<context_id>\\d+))?(?: com precisao alvo (?P<target_accuracy>\\d+(?:\\.\\d+)?))?$", "train_fragment"),
    (r"^criar fragmento neural ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]? para tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?(?: baseado em ['\"]?(?P<base_fragment_id>[a-zA-Z0-9_.-]+)['\"]?)?(?: com tipo ['\"]?(?P<fragment_type>[a-zA-Z0-9_.-]+)['\"]?)?$", "create_neural_fragment"),
    (r"^avaliar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]? na tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?(?: com contexto (?P<context_id>\\d+))?$", "evaluate_fragment"),


    # === Memory / Knowledge Management Directives ===
    (r"^salvar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]?$", "save_fragment"),
    (r"^carregar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]?$", "load_fragment"),
    (r"^deletar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]?$", "delete_fragment"),
    (r"^listar fragmentos$", "list_fragments"),
    (r"^definir tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]? com descricao ['\"](?P<description>.+)['\"]$", "define_task"),
    (r"^listar tarefas$", "list_tasks"),
    (r"^adicionar exemplo para tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]? com entrada ['\"](?P<input_text>.+)['\"] e saida ['\"](?P<output_text>.+)['\"]$", "add_example"),
    (r"^listar exemplos para tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?$", "list_examples"),
    (r"^associar fragmento ['\"]?(?P<fragment_id>[a-zA-Z0-9_.-]+)['\"]? a tarefa ['\"]?(?P<task_name>[a-zA-Z0-9_.-]+)['\"]?$", "associate_fragment_task"),


    # === Planning / Goal Management Directives ===
    (r"^definir meta ['\"](?P<goal_description>.+)['\"]$", "set_goal"),
    (r"^listar metas$", "list_goals"),
    (r"^planejar para meta ['\"]?(?P<goal_id>\\d+)['\"]?$", "plan_for_goal"), # Assumes goals might have IDs later


    # === Context Management Directives ===
    (r"^mostrar contexto$", "show_context"),
    (r"^limpar contexto$", "clear_context"),
    (r"^salvar contexto como (?P<context_id>\\d+)$", "save_context"),
    (r"^carregar contexto (?P<context_id>\\d+)$", "load_context"),
    (r"^listar contextos salvos$", "list_saved_contexts"),

    # === System/Utility Directives ===
    (r"^sair$", "exit"),
    (r"^ajuda$", "help"),

)

class A3LangInterpreter:
    def __init__(self, custom_patterns: List[Tuple[str, str]] = None):
        """Initializes the A3LangInterpreter.

        Args:
            custom_patterns: A list of custom regex patterns and associated action names.
                           Each tuple should be (regex_pattern, action_name).
        """
        self.patterns = list(DEFAULT_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.UNICODE), action)
            for pattern, action in self.patterns
        ]
        logger.info(f"A3LangInterpreter initialized with {len(self.compiled_patterns)} patterns.")

    def interpret(self, text: str) -> Optional[Dict[str, Any]]:
        """Interprets a line of text to find a matching A3L directive.

        Args:
            text: The input text string.

        Returns:
            A dictionary containing the 'directive' name and any captured parameters,
            or None if no match is found.
        """
        text = text.strip()
        if not text:
            return None

        for compiled_pattern, action in self.compiled_patterns:
            match = compiled_pattern.match(text)
            if match:
                parameters = match.groupdict()
                # Convert numeric parameters if possible
                for key, value in parameters.items():
                    if value is not None:
                        if value.isdigit():
                            parameters[key] = int(value)
                        else:
                            try:
                                parameters[key] = float(value)
                            except ValueError:
                                pass  # Keep as string if not float
                logger.debug(f"Interpreted directive: {action}, params: {parameters}")
                return {"directive": action, **parameters}

        logger.debug(f"No A3L directive matched for text: '{text}'")
        return None

# --- Main Execution Function (Imported by mem.py) ---

async def parse_and_execute(command: str, execution_context: Optional[Context] = None) -> Dict[str, Any]:
    """
    Parses an A³L command string and executes the corresponding skill/directive.
    This is the main entry point called by mem.execute for symbolic mode.
    """
    logger.info(f"Parsing and executing A³L command: '{command}'")

    # --- Quick Check for RESPONDER command ---
    logger.debug(f"Attempting to match RESPONDER on command (type {type(command)}): {repr(command)}")
    responder_match = re.search(r'^RESPONDER\\s+(?:\"|\')(.+?)(?:\"|\')$', command, re.IGNORECASE)
    if responder_match:
        response_content = responder_match.group(1)
        logger.info(f"Handled RESPONDER command directly. Response: '{response_content[:50]}...'")
        # Return structure expected by mem.execute
        return {"status": "success", "result": response_content} 
    # --- End RESPONDER Check ---

    interpreter = A3LangInterpreter() # Instantiate the parser
    parsed_directive = interpreter.interpret(command)

    if not parsed_directive:
        logger.warning(f"Could not parse A³L command: '{command}'")
        return {"status": "error", "message": f"Comando A³L não reconhecido: {command}"}

    # Directive name is the action, remaining keys are parameters
    directive_name = parsed_directive.pop("directive", None)
    parameters = parsed_directive 

    if not directive_name:
        logger.error(f"Parsed directive missing 'directive' key for command: '{command}'")
        return {"status": "error", "message": "Erro interno no parsing do comando."}

    if not execution_context:
        logger.error(f"Execution context (Context object) is required to execute directive '{directive_name}' but none was provided.")
        return {"status": "error", "message": f"Contexto de execução necessário para executar '{directive_name}'."}

    # --- Execution Logic via ToolExecutor --- 
    # Assumes the execution_context contains a valid ToolExecutor instance
    if hasattr(execution_context, 'tool_executor') and isinstance(execution_context.tool_executor, ToolExecutor):
        logger.info(f"Executing directive '{directive_name}' via ToolExecutor...")
        try:
            # Map A3L directive name to potential skill name (assuming direct mapping for now)
            # ToolExecutor should handle finding the correct skill/tool
            skill_name = directive_name # Simple mapping for now
            
            result = await execution_context.tool_executor.execute_tool(
                tool_name=skill_name, 
                tool_input=parameters,
                context=execution_context # Pass the full context to the skill
            )
            logger.info(f"Directive '{directive_name}' execution finished via ToolExecutor.")
            
            # Ensure the result is a dictionary as expected by mem.execute
            if isinstance(result, dict):
                # Check if status exists, if not, wrap it
                if 'status' not in result:
                     logger.warning(f"Skill '{skill_name}' did not return a 'status'. Wrapping result.")
                     return { "status": "success", "data": result }
                return result
            else:
                 logger.warning(f"Skill '{skill_name}' returned non-dict type: {type(result)}. Wrapping result.")
                 return { "status": "success", "data": result }

        except Exception as e:
            logger.exception(f"Error executing directive '{directive_name}' via ToolExecutor:")
            return {"status": "error", "message": f"Erro ao executar '{directive_name}': {e}"}
    else:
        logger.error(f"Cannot execute directive '{directive_name}'. ToolExecutor not found or invalid in execution_context.")
        return {"status": "error", "message": f"Mecanismo de execução (ToolExecutor) não encontrado no contexto para '{directive_name}'."}

# Example Usage (optional, for testing)
if __name__ == '__main__':
    interpreter = A3LangInterpreter()

    test_commands = [
        "refletir sobre o fragmento 'analisador_sentimento'",
        "perguntar 'tradutor_pt_en' 'Como se diz \"bom dia\" em inglês?'",
        "interpretar texto 'Qual a previsão do tempo para amanhã?'",
        "executar grafo 'processamento_pedido' com entrada '{\"item\": \"livro\", \"quantidade\": 2}'",
        "comparar desempenho entre 'modelo_a' e 'modelo_b' na tarefa 'classificacao_spam'",
        "treinar fragmento 'ner_model' na tarefa 'identificacao_entidades' por 10 epocas com precisao alvo 0.95",
        "criar fragmento neural 'classificador_imagem' para tarefa 'cifar10' com tipo 'resnet'",
        "avaliar fragmento 'detector_fraude' na tarefa 'transacoes_bancarias' com contexto 12345",
        "resolver pergunta 123",
        "salvar fragmento 'meu_modelo_final'",
        "carregar fragmento 'outro_modelo'",
        "deletar fragmento 'modelo_temporario'",
        "listar fragmentos",
        "definir tarefa 'reconhecimento_voz' com descricao 'Transcrever áudio em texto.'",
        "listar tarefas",
        "adicionar exemplo para tarefa 'traducao_fr_es' com entrada 'Bonjour le monde' e saida 'Hola mundo'",
        "listar exemplos para tarefa 'sumarizacao_texto'",
        "associar fragmento 'sumarizador_bart' a tarefa 'sumarizacao_texto'",
        "definir meta 'Aumentar a precisão da tradução em 10%'",
        "listar metas",
        "planejar para meta '1'",
        "mostrar contexto",
        "limpar contexto",
        "salvar contexto como 1",
        "carregar contexto 1",
        "listar contextos salvos",
        "sair",
        "ajuda",
        "verificar conhecimento do fragmento 'ner_model_v2' com ajuda do professor 'prof_especialista'",
        "verificar conhecimento de 'sumarizador_bart'", # Test without explicit professor
    ]

    for command in test_commands:
        result = interpreter.interpret(command)
        print(f"Command: {command}")
        print(f"Result: {result}")
        print("---") 