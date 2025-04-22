import torch
import torch.nn as nn
import re
import logging
import json
import time
from typing import List, Dict, Optional, Any, Tuple, Callable, Awaitable
import asyncio # <<< Added asyncio
from pathlib import Path

# --- A3X Imports ---
# from a3x.a3net.integration.a3x_bridge import MEMORY_BANK

# --- Import ProfessorLLMFragment conditionally for type hinting --- 
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .professor_llm_fragment import ProfessorLLMFragment
    from .neural_language_fragment import NeuralLanguageFragment

# <<< Import post_message_handler from run.py (assuming it's made available) >>>
# Need to ensure this import works based on project structure
# Or pass post_message_handler to the KI instance
# from ..run import post_message_handler # <<< COMMENT OUT DIRECT IMPORT - RELY ON INJECTION

# <<< Import interpret_a3l_line >>>
from ..integration.a3lang_interpreter import interpret_a3l_line

# Assuming these core components are accessible
from .memory_bank import MemoryBank
from .context_store import ContextStore
# <<< REMOVED incorrect import >>>
# from .llm_interface import LLMInterface 
# Assuming ProfessorLLMFragment is the type hint, not necessarily instance here
from .professor_llm_fragment import ProfessorLLMFragment 

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.7 # Example threshold

# --- A3L Command Patterns ---
# NOTE: Added VERIFY_KNOWLEDGE
# CREATE: [DEPRECATED - now handled by prefix check]
# TRAIN: (verb like treinar/ajustar/etc) fragmento 'nome' por N épocas
TRAIN_PATTERN = r"(?:treinar|treina-lo|treinado|ajustar|refinar)\b\s+fragmento\s+'([^']+)'\s+por\s+(\d+)\s+épocas?"
# REFLECT: refletir [sobre [o]] fragmento 'nome' [como a3l|json]
REFLECT_PATTERN = r"(?:refletir)\s+(?:sobre)?\s*(?:o)?\s*fragmento\s+'([^']+)'(?:\s*como\s+(a3l|json))?" # Default format is a3l
# ASK: perguntar/consultar [ao/o] fragmento 'nome' com/usando <input>
# Input capture remains non-greedy for general text, explicit for lists.
ASK_PATTERN = r"(?:perguntar|consultar)\s+(?:ao|o)?\s*fragmento\s+'([^']+)'\s+(?:com|usando)\s+(\[.*\]|.+?)"
# VERIFY_KNOWLEDGE: verificar conhecimento do fragmento 'nome' [com ajuda do professor 'prof_id']
VERIFY_KNOWLEDGE_PATTERN = r"verificar\s+conhecimento\s+d[oa]\s+fragmento\s+'([^']+)'(?:\s+com\s+ajuda\s+d[oa]\s+professor\s+'([^']+)')?"

# Type alias for message handler
MessageHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]

class KnowledgeInterpreterFragment(nn.Module):
    """
    Interpreta texto bruto (ex: respostas de LLM, logs) para extrair 
    sugestões simbólicas e convertê-las em comandos A3L executáveis.
    
    Este fragmento atua como um tradutor de conhecimento em linguagem natural 
    para a linguagem estruturada A3L, facilitando o aprendizado e a 
    adaptação do sistema com base em insights textuais.
    
    Pode consultar um ProfessorLLMFragment se a interpretação direta falhar.
    """
    
    def __init__(self, fragment_id: str,
                 professor_fragment: Optional[ProfessorLLMFragment] = None,
                 context_store: Optional[ContextStore] = None,
                 memory_bank: Optional[MemoryBank] = None,
                 llm_client: Optional[Any] = None,
                 post_chat_message_callback: Optional[MessageHandler] = None,
                 description: str = "Interprets natural language and extracts knowledge/commands.",
                 grammar_file_path: str = "a3x/a3lang/grammars/a3l_simple.gbnf",
                 prompt_template_path: str = "a3x/prompts/a3net/knowledge_interpreter_prompt.txt"):
        """Inicializa o KnowledgeInterpreterFragment.

        Args:
            fragment_id: Identificador único do fragmento.
            professor_fragment: O ProfessorLLMFragment associado ao fragmento.
            context_store: O ContextStore associado ao fragmento.
            memory_bank: O MemoryBank associado ao fragmento.
            llm_client: O cliente LLM associado ao fragmento.
            post_chat_message_callback: O callback para postar mensagens de chat.
            description: A descrição textual da sua função.
            grammar_file_path: O caminho para o arquivo de gramática.
            prompt_template_path: O caminho para o arquivo de template de prompt.
        """
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.professor = professor_fragment
        self.context_store = context_store
        self.memory_bank = memory_bank
        self.post_chat_message = post_chat_message_callback
        self.llm_client = llm_client
        self.prompt_template_path = Path(prompt_template_path)
        self.grammar_file_path = Path(grammar_file_path)

        # Load grammar
        self.grammar = None
        try:
            if self.grammar_file_path.is_file():
                 self.grammar = self.grammar_file_path.read_text()
                 logger.info(f"KI '{self.fragment_id}' loaded grammar from: {self.grammar_file_path}")
            else:
                 logger.warning(f"KI '{self.fragment_id}' grammar file not found at: {self.grammar_file_path}")
        except Exception as e:
            logger.error(f"KI '{self.fragment_id}' failed to load grammar: {e}", exc_info=True)

        # Load prompt template
        self.prompt_template = "" # Default empty template
        try:
            if self.prompt_template_path.is_file():
                 self.prompt_template = self.prompt_template_path.read_text()
                 logger.info(f"KI '{self.fragment_id}' loaded prompt template from: {self.prompt_template_path}")
            else:
                 logger.warning(f"KI '{self.fragment_id}' prompt template file not found at: {self.prompt_template_path}")
        except Exception as e:
            logger.error(f"KI '{self.fragment_id}' failed to load prompt template: {e}", exc_info=True)

        logger.info(f"KnowledgeInterpreterFragment '{self.fragment_id}' initialized.")
        if not self.context_store: logger.warning(f"KI '{self.fragment_id}' initialized without ContextStore.")
        if not self.memory_bank: logger.warning(f"KI '{self.fragment_id}' initialized without MemoryBank.")
        if not self.post_chat_message: logger.warning(f"KI '{self.fragment_id}' initialized without post_chat_message callback.")
        if not self.professor: logger.warning(f"KI '{self.fragment_id}' initialized without Professor.")

    # --- NEW: Asynchronous Message Handler ---
    async def handle_message(self, message_type: str, content: Any, **kwargs):
        """Handles incoming messages from the queue, specifically 'a3l_directive' types.
        
        Args:
            message_type: The type of the message (e.g., 'a3l_directive').
            content: The message payload.
            **kwargs: Additional keyword arguments (e.g., source_fragment_id).
        """
        logger.info(f"[{self.fragment_id}] Received message. Type: {message_type}, Content Keys: {list(content.keys()) if isinstance(content, dict) else type(content)}")

        if message_type == "a3l_directive":
            if isinstance(content, dict) and content.get("type") == "interpret_text" and isinstance(content.get("text"), str):
                text_to_interpret = content["text"]
                origin = content.get("origin", "Unknown Message Origin") # Get origin if provided
                logger.info(f"[{self.fragment_id}] Handling 'interpret_text' directive from '{origin}'. Text: '{text_to_interpret[:100]}...' ")
                try:
                    # --- Call interpret_knowledge --- 
                    # TODO: Consider if context_fragment_id from the message is useful here
                    extracted_commands, metadata = await self.interpret_knowledge(text_to_interpret)
                    
                    # Log the outcome and RE-ENQUEUE commands
                    if extracted_commands:
                        logger.info(f"[{self.fragment_id}] Interpreted {len(extracted_commands)} commands from '{origin}'. (Commands: {extracted_commands}) Metadata: {metadata}")
                        
                        # <<< ADD: Re-enqueue extracted commands >>>
                        if self.post_chat_message: # Check if handler is set
                            logger.info(f"[{self.fragment_id}] Re-enqueuing {len(extracted_commands)} extracted commands...")
                            for cmd_index, cmd_str in enumerate(extracted_commands):
                                try:
                                    # Re-interpret the string command back into a directive dictionary
                                    reinterpreted_directive = interpret_a3l_line(cmd_str)
                                    if reinterpreted_directive:
                                        # Add origin information
                                        reinterpreted_directive["_origin"] = f"Extracted by KI ({self.fragment_id}) from {origin}"
                                        # Post back to the queue for the Executor
                                        await self.post_chat_message(
                                            message_type="a3l_command",
                                            content=reinterpreted_directive,
                                            target_fragment="Executor"
                                        )
                                        logger.info(f"[{self.fragment_id}] Command {cmd_index+1} re-enqueued: {cmd_str}")
                                    else:
                                        logger.warning(f"[{self.fragment_id}] Failed to re-interpret extracted command {cmd_index+1}: {cmd_str}")
                                except Exception as reinterpr_err:
                                    logger.error(f"[{self.fragment_id}] Error re-interpreting/enqueuing command '{cmd_str}': {reinterpr_err}", exc_info=True)
                        else:
                             logger.error(f"[{self.fragment_id}] Cannot re-enqueue commands: post_chat_message not set for this KI instance.")
                        # <<< END ADD >>>
                        
                    elif metadata.get("pergunta_pendente"):
                         logger.info(f"[{self.fragment_id}] Interpreted a pending question from '{origin}'. Question: {metadata['pergunta_pendente']}")
                         # Currently handled by message_processor in run.py
                    else:
                        logger.info(f"[{self.fragment_id}] No commands or questions interpreted from text received from '{origin}'. Metadata: {metadata}")
                        
                    # Potentially post results back? Or just rely on logging for now.
                    
                except Exception as e:
                    logger.error(f"[{self.fragment_id}] Error interpreting text from message (origin: {origin}): {e}", exc_info=True)
            else:
                logger.warning(f"[{self.fragment_id}] Received 'a3l_directive' but content format is unexpected or missing 'interpret_text' type/text. Content: {content}")
        else:
            logger.warning(f"[{self.fragment_id}] Received unsupported message type '{message_type}'. Ignoring.")
    # --- End of handle_message ---

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        return cleaned

    def _extract_a3l_commands(self, text: str) -> List[str]:
        commands = []
        # Split into potential commands (lines or sentences)
        # Simple split by newline first
        potential_lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        logger.debug(f"_extract_a3l_commands: Processing {len(potential_lines)} potential lines.")
        processed_directives = set() # Track directives already added

        for line in potential_lines:
             matched = False
             # Try direct A3L command patterns first
             for directive_type, pattern in self.A3L_COMMAND_REGEX.items():
                  match = pattern.match(line)
                  if match:
                       # Simple duplicate check based on the matched line
                       if line not in processed_directives:
                           commands.append(line) 
                           processed_directives.add(line)
                           logger.debug(f"  Matched A3L pattern '{directive_type}': {line}")
                           matched = True
                           break # Stop checking patterns for this line
                       else:
                            logger.debug(f"  Skipping duplicate A3L command: {line}")
                            matched = True # Mark as matched to avoid further processing
                            break 
                            
             # Optional: If no direct match, could try sentence splitting or other heuristics
             if not matched:
                 logger.debug(f"  No direct A3L pattern matched for line: {line}")

        # Preserve order (already done by appending)
        logger.debug(f"_extract_a3l_commands: Extracted {len(commands)} commands: {commands}")
        return commands

    async def interpret_knowledge(self, text: str, context_fragment_id: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Analisa o texto bruto, extrai sugestões e as converte para comandos A3L.
        Prioriza:
        1. Extração direta (Regex)
        2. Fragmentos verificados relevantes (com 'knowledge_acquired')
        3. Preditor padrão ('language_analyzer')
        4. Professor LLM
        
        Args:
            text: O texto a ser interpretado.
            context_fragment_id: Opcional. ID do fragmento que é o contexto principal
                                 da conversa/reflexão, usado como fallback para 
                                 pronomes como "ele" ou "treiná-lo".

        Returns:
            Uma tupla: (lista de comandos A3L extraídos, metadados da interpretação).
                       Metadados podem incluir 'prediction': {'output': str, 'confidence': float}
        """
        logger.info(f"KI '{self.fragment_id}' interpreting text (length {len(text)}). Context: {context_fragment_id}")
        metadata = {"source": "KI_Unknown", "prediction": None, "error": None}
        extracted_commands = []
        
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            metadata["error"] = "Input text was empty after cleaning."
            return [], metadata

        # --- Step 1: Direct A3L Extraction via Regex --- 
        try:
            logger.debug("KI: Attempting direct A3L regex extraction...")
            regex_commands = self._extract_a3l_commands(cleaned_text)
            if regex_commands:
                 logger.info(f"KI: Extracted {len(regex_commands)} commands via regex.")
                 extracted_commands.extend(regex_commands)
                 metadata["source"] = "KI_Regex"
                 # If we got commands via regex, maybe we don't need the LLM?
                 # return extracted_commands, metadata # Option: return early if regex succeeds
            else:
                 logger.debug("KI: No direct A3L commands found via regex.")
        except Exception as regex_err:
             logger.error(f"KI: Error during regex command extraction: {regex_err}", exc_info=True)
             metadata["error"] = f"Regex extraction failed: {regex_err}"
             # Continue to LLM attempt even if regex fails?

        # --- Step 2: Professor/LLM Interpretation (if no regex commands or forced) --- 
        # Condition to use LLM: No regex commands found AND Professor is available/active
        # OR if a specific flag forces LLM use (not implemented here)
        use_llm = (not extracted_commands) and self.professor and self.professor.is_active
        
        if use_llm:
            logger.info(f"KI: Regex found no commands. Consulting Professor '{self.professor.fragment_id}'...")
            metadata["source"] = "KI_ProfessorLLM"
            try:
                # Prepare prompt using template
                prompt = self.prompt_template.format(
                    natural_language_input=cleaned_text,
                    context_info=f"Context Fragment ID: {context_fragment_id if context_fragment_id else 'None'}"
                )
                
                # Call Professor with grammar (if available)
                llm_response_text = await self.professor.ask_llm(
                    prompt,
                    grammar=self.grammar # Pass loaded grammar
                    # TODO: Add other parameters like temperature if needed
                )
                
                if llm_response_text:
                    logger.info(f"KI: Received response from Professor. Interpreting...")
                    # Attempt to extract commands from the LLM response itself
                    llm_extracted_commands = self._extract_a3l_commands(llm_response_text)
                    if llm_extracted_commands:
                         logger.info(f"KI: Extracted {len(llm_extracted_commands)} A3L commands from Professor's response.")
                         extracted_commands.extend(llm_extracted_commands)
                    else:
                         logger.warning("KI: Professor responded, but no A3L commands extracted from the response text.")
                         # Optional: Treat the raw response as natural language output or a plan?
                         # metadata["natural_language_output"] = llm_response_text
                else:
                    logger.warning("KI: Professor returned an empty response.")
                    metadata["error"] = "Professor returned empty response."

            except Exception as llm_err:
                 logger.error(f"KI: Error during LLM consultation/interpretation: {llm_err}", exc_info=True)
                 metadata["error"] = f"LLM consultation failed: {llm_err}"
        elif not extracted_commands:
             # Log why LLM wasn't used if no commands were found
             if not self.professor:
                 logger.warning("KI: No commands found and Professor not available.")
             elif not self.professor.is_active:
                  logger.warning("KI: No commands found and Professor is not active.")

        # --- Step 3: Deduplicate and Finalize --- 
        # Basic deduplication (more robust methods could be used)
        final_commands = []
        seen_commands = set()
        for cmd in extracted_commands:
            cmd_strip = cmd.strip()
            if cmd_strip and cmd_strip not in seen_commands:
                final_commands.append(cmd_strip)
                seen_commands.add(cmd_strip)
                
        logger.info(f"KI '{self.fragment_id}' interpretation complete. Found {len(final_commands)} unique commands. Source: {metadata['source']}")
        return final_commands, metadata

    def generate_reflection_a3l(self) -> str:
        """Gera uma descrição A3L do fragmento interpretador."""
        return (f"fragmento '{self.fragment_id}' ({self.__class__.__name__}), "
                f"interpreta texto para extrair comandos A3L. "
                f"Descrito como: '{self.description}'.")

    # --- Métodos Padrão nn.Module (não utilizados ativamente) ---
    def forward(self, *args, **kwargs):
        # Este fragmento não usa o forward pass padrão para tensores.
        # O método principal é interpret_knowledge.
        raise NotImplementedError("KnowledgeInterpreterFragment não implementa forward(). Use interpret_knowledge() ou handle_message().")

# Exemplo de uso (se executado diretamente)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Para ver os logs de debug
    
    interpreter = KnowledgeInterpreterFragment(
        fragment_id="interpretador_mestre",
        description="Interpreta logs e sugestões para A3L"
    )
    
    # Simular injeção do professor (requer ProfessorLLMFragment)
    # try:
    #     from professor_llm_fragment import ProfessorLLMFragment, LLM_ENABLED
    #     if LLM_ENABLED:
    #         mock_professor = ProfessorLLMFragment("prof_mock", "Mock Professor", llm_url="http://localhost:8080")
    #         interpreter.set_professor(mock_professor)
    # except ImportError:
    #     print("\n[AVISO] ProfessorLLMFragment não encontrado para teste completo.")
    
    print(interpreter.generate_reflection_a3l())
    
    test_text_1 = """
    Análise do log: A confiança para frag_alpha foi baixa (0.4). 
    Sugestão: treinar fragmento 'frag_alpha' por 5 épocas. 
    Além disso, talvez criar fragmento 'frag_alpha_especialista_confianca' com base em 'frag_alpha'.
    E também refletir sobre o fragmento 'frag_beta'.
    """
    
    test_text_2 = """
    O modelo base respondeu mal. Tente perguntar ao fragmento 'frag_gamma' com [0.1, 0.9, 0.5].
    Se não funcionar, treinar fragmento 'frag_gamma' por -3 épocas. < Inválido
    """
    
    test_text_3 = "Nada de interessante aqui."

    print("\n--- Teste 1 ---")
    commands_1, metadata_1 = interpreter.interpret_knowledge(test_text_1)
    print(f"Comandos A3L extraídos (Fonte: {metadata_1.get('source')}):")
    for cmd in commands_1:
        print(f"- {cmd}")

    print("\n--- Teste 2 ---")
    commands_2, metadata_2 = interpreter.interpret_knowledge(test_text_2)
    print(f"Comandos A3L extraídos (Fonte: {metadata_2.get('source')}):")
    for cmd in commands_2:
        print(f"- {cmd}")
        
    print("\n--- Teste 3 ---")
    commands_3, metadata_3 = interpreter.interpret_knowledge(test_text_3)
    print(f"Comandos A3L extraídos (Fonte: {metadata_3.get('source')}):")
    for cmd in commands_3:
        print(f"- {cmd}")
        
    print("\n--- Teste 4 (Natural Language) ---")
    test_text_4 = "Analise o desempenho recente e me diga como melhorar a classificação de imagens de gatos."
    commands_4, metadata_4 = interpreter.interpret_knowledge(test_text_4)
    print(f"Comandos A3L extraídos (Fonte: {metadata_4.get('source')}):")
    if commands_4:
         for cmd in commands_4:
             print(f"- {cmd}")
    else:
         print("(Nenhum comando extraído - esperado se Professor não estiver conectado ou não responder com A3L)")

"""
This defines the KnowledgeInterpreterFragment class in its own file. 
It inherits from nn.Module for framework consistency but its core logic uses regex 
to find and convert text patterns into A3L command strings via the 
`interpret_knowledge` method. Includes basic examples and logging.
""" 