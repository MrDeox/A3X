from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
# <<< REMOVED direct import of a3x_bridge >>>
# from a3x.a3net.integration import a3x_bridge 
import asyncio
# Removed Any from here as it's used later
from typing import Optional, Dict, List, Callable, Awaitable, Any 
import logging
import json
import re # For error parsing

# Dependências injetadas
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment
from a3x.a3net.core.context_store import ContextStore

# Type alias for the message handler
MessageHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]

class AutonomousSelfStarterFragment(BaseFragment):
    """
    Um Fragment especializado em iniciar ciclos autônomos de aprendizado e execução,
    capaz de decompor objetivos abstratos usando uma abordagem de engenharia reversa cognitiva.
    """
    # Default high-level goal if none is loaded
    DEFAULT_GOAL = "Aprender a entender linguagem natural autonomamente."
    # ContextStore key for persisting the goal ladder
    LADDER_CONTEXT_KEY = "autonomous_starter_goal_ladder"

    def __init__(self, fragment_id: str,
                 ki_fragment: KnowledgeInterpreterFragment,
                 professor_fragment: Optional[ProfessorLLMFragment],
                 context_store: ContextStore,
                 post_message_handler: MessageHandler,
                 description: str = "Autonomous Cognitive Self-Starter"):
        """
        Inicializa o Fragmento Autônomo.

        Args:
            fragment_id: ID único para este fragmento.
            ki_fragment: Instância do Knowledge Interpreter Fragment.
            professor_fragment: Instância do Professor LLM Fragment (pode ser None).
            context_store: Instância do Context Store.
            post_message_handler: Instância do Post Message Handler.
            description: Descrição do fragmento.
        """
        fragment_def = FragmentDef(
            name=fragment_id,
            fragment_class=AutonomousSelfStarterFragment,
            description=description,
            category="Autonomous",
            skills=[
                "decompose_abstract_goal",
                "consult_professor",
                "interpret_response",
                "initiate_learning_cycle",
                "handle_execution_error"
            ]
        )
        super().__init__(fragment_def=fragment_def)

        # --- Store Dependencies ---
        self._ki = ki_fragment
        self._professor = professor_fragment
        self._context_store = context_store
        self._post_message_handler = post_message_handler
        # ------------------------->

        # --- Initialize State ---
        self.goal_ladder: List[str] = [] # The ladder of goals/commands
        # ------------------------>

        self._logger.info(f"Fragment {self.metadata.name} initializing...")
        if not self._ki:
             self._logger.warning(f"KI Fragment dependency not provided to {self.metadata.name}.")
        if not self._professor:
             self._logger.warning(f"Professor Fragment dependency not provided or disabled for {self.metadata.name}.")
        if not self._context_store:
             self._logger.warning(f"Context Store dependency not provided to {self.metadata.name}.")
        if not self._post_message_handler:
             self._logger.warning(f"Post Message Handler not provided to {self.metadata.name}.")

        # Load persistent state at the end of initialization
        # Running this async method synchronously here is tricky.
        # It's better to call load_state at the beginning of execute().
        # asyncio.run(self._load_state()) # Avoid doing this here

        self._logger.info(f"Fragment {self.metadata.name} initialized.")

    async def _save_state(self):
        """Saves the current goal ladder to the ContextStore."""
        if not self._context_store:
            self._logger.warning("ContextStore not available, cannot save state.")
            return
        try:
            # Serialize the ladder to JSON string
            ladder_json = json.dumps(self.goal_ladder)
            await self._context_store.set(self.LADDER_CONTEXT_KEY, ladder_json)
            self._logger.debug(f"Saved goal ladder state: {ladder_json}")
        except Exception as e:
            self._logger.error(f"Failed to save goal ladder state: {e}", exc_info=True)

    async def _load_state(self):
        """Loads the goal ladder from the ContextStore."""
        if not self._context_store:
            self._logger.warning("ContextStore not available, cannot load state.")
            self.goal_ladder = [] # Start fresh if no store
            return
        try:
            ladder_json = await self._context_store.get(self.LADDER_CONTEXT_KEY)
            if ladder_json:
                self.goal_ladder = json.loads(ladder_json)
                self._logger.info(f"Loaded goal ladder state ({len(self.goal_ladder)} items). Top: {self.goal_ladder[-1] if self.goal_ladder else 'Empty'}")
            else:
                self._logger.info("No previous goal ladder state found in ContextStore.")
                self.goal_ladder = []
        except Exception as e:
            self._logger.error(f"Failed to load goal ladder state: {e}", exc_info=True)
            self.goal_ladder = [] # Reset ladder on error

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a string describing the fragment's main purpose."""
        return ("Initiate and manage autonomous learning/execution cycles, "
                "decomposing abstract goals via cognitive reverse engineering.")

    # --- Helper Methods ---
    def _is_a3l_command(self, goal: str) -> bool:
        # Simple heuristic: check for common A3L verbs at the start
        a3l_verbs = ["criar", "treinar", "refletir", "interpretar", "aprender", "executar", "definir", "comparar", "ler", "escrever", "perguntar"]
        return any(goal.strip().lower().startswith(verb) for verb in a3l_verbs)

    def _generate_and_push_correction_goal(self, failed_directive: Dict, error_message: str):
        """Analyzes an error message and pushes a corrective goal onto the ladder."""
        self._logger.info(f"Generating correction goal for error: {error_message}")
        correction_goal = None

        # --- Analyze specific errors ---
        # Example: TypeError: ...__init__() got an unexpected keyword argument '...'
        match_type_error = re.search(
            r"TypeError: .*__init__\(\) got an unexpected keyword argument '(\w+)'",
            error_message
        )
        if match_type_error:
            bad_param = match_type_error.group(1)
            fragment_type = failed_directive.get('fragment_type', 'unknown_type')
            fragment_id = failed_directive.get('fragment_id', 'unknown_id')
            # Ask professor about valid parameters for this type
            correction_goal = f"perguntar ao professor \"Quais são os parâmetros válidos e obrigatórios para criar um fragmento do tipo '{fragment_type}' na A3L? O parâmetro '{bad_param}' falhou.\""
            self._logger.warning(f"Detected TypeError for param '{bad_param}' in type '{fragment_type}'. Pushing Professor query.")

        # Example: Fragment not found error (e.g., for 'treinar fragmento')
        match_not_found = re.search(
             r"Fragment '(\w+)' not found", # Basic pattern
             error_message
        )
        if match_not_found and not correction_goal: # Only if TypeError didn't match
             missing_id = match_not_found.group(1)
             original_action = failed_directive.get('type', 'unknown_action')
             # Reflect on why the fragment is missing before this action
             correction_goal = f"refletir sobre erro 'Fragmento \\'{missing_id}\\' não encontrado ao tentar executar {original_action}'"
             self._logger.warning(f"Detected missing fragment '{missing_id}'. Pushing reflection goal.")

        # --- Default fallback: Generic reflection ---
        if not correction_goal:
            failed_cmd_str = failed_directive.get('original_line', str(failed_directive)) # Get original line if possible
            correction_goal = f"refletir sobre erro ao executar comando '{failed_cmd_str[:100]}...': {error_message[:100]}"
            self._logger.warning("Unknown error type or pattern not matched. Pushing generic reflection goal.")

        # --- Push the correction goal (with loop prevention) ---
        if correction_goal:
            # Basic loop prevention: don't push if the same goal is already at the top
            if not self.goal_ladder or self.goal_ladder[-1] != correction_goal:
                self.goal_ladder.append(correction_goal)
                self._logger.info(f"Pushed correction goal to ladder: {correction_goal}")
            else:
                self._logger.warning(f"Correction goal '{correction_goal}' is identical to the top of the ladder. Skipping push to prevent loop.")
        else:
             self._logger.error("Could not generate a correction goal.")

    # --- NEW METHOD ---
    def _simplify_error_message(self, error_message: str) -> str:
        """Simplifies known complex error messages for the Professor."""
        self._logger.debug(f"Simplifying error: {error_message[:150]}...")
        if "Fragment" in error_message and "not found" in error_message:
            # Try to extract fragment name if possible
            match = re.search(r"Fragment \'([^\']+)\'", error_message)
            fragment_name = match.group(1) if match else "unknown"
            simplified = f"Fragmento \'{fragment_name}\' não encontrado"
            self._logger.info(f"Simplified to: {simplified}")
            return simplified
        if "Unknown execution error" in error_message: # Handle generic bridge error
            simplified = "Erro ao executar comando: fragmento ausente ou mal definido"
            self._logger.info(f"Simplified to: {simplified}")
            return simplified
        if "is not ProfessorLLMFragment" in error_message:
            simplified = "Tentativa de aprender com um fragmento que não é um Professor"
            self._logger.info(f"Simplified to: {simplified}")
            return simplified
        # Add more rules here based on common errors
        
        # Default Fallback if no specific rule matches
        # Consider if a generic message is better than the full original error for the LLM
        generic_fallback = "Erro genérico durante execução simbólica"
        self._logger.info(f"Error not specifically simplified, using generic fallback: {generic_fallback}")
        # return error_message # Option 1: Return original if not simplified
        return generic_fallback # Option 2: Return generic fallback
    # --- END NEW METHOD ---

    # Modified to post command to queue instead of direct synchronous call
    async def _handle_a3l_command(self, command_str: str):
        self._logger.info(f"Handling A3L command asynchronously: {command_str}")
        directive_dict = None
        try:
            directive_dict = interpret_a3l_line(command_str)
            if directive_dict:
                 directive_dict["_origin"] = f"Autonomous Starter ({self.metadata.name})"

                 # --- Check for special handling (e.g., skip learn_from_professor) ---
                 if directive_dict.get("type") == "learn_from_professor":
                     self._logger.info(f"Intercepted 'learn_from_professor'. Skipping execution, popping from ladder.")
                     if self.goal_ladder and self.goal_ladder[-1] == command_str:
                         self.goal_ladder.pop()
                     # Return early, do not post to queue
                     return 
                 else:
                     # --- Post other directives to the message queue for Executor ---
                     if self._post_message_handler:
                         self._logger.debug(f"Posting directive to queue for Executor: {directive_dict}")
                         await self._post_message_handler(
                             message_type="a3l_command",
                             content=directive_dict,
                             target_fragment="Executor"
                         )
                         # Command is now sent, pop it from ladder. 
                         # Error handling/correction will happen based on Executor feedback or monitoring.
                         if self.goal_ladder and self.goal_ladder[-1] == command_str:
                             self.goal_ladder.pop()
                             self._logger.debug("Popped command from ladder after posting to queue.")
                     else:
                         self._logger.error("Cannot execute command: post_message_handler not set.")
                         # Optionally pop or leave on ladder if handler missing?
                         # Leaving it for now, might cause loop if handler never appears.

            else:
                self._logger.warning(f"Failed to parse A3L command: {command_str}. Skipping and popping.")
                # Pop unparseable command from ladder
                if self.goal_ladder and self.goal_ladder[-1] == command_str:
                    self.goal_ladder.pop()
                    self._logger.debug("Popped unparseable command from ladder.")

        except Exception as e:
            self._logger.error(f"Error interpreting or posting A3L command '{command_str}': {e}", exc_info=True)
            # Pop the command that caused an error during this handling phase
            if self.goal_ladder and self.goal_ladder[-1] == command_str:
                 self.goal_ladder.pop()
                 self._logger.debug("Popped command from ladder due to interpretation/posting error.")
            # We might still want to generate a correction goal here?
            # self._generate_and_push_correction_goal(directive_dict or {}, f"Error handling command: {e}")

    async def _attempt_ki_decomposition(self, goal: str) -> Optional[List[str]]:
        self._logger.debug(f"Attempting KI decomposition for goal: '{goal}'")
        if not self._ki:
            self._logger.warning("KI not available for decomposition.")
            return None
        try:
            # Prompt KI to break down the goal
            # V1 Prompt - Simple breakdown
            prompt = f"Para realizar o objetivo abstrato '{goal}', quais são os passos imediatos necessários? Responda com comandos A3L ou sub-objetivos mais simples, um por linha."
            # V2 Prompt - More focused on prerequisites
            # prompt = f"Considere o objetivo: '{goal}'. Qual é o conhecimento ou capacidade pré-requisito mais importante para alcançá-lo? Ou qual o primeiro comando A3L a ser executado?"

            self._logger.info(f"Asking KI ({self._ki.fragment_id}) to decompose: '{goal}'")
            extracted_commands, metadata = await self._ki.interpret_knowledge(prompt, context_fragment_id=self.metadata.name)
            source_info = metadata.get("source", "KI_Decomposition")

            if extracted_commands:
                self._logger.info(f"KI ({source_info}) decomposed '{goal}' into: {extracted_commands}")
                return extracted_commands
            else:
                self._logger.info(f"KI ({source_info}) did not provide decomposition steps for '{goal}'.")
                return None
        except Exception as e:
            self._logger.error(f"Error during KI decomposition for '{goal}': {e}", exc_info=True)
            return None

    async def _consult_professor(self, goal: str) -> Optional[str]:
        """Consulta o Professor para obter orientação sobre um objetivo/erro."""
        if not self._professor:
            self._logger.warning("Professor fragment not available for consultation.")
            return None
        if not hasattr(self._professor, 'ask_for_guidance'):
             self._logger.error("Professor fragment lacks 'ask_for_guidance' method.")
             return None
        if not self._professor.is_active: # Check if Professor itself is active
             self._logger.warning("Professor fragment is not active, skipping consultation.")
             return None

        self._logger.info(f"Consulting Professor about goal/error: '{goal[:100]}...'")
        try:
            # Simplify the message before sending it to the professor
            simplified_goal = self._simplify_error_message(goal)
            self._logger.debug(f"Sending simplified goal/error to professor: '{simplified_goal}'")
            # Call professor with the simplified message
            response = await self._professor.ask_for_guidance(simplified_goal)

            if response:
                self._logger.info(f"Professor provided guidance (length: {len(response)}). First 100 chars: {response[:100]}")
                return response
            else:
                self._logger.warning("Professor returned empty guidance.")
                return None
        except Exception as e:
            self._logger.error(f"Error during professor consultation: {e}", exc_info=True)
            return None

    async def _interpret_professor_response(self, response_text: str) -> Optional[List[str]]:
        self._logger.debug("Attempting KI interpretation of Professor's response.")
        if not self._ki:
            self._logger.warning("KI not available for interpretation.")
            return None
        try:
            self._logger.info("Asking KI to interpret Professor's response...")
            # Use interpret_knowledge directly on the Professor's text
            extracted_commands, metadata = await self._ki.interpret_knowledge(response_text, context_fragment_id=self.metadata.name)
            source_info = metadata.get("source", "KI_ProfInterpret")

            if extracted_commands:
                self._logger.info(f"KI ({source_info}) interpreted Professor response into: {extracted_commands}")
                return extracted_commands
            else:
                self._logger.info(f"KI ({source_info}) did not extract commands from Professor response.")
                return None
        except Exception as e:
            self._logger.error(f"Error during KI interpretation of Professor response: {e}", exc_info=True)
            return None

    async def _handle_abstract_goal(self, goal: str):
        """Handles decomposition and potential Professor consultation for an abstract goal."""
        self._logger.info(f"Handling abstract goal: {goal}")

        # Pop the abstract goal first, we will push sub-steps if found
        if self.goal_ladder and self.goal_ladder[-1] == goal:
            self.goal_ladder.pop()
            self._logger.debug(f"Popped abstract goal '{goal}' to decompose.")

        # 1. Attempt decomposition using KI
        decomposition_steps = await self._attempt_ki_decomposition(goal)

        # 2. If KI fails, consult Professor
        if not decomposition_steps:
            self._logger.info(f"KI failed to decompose '{goal}', consulting Professor...")
            # --- Simplify error message before sending to professor ---
            professor_query = goal
            if "erro" in goal.lower(): # Basic check if goal is about an error
                professor_query = self._simplify_error_message(goal)
                self._logger.info(f"Using simplified query for Professor: {professor_query[:100]}...")
            # ------------------------------------------------------->
            professor_response = await self._consult_professor(professor_query)
            if professor_response:
                # 3. Interpret Professor's response using KI
                decomposition_steps = await self._interpret_professor_response(professor_response)

        # 4. If decomposition steps were found (from KI or Professor->KI)
        if decomposition_steps:
            self._logger.info(f"Adding {len(decomposition_steps)} new steps to ladder for goal '{goal}'. Steps: {decomposition_steps}") # Log the actual steps
            # Add steps in reverse order so the first step is at the top of the stack
            for step in reversed(decomposition_steps):
                # Ensure step is a string and strip whitespace
                if isinstance(step, str):
                     self.goal_ladder.append(step.strip())
                else:
                     # Log if a step isn't a string (shouldn't happen with current KI)
                     self._logger.warning(f"Encountered non-string step in decomposition result: {step}. Skipping.")
                     
            self._logger.debug(f"Ladder top is now: {self.goal_ladder[-1] if self.goal_ladder else 'Empty'}")
        else:
            self._logger.warning(f"Failed to decompose abstract goal '{goal}' using KI and Professor. Goal discarded.")
            # Optionally, push a reflection goal?
            # self.goal_ladder.append(f"refletir sobre falha ao decompor '{goal}'")

    # --- Main Execution Loop ---
    async def execute(self):
        """
        Ponto de entrada para o ciclo autônomo contínuo.
        """
        self._logger.info(f"🚀 Starting continuous autonomous cycle for {self.metadata.name}...")

        # Ensure dependencies are met before starting loop
        if not self._ki or not self._context_store:
            self._logger.error("Missing critical dependencies (KI or ContextStore). Cannot start autonomous cycle.")
            return

        # Load initial state
        await self._load_state()

        while True:
            try:
                # Ensure state is loaded at start of each cycle iteration
                # This might be redundant if state isn't modified externally,
                # but safer if other processes could interact with the context store.
                # await self._load_state() # Consider if needed every loop

                if not self.goal_ladder:
                    self._logger.info("Goal ladder is empty. Initializing with default goal.")
                    self.goal_ladder.append(self.DEFAULT_GOAL)
                    await self._save_state() # Save the initialized ladder

                # Get current goal from the top of the stack
                current_goal = self.goal_ladder[-1] # Peek at the top
                self._logger.info(f"Processing top goal: '{current_goal}'")

                # --- Refined Goal Handling ---
                # Check if it's an error reflection goal generated internally
                is_error_reflection = current_goal.strip().lower().startswith("refletir sobre erro")
                
                # Check if it's a standard A3L command (heuristic)
                is_command = self._is_a3l_command(current_goal)
                
                # Determine if it should be treated as abstract
                # Treat as abstract if: it's the default goal, OR it's an error reflection, OR it's not recognized as a command.
                is_abstract = (current_goal == self.DEFAULT_GOAL or 
                               is_error_reflection or 
                               not is_command)
                # ------------------------------
                
                if is_abstract:
                    self._logger.debug(f"Goal '{current_goal[:50]}...' identified as abstract/reflective. Handling via _handle_abstract_goal.")
                    await self._handle_abstract_goal(current_goal)
                else:
                    # Assume it's an executable command
                    self._logger.debug(f"Goal '{current_goal[:50]}...' identified as executable command. Handling via _handle_a3l_command.")
                    await self._handle_a3l_command(current_goal)

                # Save state after potential modification
                await self._save_state()

                # Wait before next cycle
                await asyncio.sleep(10) # Configurable delay?

            except asyncio.CancelledError:
                self._logger.info(f"Autonomous cycle for {self.metadata.name} cancelled.")
                await self._save_state() # Attempt to save state on cancellation
                break
            except Exception as e:
                self._logger.error(f"Error in autonomous cycle for {self.metadata.name}: {e}", exc_info=True)
                # Avoid tight loop on persistent errors
                await self._save_state() # Save state before sleeping
                await asyncio.sleep(60) # Longer sleep on error

# Remover a necessidade de importar ki_main_instance globalmente
# O método _handle não é mais necessário 