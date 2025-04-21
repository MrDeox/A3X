from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
# Import a3x_bridge directly for synchronous calls
from a3x.a3net.integration import a3x_bridge 
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

# Type alias for the message handler - REMOVED
# MessageHandler = Callable[[str, Any, str], Awaitable[None]]

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
                 description: str = "Autonomous Cognitive Self-Starter"):
        """
        Inicializa o Fragmento Autônomo.

        Args:
            fragment_id: ID único para este fragmento.
            ki_fragment: Instância do Knowledge Interpreter Fragment.
            professor_fragment: Instância do Professor LLM Fragment (pode ser None).
            context_store: Instância do Context Store.
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

    # Modified to call handle_directive synchronously and handle errors
    async def _handle_a3l_command(self, command_str: str):
        self._logger.info(f"Handling A3L command synchronously: {command_str}")
        directive_dict = None
        try:
            directive_dict = interpret_a3l_line(command_str)
            if directive_dict:
                 # Add origin info (though less critical now it's synchronous)
                 directive_dict["_origin"] = f"Autonomous Starter ({self.metadata.name})"

                 # --- Execute Synchronously ---
                 self._logger.debug(f"Executing directive via a3x_bridge: {directive_dict}")
                 result = await a3x_bridge.handle_directive(directive_dict, fragment_instances=None, context_store=self._context_store)
                 self._logger.debug(f"Result from handle_directive: {result}")
                 # -----------------------------

                 if result and result.get("status") == "error":
                     error_msg = result.get("message", "Unknown execution error")
                     self._logger.error(f"Execution failed for '{command_str}': {error_msg}")
                     # Keep the failed command on the ladder for now? Or pop it?
                     # Let's pop it first, then push correction. Prevents re-running the fail immediately.
                     if self.goal_ladder and self.goal_ladder[-1] == command_str:
                          self.goal_ladder.pop()
                          self._logger.debug("Popped failed command from ladder before generating correction.")
                     # Generate and push a correction goal
                     self._generate_and_push_correction_goal(directive_dict, error_msg)
                 elif result and result.get("status") == "success":
                     self._logger.info(f"Successfully executed command: {command_str}")
                     # Pop AFTER successful execution
                     if self.goal_ladder and self.goal_ladder[-1] == command_str:
                         self.goal_ladder.pop()
                         self._logger.debug("Popped successful command from ladder.")
                 else:
                     # Command executed but no clear success/error status? Treat as success for now.
                     self._logger.warning(f"Command '{command_str}' executed with unclear status: {result}. Assuming success for ladder.")
                     if self.goal_ladder and self.goal_ladder[-1] == command_str:
                         self.goal_ladder.pop()
                         self._logger.debug("Popped command with unclear status from ladder.")

            else:
                self._logger.warning(f"Failed to parse A3L command: {command_str}. Skipping and popping.")
                # Pop invalid command from ladder
                if self.goal_ladder and self.goal_ladder[-1] == command_str:
                    self.goal_ladder.pop()

        except Exception as e:
            self._logger.error(f"Critical error handling A3L command '{command_str}': {e}", exc_info=True)
            # Pop command even if error occurred during handling to avoid infinite loops
            if self.goal_ladder and self.goal_ladder[-1] == command_str:
                self.goal_ladder.pop()
            # Optionally push a generic reflection on the handling error itself
            # self._generate_and_push_correction_goal(directive_dict or {}, f"Error in _handle_a3l_command: {e}")

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
         self._logger.debug(f"Attempting Professor consultation for goal: '{goal}'")
         if not self._professor or not self._professor.is_active:
             self._logger.warning(f"Professor not available or inactive. Cannot consult for goal: {goal}")
             return None
         try:
             # Formulate question for the Professor
             prompt = f"Eu sou um agente autônomo tentando alcançar o objetivo: '{goal}'. Não sei como proceder. Como devo decompor este objetivo em passos mais simples ou quais comandos A3L devo executar primeiro?"
             self._logger.info(f"Asking Professor ({self._professor.fragment_id}) for help with: '{goal}'")
             response = await self._professor.ask_llm(prompt)
             if response and not response.startswith("<"): # Basic check for error messages
                 self._logger.info(f"Professor responded: {response[:200]}...")
                 return response
             else:
                 self._logger.warning(f"Professor did not provide a usable response: {response}")
                 return None
         except Exception as e:
             self._logger.error(f"Error consulting Professor for '{goal}': {e}", exc_info=True)
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
            professor_response = await self._consult_professor(goal)
            if professor_response:
                # 3. Interpret Professor's response using KI
                decomposition_steps = await self._interpret_professor_response(professor_response)

        # 4. If decomposition steps were found (from KI or Professor->KI)
        if decomposition_steps:
            self._logger.info(f"Adding {len(decomposition_steps)} new steps to ladder for goal '{goal}'.")
            # Add steps in reverse order so the first step is at the top of the stack
            for step in reversed(decomposition_steps):
                self.goal_ladder.append(step.strip()) # Add stripped steps
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

                # Explicitly check for default goal OR use the heuristic
                if current_goal == self.DEFAULT_GOAL or not self._is_a3l_command(current_goal):
                    await self._handle_abstract_goal(current_goal)
                else:
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