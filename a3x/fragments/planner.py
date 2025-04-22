import asyncio
import logging
import random
import re
import uuid
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from pathlib import Path # Needed for plan generation target paths

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

try:
    from a3x.core.context.context_store import ContextStore
except ImportError:
    ContextStore = None

logger = logging.getLogger(__name__)

# Configuration
PLANNING_TRIGGER_THRESHOLD = 1 # Plan after seeing 1 reflection + 1 summary cycle

# Regex patterns to extract info (adjust if message formats change)
# Corrected Regex Patterns (using raw strings r'...')
SUCCESSFUL_ACTION_PATTERN = re.compile(r"-\s*'(.*?)':\s*(\d+)\s*successes")
FRAGMENT_ERROR_PATTERN = re.compile(r"-\s*'(.*?)':\s*(\d+)\s*errors/failures")
HEURISTIC_PATTERN = re.compile(r"- Count=(\d+): Sender='(.*?)', Action='(.*?)', Target='(.*?)'")

# Updated path for generated strategic targets (referenced for context)
GOAL_TARGET_PREFIX = "data/runtime/generated/strategic"
# Updated path for generated planned targets
GENERATED_PLAN_TARGET_PREFIX = "data/runtime/generated/planned"

# Define a list of alternative topics for the research skill
ALTERNATIVE_TOPICS = [
    "python basics",
    "data structures",
    "web scraping fundamentals",
    "API integration patterns",
    "async programming in python"
]

class PlannerFragment(BaseFragment):
    """Listens to reflections and learning summaries to plan sequences of actions."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self.context_store: Optional[ContextStore] = None
        self._reflection_processed = False
        self._summary_processed = False
        # Store extracted insights
        self.recent_successful_actions: Dict[str, int] = {}
        self.recent_erroring_fragments: Dict[str, int] = {}
        self.recent_top_heuristics: List[Tuple[int, str, str, str]] = [] # Changed to tuple
        self._logger.info(f"[{self.get_name()}] Initialized. Will plan after processing reflection and summary.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Analyzes reflections and learning summaries to generate 'plan_sequence' directives, proposing coordinated actions based on system insights."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the Planner."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        # --- Injetar ContextStore --- 
        if hasattr(shared_context, 'store') and isinstance(shared_context.store, ContextStore):
            self.context_store = shared_context.store
            self._logger.info(f"[{self.get_name()}] ContextStore received.")
        else:
            self._logger.warning(f"[{self.get_name()}] ContextStore not found in provided context.")
        # --- Fim Injeção ---
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming reflections and learning summaries."""
        if self._fragment_context is None:
            self.set_context(context)

        msg_type = message.get("type")
        content = message.get("content")

        if msg_type == "REFLECTION" and isinstance(content, dict):
            self._logger.info(f"[{self.get_name()}] Received reflection.")
            reflection_summary = content.get("reflection_summary")
            if reflection_summary:
                self._parse_reflection(reflection_summary)
                self._reflection_processed = True

        elif msg_type == "LEARNING_SUMMARY" and isinstance(content, dict):
            self._logger.info(f"[{self.get_name()}] Received learning summary.")
            summary_text = content.get("summary_text")
            if summary_text:
                 self._parse_summary(summary_text)
                 self._summary_processed = True

        # --- Handle REASSESS Directive ---
        elif msg_type == "ARCHITECTURE_SUGGESTION" and isinstance(content, dict) and content.get("type") == "directive" and content.get("action") == "REASSESS":
            sender = message.get("sender", "Coordinator") # Assume Coordinator if missing
            self._logger.warning(f"[{self.get_name()}] Received REASSESS directive from {sender}.")
            
            if not self._fragment_context or not self._fragment_context.store:
                self._logger.error(f"[{self.get_name()}] Cannot handle REASSESS: Context or ContextStore not available.")
                return

            # --- Adaptive Planning Logic ---
            failed_topics_key = "failed_research_topics"
            failed_topics = await self._fragment_context.store.get(failed_topics_key)
            if failed_topics is None:
                failed_topics = []
            elif not isinstance(failed_topics, list):
                self._logger.warning(f"[{self.get_name()}] Invalid data type for '{failed_topics_key}' in ContextStore. Resetting to empty list.")
                failed_topics = []
                
            self._logger.info(f"[{self.get_name()}] Previously failed topics: {failed_topics}")

            # Select the next available topic
            chosen_topic = None
            for topic in ALTERNATIVE_TOPICS:
                if topic not in failed_topics:
                    chosen_topic = topic
                    break
            
            if not chosen_topic:
                # All alternatives failed, maybe reset or use a default/random?
                self._logger.warning(f"[{self.get_name()}] All alternative topics have failed. Resetting failed list and using the first topic.")
                failed_topics = [] 
                chosen_topic = ALTERNATIVE_TOPICS[0]

            self._logger.info(f"[{self.get_name()}] Selected new topic for research: '{chosen_topic}'")

            # Update the failed topics list *before* sending the plan
            failed_topics.append(chosen_topic)
            await self._fragment_context.store.set(failed_topics_key, failed_topics)
            self._logger.info(f"[{self.get_name()}] Updated failed topics in ContextStore: {failed_topics}")
            
            # Generate a unique plan ID for this attempt
            new_plan_id = f"learn_topic_{uuid.uuid4().hex[:8]}"

            # Generate the new plan sequence using the chosen topic
            new_plan_description = f"Adaptive plan ({new_plan_id}) generated after REASSESS trigger. Attempting research on: '{chosen_topic}'."
            new_plan_sequence = [
                {
                    "type": "directive",
                    "action": "execute_skill",
                    "skill": "research_topic",
                    "parameters": { "topic": chosen_topic }, # Use the newly chosen topic
                    "message": f"Execute research step: '{chosen_topic}'"
                },
                {
                    "type": "directive",
                    "action": "summarize_research", # Assuming this exists
                    "target": f"research_summary_{chosen_topic.replace(' ', '_')}.md",
                    "message": f"Summarize findings for topic: '{chosen_topic}'."
                }
            ]
            new_plan_content = {
                 "plan_id": new_plan_id, # Include the unique ID
                 "objective": f"Learn about {chosen_topic}", # Update objective
                 "actions": new_plan_sequence # Use the generated sequence
            }
            
            # --- Log e Salvar Sugestão ---            
            suggestion_command = f"criar fragmento '{new_plan_id}' tipo 'GeneratedPlan' objective='{new_plan_content['objective']}' actions={new_plan_content['actions']}" # Exemplo A3L para plano
            suggestion_reason = f"Plano Adaptativo (ID: {new_plan_id}) gerado após REASSESS. Objetivo: Aprender sobre '{chosen_topic}'."
            self._logger.info(f"[{self.get_name()}] Sugestão: {suggestion_reason} -> {suggestion_command}")
            append_to_log(f"# [Sugestão Planner] {suggestion_reason}. Comando: {suggestion_command}. Aguardando comando A3L.")
            if self.context_store:
                 await self.context_store.push("pending_suggestions", {"command": suggestion_command, "reason": suggestion_reason, "source": self.get_name()})
            # --- Fim Log/Salvar ---

        # <<< NEW: Handle Reflection Results >>>
        elif msg_type == "reflection_result" and isinstance(content, dict):
             await self._handle_reflection_result(content)
        # <<< END NEW >>>

        # Check if conditions are met to generate a plan (from periodic reflection/summary)
        elif self._reflection_processed and self._summary_processed:
             self._logger.info(f"[{self.get_name()}] Both periodic reflection and summary processed. Attempting to generate plan...")
             await self._generate_plan_sequence() # Existing planning logic based on periodic summaries
             # Reset flags after planning
             self._reflection_processed = False
             self._summary_processed = False
             # Clear stored data for next cycle
             self.recent_successful_actions.clear()
             self.recent_erroring_fragments.clear()
             self.recent_top_heuristics.clear()
        else:
            self._logger.info(f"[{self.get_name()}] No actionable plan generated from current insights.")
            # --- New: Trigger Learning on Planning Failure ---
            try:
                objective = "Gerar um plano de ação baseado nas últimas reflexões e sumários."
                failure_context = f"Não foi possível gerar um plano acionável. Últimas ações bem-sucedidas: {self.recent_successful_actions}. Fragmentos com erro: {self.recent_erroring_fragments}. Heurísticas: {self.recent_top_heuristics}."
                learning_request = f"aprender com 'Como posso gerar um plano A3L eficaz considerando o seguinte contexto de falha: {failure_context}'"
                
                await self.post_chat_message(
                    message_type="a3l_command", # Use a specific type if available, or a general command type
                    content={"command": learning_request}, # Structure as needed by the interpreter
                    target_fragment="KnowledgeInterpreter" # Target the fragment responsible for learning
                )
                self._logger.warning(f"[{self.get_name()}] Planning failed. Sent learning request to KnowledgeInterpreter: {learning_request}")
            except Exception as learn_e:
                 self._logger.error(f"[{self.get_name()}] Failed to send learning request after planning failure: {learn_e}", exc_info=True)
            # --- End New ---

            # --- Novo Fallback: Refletir sobre Heurísticas ---            
            try:
                reflection_request = f"refletir sobre heurísticas de planejamento salvas"
                # Logar e salvar a sugestão de refletir
                self._logger.warning(f"[{self.get_name()}] Planning failed. Sugerindo reflexão sobre heurísticas.")
                append_to_log(f"# [Sugestão Planner] Falha no planejamento. Comando: {reflection_request}. Aguardando comando A3L.")
                if self.context_store:
                    await self.context_store.push("pending_suggestions", {"command": reflection_request, "reason": "Falha no planejamento heurístico", "source": self.get_name()})
            except Exception as reflect_e:
                 self._logger.error(f"[{self.get_name()}] Failed to send heuristic reflection request after planning failure: {reflect_e}", exc_info=True)
            # --- Fim Novo Fallback ---

    def _parse_reflection(self, reflection_text: str):
        """Extracts successful actions and erroring fragments from reflection text."""
        # Extract successful actions
        success_matches = SUCCESSFUL_ACTION_PATTERN.findall(reflection_text)
        for action, count_str in success_matches:
            try:
                self.recent_successful_actions[action] = int(count_str)
            except ValueError: pass
        self._logger.debug(f"[{self.get_name()}] Parsed successful actions: {self.recent_successful_actions}")
        
        # Extract erroring fragments
        error_matches = FRAGMENT_ERROR_PATTERN.findall(reflection_text)
        for fragment, count_str in error_matches:
             try:
                  self.recent_erroring_fragments[fragment] = int(count_str)
             except ValueError: pass
        self._logger.debug(f"[{self.get_name()}] Parsed erroring fragments: {self.recent_erroring_fragments}")

    def _parse_summary(self, summary_text: str):
        """Extracts top heuristics from learning summary text."""
        heuristic_matches = HEURISTIC_PATTERN.findall(summary_text)
        for count_str, sender, action, target_pattern in heuristic_matches:
             try:
                  count = int(count_str)
                  self.recent_top_heuristics.append((count, sender, action, target_pattern))
             except ValueError: pass
        # Keep sorted by count
        self.recent_top_heuristics.sort(key=lambda x: x[0], reverse=True)
        self._logger.debug(f"[{self.get_name()}] Parsed top heuristics: {self.recent_top_heuristics}")

    async def _generate_plan_sequence(self):
        """Generates a plan based on analyzed reflections and heuristics."""
        if not self._fragment_context:
             self._logger.error(f"[{self.get_name()}] Cannot generate plan: Context not set.")
             return

        plan_description = "No specific plan generated based on current insights."
        plan_sequence = [] # List of action descriptions or directives
        plan_source = "Default/NoTrigger"

        # --- Simple Planning Logic Examples --- 

        # 1. If a fragment errors often, try refactoring it based on a successful pattern
        if self.recent_erroring_fragments and self.recent_top_heuristics:
             # Find most error-prone fragment
             most_erroring = max(self.recent_erroring_fragments, key=self.recent_erroring_fragments.get)
             error_count = self.recent_erroring_fragments[most_erroring]
             
             # Find a successful heuristic (simple approach: top one)
             top_heuristic = self.recent_top_heuristics[0]
             h_count, h_sender, h_action, h_target_pattern = top_heuristic

             # Basic plan: try to apply the successful action to the erroring fragment (needs refinement)
             # This is very naive, assumes fragment name can be a target path component
             target_path_guess = f"a3x/fragments/{most_erroring.lower()}.py"
             plan_description = f"Attempt to refactor error-prone fragment '{most_erroring}' ({error_count} errors) using pattern '{h_action}' which succeeded {h_count} times."
             plan_sequence = [
                 {"action": "refactor_module", "target": target_path_guess, "message": f"Refactor {most_erroring} to address recent errors, possibly applying patterns similar to successful '{h_action}' actions."}
             ]
             plan_source = f"ErrorCorrection('{most_erroring}') + Heuristic('{h_action}')"

        # 2. If a creation action is very successful, try creating another one
        elif self.recent_top_heuristics:
            top_heuristic = self.recent_top_heuristics[0]
            h_count, h_sender, h_action, h_target_pattern = top_heuristic
            if h_action == "create_helper_module" and h_count > 5: # Example threshold
                 new_target = f"a3x/generated/planned/helper_{uuid.uuid4().hex[:6]}.py"
                 plan_description = f"Action '{h_action}' by '{h_sender}' is highly successful ({h_count} times). Planning creation of another helper module."
                 plan_sequence = [
                      {"action": "create_helper_module", "target": new_target, "message": f"Create a new helper module at {new_target}, inspired by recent successes."}
                 ]
                 plan_source = f"HeuristicReplication('{h_action}')"

        # TODO: Add more sophisticated planning logic based on combining insights.
        # Example: If 'create' succeeds but then 'refactor' often fails on the same target type -> Plan: create -> test -> maybe suggest different refactor approach.

        # --- Post the Plan --- 
        if plan_sequence:
            self._logger.info(f"[{self.get_name()}] Generated Plan: {plan_description}")
            plan_content = {
                "plan_id": f"heuristic_plan_{uuid.uuid4().hex[:6]}", # Add an ID
                "objective": plan_description, # Use description as objective
                "actions": plan_sequence, # The actual steps
                "source": plan_source
            }
            # --- Log e Salvar Sugestão ---            
            suggestion_command = f"criar fragmento '{plan_content['plan_id']}' tipo 'GeneratedPlan' objective='{plan_content['objective']}' actions={plan_content['actions']}" # Exemplo A3L para plano
            suggestion_reason = f"Plano Heurístico (ID: {plan_content['plan_id']}). Fonte: {plan_source}. Objetivo: {plan_content['objective']}."
            self._logger.info(f"[{self.get_name()}] Sugestão: {suggestion_reason} -> {suggestion_command}")
            append_to_log(f"# [Sugestão Planner] {suggestion_reason}. Comando: {suggestion_command}. Aguardando comando A3L.")
            if self.context_store:
                 await self.context_store.push("pending_suggestions", {"command": suggestion_command, "reason": suggestion_reason, "source": self.get_name()})
            # --- Fim Log/Salvar ---
            
        else:
            # self._logger.info(f"[{self.get_name()}] No actionable plan generated from current insights.") # Log now redundant if learn request is sent
            # --- Trigger Learning on Planning Failure (Already added in previous step) ---
            try:
                objective = "Gerar um plano de ação baseado nas últimas reflexões e sumários."
                failure_context = f"Não foi possível gerar um plano acionável. Últimas ações bem-sucedidas: {self.recent_successful_actions}. Fragmentos com erro: {self.recent_erroring_fragments}. Heurísticas: {self.recent_top_heuristics}."
                learning_request = f"aprender com 'Como posso gerar um plano A3L eficaz considerando o seguinte contexto de falha: {failure_context}'"
                
                await self.post_chat_message(
                    message_type="a3l_command", # Use a specific type if available, or a general command type
                    content={"command": learning_request}, # Structure as needed by the interpreter
                    target_fragment="KnowledgeInterpreter" # Target the fragment responsible for learning
                )
                self._logger.warning(f"[{self.get_name()}] Planning failed. Sent learning request to KnowledgeInterpreter: {learning_request}")
            except Exception as learn_e:
                 self._logger.error(f"[{self.get_name()}] Failed to send learning request after planning failure: {learn_e}", exc_info=True)
            # --- End Trigger Learning ---

            # --- Fallback: Refletir sobre Heurísticas ---            
            try:
                reflection_request = f"refletir sobre heurísticas de planejamento salvas"
                # Logar e salvar a sugestão de refletir
                self._logger.warning(f"[{self.get_name()}] Planning failed. Sugerindo reflexão sobre heurísticas.")
                append_to_log(f"# [Sugestão Planner] Falha no planejamento. Comando: {reflection_request}. Aguardando comando A3L.")
                if self.context_store:
                    await self.context_store.push("pending_suggestions", {"command": reflection_request, "reason": "Falha no planejamento heurístico", "source": self.get_name()})
            except Exception as reflect_e:
                 self._logger.error(f"[{self.get_name()}] Failed to send heuristic reflection request after planning failure: {reflect_e}", exc_info=True)
            # --- Fim Fallback Refletir ---
                 
            # --- Fallback: Aprender com --- (Não salva como sugestão executável direta)
            try:
                 learning_request = f"aprender com 'Como posso gerar um plano A3L eficaz considerando o seguinte contexto de falha: {failure_context}'" 
                 await self.post_chat_message(
                     message_type="a3l_command", 
                     content={"command": learning_request},
                     target_fragment="Executor" # Ou KnowledgeInterpreter se ele processar 'aprender com'
                 )
                 # ... log warning ...
            except Exception as learn_e:
                 self._logger.error(f"[{self.get_name()}] Failed to send learning request after planning failure: {learn_e}", exc_info=True)
            # --- Fim Fallback Aprender ---

    async def _handle_reflection_result(self, content: dict):
        """Generates a proactive plan based on a reflection about successful learning."""
        self._logger.info(f"[{self.get_name()}] Received reflection_result: {content}")
        source_type = content.get("source_type")
        source_content = content.get("source_content")
        reflection_text = content.get("reflection_text")

        # Only act on reflections derived from successful learning summaries for now
        if source_type != "learning_summary" or not isinstance(source_content, dict):
            self._logger.debug(f"[{self.get_name()}] Ignoring reflection_result not sourced from learning_summary.")
            return
        
        original_topic = source_content.get("topic")
        original_plan_id = source_content.get("plan_id")
        
        self._logger.info(f"[{self.get_name()}] Reflection confirms success for topic '{original_topic}' (Plan: {original_plan_id}). Planning next step.")

        # --- Simple Proactive Planning Logic --- 
        next_topic = None
        if original_topic == "python basics":
             next_topic = "intermediate python concepts"
        elif original_topic == "data structures":
             next_topic = "algorithm analysis"
        # Add more topic progression rules here
        else:
             self._logger.warning(f"[{self.get_name()}] No defined next step for successful topic: '{original_topic}'. No proactive plan generated.")
             return
        
        # Ensure we don't immediately retry the next topic if it also just failed (edge case)
        if self._fragment_context and self._fragment_context.store:
             failed_topics_key = "failed_research_topics"
             failed_topics = await self._fragment_context.store.get(failed_topics_key)
             if isinstance(failed_topics, list) and next_topic in failed_topics:
                 self._logger.warning(f"[{self.get_name()}] Proposed next topic '{next_topic}' is in the recently failed list. Skipping proactive plan.")
                 return
        
        self._logger.info(f"[{self.get_name()}] Proactively planning to research next topic: '{next_topic}'")

        # Generate plan similar to adaptive planning
        new_plan_id = f"learn_next_{uuid.uuid4().hex[:8]}"
        new_plan_description = f"Proactive plan ({new_plan_id}) generated after successful reflection on '{original_topic}'. Targeting '{next_topic}'."
        new_plan_sequence = [
            {
                "type": "directive",
                "action": "execute_skill",
                "skill": "research_topic",
                "parameters": { "topic": next_topic },
                "message": f"Execute proactive research step: '{next_topic}'"
            },
            {
                "type": "directive",
                "action": "summarize_research",
                "target": f"research_summary_{next_topic.replace(' ', '_')}.md",
                "message": f"Summarize findings for proactive topic: '{next_topic}'."
            }
        ]
        new_plan_content = {
             "plan_id": new_plan_id,
             "objective": f"Learn about {next_topic}",
             "actions": new_plan_sequence
        }
        
        # --- Log e Salvar Sugestão ---            
        suggestion_command = f"criar fragmento '{new_plan_id}' tipo 'GeneratedPlan' objective='{new_plan_content['objective']}' actions={new_plan_content['actions']}" # Exemplo A3L para plano
        suggestion_reason = f"Plano Proativo (ID: {new_plan_id}) gerado após reflexão bem-sucedida sobre '{original_topic}'. Objetivo: Aprender sobre '{next_topic}'."
        self._logger.info(f"[{self.get_name()}] Sugestão: {suggestion_reason} -> {suggestion_command}")
        append_to_log(f"# [Sugestão Planner] {suggestion_reason}. Comando: {suggestion_command}. Aguardando comando A3L.")
        if self.context_store:
             await self.context_store.push("pending_suggestions", {"command": suggestion_command, "reason": suggestion_reason, "source": self.get_name()})
        # --- Fim Log/Salvar ---

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete.")

# <<< ADD FRAGMENT DEFINITION >>>
PlannerFragmentDef = FragmentDef(
    name="Planner",
    description="Analyzes reflections and learning summaries to generate action plans.",
    # category="Planning", # Optional category
    fragment_class=PlannerFragment # <<< USE CLASS DIRECTLY >>>
)
# <<< END FRAGMENT DEFINITION >>>