# a3x/fragments/definitions.py
import os
from .basic_fragments import PlannerFragment, FinalAnswerProvider
from .file_manager_fragment import FileOpsManager
from .debugger import DebuggerFragment
from .meta_reflector_fragment import MetaReflectorFragment
from .evolution_orchestrator_fragment import EvolutionOrchestratorFragment
from .generative_mutator_fragment import GenerativeMutatorFragment
from .aggregate_evaluator_fragment import AggregateEvaluatorFragment
from .performance_manager_fragment import PerformanceManagerFragment

# Define os conjuntos de skills permitidas para cada Fragment Worker
# Os nomes das skills devem corresponder aos registrados no agente
# Make sure skill names here EXACTLY match the names returned by skill.get_name()
# <<< COMMENTED OUT: Metadata now handled by @fragment decorator >>>
# FRAGMENT_SKILLS = {
#     "FileManager": [
#         "read_file",
#         "write_file",
#         "list_directory",
#         "append_to_file",
#         "delete_path"
#     ],
#     "WebSearcher": [
#         "web_search",
#         # Add browser skills here if/when integrated and relevant for this role
#         # "get_page_content",

#      "Planner": [ # Fragment responsible for planning/decomposition
#          "hierarchical_planner",
#          # Could potentially use 'list_directory' or 'read_file' for context? Keep focused for now.
#      ],
#     "InformationRecall": [ # Fragment for memory access
#         # Add specific memory skill names here when implemented
#         # "recall_semantic",
#         # "recall_episodic",
#         "consult_learned_heuristics"
#     ],
#     "FinalAnswerProvider": [
#         "final_answer"
#     ],
#     # Add more specialized fragments as needed
#     # e.g., "ImageGenerator", "DataAnalyzer"
# }

# Lista de todos os Fragments disponíveis para o Orquestrador saber quem pode contratar
# We might want to dynamically generate this list based on loaded skills/config later
# <<< COMMENTED OUT: Registry now discovers fragments decorated with @fragment >>>
# AVAILABLE_FRAGMENTS = {
#     # Managers
#     "FileOpsManager": {
#         "module": "a3x.fragments.file_manager_fragment",
#         "class": "FileOpsManager",
#         "description": "Coordinates file operations by selecting and executing the appropriate file skill.",
#         "category": "Management",
#         "managed_skills": [
#             "read_file",
#             "write_file",
#             "list_directory",
#             "append_to_file",
#             "delete_path",
#         ],
#     },
#     "CodeExecutionManager": {
#         "module": "a3x.fragments.code_execution_fragment",
#         "class": "CodeExecutionManager",
#         "description": "Manages code execution, including validation and running code blocks.",
#         "category": "Management",
#         "managed_skills": ["execute_code"], # Example
#     },
#     # Executors
#     "PlannerFragment": {
#         "module": "a3x.fragments.basic_fragments",
#         "class": "PlannerFragment",
#         "description": "Generates a step-by-step plan to achieve an objective.",
#         "category": "Execution",
#         "skills": ["generate_plan"], # Assumes a skill named generate_plan
#     },
#     "FinalAnswerProvider": {
#         "module": "a3x.fragments.basic_fragments",
#         "class": "FinalAnswerProvider",
#         "description": "Provides the final answer or summary to the user.",
#         "category": "Execution",
#         "skills": ["final_answer"], # Assumes a skill named final_answer
#     },
#     "DebuggerFragment": {
#         "module": "a3x.fragments.debugger",
#         "class": "DebuggerFragment",
#         "description": "Analyzes persistent task failures and suggests diagnostic or corrective actions.",
#         "category": "Execution", # Or Diagnosis
#         "skills": ["llm_error_diagnosis", "read_file", "web_search"], # Skills it can use
#     },
# }

# Descrições para ajudar o Orquestrador LLM a escolher o Fragment correto
# <<< COMMENTED OUT: Description now comes from @fragment decorator via registry >>>
# FRAGMENT_DESCRIPTIONS = {
#     "FileManager": "Manages files and directories (reads, writes, lists, deletes, appends). Use for any file operations.",
#     "WebSearcher": "Searches the web using a search engine to find information or URLs.",
#     "CodeExecutor": "Executes shell commands or Python code snippets provided as input.",
#     "Planner": "Decomposes complex tasks into smaller, manageable steps or sub-tasks.",
#     "InformationRecall": "Consults learned heuristics or retrieves information from memory.",
#     "FinalAnswerProvider": "Provides the final answer or result directly to the user when the task is complete."
#     # Add descriptions for other fragments
# }

# Helper function to get skills for a fragment
# <<< COMMENTED OUT: Skills now retrieved from FragmentDef in registry >>>
# def get_skills_for_fragment(fragment_name: str) -> list[str]:
#     """Returns the list of allowed skill names for a given fragment."""
#     return FRAGMENT_SKILLS.get(fragment_name, [])

# Helper function to format fragment descriptions for the prompt
# <<< MODIFIED: Now should rely on registry.get_available_fragments_description() >>>
# (The function below is likely unused now, but kept for reference unless confirmed otherwise)
def format_fragment_descriptions_for_prompt() -> str:
    """Formats the fragment names and descriptions for inclusion in a prompt."""
    # This function might be obsolete if the agent directly calls registry.get_available_fragments_description()
    lines = ["Available Fragments (Workers):"]
    # Needs access to the registry instance or the definitions derived from it.
    # Placeholder - real implementation depends on how registry data is accessed here.
    # for name, desc in FRAGMENT_DESCRIPTIONS.items(): # Using old static list
    #     lines.append(f"- {name}: {desc}")
    return "Error: format_fragment_descriptions_for_prompt needs update to use FragmentRegistry." 