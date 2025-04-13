# a3x/fragments/definitions.py
import os

# Define os conjuntos de skills permitidas para cada Fragment Worker
# Os nomes das skills devem corresponder aos registrados no agente
# Make sure skill names here EXACTLY match the names returned by skill.get_name()
FRAGMENT_SKILLS = {
    "FileManager": [
        "read_file",
        "write_file",
        "list_directory",
        "append_to_file",
        "delete_path"
    ],
    "WebSearcher": [
        "web_search",
        # Add browser skills here if/when integrated and relevant for this role
        # "get_page_content",
    ],
    "CodeExecutor": [
        "execute_code",
        # Maybe add generate_code if this fragment should also generate?
    ],
     "Planner": [ # Fragment responsible for planning/decomposition
         "hierarchical_planner",
         # Could potentially use 'list_directory' or 'read_file' for context? Keep focused for now.
     ],
    "InformationRecall": [ # Fragment for memory access
        # Add specific memory skill names here when implemented
        # "recall_semantic",
        # "recall_episodic",
        "consult_learned_heuristics"
    ],
    "FinalAnswerProvider": [
        "final_answer"
    ],
    # Add more specialized fragments as needed
    # e.g., "ImageGenerator", "DataAnalyzer"
}

# Lista de todos os Fragments disponíveis para o Orquestrador saber quem pode contratar
# We might want to dynamically generate this list based on loaded skills/config later
AVAILABLE_FRAGMENTS = list(FRAGMENT_SKILLS.keys())

# Descrições para ajudar o Orquestrador LLM a escolher o Fragment correto
FRAGMENT_DESCRIPTIONS = {
    "FileManager": "Manages files and directories (reads, writes, lists, deletes, appends). Use for any file operations.",
    "WebSearcher": "Searches the web using a search engine to find information or URLs.",
    "CodeExecutor": "Executes shell commands or Python code snippets provided as input.",
    "Planner": "Decomposes complex tasks into smaller, manageable steps or sub-tasks.",
    "InformationRecall": "Consults learned heuristics or retrieves information from memory.",
    "FinalAnswerProvider": "Provides the final answer or result directly to the user when the task is complete."
    # Add descriptions for other fragments
}

# Helper function to get skills for a fragment
def get_skills_for_fragment(fragment_name: str) -> list[str]:
    """Returns the list of allowed skill names for a given fragment."""
    return FRAGMENT_SKILLS.get(fragment_name, [])

# Helper function to format fragment descriptions for the prompt
def format_fragment_descriptions_for_prompt() -> str:
    """Formats the fragment names and descriptions for inclusion in a prompt."""
    lines = ["Available Fragments (Workers):"]
    for name, desc in FRAGMENT_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) 