import logging
import traceback
import os
from tavily import TavilyClient # Import Tavily client

# Use config from core for API Key and enabled status
try:
    from core.config import TAVILY_ENABLED, TAVILY_API_KEY
except ImportError:
    # Fallback if running standalone or config is structured differently
    TAVILY_ENABLED = os.getenv("TAVILY_ENABLED", "False").lower() == "true"
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize logger
logger = logging.getLogger(__name__)
# Configure logger level for skill debugging (can be adjusted)
logger.setLevel(logging.INFO)

# Define the skill function using Tavily
def skill_search_web(action_input: dict, agent_memory: dict | None = None, agent_history: list | None = None, agent_logger: logging.Logger | None = None, **kwargs) -> dict:
    """
    Performs a web search using the Tavily API based on the provided query.

    Requires TAVILY_ENABLED=True in config and TAVILY_API_KEY in environment variables.

    Args:
        action_input (dict): Dictionary containing the search query and optional max_results.
                             Expected keys: "query" (string), "max_results" (int, optional, default 5).
        agent_memory (dict | None): Agent's memory (not directly used).
        agent_history (list | None): Conversation history (not directly used).
        agent_logger (logging.Logger | None): Logger instance passed from the agent.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        dict: A dictionary containing the status and search results or an error message.
              Example success: {"status": "success", "action": "web_search_completed", "data": {"results": [...], "message": "..."}}
              Example error: {"status": "error", "action": "web_search_failed/disabled", "data": {"message": "..."}}
    """
    log = agent_logger or logger # Use agent's logger if available, otherwise use the skill's logger
    log.info("[Skill: Search Web with Tavily]")
    log.debug(f"  Action Input: {action_input}")

    if not TAVILY_ENABLED:
        log.warning("  Tavily search is disabled in config.")
        return {"status": "error", "action": "web_search_disabled", "data": {"message": "Busca na web (Tavily) está desabilitada na configuração."}}

    if not TAVILY_API_KEY:
        log.error("  Error: TAVILY_API_KEY not found in environment variables.")
        return {"status": "error", "action": "web_search_failed", "data": {"message": "Chave da API Tavily (TAVILY_API_KEY) não encontrada."}}

    if not isinstance(action_input, dict):
         log.error(f"  Error: action_input is not a dictionary, but {type(action_input)}. Value: {action_input}")
         return {"status": "error", "action": "web_search_failed", "data": {"message": "Action Input inválido (não é um dicionário)."}}

    query = action_input.get("query")
    if not query or not isinstance(query, str):
        log.error("  Error: Missing or invalid 'query' parameter in action_input.")
        return {"status": "error", "action": "web_search_failed", "data": {"message": "Parâmetro 'query' ausente ou inválido."}}

    try:
        max_results = int(action_input.get("max_results", 5))
    except (ValueError, TypeError):
        log.warning(f"  Invalid 'max_results' value ({action_input.get('max_results')}). Using default 5.")
        max_results = 5

    try:
        log.info(f"  Performing Tavily search for: '{query}' (max_results: {max_results})")
        tavily = TavilyClient(api_key=TAVILY_API_KEY)

        # Perform the search
        # Include 'include_answer=False' if you only want the search results, not a summarized answer
        search_results_raw = tavily.search(query=query, search_depth="basic", max_results=max_results)

        # Extract relevant results (title, url, content/snippet)
        # Tavily's 'content' is usually the snippet.
        formatted_results = [
            {"title": r.get("title", "N/A"), "url": r.get("url", "#"), "snippet": r.get("content", "N/A")}
            for r in search_results_raw.get("results", [])
        ]

        log.info(f"  Search completed. Found {len(formatted_results)} results.")

        return {
            "status": "success",
            "action": "web_search_completed",
            "data": {
                "results": formatted_results,
                "message": f"Busca na web (Tavily) por '{query}' concluída com {len(formatted_results)} resultados."
            }
        }

    except Exception as e:
        log.error(f"  Error during Tavily search for '{query}': {e}")
        log.error(traceback.format_exc()) # Log the full traceback
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": f"Erro ao executar a busca na web (Tavily): {e}"}
        }

# Remove the old incorrect function if it still exists (should be overwritten by the edit)
# def basic_query(query): ...