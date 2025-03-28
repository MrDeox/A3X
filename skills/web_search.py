import logging
import traceback
from duckduckgo_search import DDGS # Import the library

# Initialize logger
logger = logging.getLogger(__name__)

# Define the correct skill function
def skill_search_web(action_input: dict, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Performs a web search using DuckDuckGo based on the provided query.

    Args:
        action_input (dict): Dictionary containing the search query.
                             Expected key: "query" (string).
        agent_memory (dict | None): Agent's memory (not used in this skill).
        agent_history (list | None): Conversation history (not used in this skill).

    Returns:
        dict: A dictionary containing the status and search results or an error message.
              Example success: {"status": "success", "action": "web_search_completed", "data": {"results": [...], "message": "..."}}
              Example error: {"status": "error", "action": "web_search_failed", "data": {"message": "..."}}
    """
    logger.info("[Skill: Search Web]")
    logger.debug(f"  Action Input: {action_input}")

    query = action_input.get("query")

    if not query or not isinstance(query, str):
        logger.error("  Error: Missing or invalid 'query' parameter in action_input.")
        return {"status": "error", "action": "web_search_failed", "data": {"message": "Parâmetro 'query' ausente ou inválido."}}

    try:
        logger.info(f"  Performing web search for: '{query}'")
        # Use the DDGS context manager for searching
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5)) # Get top 5 text results

        logger.info(f"  Search completed. Found {len(search_results)} results.")

        # Format results (optional, but good for consistency)
        # The results from ddgs.text are already dicts with 'title', 'href', 'body'
        formatted_results = [
            {"title": r.get("title", "N/A"), "url": r.get("href", "#"), "snippet": r.get("body", "N/A")}
            for r in search_results
        ]

        return {
            "status": "success",
            "action": "web_search_completed",
            "data": {
                "results": formatted_results,
                "message": f"Busca na web por '{query}' concluída com {len(formatted_results)} resultados."
            }
        }

    except Exception as e:
        logger.error(f"  Error during web search for '{query}': {e}")
        logger.error(traceback.format_exc()) # Log the full traceback for debugging
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": f"Erro ao executar a busca na web: {e}"}
        }

# Remove the old incorrect function if it still exists (should be overwritten by the edit)
# def basic_query(query): ...