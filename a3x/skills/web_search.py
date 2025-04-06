# skills/web_search.py
import logging
from duckduckgo_search import DDGS  # Import DDGS
from core.tools import skill
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Removed: search_tavily function
# Removed: search_stub function


# --- Skill Definition ---
@skill(
    name="web_search",
    # Updated description
    description="Performs a web search using DuckDuckGo to answer a query based on current information.",
    parameters={"query": (str, ""), "max_results": (int, 5)},
)
def web_search(query: str, max_results: int = 5) -> dict:  # Added max_results parameter
    """
    Performs a web search using the DuckDuckGo search engine and returns the results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 5.

    Returns:
        dict: Standardized dictionary with the search results or error message.
    """
    logger.debug(f"Skill 'web_search' requested with query: '{query}'")

    # Removed: Check for TAVILY_API_KEY

    if not query:
        logger.warning("Web search called with empty query.")
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": "Search query cannot be empty."},
        }

    search_results = []
    search_method = "DuckDuckGo"

    logger.info(f"Attempting web search using DuckDuckGo for query: '{query}'...")
    try:
        # Use DDGS().text() for the search
        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(query, max_results=max_results)
            )  # Get results as list

        if raw_results:
            # Map results to the expected format
            for item in raw_results:
                search_results.append(
                    {
                        "title": item.get("title", "N/A"),
                        "url": item.get("href", "#"),  # Map 'href' to 'url'
                        "snippet": item.get("body", "N/A"),  # Map 'body' to 'snippet'
                    }
                )
            logger.info(
                f"DuckDuckGo search successful, found {len(search_results)} results."
            )
        else:
            logger.warning(f"DuckDuckGo search for '{query}' returned no results.")
            # Return success but with empty results list, or a specific status?
            # Let's return success with empty list for now. Client can decide how to handle.

    except Exception as e:
        logger.error(
            f"DuckDuckGo search failed for query '{query}': {e}", exc_info=True
        )
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": f"Web search failed due to an error: {e}"},
        }

    # Always return success status, even if results are empty
    return {
        "status": "success",
        "action": "web_search_results",
        "data": {
            "message": f"Successfully retrieved {len(search_results)} web search results for query: '{query}' (Method: {search_method}).",
            "query": query,
            "search_method": search_method,
            "results": search_results,  # This will be empty if no results were found
        },
    }


@skill(
    name="search_web_tavily",
    description="Performs a web search using the Tavily API to answer a query.",
    parameters={
        "query": (str, ...),
        "max_results": (int, 5),  # Default number of results
        "include_domains": (list, None),  # REVERTED for @skill compatibility
        "exclude_domains": (list, None),  # REVERTED for @skill compatibility
    },
)
def search_web_tavily(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {}  # ADDED placeholder return
    # ... existing code ...


# Removed the original web_search function content that used Tavily/stub
