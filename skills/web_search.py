# skills/web_search.py
import logging
import os
import requests
from core.tools import skill
from core.config import TAVILY_API_KEY
from core.llm_interface import call_llm

logger = logging.getLogger(__name__)

# --- Tavily API Client (Simplified) ---
def search_tavily(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Performs a search using the Tavily API."""
    api_url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic", # or "advanced"
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Format results to match expected structure
        results = []
        if "results" in data:
            for item in data["results"]:
                results.append({
                    "title": item.get("title", "N/A"),
                    "url": item.get("url", "#"),
                    "snippet": item.get("content", "N/A") # Using content as snippet
                })
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Tavily API request failed: {e}", exc_info=True)
        return [] # Return empty list on error
    except Exception as e:
        logger.error(f"Error processing Tavily response: {e}", exc_info=True)
        return []

# --- Stub Function ---
def search_stub(query: str, max_results: int = 3) -> list[dict]:
    """Returns placeholder search results."""
    logger.warning(f"Web search is using STUB function for query: '{query}'")
    results = []
    for i in range(1, max_results + 1):
        results.append({
            "title": f"Simulated Result {i} for '{query}'",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}&result={i}",
            "snippet": f"This is placeholder snippet #{i} describing the simulated search result for your query: '{query}'. Real web search is disabled or failed."
        })
    return results

# --- Skill Definition ---
@skill(
    name="web_search",
    description="Performs a web search using Tavily API to answer a query based on current information.",
    parameters={
        "query": (str, "")
    }
)
def web_search(query: str) -> dict:
    """
    Performs a web search using the Tavily API and returns the results.

    Args:
        query (str): The search query.

    Returns:
        dict: Standardized dictionary with the search results or error message.
    """
    logger.debug(f"Skill 'web_search' requested with query: '{query}'")

    # Check if Tavily is configured by checking if the API key exists
    if not TAVILY_API_KEY:
        logger.error("Tavily API key not found. Web search skill is disabled.")
        return {"status": "error", "action": "web_search_failed", "data": {"message": "Tavily API key not configured. Web search is disabled."}}

    if not query:
        return {"status": "error", "action": "web_search_failed", "data": {"message": "Search query cannot be empty."}}

    search_results = []
    search_method = "stub"

    logger.info("Attempting web search using Tavily API...")
    search_results = search_tavily(query, TAVILY_API_KEY)
    if search_results:
        search_method = "Tavily"
        logger.info(f"Tavily search successful, found {len(search_results)} results.")
    else:
        # Tavily failed, fall back to stub? Or just return error? Let's fallback for now.
        logger.warning("Tavily search failed or returned no results. Falling back to stub.")
        search_results = search_stub(query)
        search_method = "stub (Tavily fallback)"

    if not search_results:
        # If even the stub fails (which it shouldn't) or Tavily failed with no fallback
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": f"Web search failed to retrieve results for query: '{query}' (Method: {search_method})"}
        }

    return {
        "status": "success",
        "action": "web_search_results",
        "data": {
            "message": f"Successfully retrieved {len(search_results)} web search results for query: '{query}' (Method: {search_method}).",
            "query": query,
            "search_method": search_method,
            "results": search_results
        }
    }
