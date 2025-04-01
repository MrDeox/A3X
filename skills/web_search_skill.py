import json
from duckduckgo_search import DDGS
from core.tools import skill

@skill(
    name="web_search",
    description="Searches the web using DuckDuckGo to find information based on a query.",
    parameters={
        "query": (str, ...)
    }
)
def web_search(query: str) -> dict:
    """
    Performs a web search using DuckDuckGo and returns the top results.

    Args:
        query: The search query.

    Returns:
        A dictionary containing the search results or an error message.
    """
    try:
        print(f"[Web Search Skill] Performing search for: {query}")
        with DDGS() as ddgs:
            # Fetching text results, limiting to top 5 for brevity
            results = list(ddgs.text(query, max_results=5))

        if not results:
            print("[Web Search Skill] No results found.")
            return {"status": "success", "results": "No results found."}

        # Formatting results for clarity
        formatted_results = [
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
        ]
        print(f"[Web Search Skill] Found {len(formatted_results)} results.")
        return {"status": "success", "results": formatted_results}

    except Exception as e:
        print(f"[Web Search Skill] Error during search: {e}")
        return {"status": "error", "message": f"Failed to perform web search: {str(e)}"}

# Example Usage (for testing purposes)
if __name__ == '__main__':
    search_query = "What is the weather in Paris?"
    search_results = web_search(search_query)
    print(f"\nSearch Results for '{search_query}':")
    print(search_results)

    search_query = "Latest AI news"
    search_results = web_search(search_query)
    print(f"\nSearch Results for '{search_query}':")
    print(search_results) 