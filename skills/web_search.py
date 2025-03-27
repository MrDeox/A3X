from duckduckgo_search import DDGS
import json

# Limite de resultados por busca
MAX_SEARCH_RESULTS = 3

def skill_search_web(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """Realiza busca web usando DuckDuckGo (Assinatura ReAct)."""
    print("\n[Skill: Web Search (ReAct)]")
    print(f"  Action Input: {action_input}")

    # --- Use action_input ---
    query = action_input.get("query")
    if not query:
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {"message": "Parâmetro 'query' ausente no Action Input."} # Updated error message
        }

    print(f"  Buscando por: '{query}'...")
    try:
        # --- DDGS logic remains the same ---
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
        print(f"  Encontrados {len(results)} resultados.")

        # Use href for url
        formatted_results = [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in results]

        # --- Return structure remains the same ---
        return {
            "status": "success",
            "action": "web_search_completed",
            "data": {
                "query": query,
                "results": formatted_results,
                "message": f"Busca por '{query}' concluída com {len(results)} resultado(s)."
            }
        }
    except Exception as e:
        print(f"  [Erro Web Search] Falha na busca: {e}")
        # traceback.print_exc() # Uncomment for debug
        return {"status": "error", "action": "web_search_failed", "data": {"message": f"Erro ao realizar a busca: {str(e)}"}} 