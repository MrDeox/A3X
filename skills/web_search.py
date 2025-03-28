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
        # --- INTRODUZIR BUG AQUI ---
        # Tenta concatenar string com inteiro para causar TypeError
        problematic_query = query + 1
        # --- FIM DO BUG INTRODUZIDO ---

        # A busca DDGS usaria a query original, mas o erro ocorrerá antes
        with DDGS() as ddgs:
            # Usamos a query original aqui para a busca não falhar por causa do bug
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
    except TypeError as e: # <<< Captura específica do TypeError >>>
         print(f"  [Erro Web Search - BUG INTENCIONAL] Falha ao processar query: {e}")
         # Retorna o erro específico para o agente analisar
         return {"status": "error", "action": "web_search_failed", "data": {"message": f"Erro interno ao processar query (TypeError): {e}"}}
    except Exception as e:
        print(f"  [Erro Web Search] Falha na busca: {e}")
        # Log traceback para depuração interna, mas não exponha ao LLM diretamente
        logger.error(f"[Skill: Web Search] Exception during search for '{query}': {e}", exc_info=True)
        return {"status": "error", "action": "web_search_failed", "data": {"message": f"Erro inesperado ao realizar a busca: {str(e)}"}} 