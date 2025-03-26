from duckduckgo_search import DDGS
import json

# Limite de resultados por busca
MAX_SEARCH_RESULTS = 3

def skill_search_web(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """
    Realiza uma busca na web usando DuckDuckGo.
    
    Args:
        entities (dict): Dicionário contendo as entidades extraídas do comando
        original_command (str): Comando original do usuário
        intent (str): Intenção identificada (default: None)
        history (list): Histórico de interações (default: None)
        
    Returns:
        dict: Resultado estruturado contendo os resultados da busca
    """
    print("\n[Skill: Web Search]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")
    
    # Verifica se temos uma query para buscar
    query = entities.get("query") or entities.get("topic")
    if not query:
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {
                "message": "Não entendi o que buscar."
            }
        }
    
    print(f"  Buscando por: '{query}'...")
    
    try:
        # Realiza a busca usando DuckDuckGo
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
            
        print(f"  Encontrados {len(results)} resultados.")
        
        # Formata os resultados
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "snippet": result.get("body", ""),
                "url": result.get("link", "")
            })
        
        return {
            "status": "success",
            "action": "web_search_completed",
            "data": {
                "query": query,
                "results": formatted_results,
                "message": f"Busca por '{query}' concluída."
            }
        }
        
    except Exception as e:
        print(f"  [Erro] Falha na busca: {e}")
        return {
            "status": "error",
            "action": "web_search_failed",
            "data": {
                "message": f"Erro ao realizar a busca: {str(e)}"
            }
        } 