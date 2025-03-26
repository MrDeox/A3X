import requests
import json
from .config import LLAMA_SERVER_URL, MAX_HISTORY_TURNS

def generate_natural_response(skill_result: dict, history: list) -> str:
    """Retorna a mensagem principal do resultado da skill (NLG Simplificada)."""
    print("\n[NLG Simplificada]")
    status = skill_result.get("status")
    data = skill_result.get("data", {})
    message = data.get("message", None) # Tenta pegar a mensagem do resultado

    if message:
         # Se a skill retornou uma mensagem útil, usa ela
         # Tenta limpar a parte inicial "Código ... gerado:" se for de generate_code
         if "Código " in message and " gerado:" in message and "\n---" in message:
              message = message.split("---\n", 1)[1].split("\n---")[0].strip()
         return message
    elif status == "success":
         return f"Ok, ação '{skill_result.get('action', 'desconhecida')}' concluída."
    elif status == "not_understood" or status == "cancelled":
         return "Comando não entendido ou cancelado."
    else: # Erro
         return "Desculpe, ocorreu um erro ao processar seu comando."

def generate_natural_response_old(skill_result: dict, history: list) -> str:
    """Gera uma resposta natural baseada na ação e dados recebidos."""
    print("\n[NLG] Gerando resposta natural...")
    
    # Extrair informações do skill_result
    status = skill_result.get("status")
    action = skill_result.get("action")
    data = skill_result.get("data", {})
    
    # Primeiro, criar um resumo da ação para o prompt
    action_summary = f"A ação '{action}' foi tentada."
    if status == "success":
        if action == "info_remembered":
            action_summary += f" Resultado: Sucesso."
        elif action == "info_recalled":
            action_summary += f" Resultado: Sucesso."
        elif action == "web_search_completed":
            query = data.get('query')
            results = data.get('results', [])
            if results:
                action_summary += f" Encontrei {len(results)} resultado(s) para '{query}'."
                # Incluir títulos e snippets no resumo para o LLM
                action_summary += "\nResultados:\n"
                for i, result in enumerate(results, 1):
                    action_summary += f"{i}. {result['title']}\n{result['snippet']}\n"
            else:
                action_summary += f" Não encontrei resultados para '{query}'."
    else:
        action_summary += f" Resultado: {status}. Mensagem: {data.get('message', 'Erro desconhecido')}"

    # Criar o prompt para o LLM
    prompt = f"""Você é um assistente prestativo. O usuário deu um comando, e a seguinte ação foi realizada internamente:
'{action_summary}'

Com base nessa ação e na conversa recente, formule uma resposta **concisa e natural** para o usuário final. Seja direto e útil. Evite formalidades excessivas. Não inclua o resumo da ação na sua resposta final, apenas a resposta para o usuário.
"""

    print(f"  [NLG] Prompt enviado ao LLM:\n---\n{prompt}\n---")

    try:
        # Enviar o prompt para o servidor LLM
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.7,  # Aumentado para mais criatividade nas respostas
            "stop": ["```"],
        }

        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        llm_output = response_data.get("content", "").strip()

        # Extrair apenas a resposta, removendo qualquer formatação
        if "Resposta:" in llm_output:
            llm_output = llm_output.split("Resposta:")[-1].strip()
        
        # Limpa a resposta pegando a primeira linha
        first_line_response = llm_output.splitlines()[0].strip()
        return first_line_response if first_line_response else "(Resposta natural gerada estava vazia.)"

    except Exception as e:
        print(f"[Erro NLG] Falha ao gerar resposta: {e}")
        return "(Erro ao gerar resposta natural.)" 