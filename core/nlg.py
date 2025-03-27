import requests
import json
from .config import LLAMA_SERVER_URL, MAX_HISTORY_TURNS

def generate_simplified_response(skill_result: dict, history: list) -> str:
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

def generate_natural_response(skill_result: dict, history: list) -> str:
    """Gera uma resposta natural usando o LLM local baseada na ação e histórico."""
    print("\n[NLG-LLM] Gerando resposta natural...")
    
    # Extrair informações do skill_result
    status = skill_result.get("status")
    action = skill_result.get("action")
    data = skill_result.get("data", {})
    
    # Pegar o último comando do usuário do histórico
    last_user_command = ""
    if history and len(history) > 0:
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user_command = turn.get("content", "")
                break
    
    # Criar um resumo conciso da ação
    action_summary = f"Status: {status}\nAção: {action}"

    # Adicionar detalhes específicos baseados na ação
    if status == "success":
        if action == "files_listed":
            if data.get("message"):
                action_summary += f"\nResumo dos dados: {data.get('message')}"
            else:
                action_summary += f"\nResumo dos dados: Arquivos listados (sem detalhes)."

        elif action == "web_search":
            results = data.get('results', [])
            if results:
                action_summary += f"\nResumo dos dados: Resultados encontrados:\n{results}"
            else:
                action_summary += "\nResumo dos dados: Nenhum resultado encontrado."

        elif action == "code_executed":
            output = data.get('output', '').strip()
            if output:
                # Limita o tamanho do output no prompt para não ficar gigante
                output_summary = output[:200] + ('...' if len(output) > 200 else '')
                action_summary += f"\nResumo dos dados: Código executado com a seguinte saída:\n---\n{output_summary}\n---"
            else:
                action_summary += f"\nResumo dos dados: Código executado sem saída visível."
            # Opcional: Adicionar info de 'final_locals' se desejado e não muito grande

        elif action == "web_search_completed" and isinstance(data.get('results'), list):
            num_results = len(data['results'])
            titles = [r.get('title', 'Sem título') for r in data['results']]
            action_summary += f"\nResumo dos dados: {num_results} resultado(s) da web encontrado(s), títulos: {'; '.join(titles)}"
        
        elif action in ["code_generated", "code_modified"] and data.get('file_name'):
            action_summary += f"\nResumo dos dados: Código {'gerado' if action == 'code_generated' else 'modificado'} no arquivo: {data['file_name']}"
        
        elif action == "info_recalled" and data.get('value'):
            action_summary += f"\nResumo dos dados: Informação recuperada para a chave '{data.get('key', '?')}': {data['value']}"
        
        # Para outras ações ou se nenhum caso específico se aplicar
        elif data.get('message'):
            action_summary += f"\nDetalhes: {data.get('message')}"
    else:
        # Para ações não bem-sucedidas, manter o comportamento original
        if data.get('message'):
            action_summary += f"\nDetalhes: {data.get('message')}"
    
    # Construir o prompt para o LLM com instrução atualizada
    prompt = f"""Você é A³X, um assistente de IA local e prestativo.
O usuário disse: "{last_user_command}"
A seguinte ação interna foi realizada:
{action_summary}

Com base nisso (especialmente no resumo dos dados, se houver) e na conversa recente, gere uma resposta curta, útil e amigável para o usuário final. Não explique a ação interna em detalhes, apenas responda ao usuário de forma natural.
Resposta para o usuário:"""

    print(f"[NLG-LLM] Prompt: {prompt}")

    try:
        # Enviar o prompt para o servidor LLM
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "n_predict": 128,  # Limitando para respostas mais concisas
            "temperature": 0.7,
            "stop": ["\n", "Usuário:", "A³X:"]
        }

        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        raw_response = response_data.get("content", "").strip()
        print(f"[NLG-LLM] Raw Response: {raw_response}")

        # Limpar a resposta
        if "Resposta para o usuário:" in raw_response:
            raw_response = raw_response.split("Resposta para o usuário:")[-1].strip()
        
        # Pegar apenas a primeira linha significativa
        response_lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
        if response_lines:
            return response_lines[0]
        else:
            raise ValueError("Resposta vazia do LLM")

    except Exception as e:
        print(f"[NLG-LLM] Erro ao gerar resposta: {e}")
        return f"[LLM indisponível] {generate_simplified_response(skill_result, history)}" 