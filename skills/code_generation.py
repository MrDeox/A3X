import requests
from core.config import LLAMA_SERVER_URL

def skill_generate_code(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Gera código baseado nas entidades fornecidas."""
    print(f"[DEBUG] Entidades recebidas: {entities}")
    if history:
        print(f"[DEBUG] Últimos 4 turnos do histórico: {history[-4:]}")
    
    language = entities.get("language", "python")
    construct_type = entities.get("construct_type", "function")
    purpose = entities.get("purpose", "")
    
    if not purpose:
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {
                "message": "Não foi possível entender o propósito do código a ser gerado."
            }
        }
    
    try:
        # Construir o prompt para o LLM
        prompt = f"""Gere um código em {language} que {purpose}.
        O código deve ser uma {construct_type}.
        Retorne APENAS o código, sem explicações adicionais."""
        
        # Enviar para o LLM
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        
        if response.status_code != 200:
            return {
                "status": "error",
                "action": "code_generation_failed",
                "data": {
                    "message": f"Erro ao gerar código: {response.text}"
                }
            }
        
        # Extrair o código da resposta
        code = response.json()["choices"][0]["message"]["content"].strip()
        
        return {
            "status": "success",
            "action": "code_generated",
            "data": {
                "code": code,
                "language": language,
                "construct_type": construct_type,
                "message": f"Código gerado com sucesso em {language}."
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {
                "message": f"Erro ao gerar código: {str(e)}"
            }
        } 