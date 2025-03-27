import requests
import json
import re
from core.config import LLAMA_SERVER_URL
from core.dispatcher import SKILL_DISPATCHER # Importado para validação
# from core.llm_clients import call_llm # Assuming you have this - Keep commented if not used

# Placeholder for keywords that might trigger planning
PLANNING_KEYWORDS = ["depois", "então", "primeiro", "segundo", "e então", "em seguida", "antes de", "após"]

# Placeholder for intents that are inherently simple
SIMPLE_INTENTS = ["get_value", "recall_info", "remember_info", "weather_forecast", "search_web", "unknown", "greet", "goodbye"] # Add more as needed

def generate_plan(command: str, nlu_result: dict, history: list, available_skills: list) -> list[dict]:
    """
    Analisa o comando e o resultado da NLU para determinar se um plano de múltiplos passos é necessário.
    Se sim, tenta gerar um plano (atualmente desativado).
    Se não, retorna um plano de passo único baseado na NLU.

    Args:
        command: O comando original do usuário.
        nlu_result: O resultado da interpretação da NLU.
        history: O histórico da conversa.
        available_skills: Lista de nomes de skills disponíveis.

    Returns:
        Uma lista de dicionários representando os passos do plano,
        ou uma lista com um único passo se o planejamento não for necessário/possível.
        Retorna [] se nenhuma ação puder ser determinada.
    """
    intent = nlu_result.get("intent", "unknown")
    entities = nlu_result.get("entities", {})
    needs_planning = False

    # 1. Verificar se a intenção é complexa ou se há keywords de planejamento
    #    Consideramos planejamento necessário se a intenção NÃO for simples.
    if intent not in SIMPLE_INTENTS:
        needs_planning = True
        print(f"[Planner] Intenção '{intent}' pode necessitar de planejamento.")
    # elif any(keyword in command.lower() for keyword in PLANNING_KEYWORDS):
    #     needs_planning = True
    #     print(f"[Planner] Comando contém keywords de planejamento.")

    # 2. Se o planejamento NÃO for considerado necessário
    if not needs_planning:
        print(f"[Planner] Intenção '{intent}' considerada simples. Executando como ação única.")
        # Retorna um plano de passo único com a intenção original da NLU
        if intent != "unknown": # Só retorna plano se a NLU achou algo útil
             # Usamos 'intent' como chave para consistência com NLU
             return [{"intent": intent, "entities": entities}]
        else:
             print("[Planner] NLU retornou 'unknown' e não há necessidade de planejamento complexo. Retornando plano vazio.")
             return [] # NLU não ajudou, planejamento não necessário, retorna vazio

    # 3. Se o planejamento FOI considerado necessário
    print("[Planner] Comando parece necessitar de planejamento sequencial.")

    # --- CHAMADA AO LLM PARA PLANEJAMENTO (Atualmente Desativado) ---
    use_llm_planning = False # Mudar para True para ativar
    if use_llm_planning:
        print("[Planner] Tentando gerar plano via LLM...")
        # Placeholder para lógica LLM
        # prompt = f"..."
        # try:
        #     llm_response = call_llm(prompt)
        #     plan = json.loads(llm_response)
        #     # Validar plano...
        #     print(f"[Planner] Plano gerado pelo LLM: {plan}")
        #     return plan
        # except Exception as e:
        #     print(f"[Erro Planner] Falha ao gerar ou parsear plano do LLM: {e}")
        #     # Fallback: Tentar executar o primeiro passo identificado pela NLU
        #     print("[Planner] Fallback: Executando primeiro passo da NLU.")
        #     if intent != "unknown":
        #          return [{"intent": intent, "entities": entities}]
        #     else:
        #          return []
        pass # Mantém desativado
    else:
        print("[Planner] Planejamento via LLM está DESATIVADO.")
        # Fallback: Se o planejamento é necessário mas o LLM está off,
        # retorna um plano de passo único com a intenção original da NLU.
        # Isso permite que sequências simples (onde a NLU acerta o primeiro passo) funcionem.
        if intent != "unknown":
            print(f"[Planner] Retornando plano de passo único com base na NLU: {intent}")
            return [{"intent": intent, "entities": entities}]
        else:
            # Se a NLU retornou unknown e o planejamento LLM está off, não há o que fazer.
            print("[Planner] NLU retornou 'unknown' e planejamento LLM desativado. Retornando plano vazio.")
            return []

    # --- Geração do Plano usando LLM (REMOVIDO/COMENTADO TEMPORARIAMENTE) ---
    # print("[Planner] Gerando plano com LLM...")
    # formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])
    # planning_prompt = f"""...""" # Prompt removido
    # try:
    #     headers = {"Content-Type": "application/json"}
    #     payload = {...} # Payload removido
    #     print(f"[Planner] Enviando payload ao LLM...")
    #     response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
    #     # ... resto da lógica de request e parsing removida ...
    # except requests.exceptions.RequestException as e_req:
    #     print(f"[Planner Error] Erro na requisição ao LLM: {e_req}")
    #     return []
    # except Exception as e_other:
    #     print(f"[Planner Error] Erro inesperado durante planejamento: {e_other}")
    #     return [] 