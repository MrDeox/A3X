import requests
import json
import re
from core.config import LLAMA_SERVER_URL

def generate_plan(user_goal: str, nlu_result: dict, history: list, available_skills: list) -> list:
    """
    Tenta gerar um plano sequencial de sub-tarefas (skills) para atingir um objetivo do usuário.

    Args:
        user_goal: O comando original do usuário.
        nlu_result: O resultado da interpretação inicial da NLU.
        history: O histórico da conversa.
        available_skills: Uma lista de strings com os nomes das intenções/skills disponíveis.

    Returns:
        Uma lista de dicionários representando os passos do plano, onde cada dicionário
        tem 'intent' e 'entities' (similar ao formato nlu_result).
        Retorna uma lista vazia [] se nenhum plano for necessário ou se o planejamento falhar.
    """
    print("\n[Planner] Tentando gerar plano sequencial...")
    
    # Lógica para decidir SE o planejamento é necessário
    intent = nlu_result.get("intent", "unknown")
    entities = nlu_result.get("entities", {})

    # Keywords que indicam sequência de ações
    keywords_sequence = [" e depois", " e então", " primeiro ", " segundo ", " após ", " em seguida"]
    contains_sequence_keyword = any(keyword in user_goal.lower() for keyword in keywords_sequence)

    # Verifica se a intenção é específica (não genérica)
    is_specific_intent = intent not in ["unknown", "error_parsing", "error_connection", "error_unknown"]

    # Condição Simplificada: Se NÃO houver keyword de sequência E a intenção for específica, pule.
    if not contains_sequence_keyword and is_specific_intent:
        print("[Planner] Nenhuma keyword de sequência e NLU específica. Pulando planejamento.")
        return [] # Retorna lista vazia, indicando para usar a NLU inicial
    else:
        # Se chegou aqui (tem keyword OU intenção não é específica), prossegue
        print("[Planner] Comando pode precisar de planejamento (keyword encontrada ou NLU não específica). Consultando LLM...")

    # Construção do prompt para o LLM
    planning_prompt = f"""<s> Você é um planejador sequencial especializado em decompor objetivos complexos em sub-tarefas.
    
### Objetivo do Usuário:
{user_goal}

### Skills Disponíveis:
{', '.join(available_skills)}

### Instruções:
1. Analise o objetivo do usuário e decida se ele pode ser satisfeito por uma única skill ou se precisa ser decomposto.
2. Se precisar decompor, crie uma sequência de passos usando as skills disponíveis.
3. Cada passo deve ser um dicionário com:
   - "intent": uma das skills disponíveis
   - "entities": parâmetros necessários para essa skill
4. Se o objetivo pode ser satisfeito por uma única skill, retorne uma lista vazia []
5. Para criar um arquivo de texto simples com conteúdo específico, use a skill `manage_files` com a entidade `action: create` e o `content` desejado.
6. Para adicionar conteúdo a um arquivo de texto existente, use a skill `modify_code` com a `modification` apropriada (ex: 'adicione a linha X ao final do arquivo').
7. Use `generate_code` APENAS quando o objetivo for gerar CÓDIGO executável (Python, etc.), não para simples manipulação de arquivos de texto.

### Exemplos:

Exemplo 1:
Goal: "Crie um script python hello.py que imprime ola e depois adicione um comentário # Feito"
Skills: ["generate_code", "modify_code", "manage_files", ...]
Resultado:
```json
[
    {{"intent": "generate_code", "entities": {{"language": "python", "file_name": "hello.py", "purpose": "imprime ola"}}}},
    {{"intent": "modify_code", "entities": {{"file_name": "hello.py", "modification": "adicione um comentário # Feito"}}}}
]
```

Exemplo 2:
Goal: "Lembre que meu projeto é o A3X e depois me diga qual o nome dele"
Skills: ["remember_info", "recall_info", "search_web", ...]
Resultado:
```json
[
    {{"intent": "remember_info", "entities": {{"key": "meu projeto", "value": "A3X"}}}},
    {{"intent": "recall_info", "entities": {{"key": "nome do meu projeto"}}}}
]
```

Exemplo 3:
Goal: "Qual a capital da França?"
Skills: ["search_web", "manage_files", ...]
Resultado:
```json
[]
```

Exemplo 4:
Goal: "Crie um arquivo notas.txt com 'Lembrete 1' e depois adicione 'Lembrete 2'"
Skills: ["manage_files", "modify_code", "generate_code", ...]
Resultado:
```json
[
    {{"intent": "manage_files", "entities": {{"action": "create", "file_name": "notas.txt", "content": "Lembrete 1"}}}},
    {{"intent": "modify_code", "entities": {{"file_name": "notas.txt", "modification": "adicione a linha 'Lembrete 2' ao final"}}}}
]
```

### Plano para o Objetivo Atual:
```json
"""

    print(f"[Planner] Planning Prompt:\n{planning_prompt}")

    try:
        # Chamada ao LLM
        payload = {
            "prompt": planning_prompt,
            "n_predict": 1024,
            "temperature": 0.2,
            "stop": ["```"]
        }
        
        response = requests.post(LLAMA_SERVER_URL, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        print(f"[Planner] Raw Plan Response:\n{response_data}")
        
        # Extração e limpeza da resposta
        content = response_data.get("content", "").strip()
        if not content:
            print("[Planner] Resposta vazia do LLM")
            return []
            
        # Limpeza do JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if not json_match:
            print("[Planner] Nenhuma lista JSON encontrada na resposta")
            return []
            
        json_str = json_match.group(0)
        print(f"[Planner] JSON extraído:\n{json_str}")
        
        # Parsing do JSON
        try:
            plan = json.loads(json_str)
            print(f"[Planner] Plano parseado:\n{json.dumps(plan, indent=2)}")
            
            # Validação simples do plano
            if not isinstance(plan, list):
                print("[Planner] Resultado não é uma lista")
                return []
                
            for step in plan:
                if not isinstance(step, dict) or "intent" not in step:
                    print("[Planner] Passo inválido no plano")
                    return []
                    
            return plan
            
        except json.JSONDecodeError as e:
            print(f"[Planner] Erro ao decodificar JSON: {e}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"[Planner] Erro na requisição ao LLM: {e}")
        return []
    except Exception as e:
        print(f"[Planner] Erro inesperado: {e}")
        return [] 