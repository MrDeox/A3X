import requests
import json
import re
from .config import LLAMA_SERVER_URL, MAX_HISTORY_TURNS

def create_nlu_prompt(command: str, history: list = None) -> str:
    """Cria o prompt para o LLM com exemplos few-shot."""
    
    # Exemplos essenciais (few-shot)
    examples = """
Comando: "gere um script python chamado utils.py com uma função hello world"
JSON Resultante:
```json
{
  "intent": "generate_code",
  "entities": {
    "language": "python",
    "construct_type": "function",
    "purpose": "hello world",
    "file_name": "utils.py"
  }
}
```

Comando: "crie um arquivo config.ini com [DB]
user=admin"
JSON Resultante:
```json
{
  "intent": "manage_files",
  "entities": {
    "action": "create",
    "file_name": "config.ini",
    "content": "[DB]
user=admin"
  }
}
```

Comando: "liste os arquivos .py"
JSON Resultante:
```json
{
  "intent": "manage_files",
  "entities": {
    "action": "list",
    "file_extension": ".py"
  }
}
```

Comando: "busque na web sobre a história do Python"
JSON Resultante:
```json
{
  "intent": "search_web",
  "entities": {
    "query": "história do Python"
  }
}
```

Comando: "quem descobriu o Brasil?"
JSON Resultante:
```json
{
  "intent": "search_web",
  "entities": {
    "query": "quem descobriu o Brasil"
  }
}
```

Comando: "qual a previsão do tempo para amanhã em Curitiba?"
JSON Resultante:
```json
{
  "intent": "search_web",
  "entities": {
    "query": "previsão do tempo Curitiba amanhã"
  }
}
```

Comando: "Lembre que meu email é arthur@email.com"
JSON Resultante:
```json
{
  "intent": "remember_info",
  "entities": {
    "key": "meu email",
    "value": "arthur@email.com"
  }
}
```

Comando: "Qual é o meu email?"
JSON Resultante:
```json
{
  "intent": "recall_info",
  "entities": {
    "key": "meu email"
  }
}
```

Comando: "me diga qual é o objetivo do A3X"
JSON Resultante:
```json
{
  "intent": "recall_info",
  "entities": {
    "key": "objetivo do A3X"
  }
}
```

Comando: "mostre o objetivo do projeto"
JSON Resultante:
```json
{
  "intent": "recall_info",
  "entities": {
    "key": "objetivo do projeto"
  }
}
```

Comando: "qual a senha do wifi?"
JSON Resultante:
```json
{
  "intent": "recall_info",
  "entities": {
    "key": "senha do wifi"
  }
}
```

Comando: "adicione um print 'Feito!' ao final da função anterior"
JSON Resultante:
```json
{
  "intent": "modify_code",
  "entities": {
    "target": "função anterior",
    "modification": "adicione um print 'Feito!' ao final"
  }
}
```

Comando: "refatore o script hello.py para usar uma função"
JSON Resultante:
```json
{
  "intent": "modify_code",
  "entities": {
    "target": "script hello.py",
    "modification": "refatorar para usar uma função",
    "file_name": "hello.py"
  }
}
```

Comando: "execute o código anterior"
JSON Resultante:
```json
{
  "intent": "execute_code",
  "entities": {}
}
```

Comando: "execute o script teste.py"
JSON Resultante:
```json
{
  "intent": "execute_code",
  "entities": {
    "file_name": "teste.py"
  }
}
```

Comando: "adicione a linha 'nova entrada' ao arquivo log.txt"
JSON Resultante:
```json
{
  "intent": "manage_files",
  "entities": {
    "file_name": "log.txt",
    "action": "append",
    "content": "nova entrada"
  }
}
```
"""
    
    # Construir o prompt completo
    prompt = f"""Analise o **Comando Atual** do usuário considerando o histórico e responda APENAS com JSON contendo "intent" e "entities".
               
### Histórico Recente da Conversa:
{history}

### Exemplos Essenciais

{examples}

### Comando Atual

Comando: "{command}"
JSON Resultante:
```json
"""
    
    return prompt

def interpret_command(user_input: str, history: list) -> dict:
    """Interpreta o comando do usuário usando o LLM."""
    try:
        # Criar o prompt NLU com histórico
        nlu_prompt = create_nlu_prompt(user_input, history)
        print(f"[DEBUG] Enviando prompt para o LLM:\n---\n{nlu_prompt}\n---") # DEBUG PROMPT

        # Enviar o prompt para o servidor LLM
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": nlu_prompt,
            "n_predict": 512,
            "temperature": 0.1,
            "stop": ["```"],
        }

        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status() # Levanta erro para status HTTP >= 400

        response_data = response.json()
        print("\n[DEBUG] Resposta RAW do servidor:")
        print(json.dumps(response_data, indent=2))
        
        # A resposta do /completion geralmente tem a string gerada em 'content'
        llm_output_str = response_data.get("content", "").strip()
        print("\n[DEBUG] Conteúdo extraído da resposta:")
        print("---")
        print(llm_output_str)
        print("---")

        # Tentar extrair o JSON do output do LLM
        try:
            # Remover os marcadores ```json e ``` se presentes
            if "```json" in llm_output_str:
                llm_output_str = llm_output_str.split("```json")[-1].split("```")[0].strip()
            elif "```" in llm_output_str:
                llm_output_str = llm_output_str.split("```")[1].strip()

            # Tentar encontrar o JSON válido na string
            json_start = llm_output_str.find("{")
            json_end = llm_output_str.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                llm_output_str = llm_output_str[json_start:json_end]

            print("\n[DEBUG] JSON extraído para parsing:")
            print("---")
            print(llm_output_str)
            print("---")

            parsed_json = json.loads(llm_output_str)
            # Adicionar o comando original para referência futura
            parsed_json["original_command"] = user_input
            return parsed_json

        except json.JSONDecodeError as e:
            print(f"\n[Erro NLU] Falha ao decodificar JSON da resposta do LLM:\n{llm_output_str}")
            print(f"Erro específico: {e}")
            # Tentar extrair intent e entities do texto mesmo com erro de parsing
            if '"intent":' in llm_output_str and '"entities":' in llm_output_str:
                # Se parece JSON válido mas com algum erro de formatação
                intent_match = re.search(r'"intent":\s*"([^"]+)"', llm_output_str)
                intent = intent_match.group(1) if intent_match else "unknown"
                return {
                    "intent": intent,
                    "entities": {},
                    "details": "Parsed from malformed JSON",
                    "original_command": user_input
                }
            return {"intent": "error_parsing", "entities": {}, "details": str(e), "original_command": user_input}

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP] Falha ao conectar com o servidor LLM: {e}")
        return {"intent": "error_connection", "entities": {}, "details": str(e), "original_command": user_input}
    except Exception as e:
        print(f"\n[Erro Inesperado] Ocorreu um erro: {e}")
        return {"intent": "error_unknown", "entities": {}, "details": str(e), "original_command": user_input} 