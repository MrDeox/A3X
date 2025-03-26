import requests
import json
import os
import re
from dotenv import load_dotenv

# Carregar variáveis de ambiente (pode ser útil para futuras configs)
load_dotenv()

# URL do servidor llama.cpp (padrão)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"

def create_nlu_prompt(user_input: str) -> str:
    """Cria o prompt para extrair intenção e entidades, agora com exemplos few-shot."""
    prompt = f"""Você é um assistente de IA que analisa comandos e extrai intenção e entidades em formato JSON. Responda APENAS com o JSON.

### Exemplos

Comando: "cria um arquivo python chamado utils.py com uma função hello world"
JSON Resultante:
```json
{{
  "intent": "generate_code",
  "entities": {{
    "language": "python",
    "action": "create",
    "construct_type": "file",
    "file_name": "utils.py",
    "contains": [
      {{
        "construct_type": "function",
        "function_name": "hello_world",
        "purpose": "print hello world"
      }}
    ]
  }}
}}


Comando: "liste os arquivos .txt no diretório atual"
JSON Resultante:

{{
  "intent": "manage_files",
  "entities": {{
    "action": "list",
    "file_extension": ".txt",
    "directory": "current"
  }}
}}

Comando: "qual a previsão do tempo para amanhã em São Paulo?"
JSON Resultante:

{{
  "intent": "search_info",
  "entities": {{
    "topic": "previsão do tempo",
    "timeframe": "amanhã",
    "location": "São Paulo"
  }}
}}

Comando: "lembre-me de ligar para a Maria às 16h"
JSON Resultante:

{{
  "intent": "set_reminder",
  "entities": {{
    "action": "ligar",
    "recipient": "Maria",
    "time": "16h"
  }}
}}

Comando Atual

Comando: "{user_input}"
JSON Resultante:

"""
    return prompt

def interpret_command(user_input: str) -> dict:
    """Envia o comando para o servidor LLM e retorna a interpretação."""
    nlu_prompt = create_nlu_prompt(user_input)

    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": nlu_prompt,
        "n_predict": 256,  # Máximo de tokens para a resposta JSON
        "temperature": 0.1, # Baixa temperatura para JSON mais consistente
        "top_p": 0.9,
        # "grammar": "" # Futuramente podemos adicionar gramática GBNF para forçar JSON
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status() # Levanta erro para status HTTP >= 400

        response_data = response.json()
        # A resposta do /completion geralmente tem a string gerada em 'content'
        llm_output_str = response_data.get("content", "").strip()

        # Tentar extrair o JSON do output do LLM
        try:
            # Remover os marcadores ```json e ``` se presentes
            if llm_output_str.startswith("```json"):
                llm_output_str = llm_output_str[len("```json"):].strip()
            if llm_output_str.endswith("```"):
                llm_output_str = llm_output_str[:-len("```")].strip()

            parsed_json = json.loads(llm_output_str)
            # Adicionar o comando original para referência futura
            parsed_json["original_command"] = user_input
            return parsed_json

        except json.JSONDecodeError:
            print(f"\n[Erro NLU] Falha ao decodificar JSON da resposta do LLM:\n{llm_output_str}")
            return {"intent": "error_parsing", "entities": {}, "details": "Failed to parse JSON from LLM response", "raw_response": llm_output_str, "original_command": user_input}

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP] Falha ao conectar com o servidor LLM: {e}")
        return {"intent": "error_connection", "entities": {}, "details": str(e), "original_command": user_input}
    except Exception as e:
        print(f"\n[Erro Inesperado] Ocorreu um erro: {e}")
        return {"intent": "error_unknown", "entities": {}, "details": str(e), "original_command": user_input}

# Skills (funções placeholder)
def skill_generate_code(entities: dict, original_command: str) -> str:
    """Gera código usando o LLM com base nas entidades extraídas."""
    print("\n[Skill: Generate Code]")
    print(f"  Entidades recebidas: {entities}")

    # Extrair detalhes das entidades (com valores padrão)
    language = entities.get("language", "python") # Assume Python como padrão
    file_name = entities.get("file_name", None)
    action = entities.get("action", "generate")
    construct_type = entities.get("construct_type", "code snippet")
    contains = entities.get("contains", [])
    purpose = entities.get("purpose", original_command) # Usa comando original se propósito específico não foi extraído

    # Prompt de código EXTREMAMENTE simples para teste, com newline no final
    code_prompt = "Escreva o código Python completo para uma função chamada 'somar' que aceita dois argumentos e retorna sua soma. NÃO inclua nenhuma explicação, apenas o código.\n"

    print(f"  Prompt de geração de código enviado ao LLM:\n{code_prompt}") # Para depuração

    # Enviar o prompt de código para o servidor LLM
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": code_prompt, # Mantém o prompt simples sobre a função 'somar'
        "n_predict": 1024,  # Aumentado para garantir espaço
        "temperature": 0.4, # Temperatura aumentada para mais criatividade
        "top_p": 0.9,
        # "stop" removido para deixar o modelo decidir quando parar
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        print(f"  Resposta RAW do servidor LLM: {response_data}") # DEBUG RAW RESPONSE
        generated_code = response_data.get("content", "").strip()

        # Extrair estritamente o conteúdo do primeiro bloco de código ```language ... ```
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```", generated_code)
        extracted_code = ""
        if code_match:
            extracted_code = code_match.group(1).strip()
        else:
            # Se não encontrar bloco de código, assume que a resposta inteira pode ser o código
            # (ou pelo menos a tentativa) - pode precisar de mais refinamento
            print("  [Aviso] Bloco de código não encontrado na resposta do LLM. Usando resposta completa.")
            extracted_code = generated_code # Usa a resposta completa como fallback

        # Mensagem de resultado (com a lógica de salvar comentada como antes)
        result_message = f"Código {language} gerado:\n---\n{extracted_code}\n---"
        if file_name:
            try:
                with open(file_name, "w") as f:
                    f.write(extracted_code) # Salva apenas o código extraído
                result_message += f"\nCódigo também salvo em '{file_name}'."
            except Exception as e:
                 result_message += f"\nErro ao tentar salvar o arquivo '{file_name}': {e}"

        return result_message

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP na Skill] Falha ao conectar com o servidor LLM: {e}")
        return f"Erro ao gerar código: Falha na conexão com LLM ({e})"
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill] Ocorreu um erro: {e}")
        return f"Erro inesperado ao gerar código: {e}"

def skill_manage_files(entities: dict, original_command: str):
    print("\n[Skill: Manage Files]")
    print(f"  Recebido pedido para gerenciar arquivos com entidades: {entities}")
    # Lógica futura para listar/criar/mover arquivos...
    if "action" in entities and entities["action"] == "list":
         try:
             files = os.listdir('.') # Lista arquivos no diretório atual
             return f"Arquivos no diretório atual: {', '.join(files)}"
         except Exception as e:
             return f"Erro ao listar arquivos: {e}"
    return f"Platzhalter: Ação de arquivo com base em '{original_command}' seria realizada aqui."

def skill_search_web(entities: dict, original_command: str):
    print("\n[Skill: Search Web]")
    print(f"  Recebido pedido para buscar na web com entidades: {entities}")
    return f"Platzhalter: Busca web com base em '{original_command}' seria realizada aqui."

def skill_remember_info(entities: dict, original_command: str):
    print("\n[Skill: Remember Info]")
    print(f"  Recebido pedido para lembrar informação com entidades: {entities}")
    return f"Platzhalter: Armazenamento de informação com base em '{original_command}' seria realizado aqui."

def skill_unknown(entities: dict, original_command: str):
    print("\n[Skill: Unknown]")
    print(f"  Não sei como lidar com a intenção (ou foi um erro) para: '{original_command}'")
    return f"Desculpe, não entendi ou não posso realizar a ação: '{original_command}'"

# Dispatcher (mapeia intenções para funções de skill)
SKILL_DISPATCHER = {
    "generate_code": skill_generate_code,
    "manage_files": skill_manage_files,
    "list_files": skill_manage_files,  # Mapear variações da intenção para a mesma skill
    "search_web": skill_search_web,
    "remember_info": skill_remember_info,
    "error_parsing": skill_unknown,
    "error_connection": skill_unknown,
    "error_unknown": skill_unknown,
    "unknown": skill_unknown  # Intenção padrão se o LLM não tiver certeza
}

# Loop principal da CLI
if __name__ == "__main__":
    print("Assistente Pessoal (conectado ao servidor llama.cpp)")
    print("Digite 'sair' para terminar.")
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break
            if not user_input:
                continue

            interpretation = interpret_command(user_input)
            # Opcional: imprimir para depuração
            # print("\n[Interpretação NLU]:")
            # print(json.dumps(interpretation, indent=2, ensure_ascii=False))

            intent = interpretation.get("intent", "unknown")
            entities = interpretation.get("entities", {})
            original_command = interpretation.get("original_command", user_input)  # Pega o original

            # Encontra a função da skill no dispatcher, ou usa unknown como padrão
            skill_function = SKILL_DISPATCHER.get(intent, skill_unknown)

            # Chama a skill e obtém o resultado
            result = skill_function(entities, original_command)

            print("\n[Assistente]:")
            print(result)

        except KeyboardInterrupt:
            print("\nSaindo...")
            break

    print("\nAssistente encerrado.") 