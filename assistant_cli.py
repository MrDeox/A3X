import requests
import json
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente (pode ser útil para futuras configs)
load_dotenv()

# URL do servidor llama.cpp (padrão)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"

def create_nlu_prompt(user_input: str) -> str:
    """Cria o prompt para extrair intenção e entidades."""
    # Prompt aprimorado para instruir o LLM a retornar APENAS JSON
    # (Podemos refinar isso mais tarde com exemplos few-shot se necessário)
    prompt = f"""Você é um assistente de IA especializado em analisar comandos de usuário e extrair a intenção principal e as entidades relevantes.
Sua tarefa é processar o comando fornecido e responder **APENAS** com um objeto JSON válido contendo duas chaves: "intent" (uma string descrevendo a ação principal em formato snake_case, ex: 'generate_code', 'search_web', 'manage_files', 'remember_info') e "entities" (um objeto contendo os parâmetros chave-valor extraídos do comando). Seja conciso e preciso. Se não tiver certeza, use um valor 'unknown' ou omita a entidade.

Comando do Usuário:
"{user_input}"

JSON Resultante:
```json
""" # Instruir a usar ```json pode ajudar alguns modelos
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
        "stop": ["```", "\n\n", "Comando do Usuário:"], # Parar antes de gerar texto extra
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
def skill_generate_code(entities: dict, original_command: str):
    print("\n[Skill: Generate Code]")
    print(f"  Recebido pedido para gerar código com entidades: {entities}")
    # Lógica futura para gerar código aqui...
    return f"Platzhalter: Código com base em '{original_command}' seria gerado aqui."

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