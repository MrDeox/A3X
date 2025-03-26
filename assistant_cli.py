import requests
import json
import os
from dotenv import load_dotenv
import re

# Carregar variáveis de ambiente (pode ser útil para futuras configs)
load_dotenv()

# URL do servidor llama.cpp (padrão)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"

def create_nlu_prompt(command: str, history: list) -> str:
    """
    Cria um prompt básico para o NLU, incluindo histórico recente e exemplos essenciais.
    """
    prompt = "Analise o **Comando Atual** do usuário considerando o histórico e responda APENAS com JSON contendo \"intent\" e \"entities\".\n\n"
    
    # Adiciona histórico recente se houver
    if history:
        prompt += "### Histórico Recente da Conversa:\n"
        for entry in history[-3:]:  # Últimos 3 pares de interação
            prompt += f"{entry}\n"
        prompt += "\n"
    
    prompt += "### Exemplos Essenciais\n\n"
    
    # Exemplo de geração de código
    prompt += 'Comando: "gere um script python chamado utils.py com uma função hello world"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "generate_code",
  "entities": {
    "language": "python",
    "construct_type": "function",
    "purpose": "hello world"
  }
}
```\n\n'''
    
    # Exemplo de gerenciamento de arquivos
    prompt += 'Comando: "crie um arquivo vazio teste.txt"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "manage_files",
  "entities": {
    "action": "create",
    "file_name": "teste.txt",
    "content": null
  }
}
```\n\n'''
    
    # Exemplo de listagem de arquivos
    prompt += 'Comando: "liste os arquivos .py"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "manage_files",
  "entities": {
    "action": "list",
    "file_extension": ".py"
  }
}
```\n\n'''
    
    # Exemplo de previsão do tempo
    prompt += 'Comando: "qual a previsão do tempo para amanhã em Curitiba?"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "weather_forecast",
  "entities": {
    "topic": "previsão do tempo",
    "timeframe": "amanhã",
    "location": "Curitiba"
  }
}
```\n\n'''
    
    # Adiciona o comando atual
    prompt += "### Comando Atual\n\n"
    prompt += f'Comando: "{command}"\n'
    prompt += "JSON Resultante:\n```json\n"
    
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

# Skills (funções placeholder)
def skill_generate_code(entities: dict, original_command: str, intent: str = None) -> dict:
    """Gera código usando o LLM com base nas entidades extraídas."""
    print("\n[Skill: Generate Code]")
    print(f"  Entidades recebidas: {entities}")

    # Extrair detalhes das entidades (com valores padrão)
    language = entities.get("language", "python")
    file_name = entities.get("file_name", None)
    action = entities.get("action", "generate") # Pode vir da NLU
    construct_type = entities.get("construct_type", "code") # Tipo de código (script, function, class, snippet)
    contains = entities.get("contains", []) # Lista de elementos internos (funções, classes)
    purpose = entities.get("purpose", None) # Propósito geral extraído pela NLU
    if not purpose and not contains: # Se NLU não deu propósito e não há 'contains'
         purpose = original_command # Usa comando original como último recurso

    # --- Construir o prompt de geração de código DINAMICAMENTE ---
    code_prompt = f"Gere o código {language} completo para um {construct_type}"
    if file_name:
        code_prompt += f" que será salvo como '{file_name}'"
    code_prompt += ".\n\n"

    # Descrever o conteúdo desejado
    if contains:
        code_prompt += "O código deve incluir o(s) seguinte(s) elemento(s):\n"
        for item in contains:
            item_type = item.get('construct_type', 'elemento')
            item_name = item.get('function_name') or item.get('class_name', '')
            item_purpose = item.get('purpose', '')
            code_prompt += f"- Um(a) {item_type}"
            if item_name:
                 code_prompt += f" chamado(a) '{item_name}'"
            if item_purpose:
                 code_prompt += f" que faz o seguinte: {item_purpose}\n"
            else:
                 code_prompt += "\n"
    elif purpose:
        # Usa o propósito geral se não houver 'contains' detalhado
         code_prompt += f"O propósito principal do código é: {purpose}\n"
    else:
         # Caso muito genérico, talvez pedir mais detalhes? Por enquanto, deixar simples.
         code_prompt += "Gere um exemplo de código simples.\n"

    # Instrução final modificada e adicionando linha inicial do bloco
    code_prompt += f"\nIMPORTANTE: Gere APENAS o código {language} solicitado, sem explicações.\n``` {language}\n"
    # --- Fim da construção dinâmica ---

    print(f"  Prompt de geração de código enviado ao LLM:\n---\n{code_prompt}\n---") # Para depuração

    # Enviar o prompt de código para o servidor LLM
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": code_prompt, # USA A VARIÁVEL CONSTRUÍDA ACIMA
        "n_predict": 1536, # Aumentado
        "temperature": 0.5, # Aumentado
        "stop": ["```"], # RE-ADICIONADO
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        print(f"  Resposta RAW do servidor LLM: {response_data}") # DEBUG RAW RESPONSE
        generated_code = response_data.get("content", "").strip()

        # Extrair estritamente o conteúdo do primeiro bloco de código ```language ... ```
        # (Usando re.DOTALL para '.' corresponder a newlines também)
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```", generated_code, re.DOTALL)
        extracted_code = ""
        if code_match:
            extracted_code = code_match.group(1).strip()
        else:
            # Fallback: Se não encontrar bloco, usa a resposta como estava,
            # mas limpa possíveis inícios/fins de bloco simples que o stop pode ter deixado
            print("  [Aviso] Bloco de código ```language ... ``` não encontrado na resposta. Usando fallback.")
            extracted_code = generated_code.removeprefix(f"```{language}").removesuffix("```").strip()

        # Mensagem de resultado
        result_message = f"Código {language} gerado:\n---\n{extracted_code}\n---"
        if file_name:
            try:
                # Lógica de salvamento (agora descomentada e funcional)
                with open(file_name, "w") as f:
                    f.write(extracted_code)
                result_message += f"\nCódigo também salvo em '{file_name}'."
            except Exception as e:
                 result_message += f"\nErro ao tentar salvar o arquivo '{file_name}': {e}"

        return {
            "status": "success",
            "action": "code_generated",
            "data": {
                "language": language,
                "code": extracted_code,
                "file_name": file_name, # Será None se não foi salvo
                "message": result_message
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP na Skill] Falha ao conectar com o servidor LLM: {e}")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"Erro ao gerar código: Falha na conexão com LLM ({e})"}
        }
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill] Ocorreu um erro: {e}")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"Erro inesperado ao gerar código: {e}"}
        }

def skill_manage_files(entities: dict, original_command: str) -> dict:
    """Gerencia arquivos (listar, criar, deletar) no diretório atual."""
    print("\n[Skill: Manage Files]")
    print(f"  Entidades recebidas: {entities}")

    action = entities.get("action", None)
    file_name = entities.get("file_name", None)
    directory = entities.get("directory", ".") # Padrão para diretório atual
    content = entities.get("content", None) # Conteúdo para criar arquivo
    file_extension = entities.get("file_extension", None) # Filtro para listar

    # --- Ação: Listar Arquivos ---
    if action == "list":
        try:
            # Garante que estamos operando no diretório pretendido (segurança básica)
            # Por enquanto, só permitimos '.' (diretório atual)
            if directory != ".":
                 return {
                     "status": "error",
                     "action": "list_files_failed",
                     "data": {"message": "Desculpe, por segurança, só posso listar arquivos no diretório atual por enquanto."}
                 }

            files = os.listdir(directory)
            result_files = []
            if file_extension:
                # Garante que a extensão comece com '.'
                if not file_extension.startswith('.'):
                    file_extension = '.' + file_extension
                # Filtra os arquivos pela extensão
                result_files = [f for f in files if os.path.isfile(os.path.join(directory, f)) and f.endswith(file_extension)]
                return {
                    "status": "success",
                    "action": "files_listed",
                    "data": {
                        "directory": directory,
                        "files": result_files,
                        "filter": file_extension,
                        "message": f"Arquivos '{file_extension}' no diretório atual: {', '.join(result_files) if result_files else 'Nenhum encontrado'}"
                    }
                }
            else:
                # Lista todos os arquivos e diretórios
                 result_files = files
                 return {
                     "status": "success",
                     "action": "files_listed",
                     "data": {
                         "directory": directory,
                         "files": result_files,
                         "filter": None,
                         "message": f"Conteúdo do diretório atual: {', '.join(result_files) if result_files else 'Vazio'}"
                     }
                 }

        except FileNotFoundError:
            return {
                "status": "error",
                "action": "list_files_failed",
                "data": {"message": f"Erro: O diretório '{directory}' não foi encontrado."}
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "list_files_failed",
                "data": {"message": f"Erro ao listar arquivos em '{directory}': {e}"}
            }

    # --- Ação: Criar Arquivo ---
    elif action == "create":
        if not file_name:
            return {
                "status": "error",
                "action": "create_file_failed",
                "data": {"message": "Erro: Para criar um arquivo, preciso de um nome (file_name)."}
            }
        try:
            # Medida de segurança simples: evitar caminhos absolutos ou que saiam do dir atual
            if os.path.isabs(file_name) or ".." in file_name:
                 return {
                     "status": "error",
                     "action": "create_file_failed",
                     "data": {"message": "Desculpe, por segurança, só posso criar arquivos diretamente no diretório atual."}
                 }

            if os.path.exists(file_name):
                return {
                    "status": "error",
                    "action": "create_file_failed",
                    "data": {"message": f"Erro: O arquivo '{file_name}' já existe."}
                }

            with open(file_name, "w") as f:
                if content:
                    f.write(content)
                else:
                    f.write("") # Cria arquivo vazio
            return {
                "status": "success",
                "action": "file_created",
                "data": {
                    "file_name": file_name,
                    "content": content,
                    "message": f"Arquivo '{file_name}' criado com sucesso."
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "action": "create_file_failed",
                "data": {"message": f"Erro ao criar o arquivo '{file_name}': {e}"}
            }

    # --- Ação: Deletar Arquivo ---
    elif action == "delete":
        if not file_name:
            return {
                "status": "error",
                "action": "delete_file_failed",
                "data": {"message": "Erro: Para deletar um arquivo, preciso de um nome (file_name)."}
            }
        try:
             # Medida de segurança simples
            if os.path.isabs(file_name) or ".." in file_name:
                 return {
                     "status": "error",
                     "action": "delete_file_failed",
                     "data": {"message": "Desculpe, por segurança, só posso deletar arquivos diretamente no diretório atual."}
                 }

            if not os.path.exists(file_name):
                 return {
                     "status": "error",
                     "action": "delete_file_failed",
                     "data": {"message": f"Erro: O arquivo '{file_name}' não existe."}
                 }
            if not os.path.isfile(file_name):
                 return {
                     "status": "error",
                     "action": "delete_file_failed",
                     "data": {"message": f"Erro: '{file_name}' não é um arquivo."}
                 }

            # !! Ação Destrutiva !! Adicionar confirmação seria ideal no futuro
            os.remove(file_name)
            return {
                "status": "success",
                "action": "file_deleted",
                "data": {
                    "file_name": file_name,
                    "message": f"Arquivo '{file_name}' deletado com sucesso."
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "action": "delete_file_failed",
                "data": {"message": f"Erro ao deletar o arquivo '{file_name}': {e}"}
            }

    # --- Ação Desconhecida ---
    else:
        # Se a intenção foi 'manage_files' mas a ação específica não foi reconhecida/implementada
        if entities.get("intent") == "manage_files": # Usando entities.get para evitar KeyError
             return {
                 "status": "error",
                 "action": "unknown_action",
                 "data": {"message": f"Não sei como realizar a ação específica '{action}' solicitada em '{original_command}'. Ações suportadas: list, create, delete."}
             }
        # Se a intenção não foi 'manage_files' (fallback do dispatcher)
        return {
            "status": "error",
            "action": "unknown_action",
            "data": {"message": f"Platzhalter: Ação de arquivo não reconhecida para '{original_command}'."}
        }

def skill_search_web(entities: dict, original_command: str):
    print("\n[Skill: Search Web]")
    print(f"  Recebido pedido para buscar na web com entidades: {entities}")
    return f"Platzhalter: Busca web com base em '{original_command}' seria realizada aqui."

def skill_remember_info(entities: dict, original_command: str):
    print("\n[Skill: Remember Info]")
    print(f"  Recebido pedido para lembrar informação com entidades: {entities}")
    return f"Platzhalter: Armazenamento de informação com base em '{original_command}' seria realizado aqui."

def skill_unknown(entities: dict, original_command: str, intent: str = None) -> dict:
    """Skill padrão para comandos não reconhecidos."""
    print("\n[Skill: Unknown]")
    print(f"  Não sei como lidar com a intenção (ou foi um erro) para: '{original_command}'")
    return {
        "status": "not_understood",
        "action": "unknown_intent",
        "data": {
            "message": f"Desculpe, não entendi ou não posso realizar a ação: '{original_command}'",
            "original_command": original_command,
            "intent": intent
        }
    }

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

# Inicializar histórico de conversa
conversation_history = []
MAX_HISTORY_TURNS = 5 # Quantos pares (usuário + assistente) lembrar

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

            # Interpretar o comando do usuário
            interpretation = interpret_command(user_input, conversation_history)
            # Opcional: imprimir para depuração
            # print("\n[Interpretação NLU]:")
            # print(json.dumps(interpretation, indent=2))

            # Extrair intent e entities
            intent = interpretation.get("intent", "unknown")
            entities = interpretation.get("entities", {})
            original_command = interpretation.get("original_command", user_input)

            # Chama a skill apropriada
            if intent == "generate_code":
                skill_result = skill_generate_code(entities, original_command, intent=intent)
            elif intent == "manage_files":
                skill_result = skill_manage_files(entities, original_command)
            else:
                skill_result = skill_unknown(entities, original_command, intent=intent)

            # --- Processamento do Resultado (Temporário) ---
            print("\n[Resultado da Skill (Estruturado)]:")
            print(json.dumps(skill_result, indent=2, ensure_ascii=False))

            # Cria uma resposta simples para o usuário e para o histórico (TEMPORÁRIO)
            if skill_result.get("status") == "success":
                final_response_text = f"Ação '{skill_result.get('action', 'desconhecida')}' concluída."
                # Poderia ser mais descritivo baseado na action/data
                if skill_result.get("action") == "file_created":
                     final_response_text = f"Arquivo '{skill_result['data'].get('file_name')}' criado."
                elif skill_result.get("action") == "files_listed":
                     final_response_text = f"Listei {len(skill_result['data'].get('files', []))} itens."
                elif skill_result.get("action") == "code_generated":
                     final_response_text = f"Gerei o código {skill_result['data'].get('language', '')}."
                     if skill_result['data'].get('file_name'):
                          final_response_text += f" Salvei como '{skill_result['data']['file_name']}'."

            elif skill_result.get("status") == "not_understood":
                final_response_text = skill_result["data"].get("message", "Não entendi.")
            else: # status == "error"
                 final_response_text = skill_result["data"].get("message", "Ocorreu um erro.")

            print("\n[Assistente (Resposta Temporária)]:")
            print(final_response_text)
            # --- Fim do Processamento Temporário ---

            # --- Adicionar ao Histórico ---
            conversation_history.append({"role": "user", "content": user_input})
            # Adiciona a resposta temporária ao histórico
            conversation_history.append({"role": "assistant", "content": final_response_text})
            # --- Fim da Adição ao Histórico ---

        except KeyboardInterrupt:
            print("\nSaindo...")
            break

    print("\nAssistente encerrado.") 