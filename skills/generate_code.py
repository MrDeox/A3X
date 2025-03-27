import re
import requests
import os
from core.config import LLAMA_SERVER_URL

def skill_generate_code(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
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

    # --- Construir o prompt ---
    if construct_type == 'print' and purpose:
        code_prompt = f"Gere uma única linha de código python que imprime a string: '{purpose}'."
    elif language and construct_type and purpose:
        code_prompt = f"Gere o código {language} completo para um {construct_type} que faz o seguinte: {purpose}."
        if file_name:
            code_prompt += f" O código será salvo como '{file_name}'."
        if contains:
            code_prompt += "\nO código deve incluir o(s) seguinte(s) elemento(s):\n"
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
    else:
        # Fallback se faltar informação
        code_prompt = f"Gere um exemplo simples de código {language}."

    # Instrução final
    code_prompt += f"\nIMPORTANTE: Gere APENAS o código {language} solicitado, sem explicações.\n``` {language}\n"

    print(f"  Prompt de geração de código enviado ao LLM:\n---\n{code_prompt}\n---") # Para depuração

    # Enviar o prompt de código para o servidor LLM
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": code_prompt,
        "n_predict": 1536,
        "temperature": 0.2, # Temperatura mais baixa para geração mais focada
        "stop": ["```"],
    }
    
    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        print(f"  Resposta RAW do servidor LLM: {response_data}") # DEBUG RAW RESPONSE
        generated_code = response_data.get("content", "").strip()

        # Extrair estritamente o conteúdo do primeiro bloco de código ```language ... ```
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```", generated_code, re.DOTALL)
        extracted_code = ""
        if code_match:
            extracted_code = code_match.group(1).strip()
        else:
            # Fallback: Se não encontrar bloco, usa a resposta como estava
            print("  [Aviso] Bloco de código ```language ... ``` não encontrado na resposta. Usando fallback.")
            extracted_code = generated_code.removeprefix(f"```{language}").removesuffix("```").strip()

        # Mensagem de resultado
        result_message = f"Código {language} gerado:\n---\n{extracted_code}\n---"
        if file_name:
            try:
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
                "file_name": file_name,
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

def extract_code_from_response(response):
    # Implemente a lógica para extrair o código da resposta do LLM
    # Esta é uma implementação básica e pode ser ajustada de acordo com a estrutura da resposta do LLM
    return response.json().get("content", "").strip()

def send_to_llm(prompt):
    # Implemente a lógica para enviar o prompt para o LLM
    # Esta é uma implementação básica e pode ser ajustada de acordo com a forma como você se comunica com o LLM
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "n_predict": 1536,
        "temperature": 0.5,
        "stop": ["```"],
    }
    
    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP na Skill] Falha ao conectar com o servidor LLM: {e}")
        raise
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill] Ocorreu um erro: {e}")
        raise 