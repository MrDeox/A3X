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