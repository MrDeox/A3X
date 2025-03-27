import requests
import re
import os
from core.config import LLAMA_SERVER_URL

def _find_code_in_history_or_file(file_name: str = None, history: list = None) -> tuple:
    """
    Procura o código alvo em um arquivo ou no histórico.
    
    Args:
        file_name: Nome do arquivo para procurar o código
        history: Lista de entradas do histórico
        
    Returns:
        tuple: (código_original, descrição_alvo, linguagem)
    """
    original_code = None
    target_description = ""
    language = "python"  # Padrão

    # Tenta encontrar no arquivo primeiro
    if file_name and os.path.exists(file_name):
        try:
            print(f"  Tentando ler código do arquivo: {file_name}")
            with open(file_name, "r") as f:
                original_code = f.read()
            target_description = f"o arquivo '{file_name}'"
            
            # Infere a linguagem pela extensão
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ['.js', '.jsx', '.ts', '.tsx']:
                language = "javascript"
            elif ext in ['.java']:
                language = "java"
            elif ext in ['.cpp', '.hpp', '.cc', '.hh']:
                language = "cpp"
            elif ext in ['.c', '.h']:
                language = "c"
            elif ext in ['.go']:
                language = "go"
            elif ext in ['.rs']:
                language = "rust"
            elif ext in ['.py']:
                language = "python"
        except Exception as e:
            print(f"  Erro ao ler arquivo {file_name}: {e}")

    # Se não encontrou no arquivo, tenta no histórico
    if not original_code and history:
        print("  Tentando encontrar código gerado ou modificado no histórico recente...")
        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            if entry["role"] == "assistant" and "skill_result" in entry:
                prev_skill_result = entry["skill_result"]
                action = prev_skill_result.get("action")
                
                # Verifica se é uma ação de código válida
                if prev_skill_result.get("status") == "success" and action in ["code_generated", "code_modified"]:
                    # Extrai o código baseado no tipo de ação
                    if action == "code_generated":
                        original_code = prev_skill_result.get("data", {}).get("code")
                        target_description = "o código gerado anteriormente"
                    else:  # code_modified
                        original_code = prev_skill_result.get("data", {}).get("modified_code")
                        target_description = "o código modificado anteriormente"
                    
                    if original_code:
                        language = prev_skill_result.get("data", {}).get("language", language)
                        print(f"  Código ({language}) encontrado no histórico!")
                        break

    return original_code, target_description, language

def skill_modify_code(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Modifica código existente com base no comando e histórico."""
    print("\n[Skill: Modify Code]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}")
    else:
        print("  Nenhum histórico fornecido")

    target = entities.get("target")
    modification = entities.get("modification")
    file_name = entities.get("file_name")

    if not modification:
        return {"status": "error", "action": "modify_code_failed", "data": {"message": "Não entendi qual modificação fazer."}}

    # Encontra o código alvo
    original_code, target_description, language = _find_code_in_history_or_file(file_name, history)

    if not original_code:
        return {
            "status": "error",
            "action": "code_modification_failed",
            "data": {"message": f"Não foi possível localizar o código alvo ({target_description})."}
        }

    # --- Construir o Prompt de Modificação ---
    modification_prompt = f"""Você é um assistente especializado em modificar código existente.
Aqui está o código original:

```{language}
{original_code}
```

Modifique o código de acordo com a seguinte instrução:
{modification}

Retorne APENAS o código modificado, sem explicações adicionais, dentro de um bloco de código com a linguagem especificada.
O código modificado deve manter a mesma funcionalidade base, apenas aplicando a modificação solicitada.
Se a modificação não fizer sentido ou não puder ser aplicada, retorne o código original sem alterações.

Código modificado:
```{language}
"""

    print(f"  Prompt de modificação enviado ao LLM:\n---\n{modification_prompt}\n---")

    # --- Chamar LLM para Modificar ---
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": modification_prompt,
        "n_predict": 2048,
        "temperature": 0.3,
        "stop": ["```"],
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        print(f"  Resposta RAW do servidor LLM (Modify): {response_data}")
        generated_modification = response_data.get("content", "").strip()

        # Extrair o código modificado da resposta
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```", generated_modification, re.DOTALL)
        if code_match:
            modified_code = code_match.group(1).strip()
        else:
            print(f" [Aviso] Bloco de código ```{language} ... ``` não encontrado. Assumindo resposta direta.")
            modified_code = generated_modification.strip()

        if not modified_code:
            print(" [Erro] Código modificado vazio. Mantendo código original.")
            return {
                "status": "error",
                "message": "Código modificado vazio",
                "data": {"original_code": original_code}
            }

        # --- Salvar Modificação ---
        result_message = f"Código de {target_description} modificado:\n---\n{modified_code}\n---"
        if file_name:
             try:
                 print(f"  Atualizando arquivo: {file_name}")
                 with open(file_name, "w") as f:
                     f.write(modified_code)
                 result_message += f"\nArquivo '{file_name}' atualizado."
             except Exception as e:
                  result_message += f"\nErro ao tentar atualizar o arquivo '{file_name}': {e}"

        return {
            "status": "success",
            "action": "code_modified",
            "data": {
                "target": target_description,
                "modification": modification,
                "original_code": original_code,
                "modified_code": modified_code,
                "file_name": file_name,
                "message": result_message
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP na Skill Modify] Falha ao conectar com o servidor LLM: {e}")
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro ao modificar código: Falha na conexão com LLM ({e})"}}
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill Modify] Ocorreu um erro: {e}")
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro inesperado ao modificar código: {e}"}} 