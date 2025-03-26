import requests
import re
import os
from core.config import LLAMA_SERVER_URL

def skill_modify_code(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Modifica código existente com base no comando e histórico."""
    print("\n[Skill: Modify Code]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    target = entities.get("target")
    modification = entities.get("modification")
    file_name = entities.get("file_name") # Pode vir da NLU

    if not modification:
        return {"status": "error", "action": "modify_code_failed", "data": {"message": "Não entendi qual modificação fazer."}}

    # --- Lógica para encontrar o código original ---
    original_code = None
    target_description = "" # Para o prompt NLG
    language = "python" # Assumimos Python por padrão, pode ser expandido depois

    if file_name and os.path.exists(file_name):
        try:
            print(f"  Tentando ler código do arquivo: {file_name}")
            with open(file_name, "r") as f:
                original_code = f.read()
            target_description = f"o arquivo '{file_name}'"
            # Tenta inferir a linguagem pela extensão
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
             # Continuar tentando buscar no histórico

    if not original_code and history:
        print("  Tentando encontrar código no histórico recente...")
        # Procura de trás para frente no histórico
        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            # Verifica se a entrada é do assistente e se o skill_result está lá
            if entry["role"] == "assistant" and "skill_result" in entry:
                prev_skill_result = entry["skill_result"]
                # Verifica se a ação anterior foi uma geração de código bem-sucedida
                if prev_skill_result.get("status") == "success" and prev_skill_result.get("action") == "code_generated":
                    # Pega o código dos dados do resultado anterior
                    original_code = prev_skill_result.get("data", {}).get("code")
                    if original_code:
                         target_description = "o código anterior"
                         print(f"  Código encontrado no histórico!")
                         # Tenta obter a linguagem do resultado anterior também
                         language = prev_skill_result.get("data", {}).get("language", language)
                         break # Para no primeiro código encontrado

    if not original_code:
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Não consegui encontrar o código alvo ('{target}') mencionado."}}

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

    print(f"  Prompt de modificação enviado ao LLM:\n---\n{modification_prompt}\n---") # Para depuração

    # --- Chamar LLM para Modificar ---
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": modification_prompt,
        "n_predict": 2048,  # Mais espaço, modificações podem aumentar o código
        "temperature": 0.3, # Manter baixa para seguir instruções de modificação
        "stop": ["```"],    # Parar após o bloco de código
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        print(f"  Resposta RAW do servidor LLM (Modify): {response_data}") # DEBUG
        generated_modification = response_data.get("content", "").strip()

        # Extrair código modificado (usando a mesma lógica robusta de generate_code)
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```", generated_modification, re.DOTALL)
        modified_code = ""
        if code_match:
            modified_code = code_match.group(1).strip()
        else:
            print("  [Aviso] Bloco de código modificado não encontrado na resposta. Usando fallback.")
            modified_code = generated_modification.removeprefix(f"```{language}").removesuffix("```").strip()

        if not modified_code:
             raise ValueError("LLM não retornou código modificável.")

        # --- Fim da Chamada e Extração ---

        # --- Salvar Modificação (Opcional, se file_name existe) ---
        result_message = f"Código de {target_description} modificado:\n---\n{modified_code}\n---"
        if file_name:
             try:
                 print(f"  Atualizando arquivo: {file_name}")
                 with open(file_name, "w") as f:
                     f.write(modified_code)
                 result_message += f"\nArquivo '{file_name}' atualizado."
             except Exception as e:
                  result_message += f"\nErro ao tentar atualizar o arquivo '{file_name}': {e}"
        # --- Fim do Salvamento ---

        # Retornar resultado com o código modificado REAL
        return {
            "status": "success",
            "action": "code_modified",
            "data": {
                "target": target_description,
                "modification": modification,
                "original_code": original_code, # Mantém para referência
                "modified_code": modified_code, # O código REAL modificado
                "file_name": file_name,       # Nome do arquivo, se aplicável
                "message": result_message     # Mensagem para NLG/log
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"\n[Erro HTTP na Skill Modify] Falha ao conectar com o servidor LLM: {e}")
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro ao modificar código: Falha na conexão com LLM ({e})"}}
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill Modify] Ocorreu um erro: {e}")
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro inesperado ao modificar código: {e}"}} 