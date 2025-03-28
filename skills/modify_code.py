import requests
import re
import os
import traceback # Keep for debugging
import json
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

# <<< ASSINATURA CORRIGIDA >>>
def skill_modify_code(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """Modifica código existente com base na instrução e contexto do agente (Assinatura ReAct)."""
    print("\n[Skill: Modify Code (ReAct)]")
    print(f"  Action Input: {action_input}")

    # <<< USA action_input >>>
    modification = action_input.get("modification")
    # Use a more specific default if target_code_description is missing
    target_desc = action_input.get("target_code_description", "o código anterior")

    if not modification:
        return {"status": "error", "action": "modify_code_failed", "data": {"message": "Parâmetro 'modification' ausente no Action Input."}}

    original_code = None
    language = "python" # Default
    target_found_source = None # Para logar onde achou

    # --- BUSCAR CÓDIGO ALVO (VERSÃO 3 - Itera Histórico Corretamente) ---
    if agent_history:
        print("  Iniciando busca no histórico (reverso)...") # Log inicial da busca
        # Itera o histórico de trás para frente
        for entry_index, entry in enumerate(reversed(agent_history)):
            # Log para cada entrada sendo verificada
            # print(f"    Verificando histórico (idx rev {entry_index}): {entry[:70]}...") 
            if isinstance(entry, str) and entry.startswith("Observation:"):
                 print(f"    > Encontrada Observação (idx rev {entry_index}). Verificando conteúdo...") # Log Observação encontrada
                 read_success_match = re.search(r"Observação:\s*Conteúdo do arquivo\s*'([^']+)'\s*lido com sucesso", entry)
                 if read_success_match:
                     target_filename_read = read_success_match.group(1)
                     print(f"      >> MATCH! Observação de read_file encontrada para '{target_filename_read}'. Tentando reler...") # Log
                     try:
                         abs_path = os.path.abspath(target_filename_read)
                         cwd = os.getcwd()
                         # Segurança: Garante que estamos dentro do diretório de trabalho
                         if not abs_path.startswith(cwd):
                              print(f"      [Modify Code Safety WARN] Tentativa de reler arquivo fora do diretório atual: {target_filename_read}")
                         else:
                              with open(target_filename_read, "r", encoding="utf-8") as f_read:
                                   original_code = f_read.read()
                              if original_code:
                                   language = "python" # Default
                                   ext = os.path.splitext(target_filename_read)[1].lower()
                                   if ext in ['.js']: language = "javascript"
                                   # Atualiza target_desc para refletir a origem
                                   target_desc = f"o código do arquivo '{target_filename_read}' (lido na obs. anterior)"
                                   target_found_source = f"read_file ({target_filename_read})"
                                   print(f"      Código ({language}) obtido com sucesso re-lendo arquivo.") # Log
                                   break # <<< ENCONTROU VIA READ_FILE, SAI DO LOOP DO HISTÓRICO
                              else:
                                   print(f"      [Modify Code WARN] Arquivo '{target_filename_read}' relido, mas está vazio.")
                     except FileNotFoundError:
                          print(f"      [Modify Code ERROR] Arquivo '{target_filename_read}' não encontrado ao tentar reler.")
                          original_code = None # Resetar
                     except Exception as reread_err:
                         print(f"      [Modify Code ERROR] Erro inesperado ao tentar reler '{target_filename_read}': {reread_err}")
                         original_code = None # Resetar
                     # Mesmo se a releitura falhar, encontramos a observação de read_file, paramos de procurar
                     break # <<< SAI DO LOOP DO HISTÓRICO APÓS PROCESSAR OBS DE READ_FILE

                 # Se não era observação de read_file, verifica se era de generate/modify
                 elif "Código Gerado:" in entry or "Código Modificado:" in entry:
                     print("      > Observação de generate/modify encontrada. Tentando extrair código...") # Log
                     code_match = re.search(r"```(\w*)\s*([\s\S]*?)\s*```", entry, re.DOTALL)
                     if code_match:
                         lang_found = code_match.group(1).strip().lower()
                         code_to_use = code_match.group(2).strip()
                         if code_to_use:
                             original_code = code_to_use
                             language = lang_found if lang_found else language
                             target_desc = "o código da observação anterior (generate/modify)"
                             target_found_source = "Observation (gen/mod)"
                             print(f"      Código ({language}) obtido da observação anterior (generate/modify).")
                             break # <<< ENCONTROU VIA GEN/MOD, SAI DO LOOP DO HISTÓRICO
                         else:
                              print("      > Bloco de código vazio na observação gen/mod.") # Log
                     else:
                           print("      > Regex não encontrou bloco de código na observação gen/mod.") # Log
                 else:
                      print("      > Observação não continha read_file ou gen/mod code.") # Log

            # else: # Se não for string ou não começar com Observation:
            #      print(f"    > Ignorando entrada do histórico (não é string de Observação): {type(entry)}")

    # Se saiu do loop sem achar código no histórico...
    # 2. Tenta da memória do agente (agent_memory)
    if not original_code:
        print("  Código não encontrado no histórico. Verificando memória do agente...") # Log
        last_code_from_mem = agent_memory.get('last_code')
        if last_code_from_mem:
             original_code = last_code_from_mem
             language = agent_memory.get('last_lang') if agent_memory.get('last_lang') else language
             target_desc = "o último código na memória"
             target_found_source = "Agent Memory"
             print(f"  Código ({language}) encontrado na memória do agente.")
        else:
             print("  Nenhum código encontrado na memória do agente.")

    # --- FIM DA SEÇÃO BUSCAR CÓDIGO ALVO ---

    # Agora, a verificação final se original_code foi encontrado
    if not original_code:
        # Mensagem de erro atualizada
        # Usando print pois logger pode não estar configurado aqui
        print(f"  [Modify Code ERROR] Falha ao modificar: Não foi possível localizar o código alvo via Histórico (read/gen/mod) ou Memória, baseado na descrição '{target_desc}'.")
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {"message": f"Não foi possível localizar o código alvo ('{target_desc}') via histórico ou memória."}
        }
    else:
         print(f"  Código alvo para modificação encontrado (Fonte: {target_found_source}). Descrição usada: '{target_desc}'")

    # --- Construir Prompt de Modificação (Adaptado para Chat) ---
    # System prompt defines the role
    system_prompt_modify = f"Você é um editor de código {language} preciso. Modifique o código fornecido de acordo com a instrução do usuário. Retorne APENAS o bloco de código {language} modificado, sem nenhuma explicação adicional antes ou depois."
    # User prompt provides the context and instruction
    user_prompt_modify = f"""Aqui está o código {language} original:
```{language}
{original_code}
```

Modifique este código de acordo com a seguinte instrução:
{modification}"""

    print(f"  Construindo prompt de CHAT para modificação...")
    # print(f"DEBUG User Prompt Modificação:\n{user_prompt_modify}") # Optional debug

    # --- Chamar LLM para Modificar (USA API DE CHAT AGORA!) ---
    chat_url = LLAMA_SERVER_URL # Assume config.py has the correct chat URL
    if not chat_url.endswith("/chat/completions"):
         print(f"[Modify Code WARN] URL LLM '{chat_url}' não parece ser para chat. Verifique config.py. Tentando adicionar /v1/chat/completions...")
         if chat_url.endswith("/v1") or chat_url.endswith("/v1/"):
              chat_url = chat_url.rstrip('/') + "/chat/completions"
         else:
              chat_url = chat_url.rstrip('/') + "/v1/chat/completions" # Best guess append

    headers = {"Content-Type": "application/json"}
    messages_for_modification = [
         {"role": "system", "content": system_prompt_modify},
         {"role": "user", "content": user_prompt_modify}
    ]

    payload = {
        "messages": messages_for_modification,
        "temperature": 0.2, # Low for precise modification
        "max_tokens": 2048, # Allow ample space for modified code
         "stream": False
        # "stop": ["```"] # Can add if extraction fails consistently
    }

    try:
        print(f"  Enviando requisição de CHAT para modificar: {chat_url}")
        response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        generated_content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        print(f"  [DEBUG] Raw LLM Response Content (Modify Skill):\n---\n{generated_content}\n---")

        # --- Extrair Código Modificado (Mesma lógica de generate_code) ---
        modified_code = generated_content
        extracted_via_markdown = False
        # Use re.DOTALL to match newlines within the code block
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```", modified_code, re.DOTALL)
        if code_match:
            extracted_code = next((group for group in code_match.groups() if group is not None), None)
            if extracted_code is not None:
                 modified_code = extracted_code.strip()
                 extracted_via_markdown = True
                 print("[Modify Code INFO] Código modificado extraído de bloco Markdown.")

        if not extracted_via_markdown:
             print("[Modify Code WARN] Bloco Markdown não encontrado. Tentando limpeza de fallback.")
             # Basic cleaning, remove potential ```lang and ``` markers
             cleaned_code = re.sub(rf"^\s*```{language}\s*", "", modified_code)
             cleaned_code = re.sub(r"^\s*```\s*", "", cleaned_code)
             cleaned_code = re.sub(r"\s*```\s*$", "", cleaned_code).strip()
             if cleaned_code != modified_code:
                  print("[Modify Code INFO] Marcadores ``` removidos.")
                  modified_code = cleaned_code
             else:
                  print("[Modify Code WARN] Limpeza de fallback não alterou o código. Usando raw.")
                  modified_code = modified_code.strip() # Ensure no extra whitespace


        if not modified_code:
            print(" [Erro Modify] LLM retornou modificação vazia após extração.")
            # Return original as fallback to avoid breaking the flow, but use 'warning' status
            modified_code = original_code
            status = "warning"
            message = f"LLM não conseguiu modificar o código de {target_desc} (resposta vazia). Código original mantido."
        elif modified_code == original_code:
             print(" [Info Modify] LLM retornou o código original (modificação pode não ter sido aplicável ou necessária).")
             status = "success" # Still success, just no change
             message = f"Código de {target_desc} não foi alterado pela modificação solicitada."
        else:
             print(f"  Modificação Final (Extraída):\n---\n{modified_code}\n---")
             status = "success"
             message = f"Código de {target_desc} modificado com sucesso."

        # --- Retornar Resultado ---
        return {
            "status": status,
            "action": "code_modified", # Specific action name
            "data": {
                "original_code": original_code, # Useful for context/debug
                "modified_code": modified_code, # The resulting code
                "language": language,
                "message": message
            }
        }

    # --- Exception Handling (Similar to generate_code) ---
    except requests.exceptions.Timeout:
         print(f"\n[Erro Timeout na Skill Modify] LLM demorou muito para responder (>120s).")
         return {"status": "error", "action": "modify_code_failed", "data": {"message": "Timeout: O LLM demorou muito para modificar o código."}}
    except requests.exceptions.RequestException as e:
        error_details = str(e)
        if e.response is not None:
             error_details += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}..."
        print(f"\n[Erro HTTP na Skill Modify] Falha ao comunicar com o LLM: {error_details}")
        return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro de comunicação com LLM ao tentar modificar código: {error_details}"}}
    except json.JSONDecodeError as e:
         print(f"\n[Erro JSON na Skill Modify] Falha ao decodificar resposta do LLM: {e}")
         raw_resp_text = "N/A"
         if 'response' in locals() and hasattr(response, 'text'):
              raw_resp_text = response.text[:200]
         return {"status": "error", "action": "modify_code_failed", "data": {"message": f"Erro ao decodificar JSON da resposta do LLM: {e}. Resposta recebida (início): '{raw_resp_text}'"}}
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill Modify] {e}")
        traceback.print_exc() # Print traceback for unexpected errors
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {"message": f"Erro inesperado durante a modificação de código: {e}"}
        }