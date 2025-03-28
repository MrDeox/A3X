import requests
import re
import os
import traceback # Keep for debugging
import json
import logging # <<< Add import
from core.config import LLAMA_SERVER_URL

# <<< Initialize logger >>>
logger = logging.getLogger(__name__)

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
    target_filepath = None # <<< Initialize target_filepath >>>
    language = "python" # Default
    target_found_source = None

    # <<< INÍCIO: Verificação de Overrides >>>
    # >>>>> INÍCIO: NOVA Verificação Prioritária (para compatibilidade com override do agent.py) <<<<<
    code_from_agent_override = action_input.get("code_to_modify")
    filepath_from_agent_override = action_input.get("target_filepath")

    use_override = False # Mover para cá
    if code_from_agent_override is not None and filepath_from_agent_override is not None:
        logger.info("  Código e filepath encontrados via \'code_to_modify\' e \'target_filepath\' (override do agente). Usando-os.")
        original_code = code_from_agent_override
        target_filepath = filepath_from_agent_override # <<< Use target_filepath >>>
        # Infer language from override filepath
        ext = os.path.splitext(target_filepath)[1].lower()
        if ext in ['.js']: language = "javascript"
        # Add more language inferences if needed
        target_found_source = "Agent Override (code_to_modify)"
        use_override = True
    # >>>>> FIM: NOVA Verificação Prioritária <<<<<
    # <<< Bloco de override original mantido como fallback (se necessário, pode ser removido) >>>
    elif not use_override: # Só checa este se o override do agente não foi usado
        original_code_override = action_input.get("original_code_override")
        target_filepath_override = action_input.get("target_filepath_override")

        # use_override = False # Comentado/Removido
        if original_code_override is not None and target_filepath_override is not None:
            logger.info("  Overrides de código original e filepath encontrados no Action Input (original_code_override). Usando-os diretamente.") # Log ajustado
            original_code = original_code_override
            target_filepath = target_filepath_override # <<< Use target_filepath >>>
            # Infer language from override filepath
            ext = os.path.splitext(target_filepath)[1].lower()
            if ext in ['.js']: language = "javascript"
            # Add more language inferences if needed
            target_found_source = "Action Input Override (original_code_override)" # Log ajustado
            use_override = True
    # <<< FIM: Verificação de Overrides >>>

    if not use_override: # Só busca se não usou override
        logger.debug("  Overrides não encontrados. Iniciando busca por código alvo...") # <<< Updated log >>>
        # --- BUSCAR CÓDIGO ALVO (VERSÃO 4 - Prioriza 'target_desc' para arquivo) ---

        # 1. Tenta extrair e ler o arquivo diretamente do target_desc
        logger.debug(f"  Verificando se '{target_desc}' contém um caminho de arquivo...") # <<< Use logger >>>
        # Regex simples para encontrar algo que pareça um caminho de arquivo (pode precisar de ajuste)
        path_match = re.search(r"[\\'\\\"]([a-zA-Z0-9_\\-\\./]+\\.(?:py|txt|js|json|md|log))[\\'\\\"]", target_desc)
        if path_match:
            target_filename_from_desc = path_match.group(1)
            logger.info(f"  Caminho potencial encontrado em target_desc: '{target_filename_from_desc}'. Tentando ler...") # <<< Use logger >>>
            try:
                abs_path = os.path.abspath(target_filename_from_desc)
                cwd = os.getcwd()
                # Segurança: Garante que estamos dentro do diretório de trabalho
                if not abs_path.startswith(cwd):
                     logger.warning(f"      [Modify Code Safety WARN] Tentativa de ler arquivo (de target_desc) fora do diretório atual: {target_filename_from_desc}") # <<< Use logger >>>
                else:
                     with open(target_filename_from_desc, "r", encoding="utf-8") as f_read:
                          original_code = f_read.read()
                     if original_code:
                          target_filepath = target_filename_from_desc # <<< Set target_filepath >>>
                          language = "python" # Default
                          ext = os.path.splitext(target_filename_from_desc)[1].lower()
                          if ext in ['.js']: language = "javascript"
                          # Atualiza target_desc para refletir a origem
                          # target_desc = f"o código do arquivo '{target_filename_from_desc}' (lido diretamente via target_desc)" # Opcional: Atualizar descrição
                          target_found_source = f"File ({target_filename_from_desc} from target_desc)"
                          logger.info(f"      Código ({language}) obtido com sucesso lendo arquivo do target_desc.") # <<< Use logger >>>
                     else:
                          logger.warning(f"      [Modify Code WARN] Arquivo '{target_filename_from_desc}' (de target_desc) lido, mas está vazio.") # <<< Use logger >>>
            except FileNotFoundError:
                 logger.error(f"      [Modify Code ERROR] Arquivo '{target_filename_from_desc}' (de target_desc) não encontrado.") # <<< Use logger >>>
                 original_code = None # Resetar para cair no fallback
            except Exception as direct_read_err:
                logger.exception(f"      [Modify Code ERROR] Erro inesperado ao ler '{target_filename_from_desc}' (de target_desc): {direct_read_err}") # <<< Use logger >>>
                original_code = None # Resetar para cair no fallback

        # 2. Se não achou pelo target_desc, tenta histórico (lógica anterior)
        if not original_code and agent_history:
            logger.debug("  Código não encontrado diretamente via target_desc. Iniciando busca no histórico (reverso)...") # <<< Use logger >>>
            # Itera o histórico de trás para frente
            for entry_index, entry in enumerate(reversed(agent_history)):
                # Log para cada entrada sendo verificada
                # logger.debug(f"    Verificando histórico (idx rev {entry_index}): {entry[:70]}...")
                if isinstance(entry, str) and entry.startswith("Observation:"):
                     logger.debug(f"    > Encontrada Observação (idx rev {entry_index}). Verificando conteúdo...") # <<< Use logger >>>
                     # Regex ajustada para buscar filename E preview do conteúdo lido (melhor correspondência)
                     read_success_match = re.search(r"Observação:\s*Conteúdo do arquivo\s*'([^']+)'\s*lido com sucesso", entry) # Usa regex anterior por simplicidade
                     if read_success_match:
                         target_filename_read = read_success_match.group(1)
                         logger.info(f"      >> MATCH! Observação de read_file encontrada para '{target_filename_read}'. Tentando reler...") # <<< Use logger >>>
                         try:
                             abs_path = os.path.abspath(target_filename_read)
                             cwd = os.getcwd()
                             # Segurança: Garante que estamos dentro do diretório de trabalho
                             if not abs_path.startswith(cwd):
                                  logger.warning(f"      [Modify Code Safety WARN] Tentativa de reler arquivo fora do diretório atual: {target_filename_read}") # <<< Use logger >>>
                             else:
                                  with open(target_filename_read, "r", encoding="utf-8") as f_read:
                                       original_code = f_read.read()
                                  if original_code:
                                       target_filepath = target_filename_read # <<< Set target_filepath >>>
                                       language = "python" # Default
                                       ext = os.path.splitext(target_filename_read)[1].lower()
                                       if ext in ['.js']: language = "javascript"
                                       # Atualiza target_desc para refletir a origem
                                       target_desc = f"o código do arquivo '{target_filename_read}' (lido na obs. anterior)"
                                       target_found_source = f"read_file ({target_filename_read})"
                                       logger.info(f"      Código ({language}) obtido com sucesso re-lendo arquivo.") # <<< Use logger >>>
                                       break # <<< ENCONTROU VIA READ_FILE, SAI DO LOOP DO HISTÓRICO
                                  else:
                                       logger.warning(f"      [Modify Code WARN] Arquivo '{target_filename_read}' relido, mas está vazio.") # <<< Use logger >>>
                         except FileNotFoundError:
                              logger.error(f"      [Modify Code ERROR] Arquivo '{target_filename_read}' não encontrado ao tentar reler.") # <<< Use logger >>>
                              original_code = None # Resetar
                         except Exception as reread_err:
                             logger.exception(f"      [Modify Code ERROR] Erro inesperado ao tentar reler '{target_filename_read}': {reread_err}") # <<< Use logger >>>
                             original_code = None # Resetar
                         # Mesmo se a releitura falhar, encontramos a observação de read_file, paramos de procurar
                         break # <<< SAI DO LOOP DO HISTÓRICO APÓS PROCESSAR OBS DE READ_FILE

                     # Se não era observação de read_file, verifica se era de generate/modify
                     elif "Código Gerado:" in entry or "Código Modificado:" in entry:
                         logger.debug("      > Observação de generate/modify encontrada. Tentando extrair código...") # <<< Use logger >>>
                         code_match = re.search(r"```(\\w*)\\s*([\\s\\S]*?)\\s*```", entry, re.DOTALL)
                         if code_match: # Indented correctly under elif
                             lang_found = code_match.group(1).strip().lower()
                             code_to_use = code_match.group(2).strip()
                             if code_to_use: # Indented correctly under if code_match
                                 original_code = code_to_use
                                 # <<< Tenta inferir filepath da mensagem de sucesso da skill anterior >>>
                                 prev_skill_output_match = re.search(r"salvo no arquivo: ([^\s]+)", entry, re.IGNORECASE) # Indentation fixed
                                 if prev_skill_output_match: # Indented correctly under if code_to_use
                                     target_filepath = prev_skill_output_match.group(1) # <<< Set target_filepath >>>
                                     logger.info(f"      Filepath '{target_filepath}' inferido da observação de gen/mod.")
                                 else: # Indented correctly under if prev_skill_output_match
                                     logger.warning("      Não foi possível inferir filepath da observação de gen/mod.")
                                     target_filepath = None # Garante que está None

                                 language = lang_found if lang_found else language # Indentation fixed
                                 target_desc = "o código da observação anterior (generate/modify)" # Indentation fixed
                                 target_found_source = "Observation (gen/mod)" # Indentation fixed
                                 logger.info(f"      Código ({language}) obtido da observação anterior (generate/modify).") # Indentation fixed
                                 break # <<< ENCONTROU VIA GEN/MOD, SAI DO LOOP DO HISTÓRICO
                             else: # Indented correctly under if code_to_use
                                 logger.warning("      > Bloco de código vazio na observação gen/mod.") # Indentation fixed
                         else: # Indented correctly under if code_match
                             logger.warning("      > Regex não encontrou bloco de código na observação gen/mod.") # Indentation fixed
                     else:
                          logger.debug("      > Observação não continha read_file ou gen/mod code.") # <<< Use logger >>>

                # else: # Se não for string ou não começar com Observation:
                #      logger.debug(f"    > Ignorando entrada do histórico (não é string de Observação): {type(entry)}")

        # Se saiu do loop sem achar código no histórico...
        # 3. Tenta da memória do agente (agent_memory)
    if not original_code: # Alinhado corretamente
        logger.debug("  Código não encontrado via target_desc ou histórico. Verificando memória do agente...") # <<< Use logger >>>
        last_code_from_mem = agent_memory.get('last_code') # Indentado sob o if
        if last_code_from_mem: # Indentado sob o if
             original_code = last_code_from_mem
             language = agent_memory.get('last_lang') if agent_memory.get('last_lang') else language # Indentado sob o if
             target_desc = "o último código na memória" # Indentado sob o if
             target_found_source = "Agent Memory" # Indentado sob o if
             target_filepath = None # Código da memória não tem filepath associado aqui # Indentado sob o if
             logger.info(f"  Código ({language}) encontrado na memória do agente.") # <<< Use logger >>> # Indentado sob o if
        else: # Indentado sob o if
             logger.warning("  Nenhum código encontrado na memória do agente.") # <<< Use logger >>> # Indentado sob o if

    # <<< FIM DO BLOCO if not use_override >>> -> Este comentário será removido

    # Agora, a verificação final se original_code foi encontrado (independente da origem)
    if not original_code:
        # Mensagem de erro atualizada
        logger.error(f"  [Modify Code ERROR] Falha ao modificar: Não foi possível localizar o código alvo via Override, Histórico (read/gen/mod) ou Memória, baseado na descrição '{target_desc}'.") # <<< Use logger >>>
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {"message": f"Não foi possível localizar o código alvo ('{target_desc}') via override, histórico ou memória."} # <<< Updated error message >>>
        }
    else:
         logger.info(f"  Código alvo para modificação encontrado (Fonte: {target_found_source}). Descrição usada: '{target_desc}'") # <<< Use logger >>>

    # --- Construir Prompt de Modificação (Adaptado para Chat) ---
    # System prompt defines the role
    system_prompt_modify = "Você é um editor de código Python extremamente preciso. Sua única tarefa é modificar o código fornecido na mensagem do usuário para aplicar a 'Instrução de Modificação Específica'. Retorne APENAS o código Python completo e modificado, sem NENHUM outro texto antes ou depois."
    # User prompt provides the context and instruction
    user_prompt_modify = f"""Instrução de Modificação Específica:
{modification}
(Aplique esta mudança diretamente ao código original. A instrução provavelmente pede para alterar ou remover uma linha específica.)

Código Original ({target_filepath if target_filepath else 'desconhecido'}):
```python
{original_code}
```

Código Modificado (APENAS O CÓDIGO):"""

    logger.debug(f"  Construindo prompt de CHAT para modificação...")
    # logger.debug(f"DEBUG System Prompt Modificação:\n{system_prompt_modify}") # Optional debug
    # logger.debug(f"DEBUG User Prompt Modificação:\n{user_prompt_modify}") # Optional debug

    # --- Chamar LLM para Modificar (USA API DE CHAT AGORA!) ---
    chat_url = LLAMA_SERVER_URL # Assume config.py has the correct chat URL
    if not chat_url.endswith("/chat/completions"):
         logger.warning(f"[Modify Code WARN] URL LLM '{chat_url}' não parece ser para chat. Verifique config.py. Tentando adicionar /v1/chat/completions...") # <<< Use logger >>>
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
        logger.debug(f"  [DEBUG] Raw LLM Response Content (Modify Skill):\\n---\\n{generated_content}\\n---") # <<< Use logger >>>

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
                 logger.info("[Modify Code INFO] Código modificado extraído de bloco Markdown.") # <<< Use logger >>>

        if not extracted_via_markdown:
             logger.warning("[Modify Code WARN] Bloco Markdown não encontrado. Tentando limpeza de fallback.") # <<< Use logger >>>
             # Basic cleaning, remove potential ```lang and ``` markers
             modified_code = re.sub(rf"^```{language}\s*", "", modified_code)
             modified_code = re.sub(r"\s*```$", "", modified_code)

        # --- Comparar Código ---
        if modified_code == original_code:
             message = f"Código de {target_desc} não foi alterado pela modificação solicitada."
             logger.info(f" [Info Modify] LLM retornou o código original (modificação pode não ter sido aplicável ou necessária).") # <<< Use logger >>>
        else:
             message = f"Código de {target_desc} modificado com sucesso."
             logger.info(f" [Info Modify] Modificação aplicada. Código alterado.") # <<< Use logger >>>

             # --- SALVAR NO ARQUIVO (SE target_filepath FOI DEFINIDO) ---
             if target_filepath:
                  # Segurança extra: verifica se o path é relativo e dentro do CWD
                  abs_target_path = os.path.abspath(target_filepath)
                  cwd = os.getcwd()
                  if not abs_target_path.startswith(cwd):
                       logger.error(f"  [Modify Code SAVE ERROR] Tentativa de salvar fora do diretório de trabalho bloqueada: {target_filepath}")
                       message += f" (ERRO: Falha ao salvar - acesso fora do diretório negado para '{target_filepath}')"
                       # Retorna sucesso na modificação, mas falha no salvamento
                       return {
                           "status": "partial_success", # Indica que modificou, mas não salvou
                           "action": "code_modified_not_saved",
                           "data": {
                               "original_code": original_code,
                               "modified_code": modified_code,
                               "language": language,
                               "target_filepath": target_filepath,
                               "message": message
                           }
                      }
                  else:
                      try:
                           # Garante que diretórios existam
                           os.makedirs(os.path.dirname(abs_target_path), exist_ok=True)
                           with open(abs_target_path, "w", encoding="utf-8") as f_save:
                                f_save.write(modified_code)
                           logger.info(f"  Código modificado salvo em: {target_filepath}")
                           message += f" E salvo em '{target_filepath}'."
                      except Exception as save_err:
                           logger.exception(f"  [Modify Code SAVE ERROR] Erro ao salvar arquivo '{target_filepath}':")
                           message += f" (ERRO: Falha ao salvar em '{target_filepath}': {save_err})"
                           # Retorna sucesso na modificação, mas falha no salvamento
                           return {
                               "status": "partial_success",
                               "action": "code_modified_not_saved",
                               "data": {
                                   "original_code": original_code,
                                   "modified_code": modified_code,
                                   "language": language,
                                   "target_filepath": target_filepath,
                                   "message": message
                               }
                          }
             else:
                  logger.warning("  Nenhum target_filepath definido (código veio da memória?). Código modificado não será salvo em arquivo.")
                  message += " (AVISO: Código não salvo em arquivo pois o alvo não era um arquivo específico)."

        # --- Atualizar Memória (Sempre atualiza com o último código modificado) ---
        agent_memory['last_code'] = modified_code
        agent_memory['last_lang'] = language
        logger.info("  Memória do agente atualizada com o código modificado.")

        return {
            "status": "success", # ou partial_success se salvamento falhou antes
            "action": "code_modified",
            "data": {
                "original_code": original_code,
                "modified_code": modified_code,
                "language": language,
                "target_filepath": target_filepath, # <<< Inclui o filepath no resultado >>>
                "message": message
            }
        }

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