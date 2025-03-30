import re
import requests
import json
# --- Import BASE URL from config ---
from core.config import LLAMA_SERVER_URL as BASE_LLM_URL

# --- NEW Standard Signature ---
def skill_generate_code(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """Gera código com base no input da ação, usando a assinatura padrão e a API de Chat."""

    # --- Use action_input ---
    language = action_input.get("language", "python")
    filename = action_input.get("filename") # Optional filename suggestion
    purpose = action_input.get("purpose") # Mandatory purpose

    if not purpose: # Validate mandatory parameter
         return {"status": "error", "action": "generate_code_failed", "data": {"message": "Parâmetro 'purpose' é obrigatório para gerar código."}}

    print(f"\n[Skill: Generate Code (ReAct)]")
    print(f"  Action Input: {action_input}")

    # --- REMOVED OLD PROMPT BUILDING ---
    # code_prompt = f"// Tarefa: Gere código {language} para: {purpose}\n// Código {language}:\n"

    # --- NEW CHAT COMPLETIONS LLM CALL LOGIC ---
    print(f"  Construindo prompt de CHAT para geração de código...")

    # System prompt for the generation skill
    system_prompt = f"Você é um assistente de programação especialista em {language}. Sua tarefa é gerar código conciso e funcional."
    # User prompt with the specific purpose
    user_prompt = f"Gere APENAS o código {language} para a seguinte tarefa: {purpose}. Responda SOMENTE com o bloco de código (sem explicações antes ou depois)."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    headers = {"Content-Type": "application/json"}

    # Get the base URL from config and ensure it points to chat completions
    chat_url = BASE_LLM_URL
    if not chat_url.endswith("/chat/completions"):
         print(f"[Generate Code WARN] URL base '{BASE_LLM_URL}' não parece ser para chat. Verifique config.py. Tentando adicionar /v1/chat/completions...")
         # Try to fix if it's just the base /v1/ or root
         if chat_url.endswith("/v1") or chat_url.endswith("/v1/"):
             chat_url = chat_url.rstrip('/') + "/chat/completions"
         else:
             # If it's something else, append the full path (best guess)
             chat_url = chat_url.rstrip('/') + "/v1/chat/completions"

    payload = {
        "messages": messages,
        "temperature": 0.2, # Low temperature for code
        "max_tokens": 2048, # Increased limit for potentially long code
        # "stop": [], # Generally not needed for chat
        "stream": False
    }

    try:
        print(f"  Enviando requisição de CHAT para: {chat_url}")
        # print(f"  [DEBUG] Payload (Generate Skill): {json.dumps(payload, indent=2)}") # Optional debug
        response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()

        generated_content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        print(f"  [DEBUG] Raw LLM Response Content (Generate Skill):\n---\n{generated_content}\n---")

        # --- LÓGICA DE EXTRAÇÃO DE CÓDIGO (CORRIGIDA) ---
        code = generated_content
        extracted_via_markdown = False

        # 1. Tenta extrair de bloco Markdown primeiro (mais confiável)
        # (Regex ligeiramente ajustada para ser menos 'greedy' e pegar o primeiro bloco)
        # Use re.DOTALL to make '.' match newlines within the code block
        code_match = re.search(rf"```{language}\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```", code, re.DOTALL)
        if code_match:
            # Pega o conteúdo do primeiro grupo que não for None
            extracted_code = next((group for group in code_match.groups() if group is not None), None)
            if extracted_code is not None: # Garante que algo foi capturado
                 code = extracted_code.strip()
                 extracted_via_markdown = True
                 print("[Generate Code INFO] Código extraído de bloco Markdown.")

        # 2. Se NÃO extraiu via markdown, tenta limpar texto explicativo inicial (Fallback)
        if not extracted_via_markdown:
            print("[Generate Code WARN] Bloco de código markdown não encontrado. Tentando limpeza de fallback.")
            # Padrão simplificado para remover linhas comuns ANTES do código
            # Flag (?i) movida para o INÍCIO. Adicionado re.MULTILINE.
            # Remove linhas que começam com explicações comuns ou o próprio prompt ecoado.
            # Added more robust patterns, including potential language markers
            patterns_to_remove = [
                # Common explanations, prompt echoes, language markers (case-insensitive, multiline)
                r"(?im)^\s*(?:aqui está o código|o código é|código solicitado|claro(?:,|,) aqui está|você pode usar|gerando código|defina a função|função python|função javascript|código python|código javascript|// Tarefa:.*?|// Código python:|// Código javascript:)\s*\n?",
                # Remove potential ```python or ``` markers if not caught by markdown regex
                r"^\s*```(?:\w+)?\s*\n"
            ]
            cleaned_code = code
            for pattern in patterns_to_remove:
                 # Apply each pattern sequentially
                 cleaned_code = re.sub(pattern, "", cleaned_code, count=1) # count=1 to only remove the first occurrence

            # Remove também linhas finais que possam ser explicações pós-código ou closing ```
            # Using MULTILINE flag here as well
            cleaned_code = re.sub(r"(?m)\n\s*(?:#.*|\/\/.*|Explicação:.*|Nota:.*|```)\s*$", "", cleaned_code).strip()

            if cleaned_code != code: # Se algo foi removido
                 print("[Generate Code INFO] Texto explicativo inicial/final removido via regex.")
                 code = cleaned_code.strip()
            else:
                 print("[Generate Code WARN] Limpeza de fallback não removeu nada. Usando conteúdo raw.")
                 code = code.strip() # Garante que não há espaços extras

        # Verifica se o código resultante não está vazio
        if not code:
            print(" [Erro Generate] Código vazio após extração/limpeza.")
            # Include raw response in error message for debugging
            return {"status": "error", "action": "generate_code_failed", "data": {"message": f"LLM não retornou código válido ou código vazio após limpeza. Resposta recebida: '{generated_content}'"}}

        print(f"  Código Gerado (Final Extracted):\n---\n{code}\n---")
        # --- FIM DA LÓGICA DE EXTRAÇÃO ---


        # --- Retornar Resultado (Estrutura existente OK) ---
        return {
            "status": "success",
            "action": "code_generated",
            "data": {
                "code": code, # Usa o código extraído/limpo
                "language": language,
                "purpose": purpose, # Keep original purpose
                "filename_suggested": filename,
                "message": f"Código {language} gerado com sucesso para: {purpose}"
            }
        }

    # --- Keep Existing Exception Handling ---
    except requests.exceptions.Timeout:
         print(f"\n[Erro Timeout na Skill Generate] LLM demorou muito para responder (>120s).")
         return {"status": "error", "action": "generate_code_failed", "data": {"message": "Timeout: O LLM demorou muito para gerar o código."}}
    except requests.exceptions.RequestException as e:
        # Include more details from the exception if available
        error_details = str(e)
        if e.response is not None:
             error_details += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}..." # Show beginning of response
        print(f"\n[Erro HTTP na Skill Generate] Falha ao comunicar com o LLM: {error_details}")
        return {"status": "error", "action": "generate_code_failed", "data": {"message": f"Erro de comunicação com LLM ao tentar gerar código: {error_details}"}}
    except json.JSONDecodeError as e:
         print(f"\n[Erro JSON na Skill Generate] Falha ao decodificar resposta do LLM: {e}")
         # It might be helpful to know what the response was if it wasn't JSON
         raw_resp_text = "N/A"
         if 'response' in locals() and hasattr(response, 'text'):
              raw_resp_text = response.text[:200] # Get first 200 chars
         return {"status": "error", "action": "generate_code_failed", "data": {"message": f"Erro ao decodificar JSON da resposta do LLM: {e}. Resposta recebida (início): '{raw_resp_text}'"}}
    except Exception as e:
        # Use logger.exception to include traceback info automatically
        logger.exception(f"Erro inesperado na Skill Generate: {e}")
        return {
            "status": "error",
            "action": "generate_code_failed",
            "data": {"message": f"Erro inesperado durante a geração de código: {e}"}
        }

# --- Removed helper functions not used anymore ---
# def extract_code_from_response(response): ...
# def send_to_llm(prompt): ... 