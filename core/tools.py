# core/tools.py
import json
import os
import sys

# Ajuste para importar skills do diretório pai
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Tenta importar as skills - pode precisar de ajustes dependendo da estrutura final
try:
    from skills.web_search import skill_search_web
    from skills.manage_files import skill_manage_files
    from skills.generate_code import skill_generate_code
    from skills.execute_code import skill_execute_code
    from skills.modify_code import skill_modify_code
    from skills.final_answer import skill_final_answer
    # Adicionaremos outras skills aqui depois
except ImportError as e:
    print(f"[Tools ERROR] Falha ao importar skills: {e}. Verifique os caminhos e nomes dos arquivos.")
    # Define placeholders se a importação falhar para evitar erros na inicialização
    def skill_search_web(*args, **kwargs): return {"status": "error", "data": {"message": "skill_search_web não carregada"}}
    def skill_manage_files(*args, **kwargs): return {"status": "error", "data": {"message": "skill_manage_files não carregada"}}
    def skill_generate_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_generate_code não carregada"}}
    def skill_execute_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_execute_code não carregada"}}
    def skill_modify_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_modify_code não carregada"}}


# Definição inicial das ferramentas (Formato pode evoluir)
# Usaremos um dicionário onde a chave é o nome da ferramenta
TOOLS = {
    "search_web": {
        "function": skill_search_web,
        "description": "Realiza uma busca na web usando DuckDuckGo para encontrar informações sobre um tópico ou responder a uma pergunta factual.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A string de busca detalhada ou a pergunta a ser feita na web (obrigatório)."
                }
            },
            "required": ["query"]
        }
    },
    "list_files": {
        "function": skill_manage_files,
        "description": "Lista arquivos em um diretório com base em uma extensão ou padrão. NÃO use para criar, deletar ou modificar arquivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada (DEVE ser 'list')",
                    "enum": ["list"]
                },
                "file_extension": {
                    "type": "string",
                    "description": "A extensão dos arquivos a serem listados (ex: '.py', '.txt') ou um padrão glob (ex: '*.log'). (obrigatório para listagem)"
                },
            },
            "required": ["action", "file_extension"]
        }
    },
    "generate_code": {
        "function": skill_generate_code,
        "description": "Gera código em uma linguagem de programação específica (padrão: python) com base em uma descrição do que o código deve fazer. Pode opcionalmente receber um nome de arquivo sugerido.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "A linguagem de programação para gerar o código (ex: python, javascript). Default: python."
                },
                "purpose": {
                    "type": "string",
                    "description": "Uma descrição clara e detalhada do que o código deve fazer (obrigatório)."
                },
                "filename": {
                    "type": "string",
                    "description": "Um nome de arquivo sugerido onde o código poderia ser salvo (opcional)."
                }
            },
            "required": ["purpose"]
        }
    },
    "execute_code": {
        "function": skill_execute_code,
        "description": "Executa um bloco de código Python (gerado anteriormente ou fornecido) em um ambiente seguro (sandbox). Use esta ferramenta APENAS DEPOIS que o código foi gerado por 'generate_code' ou modificado por 'modify_code' em um passo anterior.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_description": {
                     "type": "string",
                     "description": "Descrição do código a ser executado. DEVE ser 'o código do passo anterior' ou similar para indicar que deve buscar no histórico recente do agente. Não passe código diretamente aqui."
                 }
                # A skill original tentava pegar 'file_name', mas com ReAct focaremos no histórico do agente primeiro.
            },
            "required": ["target_description"] # Força o LLM a pensar sobre o alvo
        }
    },
    "modify_code": {
        "function": skill_modify_code,
        "description": "Modifica um bloco de código existente (normalmente o último gerado ou o da observação anterior) com base em uma instrução específica. Use para adicionar linhas, remover linhas, refatorar, corrigir erros, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "modification": {
                    "type": "string",
                    "description": "Instrução CLARA e específica sobre como o código deve ser modificado (obrigatório)."
                },
                "target_code_description": {
                    "type": "string",
                    "description": "Descrição de qual código modificar. Use 'o código da observação anterior' ou 'o último código gerado' para usar o código da memória do agente."
                }
                # Poderíamos adicionar 'language' se quiséssemos suportar outras linguagens
            },
            "required": ["modification", "target_code_description"]
        }
    },
    # --- Adicionaremos create_file, append_to_file, delete_file, execute_code, etc., aqui depois ---
    "final_answer": {
         "function": skill_final_answer,
         "description": "Fornece a resposta final e completa para a solicitação original do usuário, APÓS todas as ferramentas necessárias terem sido usadas e as informações coletadas.",
         "parameters": {
             "type": "object",
             "properties": {
                 "answer": {
                     "type": "string",
                     "description": "A resposta final e concisa para apresentar ao usuário."
                 }
             },
             "required": ["answer"]
         }
    }
}

def get_tool(tool_name: str) -> dict | None:
    """Busca uma ferramenta pelo nome."""
    return TOOLS.get(tool_name)

def get_tool_descriptions() -> str:
    """Retorna uma string formatada com as descrições de todas as ferramentas para o prompt do LLM."""
    descriptions = []
    tool_names = sorted([name for name in TOOLS if name != "final_answer"])
    
    for name in tool_names:
        tool_info = TOOLS[name]
        tool_info_no_func = tool_info.copy()
        if "function" in tool_info_no_func:
             del tool_info_no_func["function"]

        # Formatar como JSON ou outra string estruturada
        # Usar indentação para legibilidade no prompt
        try:
            desc_json = json.dumps(tool_info_no_func, indent=2, ensure_ascii=False)
            descriptions.append(f"Tool Name: {name}\nTool Description:\n{desc_json}")
        except TypeError as e:
             print(f"[Tools WARN] Não foi possível serializar a descrição da ferramenta '{name}': {e}")
             descriptions.append(f"Tool Name: {name}\nTool Description: Error serializing description.")

    return "\n\n".join(descriptions)

# Exemplo de como obter a string formatada (para referência)
# TOOL_DESCRIPTIONS_FOR_PROMPT = get_tool_descriptions()
# print(TOOL_DESCRIPTIONS_FOR_PROMPT) 