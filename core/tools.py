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
    from skills.memory import skill_save_memory, skill_recall_memory
    # Adicionaremos outras skills aqui depois
except ImportError as e:
    print(f"[Tools ERROR] Falha ao importar skills: {e}. Verifique os caminhos e nomes dos arquivos.")
    # Define placeholders se a importação falhar para evitar erros na inicialização
    def skill_search_web(*args, **kwargs): return {"status": "error", "data": {"message": "skill_search_web não carregada"}}
    def skill_manage_files(*args, **kwargs): return {"status": "error", "data": {"message": "skill_manage_files não carregada"}}
    def skill_generate_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_generate_code não carregada"}}
    def skill_execute_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_execute_code não carregada"}}
    def skill_modify_code(*args, **kwargs): return {"status": "error", "data": {"message": "skill_modify_code não carregada"}}
    def skill_final_answer(*args, **kwargs): return {"status": "error", "data": {"message": "skill_final_answer não carregada"}}
    def skill_save_memory(*args, **kwargs): return {"status": "error", "data": {"message": "skill_save_memory não carregada"}}
    def skill_recall_memory(*args, **kwargs): return {"status": "error", "data": {"message": "skill_recall_memory não carregada"}}

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
        "description": "Lista arquivos e diretórios em um caminho específico. Use '.' para o diretório atual.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada (DEVE ser 'list')",
                    "enum": ["list"]
                },
                "directory": {
                    "type": "string",
                    "description": "O subdiretório opcional para listar (relativo ao diretório de trabalho). Deixe vazio ou omita para listar o diretório atual."
                }
            },
            "required": ["action"] # Only action is required for list
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
    # --- File Management Tools (Using the single refactored skill) ---
    "read_file": {
        "function": skill_manage_files,
        "description": "Lê e retorna TODO o conteúdo de um arquivo de texto especificado.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada. DEVE ser EXATAMENTE a string 'read'.",
                    "enum": ["read"]
                },
                "file_name": {
                    "type": "string",
                    "description": "O nome do arquivo a ser lido (obrigatório)."
                }
            },
            "required": ["action", "file_name"]
        }
    },
    "create_file": {
        "function": skill_manage_files,
        "description": "Cria um NOVO arquivo de texto com o nome e conteúdo EXATAMENTE como especificados. Se o arquivo já existir, seu conteúdo será COMPLETAMENTE SUBSTITUÍDO.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada (DEVE ser 'create')",
                    "enum": ["create"]
                },
                "file_name": {
                    "type": "string",
                    "description": "O nome completo (incluindo extensão, ex: 'notas.txt', 'config.json') do arquivo a ser criado/sobrescrito (obrigatório)."
                },
                "content": {
                    "type": "string",
                    "description": "O conteúdo de texto EXATO a ser escrito no arquivo (obrigatório)."
                }
            },
            "required": ["action", "file_name", "content"]
        }
    },
    "append_to_file": {
        "function": skill_manage_files,
        "description": "Adiciona (anexa) conteúdo de texto AO FINAL de um arquivo JÁ EXISTENTE. Se o arquivo não existir, esta ferramenta falhará.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada (DEVE ser 'append')",
                    "enum": ["append"]
                },
                "file_name": {
                    "type": "string",
                    "description": "O nome do arquivo EXISTENTE ao qual adicionar conteúdo (obrigatório)."
                },
                "content": {
                    "type": "string",
                    "description": "O conteúdo de texto EXATO a ser adicionado ao final do arquivo (obrigatório)."
                }
            },
            "required": ["action", "file_name", "content"]
        }
    },
    "delete_file": {
        "function": skill_manage_files,
        "description": "Exclui um arquivo especificado do diretório de trabalho. Use com cuidado!",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "A ação a ser realizada (DEVE ser 'delete')",
                    "enum": ["delete"]
                },
                "file_name": {
                    "type": "string",
                    "description": "O nome do arquivo a ser excluído (obrigatório)."
                }
            },
            "required": ["action", "file_name"]
        }
    },
    # --- End File Management Tools ---
    "save_memory": {
        "function": skill_save_memory,
        "description": "Use esta ferramenta para armazenar informações textuais importantes na memória de longo prazo quando o usuário pedir explicitamente (ex: 'Lembre-se que...', 'Anote aí:', 'Guarde esta informação:', 'Salve isso:') ou quando você identificar um fato crucial de uma ferramenta (ex: search_web) que deve ser lembrado. NÃO use para o último código gerado (isso é automático).",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "O texto da informação a ser armazenada (obrigatório)."
                },
                "metadata": {
                    "type": "object",
                    "description": "Metadados opcionais em formato JSON (ex: {'source': 'user_input', 'topic': 'configuração'}).",
                    "properties": {
                         "source": {"type": "string"},
                         "topic": {"type": "string"},
                    },
                    "additionalProperties": True
                }
            },
            "required": ["content"]
        }
    },
    "recall_memory": {
        "function": skill_recall_memory,
        "description": "Busca na memória de longo prazo por informações semanticamente similares a uma consulta fornecida. Útil para recuperar fatos, contextos ou tarefas passadas relevantes. Use consultas concisas e focadas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A pergunta ou consulta textual para buscar informações similares na memória.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de resultados relevantes a serem retornados.",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    "final_answer": {
        "function": skill_final_answer,
        "description": "Fornece a resposta final ao usuário após completar a tarefa ou quando não há mais passos a seguir.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A resposta final e completa para a solicitação original do usuário."
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