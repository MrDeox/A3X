# Importa as funções de skill
from skills.generate_code import skill_generate_code
from skills.manage_files import skill_manage_files
from skills.search_web import skill_search_web
from skills.remember_info import skill_remember_info
from skills.recall_info import skill_recall_info
from skills.modify_code import skill_modify_code
from skills.execute_code import skill_execute_code
from skills.unknown import skill_unknown

# Dispatcher agora usa as funções importadas
SKILL_DISPATCHER = {
    "generate_code": skill_generate_code,
    "manage_files": skill_manage_files,
    "search_web": skill_search_web,
    "remember_info": skill_remember_info,
    "recall_info": skill_recall_info,
    "modify_code": skill_modify_code,
    "execute_code": skill_execute_code,
    # Mapeamentos de erro e fallback
    "error_parsing": skill_unknown,
    "error_connection": skill_unknown,
    "error_unknown": skill_unknown,
    "unknown": skill_unknown
}

# Poderíamos adicionar uma função get_skill(intent) aqui para encapsular a lógica .get()
def get_skill(intent: str):
    """Retorna a função da skill correspondente à intenção."""
    return SKILL_DISPATCHER.get(intent, skill_unknown) 