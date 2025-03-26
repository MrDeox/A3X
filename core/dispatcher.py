# Importa as funções de skill
from skills.generate_code import skill_generate_code
from skills.manage_files import skill_manage_files
from skills.web_search import skill_search_web
from skills.memory import skill_remember_info, skill_recall_info
from skills.modify_code import skill_modify_code
from skills.unknown import skill_unknown

# Dispatcher agora usa as funções importadas
SKILL_DISPATCHER = {
    "generate_code": skill_generate_code,
    "manage_files": skill_manage_files,
    "list_files": skill_manage_files, # Variação mapeada para a mesma função
    "search_web": skill_search_web,     # Placeholder
    "weather_forecast": skill_search_web, # Mapear intenção específica para skill genérica por enquanto
    "remember_info": skill_remember_info, # Skill real
    "recall_info": skill_recall_info,     # Nova skill
    "get_value": skill_recall_info,       # Mapear variações
    "get_population": skill_search_web, # Mapear intenção específica para skill genérica por enquanto
    "modify_code": skill_modify_code,
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