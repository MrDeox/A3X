from .generate_code import skill_generate_code
from .manage_files import skill_manage_files, execute_delete_file # execute_delete_file não é uma skill, mas é usada aqui
from .web_search import skill_search_web
from .memory import skill_remember_info, skill_recall_info
from .unknown import skill_unknown
from .modify_code import skill_modify_code

__all__ = [
    'skill_generate_code',
    'skill_manage_files',
    'skill_remember_info',
    'skill_recall_info',
    'skill_search_web',
    'skill_modify_code'
]
