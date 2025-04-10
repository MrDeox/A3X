# core/tools.py
import os
import sys
import logging
import importlib
import inspect
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, create_model  # Import Pydantic

# <<< ADD LOG HERE >>>
print(f"\n*** Executing module: a3x.core.tools (ID: {id(sys.modules.get('a3x.core.tools'))}) ***\n")

# <<< START PROJECT ROOT PATH CORRECTION >>>
# Calculate the project root (two levels up from core/tools.py)
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_current_file_dir)  # /home/arthur/Projects/A3X

# Add project root to sys.path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Clean up temporary variables
del _current_file_dir
# Keep 'project_root' as it might be useful elsewhere in this module, like in load_skills
# <<< END PROJECT ROOT PATH CORRECTION >>>

logger = logging.getLogger(__name__)

# --- Novo Registro de Skills e Decorador ---
SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {}
# <<< ADD LOG HERE >>>
print(f"\n*** Initializing SKILL_REGISTRY in a3x.core.tools (ID: {id(SKILL_REGISTRY)}) ***\n")


class SkillInputSchema(BaseModel):
    """Classe base para schemas de input de skill, pode ser estendida se necessário."""

    pass


def skill(name: str, description: str, parameters: Dict[str, tuple[type, Any]]):
    """
    Decorador para registrar uma função como uma skill disponível para o agente.

    Args:
        name: O nome único da skill (usado pelo LLM).
        description: Uma descrição clara do que a skill faz.
        parameters: Um dicionário onde as chaves são os nomes dos parâmetros
                    e os valores são tuplas (tipo_python, default_value).
                    Use ... (Ellipsis) como default_value para parâmetros obrigatórios.
    """

    def decorator(func: Callable):
        logger.debug(f"Registering skill: {name}")

        # Validar assinatura da função contra parâmetros declarados
        sig = inspect.signature(func)
        func_params = sig.parameters
        declared_param_names = set(parameters.keys())
        func_param_names = set(func_params.keys())

        # Verificar se a função aceita 'agent_memory' (opcional)
        takes_memory = "agent_memory" in func_param_names
        if takes_memory:
            func_param_names.remove(
                "agent_memory"
            )  # Não incluir na validação de schema

        # ADDED: Check for 'agent_history' as well
        takes_history = "agent_history" in func_param_names
        if takes_history:
            func_param_names.remove(
                "agent_history"
            )  # Do not include in schema validation

        # --- Start Modification: Ignore 'self' for method signature validation ---
        is_method = "self" in func_param_names
        if is_method:
            func_param_names.remove(
                "self"
            )  # Remove 'self' before comparing with declared params
        # --- End Modification ---

        # <<< START NEW: Define parameters to ignore during signature validation >>>
        ignored_params = {
            "self",
            "agent_memory",
            "agent_history",
            "resolved_path",
            "original_path_str",
            "kwargs",
            "ctx",
        }
        # <<< END NEW >>>

        # Verificar se todos os parâmetros declarados estão na assinatura da função
        # <<< MODIFIED: Remove ignored params before check >>>
        func_params_to_validate = func_param_names - ignored_params
        if not declared_param_names.issubset(func_params_to_validate):
            missing_in_func = declared_param_names - func_params_to_validate
            raise TypeError(
                f"Skill '{name}': Parameters {missing_in_func} declared in decorator but not found in function signature (excluding {ignored_params})."
            )

        # Verificar se a função não tem parâmetros extras
        # <<< MODIFIED: Remove ignored params before check >>>
        extra_in_func = func_params_to_validate - declared_param_names
        if extra_in_func:
            raise TypeError(
                f"Skill '{name}': Parameters {extra_in_func} found in function signature but not declared in decorator (excluding {ignored_params})."
            )

        # Criar schema Pydantic dinamicamente
        pydantic_fields = {}
        for param_name, (param_type, default_value) in parameters.items():
            pydantic_fields[param_name] = (param_type, default_value)

        # Nome do schema dinâmico (para melhor debug/docs)
        schema_name = f"{name.capitalize().replace('_', '')}InputSchema"
        try:
            # Usar Ellipsis diretamente para campos obrigatórios
            dynamic_schema = create_model(
                schema_name, **pydantic_fields, __base__=SkillInputSchema
            )
        except Exception as e:
            logger.error(
                f"Failed to create Pydantic schema for skill '{name}'. Error: {e}"
            )
            raise TypeError(
                f"Invalid Pydantic schema definition for skill '{name}'."
            ) from e

        # Registrar a skill
        SKILL_REGISTRY[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,  # Mantém a definição original para referência
            "schema": dynamic_schema,  # Schema Pydantic para validação
            "takes_memory": takes_memory,  # Indica se a skill usa memória
        }
        logger.info(f"Successfully registered skill: {name}")
        return func

    return decorator


def load_skills(skill_package_name: str = "a3x.skills"):
    """
    Importa dinamicamente o PACOTE de skills para garantir que __init__.py
    seja executado e registre as skills nele importadas.

    Args:
        skill_package_name (str): The Python package name for skills (e.g., 'a3x.skills').
    """
    # <<< ADDED: Prevent re-loading if already populated >>>
    if SKILL_REGISTRY:
        logger.debug("Skill registry already populated. Skipping load_skills.")
        return
    # <<< END ADDED >>>

    logger.info(f"Attempting to load skills package: {skill_package_name}")
    # # DEBUG: Print project_root and sys.path before import attempt # REMOVED DEBUGGING
    # # print(f"DEBUG: project_root in load_skills: {project_root}") # REMOVED DEBUGGING
    # # print(f"DEBUG: sys.path BEFORE import in load_skills: {sys.path}") # REMOVED DEBUGGING

    # Garantir que a raiz do projeto esteja no path ANTES de importar skills
    # (Repetido aqui para garantir, caso o módulo seja recarregado ou o path mude)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"DEBUG: Added {project_root} to sys.path in load_skills") # REMOVED DEBUGGING

    count_before = len(SKILL_REGISTRY)

    try:
        # Importa o pacote skills; seu __init__.py deve importar os módulos individuais.
        # Use the package name for import
        importlib.import_module(skill_package_name)  # Just import, don't assign
        # Recarregar o módulo se ele já foi importado, para pegar novas skills/mudanças
        # importlib.reload(module)
        count_after = len(SKILL_REGISTRY)
        newly_registered = count_after - count_before
        logger.info(
            f"Finished loading skills package '{skill_package_name}'. Newly registered skills: {newly_registered}. Total registry size: {count_after}"
        )
    except ModuleNotFoundError as e:
        # Adicionar mais detalhes ao log de erro
        logger.error(
            f"Could not find skills package '{skill_package_name}'. Check PYTHONPATH ({sys.path}) and directory structure. Error: {e}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Failed to import or process skills package '{skill_package_name}': {e}",
            exc_info=True,
        )


# --- Funções para acessar informações das Skills ---


def get_skill_registry() -> Dict[str, Dict[str, Any]]:
    """Retorna o registro completo das skills carregadas."""
    # Garante que as skills foram carregadas pelo menos uma vez
    # Chamada explícita de load_skills é preferível no ponto de entrada da aplicação (ex: Agent.__init__)
    # para controlar quando o carregamento ocorre. Remover o carregamento automático daqui.
    # <<< REMOVED Implicit Load Block >>>
    # # Se o registro estiver vazio, TENTAR carregar (útil para testes ou execuções diretas)
    # if not SKILL_REGISTRY:
    #     logger.warning(
    #         "Skill registry accessed while empty. Attempting to load skills implicitly."
    #     )
    #     load_skills()

    return SKILL_REGISTRY


def get_tool_descriptions() -> str:
    """Retorna uma string formatada com as descrições de todas as ferramentas registradas."""
    # <<< MODIFIED: Directly get registry, assume it's loaded explicitly elsewhere >>>
    registry = SKILL_REGISTRY # Directly access the global registry
    if not registry:
        # <<< REMOVED Implicit Load Attempt >>>
        # logger.warning(
        #     "Tool description requested but registry is empty. Attempting to load skills."
        # )
        # load_skills()
        # registry = get_skill_registry() # Tenta novamente
        # if not registry:
        logger.error(
            "Skill registry is empty when requesting tool descriptions. Skills should be loaded explicitly at startup."
        )
        return "No tools available (registry empty)."

    descriptions = []
    for name, skill_info in registry.items():
        # Formatar parâmetros usando a definição original para clareza no prompt
        params_str_list = []
        if skill_info.get("parameters"):
            for param_name, (param_type, default_value) in skill_info[
                "parameters"
            ].items():
                type_name = getattr(param_type, "__name__", str(param_type))
                desc = f"  - {param_name} (type: {type_name}"
                if default_value is not ...:  # Se não for obrigatório (Ellipsis)
                    desc += f", default: {repr(default_value)})"
                else:
                    desc += ", required)"
                params_str_list.append(desc)
            params_str = "\n".join(
                params_str_list
            )  # Usar \n para newline literal na string final
            descriptions.append(
                f"- {name}:\n  Description: {skill_info['description']}\n  Parameters:\n{params_str}"
            )
        else:
            descriptions.append(
                f"- {name}:\n  Description: {skill_info['description']}\n  Parameters: None"
            )

    return "\n".join(descriptions)


def get_tool(tool_name: str) -> Optional[Dict[str, Any]]:
    """Retorna a informação registrada para uma ferramenta específica (skill)."""
    registry = get_skill_registry()  # Garante que o registro está acessível
    # Tenta carregar se vazio e a ferramenta não for encontrada
    if tool_name not in registry and not SKILL_REGISTRY:
        logger.warning(
            f"Tool '{tool_name}' not found in empty registry. Attempting to load skills."
        )
        load_skills()
        registry = get_skill_registry()  # Tenta novamente

    tool_info = registry.get(tool_name)
    if not tool_info:
        logger.warning(
            f"Tool '{tool_name}' not found in registry even after attempting load."
        )
    return tool_info


# É crucial chamar load_skills() no ponto de entrada principal da aplicação
# (provavelmente na inicialização do Agent) para garantir que todas as
# skills sejam registradas antes de serem necessárias.
# Exemplo: remover a chamada automática daqui
