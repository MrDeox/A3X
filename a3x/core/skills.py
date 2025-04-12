# core/tools.py
import os
import sys
import logging
import importlib
import inspect
import pkgutil  # <<< ADDED IMPORT >>>
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel, create_model  # Import Pydantic

# Initialize logger before potentially using it in print statements
logger = logging.getLogger(__name__)

# <<< ADD LOG HERE >>>
print(f"\n*** Executing module: a3x.core.skills (ID: {id(sys.modules.get('a3x.core.skills'))}) ***\n")

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

# <<< IMPORT SKILL_REGISTRY from skill_management >>>
from a3x.core.skill_management import SKILL_REGISTRY

# --- Novo Registro de Skills e Decorador ---

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
        param_details = {}
        for param_name, param_info in parameters.items():
            # Handle both (type,) and (type, default) formats
            if isinstance(param_info, (list, tuple)):
                if len(param_info) == 1:
                    param_type = param_info[0]
                    default_value = inspect.Parameter.empty # Indicate no default
                elif len(param_info) == 2:
                    param_type, default_value = param_info
                    # <<< FIX: Correctly check against inspect.Parameter.empty >>>
                    is_required = (default_value is inspect.Parameter.empty)
                    default_desc = f" (default: {default_value})" if not is_required else ""
                else:
                    logger.warning(f"Invalid parameter format for '{param_name}' in skill '{name}'. Expected (type,) or (type, default).")
                    continue
            else:
                 logger.warning(f"Invalid parameter format for '{param_name}' in skill '{name}'. Expected tuple or list.")
                 continue

            # Validate against function signature if possible (optional but good practice)
            if param_name not in func_params:
                logger.warning(f"Parameter '{param_name}' defined in @skill for '{name}' but not found in function signature.")
                # Continue registering, but log warning

            param_details[param_name] = {
                # Store the actual type object, not its string representation
                "type_obj": param_type,
                "type_str": str(param_type.__name__) if hasattr(param_type, '__name__') else str(param_type), # Keep string for logging/display
                "required": default_value is inspect.Parameter.empty,
                "default": default_value if default_value is not inspect.Parameter.empty else None
            }

        # Criar schema Pydantic dinamicamente
        pydantic_fields = {}
        for param_name, param_info in param_details.items():
            # Use the actual type object for Pydantic model creation
            field_type = param_info["type_obj"]
            field_default = param_info["default"] if not param_info["required"] else ... # Use Ellipsis for required fields
            pydantic_fields[param_name] = (field_type, field_default)

        # Nome do schema dinâmico (para melhor debug/docs)
        schema_name = f"{name.capitalize()}SkillSchema"
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
    Carrega ou recarrega dinamicamente o PACOTE de skills e todos os seus
    submódulos para garantir que os decoradores @skill sejam executados
    e registrem as skills.

    Args:
        skill_package_name (str): O nome do pacote Python para as skills (ex: 'a3x.skills').
    """
    logger.info(f"Attempting to load/reload skills from package: {skill_package_name}")

    # Garantir que a raiz do projeto esteja no path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.debug(f"Added {project_root} to sys.path")

    count_before = len(SKILL_REGISTRY)
    module = None # Initialize module variable

    try:
        # 1. Load or Reload the main package
        if skill_package_name in sys.modules:
            logger.info(f"Reloading existing skills package: {skill_package_name}")
            module = importlib.reload(sys.modules[skill_package_name])
        else:
            logger.info(f"Importing skills package for the first time: {skill_package_name}")
            module = importlib.import_module(skill_package_name)

        # 2. Explicitly walk and import/re-import submodules to ensure decorators run
        package_paths = getattr(module, '__path__', None)
        if not package_paths:
             logger.warning(f"'{skill_package_name}' does not appear to be a package (missing __path__). Cannot discover submodules.")
             # If it's not a package but a single module file with skills?
             # This scenario is less common for organizing multiple skills.
             # If needed, one could inspect the module directly here.
        else:
            logger.debug(f"Walking package path(s) to discover skill modules: {package_paths}")
            prefix = module.__name__ + '.'
            for importer, modname, ispkg in pkgutil.walk_packages(package_paths, prefix):
                if not ispkg:
                    try:
                        if modname in sys.modules:
                            logger.debug(f"---> Reloading skill module: {modname}")
                            # Explicitly reload the submodule to re-run decorators
                            importlib.reload(sys.modules[modname])
                        else:
                            # This case is less likely during a reload, but handle it.
                            logger.debug(f"---> Importing new skill module: {modname}")
                            importlib.import_module(modname)
                    except Exception as e_mod:
                        logger.error(f"Failed to import/reload skill module '{modname}'. Error: {e_mod}", exc_info=True)
                # else: # Optional: Handle subpackages if skills can be nested deeper
                #    logger.debug(f"Skipping subpackage: {modname}")


        count_after = len(SKILL_REGISTRY)
        newly_registered = count_after - count_before
        logger.info(
            f"Finished loading/reloading skills from '{skill_package_name}'. Newly registered: {newly_registered}. Total registry: {count_after}"
        )

    except ModuleNotFoundError as e:
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


def get_skill_descriptions() -> str:
    """
    Retorna uma string formatada descrevendo todas as skills registradas,
    incluindo seus parâmetros e suas descrições do docstring.
    """
    if not SKILL_REGISTRY:
        return "No skills are currently registered."

    descriptions = []
    for name, skill_info in SKILL_REGISTRY.items():
        func = skill_info.get("function")
        desc = skill_info.get("description", "No description.")
        params_dict = skill_info.get("parameters", {})

        # --- Extract Parameter Descriptions from Docstring --- START
        param_docs = {}
        if func and func.__doc__:
            docstring = inspect.getdoc(func)
            if docstring:
                lines = docstring.split('\n')
                in_args_section = False
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.lower().startswith("args:"):
                        in_args_section = True
                        continue
                    if in_args_section:
                        if not stripped_line: # End of Args section
                            break
                        if ":" in stripped_line:
                            # Simple parse: Assume format 'param_name (type): Description.'
                            # or 'param_name: Description.' (if type is omitted in docstring)
                            parts = stripped_line.split(':', 1)
                            param_part = parts[0].strip()
                            param_desc_text = parts[1].strip()
                            # Extract just the name (handle potential type hints in docstring like 'param (str)')
                            param_name_from_doc = param_part.split('(')[0].strip()
                            param_docs[param_name_from_doc] = param_desc_text
        # --- Extract Parameter Descriptions from Docstring --- END

        # Obter docstring como descrição alternativa para a *skill* (se não houver no decorator)
        if desc == "No description." and func and func.__doc__:
            desc = inspect.getdoc(func).split('\n')[0] if inspect.getdoc(func) else "No description."

        param_strs = []
        if params_dict:
            for param_name, param_info in params_dict.items():
                param_type_str = "Any"
                default_desc = ""
                is_required = False

                if isinstance(param_info, (list, tuple)):
                    if len(param_info) == 1:
                        param_type = param_info[0]
                        is_required = True
                    elif len(param_info) == 2:
                        param_type, default_value = param_info
                        is_required = (default_value is inspect.Parameter.empty)
                        default_desc = f" (default: {default_value})" if not is_required else ""
                    else:
                        continue

                    if hasattr(param_type, '__name__'):
                        param_type_str = param_type.__name__
                    else:
                        param_type_str = str(param_type)
                else:
                    continue

                required_str = " (required)" if is_required else default_desc
                # <<< Include Parameter Description >>>
                param_desc = param_docs.get(param_name, "") # Get extracted description
                desc_str = f" - {param_desc}" if param_desc else ""
                param_strs.append(f"{param_name}: {param_type_str}{required_str}{desc_str}")

        param_section = ", ".join(param_strs)
        if param_section:
            descriptions.append(f"- {name}({param_section}): {desc}")
        else:
            descriptions.append(f"- {name}(): {desc}")

    return "\n".join(descriptions)


def get_skill(skill_name: str) -> Optional[Dict[str, Any]]:
    """Retorna a informação registrada para uma ferramenta específica (skill)."""
    registry = get_skill_registry()  # Garante que o registro está acessível
    # Tenta carregar se vazio e a ferramenta não for encontrada
    if skill_name not in registry and not SKILL_REGISTRY:
        logger.warning(
            f"Skill '{skill_name}' not found in empty registry. Attempting to load skills."
        )
        load_skills()
        registry = get_skill_registry()  # Tenta novamente

    skill_info = registry.get(skill_name)
    if not skill_info:
        logger.warning(
            f"Skill '{skill_name}' not found in registry even after attempting load."
        )
    return skill_info


# É crucial chamar load_skills() no ponto de entrada principal da aplicação
# (provavelmente na inicialização do Agent) para garantir que todas as
# skills sejam registradas antes de serem necessárias.
# Exemplo: remover a chamada automática daqui
