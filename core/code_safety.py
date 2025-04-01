# core/code_safety.py
import ast
import logging

logger = logging.getLogger(__name__)

def is_safe_ast(code_string: str) -> tuple[bool, str]:
    """Analisa a AST do código para permitir apenas construções seguras."""
    try:
        tree = ast.parse(code_string)
        allowed_nodes = (
            ast.Module, ast.Expr, ast.Constant, ast.Call, ast.Name,
            ast.Load, ast.Store, ast.Assign, ast.FunctionDef, ast.arguments,
            ast.arg, ast.Return, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            # Adicionar outros nós seguros conforme necessário (ex: loops, condicionais básicos)
            ast.For, ast.While, ast.If, ast.Compare, ast.Eq, ast.NotEq,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.Pass, ast.Break, ast.Continue, ast.List, ast.Tuple, ast.Dict,
            ast.Subscript, ast.Index, # Para acesso a listas/dicionários
        )
        allowed_calls = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'set'} # Funções built-in seguras
        allowed_imports = set() # Nenhum import permitido por padrão

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, f"Nó AST não permitido: {type(node).__name__}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_calls:
                        # Permitir chamadas a funções definidas no próprio código
                        # (Verifica se a função chamada está definida no escopo atual da AST)
                        is_defined_locally = False
                        for definition in ast.walk(tree):
                             if isinstance(definition, ast.FunctionDef) and definition.name == node.func.id:
                                 is_defined_locally = True
                                 break
                        if not is_defined_locally:
                             return False, f"Chamada de função não permitida: {node.func.id}"
                # Bloquear chamadas de atributos (ex: obj.method()) por segurança inicial
                elif isinstance(node.func, ast.Attribute):
                     return False, f"Chamada de método/atributo não permitida: {ast.dump(node.func)}"
            # Bloquear imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                 # Permitir apenas imports específicos se necessário
                 # module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                 # if module_name not in allowed_imports:
                 return False, f"Imports não são permitidos ({type(node).__name__})"
            # Bloquear acesso a atributos como __builtins__, __import__ etc.
            if isinstance(node, ast.Attribute):
                 # Permitir acesso a atributos seguros se necessário no futuro
                 # Ex: if node.attr not in {'append', 'pop', ...}:
                 if node.attr.startswith('_'): # Bloqueia atributos "privados" ou "mágicos"
                      return False, f"Acesso a atributo não permitido: {node.attr}"
            # Validação específica para BinOp
            if isinstance(node, ast.BinOp):
                 if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                      return False, f"Operação binária não permitida: {type(node.op).__name__}"


        return True, "Código parece seguro (análise AST básica)."
    except SyntaxError as e:
        return False, f"Erro de sintaxe no código: {str(e)}"
    except Exception as e:
        logger.error(f"Erro inesperado durante análise AST: {e}", exc_info=True)
        return False, f"Erro inesperado durante análise AST: {e}" 