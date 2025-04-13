# core/code_safety.py
import ast
import logging

logger = logging.getLogger(__name__)


def is_safe_ast(code_string: str) -> tuple[bool, str]:
    """Analisa a AST do código para permitir apenas construções seguras."""
    try:
        tree = ast.parse(code_string)
        allowed_nodes = (
            ast.Module,
            ast.Expr,
            ast.Constant, # Includes None, True, False, numbers, strings, bytes
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Assign,
            ast.AnnAssign, # Allow type-hinted assignments
            ast.FunctionDef,
            ast.AsyncFunctionDef, # Allow async functions
            ast.arguments,
            ast.arg,
            ast.Return,
            ast.BinOp,
            ast.UnaryOp, # Allow operators like not, -
            ast.USub,
            ast.UAdd,
            ast.Not,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            ast.BoolOp, # Allow 'and', 'or'
            ast.And,
            ast.Or,
            ast.For,
            ast.While,
            ast.If,
            ast.Pass,
            ast.Break,
            ast.Continue,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set, # Allow set literals
            ast.Subscript, # Allow indexing/slicing (e.g., my_list[0])
            ast.Slice, # Allow slicing (e.g., my_list[1:3])
            ast.Index, # Deprecated in Python 3.9, but still used by ast.parse
            # Comprehensions
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.comprehension,
            # F-strings
            ast.JoinedStr,
            ast.FormattedValue,
            # Basic Exception Handling (Careful!)
            # ast.Try,
            # ast.ExceptHandler,
            # ast.Raise, # Be cautious allowing raise
            # Control Flow
            ast.Await, # For async functions
            ast.Yield, # For generators
            ast.YieldFrom,
        )
        allowed_calls = {
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "bool", # Allow bool conversion
            "list",
            "dict",
            "tuple",
            "set",
            "abs",
            "round",
            "min",
            "max",
            "sum",
            "sorted",
            "reversed",
            "zip",
            "enumerate",
            "type", # Allow checking type
            "isinstance", # Allow isinstance checks
            "issubclass", # Allow issubclass checks
        }  # Funções built-in seguras

        allowed_methods = {
            # String methods
            "strip", "lstrip", "rstrip", "split", "join", "format",
            "upper", "lower", "capitalize", "title", "replace",
            "startswith", "endswith", "find", "index", "count",
            "isdigit", "isalpha", "isalnum", "isspace",
            # List methods
            "append", "pop", "insert", "remove", "sort", "reverse", "index", "count", "clear", "copy", "extend",
            # Dict methods
            "get", "keys", "values", "items", "update", "pop", "clear", "copy",
            # Set methods
            "add", "remove", "discard", "pop", "clear", "copy", "update",
            "intersection", "union", "difference", "issubset", "issuperset",
        } # Métodos seguros comuns

        # Imports são proibidos por padrão (nenhum ast.Import* em allowed_nodes)

        for node in ast.walk(tree):
            node_type = type(node)

            # 1. Check if node type itself is allowed
            if node_type not in allowed_nodes:
                return False, f"Construção/Nó AST não permitido: {node_type.__name__}"

            # 2. Specific checks for potentially dangerous node types
            if node_type is ast.Call:
                func = node.func
                # Check calls to built-ins/globals (e.g., print(), len())
                if isinstance(func, ast.Name):
                    if func.id not in allowed_calls:
                        # Allow calls to functions defined within the *same code string*
                        is_defined_locally = False
                        for definition in ast.walk(tree): # Check the whole tree for definitions
                            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)) and definition.name == func.id:
                                is_defined_locally = True
                                break
                        if not is_defined_locally:
                            return False, f"Chamada de função global/builtin não permitida: {func.id}"
                # Check method calls (e.g., my_list.append(), my_string.strip())
                elif isinstance(func, ast.Attribute):
                    # Check if the method name itself is in the allowed list
                    if func.attr not in allowed_methods:
                         return False, f"Chamada de método não permitida: {func.attr}"
                    # Optional: Could add checks on the object being called (func.value)
                    # but this gets complex quickly (type tracking). For now, allow
                    # approved methods on any object.
                else:
                    # Disallow other complex calls like lambda calls immediately invoked, etc.
                    return False, f"Tipo de chamada complexa não permitida: {type(func).__name__}"

            # 3. Check for direct attribute access (e.g., obj.attr)
            elif node_type is ast.Attribute:
                # Allow accessing attributes (like func.attr above),
                # but block access to potentially dangerous dunder attributes.
                # Fine-grained control can be added here if needed (e.g., allow specific attrs).
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    # Allow specific safe dunders if necessary, e.g. __len__ ?
                    # For now, block all dunders accessed directly.
                    # Note: Method calls like obj.__len__() are handled by ast.Call check.
                    if node.attr not in {'__len__'}: # Example: explicitly allow __len__
                         return False, f"Acesso a atributo dunder não permitido: {node.attr}"
                # Block accessing attributes starting with single underscore generally
                # Although this is convention, it prevents access to potentially internal state.
                elif node.attr.startswith('_'):
                     # Allow specific safe private attributes if needed
                     pass # Allow _* for now, adjust if needed

            # 4. Block specific dangerous built-in names if used directly as Name nodes
            # (Safety belt in case they slip through other checks)
            elif node_type is ast.Name:
                 if node.id in {'eval', 'exec', 'open', 'input', 'compile', '__import__', 'globals', 'locals', 'vars'}:
                     # Check context (e.g., is it being called or just assigned?)
                     # For simplicity, block their direct use as names for now.
                     return False, f"Uso direto do nome builtin perigoso: {node.id}"


        return True, "Código parece seguro (análise AST)."
    except SyntaxError as e:
        return False, f"Erro de sintaxe no código: {str(e)}"
    except Exception as e:
        logger.error(f"Erro inesperado durante análise AST: {e}", exc_info=True)
        return False, f"Erro inesperado durante análise AST: {e}"
