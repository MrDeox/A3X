# Documentação Interna do Módulo Core

## 1. Visão Geral

O módulo `core` é o componente responsável por executar blocos de código Python dentro do A³X. Ele permite que o sistema "pense com código", resolvendo problemas, automatizando tarefas e manipulando dados em tempo real através de execução segura de código Python.

Este módulo é fundamental para o raciocínio computacional do A³X, mas exige controle rigoroso devido ao seu poder de execução. É usado quando comandos em linguagem natural envolvem lógica computacional, cálculos, manipulação de dados ou automação.

## 2. Interface

```python
def run_python_code(code: str) -> str
```

### Parâmetros
- `code`: String contendo o código Python a ser executado

### Retorno
- String contendo a saída formatada (stdout/stderr)
- Em caso de erro, mensagem formatada e segura

### Ambiente de Execução
```python
# Ambiente isolado
exec_globals = {
    'print': print,
    'len': len,
    'range': range,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'pow': pow,
    'divmod': divmod,
    'bin': bin,
    'hex': hex,
    'oct': oct,
    'chr': chr,
    'ord': ord,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'reduce': reduce,
    'any': any,
    'all': all,
    'sorted': sorted,
    'reversed': reversed,
    'format': format,
    'repr': repr,
    'ascii': ascii,
    'eval': eval,
    'exec': exec,
    'compile': compile,
    'hash': hash,
    'id': id,
    'type': type,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'super': super,
    'property': property,
    'classmethod': classmethod,
    'staticmethod': staticmethod,
    'getattr': getattr,
    'setattr': setattr,
    'hasattr': hasattr,
    'delattr': delattr,
    'vars': vars,
    'locals': locals,
    'globals': globals,
    'dir': dir,
    'help': help,
    'open': open,
    'print': print,
    'input': input,
    'exit': exit,
    'quit': quit,
    'copyright': copyright,
    'credits': credits,
    'license': license,
    'help': help,
}
```

## 3. Regras de Execução

### Restrições
1. Sem acesso a módulos perigosos:
   - `os`
   - `sys`
   - `subprocess`
   - `shutil`
   - `__import__`

2. Limites de Execução:
   - Tempo máximo: 5 segundos
   - Memória máxima: 100MB
   - Profundidade de recursão: 100

3. Ambiente Isolado:
   - Sem acesso a variáveis do sistema
   - Sem acesso a arquivos do sistema
   - Sem acesso a rede

### Validação
```python
def validate_code(code: str) -> bool:
    # Palavras-chave proibidas
    forbidden = {
        'import', 'from', 'as', 'with', 'try', 'except', 'finally',
        'raise', 'assert', 'del', 'global', 'nonlocal', 'yield',
        'lambda', 'class', 'def', 'return', 'break', 'continue',
        'pass', 'exec', 'eval', 'compile', 'open', 'file',
        'os', 'sys', 'subprocess', 'shutil', '__import__'
    }
    
    # Verifica palavras-chave
    for word in forbidden:
        if word in code:
            return False
            
    # Verifica tempo de execução
    if len(code) > 1000:  # Limite de caracteres
        return False
        
    return True
```

## 4. Segurança

### Sandbox
```python
def create_sandbox():
    """Cria ambiente isolado para execução."""
    sandbox = {
        '__builtins__': {
            name: getattr(builtins, name)
            for name in SAFE_BUILTINS
        }
    }
    return sandbox
```

### Validações de Segurança
1. Filtrar palavras-chave perigosas
2. Limitar recursos (tempo, memória)
3. Isolar ambiente de execução
4. Tratar exceções com segurança

### Exemplo de Execução Segura
```python
def run_safely(code: str) -> str:
    try:
        # Criar sandbox
        sandbox = create_sandbox()
        
        # Validar código
        if not validate_code(code):
            raise SecurityError("Código não permitido")
            
        # Executar com timeout
        with timeout(5):
            exec(code, sandbox)
            
        return "Execução concluída com sucesso"
        
    except TimeoutError:
        return "Tempo limite excedido"
    except Exception as e:
        return f"Erro na execução: {str(e)}"
```

## 5. Logs

### Formato do Log
```json
{
    "timestamp": "2024-03-14T10:30:00",
    "code": "print('teste')",
    "result": "teste",
    "error": null,
    "duration": 0.1,
    "memory_used": 1024
}
```

### Localização
- Arquivo: `logs/core.log`
- Rotação: Diária
- Retenção: 30 dias

### Campos Obrigatórios
1. Timestamp
2. Código executado
3. Resultado/Saída
4. Erros (se houver)
5. Tempo de execução
6. Uso de memória

## 6. Estratégias de Recuperação

### Tratamento de Erros
```python
def handle_execution_error(error: Exception) -> str:
    """Formata erro de execução de forma segura."""
    error_types = {
        SyntaxError: "Erro de sintaxe no código",
        NameError: "Variável não definida",
        TypeError: "Tipo de dado inválido",
        ValueError: "Valor inválido",
        IndexError: "Índice fora dos limites",
        KeyError: "Chave não encontrada",
        ZeroDivisionError: "Divisão por zero",
        TimeoutError: "Tempo limite excedido",
        MemoryError: "Memória insuficiente"
    }
    
    error_type = type(error)
    message = error_types.get(error_type, "Erro na execução")
    
    return f"{message}: {str(error)}"
```

## 7. Exemplos

### Execução Básica
```python
# Cálculo simples
result = run_python_code("print(2 + 2)")

# Manipulação de lista
result = run_python_code("""
lista = [1, 2, 3, 4, 5]
print(sum(lista))
""")

# Formatação de string
result = run_python_code("""
nome = 'A³X'
print(f'Olá, {nome}!')
""")
```

### Tratamento de Erros
```python
# Código com erro
try:
    result = run_python_code("print(variavel_inexistente)")
except Exception as e:
    print(f"Erro: {e}")

# Código com timeout
try:
    result = run_python_code("while True: pass")
except TimeoutError:
    print("Tempo limite excedido")
```

## 8. Integração com o Executor

### Responsabilidades do Executor
1. **Validação**
   - Verificar se é código Python
   - Sanitizar conteúdo
   - Validar segurança

2. **Execução**
   - Rodar em ambiente isolado
   - Monitorar recursos
   - Tratar erros

3. **Logging**
   - Registrar execuções
   - Manter histórico
   - Monitorar uso

### Exemplo de Uso no Executor
```python
def process_command(self, input_text: str) -> str:
    # Extrair código do texto
    code = self._extract_code(input_text)
    
    # Validar código
    if not self._validate_code(code):
        return "Código não permitido por questões de segurança"
    
    try:
        # Executar código
        result = run_python_code(code)
        
        # Registrar no histórico
        self.command_history.append({
            'type': 'core',
            'code': code,
            'result': result
        })
        
        return result
        
    except Exception as e:
        self._log(f"Erro na execução: {e}", "ERROR")
        return f"Erro ao executar código: {e}"
``` 