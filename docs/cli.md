# Documentação Interna do Módulo CLI

## 1. Visão Geral

O módulo `cli` serve como interface segura entre o A³X e o terminal do sistema operacional. Ele encapsula a execução de comandos shell, fornecendo controle granular, rastreabilidade e mecanismos de segurança para evitar ações destrutivas ou não autorizadas.

O módulo é responsável por:
- Executar comandos de forma controlada
- Capturar e formatar saídas
- Registrar logs detalhados
- Implementar validações de segurança

## 2. Interface

```python
def execute(command: str, capture_output: bool = True) -> str
```

### Parâmetros
- `command`: Comando shell a ser executado
- `capture_output`: Se True, retorna stdout/stderr como string; se False, apenas executa

### Retorno
- String contendo a saída do comando (stdout/stderr)
- Em caso de erro, lança exceção com detalhes

## 3. Regras de Execução

### Validação
```python
def validate_command(command: str) -> bool:
    # Verifica se o comando é permitido
    if command in BLOCKED_COMMANDS:
        return False
        
    # Verifica sintaxe básica
    if not re.match(r'^[a-zA-Z0-9_\-\.\s\/]+$', command):
        return False
        
    return True
```

### Restrições
1. Apenas comandos explícitos
2. Sem geração aleatória
3. Evitar comandos destrutivos
4. Logging obrigatório

### Logs
```python
def log_execution(command: str, success: bool, output: str = None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'command': command,
        'success': success,
        'output': output
    }
    logger.info(json.dumps(log_entry))
```

## 4. Segurança

### Comandos Bloqueados
```python
BLOCKED_COMMANDS = {
    'rm', 'rmdir', 'rm -rf',
    'shutdown', 'reboot', 'halt',
    'mv /', 'dd', 'mkfs',
    'chmod', 'chown',
    'sudo', 'su',
    'wget', 'curl',
    'nc', 'netcat',
    '> /dev/sda',
    'mkfs.ext4',
    'dd if=',
    'mkfs.'
}
```

### Validações de Segurança
1. Verificar comandos bloqueados
2. Preferir comandos não-interativos
3. Validar caminhos absolutos
4. Sanitizar inputs

### Exemplo de Validação
```python
def is_safe_command(command: str) -> bool:
    # Verifica comandos bloqueados
    if any(cmd in command for cmd in BLOCKED_COMMANDS):
        return False
        
    # Verifica redirecionamentos perigosos
    if '>' in command or '>>' in command:
        if '/dev/' in command:
            return False
            
    # Verifica comandos de sistema
    if command.startswith(('sudo', 'su')):
        return False
        
    return True
```

## 5. Logs e Auditoria

### Formato do Log
```json
{
    "timestamp": "2024-03-14T10:30:00",
    "command": "ls -la",
    "success": true,
    "output": "...",
    "error": null,
    "duration": 0.1
}
```

### Localização
- Arquivo: `logs/cli.log`
- Rotação: Diária
- Retenção: 30 dias

### Campos Obrigatórios
1. Timestamp
2. Comando executado
3. Status (sucesso/falha)
4. Saída/Erro
5. Duração

## 6. Estratégias de Recuperação

### Tratamento de Erros
```python
try:
    result = execute(command)
except CommandError as e:
    logger.error(f"Erro na execução: {e}")
    return format_error(e)
except Exception as e:
    logger.critical(f"Erro crítico: {e}")
    raise
```

### Formatação de Erros
```python
def format_error(error: CommandError) -> str:
    return f"""
    Erro na execução do comando:
    Comando: {error.command}
    Código: {error.code}
    Mensagem: {error.message}
    Sugestão: {error.suggestion}
    """
```

## 7. Exemplos

### Execução Básica
```python
# Listar arquivos
output = execute("ls -la")

# Criar diretório
execute("mkdir -p data/logs")

# Verificar espaço em disco
df_output = execute("df -h")
```

### Com Tratamento de Erro
```python
try:
    # Tentar comando que pode falhar
    result = execute("git pull origin main")
except CommandError as e:
    if "conflict" in str(e):
        # Tratar conflito
        print("Conflito detectado, precisa de intervenção manual")
    else:
        # Outro erro
        print(f"Erro: {e}")
```

### Sem Captura de Output
```python
# Executar sem capturar saída
execute("notify-send 'A³X' 'Processo concluído'", capture_output=False)
```

## 8. Integração com o Executor

### Responsabilidades do Executor
1. **Validação**
   - Limpar e sanitizar comandos
   - Verificar segurança
   - Validar parâmetros

2. **Monitoramento**
   - Registrar no histórico
   - Verificar efeitos
   - Tratar erros

3. **Segurança**
   - Não executar comandos externos
   - Revisar inputs
   - Manter logs

### Exemplo de Uso no Executor
```python
def process_command(self, input_text: str) -> str:
    # Extrair comando do texto
    command = self._extract_command(input_text)
    
    # Validar comando
    if not self._validate_command(command):
        return "Comando não permitido por questões de segurança"
    
    try:
        # Executar comando
        result = execute(command)
        
        # Registrar no histórico
        self.command_history.append({
            'type': 'cli',
            'command': command,
            'result': result
        })
        
        return result
        
    except Exception as e:
        self._log(f"Erro na execução: {e}", "ERROR")
        return f"Erro ao executar comando: {e}"
``` 