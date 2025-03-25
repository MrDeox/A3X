# Documentação Interna do Módulo de Memória

## 1. Visão Geral

O módulo de memória do A³X é responsável por fornecer continuidade cognitiva ao sistema através do armazenamento local e estruturado de informações contextuais. Ele permite que o Executor mantenha um "estado mental" entre execuções, armazenando lembretes, contextos, explicações e dados relevantes para o raciocínio.

A memória é implementada como um banco de dados local (SQLite) que persiste os dados entre sessões, permitindo que o sistema mantenha um histórico de interações e aprenda com experiências anteriores.

## 2. Interface

```python
def store(key: str, value: str) -> None
def retrieve(key: str) -> Optional[str]
```

### Funções

#### store
- **Descrição**: Armazena um par chave/valor no banco local
- **Parâmetros**:
  - `key`: Chave única para identificação
  - `value`: Valor a ser armazenado
- **Retorno**: None

#### retrieve
- **Descrição**: Recupera um valor armazenado
- **Parâmetros**:
  - `key`: Chave do valor desejado
- **Retorno**: Valor armazenado ou None se não encontrado

## 3. Regras de Armazenamento

### Chaves
- Devem ser curtas e descritivas
- Sem espaços (usar underscore)
- Apenas caracteres alfanuméricos e underscore
- Máximo de 64 caracteres

### Valores
- Apenas texto (string)
- Máximo de 10KB por entrada
- Podem conter:
  - Instruções
  - Resumos
  - Contextos
  - Explicações
  - Dados de raciocínio

### Persistência
- Dados são salvos em SQLite
- Localização: `data/memory.db`
- Backup automático diário

## 4. Estratégias de Organização

### Nomenclatura de Chaves
- Usar prefixos descritivos:
  - `lembrete_`: Para lembretes e tarefas
  - `erro_`: Para registros de erros
  - `explicacao_`: Para explicações e contextos
  - `dado_`: Para dados gerais

### Exemplos de Chaves
```
lembrete_revisar_codigo
erro_execucao_llm_001
explicacao_prompt_format
dado_contexto_sessao_001
```

### Estrutura Futura
- Suporte a tags (planejado)
- Timestamps automáticos
- Contexto associado
- Versionamento de valores

## 5. Regras de Recuperação

### Busca
- Exata por chave
- Case-sensitive
- Retorna None se não encontrado
- Log de tentativas de acesso

### Tratamento de Erros
```python
try:
    value = retrieve("chave_inexistente")
    if value is None:
        # Tratar como não encontrado
        pass
except Exception as e:
    # Log do erro
    logger.error(f"Erro ao recuperar valor: {e}")
```

### Fuzzy Match (Opcional)
- Implementado via `fuzzywuzzy`
- Sugere chaves similares
- Útil para correção de erros de digitação

## 6. Limitações

### Restrições
1. Apenas texto (sem arquivos binários)
2. Máximo 10KB por valor
3. Sem sobrescrição automática
4. Acesso restrito ao escopo do sistema

### Validações
```python
# Exemplo de validação
if len(value) > 10 * 1024:  # 10KB
    raise ValueError("Valor excede limite de tamanho")

if not re.match(r'^[a-zA-Z0-9_]+$', key):
    raise ValueError("Chave inválida")
```

## 7. Exemplos

### Armazenamento
```python
# Lembrete
store("lembrete_revisar_codigo", "Revisar o executor amanhã")

# Erro
store("erro_execucao_001", "Falha na execução do modelo: CUDA out of memory")

# Contexto
store("contexto_sessao_001", "Usuário solicitou ajuda com Python")
```

### Recuperação
```python
# Busca simples
lembrete = retrieve("lembrete_revisar_codigo")
if lembrete:
    print(f"Lembrete: {lembrete}")

# Com tratamento de erro
try:
    erro = retrieve("erro_execucao_001")
    if erro:
        logger.warning(f"Erro anterior: {erro}")
except Exception as e:
    logger.error(f"Erro ao recuperar: {e}")
```

## 8. Integração com o Executor

### Responsabilidades do Executor
1. **Validação**
   - Verificar formato da chave
   - Validar tamanho do valor
   - Confirmar sobrescrição

2. **Logging**
   - Registrar todas as operações
   - Manter histórico de acessos
   - Monitorar uso da memória

3. **Uso Inteligente**
   - Usar como suporte, não verdade absoluta
   - Validar dados recuperados
   - Manter contexto atualizado

4. **Segurança**
   - Não expor dados sensíveis
   - Limpar dados temporários
   - Fazer backup regular

### Exemplo de Uso no Executor
```python
def process_command(self, input_text: str) -> str:
    # Recuperar contexto anterior
    contexto = retrieve("contexto_sessao_001")
    if contexto:
        self._log(f"Contexto anterior: {contexto}")
    
    # Processar comando
    result = self._execute(input_text)
    
    # Armazenar resultado
    store(f"resultado_{len(self.command_history)}", result)
    
    return result
``` 