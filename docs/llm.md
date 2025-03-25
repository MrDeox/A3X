# Documentação Interna do Módulo LLM

## 1. Visão Geral

O módulo `llm` é o núcleo de processamento de linguagem natural do A³X. Ele executa modelos GGUF (GPT-Generated Unified Format) via `llama.cpp`, com suporte nativo a GPU AMD através do ROCm.

A interface principal é a função `run_llm()`, que encapsula toda a complexidade de execução do modelo, gerenciamento de GPU e formatação de prompts.

## 2. Interface

```python
def run_llm(prompt: str, max_tokens: int = 128) -> str
```

### Parâmetros
- `prompt`: Texto do comando/pergunta a ser processado
- `max_tokens`: Limite de tokens na resposta (default: 128)

### Configurações Internas
- Usa 16 camadas na GPU por padrão (`--gpu-layers 16`)
- Modelo: Q4_K_M quantizado para otimização de memória
- Contexto: 2048 tokens
- Temperatura: 0.7 (balanceamento entre criatividade e precisão)

## 3. Formato do Prompt

O prompt deve seguir o formato instruct/chat:

```
<|im_start|>user MENSAGEM <|im_end|> <|im_start|>assistant
```

### Exemplos de Formatação

```python
# Pergunta simples
prompt = "<|im_start|>user Qual é a capital do Brasil? <|im_end|> <|im_start|>assistant"

# Instrução de programação
prompt = "<|im_start|>user Crie uma função Python que calcule o fatorial de um número <|im_end|> <|im_start|>assistant"

# Pedido de resumo
prompt = "<|im_start|>user Faça um resumo do texto: [TEXTO] <|im_end|> <|im_start|>assistant"
```

## 4. Regras de Uso

### Obrigatório
1. Sempre formatar o prompt corretamente com as tags `<|im_start|>` e `<|im_end|>`
2. Usar o modelo Q4_K_M para evitar problemas de memória GPU
3. Manter prompts claros e objetivos
4. Respeitar o limite de tokens (128/256)

### Proibido
1. Enviar prompts vazios
2. Usar sem GPU ROCm ativa
3. Modificar o caminho do modelo sem atualizar `inference.py`
4. Exceder o limite de memória da GPU

## 5. Exemplos

### Perguntas Simples
```python
result = run_llm("<|im_start|>user Qual é a capital do Brasil? <|im_end|> <|im_start|>assistant")
```

### Instruções de Programação
```python
result = run_llm("<|im_start|>user Implemente uma função que calcule o MDC de dois números <|im_end|> <|im_start|>assistant")
```

### Pedidos de Resumo
```python
result = run_llm("<|im_start|>user Faça um resumo do texto: [TEXTO] <|im_end|> <|im_start|>assistant")
```

## 6. Tratamento de Erros

### Captura de Erros
- O wrapper captura erros do subprocess
- Logs detalhados são gerados em caso de falha
- O executor pode tentar reformulação automática

### Estratégias de Recuperação
1. Tentar com prompt reformulado
2. Reduzir número de camadas GPU
3. Limpar cache de GPU
4. Reiniciar o processo

### Logs
```python
# Exemplo de log de erro
[ERROR] Falha na execução do modelo: CUDA out of memory
[INFO] Tentando recuperação com menos camadas GPU
```

## 7. Restrições

### Hardware
- GPU AMD com suporte a ROCm
- Mínimo 8GB VRAM
- Drivers ROCm atualizados

### Software
- Python 3.8+
- llama.cpp compilado com suporte ROCm
- Modelo GGUF Q4_K_M

### Segurança
- Não executar prompts maliciosos
- Não expor o modelo a dados sensíveis
- Manter o modelo atualizado

## 8. Integração com o Executor

O Executor A³X deve:
1. Validar o prompt antes de enviar
2. Tratar erros adequadamente
3. Manter histórico de execuções
4. Monitorar uso de GPU
5. Implementar retry com backoff 