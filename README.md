# A³X - Sistema de Inteligência Artificial Local

A³X é um sistema de inteligência artificial local, modular e avançado, desenvolvido para fornecer uma experiência de IA personalizada e segura.

## Características

- **Processamento de Linguagem Natural**: Análise de intenções e geração de respostas
- **Execução de Código**: Suporte para execução segura de código Python e comandos do terminal
- **Sistema de Memória**: Armazenamento e recuperação de informações
- **Modular**: Arquitetura modular e extensível
- **Local**: Execução local sem dependência de serviços externos
- **Seguro**: Execução segura de código e comandos

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/a3x.git
cd a3x
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

O sistema pode ser usado através da classe `Executor`:

```python
from core import Executor

# Inicializa o executor
executor = Executor()

# Processa um comando
result = executor.process_command("Olá, como vai?")
print(result['response'])
```

### Comandos Suportados

- **Saudações**: "Olá", "Oi", "Bom dia", etc.
- **Memória**: 
  - Armazenar: "lembre-se que X como Y"
  - Recuperar: "qual era Y"
- **Terminal**: "execute o comando X"
- **Python**: "execute o código python X"
- **Perguntas**: Qualquer texto terminado com "?"

## Testes

Para executar os testes:

```bash
pytest tests/
```

## Estrutura do Projeto

```
a3x/
├── core/                 # Módulo principal
│   ├── __init__.py
│   ├── executor.py      # Executor principal
│   ├── analyzer.py      # Análise de intenções
│   ├── code_runner.py   # Execução de código
│   └── llm.py          # Processamento de linguagem
├── memory/             # Sistema de memória
│   ├── __init__.py
│   ├── models.py      # Modelos de dados
│   └── system.py      # Sistema de memória
├── tests/             # Testes
│   ├── __init__.py
│   ├── test_core.py
│   └── test_memory.py
├── requirements.txt   # Dependências
└── README.md         # Este arquivo
```

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes. 