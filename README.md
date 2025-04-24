# A³X - Sistema de Inteligência Artificial Simbiótica

O A³X é um sistema de IA que integra processamento simbólico (A3L) e neural (A3Net) em uma arquitetura modular e evolutiva.

## Características Principais

- **Integração Simbólico-Neural**: Combina o poder do processamento simbólico com redes neurais
- **Evolução Autônoma**: Capacidade de aprender e evoluir através de ciclos autônomos
- **Arquitetura Modular**: Sistema baseado em fragments especializados e independentes
- **Memória Semântica**: Sistema de memória que permite aprendizado contínuo
- **LLM como Mentor**: Uso de LLMs como professores, não como executores

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/A3X.git
cd A3X
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente (copie e edite o exemplo):
```bash
cp .env.example .env
```

## Uso Básico

O A³X pode ser usado de duas formas principais:

### 1. Sistema Unificado (Recomendado)

```python
from a3x.core.a3x_unified import create_a3x_system

async def main():
    # Criar e inicializar o sistema
    system = await create_a3x_system()
    
    try:
        # Executar uma tarefa
        result = await system.execute_task(
            "Analisar este código e sugerir melhorias."
        )
        print(f"Resultado: {result}")
        
        # Iniciar ciclo autônomo
        await system.start_autonomous_cycle()
        
    finally:
        # Limpar recursos
        await system.cleanup()

# Rodar
asyncio.run(main())
```

### 2. Componentes Individuais

```python
from a3x.fragments.registry import FragmentRegistry
from a3x.core.tool_registry import ToolRegistry
from a3x.core.orchestrator import TaskOrchestrator

# Inicializar componentes
fragment_registry = FragmentRegistry()
tool_registry = ToolRegistry()

# Registrar fragments e skills
fragment_registry.discover_and_register_fragments()
tool_registry.discover_skills()

# Criar orquestrador
orchestrator = TaskOrchestrator(
    fragment_registry=fragment_registry,
    tool_registry=tool_registry
)

# Executar tarefa
result = await orchestrator.orchestrate(
    "Analisar este código e sugerir melhorias."
)
```

## Exemplos

Veja a pasta `examples/` para mais exemplos de uso, incluindo:

- Análise de código
- Ciclos autônomos
- Integração com outros sistemas
- Uso de skills personalizadas

## Arquitetura

O A³X é composto por vários subsistemas integrados:

### Core
- **Orchestrator**: Coordena a execução de tarefas
- **FragmentRegistry**: Gerencia fragments disponíveis
- **ToolRegistry**: Gerencia skills e ferramentas
- **MemoryManager**: Sistema de memória e contexto

### Fragments
- **ProfessorLLM**: Interface com modelos de linguagem
- **KnowledgeInterpreter**: Tradução entre linguagem natural e A3L
- **AutonomousStarter**: Inicia ciclos autônomos
- **MetaLearner**: Gerencia evolução do sistema

### Skills
- Módulos de funcionalidade específica
- Facilmente extensíveis
- Integração com APIs externas

## Desenvolvimento

### Adicionando Novos Fragments

1. Crie um novo arquivo em `a3x/fragments/`:
```python
from a3x.fragments.base import BaseFragment

@fragment(name="meu_fragment")
class MeuFragment(BaseFragment):
    async def execute(self, ctx, args):
        # Implementação
        pass
```

### Adicionando Novas Skills

1. Crie um novo arquivo em `a3x/skills/`:
```python
from a3x.core.skills import skill

@skill(name="minha_skill")
async def minha_skill(ctx, parametro: str):
    # Implementação
    pass
```

## Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -am 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Crie um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

- **Autor**: Seu Nome
- **Email**: seu.email@exemplo.com
- **Website**: https://seu-site.com

## Agradecimentos

- Contribuidores do projeto
- Comunidade de IA
- Frameworks e bibliotecas utilizadas
