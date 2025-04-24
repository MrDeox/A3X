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
pip install -e . # Para instalação editável (recomendado para desenvolvimento)
```

4. Configure as variáveis de ambiente (copie e edite o exemplo):
```bash
cp .env.example .env
```

## Uso Básico

O A³X pode ser utilizado de duas formas principais:

### 1. Sistema Unificado (Recomendado)

```python
from a3x.core.a3x_unified import create_a3x_system
import asyncio

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

asyncio.run(main())
```

### 2. Componentes Individuais

```python
from a3x.fragments.registry import FragmentRegistry
from a3x.core.tool_registry import ToolRegistry
from a3x.core.orchestrator import TaskOrchestrator
from a3x.core.memory_manager import MemoryManager
from a3x.core.llm_interface import LLMInterface

# Inicializar componentes
fragment_registry = FragmentRegistry()
tool_registry = ToolRegistry()
memory_manager = MemoryManager()
llm_interface = LLMInterface()

# Registrar fragments e skills
tool_registry.discover_skills()
fragment_registry.discover_and_register_fragments()

# Criar orquestrador
orchestrator = TaskOrchestrator(
    fragment_registry=fragment_registry,
    tool_registry=tool_registry,
    memory_manager=memory_manager,
    llm_interface=llm_interface
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

O A³X é estruturado em torno de componentes centrais que promovem modularidade, extensibilidade e interoperabilidade:

### Núcleo (Core)
- **TaskOrchestrator**: Componente central que coordena a execução de tarefas, delegando para fragments especializados conforme necessário.
- **FragmentRegistry**: Gerencia o registro e descoberta dinâmica de fragments disponíveis no sistema.
- **ToolRegistry**: Gerencia o registro e descoberta de skills (ferramentas) que podem ser utilizadas por fragments.
- **MemoryManager**: Sistema de memória persistente e contexto compartilhado entre fragments, garantindo histórico e aprendizado contínuo.
- **LLMInterface**: Interface padronizada para interação com modelos de linguagem de grande porte (LLMs), utilizada por fragments como ProfessorLLM.

### Fragments
Fragments são unidades de competência autônomas, cada uma responsável por um aspecto do processo de raciocínio ou execução. Exemplos:
- **ProfessorLLMFragment**: Consulta LLMs para análise e geração de conhecimento.
- **KnowledgeInterpreterFragment**: Converte respostas textuais em comandos A3L estruturados.
- **MetaReflectorFragment**: Analisa ciclos evolutivos e sugere ajustes estratégicos.
- **EvolutionOrchestratorFragment**: Orquestra ciclos completos de evolução simbólica.

Fragments são registrados automaticamente via o decorador `@fragment` e gerenciados pelo `FragmentRegistry`.

### Skills
Skills são funções especializadas que realizam operações atômicas (ex: leitura de arquivos, execução de código, consulta web). São registradas via o decorador `@skill` e gerenciadas pelo `ToolRegistry`. Fragments podem requisitar skills conforme necessário durante sua execução.

### Comunicação e Memória
- **Communication Layer**: Todos os fragments e a CLI compartilham contexto persistente via `a3x/core/communication/`, utilizando o mesmo `task_id` para interoperabilidade.
- **Ciclo Completo**: O fluxo típico envolve o TaskOrchestrator delegando para fragments, que por sua vez utilizam skills, com estados e resultados compartilhados pelo MemoryManager.

### Remoção de Componentes Obsoletos
- Componentes antigos como `ReactAgent` e `MemoryBank` foram removidos. O fluxo é agora centrado no TaskOrchestrator, FragmentExecutor, e comunicação persistente.

### Extensibilidade
- Para adicionar novos Fragments ou Skills, basta criar o arquivo correspondente e usar os decoradores `@fragment` ou `@skill`.

Consulte a seção de Desenvolvimento para exemplos práticos.
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
