# Arquitetura Técnica do A³X System

Este documento descreve a arquitetura técnica atual do sistema A³X (Agente Autônomo Adaptativo), detalhando seus componentes principais, fluxos de trabalho e interconexões.

## 1. Visão Geral e Filosofia

*   Breve descrição dos objetivos do A³X (autonomia, aprendizado local, etc.).
*   Princípios de design (modularidade, eficiência, resiliência).

## 2. Diagrama da Arquitetura

*   Diagrama Mermaid mostrando os componentes principais e suas interações.

```mermaid
graph TD
    subgraph "Interface/Entrada"
        CLI[Interface CLI / assistant_cli.py]
        UserInput(Tarefa do Usuário)
    end

    subgraph "Núcleo Cognitivo (a3x/core)"
        Cerebrum[CerebrumXAgent / cerebrumx.py]
        AgentLoop[Loop Principal (ReAct/Orquestração) / agent.py]
        Planner[Planner / planner.py]
        Reflector[Agent Reflector / agent_reflector.py]
        LLM_I[LLM Interface / llm_interface.py]
        Parser[Agent Parser / agent_parser.py]
        ToolExec[Tool Executor / tool_executor.py]
        FragReg[Fragment Registry / fragments/registry.py]
        SkillReg[Skill Registry / skills.py]
        MemorySys[Sistema de Memória (FAISS/SQLite) / memory/*]
        Config[Configuração / config.py]
        Logger[Logging / logging_config.py]
    end

    subgraph "Módulos de Capacidade"
        Fragments[Fragments (Workers/Managers) / fragments/*]
        Skills[Skills (Ferramentas Atômicas) / skills/*]
    end

    subgraph "Infraestrutura Externa"
        LLM_Server[(LLM Local - Llama.cpp Server)]
        FileSystem[(Sistema de Arquivos)]
        Shell[(Terminal/Shell)]
        Web[(Navegador/APIs Web)]
        DB[(Banco de Dados - SQLite)]
    end

    UserInput --> CLI
    CLI --> Cerebrum
    Cerebrum --> AgentLoop
    AgentLoop --> Planner
    Planner --> AgentLoop
    AgentLoop -- Raciocínio/Ação --> LLM_I
    LLM_I --> LLM_Server
    LLM_Server --> LLM_I
    LLM_I -- Resposta LLM --> Parser
    Parser --> AgentLoop
    AgentLoop -- Executar Ação --> ToolExec
    ToolExec -- Usar Skill --> SkillReg
    SkillReg -- Executar Skill --> Skills
    Skills -- Resultado --> ToolExec
    AgentLoop -- Orquestrar Fragmento --> FragReg
    FragReg -- Obter/Executar Fragmento --> Fragments
    Fragments -- Usar Skill --> SkillReg
    Fragments -- Resultado --> AgentLoop
    ToolExec -- Resultado Ação --> AgentLoop
    AgentLoop --> Reflector
    Reflector --> MemorySys
    Reflector --> AgentLoop
    AgentLoop --> MemorySys
    MemorySys --> DB
    AgentLoop --> Cerebrum
    Cerebrum -- Resposta Final --> CLI

    Skills --> FileSystem
    Skills --> Shell
    Skills --> Web
    Fragments --> FileSystem
    Fragments --> Shell
    Fragments --> Web

    FragReg --> Config
    SkillReg --> Config
    Cerebrum --> Config
    LLM_I --> Config
    MemorySys --> Config
    Logger --> Config
```

## 3. Componentes Principais (Core - `a3x/core/`)

### 3.1. `CerebrumXAgent` (`cerebrumx.py`)
*   **Responsabilidade:** Ponto de entrada principal e orquestrador de alto nível. Herdando de `ReactAgent`, ele gerencia o ciclo de vida de uma tarefa completa desde a recepção até a resposta final, implementando um ciclo cognitivo:
    1.  **Percepção:** Processa o objetivo inicial.
    2.  **Recuperação de Contexto:** Busca informações relevantes nas memórias semântica (FAISS) e episódica (SQLite).
    3.  **Seleção de Fragmento:** Utiliza o `FragmentRegistry` para selecionar o Fragment (Worker ou Manager) mais apropriado para a tarefa e contexto.
    4.  **Execução do Fragmento:** Delega a execução da tarefa ao método `run_and_optimize` do Fragment selecionado.
    5.  **Reflexão e Aprendizado:** Após a execução do Fragment, chama `_reflect_and_learn` para analisar o resultado e atualizar a memória.
    6.  **Autoavaliação:** Executa `auto_evaluate_task`.
*   **Interações:** Recebe tarefas da interface (e.g., `assistant_cli.py`), inicializa e configura componentes (`MemoryManager`, `FragmentRegistry`), interage com `LLMInterface` (indiretamente via FragmentRegistry ou skills de reflexão), `MemorySystem`, `AgentReflector`.
*   *(Status: Implementa um ciclo cognitivo de alto nível focado na delegação a Fragments, em vez de um loop ReAct passo a passo direto para toda a tarefa.)*

### 3.2. `Agent` (`agent.py` - Classe Base `ReactAgent`)
*   **Responsabilidade:** A classe `ReactAgent` (da qual `CerebrumXAgent` herda) define a lógica para um loop de execução **ReAct (Reasoning + Action)** passo a passo. Esta lógica parece ser destinada a ser usada *dentro* de Fragments ou para tarefas mais simples, enquanto `CerebrumXAgent` usa uma abordagem de delegação de nível superior. As responsabilidades do `ReactAgent` incluem:
    *   Manter o estado da conversa/tarefa atual (histórico).
    *   Construir prompts ReAct para o LLM (usando `PromptBuilder`), incluindo a tarefa/sub-tarefa, histórico, contexto e a lista de **Skills** disponíveis.
    *   Interagir com `LLMInterface` para obter a próxima decisão (pensamento e ação).
    *   Utilizar `AgentParser` para extrair a ação e seus parâmetros da resposta do LLM.
    *   Chamar `ToolExecutor` para executar a **Skill** identificada.
    *   Processar o resultado da ação (Observação).
    *   Repetir o ciclo até que a tarefa seja concluída ou ocorra um erro/limite.
*   **Interações:** `LLMInterface`, `AgentParser`, `ToolExecutor`, `SkillRegistry`, `MemorySystem`, `PromptBuilder`.
*   *(Status: Define o loop ReAct padrão. A forma como `CerebrumXAgent` utiliza ou substitui essa lógica precisa ser melhor esclarecida; atualmente, `CerebrumXAgent.run` não chama diretamente `_perform_react_iteration` de `ReactAgent`.)*

### 3.3. `LLMInterface` (`llm_interface.py`)
*   **Responsabilidade:** Abstrai a comunicação com o modelo de linguagem (LLM), independentemente de ser local ou remoto. Envia prompts formatados e recebe respostas textuais brutas. Lida com detalhes de conexão, timeouts e possíveis novas tentativas.
*   **Interações:** Chamado principalmente pelo `Agent`. Comunica-se com o `LLM_Server` externo (e.g., servidor web do llama.cpp). Pode ler configurações de `config.py`.
*   *(Status: Implementação parece focada na interação com um endpoint HTTP. Detalhar como a configuração (URL, headers) é gerenciada.)*

### 3.4. `AgentParser` (`agent_parser.py`)
*   **Responsabilidade:** Pós-processa a resposta textual do LLM para extrair informações estruturadas. Foco principal é identificar e validar o bloco de **Pensamento** e o bloco de **Ação** (incluindo nome da ferramenta/skill/fragment e seus parâmetros).
*   **Interações:** Chamado pelo `Agent` após receber a resposta do `LLMInterface`. Retorna os dados estruturados (pensamento, ação, parâmetros) ou indica falha no parsing.
*   *(Status: Crucial para a robustez do agente. Detalhar o formato de resposta esperado do LLM (JSON? Markdown? Regex?), como lida com erros de formato, e a lógica de validação dos parâmetros.)*

### 3.5. `ToolExecutor` (`tool_executor.py`)
*   **Responsabilidade:** Módulo dedicado a executar **Skills**. Recebe o nome da skill e um dicionário de parâmetros do `Agent`.
    *   Consulta o `SkillRegistry` para obter a definição e a função correspondente da skill.
    *   Valida os parâmetros recebidos contra a definição da skill.
    *   Executa a função da skill, passando o contexto necessário (`ctx`) e os parâmetros validados.
    *   Retorna o resultado (ou erro) da execução da skill para o `Agent`.
*   **Interações:** Chamado pelo `Agent`. Interage com o `SkillRegistry`. Executa funções definidas em `a3x/skills/`.
*   *(Status: Papel bem definido. Detalhar como o contexto (`ctx`) é construído e passado, e o formato esperado de retorno das skills.)*

### 3.6. `FragmentRegistry` (`a3x/fragments/registry.py`)
*   **Responsabilidade:** (Refatorado) Classe centralizada que gerencia todo o ciclo de vida dos **Fragments**:
    *   **Descoberta Automática:** Durante a inicialização, utiliza `pkgutil.walk_packages` para encontrar todos os módulos dentro do pacote `a3x.fragments`. Inspeciona cada módulo em busca de classes que herdem de `BaseFragment` ou `ManagerFragment` (excluindo as classes base em si).
    *   **Extração de Metadados:** Para cada classe Fragment descoberta, extrai automaticamente metadados como:
        *   `name`: Usa o atributo de classe `FRAGMENT_NAME` se definido, senão o nome da classe.
        *   `description`: Usa a primeira linha do docstring da classe.
        *   `category`: Define como "Management" (se herda de `ManagerFragment`) ou "Execution".
        *   `skills`/`managed_skills`: Usa atributos de classe como `DEFAULT_SKILLS` ou `MANAGED_SKILLS` (convenção).
    *   **Registro:** Armazena a definição (`FragmentDef`, contendo os metadados e a própria classe) e a classe do fragment em dicionários internos.
    *   **Disponibilização:** Fornece um método (`get_available_fragments_description`) que formata as definições de todos os fragments registrados para serem incluídas no prompt do LLM (usado pelo Orchestrator para selecionar o fragment).
    *   **Instanciação Sob Demanda:** Cria instâncias dos fragments apenas quando solicitados (`load_fragment`, chamado por `get_fragment`). Durante a instanciação, injeta dependências necessárias (passadas ao construtor do `FragmentRegistry`), como `LLMInterface`, `SkillRegistry` e configurações específicas do fragment.
    *   **Acesso:** O método `get_fragment(name)` retorna a instância cacheada ou carrega uma nova.
*   **Interações:** Usado pelo `Agent`/`CerebrumXAgent` para obter descrições e instâncias de fragments. Interage com os módulos em `a3x/fragments/`. Recebe `LLMInterface`, `SkillRegistry`, `Config` como dependências em seu construtor.
*   *(Status: Refatoração implementada. Encapsula toda a lógica de gerenciamento de fragments, promovendo descoberta automática e instanciação controlada.)*

### 3.7. `SkillRegistry` (`skills.py` e `a3x/skills/`)
*   **Responsabilidade:** Coleção global (provavelmente um dicionário singleton ou um módulo atuando como registro) que armazena as definições e as funções executáveis de todas as **Skills** disponíveis.
    *   **Descoberta Automática:** A descoberta e registro são feitos através do decorator `@skill` (definido em `a3x/core/tools.py`). Este decorator é aplicado às funções que implementam as skills, localizadas nos módulos dentro de `a3x/skills/` e seus subdiretórios.
    *   **Registro via Decorator:** Quando um módulo de skill é importado (o que é garantido pela estrutura de `__init__.py` nos diretórios `skills` e seus subdiretórios que usam `pkgutil`), o decorator `@skill` é executado. Ele recebe os metadados da skill e registra a função decorada junto com esses metadados no registro central.
    *   **Metadados do Decorator:** O decorator `@skill` espera argumentos como:
        *   `name` (str): Nome único da skill (usado pelo LLM e `ToolExecutor`).
        *   `description` (str): Descrição clara para o LLM entender o propósito da skill.
        *   `parameters` (Dict[str, tuple]): Definição dos parâmetros da função (`{nome_param: (tipo, default_ou_ellipsis)}`). Ellipsis (`...`) indica parâmetro obrigatório.
    *   **Acesso:**
        *   `ToolExecutor`: Consulta o registro pelo `name` da skill para obter a função Python correspondente a ser executada.
        *   `Agent`/`PromptBuilder`: Consulta o registro (via `get_skill_descriptions`) para obter a lista formatada de skills (nomes, descrições, parâmetros) a ser incluída no prompt do LLM.
*   **Interações:** Populado pelos decorators `@skill` durante a importação dos módulos de skills. Consultado pelo `ToolExecutor` e `Agent`/`PromptBuilder`.
*   *(Status: Mecanismo claro e comum para registro de ferramentas/plugins baseado em decorators e descoberta automática via imports.)*

### 3.8. `MemorySystem` (`a3x/core/memory/`, `a3x/memory/`, `memory.db`)
*   **Responsabilidade:** Gerencia a persistência de informações e aprendizados do agente, abstraído pela classe `MemoryManager` (`a3x/core/memory/memory_manager.py`). Inclui:
    *   **Memória de Curto Prazo:** Histórico da conversa/tarefa atual, geralmente mantido em memória pelo `Agent` durante a execução.
    *   **Memória Episódica:** Armazena um log sequencial de eventos (ações, observações, contextos) em um banco de dados SQLite (`memory.db`). A lógica de acesso (adicionar, recuperar recentes) está em `a3x/core/db_utils.py` e é exposta via `MemoryManager`.
    *   **Memória Semântica:** Armazena informações associadas a embeddings vetoriais para busca de similaridade. Utiliza:
        *   Um índice **FAISS** (`.index`) para busca rápida de vetores.
        *   Um arquivo **JSONL** (`.index.jsonl`) paralelo que armazena os metadados associados a cada vetor no índice FAISS. A lógica de interação com FAISS e o JSONL está em `a3x/core/semantic_memory_backend.py`, exposta via `MemoryManager`.
*   **Interações:**
    *   `MemoryManager` é usado pelo `Agent` (ou `CerebrumXAgent`) para recuperar contexto (semântico e episódico) no início do ciclo e para registrar eventos episódicos.
    *   `AgentReflector` pode consultar a memória para tomar decisões, mas não parece ser o principal responsável por *escrever* aprendizados estruturados.
    *   Módulos de indexação (`semantic_indexer*.py`) são responsáveis por popular a memória semântica.
*   *(Status: Arquitetura de memória combina SQLite para logs sequenciais/episódicos e FAISS+JSONL para busca semântica. `MemoryManager` atua como fachada.)*

### 3.9. `AgentReflector` (`agent_reflector.py`)
*   **Responsabilidade:** Módulo focado na **análise pós-ação** e no controle de fluxo. A função `reflect_on_observation` recebe o resultado da última ação executada e decide qual deve ser o próximo passo do agente principal.
    *   Analisa o `status` da observação ("success", "error", "no_change").
    *   Decide entre: `continue_plan`, `retry_step` (e.g., para falhas de LLM, onde pode tentar chamar `adjust_llm_parameters`), `stop_plan`, `plan_complete`.
    *   Implementa uma lógica de **auto-correção** específica para falhas da skill `execute_code`, onde tenta usar o próprio agente recursivamente para corrigir o código com erro.
*   **Interações:** Chamado pelo `Agent` (ou ciclo principal) após cada observação. Recebe o estado atual (objetivo, plano, histórico, memória). Pode chamar skills (como `adjust_llm_parameters`) ou até mesmo o `Agent` recursivamente para meta-tarefas (como a correção de código).
*   *(Status: Focado no controle de fluxo reativo pós-observação e em mecanismos de recuperação de erro específicos, como a auto-correção de código. Não parece ser o local primário para armazenamento de aprendizados complexos ou geração de heurísticas de longo prazo, que podem ocorrer em outros módulos ou processos.)*

### 3.10. Outros Componentes Core (`config.py`, `logging_config.py`, etc.)
*   **`config.py`:** Centraliza a leitura de variáveis de ambiente (`.env`) e a definição de constantes e caminhos usados em todo o projeto (e.g., `PROJECT_ROOT`, URLs de API, configurações de LLM).
*   **`logging_config.py`:** Configura o sistema de logging (níveis, formato, handlers - console, arquivo).
*   **`prompt_builder.py`:** Utilitário/classe responsável por montar dinamicamente os prompts complexos enviados ao LLM, combinando instruções, histórico, ferramentas, etc.
*   **`planner.py`:** (Potencialmente) Responsável por gerar planos de alto nível quando uma abordagem ReAct passo a passo não é suficiente.
*   **`self_optimizer.py`, `finetune_pipeline.py`:** Componentes relacionados ao aprendizado contínuo e fine-tuning local dos modelos/LoRAs.

## 4. Módulos de Capacidade

### 4.1. Fragments (`a3x/fragments/`)
*   **Conceito:** Unidades de trabalho de nível superior que encapsulam lógica de domínio ou orquestração de múltiplas skills. Representam "workers" ou "managers" que o `Agent` (ou `CerebrumXAgent`) pode delegar tarefas complexas. São projetados para serem reutilizáveis e potencialmente auto-otimizáveis.
*   **Implementação:** Classes Python localizadas em `a3x/fragments/` (e subdiretórios) que herdam de:
    *   `BaseFragment` (`a3x/fragments/base.py`): Classe abstrata base para todos os fragments. Define a interface comum, incluindo o gerenciamento de estado e métricas através de `FragmentState` (de `a3x/core/self_optimizer.py`) e o método wrapper `run_and_optimize`.
    *   `ManagerFragment` (`a3x/fragments/base.py`): Subclasse de `BaseFragment` destinada a fragments que **coordenam** outras skills ou fragments dentro de um domínio específico. Recebem uma sub-tarefa do orquestrador e devem implementar a lógica `coordinate_execution` para delegar o trabalho.
*   **Execução:** Um fragment é selecionado pelo `CerebrumXAgent` via `FragmentRegistry`. O método `run_and_optimize` é chamado, que por sua vez executa a lógica principal do fragment (seja um loop ReAct interno, a lógica `coordinate_execution` de um Manager, ou outra implementação) e lida com a coleta de métricas e otimização (se configurada).
*   **Diferença de Skills:** Fragments são classes com estado potencial, podem orquestrar múltiplas skills, implementar lógicas complexas e ter mecanismos de auto-otimização associados. Skills são funções atômicas e sem estado (na maioria dos casos).

### 4.2. Skills (`a3x/skills/`)
*   **Conceito:** As ferramentas fundamentais, atômicas e (idealmente) sem estado que o agente utiliza para interagir diretamente com o ambiente (sistema de arquivos, shell, web, APIs, etc.). Cada skill realiza uma ação específica e bem definida.
*   **Implementação:** Funções Python (geralmente `async def`) decoradas com `@skill` (definido em `a3x/core/tools.py`?). Localizadas em módulos dentro de `a3x/skills/` e seus subdiretórios. O decorator lida com o registro da skill (nome, descrição, parâmetros) no `SkillRegistry`.
*   **Execução:** São chamadas diretamente pelo `ToolExecutor` quando selecionadas pelo `Agent` (ou por um `Fragment` que precise usar uma ferramenta).
*   **Descoberta:** O `SkillRegistry` é populado automaticamente quando os módulos de skills são importados, graças à estrutura de `__init__.py`.

## 5. Fluxo de Execução Típico (Simplificado)

1.  **Entrada:** Usuário fornece tarefa via CLI (`assistant_cli.py`).
2.  **Recepção:** `CerebrumXAgent` recebe a tarefa, inicializa o `Agent` e o contexto.
3.  **Loop Cognitivo (`Agent`):**
    a.  `PromptBuilder` monta o prompt com tarefa, histórico, lista de Skills (do `SkillRegistry`) e Fragments (do `FragmentRegistry`).
    b.  `LLMInterface` envia o prompt ao LLM e recebe a resposta.
    c.  `AgentParser` extrai `thought` e `action` (nome da ferramenta + parâmetros).
    d.  **Seleção de Ferramenta:**
        *   Se `action` corresponde a uma **Skill**: `Agent` chama `ToolExecutor(skill_name, params)`. `ToolExecutor` usa `SkillRegistry` para encontrar e executar a função da skill.
        *   Se `action` corresponde a um **Fragment**: `Agent` chama `FragmentRegistry.get_fragment(fragment_name)` para obter a instância e então chama um método de execução no fragment (e.g., `fragment.execute(params)`). O fragment pode usar o `SkillRegistry` ou `ToolExecutor` internamente.
        *   Se `action` for **"Final Answer"**: O loop termina.
    e.  `Agent` recebe o resultado da execução (da Skill ou Fragment).
    f.  Atualiza o histórico.
    g.  Repete o ciclo (a-g) até "Final Answer" ou erro/limite.
4.  **Reflexão/Aprendizado (`AgentReflector`, `MemorySystem`):** Após a conclusão/falha, `AgentReflector` analisa o resultado e atualiza o `MemorySystem`.
5.  **Saída:** `CerebrumXAgent` formata a resposta final e a envia para a CLI.

## 6. Validação e Testes

*   **Framework:** Os testes são implementados usando `pytest`.
*   **Estrutura:** O diretório `tests/` contém os arquivos de teste. `tests/conftest.py` define fixtures reutilizáveis.
*   **Fixtures (`conftest.py`):
    *   **Mocks:** São fornecidas fixtures para mockar os principais componentes do core (`mock_db`, `mock_llm_interface`, `mock_planner`, etc.) e instâncias de agentes totalmente mockadas (`agent_instance`, `cerebrumx_agent_instance`). Isso permite testes unitários focados e testes de fluxo sem dependências externas.
    *   **Workspace Temporário (`temp_workspace_files`):** Cria um diretório temporário gerenciado pelo pytest para testes que precisam criar, ler ou modificar arquivos, garantindo isolamento e limpeza.
    *   **Servidor LLM Gerenciado (`managed_llama_server_session`):** Fixture crucial com escopo de **sessão**. Ela:
        *   Inicia uma instância real do servidor `llama-server` (usando o binário e o modelo configurados em `conftest.py`) em uma porta específica para testes (e.g., 8081).
        *   Verifica ativamente se o servidor está pronto consultando seu endpoint `/health`.
        *   Disponibiliza (yields) a URL base do servidor para os testes que a requisitarem.
        *   Garante que o processo do servidor seja finalizado ao término da sessão de testes.
        *   Permite testes de integração **end-to-end** que avaliam o comportamento do agente com um LLM real.
*   **Tipos de Testes:**
    *   **Unitários:** (Presumivelmente) Testam funções ou classes isoladas usando mocks.
    *   **Integração:** Testam a interação entre múltiplos componentes. Podem usar mocks parciais ou fixtures reais como `managed_llama_server_session` e `temp_workspace_files` para simular cenários mais realistas (e.g., `tests/test_integration_cli.py`).
*   **(Status: A estrutura de testes utiliza `pytest` e fixtures para permitir tanto testes isolados com mocks quanto testes de integração mais completos com um servidor LLM real gerenciado. O papel exato do `A3XValidator` mencionado anteriormente não está claro a partir do `conftest.py` e pode ser uma ferramenta separada ou parte de testes específicos.)**

--- 