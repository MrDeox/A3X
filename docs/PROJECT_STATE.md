# Resumo Detalhado do Projeto A³X

## 1. Objetivo Principal:
Construir um agente autônomo (A³X) capaz de:
*   **Compreender:** Interpretar tarefas complexas em linguagem natural.
*   **Planejar:** Decompor tarefas em passos executáveis.
*   **Agir:** Utilizar ferramentas (leitura/escrita de arquivos, execução de código/comandos, busca na web) para completar as tarefas.
*   **Aprender:** Adaptar-se a novas informações, ferramentas e feedback, melhorando seu desempenho ao longo do tempo (meta-aprendizado).
*   **Evoluir:** Eventualmente, modificar e melhorar seu próprio código base (auto-programação).

## 2. Arquitetura Atual (Visão Simplificada):

```mermaid
graph TD
    A[Usuário/Arthur] --> B(Interface CLI / assistant_cli.py);
    B --> C{Core Agent / agent.py};
    C --> D[Prompt Builder / prompt_builder.py];
    D --> E{LLM (Local - Llama.cpp Server)};
    E --> C;
    C --> F[Tool Executor / tool_executor.py];
    F --> G{{Ferramentas (Filesystem, Shell, Web)}};
    G --> F;
    F --> C;
    C --> H[Memory / memory.py (Potencial)];
    C --> B;
    B --> A;

    subgraph "Módulos Principais"
        B; C; D; F; H;
    end

    subgraph "Infraestrutura"
        E; G;
    end
```

*   **Interface CLI (`assistant_cli.py`):** Ponto de entrada para interação do usuário.
*   **Core Agent (`agent.py`):** Orquestrador principal, implementa o loop ReAct (Reason + Act).
*   **Prompt Builder (`prompt_builder.py`):** Constrói os prompts enviados ao LLM, incorporando histórico, ferramentas disponíveis e a tarefa atual.
*   **LLM (Local):** Atualmente usando `dolphin-2.2.1-mistral-7b.Q4_K_M.gguf` via `llama-cpp-python` com servidor web. Responsável pela geração de raciocínio e seleção de ações/ferramentas.
*   **Tool Executor (`tool_executor.py`):** Executa as ações/ferramentas solicitadas pelo LLM (ex: `list_files`, `read_file`, `execute_shell`).
*   **Ferramentas:** Módulos específicos que interagem com o sistema (filesystem, shell, etc.).
*   **Memory (`memory.py`):** (Ainda incipiente/planejado) Para persistência de estado e aprendizado de longo prazo.

## 3. Componentes Chave e Funcionalidades:
*   **Loop ReAct:** O agente raciocina sobre a tarefa, escolhe uma ferramenta/ação, a executa, observa o resultado e repete até completar a tarefa.
*   **Gerenciamento de Ferramentas:** O agente sabe quais ferramentas estão disponíveis e como usá-las (descrições passadas no prompt).
*   **LLM Local:** Permite experimentação rápida e controle total sobre o modelo, evitando custos e dependência de APIs externas.
*   **Fixture Pytest (`managed_llama_server`):** Gerencia o ciclo de vida do servidor Llama.cpp para testes de integração, garantindo que o servidor esteja pronto antes dos testes e seja desligado depois.
*   **Formato de Saída LLM:** Espera-se que o LLM retorne JSON estruturado contendo `pensamento` (reasoning) e `acao` (action + parameters).

## 4. Estado Atual (Pontos Relevantes):
*   **Testes Unitários/Integração:**
    *   Testes unitários básicos para algumas ferramentas existem.
    *   Um teste de integração (`test_react_agent_run_list_files` em `test_run_basic.py`) foi criado com sucesso usando mocks para simular o LLM e ferramentas.
    *   Um teste de integração *real* (`test_react_list_files` em `test_integration_cli.py`) foi adaptado para usar o LLM real via `managed_llama_server`.
*   **Falha Recente:** O teste `test_react_list_files` FALHOU.
    *   **Causa Raiz:** O LLM (dolphin-mistral) não retornou um JSON válido em sua primeira resposta quando solicitado a listar arquivos `.py`. Ele retornou texto não-estruturado, causando um erro de parsing no `agent.py`.
    *   **Observação:** A `managed_llama_server` funcionou perfeitamente, iniciando e parando o servidor conforme esperado.
*   **Prompt Engineering:** O prompt atual (`prompt_builder.py`) precisa ser refinado para instruir o LLM de forma mais robusta a *sempre* retornar JSON válido, mesmo que a resposta seja simples ou um erro.
*   **Parsing de Resposta:** O parser no `agent.py` é atualmente sensível a JSON malformado. Poderia ser tornado mais robusto (tentar extrair JSON, lidar com respostas parciais ou não-JSON).
*   **Modelo LLM:** O `dolphin-2.2.1-mistral-7b.Q4_K_M.gguf` pode não ser o ideal para seguir instruções de formato estritas como JSON. Outros modelos ou ajustes de parâmetros (temperatura, etc.) podem ser necessários.

## 5. Próximos Passos Imediatos (Sugeridos):
1.  **Corrigir a Falha do Teste `test_react_list_files`:**
    *   **Prioridade:** Focar em **Prompt Engineering** no `prompt_builder.py`. Modificar o prompt para enfatizar *criticamente* a necessidade de resposta JSON, talvez incluindo exemplos no próprio prompt.
    *   **Considerar:** Tornar o parsing da resposta do LLM no `agent.py` mais robusto como medida secundária.
    *   **Testar:** Re-executar o teste `test_react_list_files` após as modificações no prompt.
2.  **Expandir Cobertura de Testes de Integração:** Uma vez que `list_files` funcione com o LLM real, adaptar outros testes de `test_integration_cli.py` (ex: `read_file`, `search_web`) para usar a `managed_llama_server`.
3.  **Refinar `managed_llama_server`:** Lidar com o warning de `asyncio_default_fixture_loop_scope`.

## 6. Visão de Longo Prazo (Lembretes):
*   Implementar memória de longo prazo.
*   Desenvolver capacidades de meta-aprendizado.
*   Explorar mecanismos de auto-programação e auto-correção.
*   Melhorar a robustez e a capacidade de lidar com ambiguidades e erros.
