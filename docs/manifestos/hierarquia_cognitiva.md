# Manifesto da Hierarquia Cognitiva em Pirâmide (A³X)

**Ideia Central:** Estruturar a arquitetura de tomada de decisão e execução da A³X em uma hierarquia piramidal, com níveis distintos de abstração e responsabilidade: Estrategistas no topo, Coordenadores/Gerentes no meio, e Especialistas/Executores na base.

**Filosofia Fundadora:**

A inteligência complexa, seja em sistemas biológicos, organizações humanas ou inteligência artificial, raramente é monolítica. Ela emerge da interação organizada entre diferentes níveis de processamento e controle. Inspiramo-nos em estruturas hierárquicas comprovadas, onde a visão estratégica define a direção, a gestão tática coordena os recursos e a execução especializada realiza as tarefas concretas. A Pirâmide Cognitiva da A³X visa replicar essa eficiência organizacional, permitindo que a estratégia guie a execução detalhada, enquanto a informação da base informa as decisões superiores.

**Níveis da Pirâmide:**

1.  **Topo: Orquestrador (Estrategista)**
    *   **Responsabilidade:** Visão geral do objetivo final, planejamento estratégico de alto nível, decomposição inicial da tarefa, seleção e delegação de subtarefas para o nível apropriado abaixo (Managers ou Executores diretos).
    *   **Foco:** "O quê?" e "Por quê?". Define a intenção e a sequência geral.
    *   **Interação:** Recebe o objetivo do usuário/sistema externo. Delega para Managers ou Executores. Recebe status e resultados consolidados para decidir o próximo passo estratégico ou finalizar a tarefa. Utiliza o histórico geral e o `SharedTaskContext` para manter a visão global.

2.  **Meio: Managers de Domínio (Tático/Coordenador)**
    *   **Responsabilidade:** Gerenciar um conjunto específico de skills relacionadas a um domínio (ex: `FileOpsManager`, `CodeExecutionManager`). Coordenar a execução de múltiplas ações de baixo nível dentro desse domínio para cumprir a subtarefa delegada pelo Orquestrador. Pode realizar alguma decomposição tática adicional.
    *   **Foco:** "Como fazer *esta parte* eficientemente?". Coordena a execução dentro de um domínio.
    *   **Interação:** Recebe subtarefas do Orquestrador. Seleciona e invoca as skills específicas sob sua gestão (na base). Pode interagir com o `SharedTaskContext` para obter/armazenar dados relevantes ao seu domínio. Reporta o resultado consolidado da sua subtarefa ao Orquestrador.

3.  **Base: Fragmentos Executores / Skills (Especialista/Executor)**
    *   **Responsabilidade:** Realizar ações atômicas e altamente especializadas. Executar uma única skill ou uma sequência curta e bem definida de skills para completar uma tarefa muito específica delegada diretamente pelo Orquestrador (para tarefas simples) ou por um Manager.
    *   **Foco:** "Faça *esta ação específica* agora". Execução concreta.
    *   **Interação:** Recebe instruções diretas do Orquestrador ou de um Manager. Executa a(s) skill(s) necessária(s). Pode ler/escrever informações muito específicas no `SharedTaskContext`. Reporta o resultado direto da sua ação/subtarefa ao seu invocador (Orquestrador ou Manager).

**Fluxo de Informação:**

*   **Cima para Baixo (Delegação):** O Orquestrador decompõe o objetivo e delega subtarefas aos Managers ou Executores. Os Managers podem decompor ainda mais e delegar para as Skills na base.
*   **Baixo para Cima (Resultados e Contexto):** Skills retornam resultados diretos. Executores e Managers consolidam esses resultados e os retornam para o nível acima. Informações relevantes podem ser adicionadas ao `SharedTaskContext` em qualquer nível para serem potencialmente usadas por outros componentes ou pelo Orquestrador em ciclos futuros.

**Benefícios da Hierarquia:**

*   **Clareza de Responsabilidades:** Cada nível tem um papel bem definido.
*   **Abstração:** Níveis superiores lidam com conceitos mais abstratos, sem precisar se preocupar com os detalhes da execução de baixo nível.
*   **Foco Aprimorado:** Componentes em cada nível mantêm um foco cognitivo mais estreito (contexto mínimo viável).
*   **Reutilização:** Managers e Skills/Executores podem ser reutilizados em diferentes tarefas coordenadas pelo Orquestrador.
*   **Escalabilidade Cognitiva:** Permite ao agente lidar com problemas progressivamente mais complexos adicionando mais especialização na base ou coordenação no meio, sem sobrecarregar o estrategista no topo.

**Conexão com a Fragmentação Cognitiva:**

A Hierarquia Cognitiva fornece a **estrutura organizacional** para os componentes criados pela Fragmentação Cognitiva. Os "Fragmentos" do manifesto anterior podem ser vistos como os componentes nos níveis de Manager e Executor/Skill desta pirâmide. A fragmentação cria os especialistas, e a hierarquia os organiza para colaboração eficiente.

**Conclusão:**

A Hierarquia Cognitiva em Pirâmide é um modelo organizacional fundamental para a A³X. Ela estrutura a colaboração entre componentes especializados (Fragmentos), permitindo que o agente combine planejamento estratégico com execução tática e especializada de forma eficiente, escalável e evolutiva, espelhando a organização da inteligência em sistemas complexos bem-sucedidos. 