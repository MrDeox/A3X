# 1.0 Introdução e Objetivo do Documento

Este arquivo documenta observações sobre a arquitetura, fluxo de dados e comportamento do projeto A³X, feitas durante sessões de depuração e desenvolvimento. O objetivo é evitar a introdução de soluções que não se alinham com a estrutura existente.

# 2.0 Componentes Estruturais

## 2.1 Diretórios e Arquivos do Projeto

### 2.1.1 Estrutura de Diretórios Principal (`/home/arthur/projects/A3X/`)

- **`a3x/`**: Contém o código-fonte principal do projeto.
  - **`cli/`**: Lógica da interface de linha de comando (`interface.py`, `display.py`).
  - **`core/`**: Componentes centrais do agente (agente ReAct, CerebrumX, execução de ferramentas, gerenciamento de skills, LLM, memória, etc.).
  - **`skills/`**: Módulos individuais contendo as skills (ferramentas) que o agente pode usar. Organizado por categorias (e.g., `core`, `memory`, `file_manager`).
  - **`training/`**: Código relacionado ao fine-tuning (e.g., `trainer.py`).
  - **`prompts/`**: Arquivos contendo templates de prompts (e.g., `react_system_prompt.md`).
  - **`memory/`**: Diretório padrão para arquivos relacionados à memória persistente (e.g., banco de dados `memory.db`, logs de aprendizado `learned_heuristics.jsonl`).
  - **`servers/`**: (Presumivelmente) Scripts ou configurações relacionadas a servidores externos.
  - **`__init__.py`**: Torna `a3x` um pacote Python.
- **`a3x.egg-info/`**: Metadados da instalação do pacote (gerado pelo `setuptools`).
- **`llama.cpp/`**: Submódulo ou cópia do repositório `llama.cpp`.
- **`stable-diffusion-webui/`**: Submódulo ou cópia do repositório `stable-diffusion-webui`.
- **`tools/`**: Ferramentas de terceiros (e.g., `piper` para TTS).
- **`models/`**: Diretório padrão para modelos de linguagem (e.g., GGUF) e outros modelos (e.g., `yolov8n.pt`).
- **`memory.db`**: Banco de dados SQLite principal usado pela memória do agente.
- **`logs/`**: Diretório para logs gerais da aplicação.
- **`tests/`**: Testes automatizados.
- **`scripts/`**: Scripts auxiliares diversos.
- **`pyproject.toml`, `requirements.txt`**: Arquivos de gerenciamento de dependências e projeto Python.
- **`README.md`, `project_structure_notes.md`**: Documentação.
- **`.gitignore`, `.git/`**: Arquivos de controle de versão Git.

## 2.2 Configuração e Logging

### 2.2.1 Configuração Central (`a3x/core/config.py`)

Este arquivo define constantes e configurações para todo o projeto, utilizando variáveis de ambiente (`.env`) com valores padrão.

- **`PROJECT_ROOT`:** Path raiz do projeto.
- **LLM:** URL do servidor (`LLAMA_SERVER_URL`), chave de API (opcional), cabeçalhos, limite de tokens (`MAX_TOKENS_FALLBACK`), tamanho do contexto (`CONTEXT_SIZE`).
- **Agente:** Limite de iterações ReAct (`MAX_REACT_ITERATIONS`), limite de turnos de histórico (`MAX_HISTORY_TURNS`).
- **Execução de Código:** Timeout (`PYTHON_EXEC_TIMEOUT`).
- **Memória:** Caminho do DB (`MEMORY_DB_PATH`), K para busca semântica (`SEMANTIC_SEARCH_TOP_K`), limite para busca episódica (`EPISODIC_RETRIEVAL_LIMIT`).
- **Logs:** Nível (`LOG_LEVEL`), formato, diretórios para logs de reflexão e aprendizado, nome do arquivo de heurísticas (`HEURISTIC_LOG_FILE`), arquivo de log dos servidores (`SERVER_LOG_FILE`).
- **Treinamento:** Parâmetros para QLoRA (modelo base, R, alpha, diretório de saída, etc.).
- **Gerenciador de Servidor:** Caminhos para binários/diretórios (Llama.cpp, SD WebUI), hosts, portas, argumentos de linha de comando, timeouts de inicialização.

### 2.2.2 Configuração de Logging (`a3x/core/logging_config.py`)

- **`setup_logging()`:**
    - Chamada na inicialização da CLI.
    - Configura o logger raiz (`logging.basicConfig`) para enviar logs para o **console**.
    - Usa `LOG_LEVEL`, `LOG_FORMAT`, `LOG_DATE_FORMAT` de `config.py`.
    - **Nota:** A configuração do log para arquivo (`SERVER_LOG_FILE`) parece ser feita separadamente em `server_manager.py`.

### 2.2.3 Gerenciador de Servidores (`a3x/core/server_manager.py`)

Este módulo gerencia os processos dos servidores externos (Llama.cpp, SD WebUI API).

- **`managed_processes`:** Dicionário global que rastreia os processos (`asyncio.subprocess.Process`) iniciados.
- **Logging:** Configura um logger dedicado (`A3XServerManager`) que escreve em `logs/servers.log` e no console (ver seção 2.2.2).
- **`_check_server_ready(name, url, timeout)`:** Verifica periodicamente se um endpoint HTTP (`url`) está respondendo com status 200, até o `timeout`.
- **`_start_process(name, cmd_list, cwd, ready_url, ready_timeout)`:**
    - Função genérica para iniciar um processo.
    - Verifica se o `ready_url` já está respondendo (servidor já ativo).
    - Inicia o processo com `asyncio.create_subprocess_exec`.
    - Captura e loga stdout/stderr do processo usando `_log_stream`.
    - Espera o `ready_url` ficar disponível usando `_check_server_ready`.
    - Trata erros de inicialização e timeouts.
- **`start_llama_server()` / `start_sd_server()`:**
    - Funções específicas que verificam pré-requisitos (existência de binários, modelos, scripts).
    - Chamam `_start_process` com as configurações apropriadas (comandos, CWDs, URLs de verificação) vindas de `config.py`.
- **`stop_server(name)` / `stop_all_servers()`:**
    - Param os processos gerenciados, tentando `terminate()` (ou `CTRL_C_EVENT` no Windows) primeiro e depois `kill()` se necessário.
- **Uso:** Chamado pela `cli/interface.py` (ver seção 4.4.1) para iniciar os servidores automaticamente no início da execução (a menos que `--no-server` seja usado) e garantir o desligamento no final.

# 3.0 Ciclo Cognitivo e Comportamento do Agente

## 3.1 Método `run(objective)` e Ciclo Cognitivo `CerebrumXAgent` (`a3x/core/cerebrumx.py`)

O método `CerebrumXAgent.run(objective)` orquestra o ciclo principal de processamento de uma tarefa:

### 3.1.1 Percepção (`_perceive`)
- Atualmente, apenas encapsula o `objective` inicial.

### 3.1.2 Recuperação de Contexto (`_retrieve_context`)
- Consulta a memória semântica (FAISS via `embeddings.py`/`semantic_memory_backend.py` - ver seção 3.4) usando o embedding do objetivo.
- Consulta a memória episódica (SQLite via `db_utils.py` - ver seção 3.5) para buscar interações recentes.
- Gera um sumário combinado do contexto para o planejador.
- Registra a consulta semântica na memória episódica.

### 3.1.3 Planejamento (`_plan`)
- Usa `a3x.core.planner.generate_plan` para criar um plano (lista de strings representando os passos) com base no objetivo, contexto e skills disponíveis (ver seção 4.2).
- Possui um atalho para gerar planos simples para tarefas de listagem de arquivos (`_is_simple_list_files_task`).

### 3.1.4 Execução do Plano (`_execute_plan`)
- Executa planos simples diretamente chamando `execute_tool` (ver seção 4.1.2).
- Para planos complexos, itera sobre cada passo:
  - Chama `self._perform_react_iteration(step, ...)` (herdado de `a3x.core.agent.ReactAgent` - ver seção 3.3.1). Este método executa o loop ReAct (Thought -> Action -> Observation) para o passo atual, usando o LLM para decidir a próxima ação e `execute_tool` para executar a skill escolhida.
  - **Tratamento de Falha:** Se um passo falhar dentro do loop ReAct, `_execute_plan` chama as skills `reflect_on_failure` e `learn_from_failure_log` imediatamente para analisar e registrar a falha (relacionado à seção 3.6). A execução do *restante do plano é interrompida*.
- Retorna o status final (`completed`, `failed`, `error`), a mensagem final e os resultados de cada passo.

### 3.1.5 Reflexão e Aprendizado (`_reflect_and_learn`)
- Chamado *após* a conclusão de `_execute_plan`.
- Executa a skill `learning_cycle` para uma análise e aprendizado pós-execução geral (sucesso ou falha) (relacionado à seção 3.6).

## 3.2 Planejamento e Execução *(Visão Geral - Conteúdo a ser detalhado)*

*(Esta seção pode consolidar informações sobre como os planos são gerados e como a execução itera sobre eles, conectando `_plan` e `_execute_plan` de forma mais abstrata.)*

## 3.3 Ciclo ReAct

### 3.3.1 `ReactAgent._perform_react_iteration` em `a3x/core/agent.py`

Este método (herdado por `CerebrumXAgent`) executa um único ciclo de Raciocínio e Ação para um passo específico do plano:

1.  **Trim Histórico:** Reduz o histórico de interação (`self._history`) para manter o prompt do LLM conciso (`trim_history`) (ver seção 3.3.2).
2.  **Construir Prompt:** Cria o prompt para o LLM usando `build_react_prompt`, incluindo objetivo do passo, histórico, prompt do sistema e descrições das ferramentas (skills) (ver seção 3.3.2).
3.  **Chamar LLM e Parsear:**
    - Chama o LLM via `_process_llm_response`.
    - `_process_llm_response` usa `a3x.core.llm_interface.call_llm` e depois `a3x.core.agent_parser.parse_llm_response` para extrair `Thought`, `Action` (nome da ferramenta), e `Action Input` (parâmetros) da resposta do LLM.
4.  **Yield Thought:** Se um `Thought` for extraído, adiciona ao histórico e o retorna (`yield`).
5.  **Yield Action / Final Answer (Passo):**
    - Se a ação for `final_answer`, significa que o LLM considera o *passo atual* concluído. Retorna a resposta final do passo (`yield`).
    - Caso contrário, retorna a `Action` (nome da ferramenta) e `Action Input` (parâmetros) (`yield`).
6.  **Executar Ação:** Chama `_execute_action`, que:
    - Normaliza/Valida os parâmetros (`normalize_action_input`).
    - Cria o `_ToolExecutionContext` (ver seção 4.1.1).
    - Chama `a3x.core.tool_executor.execute_tool` para executar a skill real (ver seção 4.1.2).
7.  **Yield Observation:**
    - O resultado da execução da ferramenta (Observation) é adicionado ao histórico via `_handle_observation`.
    - O resultado completo da ferramenta é retornado (`yield`).
8.  **Yield Erro (Opcional):** Se a execução da ferramenta falhou (`status: "error"`), um evento de erro adicional é retornado (`yield`) para sinalizar a falha ao chamador (`_execute_plan`).

### 3.3.2 Uso do Histórico (`self._history`) no Ciclo ReAct

- O atributo `self._history` armazena o histórico completo de Thought, Action, Action Input, Observation e Final Answer ao longo do plano em execução.
- A cada passo do plano, o método `_perform_react_iteration` recebe o histórico completo e o passa para `build_react_prompt`.
- O `build_react_prompt` organiza esse histórico como mensagens alternadas de `assistant` (raciocínio e ação do agente) e `user` (respostas do ambiente).
- Antes da construção do prompt, o histórico é filtrado por `trim_history`, que remove os turnos mais antigos para respeitar o limite `MAX_HISTORY_TURNS` (atualmente 5, definido em `config.py` - seção 2.2.1).
- Esse mecanismo permite que o LLM tenha acesso ao raciocínio e aos resultados dos passos anteriores, influenciando as decisões futuras dentro do mesmo plano.
- Não há limpeza explícita do histórico entre os passos do plano, o que confirma que o raciocínio é acumulativo dentro de uma execução única de `run(objective)`.

## 3.4 Memória Semântica

### 3.4.1 Geração de Embeddings (`get_embedding` - `a3x/core/embeddings.py`)

- **Localização:** `a3x/core/embeddings.py`
- **Retorno:** A função `get_embedding(text: str)` retorna `list[float] | None`, **NÃO** um array NumPy.
- **Importante:** O código que chama `get_embedding` (e.g., `_retrieve_context` na seção 3.1.2) deve tratar o resultado como uma lista. Chamar `.tolist()` no resultado causará um `AttributeError`.
- **Exemplo de Falha:** Código em `cerebrumx.py` tratava o retorno como NumPy e chamava `.tolist()`, causando o erro.
*(Detalhes adicionais sobre o backend FAISS e a consulta podem ser adicionados aqui.)*

## 3.5 Memória Episódica

Esta seção descreve como as interações do agente (objetivo, contexto, plano, passos ReAct, resultados) são armazenadas e recuperadas, formando a memória de curto prazo e a base para o aprendizado a partir da experiência.

### 3.5.1 Armazenamento (`a3x/core/db_utils.py`)

- **Tabela:** A memória episódica reside na tabela `experience_buffer` no banco de dados SQLite (`memory.db`, geralmente na raiz do projeto).
- **Estrutura da Tabela (`experience_buffer`):
    - `id`: Identificador único do registro.
    - `context`: Descrição textual da situação ou prompt que levou à ação (e.g., objetivo do passo, estado anterior).
    - `action`: A ação específica tomada pelo agente (e.g., chamada de skill com parâmetros, resposta final).
    - `outcome`: O resultado da ação (e.g., `success`, `failure`, mensagem de erro, observação da ferramenta).
    - `priority`: Um valor numérico usado para priorizar experiências durante a amostragem para treinamento (inicialmente maior para falhas/erros).
    - `timestamp`: Data e hora do registro.
    - `metadata`: Um campo JSON para armazenar informações adicionais (e.g., skill usada, confiança da resposta do LLM, detalhes do erro).
- **Função de Registro (`add_episodic_record`):
    - Chamada para registrar uma nova "experiência" ou "episódio" na tabela `experience_buffer`.
    - Recebe `context`, `action`, `outcome`, e um dicionário `metadata` opcional.
    - Calcula uma prioridade inicial (maior para falhas).
    - Serializa o `metadata` para JSON antes de inserir.
    - **Onde é chamada?** Presumivelmente chamada após a conclusão de ações significativas ou ciclos cognitivos (e.g., dentro do ciclo ReAct ou após a execução de um plano), embora o ponto exato precise ser confirmado no código do agente (`CerebrumXAgent`, `ReactAgent`).

### 3.5.2 Recuperação (`a3x/core/db_utils.py`)

- **Recuperação por Recência (`retrieve_recent_episodes`):
    - Função principal para recuperar as interações mais recentes.
    - Recebe um parâmetro `limit` (padrão 5).
    - Retorna uma lista das `limit` linhas mais recentes da tabela `experience_buffer`, ordenadas por `timestamp` descendente.
    - **Uso Principal:** Utilizada pelo método `_retrieve_context` do `CerebrumXAgent` (Seção 3.1.2) para buscar interações recentes e construir o contexto para o planejador e o ciclo ReAct.
- **Amostragem para Treinamento (`sample_experiences`):
    - Função usada especificamente pelo processo de fine-tuning (Seção 4.5.1, `a3x/training/trainer.py` -> `prepare_dataset`).
    - Seleciona um `batch_size` de experiências da tabela, potencialmente usando a coluna `priority` para guiar a amostragem (detalhes da lógica de amostragem não visíveis no trecho).

*(Esta seção descreverá como as interações (objetivo, contexto, plano, passos ReAct, resultados) são armazenadas no banco de dados SQLite (`memory.db` via `db_utils.py`) e como são recuperadas em `_retrieve_context` - seção 3.1.2).*

## 3.6 Aprendizado Heurístico e Generalização *(Conteúdo a ser detalhado)*

*(Esta seção abordará as skills `reflect_on_failure`, `learn_from_failure_log`, `learning_cycle`, `auto_generalize_heuristics`, `consolidate_heuristics`, o arquivo `learned_heuristics.jsonl`, e como o agente aprende com sucessos e falhas.)*

# 4.0 Mecanismos Auxiliares e Internos

## 4.1 Executor de Ferramentas (`a3x/core/tool_executor.py`)

### 4.1.1 Contexto de Execução de Ferramentas (`_ToolExecutionContext`)

- **Definição:** Definido em `a3x/core/tool_executor.py` como um `namedtuple`.
- **Campos (a partir de 12/04/2025):** `logger`, `workspace_root`, `llm_url`.
- **Uso:** Passa informações essenciais (logger, workspace, URL do LLM) para as skills quando elas são executadas via `execute_tool` no ciclo ReAct (seção 3.3.1).
- **Importante:** Por ser um `namedtuple`, é **imutável** após a criação. Novos campos precisam ser adicionados à definição no `tool_executor.py`, e todas as instâncias de criação devem ser atualizadas para fornecer os novos valores.
- **Exemplo de Falha:** Tentar adicionar `llm_url` via `setattr` a uma instância existente causou `AttributeError`.

### 4.1.2 Função `execute_tool`

A função `execute_tool(tool_name, action_input, tools_dict, context)` é responsável por executar a skill escolhida pelo agente no ciclo ReAct (seção 3.3.1):

1.  **Validação:** Verifica se a `tool_name` existe no `SKILL_REGISTRY` (`tools_dict` - ver seção 4.2.2) e se a skill é válida/chamável.
2.  **Detecção de Tipo:** Usa `inspect` para determinar se a skill é uma função standalone ou um método de uma classe.
3.  **Instanciação (se Método):** Se for um método, instancia a classe correspondente, tentando passar `workspace_root` do `context` para o `__init__` da classe.
4.  **Preparação do Chamável:** Define o `executable_callable` como a função ou o método ligado à instância.
5.  **Validação de Argumentos:**
    - **Schema Pydantic:** Se um schema Pydantic foi definido para a skill no registro (ver seção 4.2.3), valida o `action_input` (parâmetros do LLM) contra ele. Retorna erro na falha.
    - **Sem Schema:** Usa o `action_input` bruto.
6.  **População de Argumentos:** Constrói o dicionário `call_args` para a chamada da skill:
    - Ignora parâmetros de decoradores conhecidos (e.g., `resolved_path` injetado por validadores - seção 4.3.1).
    - Injeciona valores do `context` (`workspace_root`, `logger`, `ctx`) se solicitados pela assinatura da skill.
    - Mapeia os parâmetros restantes do `action_input` validado.
    - Respeita valores padrão definidos na assinatura da skill.
7.  **Execução:**
    - Detecta se a skill é `async` ou síncrona.
    - Executa a skill usando `await` diretamente (para `async`) ou `run_in_executor` (para síncrona) com os `call_args` preparados.
8.  **Processamento do Resultado:**
    - Se a skill retorna algo que não é um `dict`, encapsula o resultado em um dicionário padrão de sucesso.
    - Se retorna um `dict`, retorna-o como está.
9.  **Tratamento de Erros:** Captura `TypeError` e outras exceções durante a execução, retornando dicionários de erro padronizados.

### 4.1.3 Parsers e Estados Relacionados

- **`ReactAgentOutputParser`:** Classe Pydantic (localização a confirmar, possivelmente `a3x/core/agent_parser.py`) para validar e parsear a saída JSON do LLM (pensamento + ação) usada em `_process_llm_response` (seção 3.3.1).
- **`ReactState`:** Enumeração (localização a confirmar, possivelmente `a3x/core/agent.py`) para os estados do ciclo ReAct (THINKING, ACTING, ANSWERING, ERROR).

## 4.2 Carregamento e Registro de Skills

### 4.2.1 Mecanismo Geral (`a3x/core/skills.py`)

Este módulo define como as skills são registradas e descobertas. Skills são descobertas e registradas quando seus módulos são importados.

### 4.2.2 `SKILL_REGISTRY`

- Dicionário global (importado de `a3x.core.skill_management`) que armazena as informações das skills carregadas.
- Chave: nome da skill. Valor: `dict` com `function`, `description`, `parameters`, `schema` Pydantic, etc.

### 4.2.3 Decorador `@skill`

- Usado para marcar uma função/método como skill.
- Recebe `name`, `description`, `parameters` (`dict` `{'nome': (tipo, default)}`).
- **Valida** a assinatura da função contra os parâmetros declarados.
- **Gera um schema Pydantic** dinamicamente para validação de input (usado em `execute_tool` - seção 4.1.2).
- **Registra** a skill no `SKILL_REGISTRY`.
- **Formato `parameters`:** Exige um **dicionário** onde as chaves são nomes de parâmetros e os valores são **tuplas** `(tipo, default_ou_ellipsis)`.
    - Exemplo: `parameters={'param1': (str, ...), 'param2': (int, 0)}`.
    - `...` (Ellipsis) indica parâmetro obrigatório.
- **Importante:** Usar um formato incorreto no decorator `@skill` impede o registro da skill.
- **Exemplo de Falha:** Skills `auto_generalize_heuristics` e `consolidate_heuristics` usavam um formato de lista para `parameters`, impedindo seu registro.

### 4.2.4 `load_skills(package_name)`

- Função crucial chamada na inicialização da aplicação (e.g., em `cli/interface.py` - seção 4.4.1).
- Usa `pkgutil.walk_packages` para encontrar todos os submódulos dentro do `package_name` (e.g., "a3x.skills").
- **Importa ou Recarrega** cada submódulo encontrado (tipicamente via `import` no arquivo `__init__.py` do pacote correspondente, e.g., `a3x/skills/core/__init__.py`).
- A importação/recarga executa o código do módulo, incluindo os decoradores `@skill`, populando assim o `SKILL_REGISTRY`.
- **Importante:** Se uma skill não for importada (e.g., faltando no `__init__.py`), ela não será registrada e chamadas a ela via `execute_tool` falharão com erro "Tool ... does not exist".
- **Exemplo de Falha:** Skills `auto_generalize_heuristics` e `consolidate_heuristics` não estavam sendo importadas no `a3x/skills/core/__init__.py`, impedindo seu registro.

### 4.2.5 Helpers

- `get_skill_registry()`: Retorna o registro.
- `get_skill_descriptions()`: Gera uma string formatada com a descrição de todas as skills (usada no prompt do LLM - seção 3.3.1).
- `get_skill(name)`: Obtém informações de uma skill específica.

## 4.3 Segurança de Execução

### 4.3.1 Validadores (`a3x/core/validators.py`)

Contém decoradores para validar argumentos de entrada das skills.

- **`@validate_workspace_path(arg_name, check_existence, target_type, ...)`:**
    - Decorador essencial para skills que manipulam arquivos/diretórios.
    - **Extração:** Obtém o caminho (`path_str`) do argumento especificado (`arg_name`) da função decorada.
    - **Resolução:** Resolve o `path_str` para um caminho absoluto (`Path`).
    - **Validação de Segurança:** Garante que o caminho absoluto resolvido esteja **dentro** do `workspace_root` definido na instância da skill (`self.workspace_root`). Impede acesso fora da área designada.
    - **Validação Opcional:** Verifica existência (`check_existence=True`) e tipo (`target_type='file'/'dir'`). Impede acesso a caminhos ocultos (`allow_hidden=False`).
    - **Injeção:** Se a validação for bem-sucedida, injeta `resolved_path: Path` (o caminho absoluto validado) e `original_path_str: str` nos argumentos da função decorada (usados pela lógica da skill, ignorados por `execute_tool` - seção 4.1.2).
    - **Erro:** Retorna um dicionário de erro padronizado em caso de falha na validação, interrompendo a execução da skill.

### 4.3.2 Segurança de Código AST (`a3x/core/code_safety.py`)

- **`is_safe_ast(code_string)`:**
    - Usada pela skill `execute_code` antes da execução.
    - Parseia o código em uma AST (`ast.parse`).
    - Percorre a árvore e verifica cada nó contra uma lista de permissões (`allowed_nodes`).
    - **Restrições:**
        - Bloqueia nós AST não permitidos.
        - Bloqueia chamadas a funções built-in não listadas em `allowed_calls`.
        - Bloqueia chamadas de métodos/atributos (`obj.method()`).
        - Bloqueia `import` / `import from`.
        - Bloqueia acesso a atributos iniciados com `_`.
        - Permite apenas operações binárias básicas (`+`, `-`, `*`, `/`).
    - Retorna `(True, ...)` se seguro, `(False, message)` se inseguro.

### 4.3.3 Backup (`a3x/core/backup.py`)

- **`create_backup(file_path_str, workspace_root)`:**
    - Usada pela skill `delete_path`.
    - **Local:** Salva backups em `.a3x/backups/` (relativo à raiz do projeto).
    - **Estrutura:** Mantém a estrutura de diretórios original (relativa ao `workspace_root`) dentro do diretório de backup.
    - **Nomeação:** Adiciona timestamp ao nome do arquivo + `.bak`.
    - **Cópia:** Usa `shutil.copy2` para preservar metadados.
    - **Retenção:** Remove backups antigos para o mesmo arquivo se exceder `MAX_BACKUPS_PER_FILE`.
    - **Segurança:** Verifica se o `file_path_str` resolvido está dentro do `workspace_root`.

## 4.4 Interface CLI e Modos Operacionais

### 4.4.1 Fluxo de Execução da CLI (`a3x/cli/interface.py`)

1.  **Inicialização:**
    - O script `a3x/cli/interface.py` é o ponto de entrada (`if __name__ == "__main__":`).
    - Chama `run_cli()`, que por sua vez executa `asyncio.run(main_async())`.
2.  **`main_async()`:**
    - **Setup:** Configura logging (seção 2.2.2), parseia argumentos (`_parse_arguments`), muda para o diretório raiz do projeto (`change_to_project_root`), inicializa o banco de dados (`initialize_database` - relacionado à seção 3.5).
    - **Gerenciamento de Servidor:** Se `--no-server` não for passado, tenta iniciar o servidor LLaMA usando `a3x.core.server_manager.start_llama_server()` (seção 2.2.3). Registra `a3x.core.server_manager.stop_all_servers()` para ser chamado na saída via `atexit` e no bloco `finally`.
    - **Carregamento de Skills:** Chama `a3x.core.skills.load_skills()` (seção 4.2.4) para importar e registrar todas as skills disponíveis no `SKILL_REGISTRY`.
    - **Inicialização do Agente:** Cria a instância `CerebrumXAgent` (`_initialize_agent`), passando o prompt do sistema, a URL do LLM e o `SKILL_REGISTRY` preenchido.
    - **Seleção de Modo:** Com base nos argumentos da CLI, seleciona o modo de operação:
      - `--task <tarefa>`: Chama `_handle_task_argument`. Se `<tarefa>` for um JSON, processa-o (executando skills diretamente ou passando um objetivo ao agente). Se for uma string, passa-a como objetivo para `handle_agent_interaction`.
      - `--command <comando>`: Chama `_handle_command_argument` -> `handle_agent_interaction`.
      - `--input-file <arquivo>`: Chama `_handle_file_argument` -> `_process_input_file` -> `handle_agent_interaction` para cada linha.
      - `--interactive` (ou padrão): Chama `_handle_interactive_argument` -> `_run_interactive_mode` -> `handle_agent_interaction` em loop.
      - `--stream-direct <prompt>`: Chama `a3x.cli.display.stream_direct_llm`.
      - `--train`: Chama `a3x.training.trainer.run_qlora_finetuning` (seção 4.5.1).
      - `--run-skill <skill> [--skill-args <json> | --skill-args-file <arquivo>]`: Chama `_handle_run_skill_argument` para executar uma skill específica diretamente, fora do fluxo do agente.
    - **Interação com Agente (`handle_agent_interaction` em `a3x/cli/display.py`):**
      - Recebe o agente, o comando/objetivo, e um histórico (embora o histórico principal seja gerenciado internamente pelo agente - ver seção 3.3.2).
      - Chama `agent.run(objective=command)` (o método principal do `CerebrumXAgent` - seção 3.1).
      - Processa o dicionário de resultado final retornado por `agent.run()` (status, mensagem final, etc.) e exibe usando `rich.panel`.
    - **Limpeza:** O bloco `finally` em `main_async` garante a chamada a `stop_all_servers()`.

### 4.4.2 Interface CLI - Display (`a3x/cli/display.py`)

Este módulo é responsável por formatar e apresentar a saída do agente e outras informações para o usuário na interface de linha de comando, utilizando a biblioteca `rich`.

- **`handle_agent_interaction(agent, command, history)`:**
    - Função central chamada pelos diferentes modos de operação em `interface.py` (seção 4.4.1).
    - **Execução:** Chama `agent.run(objective=command)` para iniciar o ciclo cognitivo do agente (seção 3.1).
    - **Processamento de Resultados:** Recebe o dicionário de resultados de `agent.run()`.
    - **Formatação:**
        - Extrai `final_status`, `final_message`, `step_results`.
        - Usa `rich.panel.Panel` para exibir a mensagem final com um título indicando o status (e.g., "✅ Task Completed", "❌ Task Failed").
        - Itera sobre `step_results` (se presentes) e imprime informações sobre cada passo (pensamento, ação, observação) de forma estruturada.
    - **Retorno:** Não retorna valor, apenas imprime na saída padrão.
- **`stream_direct_llm(prompt, agent)`:**
    - Usada pelo modo `--stream-direct` (seção 4.4.1).
    - Constrói um prompt de chat simples.
    - Chama `agent.llm_interface.call_llm_stream` para obter uma resposta *streaming* do LLM.
    - **Exibição:** Itera sobre os chunks recebidos do stream e imprime-os diretamente no console, proporcionando uma resposta interativa do LLM puro.
- **Funções Auxiliares (Potenciais):** Pode conter outras funções para formatação específica de erros, logs, ou outros tipos de saída, utilizando componentes `rich` como `Table`, `Syntax`, etc. (Detalhes a serem confirmados por inspeção do código).

## 4.5 Treinamento e Fine-tuning (LoRA)

### 4.5.1 Treinamento / Fine-tuning (`a3x/training/trainer.py`)

Implementa o fine-tuning do modelo base usando QLoRA e dados da memória episódica (seção 3.5).

- **`run_qlora_finetuning()`:** Função principal chamada (e.g., pela CLI com `--train` - seção 4.4.1).
    - **Configuração:** Usa parâmetros de `config.py` (modelo base, parâmetros LoRA, diretórios, hiperparâmetros de treino - seção 2.2.1).
    - **Quantização:** Configura `BitsAndBytesConfig` para carregar o modelo base em 4 bits (NF4, `bfloat16`).
    - **Carregamento:** Carrega modelo base (`AutoModelForCausalLM`) com quantização e tokenizer (`AutoTokenizer`). Define `pad_token`.
    - **PEFT (LoRA):** Prepara o modelo para treino k-bit (`prepare_model_for_kbit_training`) e aplica `LoraConfig` (`get_peft_model`), tornando apenas os adaptadores LoRA treináveis.
    - **Dataset:** Chama `prepare_dataset` para:
        - Amostrar experiências do `experience_buffer` (`db_utils.sample_experiences` - relacionado à seção 3.5).
        - Formatar cada experiência como `Context -> Action -> Outcome`.
        - Tokenizar os textos formatados.
        - Criar um `Dataset` do Hugging Face.
    - **Trainer:** Configura `TrainingArguments` (usando otimizador `paged_adamw_8bit`, batch size, épocas, etc.) e `DataCollatorForLanguageModeling` (para padding dinâmico).
    - **Treino:** Instancia e chama `trainer.train()`.
    - **Salvar:** Salva apenas os pesos do adaptador LoRA treinado (`model.save_pretrained`) no diretório de saída.

## 4.6 Skills Específicas Notáveis

### 4.6.1 Skill: Geração de Código (`a3x/skills/code_generation.py`)

- **Skill `generate_code(purpose, language, construct_type, context)`:**
    - **Objetivo:** Gerar código usando o LLM.
    - **Processo:**
        - Constrói um prompt de chat detalhado (system + user) especificando o propósito, linguagem, tipo de estrutura, contexto (opcional), e pedindo **apenas** o código como resposta.
        - Chama o LLM configurado (`LLAMA_SERVER_URL`) via `requests.post` (síncrono, não-streaming) com baixa temperatura.
        - **Extração de Código:**
            1. Tenta extrair código de blocos markdown (```lang...```) na resposta.
            2. Se falhar, aplica limpeza heurística para remover texto explicativo comum do início/fim.
        - Valida se o código extraído/limpo não está vazio.
    - **Retorno:** Dicionário de sucesso com o `code` gerado, ou dicionário de erro.

# 5.0 Metaestrutura e Rastreabilidade

## 5.1 Convenções de Anotação Técnica

### 5.1.1 Estruturação retroativa do documento com numeração hierárquica conforme padrão de rastreabilidade.
*Data:* 2024-07-27

## 5.2 Diretrizes do Auditor *(Conteúdo a ser adicionado)*

*(Esta seção registrará as diretrizes formais fornecidas pelo Auditor para a manutenção deste documento e do projeto.)*

# 7.0 Lacunas Arquiteturais Identificadas

Esta seção documenta as lacunas significativas identificadas durante a auditoria, que impactam a funcionalidade, a capacidade cognitiva ou a rastreabilidade do agente A³X.

## 7.1 Memória e Aprendizado

### 7.1.1 Detalhamento Incompleto do Ciclo de Aprendizado Heurístico

1.  **Local da Lacuna:** Seções `3.5 Memória Episódica` e `3.6 Aprendizado Heurístico e Generalização` (marcadas como `*(Conteúdo a ser detalhado)*`). Código relacionado em `a3x/core/db_utils.py` e skills `reflect_on_failure`, `learn_from_failure_log`, `learning_cycle`, `auto_generalize_heuristics`, `consolidate_heuristics` em `a3x/skills/core/`. Ciclo cognitivo em `a3x/core/cerebrumx.py` (`_execute_plan`, `_reflect_and_learn`).
2.  **Descrição da Disfunção ou Omissão:** Falta documentação detalhada sobre como a memória episódica é utilizada para registrar o contexto completo de sucessos/falhas e como as skills de aprendizado (reflexão, generalização) interagem para produzir e consultar heurísticas no arquivo `learned_heuristics.jsonl`. O fluxo exato entre a falha de um passo, a reflexão imediata, o aprendizado pós-ciclo e a subsequente consulta/aplicação de heurísticas não está claro.
3.  **Implicação Cognitiva:** Comprometimento severo da *capacidade de adaptação contínua* do agente. Impede a verificação da habilidade de evitar erros repetidos, refinar estratégias ou otimizar o uso de ferramentas com base na experiência, um pressuposto central dos manifestos do A³X.
4.  **Consequência Técnica:** Dificulta a depuração de comportamentos subótimos ou falhas recorrentes. Impossibilita a validação experimental do ciclo de aprendizado completo. As skills podem existir, mas sua integração efetiva e o impacto real no comportamento do agente são incertos.
    *   **Recomendação do Auditor:** Priorizar a documentação e validação experimental do registro/recuperação das memórias episódica e heurística, anexando logs reais e exemplos de heurísticas geradas/consultadas.

### 7.1.2 Evidência de Inoperância no Registro de Heurísticas de Falha (Cenário Específico)

1.  **Local da Lacuna:** Ciclo de aprendizado de falha acionado em `a3x/core/cerebrumx.py` (`_execute_plan`, bloco `if error_occurred:`, linhas ~450-490), que chama `reflect_on_failure` e `learn_from_failure_log`. Arquivo de registro: `a3x/memory/learning_logs/learned_heuristics.jsonl`. Lógica de execução de planos simples em `_execute_plan` (linhas ~280-340).
2.  **Descrição da Disfunção ou Omissão:** Em teste empírico (2024-07-27), uma falha foi induzida na skill `list_directory` (erro de validação de caminho fora do workspace), acionada via execução direta de "plano simples". A lógica de tratamento de erro para planos simples (linhas ~305-310) define o status como `failed` e **retorna imediatamente**, **sem acionar** o bloco `if error_occurred:` (linhas ~450-490) onde as skills `reflect_on_failure` e `learn_from_failure_log` são chamadas. Consequentemente, a inspeção do arquivo `learned_heuristics.jsonl` não revelou **nenhum novo registro** de heurística.
3.  **Implicação Cognitiva:** O agente é incapaz de aprender com falhas que ocorrem durante a execução otimizada de planos simples. Isso cria um ponto cego no ciclo de *adaptação contínua*, especificamente para erros comuns de validação ou execução em tarefas básicas.
4.  **Consequência Técnica:** O mecanismo de aprendizado de falha (`reflect_on_failure` -> `learn_from_failure_log`) está implementado, mas **não é alcançável/executado** para falhas em planos simples. A função existe, mas é **inoperante** para este subconjunto de cenários de falha. A duplicação/deslocamento da lógica de aprendizado (parte em `_execute_plan` para falhas ReAct, ausente para falhas diretas, e esperada centralização em `learning_cycle`) contribui para esta inconsistência.
    *   **Nota:** A skill `learning_cycle` em si também confirmou (logs de 2024-07-27) que atualmente não implementa o aprendizado de falha, dependendo da lógica (agora incompleta) em `_execute_plan`.

### 7.1.3 Bug Crítico Corrigido: Instanciação Incorreta de _ToolExecutionContext

1.  **Local da Lacuna:** `a3x/core/cerebrumx.py` (linhas ~298, ~452) e `a3x/skills/core/learning_cycle.py` (linha ~38). Definição em `a3x/core/tool_executor.py`.
2.  **Descrição da Disfunção ou Omissão:** A assinatura do construtor de `_ToolExecutionContext` foi modificada (presumivelmente em refatoração anterior) para exigir o argumento `tools_dict`, mas múltiplos pontos de instanciação não foram atualizados para fornecer este argumento.
3.  **Implicação Cognitiva:** Bloqueava completamente a execução de *qualquer* plano simples e a execução da skill `learning_cycle`, impedindo tanto a execução direta de tarefas quanto o ciclo de aprendizado pós-execução. Paralisava funções centrais do agente.
4.  **Consequência Técnica:** Causava `TypeError: ToolExecutionContext.__new__() missing 1 required positional argument: 'tools_dict'` em múltiplos pontos críticos. **Bug corrigido em 2024-07-27** durante auditoria empírica, passando `self.tools` ou `agent_tools` conforme apropriado.
    *   **Nota:** Este bug mascara outras possíveis falhas no ciclo de aprendizado, como a inoperância descrita em 7.1.2.

## 7.2 Runtime e Treinamento

### 7.2.1 Loop de Aprendizado Quebrado (Fine-tuning LoRA Runtime)

1.  **Local da Lacuna:** Desconexão entre Seção `4.5.1 Treinamento / Fine-tuning` (`a3x/training/trainer.py`) e a carga/utilização do modelo em tempo de execução (inferência) em `a3x/core/llm_interface.py` ou na inicialização do `CerebrumXAgent` (`a3x/core/cerebrumx.py`, `a3x/cli/interface.py`).
2.  **Descrição da Disfunção ou Omissão:** O sistema possui um mecanismo (`--train`) para fine-tuning QLoRA e salvar adaptadores, mas não há evidência (nem no código documentado, nem na documentação) de que esses adaptadores sejam carregados e aplicados ao modelo LLM base durante a operação normal do agente. O ciclo pretendido (`experiência -> fine-tuning -> especialização runtime`) está rompido na etapa final.
3.  **Implicação Cognitiva:** Frustra o princípio de *inteligência acumulativa* e *especialização*. O agente opera apenas com o modelo base pré-treinado, ignorando o conhecimento específico adquirido e refinado através do fine-tuning com dados de sua própria experiência.
4.  **Consequência Técnica:** A funcionalidade de treinamento (`--train`) torna-se isolada e sem impacto prático na performance ou comportamento do agente em execução. Recursos investidos no treinamento são desperdiçados. Trata-se de uma **lacuna arquitetural crítica**.
    *   **Recomendação do Auditor:** Confirmar ausência de carregamento LoRA runtime. Sugerir mecanismo mínimo de reintegração (e.g., carregar último adaptador treinado por padrão em `llm_interface.py` ou via flag na CLI).

## 7.3 Planejamento

### 7.3.1 Nível de Planejamento: Flat (Geração Textual Simples)

1.  **Local da Lacuna:** Seções `3.1.3 Planejamento (_plan)`, `3.2 Planejamento e Execução`. Código principal em `a3x/core/planner.py` (`generate_plan`).
2.  **Descrição da Disfunção ou Omissão:** A função `generate_plan` atua como um gerador textual simples (provavelmente via prompt ao LLM), sem implementar estruturas de raciocínio algorítmico de planejamento (e.g., decomposição de tarefas, gerenciamento de dependências, replanejamento explícito). O planejamento é "flat", uma lista de passos sem estrutura hierárquica ou condicional robusta.
3.  **Implicação Cognitiva:** Limita a capacidade do agente de abordar tarefas complexas que exigem raciocínio estratégico, decomposição de problemas, ou adaptação do plano a contingências. Prejudica a escalabilidade estratégica para objetivos não triviais.
4.  **Consequência Técnica:** O agente pode falhar em tarefas complexas, gerar planos ineficientes ou incoerentes, e ter dificuldade em se recuperar de falhas de forma estratégica (além da reflexão local pós-passo).
    *   **Recomendação do Auditor:** Formalizar limitação como "nível de planejamento: flat". Propor metas evolutivas (e.g., HTN, planos condicionais, meta-planejamento).