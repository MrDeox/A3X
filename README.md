# A³X - Agente Autônomo Adaptável Experimental

A³X é um projeto experimental para construir um sistema de agente de IA modular, localmente executado e adaptável, com foco em transparência e controle do usuário. Ele utiliza uma arquitetura baseada em ReAct (Reasoning and Acting) e é projetado para ser extensível com novas skills e capacidades.

## Visão Geral

O sistema consiste em:

*   **Core:** Componentes centrais que gerenciam o ciclo ReAct, estado do agente, comunicação com o LLM e despacho de skills (`core/`).
*   **Skills:** Módulos independentes que fornecem funcionalidades específicas ao agente (ex: executar código, buscar na web, gerenciar arquivos, acessar memória) (`skills/`).
*   **Memória:** Persistência de estado e memória semântica usando SQLite com extensão VSS (`memory.db`).
*   **LLM Local:** Interage com um servidor de LLM local (como `llama.cpp` rodando um modelo GGUF) para raciocínio e geração.
*   **CLI:** Uma interface de linha de comando (`assistant_cli.py`) para interagir com o agente.

## Configuração

1.  **Clone o Repositório:**
    ```bash
    git clone <url_do_repositorio>
    cd A3X
    ```
2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # ou venv\\Scripts\\activate # Windows
    ```
3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure o LLM Local:**
    *   Baixe um modelo GGUF compatível (ex: Mistral 7B Instruct) e coloque-o no diretório `models/`.
    *   Compile ou baixe o servidor `llama.cpp`. Coloque o executável `llama-server` (ou o nome correspondente) na raiz do projeto ou ajuste o caminho em `core/config.py`.
    *   *Opcional:* Rode o benchmark (`python tests/benchmark_ngl.py --ngl-values "..."`) para encontrar o melhor valor de `-ngl` para seu hardware e atualize `LLAMA_SERVER_ARGS` em `core/config.py`.
5.  **Variáveis de Ambiente:**
    *   Crie um arquivo `.env` na raiz do projeto (copiando `.env.example` se existir).
    *   Adicione sua chave da API Tavily:
        ```dotenv
        TAVILY_API_KEY="sua_chave_tavily_aqui"
        ```
    *   Certifique-se que `TAVILY_ENABLED=True` em `core/config.py` se quiser usar a busca Tavily.
6.  **Inicialize o Banco de Dados:**
    *   A primeira execução do agente (via CLI ou testes) geralmente inicializa o `memory.db` e as tabelas necessárias (incluindo as de VSS se a extensão estiver carregada corretamente pelo SQLite).

## Execução

1.  **Inicie o Servidor LLM:** Execute o `llama-server` em um terminal separado com as configurações desejadas (modelo, `-ngl`, etc.). Exemplo:
    ```bash
    ./llama-server -m models/mistral-7b-instruct-v0.1.Q4_K_M.gguf -c 2048 -ngl 25 --host 127.0.0.1 --port 8080
    ```
    (Ajuste os parâmetros conforme necessário e de acordo com `core/config.py`).
2.  **Execute o Agente CLI:**
    *   Para um comando único:
        ```bash
        python assistant_cli.py -c "Qual a capital do Brasil?"
        ```
    *   Para comandos interativos ou de um arquivo:
        ```bash
        python assistant_cli.py # Modo interativo
        # python assistant_cli.py -i caminho/para/arquivo_de_comandos.txt
        ```

## Arquitetura (Visão Simplificada)

1.  `assistant_cli.py`: Recebe o input do usuário.
2.  `core/agent.py` (ReactAgent): Gerencia o loop ReAct:
    *   Constrói o prompt com o objetivo, histórico e ferramentas disponíveis.
    *   Chama o LLM (`_call_llm`) via `core/config.py` (LLAMA_SERVER_URL).
    *   Parseia a resposta do LLM para extrair "Thought" e "Action".
    *   Se "Action":
        *   Identifica a skill e os parâmetros.
        *   Chama a skill correspondente do diretório `skills/` (listadas em `core/tools.py`).
        *   Recebe a "Observation" (resultado da skill).
    *   Repete o ciclo até obter "Final Answer" ou atingir limite de iterações.
3.  `skills/*.py`: Executam tarefas específicas (web search, file I/O, code execution, memory R/W).
4.  `memory.db`: Armazena o estado e a memória semântica.

## Próximos Passos / Roadmap (Exemplos)

*   Expandir cobertura de testes.
*   Implementar novas skills (ex: interação com APIs específicas, browser automation).
*   Refinar o processo de extração de "Action" e "Action Input".
*   Explorar mecanismos de planejamento mais avançados.
*   Desenvolver o loop de evolução do meta-prompt do Executor.
