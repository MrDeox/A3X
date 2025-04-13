# ✅ **Checklist Técnico — Refatoração e Fortalecimento do A³X**

> Cada item pode ser marcado como concluído conforme for implementado.
> Os blocos estão organizados por prioridade e tema.

---

## 🟥 **PRIORIDADE ALTA (Críticos para robustez e consistência)**

- [ ] **Unificar execução de planos:**
  Refatorar `CerebrumXAgent._execute_plan` para sempre usar o loop ReAct (`_perform_react_iteration`), mesmo para planos simples.

- [ ] **Corrigir generalização de heurísticas:**
  Reparar `auto_generalize_heuristics.py` e `generalize_heuristics.py`. Verificar chamadas, dependências e formatos.

- [ ] **Migrar heurísticas de JSONL para SQLite:**
  Implementar `HeuristicStore` com versionamento, score de confiança, e histórico de uso.

- [ ] **Centralizar aprendizado de falha:**
  Mover chamadas a `reflect_on_failure` e `learn_from_failure_log` de `_execute_plan` para dentro de `learning_cycle.py`.

- [ ] **Fortalecer sandbox de `execute_code`:**
  Avaliar uso de `restrictedpython`, `Docker`, ou limites de recursos via `prlimit` para evitar execução maliciosa ou exaustiva.

---

## 🟧 **PRIORIDADE MÉDIA (Estruturais e arquiteturais)**

- [ ] **Refatorar chamadas entre skills:**
  Evitar uso de `execute_tool` dentro de skills. Extrair funcionalidades para utils ou gerar planos mais ricos no planner.

- [ ] **Corrigir `heuristics_validator.py`:**
  Implementar simulação real ou método de teste baseado em logs anteriores. Substituir stub atual.

- [ ] **Expandir uso de `ExceptionPolicy`:**
  Aplicar consistentemente no ciclo cognitivo (planejamento, execução, reflexão, replanejamento).

- [ ] **Centralizar todos os caminhos em `config.py`:**
  Remover caminhos locais/hardcoded. Usar `PROJECT_ROOT` e construir tudo em `config.py`.

- [ ] **Centralizar leitura de `.env`:**
  Apenas `config.py` deve acessar `os.getenv`. Outros módulos devem importar constantes.

- [ ] **Padronizar uso de `pathlib.Path` no projeto.**

---

## 🟨 **PRIORIDADE BAIXA (Aprimoramentos e segurança)**

- [ ] **Adicionar rotação de logs:**
  Usar `TimedRotatingFileHandler` ou `RotatingFileHandler` em `logging_config.py`.

- [ ] **Unificar definição e uso do `workspace_root`:**
  Validadores, backup, execução e skills devem respeitar o mesmo root passado via `ToolExecutionContext`.

- [ ] **Adicionar rate limiting e limites configuráveis para skills críticas.**

- [ ] **Padronizar docstrings de skills:**
  Incluir propósito, parâmetros, exemplos e possíveis erros para ajudar o LLM/planner.

- [ ] **Auditar skills externas (web/API):**
  Revisar `web_search`, `browser_skill`, `api_caller` etc. para práticas seguras e controle de entrada/saída.

---

## 🧠 **EXTRA (evolução futura)**

- [ ] **Implementar `PlannerStrategy`:**
  Módulos específicos para tipos diferentes de objetivos (monetização, pesquisa, criação).

- [ ] **Adicionar score de confiança às heurísticas:**
  Baseado em uso, sucesso e generalização. Pode influenciar ranking no planejamento.

- [ ] **Criar simulação interna de heurísticas antes da execução real.**

- [ ] **Gerar visualização dos ciclos cognitivos, logs e heurísticas aplicadas.** 