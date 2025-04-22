import re
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para permitir importações relativas
# (Assume que o script está em /scripts)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Diretórios a serem verificados (relativos à raiz do projeto)
# Exclui diretórios como .venv, .git, __pycache__, etc.
# Exclui arquivos de teste, documentação e configuração conhecidos
# Exclui o próprio script
# Exclui o ProfessorLLMFragment e o Executor (onde chamadas são esperadas/já tratadas)
# Exclui arquivos já refatorados (argument_parser, generate_module_...)
DEFAULT_INCLUDE_DIRS = ["a3x"]
DEFAULT_EXCLUDE_PATTERNS = [
    "a3x/a3net/core/professor_llm_fragment.py",
    "a3x/a3net/modules/executor.py",
    "a3x/skills/generate_module_from_directive.py",
    "a3x/core/utils/argument_parser.py",
    "scripts/validate_llm_usage.py",
    "tests/",
    "docs/",
    "*.log",
    "*.md",
    "*.txt",
    "*.sh",
    "*.json",
    "__pycache__/",
    ".git/",
    ".venv/"
]
PYTHON_FILE_PATTERN = "*.py"

# --- Padrões Regex para identificar problemas ---

# 1. Chamadas diretas suspeitas ao LLM Interface fora dos locais permitidos
#    Procura por `context.llm_interface.call_llm` ou `llm_interface.ask_llm` (ou similar)
#    Ignora linhas comentadas
LLM_CALL_PATTERN = re.compile(r"^[^#]*\b(context\.llm_interface|llm_interface)\.(call_llm|ask_llm)\(\s*.*?\s*\)", re.IGNORECASE)

# 2. Planejamento Autônomo em Fragmentos (Exemplo: PlannerFragment já tratado)
#    Este é mais difícil de detectar genericamente via regex.
#    Procura por post_chat_message enviando 'plan_sequence' ou 'architecture_suggestion'
#    Pode gerar falsos positivos, mas ajuda a revisar.
#    Regex simplificada para evitar erros de parsing
PLAN_POST_PATTERN = re.compile(r"^[^#]*await\s+self\.post_chat_message\(.*(plan_sequence|architecture_suggestion).*\)", re.IGNORECASE)

# 3. Uso direto de URLs de LLM (exceto em config talvez)
LLM_URL_PATTERN = re.compile(r"^[^#]*['\"]https?://.*/(?:complete|completions|chat)['\"]", re.IGNORECASE)


def scan_file(filepath: Path, patterns: Dict[str, re.Pattern]) -> Dict[str, List[Tuple[int, str]]]:
    """Escaneia um único arquivo procurando pelos padrões regex."""
    issues_found = {name: [] for name in patterns}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_num = i + 1
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue # Ignora linhas vazias ou totalmente comentadas

                for name, pattern in patterns.items():
                    if pattern.search(line): # Verifica a linha inteira, não apenas a stripada
                        issues_found[name].append((line_num, line.strip()))
                        # Não precisa checar outros padrões para a mesma linha se um já bateu?
                        # Ou pode haver múltiplas issues na mesma linha? Deixar checar todas.

    except Exception as e:
        print(f"  [Erro] Falha ao ler {filepath}: {e}", file=sys.stderr)
    return issues_found


def main():
    """Função principal para executar o scan."""
    print("--- Iniciando Verificação de Segurança A³X --- ")
    print(f"Raiz do Projeto: {project_root}")

    patterns_to_check = {
        "LLM_CALL": LLM_CALL_PATTERN,
        "PLAN_POST": PLAN_POST_PATTERN,
        "LLM_URL": LLM_URL_PATTERN,
    }

    total_issues = 0
    files_scanned = 0

    # Usar Path.glob para encontrar arquivos .py nos diretórios incluídos
    py_files = []
    for dir_pattern in DEFAULT_INCLUDE_DIRS:
        py_files.extend(project_root.glob(f"{dir_pattern}/**/{PYTHON_FILE_PATTERN}"))

    print(f"Encontrados {len(py_files)} arquivos Python para escanear.")

    for filepath in py_files:
        relative_path_str = str(filepath.relative_to(project_root))

        # Checar se o arquivo deve ser excluído
        exclude = False
        for pattern in DEFAULT_EXCLUDE_PATTERNS:
            if filepath.match(pattern) or pattern in relative_path_str:
                # print(f"  Excluindo: {relative_path_str} (Match: {pattern})")
                exclude = True
                break
        if exclude:
            continue

        files_scanned += 1
        # print(f"  Escaneando: {relative_path_str}")
        issues_in_file = scan_file(filepath, patterns_to_check)

        for issue_type, findings in issues_in_file.items():
            if findings:
                print(f"\n[ALERTA: {issue_type}] em {relative_path_str}:", file=sys.stderr)
                for line_num, line_content in findings:
                    print(f"  - Linha {line_num}: {line_content}", file=sys.stderr)
                total_issues += len(findings)

    print(f"\n--- Verificação Concluída --- ")
    print(f"Arquivos Python escaneados: {files_scanned}")
    if total_issues > 0:
        print(f"Total de problemas potenciais encontrados: {total_issues}", file=sys.stderr)
        print("\nREVISÃO MANUAL RECOMENDADA para os alertas acima.", file=sys.stderr)
        sys.exit(1) # Retorna código de erro se encontrar problemas
    else:
        print("Nenhum problema potencial detectado pelos padrões atuais.")
        sys.exit(0)


if __name__ == "__main__":
    main() 