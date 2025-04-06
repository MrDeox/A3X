import sys
import os

# <<< START VENV WORKAROUND >>>
# Manually add the venv site-packages directory to sys.path
# This is needed because the venv creation linked to Cursor, not system Python.
_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_site_packages = os.path.join(
    _project_root, "venv", "lib", "python3.13", "site-packages"
)
if _venv_site_packages not in sys.path:
    sys.path.insert(0, _venv_site_packages)
# Cleanup temporary variables
del _project_root
del _venv_site_packages
# <<< END VENV WORKAROUND >>>

# Adiciona o diretório raiz ao sys.path para encontrar 'cli'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Changed from os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# <<< ADDED: Explicitly add project root again just in case >>>
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Importa a função principal da interface
    # from cli.interface import run_cli
    from a3x.cli.interface import run_cli
except Exception:
    import traceback

    print("Erro DETALHADO ao tentar importar 'run_cli' de 'cli.interface':")
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    # Chama a função principal da interface CLI
    run_cli()
