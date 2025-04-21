# test_sandbox_skill.py
import asyncio
import logging
from pathlib import Path
import sys
import os # Import os for path manipulation if needed

# --- Configuração Inicial ---
# Tenta determinar o root do projeto de forma mais robusta
try:
    # Assume que este script está na raiz ou um nível abaixo
    script_path = Path(__file__).resolve()
    # Tenta encontrar um marcador do projeto (ex: .git, pyproject.toml) subindo na árvore
    project_root_found = script_path.parent
    while not (project_root_found / ".git").exists() and not (project_root_found / "pyproject.toml").exists() and project_root_found != project_root_found.parent:
         project_root_found = project_root_found.parent
    if (project_root_found / ".git").exists() or (project_root_found / "pyproject.toml").exists():
         PROJECT_ROOT_AUTO = str(project_root_found)
    else:
         # Fallback se não encontrar marcador (menos ideal)
         PROJECT_ROOT_AUTO = str(script_path.parent)
         print(f"Aviso: Não foi possível determinar o root do projeto com certeza. Usando: {PROJECT_ROOT_AUTO}")

except NameError:
    # __file__ não definido (ex: rodando interativamente) - use CWD
    PROJECT_ROOT_AUTO = os.getcwd()
    print(f"Aviso: __file__ não definido. Usando diretório atual como root: {PROJECT_ROOT_AUTO}")

# Adiciona o root ao sys.path para garantir imports do A³X
if PROJECT_ROOT_AUTO not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_AUTO)
    print(f"Adicionado ao sys.path: {PROJECT_ROOT_AUTO}")


# --- Imports do A³X ---
# Coloque os imports APÓS ajustar o sys.path
try:
    from a3x.skills.code_execution import execute_python_in_sandbox
    from a3x.core.skills import SkillContext
    # Importa PROJECT_ROOT do config para que a skill o use
    from a3x.core.config import PROJECT_ROOT as A3X_CONFIG_ROOT
except ImportError as e:
    print(f"Erro ao importar componentes A³X: {e}")
    print("Verifique se o script está no diretório correto e o sys.path inclui o root do A³X.")
    sys.exit(1)

# Verifica se o PROJECT_ROOT do config corresponde ao detectado
if A3X_CONFIG_ROOT != PROJECT_ROOT_AUTO:
     print(f"AVISO: PROJECT_ROOT do config ({A3X_CONFIG_ROOT}) difere do detectado ({PROJECT_ROOT_AUTO}).")
     print("Usando o PROJECT_ROOT do config para consistência com a skill.")
     PROJECT_ROOT = A3X_CONFIG_ROOT
else:
     PROJECT_ROOT = A3X_CONFIG_ROOT # ou PROJECT_ROOT_AUTO, são iguais

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestSandboxSkill")

async def main():
    if not PROJECT_ROOT:
        logger.error("Erro: PROJECT_ROOT não está configurado no a3x.core.config ou não pôde ser detectado.")
        return

    logger.info(f"Usando PROJECT_ROOT: {PROJECT_ROOT}")

    # --- Cria um contexto MÍNIMO mas VÁLIDO para a skill --- 
    # Define uma função async vazia para llm_call
    async def _dummy_llm_call(*args, **kwargs):
        logger.warning("_dummy_llm_call invoked!")
        return None # Ou um valor apropriado se a skill esperar algo

    # Cria o SkillContext com os argumentos necessários
    # Nota: 'task' pode precisar ser um objeto mais complexo dependendo do uso
    mock_context = SkillContext(
        logger=logger, 
        llm_call=_dummy_llm_call, 
        is_test=True, 
        workspace_root=Path(PROJECT_ROOT), 
        task=None # Usando None como placeholder para 'task'
    )

    script_to_run = "sandbox_hello.py" # Relativo ao PROJECT_ROOT

    logger.info(f"--- Executando {script_to_run} via execute_python_in_sandbox ---")
    result = await execute_python_in_sandbox(
        context=mock_context,
        script_path=script_to_run,
        timeout_seconds=30
    )

    print("\n--- Resultado da Execução da Skill ---")
    print(f"Status: {result.get('status')}")
    print(f"Exit Code: {result.get('exit_code')}")
    print("\n--- Stdout ---")
    print(result.get('stdout'))
    print("\n--- Stderr ---")
    print(result.get('stderr'))
    print("------------------------------------")

    # Verifica se o arquivo de teste foi criado fora do sandbox
    expected_file_path_str = "teste_sandbox.txt" # Nome do arquivo criado pelo script de teste
    expected_file = Path(PROJECT_ROOT) / expected_file_path_str

    if result.get("status") == "success" and result.get("exit_code") == 0:
        logger.info("Skill executada com sucesso.")
        if expected_file.exists():
            logger.info(f"Arquivo '{expected_file_path_str}' foi criado com sucesso em {expected_file}.")
            try:
                # Limpa o arquivo
                expected_file.unlink()
                logger.info(f"Arquivo '{expected_file_path_str}' removido.")
            except Exception as e:
                logger.error(f"Erro ao remover '{expected_file_path_str}': {e}")
        else:
            logger.error(f"!!! ALERTA: Arquivo '{expected_file_path_str}' NÃO foi encontrado em {expected_file} após execução bem-sucedida!")
    elif expected_file.exists():
         logger.warning(f"A execução da skill falhou ou teve erro (status: {result.get('status')}, exit: {result.get('exit_code')}), mas o arquivo '{expected_file_path_str}' foi encontrado. Limpando...")
         try:
             expected_file.unlink()
             logger.info(f"Arquivo '{expected_file_path_str}' removido.")
         except Exception as e:
                logger.error(f"Erro ao remover '{expected_file_path_str}': {e}")
    else:
         logger.warning(f"A execução da skill falhou ou teve erro (status: {result.get('status')}, exit: {result.get('exit_code')}) e o arquivo '{expected_file_path_str}' não foi encontrado (consistente).")


if __name__ == "__main__":
    # Garante que temos um loop de eventos, mesmo se rodado como script simples
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run loop" in str(e):
             logger.warning("Loop de eventos já rodando. Tentando obter loop existente.")
             loop = asyncio.get_event_loop()
             loop.run_until_complete(main())
        else:
             raise 