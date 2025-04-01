# core/backup.py
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from core.config import PROJECT_ROOT  # Import the base project root

logger = logging.getLogger(__name__)

# Define the root directory for backups relative to the project root
# <<< MODIFIED: Use PROJECT_ROOT from config >>>
BACKUP_ROOT_DIR = Path(PROJECT_ROOT).resolve() / ".a3x" / "backups"
MAX_BACKUPS_PER_FILE = 5
COMPRESS_AFTER_DAYS = 7  # TODO: Implement compression logic later


def create_backup(
    file_path_str: str, workspace_root: Path | None = None
) -> Path | None:
    """
    Cria um backup de um arquivo dentro do diretório .a3x/backups.
    Mantém um número limitado de backups por arquivo.

    Args:
        file_path_str (str): O caminho absoluto ou relativo (dentro do workspace) para o arquivo original.
        workspace_root (Path | None): The workspace root to resolve relative paths against and check containment.
                                      Defaults to core.config.PROJECT_ROOT.

    Returns:
        Path | None: O caminho para o arquivo de backup criado, ou None se falhar.
    """
    # Use provided workspace_root, fallback to PROJECT_ROOT from config
    effective_workspace_root = (workspace_root or Path(PROJECT_ROOT)).resolve()
    logger.debug(
        f"create_backup called for '{file_path_str}' with effective workspace '{effective_workspace_root}'"
    )
    # The BASE directory for all backups is still relative to the REAL project root
    backup_base_dir = Path(PROJECT_ROOT).resolve() / ".a3x" / "backups"

    try:
        original_path = Path(file_path_str)
        # Resolve para garantir que estamos trabalhando com um caminho absoluto,
        # mas a validação de workspace usa effective_workspace_root.
        if not original_path.is_absolute():
            # Resolve RELATIVE to the effective workspace for validation check
            resolved_original_path = (
                effective_workspace_root / original_path
            ).resolve()
        else:
            resolved_original_path = original_path.resolve()

        # Verificação de segurança: path deve estar DENTRO do workspace EFETIVO.
        if not resolved_original_path.is_relative_to(effective_workspace_root):
            logger.error(
                f"Backup negado: Caminho resolvido '{resolved_original_path}' fora do workspace efetivo '{effective_workspace_root}'"
            )
            return None

        if not resolved_original_path.is_file():
            logger.error(
                f"Backup falhou: Caminho não é um arquivo válido ou não existe: {resolved_original_path}"
            )
            return None

        # Garante que o diretório de backup BASE (relativo ao PROJECT_ROOT) existe
        backup_base_dir.mkdir(parents=True, exist_ok=True)

        # Cria um subdiretório DENTRO do backup_base_dir.
        # A estrutura do subdiretório reflete o path relativo ao workspace EFETIVO.
        relative_path = resolved_original_path.relative_to(effective_workspace_root)
        file_backup_dir = backup_base_dir / relative_path.parent
        file_backup_dir.mkdir(parents=True, exist_ok=True)

        # Gera nome do backup com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_filename = f"{resolved_original_path.name}.{timestamp}.bak"
        backup_path = file_backup_dir / backup_filename

        # Copia o arquivo original para o local de backup
        shutil.copy2(resolved_original_path, backup_path)
        logger.info(f"Backup criado com sucesso: {backup_path}")

        # Gerencia backups antigos (dentro do diretório específico do arquivo)
        existing_backups = sorted(
            file_backup_dir.glob(f"{resolved_original_path.name}.*.bak"),
            key=os.path.getmtime,
            reverse=True,
        )

        if len(existing_backups) > MAX_BACKUPS_PER_FILE:
            for old_backup in existing_backups[MAX_BACKUPS_PER_FILE:]:
                try:
                    old_backup.unlink()
                    logger.info(f"Backup antigo removido: {old_backup}")
                except OSError as e:
                    logger.error(f"Erro ao remover backup antigo {old_backup}: {e}")

        return backup_path

    except Exception:
        logger.exception(
            f"Erro inesperado durante a criação do backup para {file_path_str}:"
        )
        return None


def _manage_old_backups(file_backup_dir: Path, original_filename: str):
    """Mantém apenas os MAX_BACKUPS_PER_FILE mais recentes."""
    try:
        backups = sorted(
            file_backup_dir.glob(f"{original_filename}.*.bak"),
            key=os.path.getmtime,
            reverse=True,  # Mais recente primeiro
        )

        if len(backups) > MAX_BACKUPS_PER_FILE:
            backups_to_delete = backups[MAX_BACKUPS_PER_FILE:]
            for old_backup in backups_to_delete:
                try:
                    old_backup.unlink()
                    logger.info(f"Backup antigo removido: {old_backup.name}")
                except OSError as del_err:
                    logger.warning(
                        f"Não foi possível remover o backup antigo '{old_backup.name}': {del_err}"
                    )
    except Exception:
        logger.exception(
            f"Erro ao gerenciar backups antigos em {file_backup_dir} para {original_filename}:"
        )


# TODO: Implementar lógica de compressão para backups com mais de COMPRESS_AFTER_DAYS
