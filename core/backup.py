# core/backup.py
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (DEVE SER CONSISTENTE)
try:
    from core.config import PROJECT_ROOT as WORKSPACE_ROOT_STR
    WORKSPACE_ROOT = Path(WORKSPACE_ROOT_STR).resolve()
    logger.debug(f"backup using WORKSPACE_ROOT from core.config: {WORKSPACE_ROOT}")
except ImportError:
    logger.warning("Could not import PROJECT_ROOT from core.config in backup. Using hardcoded path relative to core.")
    core_dir = Path(__file__).parent
    WORKSPACE_ROOT = core_dir.parent.resolve()
    logger.debug(f"backup using calculated WORKSPACE_ROOT: {WORKSPACE_ROOT}")

BACKUP_ROOT_DIR = WORKSPACE_ROOT / ".a3x" / "backups"
MAX_BACKUPS_PER_FILE = 5
COMPRESS_AFTER_DAYS = 7 # TODO: Implement compression logic later

def create_backup(file_path_str: str) -> Path | None:
    """
    Cria um backup de um arquivo dentro do diretório .a3x/backups.
    Mantém um número limitado de backups por arquivo.

    Args:
        file_path_str (str): O caminho absoluto ou relativo (dentro do workspace) para o arquivo original.

    Returns:
        Path | None: O caminho para o arquivo de backup criado, ou None se falhar.
    """
    try:
        original_path = Path(file_path_str)
        # Resolve para garantir que estamos trabalhando com um caminho absoluto dentro do workspace
        # Reutilizando lógica de validação (simplificada aqui, assumindo que a skill chamadora já validou)
        if not original_path.is_absolute():
            resolved_original_path = (WORKSPACE_ROOT / original_path).resolve()
        else:
            resolved_original_path = original_path.resolve()

        # Verificação final de segurança
        if not resolved_original_path.is_relative_to(WORKSPACE_ROOT):
             logger.error(f"Backup negado: Caminho resolvido fora do workspace: {resolved_original_path}")
             return None
             
        if not resolved_original_path.is_file():
            logger.error(f"Backup falhou: Caminho não é um arquivo válido ou não existe: {resolved_original_path}")
            return None # Não faz backup se não for arquivo ou não existir

        # Garante que o diretório de backup raiz existe
        BACKUP_ROOT_DIR.mkdir(parents=True, exist_ok=True)

        # Cria um subdiretório para o arquivo específico se necessário (baseado no caminho relativo)
        relative_path = resolved_original_path.relative_to(WORKSPACE_ROOT)
        file_backup_dir = BACKUP_ROOT_DIR / relative_path.parent
        file_backup_dir.mkdir(parents=True, exist_ok=True)

        # Gera nome do backup com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_filename = f"{resolved_original_path.name}.{timestamp}.bak"
        backup_path = file_backup_dir / backup_filename

        # Copia o arquivo (copy2 preserva metadados)
        shutil.copy2(resolved_original_path, backup_path)
        logger.info(f"Backup criado com sucesso para '{resolved_original_path.name}' em: {backup_path}")

        # Gerencia backups antigos (deleção simples por enquanto)
        _manage_old_backups(file_backup_dir, resolved_original_path.name)

        return backup_path

    except Exception as e:
        logger.exception(f"Erro inesperado ao criar backup para {file_path_str}:")
        return None

def _manage_old_backups(file_backup_dir: Path, original_filename: str):
    """Mantém apenas os MAX_BACKUPS_PER_FILE mais recentes."""
    try:
        backups = sorted(
            file_backup_dir.glob(f"{original_filename}.*.bak"),
            key=os.path.getmtime,
            reverse=True # Mais recente primeiro
        )

        if len(backups) > MAX_BACKUPS_PER_FILE:
            backups_to_delete = backups[MAX_BACKUPS_PER_FILE:]
            for old_backup in backups_to_delete:
                try:
                    old_backup.unlink()
                    logger.info(f"Backup antigo removido: {old_backup.name}")
                except OSError as del_err:
                    logger.warning(f"Não foi possível remover o backup antigo '{old_backup.name}': {del_err}")
    except Exception as e:
        logger.exception(f"Erro ao gerenciar backups antigos em {file_backup_dir} para {original_filename}:")

# TODO: Implementar lógica de compressão para backups com mais de COMPRESS_AFTER_DAYS
