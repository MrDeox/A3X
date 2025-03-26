import sqlite3
import os
from datetime import datetime

# Define o caminho do banco de dados na raiz do projeto
DATABASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory.db'))

def _initialize_memory_db():
    """Cria a tabela 'knowledge' se ela não existir."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Trigger para atualizar updated_at (Opcional, mas bom)
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_knowledge_updated_at
            AFTER UPDATE ON knowledge FOR EACH ROW
            BEGIN
                UPDATE knowledge SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        ''')
        conn.commit()
        conn.close()
        print("[Memory] Banco de dados inicializado.")
    except sqlite3.Error as e:
        print(f"[Memory Error] Erro ao inicializar banco de dados: {e}")

# Chama a inicialização quando o módulo é carregado
_initialize_memory_db()

def skill_remember_info(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Armazena informações na memória do assistente."""
    print("\n[Skill: Remember Info]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    info = entities.get("info")
    if not info:
        return {"status": "error", "action": "remember_info_failed", "data": {"message": "Não entendi qual informação armazenar."}}

    try:
        # Armazena a informação
        store_info(info)
        return {"status": "success", "action": "info_remembered", "data": {"message": f"Informação armazenada: {info}"}}
    except Exception as e:
        print(f"\n[Erro na Skill Remember Info] Ocorreu um erro: {e}")
        return {"status": "error", "action": "remember_info_failed", "data": {"message": f"Erro ao armazenar informação: {e}"}}

def skill_recall_info(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Recupera informações da memória do assistente."""
    print("\n[Skill: Recall Info]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    info = entities.get("info")
    if not info:
        return {"status": "error", "action": "recall_info_failed", "data": {"message": "Não entendi qual informação recuperar."}}

    try:
        # Recupera a informação
        retrieved_info = recall_info(info)
        if not retrieved_info:
            return {"status": "error", "action": "recall_info_failed", "data": {"message": f"Não encontrei informação sobre '{info}'."}}
        
        return {"status": "success", "action": "info_recalled", "data": {"message": f"Informação recuperada: {retrieved_info}"}}
    except Exception as e:
        print(f"\n[Erro na Skill Recall Info] Ocorreu um erro: {e}")
        return {"status": "error", "action": "recall_info_failed", "data": {"message": f"Erro ao recuperar informação: {e}"}}

# Outras funções de memória podem ser adicionadas aqui (ex: deletar, listar chaves) 