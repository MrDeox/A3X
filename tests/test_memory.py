# tests/test_memory.py
import pytest
import sqlite3
import struct
import numpy as np
from unittest.mock import MagicMock, call, patch, AsyncMock
from unittest import mock
import json

# Importar as funções a serem testadas
from skills.memory import skill_save_memory, skill_recall_memory

# Mock Embedding
MOCK_EMBEDDING_DIM = 4 # Usar dimensão pequena para teste
MOCK_EMBEDDING_ARRAY = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
MOCK_EMBEDDING_BLOB = struct.pack(f'<{MOCK_EMBEDDING_DIM}f', *MOCK_EMBEDDING_ARRAY)

# --- Fixtures --- 

@pytest.fixture
def mock_embedding_functions(mocker):
    """Mocks functions from core.embeddings."""
    # Update paths for save.py
    mock_get_save = mocker.patch('skills.memory.save.get_embedding', return_value=MOCK_EMBEDDING_ARRAY)
    mocker.patch('skills.memory.save.EMBEDDING_DIM', MOCK_EMBEDDING_DIM)
    # Update paths for recall.py
    mock_get_recall = mocker.patch('skills.memory.recall.get_embedding', return_value=MOCK_EMBEDDING_ARRAY)
    mocker.patch('skills.memory.recall.EMBEDDING_DIM', MOCK_EMBEDDING_DIM)
    # Keep mocking core.embeddings directly in case it's imported there too (belt and suspenders)
    mocker.patch('core.embeddings.EMBEDDING_DIM', MOCK_EMBEDDING_DIM, create=True)
    # Return one of the mocks, or maybe a tuple if tests need both? Let's return the recall one for now,
    # as recall tests might depend more specifically on its mock return. Adjust if needed.
    # Consider returning a dict {'save': mock_get_save, 'recall': mock_get_recall} if tests become complex.
    return mock_get_recall # Or adjust based on test needs

@pytest.fixture
def mock_db_connection(mocker):
    """Mocks the database connection and cursor."""
    mock_conn = MagicMock(spec=sqlite3.Connection)
    mock_cursor = MagicMock(spec=sqlite3.Cursor)
    mock_conn.cursor.return_value = mock_cursor
    # Mock get_db_connection for both save and recall modules
    mocker.patch('skills.memory.save.get_db_connection', return_value=mock_conn)
    mocker.patch('skills.memory.recall.get_db_connection', return_value=mock_conn)
    return mock_conn, mock_cursor

# --- Testes para skill_save_memory --- 

@pytest.mark.asyncio
async def test_save_memory_success(mock_db_connection, mock_embedding_functions):
    """Testa o fluxo de sucesso de save_memory, incluindo inserção VSS."""
    mock_conn, mock_cursor = mock_db_connection
    mock_get_embedding_recall = mock_embedding_functions # Mock returned is recall's
    # Need to access the save mock separately if needed, e.g., by modifying the fixture return
    # Or by patching specifically within the test. For now, assume recall mock covers it,
    # or adjust fixture/test if call_args check fails. Let's try asserting on the specific mock:

    # Re-patch specifically for save if needed, or modify fixture to return both mocks
    with patch('skills.memory.save.get_embedding') as mock_get_embedding_save:
        mock_get_embedding_save.return_value = MOCK_EMBEDDING_ARRAY

        action_input = {"content": "Test content", "metadata": {"source": "test"}}

        # Simular inserção bem-sucedida na tabela principal e VSS
        mock_cursor.rowcount = 1 # Simula que a inserção principal aconteceu
        mock_cursor.lastrowid = 123 # ID retornado
        # Ajustado: Primeiro fetchone para check vss (assume que existe, retorna tupla)
        # Segundo fetchone não é chamado neste fluxo, mas podemos deixar None por segurança
        mock_cursor.fetchone.side_effect = [
            ('vss_semantic_memory',), # Simula vss_table_exists = True
            # Não deveria chegar aqui no fluxo de sucesso puro, mas None é seguro.
            None
        ]

        result = skill_save_memory(action_input)

        assert result["status"] == "success"
        assert result["action"] == "memory_saved"
        assert result["data"]["rowid"] == 123
        assert result["data"]["vss_updated"] is True
        # Assert against the specific mock used by skill_save_memory
        mock_get_embedding_save.assert_called_once()
        assert "Test content" in mock_get_embedding_save.call_args[0][0]
        # Verificar chamadas ao cursor (INSERT principal, checar VSS, INSERT VSS)
        assert mock_cursor.execute.call_count == 3 # INSERT semantic, CHECK vss, INSERT vss
        mock_cursor.execute.assert_any_call(
            mock.ANY, 
            ("Test content", MOCK_EMBEDDING_BLOB, '{"source": "test"}')
        ) # Check main INSERT call args
        mock_cursor.execute.assert_any_call(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'"
        ) # Check VSS table existence check
        mock_cursor.execute.assert_any_call(
            mock.ANY, 
            (123, MOCK_EMBEDDING_BLOB)
        ) # Check VSS INSERT call args
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_save_memory_duplicate_content(mock_db_connection, mock_embedding_functions):
    """Testa o fluxo quando o conteúdo já existe (IGNORE), mas VSS é atualizado."""
    mock_conn, mock_cursor = mock_db_connection
    mock_get_embedding_recall = mock_embedding_functions # Mock returned is recall's

    action_input = {"content": "Duplicate content"}

    # Simular IGNORE na inserção principal (rowcount 0), mas encontra ID existente
    mock_cursor.rowcount = 0
    # CORRIGIDO: Inverter a ordem do side_effect para fetchone
    # 1ª chamada (SELECT id): deve retornar (456,)
    # 2ª chamada (SELECT name FROM sqlite_master): deve retornar ('vss_semantic_memory',)
    mock_cursor.fetchone.side_effect = [
        (456,), # Simula encontrar o ID existente via SELECT
        ('vss_semantic_memory',) # Simula vss_table_exists = True
    ]
    # Simular que a inserção VSS acontece (ou já existia, rowcount 0 ou 1)
    # Para simplificar, vamos assumir que aconteceu (rowcount=1)
    # Mockar a segunda execute call para devolver rowcount = 1
    def execute_side_effect(*args, **kwargs):
        sql = args[0]
        if "INSERT OR IGNORE INTO vss_semantic_memory" in sql:
             mock_cursor.rowcount = 1 # Simula inserção VSS
        elif "INSERT OR IGNORE INTO semantic_memory" in sql:
             mock_cursor.rowcount = 0 # Simula IGNORE
        elif "SELECT id FROM semantic_memory" in sql:
             mock_cursor.rowcount = 1 # Simula SELECT encontrou
        # Adicione outros comportamentos se necessário
        # Corrigido: Retornar o próprio cursor, como o execute real faz (CORRIGIDA INDENTAÇÃO)
        return mock_cursor
    mock_cursor.execute.side_effect = execute_side_effect

    result = skill_save_memory(action_input)

    assert result["status"] == "success"
    assert result["action"] == "memory_saved"
    assert result["data"]["rowid"] == 456 # ID encontrado
    assert result["data"]["vss_updated"] is True # Mesmo duplicado, VSS foi atualizado/verificado
    # Verificar chamadas (INSERT IGNORE principal, SELECT ID, checar VSS, INSERT IGNORE VSS)
    # A ordem exata pode variar um pouco dependendo da lógica exata no código
    assert mock_cursor.execute.call_count == 4 # INSERT semantic(IGNORE), SELECT ID, CHECK vss, INSERT vss
    mock_cursor.execute.assert_any_call(
        mock.ANY, 
        ("Duplicate content", MOCK_EMBEDDING_BLOB, None)
    ) # Check main INSERT call args
    mock_cursor.execute.assert_any_call(
        "SELECT id FROM semantic_memory WHERE content = ?", 
        ("Duplicate content",)
    ) # Check SELECT ID call
    mock_cursor.execute.assert_any_call(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'"
    ) # Check VSS table existence check
    mock_cursor.execute.assert_any_call(
        mock.ANY, 
        (456, MOCK_EMBEDDING_BLOB)
    ) # Check VSS INSERT call args
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_save_memory_missing_content(mock_db_connection, mock_embedding_functions):
    """Testa falha se 'content' estiver ausente."""
    action_input = {"metadata": {"source": "test"}} # Sem content
    result = skill_save_memory(action_input)
    assert result["status"] == "error"
    assert result["action"] == "save_memory_failed"
    assert "Parâmetro 'content' obrigatório ausente" in result["data"]["message"]

@pytest.mark.asyncio
async def test_save_memory_embedding_error(mock_db_connection, mock_embedding_functions):
    """Testa falha se get_embedding falhar."""
    # Mock the specific function that will be called
    with patch('skills.memory.save.get_embedding') as mock_get_embedding_save:
        mock_get_embedding_save.side_effect = Exception("Embedding model failed")
        action_input = {"content": "Test content"}
        result = skill_save_memory(action_input)
        assert result["status"] == "error"
        assert result["action"] == "save_memory_failed"
        assert "Erro inesperado ao gerar embedding" in result["data"]["message"]

@pytest.mark.asyncio
async def test_save_memory_db_error(mock_db_connection, mock_embedding_functions):
    """Testa falha se ocorrer erro no DB."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = sqlite3.Error("DB write error")
    action_input = {"content": "Test content"}
    result = skill_save_memory(action_input)
    assert result["status"] == "error"
    assert result["action"] == "save_memory_failed"
    assert "Erro de DB: DB write error" in result["data"]["message"]
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

# --- Testes para skill_recall_memory --- 

@pytest.mark.asyncio
async def test_recall_memory_success(mock_db_connection, mock_embedding_functions):
    """Testa o fluxo de sucesso de recall_memory."""
    mock_conn, mock_cursor = mock_db_connection
    mock_get_embedding_recall = mock_embedding_functions # Fixture returns recall mock

    action_input = {"query": "Find test", "max_results": 2}

    # Simular que VSS existe e não está vazia, e a busca retorna resultados
    mock_cursor.fetchone.side_effect = [
        ('vss_semantic_memory',), # vss_table_exists = True
        (1,) # memory_count > 0
    ]
    mock_cursor.fetchall.return_value = [
        (10, "Test content 1", 0.5), # (id, content, distance)
        (20, "Test content 2", 0.8)
    ]

    result = skill_recall_memory(action_input)

    assert result["status"] == "success"
    assert result["action"] == "memory_recalled"
    assert len(result["data"]["results"]) == 2
    assert result["data"]["results"][0]["id"] == 10
    assert result["data"]["results"][0]["content"] == "Test content 1"
    assert result["data"]["results"][0]["distance"] == 0.5
    assert result["data"]["results"][1]["id"] == 20
    assert result["data"]["results"][1]["distance"] == 0.8
    # Assert against the recall mock provided by the fixture
    mock_get_embedding_recall.assert_called_once()
    assert "Find test" in mock_get_embedding_recall.call_args[0][0]
    # Verificar chamadas ao cursor (check VSS, check count, SELECT VSS)
    assert mock_cursor.execute.call_count == 3
    # Verificar a chamada VSS search
    vss_call = mock_cursor.execute.call_args_list[2]
    assert "FROM vss_semantic_memory vss" in vss_call[0][0]
    assert vss_call[0][1] == (MOCK_EMBEDDING_BLOB, 2) # Query embedding e max_results
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_recall_memory_no_results(mock_db_connection, mock_embedding_functions):
    """Testa o fluxo quando a busca não retorna resultados."""
    mock_conn, mock_cursor = mock_db_connection
    mock_get_embedding_recall = mock_embedding_functions # Fixture returns recall mock

    action_input = {"query": "Find nothing"}

    # Simular VSS existe e não vazia, mas busca retorna lista vazia
    mock_cursor.fetchone.side_effect = [
        ('vss_semantic_memory',), # vss_table_exists = True
        (5,) # memory_count > 0
    ]
    mock_cursor.fetchall.return_value = []

    result = skill_recall_memory(action_input)

    assert result["status"] == "success"
    assert result["action"] == "memory_recalled"
    assert len(result["data"]["results"]) == 0
    assert "Nenhuma informação relevante encontrada" in result["data"]["message"]
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_recall_memory_vss_not_exists(mock_db_connection, mock_embedding_functions):
    """Testa falha se a tabela VSS não existir."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.fetchone.return_value = None # Simula vss_table_exists = False

    action_input = {"query": "Find test"}
    result = skill_recall_memory(action_input)

    assert result["status"] == "error"
    assert result["action"] == "recall_memory_failed"
    assert "Índice de busca semântica não está disponível" in result["data"]["message"]
    mock_conn.close.assert_called_once() # Conexão é fechada mesmo com erro

@pytest.mark.asyncio
async def test_recall_memory_db_empty(mock_db_connection, mock_embedding_functions):
    """Testa o caso onde a tabela semantic_memory está vazia."""
    mock_conn, mock_cursor = mock_db_connection

    # Simular VSS existe, mas count é 0
    mock_cursor.fetchone.side_effect = [
        ('vss_semantic_memory',), # vss_table_exists = True
        (0,) # memory_count = 0
    ]

    action_input = {"query": "Find test"}
    result = skill_recall_memory(action_input)

    assert result["status"] == "success" # Retorna sucesso, mas informa que está vazia
    assert result["action"] == "memory_recalled"
    assert len(result["data"]["results"]) == 0
    assert "Memória semântica está vazia" in result["data"]["message"]
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_recall_memory_missing_query(mock_db_connection, mock_embedding_functions):
    """Testa falha se 'query' estiver ausente."""
    action_input = {"max_results": 5} # Sem query
    result = skill_recall_memory(action_input)
    assert result["status"] == "error"
    assert result["action"] == "recall_memory_failed"
    assert "Parâmetro 'query' obrigatório ausente" in result["data"]["message"]

@pytest.mark.asyncio
async def test_recall_memory_embedding_error(mock_db_connection, mock_embedding_functions):
    """Testa falha se get_embedding para a query falhar."""
    mock_get_embedding_recall = mock_embedding_functions
    mock_get_embedding_recall.side_effect = Exception("Query embedding failed")
    action_input = {"query": "Find test"}
    result = skill_recall_memory(action_input)
    assert result["status"] == "error"
    assert result["action"] == "recall_memory_failed"
    assert "Erro inesperado ao gerar query embedding" in result["data"]["message"]

@pytest.mark.asyncio
async def test_recall_memory_db_error(mock_db_connection, mock_embedding_functions):
    """Testa falha se ocorrer erro no DB durante a busca."""
    mock_conn, mock_cursor = mock_db_connection
    # Simular VSS existe e não vazia
    mock_cursor.fetchone.side_effect = [
        ('vss_semantic_memory',), 
        (1,) 
    ]
    mock_cursor.execute.side_effect = [
        MagicMock(), # Chamada para check VSS
        MagicMock(), # Chamada para check count
        sqlite3.Error("VSS search failed") # Erro na busca VSS
    ]
        
    action_input = {"query": "Find test"}
    result = skill_recall_memory(action_input)

    # Ajustado para verificar o formato de erro padrão
    assert result.get("status") == "error"
    assert result.get("action") == "recall_memory_failed"
    assert "Erro inesperado ao buscar memória: VSS search failed" in result.get("data", {}).get("message", "")
    mock_conn.close.assert_called_once() # Deve fechar mesmo com erro
