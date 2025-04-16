import logging

from a3x.core.context import SharedTaskContext, Context
from a3x.skills.file_manager import FileManagerSkill
# Importar execute_code como função assíncrona diretamente
# from a3x.skills.execute_code import execute_code

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_skills_interaction():
    """
    Testa a interação entre read_file e execute_code usando SharedTaskContext.
    """
    logger.info("Iniciando teste de interação entre habilidades")
    
    # Inicializar o contexto compartilhado
    context = SharedTaskContext(task_id="test_task", initial_objective="Testar interação entre habilidades")
    context2 = Context()
    
    # Inicializar o gerenciador de arquivos
    file_manager = FileManagerSkill()
    
    # Criar um arquivo de teste
    test_file_path = "test.txt"
    test_content = "Este é um arquivo de teste para verificar a leitura."
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    logger.info(f"Arquivo de teste criado: {test_file_path}")
    
    # Testar leitura do arquivo
    read_result = await file_manager.read_file(path=test_file_path, shared_task_context=context)
    logger.info(f"Resultado da leitura: {read_result}")
    
    # Testar execução de código com placeholder
    test_code = "print('Caminho do último arquivo lido: ' + '$LAST_READ_FILE')"
    # Importar execute_code dinamicamente para evitar problemas de importação
    from a3x.skills.execute_code import execute_code
    # Chamar execute_code sem await, pois pode não ser assíncrono no contexto atual
    exec_result = execute_code(context=context2, code=test_code, shared_task_context=context)
    logger.info(f"Resultado da execução: {exec_result}")
    
    logger.info("Teste de interação concluído")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_skills_interaction()) 