import os
import json
from core.nlu import interpret_command
from core.nlg import generate_natural_response
from core.dispatcher import get_skill

# Setup inicial
try:
    os.chdir("/home/arthur/Projects/A3X/")
except FileNotFoundError:
    print("Erro: Diretório do projeto não encontrado!")
    exit(1)

conversation_history = []
print("\n=== Iniciando Bloco 1: Geração Inicial ===")

# Bloco 1: Geração Inicial
try:
    user_input_1 = 'gere uma função python chamada step1 que imprime "Passo 1 concluído"'
    conversation_history.append({"role": "user", "content": user_input_1})
    
    # Processamento do comando 1
    nlu_result_1 = interpret_command(user_input_1, conversation_history)
    intent_1 = nlu_result_1["intent"]
    entities_1 = nlu_result_1["entities"]
    
    skill_function_1 = get_skill(intent_1)
    skill_result_1 = skill_function_1(entities_1, user_input_1, intent=intent_1, history=conversation_history)
    response_1 = generate_natural_response(skill_result_1, conversation_history)
    
    conversation_history.append({
        "role": "assistant",
        "content": response_1,
        "skill_result": skill_result_1
    })
    
    # Verificações do Bloco 1
    assert skill_result_1.get("status") == "success", "Falha no status do Bloco 1"
    assert skill_result_1.get("action") == "code_generated", "Falha na ação do Bloco 1"
    assert 'print("Passo 1 concluído")' in skill_result_1['data']['code'], "Código do Bloco 1 não contém o print esperado"
    
    print("✓ Bloco 1 OK")
    print("\n=== Iniciando Bloco 2: Primeira Modificação ===")
    
except Exception as e:
    print(f"\nErro no Bloco 1: {e}")
    print("\nHistórico atual:")
    print(json.dumps(conversation_history, indent=2))
    exit(1)

# Bloco 2: Primeira Modificação
try:
    user_input_2 = 'agora adicione um print "Passo 2 adicionado" no final dela'
    conversation_history.append({"role": "user", "content": user_input_2})
    
    # Processamento do comando 2
    nlu_result_2 = interpret_command(user_input_2, conversation_history)
    intent_2 = nlu_result_2["intent"]
    entities_2 = nlu_result_2["entities"]
    
    skill_function_2 = get_skill(intent_2)
    skill_result_2 = skill_function_2(entities_2, user_input_2, intent=intent_2, history=conversation_history)
    response_2 = generate_natural_response(skill_result_2, conversation_history)
    
    conversation_history.append({
        "role": "assistant",
        "content": response_2,
        "skill_result": skill_result_2
    })
    
    # Verificações do Bloco 2
    assert skill_result_2.get("status") == "success", "Falha no status do Bloco 2"
    assert skill_result_2.get("action") == "code_modified", "Falha na ação do Bloco 2"
    assert 'print("Passo 1 concluído")' in skill_result_2['data']['modified_code'], "Código do Bloco 2 não mantém o print do Passo 1"
    assert 'print("Passo 2 adicionado")' in skill_result_2['data']['modified_code'], "Código do Bloco 2 não contém o print do Passo 2"
    
    print("✓ Bloco 2 OK")
    print("\n=== Iniciando Bloco 3: Segunda Modificação (Encadeada) ===")
    
except Exception as e:
    print(f"\nErro no Bloco 2: {e}")
    print("\nHistórico atual:")
    print(json.dumps(conversation_history, indent=2))
    exit(1)

# Bloco 3: Segunda Modificação (Encadeada)
try:
    user_input_3 = 'por fim, envolva o conteúdo da função com um try/except generico que imprime "Erro ocorreu"'
    conversation_history.append({"role": "user", "content": user_input_3})
    
    # Processamento do comando 3
    nlu_result_3 = interpret_command(user_input_3, conversation_history)
    intent_3 = nlu_result_3["intent"]
    entities_3 = nlu_result_3["entities"]
    
    skill_function_3 = get_skill(intent_3)
    skill_result_3 = skill_function_3(entities_3, user_input_3, intent=intent_3, history=conversation_history)
    response_3 = generate_natural_response(skill_result_3, conversation_history)
    
    conversation_history.append({
        "role": "assistant",
        "content": response_3,
        "skill_result": skill_result_3
    })
    
    # Verificações do Bloco 3
    assert skill_result_3.get("status") == "success", "Falha no status do Bloco 3"
    assert skill_result_3.get("action") == "code_modified", "Falha na ação do Bloco 3"
    
    modified_code = skill_result_3['data']['modified_code']
    assert 'try:' in modified_code, "Código do Bloco 3 não contém try"
    assert 'except' in modified_code, "Código do Bloco 3 não contém except"
    assert 'print("Erro ocorreu")' in modified_code, "Código do Bloco 3 não contém mensagem de erro"
    assert 'print("Passo 1 concluído")' in modified_code, "Código do Bloco 3 não mantém o print do Passo 1"
    assert 'print("Passo 2 adicionado")' in modified_code, "Código do Bloco 3 não mantém o print do Passo 2"
    
    print("✓ Bloco 3 OK")
    print("\n=== Teste de Modificação Encadeada Concluído com Sucesso! ===")
    
except Exception as e:
    print(f"\nErro no Bloco 3: {e}")
    print("\nHistórico atual:")
    print(json.dumps(conversation_history, indent=2))
    exit(1) 