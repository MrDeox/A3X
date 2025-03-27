import json
import os
# Garante que estamos no diretório raiz para imports relativos funcionarem
# (Ajuste se necessário, dependendo de onde o agente executa - ex: /home/arthur/Projects/A3X/)
try:
    # Tenta mudar para o diretório correto se não estiver lá
    if os.path.basename(os.getcwd()) != 'A3X':
         os.chdir('/home/arthur/Projects/A3X') # AJUSTE O CAMINHO SE NECESSÁRIO
         print(f"Diretório alterado para: {os.getcwd()}")
except FileNotFoundError:
    print("ERRO: Diretório raiz do projeto não encontrado. Ajuste o caminho no código.")
    exit()

print("--- Bloco 1: Importando e Gerando Código Inicial ---")
try:
    # Imports principais
    from core.nlu import interpret_command
    from core.nlg import generate_natural_response
    from core.dispatcher import get_skill
    from skills.manage_files import execute_delete_file # Apenas para garantir que importa
    print("Imports concluídos.")

    # Inicializa o histórico (SERÁ MANTIDO PELO AGENTE)
    conversation_history = []
    print("Histórico inicializado.")

    # Comando 1
    user_input_1 = 'gere uma função python chamada \'etapa1\' que imprime "Etapa 1"'
    conversation_history.append({"role": "user", "content": user_input_1})
    print(f"Comando 1: {user_input_1}")

    # Processamento 1
    interpretation_1 = interpret_command(user_input_1, conversation_history)
    print(f"[DEBUG] Interpretação 1: {json.dumps(interpretation_1, indent=2, ensure_ascii=False)}")
    intent_1 = interpretation_1.get("intent", "unknown")
    entities_1 = interpretation_1.get("entities", {})
    skill_function_1 = get_skill(intent_1)
    skill_result_1 = skill_function_1(entities_1, user_input_1, intent=intent_1, history=conversation_history)
    print(f"[DEBUG] Resultado Skill 1: {json.dumps(skill_result_1, indent=2, ensure_ascii=False)}")
    response_text_1 = generate_natural_response(skill_result_1, conversation_history)
    print(f"[DEBUG] Resposta NLG 1: {response_text_1}")

    # Atualiza histórico
    conversation_history.append({"role": "assistant", "content": response_text_1, "skill_result": skill_result_1})

    # Verifica se a ação foi bem-sucedida
    assert skill_result_1.get("status") == "success", "Bloco 1 falhou: Geração inicial não teve status 'success'"
    assert skill_result_1.get("action") == "code_generated", "Bloco 1 falhou: Ação não foi 'code_generated'"
    print("--- Bloco 1: OK ---")

except Exception as e:
    print(f"ERRO NO BLOCO 1: {e}")
    # Mostra o histórico mesmo em caso de erro
    print("\nHistórico (Bloco 1):")
    print(conversation_history)

# IMPORTANTE: A variável 'conversation_history' agora contém dados para o próximo bloco.

print("\n--- Bloco 2: Modificando Código via Histórico ---")
try:
    # Comando 2
    user_input_2 = 'agora adicione um print "Etapa 2" ao final dela'
    conversation_history.append({"role": "user", "content": user_input_2})
    print(f"Comando 2: {user_input_2}")

    # Processamento 2
    interpretation_2 = interpret_command(user_input_2, conversation_history)
    print(f"[DEBUG] Interpretação 2: {json.dumps(interpretation_2, indent=2, ensure_ascii=False)}")
    intent_2 = interpretation_2.get("intent", "unknown")
    entities_2 = interpretation_2.get("entities", {})
    skill_function_2 = get_skill(intent_2)
    skill_result_2 = skill_function_2(entities_2, user_input_2, intent=intent_2, history=conversation_history)
    print(f"[DEBUG] Resultado Skill 2: {json.dumps(skill_result_2, indent=2, ensure_ascii=False)}")
    response_text_2 = generate_natural_response(skill_result_2, conversation_history)
    print(f"[DEBUG] Resposta NLG 2: {response_text_2}")

    # Atualiza histórico
    conversation_history.append({"role": "assistant", "content": response_text_2, "skill_result": skill_result_2})

    # Verifica se a modificação foi bem-sucedida e contém o novo print
    assert skill_result_2.get("status") == "success", "Bloco 2 falhou: Modificação não teve status 'success'"
    assert skill_result_2.get("action") == "code_modified", "Bloco 2 falhou: Ação não foi 'code_modified'"
    assert 'print("Etapa 1")' in skill_result_2["data"]["modified_code"], "Bloco 2 falhou: Código original não encontrado no modificado"
    assert 'print("Etapa 2")' in skill_result_2["data"]["modified_code"], "Bloco 2 falhou: Modificação não encontrada no código final"
    print("--- Bloco 2: OK ---")

except Exception as e:
    print(f"ERRO NO BLOCO 2: {e}")
    # Mostra o histórico mesmo em caso de erro
    print("\nHistórico (Bloco 2):")
    print(conversation_history) 