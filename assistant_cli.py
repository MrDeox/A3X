import requests
import json
import os
from dotenv import load_dotenv
import argparse
import time

# Carregar variáveis de ambiente (pode ser útil para futuras configs)
load_dotenv()

# Imports dos módulos core
from core.nlu import interpret_command
from core.nlg import generate_natural_response
from core.config import MAX_HISTORY_TURNS
from core.dispatcher import get_skill, SKILL_DISPATCHER
from core.planner import generate_plan

def process_command(command: str, conversation_history: list, is_sequence: bool = False) -> None:
    """
    Processa um comando do usuário, incluindo planejamento sequencial se necessário.
    
    Args:
        command: O comando a ser processado
        conversation_history: Histórico da conversa
        is_sequence: Se True, indica que este comando faz parte de uma sequência
    """
    print(f"\n> {command}") # Ecoa o comando recebido
    conversation_history.append({"role": "user", "content": command})

    # NLU
    print("[DEBUG] NLU: Analisando comando...")
    nlu_result = interpret_command(command, history=conversation_history)
    print(f"[DEBUG] NLU: Resultado: {json.dumps(nlu_result, indent=2, ensure_ascii=False)}")

    # Obter skills disponíveis
    available_skills = [name for name in SKILL_DISPATCHER.keys() if "error" not in name and "unknown" not in name]

    # Planner
    print("[Planner] Verificando necessidade de plano...")
    # Se for parte de uma sequência, força o planejamento
    if is_sequence:
        print("[Planner] Comando faz parte de uma sequência. Forçando planejamento...")
        nlu_result["intent"] = "unknown" # Força intenção genérica para garantir planejamento
    plan = generate_plan(command, nlu_result, conversation_history, available_skills)

    # Execução: Plano ou Ação Única
    if plan: # Se o planner retornou uma lista de passos
        print(f"[Planner] Executando plano com {len(plan)} passos...")
        plan_successful = True # Flag para rastrear sucesso do plano
        for i, step in enumerate(plan):
            step_intent = step.get("intent")
            step_entities = step.get("entities", {})
            print(f"\n--- Executando Passo {i+1}/{len(plan)}: {step_intent} ---")

            if not step_intent or step_intent not in SKILL_DISPATCHER:
                print(f"[Erro] Skill inválida no plano: {step_intent}")
                plan_successful = False
                break # Aborta o plano

            skill_function = get_skill(step_intent) # USA get_skill
            try:
                skill_result = skill_function(
                    step_entities,
                    f"Passo {i+1} do plano: {step_intent}", # Comando original descritivo
                    intent=step_intent,
                    history=conversation_history # Passa o histórico atual
                )
                print(f"[DEBUG] Skill: Resultado: {json.dumps(skill_result, indent=2, ensure_ascii=False)}")

                response_text = "[Falha na NLG]" # Default
                if skill_result.get("status") == "success":
                     print("[NLG] Gerando resposta do passo...")
                     response_text = generate_natural_response(skill_result, conversation_history)
                     print(f"[A³X - Passo {i+1}/{len(plan)}]: {response_text}")
                     conversation_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "skill_result": skill_result
                     })
                elif skill_result.get("status") == "confirmation_required":
                     print("[Confirmação Necessária] A skill requer confirmação.")
                     response_text = skill_result.get("data", {}).get("confirmation_prompt", "Confirmação necessária.")
                     print(f"[A³X - Passo {i+1}/{len(plan)}]: {response_text}")
                     plan_successful = False
                     break
                else: # Erro no passo
                    print(f"[Erro] Falha ao executar passo {i+1}: {skill_result.get('data', {}).get('message', 'Erro desconhecido')}")
                    response_text = generate_natural_response(skill_result, conversation_history)
                    print(f"[A³X - Passo {i+1}/{len(plan)}]: {response_text}")
                    conversation_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "skill_result": skill_result
                    })
                    plan_successful = False
                    break

            except Exception as e_skill:
                print(f"[Erro Fatal Skill] Erro inesperado ao executar skill '{step_intent}': {e_skill}")
                skill_result = {"status": "error", "action": f"{step_intent}_failed", "data": {"message": f"Erro inesperado na skill: {e_skill}"}}
                response_text = generate_natural_response(skill_result, conversation_history)
                print(f"[A³X - Erro Plano]: {response_text}")
                conversation_history.append({"role": "assistant", "content": response_text, "skill_result": skill_result})
                plan_successful = False
                break

        if plan_successful:
            print("\n--- Plano concluído com sucesso! ---")
        else:
            print("\n--- Plano interrompido. ---")

    else: # Se o planner retornou [], executa ação única
        print("[Planner] Executando como ação única...")
        intent = nlu_result.get("intent", "unknown")
        entities = nlu_result.get("entities", {})

        skill_function = get_skill(intent) # USA get_skill
        try:
            skill_result = skill_function(entities, command, intent=intent, history=conversation_history)
            print(f"[DEBUG] Skill: Resultado: {json.dumps(skill_result, indent=2, ensure_ascii=False)}")

            response_text = "[Falha na NLG]"
            if skill_result.get("status") == "confirmation_required":
                 print("[Confirmação Necessária] A skill requer confirmação.")
                 response_text = skill_result.get("data", {}).get("confirmation_prompt", "Confirmação necessária.")
                 print(f"[A³X]: {response_text}")
                 # Não atualiza histórico até confirmação
            else:
                print("[NLG] Gerando resposta...")
                response_text = generate_natural_response(skill_result, conversation_history)
                print(f"[A³X]: {response_text}")
                conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "skill_result": skill_result
                })

        except Exception as e_skill_single:
            print(f"[Erro Fatal Skill] Erro inesperado ao executar skill '{intent}': {e_skill_single}")
            skill_result = {"status": "error", "action": f"{intent}_failed", "data": {"message": f"Erro inesperado na skill: {e_skill_single}"}}
            response_text = generate_natural_response(skill_result, conversation_history)
            print(f"[A³X - Erro]: {response_text}")
            conversation_history.append({"role": "assistant", "content": response_text, "skill_result": skill_result})

    # Limitar tamanho do histórico (OPCIONAL) - Descomente se quiser
    # if len(conversation_history) > MAX_HISTORY_TURNS * 2:
    #     print("[DEBUG] Histórico antigo removido.")
    #     keep_turns = MAX_HISTORY_TURNS * 2
    #     conversation_history[:] = conversation_history[-keep_turns:]

def main():
    # Garante execução no diretório raiz do projeto
    project_root = "/home/arthur/Projects/A3X" # AJUSTE SE NECESSÁRIO
    try:
        os.chdir(project_root)
        print(f"[Info] Executando em: {os.getcwd()}")
    except FileNotFoundError:
        print(f"[Erro Fatal] Diretório do projeto não encontrado: {project_root}")
        exit(1)

    parser = argparse.ArgumentParser(description='Assistente CLI A³X')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')

    args = parser.parse_args()

    conversation_history = []

    if args.command:
        # Modo comando único
        process_command(args.command, conversation_history)

    elif args.input_file:
        # Modo arquivo de entrada
        try:
            print(f"[Info] Lendo comandos de: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"[Info] Encontrados {len(commands)} comandos para processar.")
                
                for line_num, command in enumerate(commands, 1):
                    print(f"\n--- Comando da Linha {line_num} ---")
                    # Marca como sequência se não for o último comando
                    is_sequence = line_num < len(commands)
                    process_command(command, conversation_history, is_sequence)
                    time.sleep(1) # Pausa opcional
            print("\n[Info] Fim do arquivo de entrada.")
        except FileNotFoundError:
            print(f"[Erro Fatal] Arquivo de entrada não encontrado: {args.input_file}")
        except Exception as e:
            print(f"\n[Erro Fatal] Ocorreu um erro ao processar o arquivo '{args.input_file}': {e}")

    else:
        # Modo interativo
        print("Assistente CLI A³X iniciado. Digite 'sair' para encerrar.")
        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() in ['sair', 'exit', 'quit']:
                    print("Encerrando o assistente...")
                    break
                if not command:
                    continue
                process_command(command, conversation_history)
            except KeyboardInterrupt:
                print("\nEncerrando o assistente...")
                break
            except Exception as e:
                print(f"\n[Erro Inesperado] {e}")
                # Opcional: adicionar mais detalhes/traceback aqui se necessário
                # traceback.print_exc()

if __name__ == "__main__":
    main() 