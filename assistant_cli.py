import requests
import json
import os
from dotenv import load_dotenv
import argparse
import time
import traceback # Para debug de erros

# Carregar variáveis de ambiente
load_dotenv()

# Imports dos módulos core
# Removidos NLU, Planner, Dispatcher por enquanto
# from core.nlu import interpret_command
from core.nlg import generate_natural_response, generate_simplified_response # Mantém NLG por enquanto
from core.config import MAX_HISTORY_TURNS # Pode ser usado para histórico geral
# from core.dispatcher import get_skill, SKILL_DISPATCHER
# from core.planner import generate_plan
from core.agent import ReactAgent # <-- NOVO IMPORT

# Função process_command agora recebe a instância do agente
def process_command(agent: ReactAgent, command: str, conversation_history: list) -> None:
    """Processa um único comando usando a instância fornecida do Agente ReAct."""
    print(f"\n> {command}")
    conversation_history.append({"role": "user", "content": command})

    # NÃO instancia mais o agente aqui dentro

    final_response = ""
    agent_outcome = None

    try:
        # Usa a instância do agente passada como argumento
        final_response = agent.run(objective=command)
        agent_outcome = {"status": "success", "action": "react_cycle_completed", "data": {"message": final_response}}
        print(f"\n[CLI] Agente concluiu. Resposta final: {final_response}")

    except Exception as e:
        print(f"[Erro Fatal Agent] Erro ao executar agente: {e}")
        traceback.print_exc() # Imprime traceback completo para debug
        final_response = f"Desculpe, ocorreu um erro interno grave ao processar seu comando."
        agent_outcome = {"status": "error", "action": "react_cycle_failed", "data": {"message": str(e)}}

    # --- LÓGICA DE RESPOSTA (NLG) ---
    # Usamos a resposta final direta do agente
    response_text = final_response
    print(f"[A³X]: {response_text}")
    conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "agent_outcome": agent_outcome # Guarda o resultado do agente para referência
    })
    # ... (limitar histórico se necessário) ...


def main():
    # Garante execução no diretório raiz do projeto
    project_root = "/home/arthur/Projects/A3X" # AJUSTE SE NECESSÁRIO
    try:
        os.chdir(project_root)
        print(f"[Info] Executando em: {os.getcwd()}")
    except FileNotFoundError:
        print(f"[Erro Fatal] Diretório do projeto não encontrado: {project_root}")
        exit(1)

    parser = argparse.ArgumentParser(description='Assistente CLI A³X (ReAct)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')

    args = parser.parse_args()

    conversation_history = [] # Histórico da conversa geral

    # <<< MOVER INSTANCIAÇÃO PARA CÁ >>>
    print("[Info] Inicializando Agente ReAct...")
    agent = ReactAgent() # Instancia o agente UMA VEZ AQUI
    print("[Info] Agente pronto.")
    # <<< FIM DA MUDANÇA >>>

    if args.command:
        # Modo comando único
        process_command(agent, args.command, conversation_history) # Passa a instância agent

    elif args.input_file:
        # Modo arquivo de entrada
        try:
            print(f"[Info] Lendo comandos de: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"[Info] Encontrados {len(commands)} comandos para processar.")

                for line_num, command in enumerate(commands, 1):
                    print(f"\n--- Comando da Linha {line_num} ---")
                    process_command(agent, command, conversation_history) # Passa a MESMA instância agent
                    time.sleep(1) # Pausa opcional
            print("\n[Info] Fim do arquivo de entrada.")
        except FileNotFoundError:
            print(f"[Erro Fatal] Arquivo de entrada não encontrado: {args.input_file}")
        except Exception as e:
            print(f"\n[Erro Fatal] Ocorreu um erro ao processar o arquivo '{args.input_file}': {e}")
            traceback.print_exc()

    else:
        # Modo interativo
        print("Assistente CLI A³X (ReAct) iniciado. Digite 'sair' para encerrar.")
        while True:
            try:
                command = input("\n> ").strip()
                if command.lower() in ['sair', 'exit', 'quit']:
                    print("Encerrando o assistente...")
                    break
                if not command:
                    continue
                process_command(agent, command, conversation_history) # Passa a MESMA instância agent
            except KeyboardInterrupt:
                print("\nEncerrando o assistente...")
                break
            except Exception as e:
                print(f"\n[Erro Inesperado no Loop Principal] {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main() 