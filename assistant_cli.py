import requests
import json
import os
from dotenv import load_dotenv
import argparse

# Carregar variáveis de ambiente (pode ser útil para futuras configs)
load_dotenv()

# Imports dos módulos core
from core.nlu import interpret_command
from core.nlg import generate_natural_response
from core.config import MAX_HISTORY_TURNS
from core.dispatcher import get_skill
from skills.manage_files import execute_delete_file

# Inicializar histórico de conversa
conversation_history = []

def create_nlu_prompt(command: str, history: list) -> str:
    """
    Cria um prompt básico para o NLU, incluindo histórico recente e exemplos essenciais.
    """
    prompt = "Analise o **Comando Atual** do usuário considerando o histórico e responda APENAS com JSON contendo \"intent\" e \"entities\".\n\n"
    
    # Adiciona histórico recente se houver
    if history:
        prompt += "### Histórico Recente da Conversa:\n"
        for entry in history[-3:]:  # Últimos 3 pares de interação
            prompt += f"{entry}\n"
        prompt += "\n"
    
    prompt += "### Exemplos Essenciais\n\n"
    
    # Exemplo de geração de código
    prompt += 'Comando: "gere um script python chamado utils.py com uma função hello world"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "generate_code",
  "entities": {
    "language": "python",
    "construct_type": "function",
    "purpose": "hello world"
  }
}
```\n\n'''
    
    # Exemplo de gerenciamento de arquivos
    prompt += 'Comando: "crie um arquivo vazio teste.txt"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "manage_files",
  "entities": {
    "action": "create",
    "file_name": "teste.txt",
    "content": null
  }
}
```\n\n'''
    
    # Exemplo de listagem de arquivos
    prompt += 'Comando: "liste os arquivos .py"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "manage_files",
  "entities": {
    "action": "list",
    "file_extension": ".py"
  }
}
```\n\n'''
    
    # Exemplo de previsão do tempo
    prompt += 'Comando: "qual a previsão do tempo para amanhã em Curitiba?"\n'
    prompt += "JSON Resultante:\n```json\n"
    prompt += '''{
  "intent": "weather_forecast",
  "entities": {
    "topic": "previsão do tempo",
    "timeframe": "amanhã",
    "location": "Curitiba"
  }
}
```\n\n'''
    
    # Adiciona o comando atual
    prompt += "### Comando Atual\n\n"
    prompt += f'Comando: "{command}"\n'
    prompt += "JSON Resultante:\n```json\n"
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description='Assistente CLI com habilidades de IA')
    parser.add_argument('-c', '--command', help='Comando único para executar')
    args = parser.parse_args()

    # Lista para armazenar o histórico da conversa
    conversation_history = []

    if args.command:
        # Modo comando único
        print(f"> {args.command}")
        conversation_history.append({"role": "user", "content": args.command})
        
        # Interpreta o comando
        interpretation = interpret_command(args.command, history=conversation_history)
        print(f"[DEBUG] Interpretação: {interpretation}")
        
        # Extrai intenção e entidades
        intent = interpretation.get("intent", "unknown")
        entities = interpretation.get("entities", {})
        
        # Obtém a função de skill apropriada
        skill_function = get_skill(intent)
        
        # Executa a skill
        skill_result = skill_function(entities, args.command, intent=intent, history=conversation_history)
        print(f"\n[Resultado da Skill (Estruturado)]:\n{json.dumps(skill_result, indent=2, ensure_ascii=False)}")
        
        # Gera resposta natural
        natural_response = generate_natural_response(skill_result, conversation_history)
        print(f"\n[Assistente]:\n{natural_response}")
        
        # Adiciona a resposta ao histórico com o resultado estruturado
        conversation_history.append({
            "role": "assistant",
            "content": natural_response,
            "skill_result": skill_result
        })
        
    else:
        # Modo interativo
        print("Assistente CLI iniciado. Digite 'sair' para encerrar.")
        while True:
            try:
                # Lê o comando do usuário
                command = input("\n> ").strip()
                
                # Verifica se deve sair
                if command.lower() in ['sair', 'exit', 'quit']:
                    print("Encerrando o assistente...")
                    break
                
                # Adiciona o comando ao histórico
                conversation_history.append({"role": "user", "content": command})
                
                # Interpreta o comando
                interpretation = interpret_command(command, history=conversation_history)
                print(f"[DEBUG] Interpretação: {interpretation}")
                
                # Extrai intenção e entidades
                intent = interpretation.get("intent", "unknown")
                entities = interpretation.get("entities", {})
                
                # Obtém a função de skill apropriada
                skill_function = get_skill(intent)
                
                # Executa a skill
                skill_result = skill_function(entities, command, intent=intent, history=conversation_history)
                print(f"\n[Resultado da Skill (Estruturado)]:\n{json.dumps(skill_result, indent=2, ensure_ascii=False)}")
                
                # Gera resposta natural
                natural_response = generate_natural_response(skill_result, conversation_history)
                print(f"\n[Assistente]:\n{natural_response}")
                
                # Adiciona a resposta ao histórico com o resultado estruturado
                conversation_history.append({
                    "role": "assistant",
                    "content": natural_response,
                    "skill_result": skill_result
                })
                
            except KeyboardInterrupt:
                print("\nEncerrando o assistente...")
                break
            except Exception as e:
                print(f"\n[Erro] Ocorreu um erro: {e}")
                continue

if __name__ == "__main__":
    main() 