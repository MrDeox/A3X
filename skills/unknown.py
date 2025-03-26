def skill_unknown(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Skill padrão para quando não entendemos o comando."""
    print("\n[Skill: Unknown]")
    print(f"  Comando não reconhecido: {original_command}")
    return {
        "status": "not_understood",
        "action": "unknown_command",
        "data": {
            "message": "Desculpe, não entendi o comando ou não sei como executá-lo ainda."
        }
    } 