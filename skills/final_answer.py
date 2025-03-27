def skill_final_answer(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """
    Fornece a resposta final (assinatura padrão).
    Esta skill é chamada pelo agente quando ele decide concluir.
    """
    print("\n[Skill: Final Answer]")
    # --- Use action_input ---
    answer = action_input.get("answer", "Não foi possível determinar a resposta final.")
    print(f"  Final Answer (from Action Input): {answer}")

    # This skill doesn't typically fail, it just delivers the LLM's decided answer.
    return {
        "status": "success", # Always success from the skill's perspective
        "action": "final_answer_provided", # Action name indicates purpose
        "data": {
            "answer": answer,
            "message": "Resposta final fornecida pelo agente." # Simple message
        }
    } 