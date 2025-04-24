import logging
from typing import Dict, Any, Optional, List
from a3x.core.skills import skill
from a3x.fragments.base import FragmentContext

@skill(
    name="reorganizar_fragmentos",
    description="Aplica reorganização dos fragmentos com base nas avaliações e gera comandos A3L para promoção/arquivamento.",
    parameters={
        "type": "object",
        "properties": {
            "avaliacoes": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Lista de avaliações dos fragmentos. Cada avaliação deve conter nome e score."
            }
        },
        "required": ["avaliacoes"]
    }
)
async def reorganizar_fragmentos(ctx: FragmentContext, avaliacoes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aplica a lógica de reorganização dos fragmentos com base nas avaliações fornecidas.
    Gera comandos A3L para promover ou arquivar fragmentos conforme necessário.

    Args:
        ctx: Contexto do fragmento.
        avaliacoes: Lista de avaliações dos fragmentos.

    Returns:
        Dict contendo os comandos A3L gerados e um sumário da reorganização.
    """
    logger = logging.getLogger(__name__)
    try:
        promover = []
        arquivar = []
        # Exemplo simples: promover os top 1-2, arquivar os últimos
        avaliacoes_ordenadas = sorted(avaliacoes, key=lambda x: x.get("score", 0), reverse=True)
        if not avaliacoes_ordenadas:
            return {"status": "error", "message": "Nenhuma avaliação fornecida."}
        # Promover o melhor
        promover.append(avaliacoes_ordenadas[0]["nome"])
        # Opcional: promover o segundo se score acima de threshold
        if len(avaliacoes_ordenadas) > 1 and avaliacoes_ordenadas[1]["score"] > 0.8 * avaliacoes_ordenadas[0]["score"]:
            promover.append(avaliacoes_ordenadas[1]["nome"])
        # Arquivar o pior
        if len(avaliacoes_ordenadas) > 2:
            arquivar.append(avaliacoes_ordenadas[-1]["nome"])
        # Gerar comandos A3L
        a3l_commands = []
        for nome in promover:
            a3l_commands.append(f"promover fragmento '{nome}'")
        for nome in arquivar:
            a3l_commands.append(f"arquivar fragmento '{nome}'")
        summary = f"Promovidos: {promover}. Arquivados: {arquivar}."
        logger.info(summary)
        return {
            "status": "success",
            "a3l_commands": a3l_commands,
            "summary": summary
        }
    except Exception as e:
        logger.exception("Erro ao reorganizar fragmentos:")
        return {"status": "error", "message": str(e)}
