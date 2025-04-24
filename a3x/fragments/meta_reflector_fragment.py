import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

class MetaReflectorFragment(BaseFragment):
    """
    Analisa a memória dos ciclos evolutivos passados, identifica padrões de decisões ruins, excesso de mutações inúteis,
    fragmentos obsoletos e sugere ajustes estratégicos para o ciclo evolutivo do A³X.
    """
    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        self._fragment_context: Optional[FragmentContext] = ctx
        self.logger = ctx.logger

    def set_context(self, context: FragmentContext):
        super().set_context(context)
        self._fragment_context = context

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """
        Retorna a descrição robusta e detalhada do propósito deste fragmento.
        """
        return (
            "Analisa a memória dos ciclos evolutivos passados, identifica padrões de decisões ruins, "
            "excesso de mutações inúteis, fragmentos obsoletos e sugere ajustes estratégicos para o ciclo evolutivo do A³X. "
            "Gera recomendações para promover estabilidade, eficiência e evolução contínua do sistema, "
            "baseando-se em evidências históricas e heurísticas extraídas dos resultados de execução dos fragmentos."
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Lê a memória dos ciclos passados, analisa padrões e gera sugestões de ajustes estratégicos.
        Pode retornar ActionIntent ou PendingRequest com capability_needed="strategy_adjustment".
        """
        ctx = self._fragment_context or kwargs.get("ctx")
        if ctx is None:
            self.logger.error("MetaReflectorFragment: contexto não fornecido.")
            return {"status": "error", "message": "Contexto não fornecido."}

        # --- Recupera histórico de ciclos evolutivos ---
        try:
            # Acesso robusto à memória compartilhada
            if hasattr(ctx, 'shared_task_context') and ctx.shared_task_context is not None:
                # Pode ser dict, MagicMock, ou objeto
                stc = ctx.shared_task_context
                if hasattr(stc, 'get') and callable(getattr(stc, 'get')):
                    ciclos = stc.get("evolution_cycles", [])
                elif hasattr(stc, 'evolution_cycles'):
                    ciclos = getattr(stc, 'evolution_cycles', [])
                else:
                    ciclos = []
            else:
                ciclos = []
        except Exception as e:
            self.logger.error(f"Erro ao acessar memória dos ciclos: {e}")
            ciclos = []

        # --- Analisa padrões ---
        mutacoes_inuteis = 0
        fragmentos_obsoletos = set()
        fragmentos_promovidos = defaultdict(int)
        fragmentos_desativados = set()
        decisoes_ruins = 0
        total_ciclos = len(ciclos)
        insights = []

        for ciclo in ciclos:
            # Exemplo de ciclo: {
            #   "mutacoes": ["fragA", ...],
            #   "avaliacoes": [ ... ],
            #   "promovidos": ["fragB"],
            #   "desativados": ["fragX"],
            #   "falhas": [ ... ]
            # }
            mutacoes = ciclo.get("mutacoes", [])
            avaliacoes = ciclo.get("avaliacoes", [])
            promovidos = ciclo.get("promovidos", [])
            desativados = ciclo.get("desativados", [])
            falhas = ciclo.get("falhas", [])

            if len(mutacoes) > 3 and len(promovidos) == 0:
                mutacoes_inuteis += 1
            for f in desativados:
                fragmentos_desativados.add(f)
            for f in promovidos:
                fragmentos_promovidos[f] += 1
            if len(falhas) > 2:
                decisoes_ruins += 1
            # Detecta fragmentos obsoletos (nunca promovidos, sempre desativados)
            for f in desativados:
                if fragmentos_promovidos[f] == 0:
                    fragmentos_obsoletos.add(f)

        # --- Gera sugestões de ajuste ---
        suggested_adjustments = []
        if mutacoes_inuteis > 2:
            suggested_adjustments.append("reduzir taxa de mutações")
        if decisoes_ruins > 1:
            suggested_adjustments.append("ajustar critérios de avaliação")
        for f in fragmentos_obsoletos:
            suggested_adjustments.append(f"desativar fragmento {f}")
        for f, count in fragmentos_promovidos.items():
            if count > 2:
                suggested_adjustments.append(f"promover fragmento {f} por consistência")

        insights.append({"suggested_adjustments": suggested_adjustments})

        # Pode retornar ActionIntent ou PendingRequest
        result = {
            "status": "success",
            "message": "Meta-reflexão concluída.",
            "insights": insights,
        }
        if suggested_adjustments:
            result["action_intent"] = {
                "capability_needed": "strategy_adjustment",
                "suggestions": suggested_adjustments
            }
        return result

MetaReflectorFragmentDef = FragmentDef(
    name="MetaReflector",
    description="Analisa a memória dos ciclos evolutivos e sugere ajustes estratégicos.",
    fragment_class=MetaReflectorFragment,
    skills=["refletir_ciclo"],
    managed_skills=["refletir_ciclo"],
    prompt_template="Analise os ciclos evolutivos e sugira ajustes estratégicos."
)
