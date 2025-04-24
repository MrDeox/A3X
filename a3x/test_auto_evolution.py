import asyncio
from a3x.fragments.definitions import (
    MetaReflectorFragment,
    EvolutionOrchestratorFragment,
    GenerativeMutatorFragment,
    AggregateEvaluatorFragment,
    PerformanceManagerFragment
)
from a3x.core.context import FragmentContext

# Simulação de contexto persistente compartilhado
import logging

from a3x.fragments.meta_reflector_fragment import MetaReflectorFragment, MetaReflectorFragmentDef

from a3x.core.context import SharedTaskContext, FragmentContext
from a3x.core.llm_interface import LLMInterface
from a3x.core.tool_registry import ToolRegistry
from a3x.core.tool_executor import ToolExecutor
from a3x.fragments.registry import FragmentRegistry
from a3x.core.memory.memory_manager import MemoryManager
import os
import a3x.core.config as config_mod

# Inicializa recursos reais
logger = logging.getLogger("test_auto_evolution")
llm_interface = LLMInterface()
tool_registry = ToolRegistry()
fragment_registry = FragmentRegistry()

config_dict = {
    "SEMANTIC_INDEX_PATH": getattr(config_mod, "SEMANTIC_INDEX_PATH", None),
    "DATABASE_PATH": getattr(config_mod, "DATABASE_PATH", None),
    "EPISODIC_MEMORY_LIMIT": getattr(config_mod, "EPISODIC_RETRIEVAL_LIMIT", 10),
    "HEURISTIC_LOG_PATH": getattr(config_mod, "HEURISTIC_LOG_FILE", None),
    # Adicione outros campos obrigatórios conforme necessário
}
memory_manager = MemoryManager(config=config_dict)

# Inicializa ToolExecutor real e injeta no contexto compartilhado para robustez simbólica
tool_executor = ToolExecutor(tool_registry=ToolRegistry())  # <<< Instanciação robusta >>>

# Inicializa contexto compartilhado real
shared_task_context = SharedTaskContext(
    task_id="test-task-001",
    initial_objective="Testar ciclo robusto de autoevolução",
    tool_executor=tool_executor  # <<< Injeta ToolExecutor real >>>
)

class FragmentContextWithSymbolic(FragmentContext):
    async def run_symbolic_command(self, command: str, **kwargs):
        if hasattr(self.shared_task_context, "run_symbolic_command"):
            return await self.shared_task_context.run_symbolic_command(command, **kwargs)
        raise NotImplementedError("Underlying shared_task_context does not support symbolic command execution.")

shared_ctx = FragmentContextWithSymbolic(
    logger=logger,
    llm_interface=llm_interface,
    tool_registry=tool_registry,
    fragment_registry=fragment_registry,
    shared_task_context=shared_task_context,
    workspace_root=os.getcwd(),
    memory_manager=memory_manager,
    fragment_id="MetaReflector",
    fragment_name="MetaReflector",
    fragment_class=MetaReflectorFragment,
    fragment_def=MetaReflectorFragmentDef,
    config={}
)

async def run_meta_reflector():
    # Injeta ciclos simulados na memória
    shared_ctx.shared_task_context.evolution_cycles = [
        {"mutacoes": ["fragA"], "avaliacoes": [], "promovidos": [], "desativados": ["fragX"], "falhas": ["erro1", "erro2", "erro3"]}
    ]
    frag = MetaReflectorFragment(shared_ctx)
    result = await frag.execute()
    print("MetaReflectorFragment result:", result)
    assert "insights" in result and isinstance(result["insights"], list)
    assert "suggested_adjustments" in result["insights"][0]

async def run_evolution_cycle():
    frag = EvolutionOrchestratorFragment(shared_ctx)
    result = await frag.execute()
    print("EvolutionOrchestratorFragment result:", result)
    # Espera promover ou arquivar algum fragmento
    assert "cycle_summary" in result

async def run_mutator():
    frag = GenerativeMutatorFragment(None)
    frag.set_context(shared_ctx)
    result = await frag.execute(fragment_to_expand="fragA")
    print("GenerativeMutatorFragment result:", result)
    assert "mutations" in result

async def run_evaluator():
    frag = AggregateEvaluatorFragment(None)
    frag.set_context(shared_ctx)
    result = await frag.execute(fragments=["fragA", "fragA_mut_1"])
    print("AggregateEvaluatorFragment result:", result)
    assert "evaluation_summary" in result

async def run_performance_manager():
    frag = PerformanceManagerFragment(None)
    frag.set_context(shared_ctx)
    result = await frag.execute()
    print("PerformanceManagerFragment result:", result)
    assert "action_intent" in result

async def main():
    print("\n==== Testando MetaReflectorFragment ====")
    await run_meta_reflector()
    print("\n==== Testando EvolutionOrchestratorFragment ====")
    await run_evolution_cycle()
    print("\n==== Testando GenerativeMutatorFragment ====")
    await run_mutator()
    print("\n==== Testando AggregateEvaluatorFragment ====")
    await run_evaluator()
    print("\n==== Testando PerformanceManagerFragment ====")
    await run_performance_manager()
    print("\n==== Teste de autoevolução completo! ====")

if __name__ == "__main__":
    asyncio.run(main())
