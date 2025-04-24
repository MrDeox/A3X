import logging
from typing import List, Dict, Any, Optional

# Core A3X Imports (adjust based on actual project structure)
try:
    from a3x.fragments.base import BaseFragment, FragmentContext, FragmentDef
    from a3x.fragments.registry import fragment
    from a3x.core.memory.memory_manager import MemoryManager
except ImportError as e:
    print(f"[EvolutionPlannerFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        memory: Any = None
        tool_registry: Any = None
        llm: Any = None
        workspace_root: str = "."

    class BaseFragment:
        def __init__(self, *args, **kwargs): pass
    # Placeholder for the decorator if the real one fails to import
    if 'fragment' not in locals():
        def fragment(*args, **kwargs):
            def decorator(cls): return cls
            return decorator

logger = logging.getLogger(__name__)

@fragment(
    name="evolution_planner",
    description="Analyzes system state and proposes evolution steps",
    category="evolution",
    skills=["analyze_state", "plan_steps", "validate_plan"]
)
class EvolutionPlannerFragment(BaseFragment):
    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        self._logger.info(f"EvolutionPlannerFragment '{self.get_name()}' initialized.")

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "Analyzes system state (performance, errors, dependencies) and proposes concrete evolution steps (e.g., A3L commands, code changes) to improve the system."

    async def execute(self, **kwargs) -> Dict[str, Any]:
        ctx = kwargs.get('ctx')
        if not ctx:
            self._logger.error("FragmentContext (ctx) not provided to execute method.")
            return {"status": "error", "error": "Context not provided"}
        
        try:
            # Analyze current system state
            state = await self._analyze_state()
            
            # Plan evolution steps
            plan = await self._plan_steps(state)
            
            # Validate the plan
            validation = await self._validate_plan(plan)
            
            return {
                "state": state,
                "plan": plan,
                "validation": validation
            }
            
        except Exception as e:
            self._logger.error(f"Error in EvolutionPlannerFragment: {str(e)}")
            raise

    async def _analyze_state(self) -> Dict[str, Any]:
        # TODO: Implement state analysis logic
        pass

    async def _plan_steps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement step planning logic
        pass

    async def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement plan validation logic
        pass

    async def _analyze_system_state(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Analisa o estado atual do sistema."""
        try:
            # Obtém estado dos fragments
            fragments_state = await self._get_fragments_state(ctx)
            
            # Analisa performance
            performance_metrics = await self._analyze_performance(ctx)
            
            # Analisa dependências
            dependencies = await self._analyze_dependencies(ctx)
            
            # Identifica gargalos
            bottlenecks = await self._identify_bottlenecks(ctx, fragments_state, performance_metrics)
            
            return {
                "status": "success",
                "fragments_state": fragments_state,
                "performance_metrics": performance_metrics,
                "dependencies": dependencies,
                "bottlenecks": bottlenecks
            }
            
        except Exception as e:
            logger.exception("Erro durante análise do estado do sistema:")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _get_fragments_state(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Obtém estado atual dos fragments."""
        state = {}
        
        for fragment in ctx.fragment_registry.get_fragments():
            # Obtém métricas do fragment
            metrics = await self._get_fragment_metrics(ctx, fragment)
            
            # Obtém histórico recente
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=50
            )
            
            # Calcula estatísticas
            stats = self._calculate_fragment_stats(history)
            
            state[fragment.name] = {
                "metrics": metrics,
                "stats": stats,
                "is_active": fragment.is_active
            }
            
        return state

    async def _get_fragment_metrics(self, ctx: FragmentContext, fragment: Any) -> Dict[str, Any]:
        """Obtém métricas de um fragment específico."""
        try:
            # Obtém histórico de execução
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=100
            )
            
            if not history:
                return {
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "error_rate": 0.0
                }
                
            # Calcula métricas
            success_count = sum(1 for h in history if h.get("status") == "success")
            error_count = sum(1 for h in history if h.get("status") == "error")
            total_count = len(history)
            
            execution_times = [h.get("execution_time", 0.0) for h in history]
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            return {
                "success_rate": success_count / total_count,
                "avg_execution_time": avg_execution_time,
                "error_rate": error_count / total_count
            }

        except Exception as e:
            logger.exception(f"Erro ao obter métricas do fragment {fragment.name}:")
            return {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "error_rate": 1.0
            }

    def _calculate_fragment_stats(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estatísticas do histórico de um fragment."""
        if not history:
            return {
                "total_executions": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_input_size": 0,
                "avg_output_size": 0
            }
            
        # Conta execuções
        total_executions = len(history)
        success_count = sum(1 for h in history if h.get("status") == "success")
        error_count = sum(1 for h in history if h.get("status") == "error")
        
        # Calcula tamanhos médios
        input_sizes = [len(str(h.get("input", ""))) for h in history]
        output_sizes = [len(str(h.get("output", ""))) for h in history]
        
        return {
            "total_executions": total_executions,
            "success_count": success_count,
            "error_count": error_count,
            "avg_input_size": sum(input_sizes) / len(input_sizes),
            "avg_output_size": sum(output_sizes) / len(output_sizes)
        }

    async def _analyze_performance(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Analisa performance geral do sistema."""
        try:
            # Obtém métricas de todos os fragments
            fragments_metrics = {}
            for fragment in ctx.fragment_registry.get_fragments():
                metrics = await self._get_fragment_metrics(ctx, fragment)
                fragments_metrics[fragment.name] = metrics
                
            # Calcula métricas globais
            global_metrics = self._calculate_global_metrics(fragments_metrics)
            
            # Identifica gargalos
            bottlenecks = self._identify_performance_bottlenecks(fragments_metrics)
            
            return {
                "fragments_metrics": fragments_metrics,
                "global_metrics": global_metrics,
                "bottlenecks": bottlenecks
            }

        except Exception as e:
            logger.exception("Erro durante análise de performance:")
            return {
                "fragments_metrics": {},
                "global_metrics": {},
                "bottlenecks": []
            }

    def _calculate_global_metrics(self, fragments_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula métricas globais do sistema."""
        if not fragments_metrics:
            return {
                "avg_success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_error_rate": 0.0,
                "system_health": 0.0
            }
            
        # Calcula médias
        success_rates = [m["success_rate"] for m in fragments_metrics.values()]
        execution_times = [m["avg_execution_time"] for m in fragments_metrics.values()]
        error_rates = [m["error_rate"] for m in fragments_metrics.values()]
        
        avg_success_rate = sum(success_rates) / len(success_rates)
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_error_rate = sum(error_rates) / len(error_rates)
        
        # Calcula saúde do sistema
        system_health = (avg_success_rate * 0.6) + ((1 - avg_error_rate) * 0.4)
        
        return {
            "avg_success_rate": avg_success_rate,
            "avg_execution_time": avg_execution_time,
            "avg_error_rate": avg_error_rate,
            "system_health": system_health
        }

    def _identify_performance_bottlenecks(self, fragments_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identifica gargalos de performance."""
        bottlenecks = []
        
        for fragment_name, metrics in fragments_metrics.items():
            # Verifica sucesso
            if metrics["success_rate"] < 0.7:
                bottlenecks.append(f"Baixa taxa de sucesso em {fragment_name}")
                
            # Verifica tempo de execução
            if metrics["avg_execution_time"] > 5.0:  # 5 segundos
                bottlenecks.append(f"Alto tempo de execução em {fragment_name}")
                
            # Verifica erros
            if metrics["error_rate"] > 0.3:
                bottlenecks.append(f"Alta taxa de erro em {fragment_name}")
                
        return bottlenecks

    async def _analyze_dependencies(self, ctx: FragmentContext) -> Dict[str, List[str]]:
        """Analisa dependências entre fragments."""
        dependencies = {}
        
        for fragment in ctx.fragment_registry.get_fragments():
            # Obtém dependências do fragment
            fragment_deps = await self._get_fragment_dependencies(ctx, fragment)
            dependencies[fragment.name] = fragment_deps
            
        return dependencies

    async def _get_fragment_dependencies(self, ctx: FragmentContext, fragment: Any) -> List[str]:
        """Obtém dependências de um fragment específico."""
        try:
            # Obtém histórico de execução
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=100
            )
            
            if not history:
                return []
                
            # Extrai dependências do histórico
            dependencies = set()
            for h in history:
                deps = h.get("dependencies", [])
                if isinstance(deps, list):
                    dependencies.update(deps)
                    
            return list(dependencies)
            
        except Exception as e:
            logger.exception(f"Erro ao obter dependências do fragment {fragment.name}:")
            return []

    async def _identify_bottlenecks(self, ctx: FragmentContext, fragments_state: Dict[str, Any], performance_metrics: Dict[str, Any]) -> List[str]:
        """Identifica gargalos no sistema."""
        bottlenecks = []
        
        # Verifica fragments inativos
        for fragment_name, state in fragments_state.items():
            if not state["is_active"]:
                bottlenecks.append(f"Fragment {fragment_name} está inativo")
                
        # Verifica métricas de performance
        global_metrics = performance_metrics.get("global_metrics", {})
        if global_metrics.get("system_health", 0.0) < 0.7:
            bottlenecks.append("Saúde geral do sistema abaixo do esperado")
            
        # Verifica gargalos de performance
        bottlenecks.extend(performance_metrics.get("bottlenecks", []))
        
        return bottlenecks

    async def _generate_evolution_steps(self, ctx: FragmentContext, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera passos de evolução baseados na análise."""
        try:
            steps = []
            
            # Extrai dados da análise
            fragments_state = analysis.get("fragments_state", {})
            performance_metrics = analysis.get("performance_metrics", {})
            dependencies = analysis.get("dependencies", {})
            bottlenecks = analysis.get("bottlenecks", [])
            
            # Gera passos para fragments inativos
            for fragment_name, state in fragments_state.items():
                if not state["is_active"]:
                    steps.append({
                        "type": "activate_fragment",
                        "fragment": fragment_name,
                        "priority": "high"
                    })
                    
            # Gera passos para gargalos de performance
            for bottleneck in bottlenecks:
                if "Baixa taxa de sucesso" in bottleneck:
                    fragment = bottleneck.split()[-1]
                    steps.append({
                        "type": "optimize_fragment",
                        "fragment": fragment,
                        "action": "improve_success_rate",
                        "priority": "high"
                    })
                elif "Alto tempo de execução" in bottleneck:
                    fragment = bottleneck.split()[-1]
                    steps.append({
                        "type": "optimize_fragment",
                        "fragment": fragment,
                        "action": "reduce_execution_time",
                        "priority": "medium"
                    })
                elif "Alta taxa de erro" in bottleneck:
                    fragment = bottleneck.split()[-1]
                    steps.append({
                        "type": "optimize_fragment",
                        "fragment": fragment,
                        "action": "reduce_error_rate",
                        "priority": "high"
                    })
                    
            # Gera passos para melhorar saúde do sistema
            if performance_metrics.get("global_metrics", {}).get("system_health", 0.0) < 0.7:
                steps.append({
                    "type": "system_optimization",
                    "action": "improve_overall_health",
                    "priority": "high"
                })
                
            return steps
            
        except Exception as e:
            logger.exception("Erro durante geração de passos de evolução:")
            return [] 