# a3x/fragments/symbolic_guidance_fragment.py
import logging
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
from a3x.fragments.base import BaseFragment, FragmentContext
from a3x.fragments.registry import fragment
from a3x.core.memory.memory_manager import MemoryManager

# Scikit-learn imports (needed for loading the pipeline)
try:
    # Need these to potentially unpickle the model object correctly, even if not used directly
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[SymbolicGuidanceFragment] Warning: scikit-learn not found. Cannot load or use the model.")
    SKLEARN_AVAILABLE = False
    # Placeholders needed if sklearn isn't installed, otherwise joblib.load might fail
    class TfidfVectorizer: pass
    class LogisticRegression: pass
    class Pipeline: pass
    class NotFittedError(Exception): pass

# Core A3X Imports
try:
    from a3x.fragment_registry import register_fragment, fragment_decorator
except ImportError as e:
    print(f"[SymbolicGuidanceFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    
    class FragmentContext:
        workspace_root: Optional[str] = None
        
    class BaseFragment:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

@fragment(
    name="symbolic_guidance",
    description="Fornece orientação simbólica para o sistema.",
    category="evolution",
    skills=["analyze_system", "generate_guidance", "validate_changes"]
)
class SymbolicGuidanceFragment(BaseFragment):
    """
    Provides symbolic guidance for system behavior and decision making.
    """

    # Path relative to workspace root
    MODEL_FILE = "a3x/a3net/models/fragment_success_predictor.pkl"
    CONFIDENCE_THRESHOLD = 0.75 # Minimum probability for suggesting 'usar fragmento'

    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        self.ctx = ctx
        self.memory_manager = MemoryManager()
        self.model = None
        self._model_load_attempted = False
        self._model_load_error = None

    async def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # Load prediction model
            await self._load_model()
            
            # Analyze system state
            analysis = await self._analyze_system()
            
            # Generate guidance
            guidance = await self._generate_guidance(analysis)
            
            # Validate changes
            validated = await self._validate_changes(guidance)
            
            return {
                "status": "success",
                "analysis": analysis,
                "guidance": guidance,
                "validated": validated
            }
        except Exception as e:
            logger.error(f"Error in SymbolicGuidanceFragment.execute: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _load_model(self) -> None:
        """Loads the prediction model using joblib. Returns True on success."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn not available. Cannot load the model.")
            self._model_load_error = "Scikit-learn not installed."
            return

        if not self.ctx.workspace_root:
            logger.error("Workspace root not set. Cannot load the model.")
            self._model_load_error = "Workspace root not set."
            return

        model_path = Path(self.ctx.workspace_root) / self.MODEL_FILE

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            self._model_load_error = f"Model file not found at {model_path}"
            return

        try:
            logger.info(f"Loading success predictor model from: {model_path}")
            start_time = time.time() # Requires import time
            self.model = joblib.load(model_path)
            end_time = time.time()
            logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")
            # Basic check if it looks like a pipeline with predict_proba
            if not hasattr(self.model, 'predict_proba') or not callable(getattr(self.model, 'predict_proba')):
                logger.error("Loaded object is not a valid scikit-learn pipeline/model with predict_proba.")
                self.model = None
                self._model_load_error = "Invalid model object loaded."
                return
            self._model_load_attempted = True
            return
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
            self.model = None
            self._model_load_error = f"Error loading model: {e}"
            return

    async def _analyze_system(self) -> Dict[str, Any]:
        # TODO: Implement system analysis logic
        return {}

    async def _generate_guidance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement guidance generation logic
        return {}

    async def _validate_changes(self, guidance: Dict[str, Any]) -> bool:
        # TODO: Implement change validation logic
        return True

    async def _evaluate_fragments(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Avalia o desempenho dos fragments ativos."""
        try:
            # Obtém fragments ativos
            active_fragments = ctx.fragment_registry.get_active_fragments()
            
            # Coleta métricas de cada fragment
            fragment_metrics = {}
            for fragment in active_fragments:
                metrics = await self._collect_fragment_metrics(ctx, fragment)
                fragment_metrics[fragment.name] = metrics
            
            # Analisa tendências
            trends = self._analyze_trends(fragment_metrics)
            
            return {
                "status": "success",
                "fragment_metrics": fragment_metrics,
                "trends": trends
            }
            
        except Exception as e:
            logger.exception("Erro durante avaliação dos fragments:")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _collect_fragment_metrics(self, ctx: FragmentContext, fragment: BaseFragment) -> Dict[str, Any]:
        """Coleta métricas de um fragment específico."""
        try:
            # Obtém histórico de execução
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=100
            )
            
            # Calcula métricas
            success_rate = self._calculate_success_rate(history)
            avg_execution_time = self._calculate_avg_execution_time(history)
            error_rate = self._calculate_error_rate(history)
            
            return {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "error_rate": error_rate,
                "total_executions": len(history)
            }
            
        except Exception:
            return {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "error_rate": 1.0,
                "total_executions": 0
            }

    def _calculate_success_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calcula taxa de sucesso das execuções."""
        if not history:
            return 0.0
            
        successful = sum(1 for h in history if h.get("status") == "success")
        return successful / len(history)

    def _calculate_avg_execution_time(self, history: List[Dict[str, Any]]) -> float:
        """Calcula tempo médio de execução."""
        if not history:
            return 0.0
            
        times = [h.get("execution_time", 0) for h in history]
        return sum(times) / len(times)

    def _calculate_error_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calcula taxa de erros."""
        if not history:
            return 1.0
            
        errors = sum(1 for h in history if h.get("status") == "error")
        return errors / len(history)

    def _analyze_trends(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa tendências nos dados coletados."""
        trends = {
            "improving": [],
            "degrading": [],
            "stable": []
        }
        
        for fragment_name, fragment_metrics in metrics.items():
            # Analisa tendência de sucesso
            if fragment_metrics["success_rate"] > 0.8:
                trends["improving"].append(fragment_name)
            elif fragment_metrics["success_rate"] < 0.5:
                trends["degrading"].append(fragment_name)
        else:
                trends["stable"].append(fragment_name)
        
        return trends

    async def _analyze_performance(self, ctx: FragmentContext, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa o desempenho geral do sistema."""
        try:
            # Extrai métricas
            fragment_metrics = evaluation.get("fragment_metrics", {})
            trends = evaluation.get("trends", {})
            
            # Calcula métricas globais
            global_metrics = self._calculate_global_metrics(fragment_metrics)
            
            # Identifica gargalos
            bottlenecks = self._identify_bottlenecks(fragment_metrics)
            
            # Analisa dependências
            dependencies = self._analyze_dependencies(ctx, fragment_metrics)
            
            return {
                "status": "success",
                "global_metrics": global_metrics,
                "bottlenecks": bottlenecks,
                "dependencies": dependencies,
                "trends": trends
            }
            
        except Exception as e:
            logger.exception("Erro durante análise de performance:")
            return {
                "status": "error",
                "message": str(e)
            }

    def _calculate_global_metrics(self, fragment_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calcula métricas globais do sistema."""
        if not fragment_metrics:
            return {
                "avg_success_rate": 0.0,
                "avg_execution_time": 0.0,
                "system_health": 0.0
            }
            
        # Calcula médias
        success_rates = [m["success_rate"] for m in fragment_metrics.values()]
        execution_times = [m["avg_execution_time"] for m in fragment_metrics.values()]
        
        avg_success = sum(success_rates) / len(success_rates)
        avg_time = sum(execution_times) / len(execution_times)
        
        # Calcula saúde do sistema
        health = avg_success * (1 - min(1, avg_time / 1000))  # Normaliza tempo
        
        return {
            "avg_success_rate": avg_success,
            "avg_execution_time": avg_time,
            "system_health": health
        }

    def _identify_bottlenecks(self, fragment_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identifica gargalos no sistema."""
        bottlenecks = []
        
        for fragment_name, metrics in fragment_metrics.items():
            # Verifica taxa de erro alta
            if metrics["error_rate"] > 0.3:
                bottlenecks.append(f"{fragment_name}: Alta taxa de erros")
            
            # Verifica tempo de execução alto
            if metrics["avg_execution_time"] > 5000:  # 5 segundos
                bottlenecks.append(f"{fragment_name}: Tempo de execução alto")
            
            # Verifica taxa de sucesso baixa
            if metrics["success_rate"] < 0.5:
                bottlenecks.append(f"{fragment_name}: Baixa taxa de sucesso")
        
        return bottlenecks

    def _analyze_dependencies(self, ctx: FragmentContext, fragment_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analisa dependências entre fragments."""
        dependencies = {}
        
        for fragment_name in fragment_metrics.keys():
            fragment = ctx.fragment_registry.get_fragment(fragment_name)
            if fragment:
                # Obtém skills do fragment
                skills = fragment.skills
                
                # Verifica dependências
                deps = []
                for skill in skills:
                    # Verifica se outros fragments usam a mesma skill
                    for other_name, other_metrics in fragment_metrics.items():
                        if other_name != fragment_name:
                            other_fragment = ctx.fragment_registry.get_fragment(other_name)
                            if other_fragment and skill in other_fragment.skills:
                                deps.append(f"{other_name}: {skill}")
                
                dependencies[fragment_name] = deps
        
        return dependencies

    async def _generate_recommendations(self, ctx: FragmentContext, analysis: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise."""
        try:
            recommendations = []
            
            # Analisa métricas globais
            global_metrics = analysis.get("global_metrics", {})
            if global_metrics["system_health"] < 0.7:
                recommendations.append("Sistema com saúde abaixo do ideal - considerar otimizações gerais")
            
            # Analisa gargalos
            bottlenecks = analysis.get("bottlenecks", [])
            for bottleneck in bottlenecks:
                recommendations.append(f"Gargalo identificado: {bottleneck}")
            
            # Analisa dependências
            dependencies = analysis.get("dependencies", {})
            for fragment, deps in dependencies.items():
                if len(deps) > 3:  # Muitas dependências
                    recommendations.append(f"Fragment {fragment} tem muitas dependências - considerar refatoração")
            
            # Analisa tendências
            trends = analysis.get("trends", {})
            for fragment in trends.get("degrading", []):
                recommendations.append(f"Fragment {fragment} mostrando degradação - investigar causa")
            
            return recommendations
            
        except Exception as e:
            logger.exception("Erro durante geração de recomendações:")
            return ["Erro ao gerar recomendações"]

# --- Example Usage/Testing (Optional) ---
# async def main():
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     import time # Ensure time is imported

#     # Create dummy workspace and model file (Requires a trained model)
#     test_workspace = Path("./temp_guidance_workspace")
#     test_workspace.mkdir(exist_ok=True)
#     model_dir = test_workspace / "a3x/a3net/models"
#     model_dir.mkdir(parents=True, exist_ok=True)
#     model_path = model_dir / "fragment_success_predictor.pkl"

#     # --- Create a Dummy Trained Model for Testing ---
#     # NOTE: This requires the NeuralTrainerFragment code or similar logic
#     # to be available to create a valid model file.
#     print("\n--- Creating dummy model for testing ---")
#     try:
#         if not SKLEARN_AVAILABLE:
#             raise ImportError("Skipping dummy model creation: scikit-learn not installed.")

#         # Dummy data
#         dummy_texts = [
#             "TASK: Write code FRAGMENT: CodeWriter",
#             "TASK: Analyze data FRAGMENT: DataAnalyzer",
#             "TASK: Write code FRAGMENT: OldCodeWriter",
#             "TASK: Debug issue FRAGMENT: Debugger",
#             "TASK: Analyze data FRAGMENT: SlowAnalyzer"
#         ]
#         dummy_labels = [1, 1, 0, 1, 0] # 1=success, 0=failure

#         # Create and fit pipeline
#         dummy_pipeline = Pipeline([
#             ('tfidf', TfidfVectorizer()),
#             ('clf', LogisticRegression(solver='liblinear'))
#         ])
#         dummy_pipeline.fit(dummy_texts, dummy_labels)
#         print(f"Dummy pipeline fitted: {dummy_pipeline}")

#         # Save dummy model
#         joblib.dump(dummy_pipeline, model_path)
#         print(f"Dummy model saved to: {model_path}")
#         model_created = True
#     except Exception as e:
#         print(f"\n!!! Failed to create dummy model: {e}. Guidance test may fail. !!!")
#         model_created = False
#         if model_path.exists(): model_path.unlink() # Clean up partial file
#     print("--- End dummy model creation ---\n")

#     if not model_created:
#         print("Cannot run test without a model file.")
#         return

#     # Mock Context
#     class MockContext(FragmentContext):
#          def __init__(self, workspace):
#               self.workspace_root = workspace

#     # Instantiate Fragment
#     fragment = SymbolicGuidanceFragment()
#     context = MockContext(str(test_workspace.resolve()))

#     # --- Test Case 1: One fragment clearly better ---
#     print("--- Test Case 1: Clear Winner ---")
#     options1 = [
#         {"task": "Write code", "fragment": "CodeWriter"}, # Should have high prob
#         {"task": "Write code", "fragment": "OldCodeWriter"} # Should have low prob
#     ]
#     result1 = await fragment.execute(context, fragment_options=options1)
#     print(f"Execution result 1: {result1}")

#     # --- Test Case 2: No fragment meets threshold ---
#     print("\n--- Test Case 2: Low Confidence ---")
#     options2 = [
#         {"task": "Refactor complex system", "fragment": "CodeWriter"}, # Assume low prob
#         {"task": "Refactor complex system", "fragment": "LegacyRefactorTool"} # Assume low prob
#     ]
#     # Reset internal model state if needed (or create new fragment instance)
#     # fragment = SymbolicGuidanceFragment() # Re-instantiate to force reload if desired
#     result2 = await fragment.execute(context, fragment_options=options2)
#     print(f"Execution result 2: {result2}")

#     # --- Test Case 3: Model file missing ---
#     print("\n--- Test Case 3: Model Missing ---")
#     if model_path.exists(): model_path.unlink()
#     fragment_new = SymbolicGuidanceFragment() # New instance to trigger load attempt
#     result3 = await fragment_new.execute(context, fragment_options=options1)
#     print(f"Execution result 3: {result3}")


#     # Clean up dummy files/dirs
#     # if model_path.exists(): model_path.unlink()
#     # model_dir.rmdir()
#     # Path(test_workspace / "a3x/a3net").rmdir()
#     # Path(test_workspace / "a3x").rmdir()
#     # test_workspace.rmdir()

# if __name__ == "__main__":
#     import asyncio
#     import time # Ensure time is imported
#     asyncio.run(main()) 