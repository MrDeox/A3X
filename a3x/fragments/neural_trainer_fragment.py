import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import joblib # For saving sklearn models (more robust than pickle for numpy arrays)
from a3x.fragments.base import BaseFragment, FragmentContext
from a3x.fragments.registry import fragment
from a3x.core.memory.memory_manager import MemoryManager
from a3x.a3net.core.train_loop import TrainingLoop
from a3x.fragments.dataset_builder import DatasetBuilder

# Scikit-learn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[NeuralTrainerFragment] Warning: scikit-learn not found. Please install it: pip install scikit-learn joblib")
    # Define placeholders if sklearn is not available
    SKLEARN_AVAILABLE = False
    class TfidfVectorizer: pass
    class LogisticRegression: pass
    class Pipeline: pass
    class NotFittedError(Exception): pass

# Core A3X Imports (adjust based on actual project structure)
try:
    from a3x.fragments.base import BaseFragment, FragmentContext # Base class and context
    # Fragment registration mechanism (replace with actual import if known)
    # from a3x.fragment_registry import register_fragment, fragment_decorator
except ImportError as e:
    print(f"[NeuralTrainerFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        workspace_root: Optional[str] = None
    class BaseFragment:
        def __init__(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

# --- Fragment Registration (Placeholder) ---
# Replace with the actual registration mechanism
# @fragment_decorator(name="neural_trainer", trigger_phrases=["treinar modelo de sucesso simbolico"])
# --- End Placeholder ---

@fragment(
    name="neural_trainer",
    description="Manages neural network training process",
    category="learning",
    skills=["train_model", "evaluate_model", "save_model"]
)
class NeuralTrainerFragment(BaseFragment):
    """Fragment responsável pelo treinamento e evolução do componente neural (A3Net)."""

    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        self.memory_manager = MemoryManager()
        self.training_loop = TrainingLoop(self.memory_manager)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # Prepare dataset
            dataset = await self._prepare_dataset()
            
            # Train model
            training_result = await self._train_model(dataset)
            
            # Evaluate model
            evaluation_result = await self._evaluate_model(training_result["model"])
            
            # Save model
            await self._save_model(training_result["model"])
            
            return {
                "training_result": training_result,
                "evaluation_result": evaluation_result
            }
            
        except Exception as e:
            self.ctx.logger.error(f"Error in NeuralTrainerFragment: {str(e)}")
            raise

    async def _prepare_dataset(self) -> Dict[str, Any]:
        """Prepara o dataset para treinamento."""
        try:
            # Inicializa o builder
            builder = DatasetBuilder(self.memory_manager)
            
            # Gera o dataset
            dataset_info = await builder.build_dataset()
            
            # Valida o dataset
            if not self._validate_dataset(dataset_info):
                raise ValueError("Dataset inválido gerado")
            
            return dataset_info
            
        except Exception as e:
            logger.exception("Erro durante preparação do dataset:")
            raise

    def _validate_dataset(self, dataset_info: Dict[str, Any]) -> bool:
        """Valida se o dataset gerado é adequado para treinamento."""
        try:
            # Verifica se o arquivo existe
            dataset_path = Path(dataset_info["path"])
            if not dataset_path.exists():
                return False
            
            # Verifica se tem dados suficientes
            if dataset_info["size"] < self.MIN_DATASET_SIZE:
                return False
            
            # Verifica a distribuição dos dados
            if not self._check_data_distribution(dataset_info):
                return False
            
            return True
            
        except Exception:
            return False

    def _check_data_distribution(self, dataset_info: Dict[str, Any]) -> bool:
        """Verifica se a distribuição dos dados é adequada."""
        try:
            # Carrega uma amostra do dataset
            sample_size = min(100, dataset_info["size"])
            sample = self._load_dataset_sample(dataset_info["path"], sample_size)
            
            # Verifica distribuição de classes/tipos
            class_distribution = self._analyze_class_distribution(sample)
            
            # Verifica se há classes minoritárias
            min_samples = sample_size * 0.1  # Mínimo 10% por classe
            if any(count < min_samples for count in class_distribution.values()):
                return False
            
            return True
            
        except Exception:
            return False

    def _load_dataset_sample(self, path: str, size: int) -> List[Dict[str, Any]]:
        """Carrega uma amostra do dataset."""
        sample = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= size:
                    break
                sample.append(json.loads(line))
        return sample

    def _analyze_class_distribution(self, sample: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analisa a distribuição de classes no dataset."""
        distribution = {}
        for item in sample:
            class_name = item.get("class", "unknown")
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    async def _train_model(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Executa o ciclo de treinamento neural."""
        try:
            # 1. Coletar dados do contexto
            dataset_path = await self._prepare_dataset()
            
            # 2. Executar treinamento
            training_result = await self.training_loop.train(dataset_path)
            
            # 3. Avaliar resultados
            evaluation = await self._evaluate_training(training_result)
            
            return {
                "status": "success",
                "training_result": training_result,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.exception("Erro durante treinamento neural:")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _evaluate_model(self, model: Any) -> Dict[str, Any]:
        """Avalia os resultados do treinamento."""
        try:
            # Extrai métricas do treinamento
            metrics = model.get("metrics", {})
            
            # Avalia performance
            performance = self._calculate_performance(metrics)
            
            # Verifica se houve overfitting
            overfitting = self._check_overfitting(metrics)
            
            # Gera recomendações
            recommendations = self._generate_recommendations(performance, overfitting)
            
            return {
                "status": "success",
                "performance": performance,
                "overfitting": overfitting,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.exception("Erro durante avaliação do treinamento:")
            return {
                "status": "error",
                "message": str(e)
            }

    def _calculate_performance(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calcula métricas de performance do modelo."""
        return {
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0)
        }

    def _check_overfitting(self, metrics: Dict[str, float]) -> bool:
        """Verifica se há indícios de overfitting."""
        train_loss = metrics.get("train_loss", 0.0)
        val_loss = metrics.get("val_loss", 0.0)
        
        # Considera overfitting se a diferença for maior que 20%
        return abs(train_loss - val_loss) > 0.2

    def _generate_recommendations(self, performance: Dict[str, float], overfitting: bool) -> List[str]:
        """Gera recomendações baseadas nos resultados."""
        recommendations = []
        
        # Verifica performance geral
        if all(v < 0.7 for v in performance.values()):
            recommendations.append("Considerar aumento do tamanho do dataset")
            recommendations.append("Avaliar arquitetura do modelo")
        
        # Verifica overfitting
        if overfitting:
            recommendations.append("Aplicar técnicas de regularização")
            recommendations.append("Considerar early stopping")
        
        return recommendations

    async def _save_model(self, model: Any) -> None:
        # TODO: Implement model saving logic
        pass

# --- Example Usage/Testing (Optional) ---
# async def main():
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     import time # Add import for timing

#     # Create dummy workspace and data file
#     test_workspace = Path("./temp_trainer_workspace")
#     test_workspace.mkdir(exist_ok=True)
#     data_dir = test_workspace / "a3x/a3net/data"
#     data_dir.mkdir(parents=True, exist_ok=True)
#     input_file = data_dir / "symbolic_experience.jsonl"

#     dummy_data = [
#         {"task": "Write code", "fragment_used": "CodeWriter", "status": "success", "skill_used": "write_file", "observation": "File written"},
#         {"task": "Analyze data", "fragment_used": "DataAnalyzer", "status": "success", "skill_used": "run_sql", "observation": {"rows": 5}},
#         {"task": "Write code", "fragment_used": "CodeWriter", "status": "failure", "skill_used": "write_file", "observation": "Permission denied"},
#         {"task": "Debug issue", "fragment_used": "Debugger", "status": "success", "skill_used": "read_log", "observation": "Error found"},
#         {"task": "Refactor module", "fragment_used": "CodeWriter", "status": "failure", "skill_used": "apply_edit", "observation": "Conflict detected"},
#         {"task": "Analyze data", "fragment_used": "DataAnalyzer", "status": "success", "skill_used": "run_sql", "observation": {"rows": 10}},
#         {"task": "Debug issue", "fragment_used": "Debugger", "status": "failure", "skill_used": "read_log", "observation": "Log empty"},
#         {"task": "Write code", "fragment_used": "CodeWriter", "status": "success", "skill_used": "write_file", "observation": "File updated"},
#     ]
#     with open(input_file, "w", encoding="utf-8") as f:
#         for record in dummy_data:
#             f.write(json.dumps(record) + "\n")

#     # Mock Context
#     class MockContext(FragmentContext):
#          def __init__(self, workspace):
#               self.workspace_root = workspace

#     # Instantiate and Execute Fragment
#     fragment = NeuralTrainerFragment() # Assuming no complex init args
#     context = MockContext(str(test_workspace.resolve()))

#     result = await fragment.execute(context)
#     print(f"\nExecution result: {result}")

#     # Check if model file was created
#     model_path = test_workspace / NeuralTrainerFragment.OUTPUT_MODEL_FILE
#     if model_path.exists():
#         print(f"Model file created at: {model_path}")
#         # Optional: Load and test the model
#         try:
#             loaded_pipeline = joblib.load(model_path)
#             print(f"Model loaded successfully: {loaded_pipeline}")
#             # Example prediction
#             test_texts = ["TASK: Write code FRAGMENT: CodeWriter", "TASK: Analyze data FRAGMENT: DataAnalyzer"]
#             predictions = loaded_pipeline.predict(test_texts)
#             probabilities = loaded_pipeline.predict_proba(test_texts)
#             print(f"Test Predictions for {test_texts}: {predictions}") # 0=failure, 1=success
#             print(f"Test Probabilities: {probabilities}")
#         except Exception as load_err:
#             print(f"Error loading/testing saved model: {load_err}")
#         # Clean up dummy files/dirs
#         # model_path.unlink()
#         # model_path.parent.rmdir()
#     else:
#         print(f"Model file {model_path} was NOT created.")

#     # Clean up input file and dir
#     # input_file.unlink()
#     # data_dir.rmdir()
#     # Path(test_workspace / "a3x/a3net").rmdir()
#     # Path(test_workspace / "a3x").rmdir()
#     # test_workspace.rmdir()

# if __name__ == "__main__":
#     import asyncio
#     import time # Make sure time is imported
#     asyncio.run(main()) 