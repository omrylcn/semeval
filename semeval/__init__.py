"""SemEval: Semantic Evaluation Package for Turkish NLP Models.

This package provides a comprehensive framework for evaluating semantic
search and NLP models.

Main Components
---------------
- **Encoders**: Text encoding backends (SentenceTransformer, HuggingFace)
- **Data Loaders**: Load and validate test data (JSON)
- **Tasks**: Evaluation tasks (Information Retrieval, Semantic Similarity, etc.)
- **Runner**: Orchestrate multi-task evaluations
- **Schemas**: Pydantic models for data validation

Quick Start
-----------
Basic usage:

    >>> from semeval import TaskRunner
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run evaluation
    >>> runner = TaskRunner(encoder=encoder, verbose=True)
    >>> result = runner.run("data/test_data.json")
    >>>
    >>> # Get results
    >>> summary = result.get_summary()
    >>> print(f"NDCG@10: {summary['tasks']['information_retrieval']['metrics']['cosine-NDCG@10']:.4f}")

Advanced usage with custom encoder:

    >>> from semeval import TaskRunner
    >>> from semeval.core.encoders import HuggingFaceEncoder
    >>>
    >>> # Use custom HuggingFace model
    >>> encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
    >>> runner = TaskRunner(encoder=encoder)
    >>> result = runner.run("data/test_data.json")

Run specific task only:

    >>> result = runner.run_task("information_retrieval", "data/test_data.json")
    >>> print(f"Status: {result.status}")
    >>> print(f"NDCG@10: {result.get_metric('cosine-NDCG@10'):.4f}")

Modules
-------
core
    Core functionality (encoders, loaders, runner, schemas)
tasks
    Evaluation task implementations
metrics
    Metric computation functions
scripts
    Utility scripts and examples
data
    Test data files

Classes
-------
TaskRunner
    Main runner for executing evaluation tasks
EvaluationResult
    Container for evaluation results
SentenceTransformerEncoder
    Encoder using Sentence Transformers
HuggingFaceEncoder
    Encoder using HuggingFace transformers
JSONDataLoader
    Load test data from JSON files

See Also
--------
semeval.core : Core modules
semeval.tasks : Task implementations
semeval.metrics : Metric functions
"""

__version__ = "0.1.0"
__author__ = "No-One"


# Lazy imports for better CLI performance
def __getattr__(name):
    """Lazy import for better performance.

    This allows CLI commands to load quickly without importing heavy dependencies
    like torch, transformers, and sentence-transformers until they're actually needed.
    """
    # Encoders
    if name == "BaseEncoder":
        from .core.base_encoder import BaseEncoder

        return BaseEncoder
    if name == "SentenceTransformerEncoder":
        from .core.encoders import SentenceTransformerEncoder

        return SentenceTransformerEncoder
    if name == "HuggingFaceEncoder":
        from .core.encoders import HuggingFaceEncoder

        return HuggingFaceEncoder

    # Data loaders
    if name == "BaseDataLoader":
        from .core.base_loader import BaseDataLoader

        return BaseDataLoader
    if name == "DataValidationError":
        from .core.base_loader import DataValidationError

        return DataValidationError
    if name == "JSONDataLoader":
        from .core.loaders import JSONDataLoader

        return JSONDataLoader

    # Configuration
    if name == "SemEvalSettings":
        from .core.config import SemEvalSettings

        return SemEvalSettings
    if name == "load_settings":
        from .core.config import load_settings

        return load_settings

    # Runner
    if name == "TaskRunner":
        from .core.runner import TaskRunner

        return TaskRunner
    if name == "EvaluationResult":
        from .core.runner import EvaluationResult

        return EvaluationResult

    # Schemas
    if name == "TestDataModel":
        from .core.schemas import TestDataModel

        return TestDataModel
    if name == "InformationRetrievalData":
        from .core.schemas import InformationRetrievalData

        return InformationRetrievalData
    if name == "SemanticSimilarityData":
        from .core.schemas import SemanticSimilarityData

        return SemanticSimilarityData
    if name == "LinguisticRobustnessData":
        from .core.schemas import LinguisticRobustnessData

        return LinguisticRobustnessData
    if name == "VectorArithmeticData":
        from .core.schemas import VectorArithmeticData

        return VectorArithmeticData

    # Tasks
    if name == "BaseTask":
        from .tasks import BaseTask

        return BaseTask
    if name == "InformationRetrieval":
        from .tasks import InformationRetrieval

        return InformationRetrieval
    if name == "SemanticSimilarity":
        from .tasks import SemanticSimilarity

        return SemanticSimilarity
    if name == "LinguisticRobustness":
        from .tasks import LinguisticRobustness

        return LinguisticRobustness
    if name == "VectorArithmetic":
        from .tasks import VectorArithmetic

        return VectorArithmetic
    if name == "TaskResult":
        from .tasks import TaskResult

        return TaskResult

    # Post-processing
    if name == "ResultsExporter":
        from .postprocess import ResultsExporter

        return ResultsExporter
    if name == "ReportGenerator":
        from .postprocess import ReportGenerator

        return ReportGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    "__author__",
    # Main runner
    "TaskRunner",
    "EvaluationResult",
    # Encoders
    "SentenceTransformerEncoder",
    "HuggingFaceEncoder",
    "BaseEncoder",
    # Data loaders
    "JSONDataLoader",
    "BaseDataLoader",
    # Configuration
    "load_settings",
    "SemEvalSettings",
    # Tasks
    "InformationRetrieval",
    "SemanticSimilarity",
    "LinguisticRobustness",
    "VectorArithmetic",
    "BaseTask",
    "TaskResult",
    # Schemas
    "TestDataModel",
    "InformationRetrievalData",
    "SemanticSimilarityData",
    "LinguisticRobustnessData",
    "VectorArithmeticData",
    # Exceptions
    "DataValidationError",
    # Post-processing
    "ResultsExporter",
    "ReportGenerator",
]
