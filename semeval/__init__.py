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

# Main API exports
from .core.base_encoder import BaseEncoder
from .core.base_loader import BaseDataLoader, DataValidationError
from .core.config import SemEvalSettings, load_settings
from .core.encoders import HuggingFaceEncoder, SentenceTransformerEncoder
from .core.loaders import JSONDataLoader
from .core.runner import EvaluationResult, TaskRunner
from .core.schemas import (
    InformationRetrievalData,
    LinguisticRobustnessData,
    SemanticSimilarityData,
    TestDataModel,
    VectorArithmeticData,
)
from .postprocess import ReportGenerator, ResultsExporter
from .tasks import (
    BaseTask,
    InformationRetrieval,
    LinguisticRobustness,
    SemanticSimilarity,
    TaskResult,
    VectorArithmetic,
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    # Main runner
    'TaskRunner',
    'EvaluationResult',
    # Encoders
    'SentenceTransformerEncoder',
    'HuggingFaceEncoder',
    'BaseEncoder',
    # Data loaders
    'JSONDataLoader',
    'BaseDataLoader',
    # Configuration
    'load_settings',
    'SemEvalSettings',
    # Tasks
    'InformationRetrieval',
    'SemanticSimilarity',
    'LinguisticRobustness',
    'VectorArithmetic',
    'BaseTask',
    'TaskResult',
    # Schemas
    'TestDataModel',
    'InformationRetrievalData',
    'SemanticSimilarityData',
    'LinguisticRobustnessData',
    'VectorArithmeticData',
    # Exceptions
    'DataValidationError',
    # Post-processing
    'ResultsExporter',
    'ReportGenerator',
]
