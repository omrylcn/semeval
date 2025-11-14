"""Core modules for semeval package.

This package contains the core functionality including schemas,
encoders, and data loaders.

Modules
-------
schemas
    Pydantic data models for validation
base_encoder
    Abstract base class for encoders
base_loader
    Abstract base class for data loaders
encoders
    Encoder implementations (SentenceTransformer, HuggingFace)
loaders
    Data loader implementations (JSON)
runner
    Task runner for orchestrating evaluations

Examples
--------
>>> from semeval.core.schemas import TestDataModel
>>> from semeval.core.encoders import SentenceTransformerEncoder
>>> from semeval.core.loaders import JSONDataLoader
>>> from semeval.core.runner import TaskRunner
>>>
>>> # Load data and run evaluation
>>> encoder = SentenceTransformerEncoder("model-name")
>>> runner = TaskRunner(encoder=encoder, verbose=True)
>>> result = runner.run("data/test_data.json")
>>> print(result.get_summary())
"""

from .base_encoder import BaseEncoder
from .base_loader import BaseDataLoader, DataValidationError
from .config import (
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    SemEvalSettings,
    TasksConfig,
    load_settings,
)
from .runner import EvaluationResult, TaskRunner
from .schemas import (
    InformationRetrievalData,
    LinguisticRobustnessData,
    SemanticSimilarityData,
    TestDataModel,
    VectorArithmeticData,
)

__all__ = [
    # Schemas
    'TestDataModel',
    'InformationRetrievalData',
    'SemanticSimilarityData',
    'LinguisticRobustnessData',
    'VectorArithmeticData',
    # Base classes
    'BaseEncoder',
    'BaseDataLoader',
    # Runner
    'TaskRunner',
    'EvaluationResult',
    # Configuration
    'SemEvalSettings',
    'load_settings',
    'ModelConfig',
    'OutputConfig',
    'TasksConfig',
    'LoggingConfig',
    # Exceptions
    'DataValidationError',
]
