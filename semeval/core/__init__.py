"""Core modules for semeval package.

This package contains the core functionality including schemas,
encoders, data loaders, exceptions, and logging.

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
exceptions
    Custom exception hierarchy
logging
    Structured logging utilities

Examples
--------
>>> from semeval.core.schemas import TestDataModel
>>> from semeval.core.encoders import SentenceTransformerEncoder
>>> from semeval.core.loaders import JSONDataLoader
>>> from semeval.core.runner import TaskRunner
>>> from semeval.core.logging import setup_logger
>>>
>>> # Load data and run evaluation
>>> logger = setup_logger(verbose=True)
>>> encoder = SentenceTransformerEncoder("model-name")
>>> runner = TaskRunner(encoder=encoder, verbose=True)
>>> result = runner.run("data/test_data.json")
>>> print(result.get_summary())
"""

from .base_encoder import BaseEncoder
from .base_loader import BaseDataLoader
from .config import (
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    SemEvalSettings,
    TasksConfig,
    load_settings,
)
from .exceptions import (
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    DataError,
    DataFormatError,
    DataLoadError,
    DataValidationError,
    EncoderError,
    ExportError,
    ExportFormatError,
    MetricComputationError,
    MetricError,
    MetricInputError,
    ModelDimensionError,
    ModelError,
    ModelLoadError,
    ReportGenerationError,
    SemEvalError,
    TaskConfigError,
    TaskError,
    TaskExecutionError,
)
from .logging import (
    PerformanceLogger,
    StructuredLogger,
    get_logger,
    log_errors,
    log_execution_time,
    setup_logger,
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
    "TestDataModel",
    "InformationRetrievalData",
    "SemanticSimilarityData",
    "LinguisticRobustnessData",
    "VectorArithmeticData",
    # Base classes
    "BaseEncoder",
    "BaseDataLoader",
    # Runner
    "TaskRunner",
    "EvaluationResult",
    # Configuration
    "SemEvalSettings",
    "load_settings",
    "ModelConfig",
    "OutputConfig",
    "TasksConfig",
    "LoggingConfig",
    # Exceptions
    "SemEvalError",
    "DataError",
    "DataValidationError",
    "DataLoadError",
    "DataFormatError",
    "ModelError",
    "ModelLoadError",
    "EncoderError",
    "ModelDimensionError",
    "TaskError",
    "TaskExecutionError",
    "TaskConfigError",
    "MetricError",
    "MetricComputationError",
    "MetricInputError",
    "ConfigError",
    "ConfigValidationError",
    "ConfigLoadError",
    "ExportError",
    "ExportFormatError",
    "ReportGenerationError",
    # Logging
    "setup_logger",
    "get_logger",
    "StructuredLogger",
    "PerformanceLogger",
    "log_execution_time",
    "log_errors",
]
