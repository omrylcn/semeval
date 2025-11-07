"""Evaluation tasks.

This package contains implementations of evaluation tasks.
"""

from .base import BaseTask, TaskResult
from .information_retrieval import InformationRetrieval
from .semantic_similarity import SemanticSimilarity
from .linguistic_robustness import LinguisticRobustness
from .vector_arithmetic import VectorArithmetic
from .registry import get_task_class, register_task, TASK_REGISTRY

__all__ = [
    'BaseTask',
    'TaskResult',
    'InformationRetrieval',
    'SemanticSimilarity',
    'LinguisticRobustness',
    'VectorArithmetic',
    'get_task_class',
    'register_task',
    'TASK_REGISTRY',
]
