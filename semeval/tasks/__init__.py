"""Evaluation tasks.

This package contains implementations of evaluation tasks.
"""

from .base import BaseTask, TaskResult
from .information_retrieval import InformationRetrieval
from .linguistic_robustness import LinguisticRobustness
from .registry import TASK_REGISTRY, get_task_class, register_task
from .semantic_similarity import SemanticSimilarity
from .vector_arithmetic import VectorArithmetic

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
