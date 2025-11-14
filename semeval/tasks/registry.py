"""Task registry for mapping task names to task classes.

This module provides a registry that maps task names to their corresponding
task class implementations, enabling flexible export and reporting.
"""

from typing import Dict, Optional, Type

from .base import BaseTask
from .information_retrieval import InformationRetrieval
from .linguistic_robustness import LinguisticRobustness
from .semantic_similarity import SemanticSimilarity
from .vector_arithmetic import VectorArithmetic

# Task registry mapping task names to task classes
TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "information_retrieval": InformationRetrieval,
    "semantic_similarity": SemanticSimilarity,
    "linguistic_robustness": LinguisticRobustness,
    "vector_arithmetic": VectorArithmetic,
}


def get_task_class(task_name: str) -> Optional[Type[BaseTask]]:
    """Get task class by name.

    Parameters
    ----------
    task_name : str
        Name of the task

    Returns
    -------
    Type[BaseTask] or None
        Task class or None if not found

    Examples
    --------
    >>> from semeval.tasks.registry import get_task_class
    >>> task_cls = get_task_class('information_retrieval')
    >>> print(task_cls.__name__)
    InformationRetrieval
    """
    return TASK_REGISTRY.get(task_name)


def register_task(task_name: str, task_class: Type[BaseTask]) -> None:
    """Register a new task class.

    Parameters
    ----------
    task_name : str
        Name of the task
    task_class : Type[BaseTask]
        Task class to register

    Examples
    --------
    >>> from semeval.tasks.registry import register_task
    >>> from semeval.tasks.base import BaseTask
    >>>
    >>> class CustomTask(BaseTask):
    ...     def get_task_name(self):
    ...         return 'custom_task'
    ...     def run(self):
    ...         pass
    >>>
    >>> register_task('custom_task', CustomTask)
    """
    TASK_REGISTRY[task_name] = task_class
