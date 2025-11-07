"""Evaluation tasks.

This package contains implementations of evaluation tasks.
"""

from .base import BaseTask, TaskResult
from .information_retrieval import InformationRetrieval

__all__ = ['BaseTask', 'TaskResult', 'InformationRetrieval']
