"""Task runner for orchestrating evaluation tasks.

This module provides the main runner class that executes multiple
evaluation tasks and aggregates their results.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_encoder import BaseEncoder
from .base_loader import BaseDataLoader
from .loaders import JSONDataLoader
from .schemas import TestDataModel
from .config import SemEvalSettings, load_settings
from ..tasks import (
    InformationRetrieval,
    SemanticSimilarity,
    LinguisticRobustness,
    VectorArithmetic,
    TaskResult
)


class EvaluationResult:
    """Container for all evaluation results.

    Parameters
    ----------
    metadata : dict
        Test metadata
    task_results : list of TaskResult
        Results from individual tasks
    total_runtime : float
        Total runtime in seconds

    Attributes
    ----------
    metadata : dict
        Test metadata
    task_results : list of TaskResult
        Results from individual tasks
    total_runtime : float
        Total runtime in seconds
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        task_results: List[TaskResult],
        total_runtime: float
    ):
        """Initialize evaluation result."""
        self.metadata = metadata
        self.task_results = task_results
        self.total_runtime = total_runtime

    def get_task_result(self, task_name: str) -> Optional[TaskResult]:
        """Get result for a specific task.

        Parameters
        ----------
        task_name : str
            Name of the task

        Returns
        -------
        TaskResult or None
            Task result if found
        """
        for result in self.task_results:
            if result.task_name == task_name:
                return result
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results.

        Returns
        -------
        dict
            Summary dictionary with metadata, task statuses, and key metrics
        """
        summary = {
            "metadata": self.metadata,
            "total_runtime": self.total_runtime,
            "tasks": {}
        }

        for result in self.task_results:
            summary["tasks"][result.task_name] = {
                "status": result.status,
                "runtime": result.runtime_seconds,
                "metrics": result.metrics,
                "error": result.error_message if result.status == "failed" else None
            }

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return f"EvaluationResult(tasks={len(self.task_results)}, runtime={self.total_runtime:.2f}s)"


class TaskRunner:
    """Main runner for executing evaluation tasks.

    This class orchestrates the execution of multiple evaluation tasks
    on a given encoder using specified test data.

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to evaluate
    data_loader : BaseDataLoader, optional
        Data loader to use (default: JSONDataLoader)
    device : str, optional
        Device for computation
    verbose : bool, optional
        If True, print progress information

    Attributes
    ----------
    encoder : BaseEncoder
        Text encoder being evaluated
    data_loader : BaseDataLoader
        Data loader instance
    device : str or None
        Computation device
    verbose : bool
        Verbosity flag

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.core.runner import TaskRunner
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run all tasks
    >>> runner = TaskRunner(encoder=encoder, verbose=True)
    >>> result = runner.run("data/test_data.json")
    >>>
    >>> # Get summary
    >>> summary = result.get_summary()
    >>> print(f"Total runtime: {summary['total_runtime']:.2f}s")
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        data_loader: Optional[BaseDataLoader] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        settings: Optional[SemEvalSettings] = None
    ):
        """Initialize task runner.

        Parameters
        ----------
        encoder : BaseEncoder
            Text encoder
        data_loader : BaseDataLoader, optional
            Data loader (default: JSONDataLoader)
        device : str, optional
            Device override (overrides settings if provided)
        verbose : bool, optional
            Verbose override (overrides settings if provided)
        settings : SemEvalSettings, optional
            Configuration settings. If not provided, loads from default config.
        """
        self.encoder = encoder
        self.data_loader = data_loader or JSONDataLoader()

        # Load settings if not provided (backward compatibility)
        self.settings = settings or load_settings()

        # Allow parameter overrides (for backward compatibility)
        self.device = device if device is not None else self.settings.model.device
        self.verbose = verbose if verbose else self.settings.logging.verbose

    def _log(self, message: str, level: str = "INFO"):
        """Log a message if verbose.

        Parameters
        ----------
        message : str
            Message to log
        level : str, optional
            Log level (INFO, WARNING, ERROR)
        """
        if self.verbose:
            print(f"[{level}] {message}")

    def run(self, data_source: str) -> EvaluationResult:
        """Run all available tasks on the provided data.

        Parameters
        ----------
        data_source : str
            Path to test data file

        Returns
        -------
        EvaluationResult
            Aggregated results from all tasks

        Raises
        ------
        FileNotFoundError
            If data file not found
        ValueError
            If data validation fails

        Examples
        --------
        >>> runner = TaskRunner(encoder=encoder, verbose=True)
        >>> result = runner.run("data/test_data.json")
        >>> print(result.get_summary())
        """
        self._log("="*70)
        self._log("Starting Evaluation")
        self._log("="*70)

        start_time = time.time()

        # Load test data
        self._log(f"Loading test data from: {data_source}")
        test_data = self.data_loader.load(data_source)

        self._log(f"Model: {self.encoder.model_name}")
        self._log(f"Embedding dimension: {self.encoder.get_embedding_dim()}")

        # Run available tasks
        task_results = []

        # Information Retrieval
        if test_data.tasks.information_retrieval is not None:
            self._log("\n" + "="*70)
            self._log("Running Information Retrieval Task")
            self._log("="*70)

            task = InformationRetrieval(
                encoder=self.encoder,
                task_data=test_data.tasks.information_retrieval,
                device=self.device,
                verbose=self.verbose
            )
            result = task.run()
            task_results.append(result)

        # Semantic Similarity
        if test_data.tasks.semantic_similarity is not None:
            self._log("\n" + "="*70)
            self._log("Running Semantic Similarity Task")
            self._log("="*70)

            task = SemanticSimilarity(
                encoder=self.encoder,
                task_data=test_data.tasks.semantic_similarity,
                device=self.device,
                verbose=self.verbose
            )
            result = task.run()
            task_results.append(result)

        # Linguistic Robustness
        if test_data.tasks.linguistic_robustness is not None:
            self._log("\n" + "="*70)
            self._log("Running Linguistic Robustness Task")
            self._log("="*70)

            task = LinguisticRobustness(
                encoder=self.encoder,
                task_data=test_data.tasks.linguistic_robustness,
                device=self.device,
                verbose=self.verbose
            )
            result = task.run()
            task_results.append(result)

        # Vector Arithmetic
        if test_data.tasks.vector_arithmetic is not None:
            self._log("\n" + "="*70)
            self._log("Running Vector Arithmetic Task")
            self._log("="*70)

            task = VectorArithmetic(
                encoder=self.encoder,
                task_data=test_data.tasks.vector_arithmetic,
                device=self.device,
                verbose=self.verbose
            )
            result = task.run()
            task_results.append(result)

        total_runtime = time.time() - start_time

        self._log("\n" + "="*70)
        self._log(f"Evaluation Completed in {total_runtime:.2f}s")
        self._log("="*70)

        # Create evaluation result
        evaluation_result = EvaluationResult(
            metadata=test_data.metadata.model_dump(),
            task_results=task_results,
            total_runtime=total_runtime
        )

        return evaluation_result

    def run_task(self, task_name: str, data_source: str) -> TaskResult:
        """Run a specific task only.

        Parameters
        ----------
        task_name : str
            Name of the task to run
        data_source : str
            Path to test data file

        Returns
        -------
        TaskResult
            Task execution result

        Raises
        ------
        ValueError
            If task not available in data or unknown task name

        Examples
        --------
        >>> runner = TaskRunner(encoder=encoder)
        >>> result = runner.run_task("information_retrieval", "data/test_data.json")
        >>> print(f"NDCG@10: {result.get_metric('cosine-NDCG@10'):.4f}")
        """
        # Load test data
        test_data = self.data_loader.load(data_source)

        # Run specific task
        if task_name == "information_retrieval":
            if test_data.tasks.information_retrieval is None:
                raise ValueError("Information Retrieval task not available in test data")

            task = InformationRetrieval(
                encoder=self.encoder,
                task_data=test_data.tasks.information_retrieval,
                device=self.device,
                verbose=self.verbose
            )
            return task.run()

        elif task_name == "semantic_similarity":
            if test_data.tasks.semantic_similarity is None:
                raise ValueError("Semantic Similarity task not available in test data")

            task = SemanticSimilarity(
                encoder=self.encoder,
                task_data=test_data.tasks.semantic_similarity,
                device=self.device,
                verbose=self.verbose
            )
            return task.run()

        elif task_name == "linguistic_robustness":
            if test_data.tasks.linguistic_robustness is None:
                raise ValueError("Linguistic Robustness task not available in test data")

            task = LinguisticRobustness(
                encoder=self.encoder,
                task_data=test_data.tasks.linguistic_robustness,
                device=self.device,
                verbose=self.verbose
            )
            return task.run()

        elif task_name == "vector_arithmetic":
            if test_data.tasks.vector_arithmetic is None:
                raise ValueError("Vector Arithmetic task not available in test data")

            task = VectorArithmetic(
                encoder=self.encoder,
                task_data=test_data.tasks.vector_arithmetic,
                device=self.device,
                verbose=self.verbose
            )
            return task.run()

        else:
            raise ValueError(f"Unknown task name: {task_name}")
