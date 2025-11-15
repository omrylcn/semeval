"""Task runner for orchestrating evaluation tasks.

This module provides the main runner class that executes multiple
evaluation tasks and aggregates their results.
"""

import time
from typing import Any, Dict, List, Optional

from ..tasks import (
    InformationRetrieval,
    LinguisticRobustness,
    SemanticSimilarity,
    TaskResult,
    VectorArithmetic,
)
from .base_encoder import BaseEncoder
from .base_loader import BaseDataLoader
from .config import SemEvalSettings, load_settings
from .exceptions import TaskConfigError, TaskError, TaskExecutionError
from .loaders import JSONDataLoader
from .logging import get_logger, log_execution_time

logger = get_logger("semeval")


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
        total_runtime: float,
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
            "tasks": {},
        }

        for result in self.task_results:
            summary["tasks"][result.task_name] = {
                "status": result.status,
                "runtime": result.runtime_seconds,
                "metrics": result.metrics,
                "error": result.error_message if result.status == "failed" else None,
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
        settings: Optional[SemEvalSettings] = None,
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
        logger.info("Initializing TaskRunner")
        self.encoder = encoder
        self.data_loader = data_loader or JSONDataLoader()

        # Load settings if not provided (backward compatibility)
        self.settings = settings or load_settings()

        # Allow parameter overrides (for backward compatibility)
        self.device = device if device is not None else self.settings.model.device
        self.verbose = verbose if verbose else self.settings.logging.verbose

        logger.info(
            f"TaskRunner initialized (model: {encoder.model_name}, "
            f"device: {self.device}, verbose: {self.verbose})"
        )

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

        # Also log to logger
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.debug(message)

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
        DataLoadError
            If data file not found
        DataValidationError
            If data validation fails
        TaskExecutionError
            If task execution fails

        Examples
        --------
        >>> runner = TaskRunner(encoder=encoder, verbose=True)
        >>> result = runner.run("data/test_data.json")
        >>> print(result.get_summary())
        """
        self._log("=" * 70)
        self._log("Starting Evaluation")
        self._log("=" * 70)
        logger.info(f"Starting evaluation with data source: {data_source}")

        start_time = time.time()

        # Load test data
        self._log(f"Loading test data from: {data_source}")
        try:
            with log_execution_time(logger, "data_loading"):
                test_data = self.data_loader.load(data_source)
        except Exception as e:
            logger.error(f"Failed to load test data from {data_source}: {str(e)}")
            raise

        self._log(f"Model: {self.encoder.model_name}")
        self._log(f"Embedding dimension: {self.encoder.get_embedding_dim()}")
        logger.info(
            f"Loaded test data (model: {self.encoder.model_name}, "
            f"dim: {self.encoder.get_embedding_dim()}, "
            f"enabled_tasks: {', '.join(test_data.get_enabled_tasks())})"
        )

        # Run available tasks
        task_results = []

        # Information Retrieval
        if test_data.tasks.information_retrieval is not None:
            self._log("\n" + "=" * 70)
            self._log("Running Information Retrieval Task")
            self._log("=" * 70)
            logger.info("Starting Information Retrieval task")

            try:
                task = InformationRetrieval(
                    encoder=self.encoder,
                    task_data=test_data.tasks.information_retrieval,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, "information_retrieval_task"):
                    result = task.run()
                task_results.append(result)
                logger.info(f"Information Retrieval task completed (status: {result.status})")
            except Exception as e:
                logger.error(f"Information Retrieval task failed: {str(e)}")
                self._log(f"Task failed: {str(e)}", level="ERROR")
                # Continue with other tasks instead of failing completely

        # Semantic Similarity
        if test_data.tasks.semantic_similarity is not None:
            self._log("\n" + "=" * 70)
            self._log("Running Semantic Similarity Task")
            self._log("=" * 70)
            logger.info("Starting Semantic Similarity task")

            try:
                task = SemanticSimilarity(
                    encoder=self.encoder,
                    task_data=test_data.tasks.semantic_similarity,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, "semantic_similarity_task"):
                    result = task.run()
                task_results.append(result)
                logger.info(f"Semantic Similarity task completed (status: {result.status})")
            except Exception as e:
                logger.error(f"Semantic Similarity task failed: {str(e)}")
                self._log(f"Task failed: {str(e)}", level="ERROR")

        # Linguistic Robustness
        if test_data.tasks.linguistic_robustness is not None:
            self._log("\n" + "=" * 70)
            self._log("Running Linguistic Robustness Task")
            self._log("=" * 70)
            logger.info("Starting Linguistic Robustness task")

            try:
                task = LinguisticRobustness(
                    encoder=self.encoder,
                    task_data=test_data.tasks.linguistic_robustness,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, "linguistic_robustness_task"):
                    result = task.run()
                task_results.append(result)
                logger.info(f"Linguistic Robustness task completed (status: {result.status})")
            except Exception as e:
                logger.error(f"Linguistic Robustness task failed: {str(e)}")
                self._log(f"Task failed: {str(e)}", level="ERROR")

        # Vector Arithmetic
        if test_data.tasks.vector_arithmetic is not None:
            self._log("\n" + "=" * 70)
            self._log("Running Vector Arithmetic Task")
            self._log("=" * 70)
            logger.info("Starting Vector Arithmetic task")

            try:
                task = VectorArithmetic(
                    encoder=self.encoder,
                    task_data=test_data.tasks.vector_arithmetic,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, "vector_arithmetic_task"):
                    result = task.run()
                task_results.append(result)
                logger.info(f"Vector Arithmetic task completed (status: {result.status})")
            except Exception as e:
                logger.error(f"Vector Arithmetic task failed: {str(e)}")
                self._log(f"Task failed: {str(e)}", level="ERROR")

        total_runtime = time.time() - start_time

        self._log("\n" + "=" * 70)
        self._log(f"Evaluation Completed in {total_runtime:.2f}s")
        self._log("=" * 70)

        logger.info(
            f"Evaluation completed in {total_runtime:.2f}s "
            f"(total_tasks: {len(task_results)}, "
            f"successful: {sum(1 for r in task_results if r.status == 'completed')})"
        )

        # Create evaluation result
        evaluation_result = EvaluationResult(
            metadata=test_data.metadata.model_dump(),
            task_results=task_results,
            total_runtime=total_runtime,
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
        TaskConfigError
            If task not available in data or unknown task name
        TaskExecutionError
            If task execution fails

        Examples
        --------
        >>> runner = TaskRunner(encoder=encoder)
        >>> result = runner.run_task("information_retrieval", "data/test_data.json")
        >>> print(f"NDCG@10: {result.get_metric('cosine-NDCG@10'):.4f}")
        """
        logger.info(f"Running single task: {task_name} (source: {data_source})")

        # Load test data
        try:
            test_data = self.data_loader.load(data_source)
        except Exception as e:
            logger.error(f"Failed to load test data: {str(e)}")
            raise

        # Run specific task
        if task_name == "information_retrieval":
            if test_data.tasks.information_retrieval is None:
                logger.error(f"Task {task_name} not available in test data")
                raise TaskConfigError(
                    "Information Retrieval task not available in test data",
                    task_name=task_name,
                )

            try:
                task = InformationRetrieval(
                    encoder=self.encoder,
                    task_data=test_data.tasks.information_retrieval,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, f"{task_name}_execution"):
                    result = task.run()
                logger.info(f"Task {task_name} completed successfully (status: {result.status})")
                return result
            except Exception as e:
                logger.error(f"Task {task_name} execution failed: {str(e)}")
                raise TaskExecutionError(f"Task execution failed: {str(e)}", task_name=task_name) from e

        elif task_name == "semantic_similarity":
            if test_data.tasks.semantic_similarity is None:
                logger.error(f"Task {task_name} not available in test data")
                raise TaskConfigError(
                    "Semantic Similarity task not available in test data",
                    task_name=task_name,
                )

            try:
                task = SemanticSimilarity(
                    encoder=self.encoder,
                    task_data=test_data.tasks.semantic_similarity,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, f"{task_name}_execution"):
                    result = task.run()
                logger.info(f"Task {task_name} completed successfully (status: {result.status})")
                return result
            except Exception as e:
                logger.error(f"Task {task_name} execution failed: {str(e)}")
                raise TaskExecutionError(f"Task execution failed: {str(e)}", task_name=task_name) from e

        elif task_name == "linguistic_robustness":
            if test_data.tasks.linguistic_robustness is None:
                logger.error(f"Task {task_name} not available in test data")
                raise TaskConfigError(
                    "Linguistic Robustness task not available in test data",
                    task_name=task_name,
                )

            try:
                task = LinguisticRobustness(
                    encoder=self.encoder,
                    task_data=test_data.tasks.linguistic_robustness,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, f"{task_name}_execution"):
                    result = task.run()
                logger.info(f"Task {task_name} completed successfully (status: {result.status})")
                return result
            except Exception as e:
                logger.error(f"Task {task_name} execution failed: {str(e)}")
                raise TaskExecutionError(f"Task execution failed: {str(e)}", task_name=task_name) from e

        elif task_name == "vector_arithmetic":
            if test_data.tasks.vector_arithmetic is None:
                logger.error(f"Task {task_name} not available in test data")
                raise TaskConfigError(
                    "Vector Arithmetic task not available in test data",
                    task_name=task_name,
                )

            try:
                task = VectorArithmetic(
                    encoder=self.encoder,
                    task_data=test_data.tasks.vector_arithmetic,
                    device=self.device,
                    verbose=self.verbose,
                )
                with log_execution_time(logger, f"{task_name}_execution"):
                    result = task.run()
                logger.info(f"Task {task_name} completed successfully (status: {result.status})")
                return result
            except Exception as e:
                logger.error(f"Task {task_name} execution failed: {str(e)}")
                raise TaskExecutionError(f"Task execution failed: {str(e)}", task_name=task_name) from e

        else:
            logger.error(f"Unknown task name: {task_name}")
            raise TaskConfigError(f"Unknown task name: {task_name}", task_name=task_name)
