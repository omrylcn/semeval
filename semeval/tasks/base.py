"""Abstract base class for evaluation tasks.

This module defines the interface that all evaluation task implementations
must follow.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskResult(BaseModel):
    """Result of running an evaluation task.

    Attributes
    ----------
    task_name : str
        Name of the task
    status : str
        Execution status ('success', 'failed', 'skipped')
    metrics : dict
        Dictionary of metric names to values
    runtime_seconds : float
        Task execution time in seconds
    timestamp : str
        ISO format timestamp of task completion
    error_message : str, optional
        Error message if task failed
    metadata : dict, optional
        Additional task-specific metadata

    Examples
    --------
    >>> result = TaskResult(
    ...     task_name="information_retrieval",
    ...     status="success",
    ...     metrics={"ndcg@10": 0.95, "mrr@10": 1.0},
    ...     runtime_seconds=12.5
    ... )
    >>> print(result.metrics["ndcg@10"])
    0.95
    """

    task_name: str
    status: str
    metrics: Dict[str, Any]
    runtime_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def is_success(self) -> bool:
        """Check if task completed successfully.

        Returns
        -------
        bool
            True if status is 'success'
        """
        return self.status == "success"

    def get_metric(self, metric_name: str, default: Any = None) -> Any:
        """Get a specific metric value.

        Parameters
        ----------
        metric_name : str
            Name of the metric to retrieve
        default : any, optional
            Default value if metric not found

        Returns
        -------
        any
            Metric value or default

        Examples
        --------
        >>> result.get_metric("ndcg@10")
        0.95
        >>> result.get_metric("missing_metric", default=0.0)
        0.0
        """
        return self.metrics.get(metric_name, default)


class BaseTask(ABC):
    """Abstract base class for evaluation tasks.

    All evaluation tasks must inherit from this class and implement
    the required methods.

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to use for embeddings
    task_data : any
        Task-specific data (schema depends on task type)
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps')
    verbose : bool, optional
        If True, print progress information. Default is False

    Attributes
    ----------
    encoder : BaseEncoder
        Text encoder instance
    task_data : any
        Task configuration and test data
    device : str
        Computation device
    verbose : bool
        Verbosity flag

    Notes
    -----
    Subclasses must implement:
    - run(): Execute the evaluation task
    - get_task_name(): Return task identifier

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.tasks import InformationRetrieval
    >>>
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>> task = InformationRetrieval(encoder=encoder, task_data=ir_data)
    >>> result = task.run()
    >>> print(result.metrics["ndcg@10"])
    """

    def __init__(
        self,
        encoder,  # BaseEncoder type hint causes circular import
        task_data: Any,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize base task.

        Parameters
        ----------
        encoder : BaseEncoder
            Text encoder instance
        task_data : any
            Task-specific data
        device : str, optional
            Device for computation
        verbose : bool, optional
            Verbosity flag
        """
        self.encoder = encoder
        self.task_data = task_data
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def run(self) -> TaskResult:
        """Execute the evaluation task.

        Returns
        -------
        TaskResult
            Task execution results including metrics

        Raises
        ------
        RuntimeError
            If task execution fails

        Examples
        --------
        >>> task = InformationRetrieval(encoder=encoder, task_data=data)
        >>> result = task.run()
        >>> print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
        NDCG@10: 0.9872
        """
        pass

    @abstractmethod
    def get_task_name(self) -> str:
        """Return the name/identifier of this task.

        Returns
        -------
        str
            Task name (e.g., 'information_retrieval')

        Examples
        --------
        >>> task = InformationRetrieval(encoder=encoder, task_data=data)
        >>> task.get_task_name()
        'information_retrieval'
        """
        pass

    def _log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled.

        Parameters
        ----------
        message : str
            Message to log
        level : str, optional
            Log level ('INFO', 'WARNING', 'ERROR'). Default is 'INFO'

        Examples
        --------
        >>> task = InformationRetrieval(encoder=encoder, task_data=data, verbose=True)
        >>> task._log("Starting evaluation")
        [INFO] Starting evaluation
        """
        if self.verbose:
            print(f"[{level}] {message}")

    @staticmethod
    def get_export_columns(result: TaskResult) -> Dict[str, Any]:
        """Get columns for CSV/DataFrame export.

        Tasks can override this to provide custom export columns.
        Default implementation returns basic columns.

        Parameters
        ----------
        result : TaskResult
            Task result to export

        Returns
        -------
        dict
            Dictionary of column_name -> value for export

        Examples
        --------
        >>> class MyTask(BaseTask):
        ...     @staticmethod
        ...     def get_export_columns(result):
        ...         return {'custom_metric': result.metrics.get('custom', 0)}
        """
        # Default: return all metrics as-is
        return result.metrics

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format task results as markdown lines.

        Tasks can override this to provide custom markdown formatting.
        Default implementation returns basic metrics table.

        Parameters
        ----------
        result : TaskResult
            Task result to format

        Returns
        -------
        list of str
            List of markdown lines

        Examples
        --------
        >>> class MyTask(BaseTask):
        ...     @staticmethod
        ...     def format_markdown_report(result):
        ...         return ['## My Task', f'Score: {result.metrics["score"]}']
        """
        import pandas as pd

        lines = []
        if result.metrics:
            # Build data for DataFrame
            data = {"Metric": [], "Value": []}
            for key, value in result.metrics.items():
                data["Metric"].append(key)
                if isinstance(value, float):
                    data["Value"].append(f"{value:.4f}")
                else:
                    data["Value"].append(str(value))

            df = pd.DataFrame(data)
            lines.append(df.to_markdown(index=False))

        return lines

    def __repr__(self) -> str:
        """Return string representation of task.

        Returns
        -------
        str
            String representation

        Examples
        --------
        >>> task = InformationRetrieval(encoder=encoder, task_data=data)
        >>> repr(task)
        'InformationRetrieval(encoder=SentenceTransformerEncoder(...), enabled=True)'
        """
        enabled = getattr(self.task_data, "enabled", True)
        return (
            f"{self.__class__.__name__}("
            f"encoder={self.encoder.__class__.__name__}(...), "
            f"enabled={enabled})"
        )
