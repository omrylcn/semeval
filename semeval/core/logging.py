"""Structured logging utilities for SemEval.

This module provides a centralized logging configuration with structured logging,
contextual information, and performance tracking.
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)


def setup_logger(
    name: str = "semeval",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Setup and configure a logger for SemEval.

    Parameters
    ----------
    name : str, optional
        Logger name (default: "semeval")
    level : int, optional
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file. If None, only console logging is enabled.
    verbose : bool, optional
        If True, use detailed format with file/line information

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger(verbose=True)
    >>> logger.info("Starting evaluation")

    >>> logger = setup_logger(log_file="logs/semeval.log")
    >>> logger.debug("Debug information")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Choose format based on verbosity
    formatter = logging.Formatter(DETAILED_FORMAT if verbose else DEFAULT_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "semeval") -> logging.Logger:
    """Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name (default: "semeval")

    Returns
    -------
    logging.Logger
        Logger instance

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Message")
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Structured logger with contextual information.

    Provides easy-to-use methods for logging with structured context,
    making logs more searchable and analyzable.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance to use. If None, creates a new one.
    default_context : dict, optional
        Default context to include in all log messages

    Examples
    --------
    >>> slogger = StructuredLogger(default_context={"model": "bert-base"})
    >>> slogger.info("Encoding started", num_texts=100, device="cuda")
    >>> # Output: INFO - Encoding started [model=bert-base, num_texts=100, device=cuda]

    >>> slogger.error("Encoding failed", error="CUDA out of memory", num_texts=1000)
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        default_context: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or get_logger()
        self.default_context = default_context or {}

    def _format_message(self, message: str, **context) -> str:
        """Format message with context.

        Parameters
        ----------
        message : str
            Log message
        **context : dict
            Context key-value pairs

        Returns
        -------
        str
            Formatted message
        """
        # Merge default context with provided context
        full_context = {**self.default_context, **context}

        if full_context:
            context_str = ", ".join(f"{k}={v}" for k, v in full_context.items())
            return f"{message} [{context_str}]"
        return message

    def debug(self, message: str, **context):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message, **context))

    def info(self, message: str, **context):
        """Log info message with context."""
        self.logger.info(self._format_message(message, **context))

    def warning(self, message: str, **context):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message, **context))

    def error(self, message: str, **context):
        """Log error message with context."""
        self.logger.error(self._format_message(message, **context))

    def critical(self, message: str, **context):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message, **context))

    def exception(self, message: str, **context):
        """Log exception with context and traceback."""
        formatted_msg = self._format_message(message, **context)
        self.logger.exception(formatted_msg)


@contextmanager
def log_execution_time(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **context,
):
    """Context manager to log execution time of an operation.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    operation : str
        Name of the operation being timed
    level : int, optional
        Logging level (default: INFO)
    **context : dict
        Additional context to include in log

    Examples
    --------
    >>> logger = get_logger()
    >>> with log_execution_time(logger, "model_loading", model="bert-base"):
    ...     model = load_model("bert-base")
    INFO - model_loading started [model=bert-base]
    INFO - model_loading completed in 2.34s [model=bert-base]

    >>> with log_execution_time(logger, "encoding", num_texts=1000):
    ...     embeddings = encoder.encode(texts)
    """
    start_time = time.time()

    # Format context
    context_str = ""
    if context:
        context_items = ", ".join(f"{k}={v}" for k, v in context.items())
        context_str = f" [{context_items}]"

    logger.log(level, f"{operation} started{context_str}")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.log(level, f"{operation} completed in {elapsed:.2f}s{context_str}")


@contextmanager
def log_errors(
    logger: logging.Logger,
    operation: str,
    reraise: bool = True,
    **context,
):
    """Context manager to log errors with context.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    operation : str
        Name of the operation
    reraise : bool, optional
        Whether to reraise the exception after logging (default: True)
    **context : dict
        Additional context to include in log

    Examples
    --------
    >>> logger = get_logger()
    >>> with log_errors(logger, "model_loading", model="bert-base"):
    ...     model = load_model("bert-base")
    # If error occurs:
    ERROR - model_loading failed [model=bert-base, error_type=RuntimeError, error=CUDA not available]

    >>> with log_errors(logger, "encoding", reraise=False, num_texts=100):
    ...     embeddings = encoder.encode(texts)
    # Error is logged but not reraised
    """
    try:
        yield
    except Exception as e:
        error_context = {
            **context,
            "error_type": type(e).__name__,
            "error": str(e),
        }
        context_str = ", ".join(f"{k}={v}" for k, v in error_context.items())
        logger.error(f"{operation} failed [{context_str}]", exc_info=True)

        if reraise:
            raise


class PerformanceLogger:
    """Logger for tracking performance metrics.

    Tracks operation counts, timings, and provides summary statistics.

    Examples
    --------
    >>> perf_logger = PerformanceLogger()
    >>> perf_logger.record("encoding", 2.5, num_texts=1000)
    >>> perf_logger.record("encoding", 2.3, num_texts=1000)
    >>> perf_logger.summary()
    {'encoding': {'count': 2, 'total_time': 4.8, 'avg_time': 2.4, ...}}
    """

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def record(self, operation: str, elapsed_time: float, **metadata):
        """Record an operation's performance.

        Parameters
        ----------
        operation : str
            Operation name
        elapsed_time : float
            Time elapsed in seconds
        **metadata : dict
            Additional metadata about the operation
        """
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append({"time": elapsed_time, "metadata": metadata})

    @contextmanager
    def measure(self, operation: str, **metadata):
        """Context manager to measure operation time.

        Parameters
        ----------
        operation : str
            Operation name
        **metadata : dict
            Additional metadata

        Examples
        --------
        >>> perf_logger = PerformanceLogger()
        >>> with perf_logger.measure("encoding", num_texts=1000):
        ...     embeddings = encoder.encode(texts)
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.record(operation, elapsed, **metadata)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all operations.

        Returns
        -------
        dict
            Summary statistics with count, total_time, avg_time, min_time, max_time
        """
        summary = {}
        for operation, records in self.metrics.items():
            times = [r["time"] for r in records]
            summary[operation] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
            }
        return summary

    def report(self, logger: Optional[logging.Logger] = None):
        """Log performance summary.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger to use. If None, uses default logger.
        """
        logger = logger or get_logger()
        summary = self.summary()

        logger.info("=== Performance Summary ===")
        for operation, stats in summary.items():
            logger.info(
                f"{operation}: "
                f"count={stats['count']}, "
                f"total={stats['total_time']:.2f}s, "
                f"avg={stats['avg_time']:.2f}s, "
                f"min={stats['min_time']:.2f}s, "
                f"max={stats['max_time']:.2f}s"
            )


# Module-level default logger
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """Get or create the default SemEval logger.

    Returns
    -------
    logging.Logger
        Default logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


# Convenience functions using default logger
def debug(message: str, **context):
    """Log debug message using default logger."""
    slogger = StructuredLogger(get_default_logger())
    slogger.debug(message, **context)


def info(message: str, **context):
    """Log info message using default logger."""
    slogger = StructuredLogger(get_default_logger())
    slogger.info(message, **context)


def warning(message: str, **context):
    """Log warning message using default logger."""
    slogger = StructuredLogger(get_default_logger())
    slogger.warning(message, **context)


def error(message: str, **context):
    """Log error message using default logger."""
    slogger = StructuredLogger(get_default_logger())
    slogger.error(message, **context)


def critical(message: str, **context):
    """Log critical message using default logger."""
    slogger = StructuredLogger(get_default_logger())
    slogger.critical(message, **context)
