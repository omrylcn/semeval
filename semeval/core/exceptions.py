"""Custom exception hierarchy for SemEval.

This module defines all custom exceptions used throughout the SemEval framework,
providing clear error messages and proper exception chaining.
"""

from typing import List, Optional


class SemEvalError(Exception):
    """Base exception for all SemEval errors.

    This is the root exception class. All custom exceptions should inherit from this.

    Parameters
    ----------
    message : str
        Error message
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# ============================================================================
# Data-related Errors
# ============================================================================


class DataError(SemEvalError):
    """Base class for data-related errors.

    Raised when there are issues with test data loading, validation, or format.
    """

    pass


class DataValidationError(DataError):
    """Exception raised when test data validation fails.

    Parameters
    ----------
    message : str
        Error message describing the validation failure
    errors : list, optional
        List of specific validation errors
    file_path : str, optional
        Path to the file that failed validation
    """

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        file_path: Optional[str] = None,
    ):
        self.errors = errors or []
        self.file_path = file_path
        super().__init__(message)

    def __str__(self):
        msg = self.message
        if self.file_path:
            msg = f"{msg}\nFile: {self.file_path}"
        if self.errors:
            error_details = "\n".join(f"  • {err}" for err in self.errors)
            msg = f"{msg}\nValidation Errors:\n{error_details}"
        return msg


class DataLoadError(DataError):
    """Exception raised when data loading fails.

    Parameters
    ----------
    message : str
        Error message
    file_path : str, optional
        Path to the file that failed to load
    """

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        if file_path:
            message = f"{message}\nFile: {file_path}"
        super().__init__(message)


class DataFormatError(DataError):
    """Exception raised when data format is invalid.

    Parameters
    ----------
    message : str
        Error message
    expected_format : str, optional
        Expected data format
    actual_format : str, optional
        Actual data format received
    """

    def __init__(
        self,
        message: str,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
    ):
        self.expected_format = expected_format
        self.actual_format = actual_format

        if expected_format and actual_format:
            message = (
                f"{message}\n"
                f"Expected format: {expected_format}\n"
                f"Actual format: {actual_format}"
            )
        super().__init__(message)


# ============================================================================
# Model-related Errors
# ============================================================================


class ModelError(SemEvalError):
    """Base class for model-related errors.

    Raised when there are issues with model loading or inference.
    """

    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails.

    Parameters
    ----------
    message : str
        Error message
    model_name : str, optional
        Name of the model that failed to load
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        if model_name:
            message = f"{message}\nModel: {model_name}"
        super().__init__(message)


class EncoderError(ModelError):
    """Exception raised when encoding fails.

    Parameters
    ----------
    message : str
        Error message
    model_name : str, optional
        Name of the encoder model
    num_texts : int, optional
        Number of texts being encoded
    device : str, optional
        Device being used (cpu, cuda, mps)
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        num_texts: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_texts = num_texts
        self.device = device

        context_parts = []
        if model_name:
            context_parts.append(f"Model: {model_name}")
        if num_texts is not None:
            context_parts.append(f"Texts: {num_texts}")
        if device:
            context_parts.append(f"Device: {device}")

        if context_parts:
            message = f"{message}\nContext: {', '.join(context_parts)}"

        super().__init__(message)


class ModelDimensionError(ModelError):
    """Exception raised when model dimension is incompatible.

    Parameters
    ----------
    message : str
        Error message
    expected_dim : int, optional
        Expected embedding dimension
    actual_dim : int, optional
        Actual embedding dimension
    model_name : str, optional
        Name of the model
    """

    def __init__(
        self,
        message: str,
        expected_dim: Optional[int] = None,
        actual_dim: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.model_name = model_name

        if expected_dim and actual_dim:
            message = (
                f"{message}\n"
                f"Expected dimension: {expected_dim}\n"
                f"Actual dimension: {actual_dim}"
            )
        if model_name:
            message = f"{message}\nModel: {model_name}"

        super().__init__(message)


# ============================================================================
# Task-related Errors
# ============================================================================


class TaskError(SemEvalError):
    """Base class for task execution errors.

    Raised when evaluation tasks fail to execute.
    """

    pass


class TaskExecutionError(TaskError):
    """Exception raised when task execution fails.

    Parameters
    ----------
    message : str
        Error message
    task_name : str, optional
        Name of the task that failed
    """

    def __init__(self, message: str, task_name: Optional[str] = None):
        self.task_name = task_name
        if task_name:
            message = f"{message}\nTask: {task_name}"
        super().__init__(message)


class TaskConfigError(TaskError):
    """Exception raised when task configuration is invalid.

    Parameters
    ----------
    message : str
        Error message
    task_name : str, optional
        Name of the task
    config_field : str, optional
        Configuration field that is invalid
    """

    def __init__(
        self,
        message: str,
        task_name: Optional[str] = None,
        config_field: Optional[str] = None,
    ):
        self.task_name = task_name
        self.config_field = config_field

        context_parts = []
        if task_name:
            context_parts.append(f"Task: {task_name}")
        if config_field:
            context_parts.append(f"Field: {config_field}")

        if context_parts:
            message = f"{message}\n{', '.join(context_parts)}"

        super().__init__(message)


# ============================================================================
# Metric-related Errors
# ============================================================================


class MetricError(SemEvalError):
    """Base class for metric computation errors.

    Raised when metric calculations fail.
    """

    pass


class MetricComputationError(MetricError):
    """Exception raised when metric computation fails.

    Parameters
    ----------
    message : str
        Error message
    metric_name : str, optional
        Name of the metric
    """

    def __init__(self, message: str, metric_name: Optional[str] = None):
        self.metric_name = metric_name
        if metric_name:
            message = f"{message}\nMetric: {metric_name}"
        super().__init__(message)


class MetricInputError(MetricError):
    """Exception raised when metric input is invalid.

    Parameters
    ----------
    message : str
        Error message
    metric_name : str, optional
        Name of the metric
    parameter : str, optional
        Parameter that is invalid
    """

    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        parameter: Optional[str] = None,
    ):
        self.metric_name = metric_name
        self.parameter = parameter

        context_parts = []
        if metric_name:
            context_parts.append(f"Metric: {metric_name}")
        if parameter:
            context_parts.append(f"Parameter: {parameter}")

        if context_parts:
            message = f"{message}\n{', '.join(context_parts)}"

        super().__init__(message)


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigError(SemEvalError):
    """Base class for configuration errors.

    Raised when configuration is invalid or missing.
    """

    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails.

    Parameters
    ----------
    message : str
        Error message
    config_field : str, optional
        Configuration field that is invalid
    config_file : str, optional
        Path to config file
    """

    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.config_field = config_field
        self.config_file = config_file

        context_parts = []
        if config_field:
            context_parts.append(f"Field: {config_field}")
        if config_file:
            context_parts.append(f"File: {config_file}")

        if context_parts:
            message = f"{message}\n{', '.join(context_parts)}"

        super().__init__(message)


class ConfigLoadError(ConfigError):
    """Exception raised when configuration loading fails.

    Parameters
    ----------
    message : str
        Error message
    config_file : str, optional
        Path to config file that failed to load
    """

    def __init__(self, message: str, config_file: Optional[str] = None):
        self.config_file = config_file
        if config_file:
            message = f"{message}\nConfig file: {config_file}"
        super().__init__(message)


# ============================================================================
# Export/Report Errors
# ============================================================================


class ExportError(SemEvalError):
    """Base class for export/report errors.

    Raised when result export or report generation fails.
    """

    pass


class ExportFormatError(ExportError):
    """Exception raised when export format is unsupported.

    Parameters
    ----------
    message : str
        Error message
    format : str, optional
        Requested format
    supported_formats : list, optional
        List of supported formats
    """

    def __init__(
        self,
        message: str,
        format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
    ):
        self.format = format
        self.supported_formats = supported_formats

        if format and supported_formats:
            message = (
                f"{message}\n"
                f"Requested format: {format}\n"
                f"Supported formats: {', '.join(supported_formats)}"
            )

        super().__init__(message)


class ReportGenerationError(ExportError):
    """Exception raised when report generation fails.

    Parameters
    ----------
    message : str
        Error message
    output_path : str, optional
        Path where report was to be saved
    """

    def __init__(self, message: str, output_path: Optional[str] = None):
        self.output_path = output_path
        if output_path:
            message = f"{message}\nOutput path: {output_path}"
        super().__init__(message)
