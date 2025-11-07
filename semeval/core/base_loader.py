"""Abstract base class for test data loaders.

This module defines the interface for loading and validating test data
from various sources (JSON, CSV, databases, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from .schemas import TestDataModel


class DataValidationError(Exception):
    """Exception raised when test data validation fails.

    Parameters
    ----------
    message : str
        Error message describing the validation failure
    errors : list, optional
        List of specific validation errors
    """

    def __init__(self, message: str, errors: Optional[list] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)

    def __str__(self):
        if self.errors:
            error_details = "\n".join(f"  - {err}" for err in self.errors)
            return f"{self.message}\nErrors:\n{error_details}"
        return self.message


class BaseDataLoader(ABC):
    """Abstract base class for test data loaders.

    This class defines the interface for loading test data from various
    sources and validating it against the expected schema.

    Notes
    -----
    Subclasses must implement:
    - load(): Load and validate test data
    - validate(): Validate data format

    Examples
    --------
    >>> class MyLoader(BaseDataLoader):
    ...     def load(self, source):
    ...         data = self._load_from_source(source)
    ...         return TestDataModel(**data)
    ...
    ...     def validate(self, data):
    ...         try:
    ...             TestDataModel(**data)
    ...             return True
    ...         except Exception:
    ...             return False
    """

    @abstractmethod
    def load(self, source: str, **kwargs) -> TestDataModel:
        """Load and validate test data from source.

        Parameters
        ----------
        source : str
            Data source identifier (file path, URL, database connection, etc.)
        **kwargs : dict
            Additional loader-specific parameters

        Returns
        -------
        TestDataModel
            Validated test data model instance

        Raises
        ------
        FileNotFoundError
            If source file/resource not found
        DataValidationError
            If data format is invalid
        ValueError
            If source format is not supported

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> test_data = loader.load("data/test_data.json")
        >>> print(test_data.metadata.version)
        '1.0'
        >>> print(test_data.get_enabled_tasks())
        ['information_retrieval', 'semantic_similarity']
        """
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data format against schema.

        Parameters
        ----------
        data : dict
            Raw data dictionary to validate

        Returns
        -------
        bool
            True if data is valid

        Raises
        ------
        DataValidationError
            If validation fails (with detailed error messages)

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> data = {"metadata": {...}, "information_retrieval": {...}}
        >>> loader.validate(data)
        True
        >>>
        >>> invalid_data = {"metadata": {}}  # Missing required fields
        >>> loader.validate(invalid_data)
        DataValidationError: Invalid test data format
        """
        pass

    def _check_source_exists(self, source: str) -> Path:
        """Check if source file exists and return Path object.

        Parameters
        ----------
        source : str
            File path to check

        Returns
        -------
        Path
            Validated Path object

        Raises
        ------
        FileNotFoundError
            If file does not exist

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> path = loader._check_source_exists("data/test_data.json")
        >>> print(path.exists())
        True
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(
                f"Test data file not found: {source}\n"
                f"Absolute path: {path.absolute()}"
            )

        if not path.is_file():
            raise ValueError(
                f"Source must be a file, not a directory: {source}"
            )

        return path

    def __repr__(self) -> str:
        """Return string representation of loader.

        Returns
        -------
        str
            String representation

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> repr(loader)
        'JSONDataLoader()'
        """
        return f"{self.__class__.__name__}()"
