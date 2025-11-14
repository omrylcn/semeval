"""JSON data loader implementation.

This module provides functionality to load and validate test data from JSON files.
"""

import json
from typing import Any, Dict

from pydantic import ValidationError

from ..base_loader import BaseDataLoader, DataValidationError
from ..schemas import TestDataModel


class JSONDataLoader(BaseDataLoader):
    """Load test data from JSON files.

    This loader reads JSON files and validates them against the
    TestDataModel schema using Pydantic.

    Parameters
    ----------
    encoding : str, optional
        File encoding. Default is 'utf-8'

    Examples
    --------
    >>> loader = JSONDataLoader()
    >>> test_data = loader.load("data/test_data.json")
    >>> print(test_data.metadata.version)
    '1.0'
    >>>
    >>> # Check which tasks are enabled
    >>> print(test_data.get_enabled_tasks())
    ['information_retrieval', 'semantic_similarity']
    >>>
    >>> # Validate before loading
    >>> import json
    >>> with open("data/test_data.json") as f:
    ...     data = json.load(f)
    >>> loader.validate(data)
    True
    """

    def __init__(self, encoding: str = 'utf-8'):
        """Initialize JSON data loader.

        Parameters
        ----------
        encoding : str, optional
            File encoding. Default is 'utf-8'
        """
        self.encoding = encoding

    def load(self, source: str, **kwargs) -> TestDataModel:
        """Load and validate test data from JSON file.

        Parameters
        ----------
        source : str
            Path to JSON file
        **kwargs : dict
            Additional parameters (not used, kept for compatibility)

        Returns
        -------
        TestDataModel
            Validated test data model

        Raises
        ------
        FileNotFoundError
            If source file not found
        DataValidationError
            If JSON is invalid or doesn't match schema
        json.JSONDecodeError
            If file contains invalid JSON

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> test_data = loader.load("data/test_data.json")
        >>>
        >>> # Access specific task data
        >>> ir_data = test_data.information_retrieval
        >>> print(f"Corpus size: {len(ir_data.corpus)}")
        Corpus size: 35
        >>> print(f"Query count: {len(ir_data.queries)}")
        Query count: 12
        """
        # Check if file exists
        path = self._check_source_exists(source)

        # Load JSON
        try:
            with open(path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataValidationError(
                f"Invalid JSON in file: {source}",
                errors=[f"Line {e.lineno}, Column {e.colno}: {e.msg}"]
            ) from e
        except Exception as e:
            raise DataValidationError(
                f"Failed to read file: {source}",
                errors=[str(e)]
            ) from e

        # Validate and parse
        try:
            return TestDataModel(**data)
        except ValidationError as e:
            # Extract error details from Pydantic
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")

            raise DataValidationError(
                f"Invalid test data format in file: {source}",
                errors=errors
            ) from e

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
        >>> data = {
        ...     "metadata": {
        ...         "version": "1.0",
        ...         "description": "Test data"
        ...     },
        ...     "information_retrieval": {
        ...         "name": "IR Test",
        ...         "corpus": {"c0": "doc text"},
        ...         "queries": {"q0": "query text"},
        ...         "relevant_docs": {"q0": {"c0": 2}}
        ...     }
        ... }
        >>> loader.validate(data)
        True
        >>>
        >>> # Invalid data
        >>> invalid_data = {"metadata": {}}  # Missing required fields
        >>> try:
        ...     loader.validate(invalid_data)
        ... except DataValidationError as e:
        ...     print(e)
        Invalid test data format
        Errors:
          - metadata -> version: field required
          - metadata -> description: field required
        """
        try:
            TestDataModel(**data)
            return True
        except ValidationError as e:
            # Extract error details
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")

            raise DataValidationError(
                "Invalid test data format",
                errors=errors
            ) from e

    def load_from_string(self, json_string: str) -> TestDataModel:
        """Load test data from JSON string.

        Parameters
        ----------
        json_string : str
            JSON string containing test data

        Returns
        -------
        TestDataModel
            Validated test data model

        Raises
        ------
        DataValidationError
            If JSON is invalid or doesn't match schema

        Examples
        --------
        >>> loader = JSONDataLoader()
        >>> json_str = '''
        ... {
        ...     "metadata": {"version": "1.0", "description": "Test"},
        ...     "information_retrieval": {...}
        ... }
        ... '''
        >>> test_data = loader.load_from_string(json_str)
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise DataValidationError(
                "Invalid JSON string",
                errors=[f"Line {e.lineno}, Column {e.colno}: {e.msg}"]
            ) from e

        try:
            return TestDataModel(**data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error['loc'])
                msg = error['msg']
                errors.append(f"{loc}: {msg}")

            raise DataValidationError(
                "Invalid test data format",
                errors=errors
            ) from e
