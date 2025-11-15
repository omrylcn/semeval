"""JSON data loader implementation.

This module provides functionality to load and validate test data from JSON files.
"""

import json
from typing import Any, Dict

from pydantic import ValidationError

from ..base_loader import BaseDataLoader
from ..exceptions import DataFormatError, DataValidationError
from ..logging import get_logger
from ..schemas import TestDataModel

logger = get_logger("semeval")


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

    def __init__(self, encoding: str = "utf-8"):
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
        DataLoadError
            If source file not found
        DataValidationError
            If JSON is invalid or doesn't match schema
        DataFormatError
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
        logger.info(f"Loading test data from: {source}")

        # Check if file exists
        path = self._check_source_exists(source)

        # Load JSON
        try:
            with open(path, "r", encoding=self.encoding) as f:
                data = json.load(f)
            logger.debug(f"JSON file loaded successfully: {source}")
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in file: {source} (Line {e.lineno}, Column {e.colno})"
            )
            raise DataFormatError(
                f"Invalid JSON in file: {source}",
                expected_format="Valid JSON",
                actual_format=f"JSON parse error at line {e.lineno}",
            ) from e
        except Exception as e:
            logger.error(f"Failed to read file: {source} - {str(e)}")
            raise DataValidationError(
                f"Failed to read file: {source}",
                errors=[str(e)],
                file_path=source,
            ) from e

        # Validate and parse
        try:
            test_data = TestDataModel(**data)
            logger.info(
                f"Test data validated successfully: {source} "
                f"(tasks: {', '.join(test_data.get_enabled_tasks())})"
            )
            return test_data
        except ValidationError as e:
            # Extract error details from Pydantic
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            logger.error(
                f"Invalid test data format in file: {source} "
                f"({len(errors)} validation errors)"
            )
            raise DataValidationError(
                f"Invalid test data format in file: {source}",
                errors=errors,
                file_path=source,
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
        logger.debug("Validating test data format")
        try:
            TestDataModel(**data)
            logger.debug("Test data validation successful")
            return True
        except ValidationError as e:
            # Extract error details
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            logger.error(f"Test data validation failed ({len(errors)} errors)")
            raise DataValidationError("Invalid test data format", errors=errors) from e

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
        DataFormatError
            If JSON is invalid
        DataValidationError
            If data doesn't match schema

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
        logger.debug("Loading test data from JSON string")
        try:
            data = json.loads(json_string)
            logger.debug("JSON string parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON string (Line {e.lineno}, Column {e.colno}: {e.msg})"
            )
            raise DataFormatError(
                "Invalid JSON string",
                expected_format="Valid JSON",
                actual_format=f"JSON parse error at line {e.lineno}",
            ) from e

        try:
            test_data = TestDataModel(**data)
            logger.info(
                f"Test data from string validated successfully "
                f"(tasks: {', '.join(test_data.get_enabled_tasks())})"
            )
            return test_data
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            logger.error(f"Invalid test data format ({len(errors)} errors)")
            raise DataValidationError("Invalid test data format", errors=errors) from e
