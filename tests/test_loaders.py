"""Tests for data loaders."""

import json
import tempfile
from pathlib import Path

import pytest

from semeval.core.exceptions import DataFormatError, DataLoadError, DataValidationError
from semeval.core.loaders import JSONDataLoader
from semeval.core.schemas import TestDataModel


def test_json_loader_initialization():
    """Test JSONDataLoader can be initialized."""
    loader = JSONDataLoader()
    assert loader is not None


def test_json_loader_with_minimal_valid_data():
    """Test loading minimal valid JSON data."""
    minimal_data = {
        "metadata": {
            "version": "1.0",
            "description": "Test",
            "language": "tr",
            "created_by": "test",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Semantic Similarity Test",
                "triplets": [
                    {
                        "id": "test-1",
                        "anchor": "Test anchor",
                        "positive": "Test positive",
                        "negative": "Test negative",
                    }
                ],
            }
        },
    }

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(minimal_data, tmp_file)
        tmp_path = tmp_file.name

    try:
        loader = JSONDataLoader()
        test_data = loader.load(tmp_path)

        assert isinstance(test_data, TestDataModel)
        assert test_data.metadata.version == "1.0"
        assert test_data.tasks.semantic_similarity is not None
        assert len(test_data.tasks.semantic_similarity.triplets) == 1
    finally:
        Path(tmp_path).unlink()


def test_json_loader_invalid_json():
    """Test loader handles invalid JSON gracefully."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_file.write("{invalid json content")
        tmp_path = tmp_file.name

    try:
        loader = JSONDataLoader()
        with pytest.raises(DataFormatError):
            loader.load(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_json_loader_missing_file():
    """Test loader handles missing file."""
    loader = JSONDataLoader()
    with pytest.raises(DataLoadError):
        loader.load("/nonexistent/path/file.json")


def test_json_loader_directory_not_file():
    """Test loader rejects directory instead of file."""
    loader = JSONDataLoader()
    with pytest.raises(DataLoadError, match="must be a file, not a directory"):
        loader.load(str(Path.cwd()))


def test_json_loader_missing_required_metadata_fields():
    """Test loader validates required metadata fields."""
    invalid_data = {
        "metadata": {
            "version": "1.0"
            # Missing description, language, created_by
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Test",
                "triplets": [
                    {
                        "id": "test-1",
                        "anchor": "anchor",
                        "positive": "positive",
                        "negative": "negative",
                    }
                ],
            }
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(invalid_data, tmp_file)
        tmp_path = tmp_file.name

    try:
        loader = JSONDataLoader()
        with pytest.raises(DataValidationError) as exc_info:
            loader.load(tmp_path)
        # Check error message contains field details
        assert "description" in str(exc_info.value) or "Field required" in str(
            exc_info.value
        )
    finally:
        Path(tmp_path).unlink()


def test_json_loader_validate_valid_data():
    """Test validate method with valid data."""
    valid_data = {
        "metadata": {
            "version": "1.0",
            "description": "Test dataset",
            "language": "tr",
            "created_by": "test",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Semantic Similarity",
                "triplets": [
                    {
                        "id": "test-1",
                        "anchor": "Anchor text",
                        "positive": "Positive text",
                        "negative": "Negative text",
                    }
                ],
            }
        },
    }

    loader = JSONDataLoader()
    # Should return True for valid data
    assert loader.validate(valid_data) is True


def test_json_loader_validate_invalid_data():
    """Test validate method with invalid data."""
    invalid_data = {
        "metadata": {"version": "1.0"}  # Missing required fields
    }

    loader = JSONDataLoader()
    with pytest.raises(DataValidationError) as exc_info:
        loader.validate(invalid_data)

    # Check error message has details
    error_str = str(exc_info.value)
    assert "Invalid test data format" in error_str


def test_json_loader_load_from_string_valid():
    """Test loading from JSON string."""
    json_string = json.dumps(
        {
            "metadata": {
                "version": "1.0",
                "description": "Test",
                "language": "tr",
                "created_by": "test",
            },
            "tasks": {
                "semantic_similarity": {
                    "name": "Test",
                    "triplets": [
                        {
                            "id": "test-1",
                            "anchor": "A",
                            "positive": "B",
                            "negative": "C",
                        }
                    ],
                }
            },
        }
    )

    loader = JSONDataLoader()
    test_data = loader.load_from_string(json_string)

    assert isinstance(test_data, TestDataModel)
    assert test_data.metadata.version == "1.0"


def test_json_loader_load_from_string_invalid_json():
    """Test load_from_string with invalid JSON."""
    invalid_json = "{not valid json"

    loader = JSONDataLoader()
    with pytest.raises(DataFormatError, match="Invalid JSON string"):
        loader.load_from_string(invalid_json)


def test_json_loader_load_from_string_invalid_schema():
    """Test load_from_string with valid JSON but invalid schema."""
    json_string = json.dumps({"metadata": {}})  # Missing required fields

    loader = JSONDataLoader()
    with pytest.raises(DataValidationError, match="Invalid test data format"):
        loader.load_from_string(json_string)


def test_json_loader_custom_encoding():
    """Test JSONDataLoader with custom encoding."""
    loader = JSONDataLoader(encoding="utf-8")
    assert loader.encoding == "utf-8"

    loader_latin = JSONDataLoader(encoding="latin-1")
    assert loader_latin.encoding == "latin-1"


def test_json_loader_repr():
    """Test string representation of loader."""
    loader = JSONDataLoader()
    assert repr(loader) == "JSONDataLoader()"


def test_data_validation_error_with_errors():
    """Test DataValidationError formats multiple errors correctly."""
    errors = ["Error 1: Field missing", "Error 2: Invalid type"]
    exc = DataValidationError("Validation failed", errors=errors)

    error_str = str(exc)
    assert "Validation failed" in error_str
    assert "Error 1: Field missing" in error_str
    assert "Error 2: Invalid type" in error_str


def test_data_validation_error_without_errors():
    """Test DataValidationError with no detailed errors."""
    exc = DataValidationError("Simple error")

    error_str = str(exc)
    assert error_str == "Simple error"
    assert exc.errors == []


def test_json_loader_missing_triplets():
    """Test loader validates triplets list is not empty."""
    invalid_data = {
        "metadata": {
            "version": "1.0",
            "description": "Test",
            "language": "tr",
            "created_by": "test",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Test",
                "triplets": [],  # Empty list should fail validation
            }
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(invalid_data, tmp_file)
        tmp_path = tmp_file.name

    try:
        loader = JSONDataLoader()
        with pytest.raises(DataValidationError):
            loader.load(tmp_path)
    finally:
        Path(tmp_path).unlink()
