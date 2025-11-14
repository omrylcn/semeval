"""Tests for data loaders."""

import json
import tempfile
from pathlib import Path

import pytest

from semeval.core.base_loader import DataValidationError
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
        with pytest.raises(DataValidationError):
            loader.load(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_json_loader_missing_file():
    """Test loader handles missing file."""
    loader = JSONDataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path/file.json")
