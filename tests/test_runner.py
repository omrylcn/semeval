"""Tests for TaskRunner."""

import json
import tempfile
from pathlib import Path

import pytest

from semeval.core.encoders import SentenceTransformerEncoder
from semeval.core.exceptions import DataLoadError
from semeval.core.runner import EvaluationResult, TaskRunner


@pytest.fixture
def small_model_encoder():
    """Create a small encoder for fast testing."""
    return SentenceTransformerEncoder(
        "sentence-transformers/paraphrase-albert-small-v2", device="cpu"
    )


@pytest.fixture
def minimal_test_data():
    """Create minimal valid test data."""
    return {
        "metadata": {
            "version": "1.0.0",
            "description": "Minimal test data",
            "language": "en",
            "created_by": "pytest",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Semantic Similarity Test",
                "enabled": True,
                "triplets": [
                    {
                        "id": "test_triplet_1",
                        "anchor": "The cat sits on the mat",
                        "positive": "A cat is sitting on a mat",
                        "negative": "Dogs are playing in the park",
                        "category": "animals",
                    },
                    {
                        "id": "test_triplet_2",
                        "anchor": "Machine learning is fascinating",
                        "positive": "AI and ML are very interesting",
                        "negative": "Cooking recipes for beginners",
                        "category": "technology",
                    },
                ],
            }
        },
    }


class TestTaskRunner:
    """Test cases for TaskRunner."""

    def test_runner_initialization(self, small_model_encoder):
        """Test TaskRunner can be initialized."""
        runner = TaskRunner(encoder=small_model_encoder)
        assert runner is not None
        assert runner.encoder == small_model_encoder
        assert runner.data_loader is not None

    def test_runner_with_verbose(self, small_model_encoder):
        """Test TaskRunner with verbose mode."""
        runner = TaskRunner(encoder=small_model_encoder, verbose=True)
        assert runner.verbose is True

    def test_runner_run_with_minimal_data(self, small_model_encoder, minimal_test_data):
        """Test running evaluation with minimal data."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(minimal_test_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            runner = TaskRunner(encoder=small_model_encoder, verbose=False)
            result = runner.run(tmp_path)

            # Check result type
            assert isinstance(result, EvaluationResult)

            # Check metadata
            assert result.metadata["version"] == "1.0.0"
            assert result.metadata["description"] == "Minimal test data"

            # Check task results
            assert len(result.task_results) > 0
            assert result.task_results[0].task_name == "semantic_similarity"

            # Check runtime
            assert result.total_runtime > 0

            # Check summary can be generated
            summary = result.get_summary()
            assert isinstance(summary, dict)

        finally:
            Path(tmp_path).unlink()

    def test_runner_missing_file(self, small_model_encoder):
        """Test runner handles missing file gracefully."""
        runner = TaskRunner(encoder=small_model_encoder)

        with pytest.raises(DataLoadError):
            runner.run("/nonexistent/file.json")

    def test_runner_run_task_single(self, small_model_encoder, minimal_test_data):
        """Test running a single specific task."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(minimal_test_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            runner = TaskRunner(encoder=small_model_encoder, verbose=False)
            result = runner.run_task("semantic_similarity", tmp_path)

            # Check result
            assert result.task_name == "semantic_similarity"
            assert result.status in ["success", "failed", "completed"]
            assert result.metrics is not None

        finally:
            Path(tmp_path).unlink()

    def test_evaluation_result_methods(self, small_model_encoder, minimal_test_data):
        """Test EvaluationResult methods."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(minimal_test_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            runner = TaskRunner(encoder=small_model_encoder, verbose=False)
            result = runner.run(tmp_path)

            # Test get_summary
            summary = result.get_summary()
            assert "total_runtime" in summary
            assert "metadata" in summary
            assert "tasks" in summary

            # Test get_task_result
            task_result = result.get_task_result("semantic_similarity")
            assert task_result is not None
            assert task_result.task_name == "semantic_similarity"

            # Test that summary contains task data
            assert "semantic_similarity" in summary["tasks"]
            assert summary["tasks"]["semantic_similarity"]["status"] == "success"

        finally:
            Path(tmp_path).unlink()

    def test_runner_log_method(self, small_model_encoder):
        """Test runner's internal logging method."""
        runner = TaskRunner(encoder=small_model_encoder, verbose=True)

        # Should not raise any errors
        runner._log("Test message")
        runner._log("Warning message", level="WARNING")
        runner._log("Error message", level="ERROR")
