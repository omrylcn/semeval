"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from semeval.cli.commands.eval_cmd import eval as eval_cmd
from semeval.cli.commands.validate import validate as validate_cmd


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_test_data_file():
    """Create a temporary valid test data file."""
    data = {
        "metadata": {
            "version": "1.0",
            "description": "Test data",
            "language": "en",
            "created_by": "pytest",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "Test Similarity",
                "triplets": [
                    {
                        "id": "t1",
                        "anchor": "hello",
                        "positive": "hi",
                        "negative": "goodbye",
                    }
                ],
            }
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(data, tmp_file)
        tmp_path = tmp_file.name

    yield tmp_path

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_with_valid_file(self, cli_runner, valid_test_data_file):
        """Test validate command with valid file."""
        app = typer.Typer()
        app.command()(validate_cmd)

        result = cli_runner.invoke(app, [valid_test_data_file])

        assert result.exit_code == 0
        assert "Schema validation passed" in result.stdout
        assert "Enabled tasks:" in result.stdout

    def test_validate_with_nonexistent_file(self, cli_runner):
        """Test validate command with non-existent file."""
        app = typer.Typer()
        app.command()(validate_cmd)

        result = cli_runner.invoke(app, ["/nonexistent/file.json"])

        # Typer will fail before reaching the command due to exists=True
        assert result.exit_code != 0

    def test_validate_with_invalid_json(self, cli_runner):
        """Test validate command with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write("{invalid json")
            tmp_path = tmp_file.name

        try:
            app = typer.Typer()
            app.command()(validate_cmd)

            result = cli_runner.invoke(app, [tmp_path])

            assert result.exit_code == 1
            assert "Validation failed" in result.stdout or "error" in result.stdout.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_with_strict_mode_and_warnings(self, cli_runner):
        """Test validate with strict mode when there are warnings."""
        # Create data with < 50 triplets (will trigger warning)
        data = {
            "metadata": {
                "version": "1.0",
                "description": "Small dataset",
                "language": "en",
                "created_by": "pytest",
            },
            "tasks": {
                "semantic_similarity": {
                    "name": "Small Test",
                    "triplets": [
                        {
                            "id": f"t{i}",
                            "anchor": f"text{i}",
                            "positive": f"pos{i}",
                            "negative": f"neg{i}",
                        }
                        for i in range(10)  # Less than 50
                    ],
                }
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(data, tmp_file)
            tmp_path = tmp_file.name

        try:
            app = typer.Typer()
            app.command()(validate_cmd)

            result = cli_runner.invoke(app, [tmp_path, "--strict"])

            assert result.exit_code == 1
            assert "Warnings" in result.stdout or "warnings" in result.stdout
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestEvalCommand:
    """Tests for eval command."""

    @patch("semeval.core.runner.TaskRunner")
    @patch("semeval.core.encoders.SentenceTransformerEncoder")
    def test_eval_basic_execution(
        self, mock_encoder_class, mock_runner_class, cli_runner, valid_test_data_file
    ):
        """Test eval command basic execution."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.get_embedding_dim.return_value = 768
        mock_encoder_class.return_value = mock_encoder

        # Mock runner and result
        mock_result = Mock()
        mock_result.get_summary.return_value = {
            "total_runtime": 1.5,
            "semantic_similarity": {"accuracy": 0.95},
        }
        mock_result.save = Mock()

        mock_runner = Mock()
        mock_runner.run.return_value = mock_result
        mock_runner_class.return_value = mock_runner

        app = typer.Typer()
        app.command()(eval_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = cli_runner.invoke(
                app,
                [
                    "--model",
                    "test-model",
                    "--data",
                    valid_test_data_file,
                    "--output",
                    tmpdir,
                ],
            )

        assert result.exit_code == 0
        assert mock_encoder_class.called
        assert mock_runner.run.called
        assert "Evaluation complete" in result.stdout or "complete" in result.stdout.lower()

    @patch("semeval.core.runner.TaskRunner")
    @patch("semeval.core.encoders.HuggingFaceEncoder")
    def test_eval_with_huggingface_encoder(
        self, mock_encoder_class, mock_runner_class, cli_runner, valid_test_data_file
    ):
        """Test eval command with HuggingFace encoder."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.get_embedding_dim.return_value = 512
        mock_encoder_class.return_value = mock_encoder

        # Mock runner and result
        mock_result = Mock()
        mock_result.get_summary.return_value = {"total_runtime": 1.0}
        mock_result.save = Mock()

        mock_runner = Mock()
        mock_runner.run.return_value = mock_result
        mock_runner_class.return_value = mock_runner

        app = typer.Typer()
        app.command()(eval_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = cli_runner.invoke(
                app,
                [
                    "--model",
                    "test-model",
                    "--data",
                    valid_test_data_file,
                    "--output",
                    tmpdir,
                    "--encoder",
                    "huggingface",
                ],
            )

        assert result.exit_code == 0
        assert mock_encoder_class.called

    @patch("semeval.core.runner.TaskRunner")
    @patch("semeval.core.encoders.SentenceTransformerEncoder")
    def test_eval_with_verbose_flag(
        self, mock_encoder_class, mock_runner_class, cli_runner, valid_test_data_file
    ):
        """Test eval command with verbose flag."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.get_embedding_dim.return_value = 768
        mock_encoder_class.return_value = mock_encoder

        # Mock runner
        mock_result = Mock()
        mock_result.get_summary.return_value = {"total_runtime": 1.0}
        mock_result.save = Mock()

        mock_runner = Mock()
        mock_runner.run.return_value = mock_result
        mock_runner_class.return_value = mock_runner

        app = typer.Typer()
        app.command()(eval_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = cli_runner.invoke(
                app,
                [
                    "--model",
                    "test-model",
                    "--data",
                    valid_test_data_file,
                    "--output",
                    tmpdir,
                    "--verbose",
                ],
            )

        # Verbose mode should be passed to runner
        assert mock_runner_class.call_args[1]["verbose"] is True

    @patch("semeval.core.encoders.SentenceTransformerEncoder")
    def test_eval_with_invalid_encoder_type(
        self, mock_encoder_class, cli_runner, valid_test_data_file
    ):
        """Test eval command with invalid encoder type."""
        app = typer.Typer()
        app.command()(eval_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = cli_runner.invoke(
                app,
                [
                    "--model",
                    "test-model",
                    "--data",
                    valid_test_data_file,
                    "--output",
                    tmpdir,
                    "--encoder",
                    "invalid-type",
                ],
            )

        assert result.exit_code == 1
        assert "Unknown encoder" in result.stdout or "error" in result.stdout.lower()

    def test_eval_with_config_file(self, cli_runner, valid_test_data_file):
        """Test eval command with config file (should show not implemented)."""
        config_data = {"model": "test-model", "device": "cpu"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            json.dump(config_data, config_file)
            config_path = config_file.name

        try:
            app = typer.Typer()
            app.command()(eval_cmd)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = cli_runner.invoke(
                    app,
                    [
                        "--model",
                        "test-model",
                        "--data",
                        valid_test_data_file,
                        "--output",
                        tmpdir,
                        "--config",
                        config_path,
                    ],
                )

            assert result.exit_code == 1
            assert "not yet implemented" in result.stdout.lower()
        finally:
            Path(config_path).unlink(missing_ok=True)

    @patch("semeval.core.runner.TaskRunner")
    @patch("semeval.core.encoders.SentenceTransformerEncoder")
    def test_eval_creates_output_directory(
        self, mock_encoder_class, mock_runner_class, cli_runner, valid_test_data_file
    ):
        """Test that eval command creates output directory."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.get_embedding_dim.return_value = 768
        mock_encoder_class.return_value = mock_encoder

        # Mock runner
        mock_result = Mock()
        mock_result.get_summary.return_value = {"total_runtime": 1.0}
        mock_result.save = Mock()

        mock_runner = Mock()
        mock_runner.run.return_value = mock_result
        mock_runner_class.return_value = mock_runner

        app = typer.Typer()
        app.command()(eval_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_output_dir"

            result = cli_runner.invoke(
                app,
                [
                    "--model",
                    "test-model",
                    "--data",
                    valid_test_data_file,
                    "--output",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert output_dir.exists()


class TestCLIUtilityFunctions:
    """Tests for CLI utility functions."""

    def test_console_import(self):
        """Test that console can be imported."""
        from semeval.cli.utils.output import console

        assert console is not None

    def test_output_functions_exist(self):
        """Test that output functions exist."""
        from semeval.cli.utils.output import error, success, warning

        assert callable(error)
        assert callable(success)
        assert callable(warning)

    def test_print_banner_exists(self):
        """Test that print_banner function exists."""
        from semeval.cli.utils.output import print_banner

        assert callable(print_banner)

    def test_create_progress_exists(self):
        """Test that create_progress function exists."""
        from semeval.cli.utils.output import create_progress

        assert callable(create_progress)
