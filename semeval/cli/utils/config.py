"""Configuration utilities for CLI - thin wrapper over existing config system."""

from pathlib import Path
from typing import Any, Dict

from semeval.core.config import SemEvalSettings, load_settings
from semeval.core.loaders import JSONDataLoader


def load_cli_config(config_path: Path) -> SemEvalSettings:
    """Load SemEval configuration from file.

    Uses the existing SemEvalSettings system.

    Parameters
    ----------
    config_path : Path
        Path to configuration file (YAML or JSON)

    Returns
    -------
    SemEvalSettings
        Loaded configuration
    """
    # Use the load_settings helper which handles config path properly
    return load_settings(config_path=str(config_path))


def load_test_data(data_path: Path) -> Dict[str, Any]:
    """Load test data using existing JSONDataLoader.

    Parameters
    ----------
    data_path : Path
        Path to test data JSON file

    Returns
    -------
    dict
        Loaded and validated test data
    """
    loader = JSONDataLoader()
    return loader.load(str(data_path))
