"""Configuration management for SemEval framework.

This module provides configuration classes using pydantic-settings,
supporting configuration from YAML files, environment variables, and defaults.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .yaml_source import YamlConfigSettingsSource

# Default config path (can be overridden with CONFIG_PATH env var)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class ModelConfig(BaseSettings):
    """Model configuration settings.

    Attributes
    ----------
    name : str
        Model identifier or path
    device : str
        Device to use ('auto', 'cpu', 'cuda', 'mps')
    batch_size : int
        Batch size for encoding
    """

    name: str = Field(
        default="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        description="Model name or path",
    )
    device: str = Field(
        default="auto", description="Device for computation: auto, cpu, cuda, mps"
    )
    batch_size: int = Field(
        default=32, gt=0, le=512, description="Batch size for encoding"
    )


class OutputConfig(BaseSettings):
    """Output configuration settings.

    Attributes
    ----------
    base_dir : str
        Base directory for outputs
    export_formats : list of str
        Export formats to generate
    save_comprehensive_report : bool
        Whether to generate comprehensive report
    """

    base_dir: str = Field(default="./output", description="Base output directory")
    export_formats: List[str] = Field(
        default=["json", "csv", "markdown"],
        description="Export formats: json, csv, markdown",
    )
    save_comprehensive_report: bool = Field(
        default=True, description="Generate comprehensive report"
    )


class InformationRetrievalConfig(BaseSettings):
    """Information Retrieval task configuration.

    Attributes
    ----------
    enabled : bool
        Whether task is enabled
    ndcg_at_k : list of int
        K values for NDCG@K metric
    map_at_k : list of int
        K values for MAP@K metric
    mrr_at_k : list of int
        K values for MRR@K metric
    """

    enabled: bool = Field(default=True, description="Enable this task")
    ndcg_at_k: List[int] = Field(
        default=[1, 3, 5, 10], description="K values for NDCG@K"
    )
    map_at_k: List[int] = Field(default=[10], description="K values for MAP@K")
    mrr_at_k: List[int] = Field(default=[10], description="K values for MRR@K")


class SemanticSimilarityConfig(BaseSettings):
    """Semantic Similarity task configuration.

    Attributes
    ----------
    enabled : bool
        Whether task is enabled
    margin_threshold : float
        Minimum margin for successful triplet
    """

    enabled: bool = Field(default=True, description="Enable this task")
    margin_threshold: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Minimum margin threshold"
    )


class LinguisticRobustnessConfig(BaseSettings):
    """Linguistic Robustness task configuration.

    Attributes
    ----------
    enabled : bool
        Whether task is enabled
    morphology_threshold : float
        Similarity threshold for morphology (high = good)
    typo_threshold : float
        Similarity threshold for typos (high = good)
    negation_threshold : float
        Similarity threshold for negation (low = good)
    """

    enabled: bool = Field(default=True, description="Enable this task")
    morphology_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Morphology similarity threshold"
    )
    typo_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Typo similarity threshold"
    )
    negation_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Negation similarity threshold (lower is better)",
    )


class VectorArithmeticConfig(BaseSettings):
    """Vector Arithmetic task configuration.

    Attributes
    ----------
    enabled : bool
        Whether task is enabled
    top_k : list of int
        K values for Top-K accuracy
    """

    enabled: bool = Field(default=True, description="Enable this task")
    top_k: List[int] = Field(
        default=[1, 5, 10], description="K values for Top-K accuracy"
    )


class TasksConfig(BaseSettings):
    """All task configurations.

    Attributes
    ----------
    information_retrieval : InformationRetrievalConfig
        IR task config
    semantic_similarity : SemanticSimilarityConfig
        Semantic similarity task config
    linguistic_robustness : LinguisticRobustnessConfig
        Linguistic robustness task config
    vector_arithmetic : VectorArithmeticConfig
        Vector arithmetic task config
    """

    information_retrieval: InformationRetrievalConfig = Field(
        default_factory=InformationRetrievalConfig
    )
    semantic_similarity: SemanticSimilarityConfig = Field(
        default_factory=SemanticSimilarityConfig
    )
    linguistic_robustness: LinguisticRobustnessConfig = Field(
        default_factory=LinguisticRobustnessConfig
    )
    vector_arithmetic: VectorArithmeticConfig = Field(
        default_factory=VectorArithmeticConfig
    )


class LoggingConfig(BaseSettings):
    """Logging configuration.

    Attributes
    ----------
    verbose : bool
        Enable verbose logging
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : str
        Directory for log files
    """

    verbose: bool = Field(default=False, description="Verbose mode")
    level: str = Field(default="INFO", description="Log level")
    log_dir: str = Field(default="logs", description="Log directory")

    def get_log_level_int(self) -> int:
        """Convert log level string to int.

        Returns
        -------
        int
            Logging level constant
        """
        return getattr(logging, self.level.upper(), logging.INFO)


class SemEvalSettings(BaseSettings):
    """Main SemEval framework settings.

    This class manages all configuration for the SemEval framework,
    loading settings from YAML files, environment variables, and defaults
    in that priority order.

    Configuration Priority:
    1. Environment variables (highest)
    2. .env file
    3. YAML configuration file
    4. Default values (lowest)

    Attributes
    ----------
    app_name : str
        Application name
    app_version : str
        Application version
    app_description : str
        Application description
    model : ModelConfig
        Model configuration
    output : OutputConfig
        Output configuration
    tasks : TasksConfig
        Task configurations
    logging : LoggingConfig
        Logging configuration

    Examples
    --------
    >>> # Load from default config.yaml
    >>> settings = SemEvalSettings()
    >>>
    >>> # Load from custom config
    >>> import os
    >>> os.environ['CONFIG_PATH'] = 'config.prod.yaml'
    >>> settings = SemEvalSettings()
    >>>
    >>> # Override with env variable
    >>> os.environ['SEMEVAL_MODEL__NAME'] = 'custom-model'
    >>> settings = SemEvalSettings()
    >>> print(settings.model.name)
    custom-model
    """

    # Application metadata
    app_name: str = Field(default="SemEval", description="Application name")
    app_version: str = Field(default="0.1.0", description="Version")
    app_description: str = Field(
        default="Semantic Evaluation Framework for Turkish NLP",
        description="Description",
    )

    # Configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SEMEVAL_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Customize configuration sources priority order.

        Priority (highest to lowest):
        1. Environment variables
        2. .env file
        3. YAML configuration
        4. Default values

        Parameters
        ----------
        settings_cls : Type[BaseSettings]
            Settings class
        init_settings : PydanticBaseSettingsSource
            Init settings source
        env_settings : PydanticBaseSettingsSource
            Environment variable source
        dotenv_settings : PydanticBaseSettingsSource
            .env file source
        file_secret_settings : PydanticBaseSettingsSource
            File secrets source

        Returns
        -------
        tuple
            Ordered sources
        """
        yaml_path = os.getenv("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
            init_settings,
            file_secret_settings,
        )


def load_settings(
    config_path: Optional[str] = None, env: Optional[str] = None
) -> SemEvalSettings:
    """Load SemEval settings from configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML config file. If not provided, uses CONFIG_PATH
        environment variable or default path.
    env : str, optional
        Environment name ('dev', 'prod'). If provided, loads config.{env}.yaml.
        Takes precedence over config_path.

    Returns
    -------
    SemEvalSettings
        Loaded settings

    Examples
    --------
    >>> settings = load_settings()  # Loads config.yaml
    >>> settings = load_settings(env="dev")  # Loads config.dev.yaml
    >>> settings = load_settings(env="prod")  # Loads config.prod.yaml
    >>> settings = load_settings("custom_config.yaml")
    """
    if env:
        # Environment-specific config takes precedence
        config_path = f"config.{env}.yaml"

    if config_path:
        os.environ["CONFIG_PATH"] = config_path

    return SemEvalSettings()
