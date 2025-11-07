"""YAML configuration source for pydantic-settings.

This module provides a custom settings source that reads configuration
from YAML files, enabling hierarchical configuration management.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that reads from YAML files.

    This source allows pydantic-settings to load configuration from
    YAML files, supporting nested configuration structures.

    Parameters
    ----------
    settings_cls : Type[BaseSettings]
        The settings class being configured
    yaml_file : str or Path, optional
        Path to YAML configuration file. If not provided or file doesn't exist,
        returns empty configuration.

    Examples
    --------
    >>> class MySettings(BaseSettings):
    ...     model_config = SettingsConfigDict(...)
    ...
    ...     @classmethod
    ...     def settings_customise_sources(cls, ...):
    ...         return (
    ...             YamlConfigSettingsSource(settings_cls, yaml_file="config.yaml"),
    ...             ...
    ...         )
    """

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        yaml_file: str | Path | None = None
    ):
        """Initialize YAML config source.

        Parameters
        ----------
        settings_cls : Type[BaseSettings]
            Settings class
        yaml_file : str or Path, optional
            Path to YAML file
        """
        super().__init__(settings_cls)
        self.yaml_file = Path(yaml_file) if yaml_file else None
        self._config_data: Dict[str, Any] = {}

        if self.yaml_file and self.yaml_file.exists():
            self._load_yaml()

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except Exception as e:
            # Fail silently, let defaults handle it
            self._config_data = {}

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str
    ) -> Tuple[Any, str, bool]:
        """Get value for a specific field from YAML config.

        Parameters
        ----------
        field : FieldInfo
            Field information
        field_name : str
            Name of the field

        Returns
        -------
        tuple
            (value, field_key, value_is_complex)
        """
        # Support nested field access with dot notation
        value = self._config_data
        for key in field_name.split('.'):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break

        return value, field_name, False

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool
    ) -> Any:
        """Prepare field value for validation.

        Parameters
        ----------
        field_name : str
            Field name
        field : FieldInfo
            Field information
        value : Any
            Raw value
        value_is_complex : bool
            Whether value is complex

        Returns
        -------
        Any
            Prepared value
        """
        return value

    def __call__(self) -> Dict[str, Any]:
        """Return all configuration data.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return self._config_data
