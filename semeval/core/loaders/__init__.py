"""Data loader implementations.

This package provides data loaders for various file formats.

Available Loaders
-----------------
JSONDataLoader
    Load test data from JSON files

Examples
--------
>>> from semeval.core.loaders import JSONDataLoader
>>> loader = JSONDataLoader()
>>> test_data = loader.load("data/test_data.json")
"""

from .json_loader import JSONDataLoader

__all__ = ['JSONDataLoader']
