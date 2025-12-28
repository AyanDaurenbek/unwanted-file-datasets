"""Utilities for building real and synthetic datasets of file features."""

__all__ = [
    "SyntheticDatasetGenerator",
    "DatasetBuilder",
    "RealDataCollector",
]

from .synthetic import SyntheticDatasetGenerator
from .builder import DatasetBuilder
from .collect import RealDataCollector

__version__ = "0.1.0"
