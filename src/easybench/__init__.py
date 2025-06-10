"""
EasyBench: A simple and easy-to-use Python benchmarking library.

This package provides tools for benchmarking Python code with an easy-to-use interface.
"""

from .core import BenchConfig, BenchParams, EasyBench, fixture, parametrize
from .decorator import bench

__version__ = "0.1.18"

__all__ = [
    "BenchConfig",
    "BenchParams",
    "EasyBench",
    "bench",
    "fixture",
    "parametrize",
]
