"""
EasyBench: A simple and easy-to-use Python benchmarking library.

This package provides tools for benchmarking Python code with an easy-to-use interface.
"""

from .core import BenchConfig, EasyBench, fixture
from .decorator import BenchDecoParams, bench

__version__ = "0.1.0"

__all__ = [
    "BenchConfig",
    "BenchDecoParams",
    "EasyBench",
    "bench",
    "fixture",
]
