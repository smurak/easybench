"""
EasyBench: A simple and easy-to-use Python benchmarking library.

This package provides tools for benchmarking Python code with an easy-to-use interface.
"""

from .core import BenchConfig, BenchParams, EasyBench, fixture, parametrize
from .decorator import bench
from .utils import get_bench_env

__version__ = "0.1.21"

__all__ = [
    "BenchConfig",
    "BenchParams",
    "EasyBench",
    "bench",
    "fixture",
    "get_bench_env",
    "parametrize",
]
