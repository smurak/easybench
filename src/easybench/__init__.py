"""
EasyBench: A simple and easy-to-use Python benchmarking library.

This package provides tools for benchmarking Python code with an easy-to-use interface.
"""

from .core import (
    BenchConfig,
    BenchParams,
    EasyBench,
    customize,
    fixture,
    parametrize,
    set_reporter,
)
from .decorator import bench
from .utils import get_bench_env

# Register IPython magic if in IPython/Jupyter environment
try:
    from IPython import get_ipython

    from .notebook import load_ipython_extension

    ipython = get_ipython()
    if ipython is not None:
        load_ipython_extension(ipython)
except (NameError, ImportError):
    # Not in IPython environment
    pass

__version__ = "0.1.36"

__all__ = [
    "BenchConfig",
    "BenchParams",
    "EasyBench",
    "bench",
    "customize",
    "fixture",
    "get_bench_env",
    "parametrize",
    "set_reporter",
]
