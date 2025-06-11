"""Collections of utility functions for easybench."""

import os
import platform
import sys
import time
from typing import Any


def measure_timer_overhead(iterations: int = 1_000_000) -> float:
    """Measure the overhead of the time.perf_counter() function."""
    # Variable to prevent loop optimization
    counter = 0

    # 1. Measure loop overhead only
    start_empty = time.perf_counter()
    for _ in range(iterations):
        counter += 1
    end_empty = time.perf_counter()
    loop_overhead = end_empty - start_empty

    # 2. Measure loop with timer calls
    start_with_timer = time.perf_counter()
    for _ in range(iterations):
        counter += 1
        time.perf_counter()
    end_with_timer = time.perf_counter()
    total_time = end_with_timer - start_with_timer

    # 3. Calculate timer function overhead
    # Use `counter` instead of `iterations` in the calculation
    # to ensure loop and variable are not optimized away in any interpreter
    return (total_time - loop_overhead) / (counter / 2)


def get_bench_env() -> dict[str, Any]:
    """
    Collect environment information relevant to benchmarking.

    Returns:
        Dictionary containing benchmark-relevant environment details

    """
    env_info: dict[str, Any] = {}

    # OS information (affects performance)
    env_info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
    }

    # CPU information (major performance factor)
    env_info["cpu"] = {
        "count": os.cpu_count() or "Unknown",
        "processor": platform.processor(),
    }

    # Python runtime environment (affects execution speed)
    env_info["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
    }

    # Add Pyodide info if applicable
    if "pyodide" in sys.modules:
        pyodide = sys.modules["pyodide"]
        env_info["python"]["environment"] = "pyodide"
        env_info["python"]["pyodide_version"] = getattr(
            pyodide,
            "__version__",
            "unknown",
        )

    # Performance counter information (affects measurement precision)
    perf_info = time.get_clock_info("perf_counter")
    env_info["perf_counter"] = {
        "resolution": perf_info.resolution,
        "implementation": perf_info.implementation,
        "monotonic": perf_info.monotonic,
        "timer_overhead": measure_timer_overhead(),
    }

    return env_info
