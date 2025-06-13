"""Collections of utility functions for easybench."""

import os
import platform
import sys
import time
import unicodedata
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


def visual_width(text: str) -> int:
    """
    Calculate the visual width of a string.

    Assuming:
        - Fullwidth ('F') and Wide ('W') characters count as 2.
        - All other characters count as 1.

    This function is useful for determining display width, especially for
    East Asian characters in fixed-width terminal environments.

    Args:
        text (str): The input string.

    Returns:
        int: The total visual width of the string.

    """
    width = 0
    for ch in text:
        ea_width = unicodedata.east_asian_width(ch)
        if ea_width in ("F", "W"):  # Fullwidth or Wide
            width += 2
        else:
            width += 1  # Narrow, Halfwidth, Ambiguous, Neutral
    return width


def visual_ljust(text: str, width: int, fillchar: str = " ") -> str:
    """
    Ljust for visual width.

    Args:
        text (str): The input string to pad.
        width (int): Target visual width.
        fillchar (str, optional): Character used for padding. Defaults to a space.

    Returns:
        str: Padded string with the specified visual width.

    """
    current_width = visual_width(text)
    padding_width = max(0, width - current_width)
    return text + (fillchar * padding_width)


def visual_rjust(text: str, width: int, fillchar: str = " ") -> str:
    """
    Rjust for visual width.

    Args:
        text (str): The input string to pad.
        width (int): Target visual width.
        fillchar (str, optional): Character used for padding. Defaults to a space.

    Returns:
        str: Right-padded string with the specified visual width.

    """
    current_width = visual_width(text)
    padding_width = max(0, width - current_width)
    return (fillchar * padding_width) + text
