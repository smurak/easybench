"""
A collection of pytest fixtures and testing utilities for easybench.

This module provides common fixtures and helper functions
used across multiple test files:
- Parsing benchmark output for verification
- Allocating memory for testing memory usage measurements
- Stripping ANSI color codes for output analysis
"""

import re
from collections.abc import Callable
from typing import Any

import pytest


def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI color codes from text.

    This function removes ANSI escape sequences used for terminal coloring
    to make the text easier to parse and compare in tests.

    Args:
        text: Text that may contain ANSI color codes

    Returns:
        Text with all ANSI color codes removed

    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def _prepare_lines(output: str) -> tuple[list[str], list[str]]:
    """Prepare output lines by splitting and stripping ANSI codes."""
    original_lines: list[str] = output.strip().split("\n")
    clean_lines: list[str] = [strip_ansi_codes(line) for line in original_lines]
    return original_lines, clean_lines


def _parse_header(line: str) -> tuple[int, bool]:
    """Parse the header to get trial count."""
    header_match = re.search(
        r"Benchmark Results \((\d+) trial(?:s)?\)",
        line,
    )
    if not header_match:
        return 0, False
    trials: int = int(header_match.group(1))
    return trials, True


def _find_header_idx(clean_lines: list[str]) -> int:
    """Find the table header index."""
    return next(
        (
            i
            for i, line in enumerate(clean_lines)
            if "Function" in line
            and (
                re.search(r"Time \([^\)]+\)", line)
                or re.search(r"Avg Time \([^\)]+\)", line)
            )
        ),
        -1,
    )


def _detect_table_format(header_line: str) -> tuple[bool, bool, str, str]:
    """Determine the table format and extract units."""
    is_single_trial = not re.search(r"Avg Time \([^\)]+\)", header_line)

    # Extract time unit
    time_match = re.search(r"(?:Avg )?Time \(([^\)]+)\)", header_line)
    time_unit = time_match.group(1) if time_match else "s"

    # Detect memory metrics and extract memory unit
    memory_match = re.search(r"(?:Avg )?Mem(?:ory)? \(([^\)]+)\)", header_line)
    has_memory_metrics = bool(memory_match)
    memory_unit = memory_match.group(1) if memory_match else "KB"

    return is_single_trial, has_memory_metrics, time_unit, memory_unit


def _find_data_section(clean_lines: list[str], header_idx: int) -> tuple[int, int]:
    """Find the start and end indices of the data section."""
    data_start = header_idx + 2  # Skip the header line and separator
    data_end = next(
        (
            i
            for i in range(data_start, len(clean_lines))
            if not clean_lines[i].strip()
            or "Benchmark Return Values:" in clean_lines[i]
        ),
        len(clean_lines),
    )
    return data_start, data_end


def _find_return_values_idx(clean_lines: list[str]) -> int:
    """Find the index of return values section."""
    return next(
        (i for i, line in enumerate(clean_lines) if "Benchmark Return Values:" in line),
        -1,
    )


def _get_benchmark_columns(*, is_single_trial: bool, has_memory: bool) -> list[str]:
    """Determine column names based on benchmark format."""
    if is_single_trial:
        columns = ["function", "time"]
        if has_memory:
            columns.append("memory")
    else:
        columns = ["function", "avg", "min", "max"]
        if has_memory:
            columns.extend(["avg_memory", "max_memory"])
    return columns


def _initialize_color_info(columns: list[str]) -> dict[str, dict[str, str]]:
    """Initialize the color information dictionary."""
    color_info: dict[str, dict[str, str]] = {}
    for col in columns:
        if col != "function":  # Function name column doesn't have color info
            color_info[col] = {}
    return color_info


def _extract_metrics(values: list[str], columns: list[str]) -> dict[str, float]:
    """Extract numerical metrics from values based on column names."""
    metrics = {}
    for j, col_name in enumerate(columns[1:]):  # Skip 'function' column
        if j < len(values):
            try:
                metrics[col_name] = float(values[j])
            except ValueError:
                continue
    return metrics


def _extract_color_info(
    orig_line: str,
    data: dict[str, Any],
    color_info: dict[str, dict[str, str]],
) -> None:
    """Extract color information from original line with ANSI codes."""
    values = data["values"]
    columns = data["columns"]
    metrics = data["metrics"]
    func_name = data["func_name"]

    for j, col_name in enumerate(columns[1:]):
        if j >= len(values) or col_name not in metrics:
            continue

        # Get the value as a string - strip spaces to handle right-justified values
        value_str = values[j].strip()
        value_pattern = re.escape(value_str)

        # Detect green code (appearing before the value, with potential spaces)
        green_match = re.search(
            r"\033\[32m\s*" + value_pattern,
            orig_line,
        ) or re.search(
            r"\x1b\[32m\s*" + value_pattern,
            orig_line,
        )
        if green_match:
            if col_name not in color_info:
                color_info[col_name] = {}
            color_info[col_name]["green"] = func_name

        # Detect red code (appearing before the value, with potential spaces)
        red_match = re.search(
            r"\033\[31m\s*" + value_pattern,
            orig_line,
        ) or re.search(
            r"\x1b\[31m\s*" + value_pattern,
            orig_line,
        )
        if red_match:
            if col_name not in color_info:
                color_info[col_name] = {}
            color_info[col_name]["red"] = func_name


def _parse_functions_data(
    original_lines: list[str],
    clean_lines: list[str],
    *,
    is_single_trial: bool,
    has_memory: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Parse function benchmark data and color information."""
    functions: dict[str, dict[str, float]] = {}

    if not clean_lines:
        return functions, {}

    # Define columns and initialize color info
    columns = _get_benchmark_columns(
        is_single_trial=is_single_trial,
        has_memory=has_memory,
    )
    color_info = _initialize_color_info(columns)

    # Process each function row
    for i, clean_line in enumerate(clean_lines):
        if not clean_line.strip() or "-" in clean_line[:5]:
            continue

        orig_line = original_lines[i] if i < len(original_lines) else clean_line

        # Extract function name and values
        # Split on spaces of length 2 or more to handle right-justified columns
        parts = re.split(r"\s{2,}", clean_line.strip())
        min_parts_required = 2
        if len(parts) < min_parts_required:
            continue

        func_name = parts[0]
        values = [part.strip() for part in parts[1:]]  # Strip spaces from each part

        # Parse metrics and extract color information
        metrics = _extract_metrics(values, columns)
        functions[func_name] = metrics

        # Create data dictionary for color info extraction
        data = {
            "values": values,
            "columns": columns,
            "metrics": metrics,
            "func_name": func_name,
        }
        _extract_color_info(orig_line, data, color_info)

    # Clean up empty color information
    cleaned_colors = {col: colors for col, colors in color_info.items() if colors}
    return functions, cleaned_colors


def _parse_return_values(lines: list[str]) -> dict[str, str | list[str]]:
    """Parse function return value information."""
    return_values: dict[str, str | list[str]] = {}
    current_func: str | None = None
    trial_values: list[str] = []

    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        # Function name line pattern (e.g., "append_list:")
        if ":" in line and not line.startswith(" ") and not line.startswith("Trial"):
            # Save the data of the previous function
            if current_func and trial_values:
                return_values[current_func] = trial_values.copy()

            # Extract the new function name
            if line.endswith(":"):
                # Format: "function_name:"  # noqa: ERA001
                current_func = line[:-1].strip()
                trial_values = []
            else:
                # Format: "function_name: value"  # noqa: ERA001
                parts = line.split(":", 1)
                func_name = parts[0].strip()
                value = parts[1].strip()

                # If there is a value (single return value)
                if value:
                    return_values[func_name] = value
                    current_func = None
                    trial_values = []
                else:
                    # No value, but multiple lines of values will follow
                    current_func = func_name
                    trial_values = []

        # Trial line pattern (e.g., "  Trial 1: 101")
        elif current_func and line.startswith(("Trial", "  Trial")):
            trial_match = re.match(r"(?:\s*Trial\s+\d+:?\s*)(.*)", line)
            if trial_match:
                trial_value = trial_match.group(1).strip()
                trial_values.append(trial_value)

    if current_func and trial_values:
        return_values[current_func] = trial_values.copy()

    return return_values


@pytest.fixture
def parse_benchmark_output() -> Callable[[str], dict[str, Any]]:
    """
    Parse the output from EasyBench's _display_results method.

    Returns:
        A function that takes a string output and returns a dictionary
        with parsed benchmark results

    """

    def _parse(output: str) -> dict[str, Any]:
        """Parse benchmark output and return a structured dictionary."""
        result: dict[str, Any] = {}

        # Prepare lines
        original_lines, clean_lines = _prepare_lines(output)
        if not clean_lines:
            return result

        # Parse header to get trial count
        trials, valid_header = _parse_header(clean_lines[0] if clean_lines else "")
        if not valid_header:
            return result

        result["trials"] = trials

        # Find the header row
        header_idx = _find_header_idx(clean_lines)
        if header_idx < 0:
            return result  # Header not found

        # Determine the table format and extract units
        header_line = clean_lines[header_idx]
        is_single_trial, has_memory_metrics, time_unit, memory_unit = (
            _detect_table_format(header_line)
        )
        result["is_single_trial"] = is_single_trial
        result["has_memory_metrics"] = has_memory_metrics
        result["time_unit"] = time_unit
        result["memory_unit"] = memory_unit

        # Identify the start and end of the data section
        data_start, data_end = _find_data_section(clean_lines, header_idx)

        # Parse function data and color information
        functions_data, color_info = _parse_functions_data(
            original_lines[data_start:data_end],
            clean_lines[data_start:data_end],
            is_single_trial=is_single_trial,
            has_memory=has_memory_metrics,
        )

        result["functions"] = functions_data
        if color_info:
            result["color"] = color_info

        # Parse function return value information
        return_values_idx = _find_return_values_idx(clean_lines)
        result["has_return_values"] = return_values_idx >= 0

        if result["has_return_values"] and return_values_idx + 2 < len(clean_lines):
            return_values = _parse_return_values(
                clean_lines[return_values_idx + 2 :],
            )
            result["return_values"] = return_values

        return result

    return _parse


@pytest.fixture
def allocate_memory() -> Callable[[int], bytearray]:
    """
    Provide a function to allocate memory for testing.

    Returns:
        A function that allocates memory of specified size in kilobytes

    """

    def _allocate_kb(kb_size: int) -> bytearray:
        """
        Allocate memory of a specified size in kilobytes.

        This function reliably allocates and retains memory of a specified size
        in kilobytes.

        Args:
            kb_size: Amount of memory to allocate (in kilobytes)

        Returns:
            An object (bytearray) holding the allocated memory

        """
        # 1KB = 1024 bytes
        bytes_size = kb_size * 1024

        # Use bytearray to allocate memory of the specified size
        # bytearray actually allocates memory internally, ensuring accurate measurement
        memory = bytearray(bytes_size)

        # Modify some values to prevent memory from being released by optimization
        step_size = 4096  # Page size, extracted as a constant
        for i in range(0, bytes_size, step_size):
            if i < bytes_size:
                memory[i] = 1

        return memory

    return _allocate_kb
