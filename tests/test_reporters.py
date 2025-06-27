"""
Tests for the reporter classes and formatters.

This module tests the various reporter classes and formatters, including:
- TableFormatter
- CSVFormatter
- JSONFormatter
- DataFrameFormatter
- Reporter
- StreamReporter
- ConsoleReporter
- FileReporter
- CallbackReporter
- Helper functions for creating reporters
"""

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar, get_args
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from easybench.core import BenchConfig, ResultType, StatType, get_reporter
from easybench.reporters import (
    CallbackReporter,
    ConsoleReporter,
    CSVFormatter,
    DataFrameFormatter,
    FileReporter,
    Formatted,
    Formatter,
    JSONFormatter,
    MemoryUnit,
    MetricType,
    Reporter,
    SimpleConsoleReporter,
    SimpleFormatter,
    SimpleStreamReporter,
    StreamReporter,
    TableFormatter,
)
from easybench.utils import visual_width
from easybench.visualization import PlotReporter

# Constants for test values to avoid magic numbers
TEST_TIME_VALUE = 0.1
TEST_SLOW_TIME = 0.2
TEST_SLOWER_TIME = 0.3
TEST_AVG_TIME = 0.2
TEST_MEMORY_VALUE = 1.0
TEST_AVG_MEMORY = 2.0
TEST_MAX_MEMORY = 3.0
TEST_METRIC_MIN = 0.05
TEST_METRIC_MAX = 0.15
TEST_FLOAT_VALUE = 0.1
EXPECTED_ROW_COUNT = 2

# Constants for comparison
MIN_CSV_ROWS = 2  # Header + data row
MIN_VALUE_COLUMNS = 2  # Function name + value
MIN_STAT_COLUMNS = 4  # Function name + avg + min + max
FLOAT_TOLERANCE = 0.001  # Tolerance for float comparisons

# Type alias for import function return type
ImportReturnT = TypeVar("ImportReturnT")

# Type alias for pandas DataFrame-like objects
DataFrameLike = dict[str, list[str | float]] | Mapping[str, Sequence[str | float | int]]


def complete_stat(
    dic: dict[str, float],
    memory: bool = False,  # noqa: FBT002, FBT001
) -> StatType:
    """Complete dictionaries for StatType."""
    stat: StatType = {
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
    if memory:
        stat.update(
            {
                "avg_memory": 0.0,
                "max_memory": 0.0,
            },
        )

    stat.update(dic)  # type: ignore [literal-required, typeddict-item]
    return stat


class TestTableFormatter:
    """Tests for the TableFormatter class."""

    def test_format_single_trial(self) -> None:
        """Test formatting results for a single trial."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE], "memory": [1024]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {"avg": TEST_TIME_VALUE, "avg_memory": TEST_MEMORY_VALUE},
            ),
        }
        config = BenchConfig(trials=1, memory=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (1 trial)" in output
        assert "Function" in output
        assert "Time (s)" in output
        assert "Memory (KB)" in output
        assert "test_func" in output

    def test_format_multiple_trials(self) -> None:
        """Test formatting results for multiple trials."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                },
            ),
        }
        config = BenchConfig(trials=3)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (3 trials)" in output
        assert "Function" in output
        assert "Avg Time (s)" in output
        assert "Min Time (s)" in output
        assert "Max Time (s)" in output
        assert "test_func" in output

    def test_format_with_memory_metrics(self) -> None:
        """Test formatting results with memory metrics."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (3 trials)" in output
        assert "Function" in output
        assert "Avg Mem (KB)" in output
        assert "Max Mem (KB)" in output
        assert "test_func" in output

    def test_format_with_return_values(self) -> None:
        """Test formatting results with return values."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (1 trial)" in output
        assert "Benchmark Return Values" in output
        assert "test_func: result" in output

    def test_format_with_different_return_values(self) -> None:
        """Test formatting results with different return values across trials."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "output": [1, 2, 3],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                },
            ),
        }
        config = BenchConfig(trials=3, show_output=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (3 trials)" in output
        assert "Benchmark Return Values" in output
        assert "test_func:" in output
        assert "Trial 1: 1" in output
        assert "Trial 2: 2" in output
        assert "Trial 3: 3" in output

    def test_format_metric_no_color(self) -> None:
        """Test formatting metrics without color."""
        formatter = TableFormatter()
        value = TEST_FLOAT_VALUE
        min_value = TEST_METRIC_MIN
        max_value = TEST_METRIC_MAX
        width = 15

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
            width=width,
            color=False,
        )

        assert "0.100000" in result
        assert "\033[" not in result  # No color codes

    def test_format_metric_with_color_min(self) -> None:
        """Test formatting minimum value with color."""
        formatter = TableFormatter()
        value = TEST_METRIC_MIN
        min_value = TEST_METRIC_MIN
        max_value = TEST_METRIC_MAX
        width = 15

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
            width=width,
            color=True,
        )

        assert "0.050000" in result
        assert "\033[32m" in result  # Green color code

    def test_format_metric_with_color_max(self) -> None:
        """Test formatting maximum value with color."""
        formatter = TableFormatter()
        value = TEST_METRIC_MAX
        min_value = TEST_METRIC_MIN
        max_value = TEST_METRIC_MAX
        width = 15

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
            width=width,
            color=True,
        )

        assert "0.150000" in result
        assert "\033[31m" in result  # Red color code

    def test_format_with_memory_unit(self) -> None:
        """Test formatting results with different memory units."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024 * 1024],  # 1 MB in bytes
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_TIME_VALUE,
                    "avg_memory": 1024 * 1024,  # 1 MB in bytes
                    "max_memory": 2 * 1024 * 1024,  # 2 MB in bytes
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=1, memory="MB")  # type: ignore [arg-type]

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (1 trial)" in output
        assert "Memory (MB)" in output
        assert "1.000000" in output  # 1 MB displayed

    def test_format_with_wide_characters(self) -> None:
        """Test formatting with function names containing wide characters."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE]},
            "テスト関数": {"times": [TEST_SLOW_TIME]},  # Japanese characters
            "测试函数": {"times": [TEST_SLOWER_TIME]},  # Chinese characters
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
            "テスト関数": complete_stat({"avg": TEST_SLOW_TIME}),
            "测试函数": complete_stat({"avg": TEST_SLOWER_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Verify output contains all function names
        assert "test_func" in output
        assert "テスト関数" in output
        assert "测试函数" in output

        # Check if function names are properly spaced
        # - each line should start with a function name
        # followed by appropriate spacing before the time value
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        function_lines = [
            line
            for line in lines
            if "test_func" in line or "テスト関数" in line or "测试函数" in line
        ]

        for line in function_lines:
            # Extract function name - it's at the start of the line
            function_name = None
            for name in results:
                if line.startswith(name):
                    function_name = name
                    break

            assert (
                function_name is not None
            ), f"Could not find function name in line: {line}"

            # Verify proper alignment - after the function name there should be spaces
            # followed by the time value
            remainder = line[len(function_name) :].strip()
            assert remainder.startswith("0."), f"Value not properly aligned in: {line}"

    def test_format_alignment_with_mixed_width_characters(self) -> None:
        """Test column alignment with mixed regular and wide characters."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "ascii_only": {"times": [TEST_TIME_VALUE]},
            "漢字_mixed_with_ascii": {"times": [TEST_SLOW_TIME]},
            "all_アジア文字_chars": {"times": [TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            "ascii_only": complete_stat({"avg": TEST_TIME_VALUE}),
            "漢字_mixed_with_ascii": complete_stat({"avg": TEST_SLOW_TIME}),
            "all_アジア文字_chars": complete_stat({"avg": TEST_SLOWER_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Get all lines with function names
        lines = [
            line for line in output.split("\n") if any(name in line for name in results)
        ]

        # All line length are equal
        assert len({visual_width(line) for line in lines}) == 1

    def test_format_visual_width_calculation(self) -> None:
        """Test that visual_width is used correctly for width calculation."""
        # Create function names with different visual widths
        names = {
            "ascii": "test_function",  # 13 chars, visual width 13
            "wide": "テスト関数",  # 5 chars, visual width 10
            "mixed": "test_漢字_func",  # 14 chars, visual width 18
        }

        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            names["ascii"]: {"times": [TEST_TIME_VALUE]},
            names["wide"]: {"times": [TEST_SLOW_TIME]},
            names["mixed"]: {"times": [TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            names["ascii"]: complete_stat({"avg": TEST_TIME_VALUE}),
            names["wide"]: complete_stat({"avg": TEST_SLOW_TIME}),
            names["mixed"]: complete_stat({"avg": TEST_SLOWER_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Verify max_name_len is using visual_width rather than len
        # Max visual width should be 18 (test_漢字_func) not 13 (test_function)
        expected_max_visual_width = max(visual_width(name) for name in names.values())
        assert expected_max_visual_width == 14  # noqa: PLR2004

        # Extract all lines with function names
        lines = [
            line
            for line in output.split("\n")
            if any(name in line for name in names.values())
        ]

        # All line length are equal
        assert len({visual_width(line) for line in lines}) == 1

    def test_precision_parameter(self) -> None:
        """Test the precision parameter for controlling decimal places."""
        # Test with different precision values
        for precision in [2, 3, 4, 8]:
            formatter = TableFormatter(precision=precision)
            results: dict[str, ResultType] = {
                "test_func": {"times": [1.123456789]},
            }
            stats: dict[str, StatType] = {
                "test_func": complete_stat({"avg": 1.123456789}),
            }
            config = BenchConfig(trials=1)

            output = formatter.format(results, stats, config)

            # Check that the formatted number has the correct precision
            expected_format = f"{1.123456789:.{precision}f}"
            assert expected_format in output

    def test_calculate_column_widths_single_trial(self) -> None:
        """Test column width calculation for single trial."""
        formatter = TableFormatter(precision=3)
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": 1.123, "avg_memory": 1024.5}),
        }
        config = BenchConfig(trials=1, memory=True)

        # Use _prepare_formatting_data instead of _calculate_column_widths directly
        formatting_data = formatter._prepare_formatting_data(stats, config)
        widths = formatting_data["column_widths"]

        # Width should be at least the length of the formatted value
        expected_time_width = len("1.123") + 2  # +2 for padding
        expected_memory_width = len("1.000") + 2  # 1024.5 bytes = 1.000 KB

        assert widths["time"] >= expected_time_width
        assert widths["memory"] >= expected_memory_width

    def test_calculate_column_widths_multiple_trials(self) -> None:
        """Test column width calculation for multiple trials."""
        formatter = TableFormatter(precision=2)
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": 12.34,
                    "min": 1.23,
                    "max": 123.45,
                    "avg_memory": 1024,
                    "max_memory": 2048,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=True)

        # Use _prepare_formatting_data instead of _calculate_column_widths directly
        formatting_data = formatter._prepare_formatting_data(stats, config)
        widths = formatting_data["column_widths"]

        # Check that widths accommodate the largest values
        assert "avg_time" in widths
        assert "min_time" in widths
        assert "max_time" in widths
        assert "avg_memory" in widths
        assert "max_memory" in widths

        # Width should be at least the length of the largest formatted value
        max_time_value = max(12.34, 1.23, 123.45)
        expected_min_width = len(f"{max_time_value:.2f}") + 2

        assert widths["avg_time"] >= expected_min_width
        assert widths["min_time"] >= expected_min_width
        assert widths["max_time"] >= expected_min_width

    def test_calculate_column_widths_considers_headers(self) -> None:
        """Test that column width calculation considers header lengths."""
        formatter = TableFormatter(precision=1)
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": 1.0}),  # Short value
        }
        config = BenchConfig(trials=1)

        # Use _prepare_formatting_data instead of _calculate_column_widths directly
        formatting_data = formatter._prepare_formatting_data(stats, config)
        widths = formatting_data["column_widths"]

        # Width should be at least the header length
        header_length = len("Time (s)")
        assert widths["time"] >= header_length + 2  # +2 for padding

    def test_calculate_column_widths_with_different_memory_units(self) -> None:
        """Test column width calculation with different memory units."""
        formatter = TableFormatter(precision=2)
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": 1.0,
                    "avg_memory": 1024 * 1024 * 1024,  # 1 GB in bytes
                    "max_memory": 2 * 1024 * 1024 * 1024,  # 2 GB in bytes
                },
            ),
        }
        config = BenchConfig(trials=1, memory="GB")

        # Use _prepare_formatting_data instead of _calculate_column_widths directly
        formatting_data = formatter._prepare_formatting_data(stats, config)
        widths = formatting_data["column_widths"]

        # When converted to GB, values should be 1.00 and 2.00
        expected_width = max(len("1.00"), len("2.00"), len("Memory (GB)")) + 2
        assert widths["memory"] >= expected_width

    def test_dynamic_width_with_large_values(self) -> None:
        """Test that dynamic width handles large values correctly."""
        formatter = TableFormatter(precision=2)
        results: dict[str, ResultType] = {
            "test_func": {"times": [1234567.89]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": 1234567.89}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Check that the large value is properly formatted and displayed
        expected_format = f"{1234567.89:.2f}"
        assert expected_format in output

    def test_dynamic_width_with_mixed_value_sizes(self) -> None:
        """Test dynamic width with mixed small and large values."""
        formatter = TableFormatter(precision=3)
        results: dict[str, ResultType] = {
            "small_func": {"times": [0.001]},
            "large_func": {"times": [1000.123]},
        }
        stats: dict[str, StatType] = {
            "small_func": complete_stat({"avg": 0.001}),
            "large_func": complete_stat({"avg": 1000.123}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Both values should be properly formatted
        assert "0.001" in output
        assert "1000.123" in output

        # Check that columns are properly aligned
        lines = [line for line in output.split("\n") if "func" in line]
        assert len(lines) == len(results)

        # All data lines should have the same visual width
        assert len({visual_width(line) for line in lines}) == 1


class TestCSVFormatter:
    """Tests for the CSVFormatter class."""

    def test_format_single_trial(self) -> None:
        """Test formatting single trial results as CSV."""
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        # Parse the CSV output
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)

        assert len(rows) == EXPECTED_ROW_COUNT  # Header + 1 data row
        assert rows[0] == ["Function", "Time (s)"]
        assert rows[1][0] == "test_func"
        assert float(rows[1][1]) == TEST_TIME_VALUE

    def test_format_multiple_trials(self) -> None:
        """Test formatting multiple trial results as CSV."""
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                },
            ),
        }
        config = BenchConfig(trials=3)

        output = formatter.format(results, stats, config)

        # Parse the CSV output
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)

        assert len(rows) == EXPECTED_ROW_COUNT  # Header + 1 data row
        assert rows[0] == ["Function", "Avg Time (s)", "Min Time (s)", "Max Time (s)"]
        assert rows[1][0] == "test_func"
        assert float(rows[1][1]) == TEST_AVG_TIME
        assert float(rows[1][2]) == TEST_TIME_VALUE
        assert float(rows[1][3]) == TEST_SLOWER_TIME

    def test_format_with_memory(self) -> None:
        """Test formatting results with memory metrics as CSV."""
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE], "memory": [1024]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_TIME_VALUE,
                    "avg_memory": TEST_MEMORY_VALUE * 1024,
                },
            ),
        }
        config = BenchConfig(trials=1, memory=True)

        output = formatter.format(results, stats, config)

        # Parse the CSV output
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)

        assert len(rows) == EXPECTED_ROW_COUNT  # Header + 1 data row
        assert rows[0] == ["Function", "Time (s)", "Memory (KB)"]
        assert rows[1][0] == "test_func"
        assert float(rows[1][1]) == TEST_TIME_VALUE
        assert float(rows[1][2]) == TEST_MEMORY_VALUE

    def test_format_multiple_trials_with_memory(self) -> None:
        """Test formatting multiple trial results with memory as CSV."""
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)

        # Parse the CSV output
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)

        assert len(rows) == EXPECTED_ROW_COUNT  # Header + 1 data row
        assert rows[0] == [
            "Function",
            "Avg Time (s)",
            "Min Time (s)",
            "Max Time (s)",
            "Avg Memory (KB)",
            "Max Memory (KB)",
        ]
        assert rows[1][0] == "test_func"
        assert float(rows[1][1]) == TEST_AVG_TIME
        assert float(rows[1][4]) == TEST_AVG_MEMORY
        assert float(rows[1][5]) == TEST_MAX_MEMORY

    def test_format_with_memory_unit(self) -> None:
        """Test formatting results with different memory units."""
        unit = MemoryUnit("MB")
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024 * 1024],  # 1 MB in bytes
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_TIME_VALUE,
                    "avg_memory": 1024 * 1024,  # 1 MB in bytes
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=1, memory=unit)

        output = formatter.format(results, stats, config)

        reader = csv.reader(io.StringIO(output))
        rows = list(reader)

        assert rows[0] == ["Function", "Time (s)", "Memory (MB)"]
        assert float(rows[1][2]) == 1.0  # 1 MB


class TestJSONFormatter:
    """Tests for the JSONFormatter class."""

    def test_format_basic(self) -> None:
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert "config" in data
        assert "stats" in data
        assert "results" in data
        assert data["config"]["trials"] == 1
        assert "test_func" in data["stats"]
        assert data["stats"]["test_func"]["avg"] == TEST_TIME_VALUE
        assert data["results"] == results

    def test_format_with_memory(self) -> None:
        """Test JSON formatting with memory metrics."""
        formatter = JSONFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert data["config"]["memory"] is True
        assert data["stats"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["stats"]["test_func"]["max_memory"] == TEST_MAX_MEMORY
        assert data["results"] == results

    def test_format_with_output(self) -> None:
        """Test JSON formatting with function outputs."""
        formatter = JSONFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert data["results"]["test_func"]["output"] == ["result"]
        assert "test_func" in data["stats"]
        assert data["stats"]["test_func"]["avg"] == TEST_TIME_VALUE

    def test_format_with_memory_unit(self) -> None:
        """Test formatting results with different memory units."""
        unit = MemoryUnit("MB")
        formatter = JSONFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024 * 1024],  # 1 MB in bytes
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": TEST_AVG_MEMORY * 1024 * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024 * 1024,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=unit)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert data["config"]["memory"] == "MB"
        assert data["stats"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["stats"]["test_func"]["max_memory"] == TEST_MAX_MEMORY
        assert data["results"] == results


class TestSimpleFormatter:
    """Tests for the SimpleFormatter class."""

    def test_init_with_valid_metric(self) -> None:
        """Test initialization with valid metrics."""
        for metric in get_args(MetricType):
            if metric == "def":
                continue
            formatter = SimpleFormatter(metric=metric)
            assert formatter.metric == metric

    def test_init_with_invalid_metric(self) -> None:
        """Test that 'def' is not a valid metric for SimpleFormatter."""
        with pytest.raises(
            ValueError,
            match="'def' is not a valid metric for Simple formatter",
        ):
            SimpleFormatter(metric="def")  # type: ignore [arg-type]

    def test_format_single_trial(self) -> None:
        """Test formatting with single trial results."""
        metric: MetricType = "avg"
        formatter = SimpleFormatter(metric=metric)
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        assert output.strip() == str(TEST_TIME_VALUE)

    def test_format_multiple_trials_avg(self) -> None:
        """Test formatting multiple trial results with 'avg' metric."""
        formatter = SimpleFormatter(metric="avg")
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                },
            ),
        }
        config = BenchConfig(trials=3)

        output = formatter.format(results, stats, config)
        assert output.strip() == str(TEST_AVG_TIME)

    def test_format_multiple_trials_min(self) -> None:
        """Test formatting multiple trial results with 'min' metric."""
        formatter = SimpleFormatter(metric="min")
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                },
            ),
        }
        config = BenchConfig(trials=3)

        output = formatter.format(results, stats, config)
        assert output.strip() == str(TEST_TIME_VALUE)

    def test_format_with_memory_metrics(self) -> None:
        """Test formatting results with memory metrics."""
        formatter = SimpleFormatter(metric="avg_memory")
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_TIME_VALUE,
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
            ),
        }
        config = BenchConfig(trials=1, memory=True)

        output = formatter.format(results, stats, config)
        assert output.strip() == str(TEST_AVG_MEMORY)

    def test_format_multiple_functions(self) -> None:
        """Test formatting results with multiple functions."""
        formatter = SimpleFormatter(metric="avg")
        results: dict[str, ResultType] = {
            "test_func1": {"times": [TEST_TIME_VALUE]},
            "test_func2": {"times": [TEST_SLOW_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func1": complete_stat({"avg": TEST_TIME_VALUE}),
            "test_func2": complete_stat({"avg": TEST_SLOW_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        lines = output.strip().split("\n")
        length = 2
        assert len(lines) == length
        assert lines[0] == str(TEST_TIME_VALUE)
        assert lines[1] == str(TEST_SLOW_TIME)

    def test_format_metric_fallback_for_single_trial(self) -> None:
        """Test fallback to 'time' when 'avg' requested for single trial."""
        formatter = SimpleFormatter(metric="avg")
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }  # No 'avg' key
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        assert output.strip() == str(TEST_TIME_VALUE)

    def test_format_missing_metric(self) -> None:
        """Test handling of missing metrics."""
        with pytest.raises(ValueError, match="'non_existent' is not a valid metric"):
            SimpleFormatter(metric="non_existent")  # type: ignore [assignment, arg-type]

    def test_format_with_custom_formatter(self) -> None:
        """Test SimpleFormatter with custom format function."""

        # Define a custom formatter that rounds to 3 decimal places and adds a unit
        def custom_format(method_name: str, value: float) -> str:
            _ = method_name
            return f"{value:.3f}s"

        formatter = SimpleFormatter(metric="avg", item_format=custom_format)
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        assert output.strip() == f"{TEST_TIME_VALUE:.3f}s"

    def test_format_with_custom_joiner(self) -> None:
        """Test SimpleFormatter with custom join function."""

        # Define a custom joiner that uses commas
        def custom_join(values: list[str]) -> str:
            return ", ".join(values)

        formatter = SimpleFormatter(metric="avg", list_format=custom_join)
        results: dict[str, ResultType] = {
            "test_func1": {"times": [TEST_TIME_VALUE]},
            "test_func2": {"times": [TEST_SLOW_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func1": complete_stat({"avg": TEST_TIME_VALUE}),
            "test_func2": complete_stat({"avg": TEST_SLOW_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        expected = f"{TEST_TIME_VALUE}, {TEST_SLOW_TIME}"
        assert output == expected

    def test_format_with_combined_custom_functions(self) -> None:
        """Test SimpleFormatter with both custom format and join functions."""

        # Define custom formatter and joiner
        def custom_format(method_name: str, value: float) -> str:
            _ = method_name
            return f"{value*1000:.1f}ms"

        def custom_join(values: list[str]) -> str:
            return " | ".join(values)

        formatter = SimpleFormatter(
            metric="avg",
            item_format=custom_format,
            list_format=custom_join,
        )
        results: dict[str, ResultType] = {
            "test_func1": {"times": [TEST_TIME_VALUE]},
            "test_func2": {"times": [TEST_SLOW_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func1": complete_stat({"avg": TEST_TIME_VALUE}),
            "test_func2": complete_stat({"avg": TEST_SLOW_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        expected = f"{TEST_TIME_VALUE*1000:.1f}ms | {TEST_SLOW_TIME*1000:.1f}ms"
        assert output == expected

    def test_format_with_method_name(self) -> None:
        """Test SimpleFormatter with custom format function using method name."""

        # Define a custom formatter that includes method name in output
        def custom_format(method_name: str, value: float) -> str:
            return f"{method_name}={value:.2f}"

        formatter = SimpleFormatter(metric="avg", item_format=custom_format)
        results: dict[str, ResultType] = {
            "test_func1": {"times": [TEST_TIME_VALUE]},
            "test_func2": {"times": [TEST_SLOW_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func1": complete_stat({"avg": TEST_TIME_VALUE}),
            "test_func2": complete_stat({"avg": TEST_SLOW_TIME}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        lines = output.strip().split("\n")
        assert lines[0] == f"test_func1={TEST_TIME_VALUE:.2f}"
        assert lines[1] == f"test_func2={TEST_SLOW_TIME:.2f}"


class TestDataFrameFormatter:
    """Tests for the DataFrameFormatter class."""

    @pytest.mark.skipif(
        "pandas" not in pytest.importorskip("pandas").__name__,
        reason="pandas is not installed",
    )
    def test_format_basic(self) -> None:
        """Test basic DataFrame formatting."""
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)

        assert isinstance(output, pd.DataFrame)
        assert "Function" in output.columns
        assert "Time (s)" in output.columns
        assert len(output) == 1
        assert output["Function"].iloc[0] == "test_func"
        assert output["Time (s)"].iloc[0] == TEST_TIME_VALUE

    @pytest.mark.skipif(
        "pandas" not in pytest.importorskip("pandas").__name__,
        reason="pandas is not installed",
    )
    def test_format_with_memory(self) -> None:
        """Test DataFrame formatting with memory metrics."""
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
            ),
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)

        assert "Avg Memory (KB)" in output.columns
        assert "Max Memory (KB)" in output.columns
        assert output["Avg Memory (KB)"].iloc[0] == TEST_AVG_MEMORY
        assert output["Max Memory (KB)"].iloc[0] == TEST_MAX_MEMORY

    @pytest.mark.skipif(
        "pandas" not in pytest.importorskip("pandas").__name__,
        reason="pandas is not installed",
    )
    def test_format_with_output(self) -> None:
        """Test DataFrame formatting with function outputs."""
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {
            "test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]},
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)

        assert "Output" in output.columns
        assert output["Output"].iloc[0] == "result"

    def test_import_error(self) -> None:
        """Test handling of missing pandas import."""
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        # Mock the import statement to raise ImportError
        with mock.patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'pandas'")

            with pytest.raises(
                ImportError,
                match="pandas is required for DataFrame output",
            ):
                formatter.format(results, stats, config)

    def test_format_with_memory_unit(self) -> None:
        """Test formatting results with different memory units."""
        unit = MemoryUnit("MB")
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024 * 1024],  # 1 MB in bytes
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg": TEST_TIME_VALUE,
                    "avg_memory": 1024 * 1024,  # 1 MB in bytes
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=1, memory=unit)

        # Skip test if pandas is not installed
        try:
            output = formatter.format(results, stats, config)
            assert "Memory (MB)" in output.columns
            assert output["Memory (MB)"].iloc[0] == 1.0  # 1 MB
        except ImportError:
            pytest.skip("pandas is not installed")


class TestReporters:
    """Tests for reporter classes."""

    def test_reporter_base_class(self) -> None:
        """Test the base Reporter class."""
        formatter = TableFormatter()

        # Create a concrete subclass that implements report_formatted
        class ConcreteReporter(Reporter):
            def report_formatted(self, formatted_output: Formatted) -> None:
                pass

        reporter = ConcreteReporter(formatter)

        assert reporter.formatter is formatter

        # Mock the report_formatted method to track if it's called
        original_send = reporter.report_formatted
        mock_send = MagicMock()
        reporter.report_formatted = mock_send  # type: ignore [method-assign]

        # Test data
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        # Call report which should call report_formatted
        reporter.report(results, stats, config)

        # Verify report_formatted was called
        mock_send.assert_called_once()

        # Restore the original method
        reporter.report_formatted = original_send  # type: ignore [method-assign]

    def test_stream_reporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the StreamReporter class."""
        formatter = TableFormatter()
        reporter = StreamReporter(formatter)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        captured = capsys.readouterr()
        assert test_output in captured.out

    def test_stream_reporter_with_custom_file(self) -> None:
        """Test StreamReporter with a custom file."""
        formatter = TableFormatter()
        output = io.StringIO()
        reporter = StreamReporter(formatter, file=output)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        assert test_output in output.getvalue()

    def test_console_reporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the ConsoleReporter class."""
        formatter = TableFormatter()
        reporter = ConsoleReporter(formatter)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        captured = capsys.readouterr()
        assert test_output in captured.out

    def test_file_reporter(self, tmp_path: Path) -> None:
        """Test the FileReporter class with a path object."""
        formatter = TableFormatter()
        output_file = tmp_path / "benchmark_results.txt"
        # Fixed parameter order: path first, then formatter
        reporter = FileReporter(str(output_file), formatter)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        assert output_file.exists()
        content = output_file.read_text()
        assert test_output in content

    def test_file_reporter_with_string_path(self, tmp_path: Path) -> None:
        """Test the FileReporter class with a string path."""
        formatter = TableFormatter()
        output_file = tmp_path / "benchmark_results.txt"
        # Fixed parameter order: path first, then formatter
        reporter = FileReporter(str(output_file), formatter)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        assert output_file.exists()
        content = output_file.read_text()
        assert test_output in content

    def test_file_reporter_append(self, tmp_path: Path) -> None:
        """Test the FileReporter class with append mode."""
        formatter = TableFormatter()
        output_file = tmp_path / "benchmark_results.txt"

        # First write - fixed parameter order
        reporter = FileReporter(str(output_file), formatter)
        reporter.report_formatted("First line")

        # Second write with append mode - fixed parameter order
        reporter = FileReporter(str(output_file), formatter, mode="a")
        reporter.report_formatted("Second line")

        content = output_file.read_text()
        assert "First line" in content
        assert "Second line" in content

    def test_callback_reporter(self) -> None:
        """Test the CallbackReporter class."""
        formatter = TableFormatter()
        callback_results = []

        def callback_func(output: str | object) -> None:
            callback_results.append(output)

        reporter = CallbackReporter(callback_func, formatter)

        test_output = "Test benchmark output"
        reporter.report_formatted(test_output)

        assert len(callback_results) == 1
        assert callback_results[0] == test_output

    def test_reporter_report(self) -> None:
        """Test the report method of Reporter."""
        # Import MagicMock
        # Create a mock formatter
        formatter = MagicMock(spec=Formatter)
        formatter.format.return_value = "Formatted output"

        # Create a concrete reporter subclass that implements report_formatted
        class TestReporter(Reporter):
            def report_formatted(self, formatted_output: str | object) -> None:
                pass

        # Create a reporter with the mock formatter and mock its report_formatted method
        reporter = TestReporter(formatter)
        reporter.report_formatted = MagicMock()  # type: ignore [method-assign]

        # Test data
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        # Call the report method
        reporter.report(results, stats, config)

        # Use keyword arguments to match the actual call pattern
        formatter.format.assert_called_once_with(
            results=results,
            stats=stats,
            config=config,
        )

        # Verify report_formatted was called with the formatter's output
        reporter.report_formatted.assert_called_once_with("Formatted output")


class TestSimpleConsoleReporter:
    """Tests for the SimpleConsoleReporter class."""

    def test_simple_console_reporter_basic(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test basic SimpleConsoleReporter functionality."""
        reporter = SimpleConsoleReporter(metric="avg")
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        reporter.report(results, stats, config)
        captured = capsys.readouterr()
        assert captured.out.strip() == str(TEST_TIME_VALUE)

    def test_simple_console_reporter_with_custom_formatter(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test SimpleConsoleReporter with custom formatter."""

        def custom_format(method_name: str, value: float) -> str:
            _ = method_name
            return f"{value:.3f}s"

        reporter = SimpleConsoleReporter(metric="avg", item_format=custom_format)
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        reporter.report(results, stats, config)
        captured = capsys.readouterr()
        assert captured.out.strip() == f"{TEST_TIME_VALUE:.3f}s"

    def test_simple_console_reporter_with_custom_joiner(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test SimpleConsoleReporter with custom joiner."""

        def custom_join(values: list[str]) -> str:
            return ", ".join(values)

        reporter = SimpleConsoleReporter(metric="avg", list_format=custom_join)
        results: dict[str, ResultType] = {
            "test_func1": {"times": [TEST_TIME_VALUE]},
            "test_func2": {"times": [TEST_SLOW_TIME]},
        }
        stats: dict[str, StatType] = {
            "test_func1": complete_stat({"avg": TEST_TIME_VALUE}),
            "test_func2": complete_stat({"avg": TEST_SLOW_TIME}),
        }
        config = BenchConfig(trials=1)

        reporter.report(results, stats, config)
        captured = capsys.readouterr()
        expected = f"{TEST_TIME_VALUE}, {TEST_SLOW_TIME}"
        assert captured.out.strip() == expected

    def test_simple_console_reporter_with_custom_file(self) -> None:
        """Test SimpleConsoleReporter with custom file."""
        output_file = io.StringIO()
        reporter = SimpleStreamReporter(metric="avg", file=output_file)
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        reporter.report(results, stats, config)
        output_file.seek(0)
        content = output_file.read()
        assert content.strip() == str(TEST_TIME_VALUE)


class TestGetReporter:
    """Tests for the get_reporter function."""

    def test_get_console_reporter(self) -> None:
        """Test getting a console reporter."""
        reporter = get_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_get_simple_reporter(self) -> None:
        """Test getting a simple reporter."""
        reporter = get_reporter("simple")
        assert isinstance(reporter, SimpleConsoleReporter)

    def test_get_plot_reporter(self) -> None:
        """Test getting a plot reporter."""
        reporter = get_reporter("plot")
        assert isinstance(reporter, PlotReporter)

    def test_get_reporter_with_kwargs(self) -> None:
        """Test getting a reporter with custom kwargs."""
        precision = 3
        reporter = get_reporter("console", {"precision": precision})

        assert isinstance(reporter, ConsoleReporter)
        assert isinstance(reporter.formatter, TableFormatter)
        assert reporter.formatter.precision == precision

    def test_get_reporter_with_invalid_name(self) -> None:
        """Test getting a reporter with an invalid name."""
        with pytest.raises(ValueError, match="Unknown reporter type:"):
            get_reporter("invalid_name")

    def test_get_file_reporter(self, tmp_path: Path) -> None:
        """Test getting a file reporter with 'file' name."""
        output_path = tmp_path / "results.txt"
        reporter = get_reporter("file", {"path": str(output_path)})

        assert isinstance(reporter, FileReporter)
        assert reporter.path == output_path

    def test_get_reporter_from_file_extension(self, tmp_path: Path) -> None:
        """Test getting a file reporter from file extension."""
        # Test CSV file extension
        csv_path = str(tmp_path / "results.csv")
        csv_reporter = get_reporter(csv_path)

        assert isinstance(csv_reporter, FileReporter)
        assert csv_reporter.path == Path(csv_path)
        assert isinstance(csv_reporter.formatter, CSVFormatter)

        # Test JSON file extension
        json_path = str(tmp_path / "results.json")
        json_reporter = get_reporter(json_path)

        assert isinstance(json_reporter, FileReporter)
        assert json_reporter.path == Path(json_path)
        assert isinstance(json_reporter.formatter, JSONFormatter)

        # Test with formatter override
        custom_formatter = CSVFormatter()
        json_with_csv = get_reporter(json_path, {"formatter": custom_formatter})

        assert isinstance(json_with_csv, FileReporter)
        assert json_with_csv.formatter is custom_formatter


class TestFormatterTimeUnits:
    """Test formatter handling of different time units from BenchConfig."""

    def test_formatters_with_time_units(self) -> None:
        """Test all formatters with different time units."""
        time_units = ["s", "ms", "μs", "us", "ns", "m"]
        # Create sample results and stats
        results: dict[str, ResultType] = {"test_func": {"times": [1.0, 2.0, 3.0]}}
        stats = {"test_func": complete_stat({"avg": 2.0, "min": 1.0, "max": 3.0})}

        for time_unit in time_units:
            config = BenchConfig(time=time_unit)

            # Test TableFormatter
            table_formatter = TableFormatter()
            table_output = table_formatter.format(results, stats, config)
            display_unit = "μs" if time_unit == "us" else time_unit
            assert f"Time ({display_unit})" in table_output

            # Test CSVFormatter
            csv_formatter = CSVFormatter()
            csv_output = csv_formatter.format(results, stats, config)
            assert f"Avg Time ({display_unit})" in csv_output

            # Test JSONFormatter
            json_formatter = JSONFormatter()
            json_output = json_formatter.format(results, stats, config)
            decoded_str = json_output.encode("utf-8").decode("unicode_escape")
            assert f'"time": "{time_unit}"' in decoded_str

            # Test DataFrameFormatter (skip if pandas not installed)
            try:

                df_formatter = DataFrameFormatter()
                df_output = df_formatter.format(results, stats, config)
                assert f"Avg Time ({display_unit})" in df_output.columns
            except ImportError:
                pass  # Skip if pandas not installed

    def test_time_unit_conversion_values(self) -> None:
        """Test that time values are correctly converted to different units."""
        # Base time in seconds
        base_seconds = 1.0

        # Expected conversion factors for different time units
        conversion_factors = {
            "s": 1.0,  # seconds (no conversion)
            "ms": 1000.0,  # milliseconds
            "μs": 1000000.0,  # microseconds
            "us": 1000000.0,  # microseconds (alternative notation)
            "ns": 1000000000.0,  # nanoseconds
            "m": 1 / 60.0,  # minutes
        }

        # Create sample results and stats with time in seconds
        results: dict[str, ResultType] = {"test_func": {"times": [base_seconds]}}
        stats = {
            "test_func": complete_stat(
                {
                    "avg": base_seconds,
                    "min": base_seconds,
                    "max": base_seconds,
                },
            ),
        }

        for time_unit, factor in conversion_factors.items():
            config = BenchConfig(time=time_unit, trials=1)
            expected_value = base_seconds * factor

            # Test TableFormatter
            table_formatter = TableFormatter()
            table_output = table_formatter.format(results, stats, config)
            assert f"{expected_value:.6f}" in table_output

            # Test CSVFormatter
            csv_formatter = CSVFormatter()
            csv_output = csv_formatter.format(results, stats, config)
            csv_lines = csv_output.strip().split("\n")
            if len(csv_lines) >= MIN_CSV_ROWS:  # Header + data line
                data_line = csv_lines[1].split(",")
                if len(data_line) >= MIN_VALUE_COLUMNS:  # Function name + value
                    converted_value = float(data_line[1])
                    assert abs(converted_value - expected_value) < FLOAT_TOLERANCE

            # Test JSONFormatter
            json_formatter = JSONFormatter()
            json_output = json_formatter.format(results, stats, config)
            json_data = json.loads(json_output)
            converted_value = json_data["stats"]["test_func"]["avg"]
            assert abs(converted_value - expected_value) < FLOAT_TOLERANCE

            # Test DataFrameFormatter (skip if pandas not installed)
            try:
                df_formatter = DataFrameFormatter()
                df_output = df_formatter.format(results, stats, config)
                display_unit = "μs" if time_unit == "us" else time_unit
                column_name = f"Time ({display_unit})"
                converted_value = df_output[column_name].iloc[0]
                assert abs(converted_value - expected_value) < FLOAT_TOLERANCE
            except ImportError:
                pass  # Skip if pandas not installed

    def test_time_unit_conversion_multiple_trials(self) -> None:
        """Test time unit conversion with multiple trials."""
        # Base times in seconds
        base_times = [1.0, 2.0, 3.0]
        avg_time = 2.0
        min_time = 1.0
        max_time = 3.0

        # Create sample results and stats
        results: dict[str, ResultType] = {"test_func": {"times": base_times}}
        stats = {
            "test_func": complete_stat(
                {"avg": avg_time, "min": min_time, "max": max_time},
            ),
        }

        # Test conversion to milliseconds
        time_unit = "ms"
        factor = 1000.0
        config = BenchConfig(time=time_unit, trials=3)

        # Expected converted values
        expected_avg = avg_time * factor
        expected_min = min_time * factor
        expected_max = max_time * factor

        # Test TableFormatter
        table_formatter = TableFormatter()
        table_output = table_formatter.format(results, stats, config)
        assert f"{expected_avg:.6f}" in table_output
        assert f"{expected_min:.6f}" in table_output
        assert f"{expected_max:.6f}" in table_output

        # Test CSVFormatter
        csv_formatter = CSVFormatter()
        csv_output = csv_formatter.format(results, stats, config)
        csv_lines = csv_output.strip().split("\n")
        if len(csv_lines) >= MIN_CSV_ROWS:  # Header + data line
            data_line = csv_lines[1].split(",")
            if len(data_line) >= MIN_STAT_COLUMNS:  # Function name + avg + min + max
                assert abs(float(data_line[1]) - expected_avg) < FLOAT_TOLERANCE
                assert abs(float(data_line[2]) - expected_min) < FLOAT_TOLERANCE
                assert abs(float(data_line[3]) - expected_max) < FLOAT_TOLERANCE

        # Test JSONFormatter
        json_formatter = JSONFormatter()
        json_output = json_formatter.format(results, stats, config)
        json_data = json.loads(json_output)
        assert (
            abs(json_data["stats"]["test_func"]["avg"] - expected_avg) < FLOAT_TOLERANCE
        )
        assert (
            abs(json_data["stats"]["test_func"]["min"] - expected_min) < FLOAT_TOLERANCE
        )
        assert (
            abs(json_data["stats"]["test_func"]["max"] - expected_max) < FLOAT_TOLERANCE
        )


class TestFormattersWithTimeDisabled:
    """Test how formatters behave when time=False is set."""

    def test_table_formatter_with_time_disabled(self) -> None:
        """Test that TableFormatter correctly handles time=False."""
        formatter = TableFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024, 2048, 3072],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=3, memory=True, time=False)

        output = formatter.format(results, stats, config)

        # Time columns should not be present
        assert "Time" not in output
        assert "Avg Time" not in output
        assert "Min Time" not in output
        assert "Max Time" not in output

        # But memory columns should be present
        assert "Avg Mem (KB)" in output
        assert "Max Mem (KB)" in output
        assert "test_func" in output

    def test_csv_formatter_with_time_disabled(self) -> None:
        """Test that CSVFormatter correctly handles time=False."""
        formatter = CSVFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024, 2048, 3072],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=3, memory=True, time=False)

        output = formatter.format(results, stats, config)

        # Parse the CSV output
        reader = csv.reader(io.StringIO(output))
        headers = next(reader)

        # Check that time-related columns are not present
        assert "Time (s)" not in headers
        assert "Avg Time (s)" not in headers
        assert "Min Time (s)" not in headers
        assert "Max Time (s)" not in headers

        # But memory columns are present
        assert "Avg Memory (KB)" in headers
        assert "Max Memory (KB)" in headers

        # Check that data row exists and contains function name
        data_row = next(reader)
        assert data_row[0] == "test_func"

        # Verify memory values exist in the data row
        assert float(data_row[1]) == TEST_AVG_MEMORY
        assert float(data_row[2]) == TEST_MAX_MEMORY

    def test_json_formatter_with_time_disabled(self) -> None:
        """Test that JSONFormatter correctly handles time=False."""
        formatter = JSONFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024, 2048, 3072],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=3, memory=True, time=False)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        # Check that config, stats and results exist
        assert "config" in data
        assert "stats" in data
        assert "results" in data

        # Verify config has time=False
        assert data["config"]["time"] is False

        # Check that time-related stats are not present for test_func
        assert "avg" not in data["stats"]["test_func"]
        assert "min" not in data["stats"]["test_func"]
        assert "max" not in data["stats"]["test_func"]

        # But memory stats are present
        assert "avg_memory" in data["stats"]["test_func"]
        assert "max_memory" in data["stats"]["test_func"]
        assert data["stats"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["stats"]["test_func"]["max_memory"] == TEST_MAX_MEMORY

        # Check that results contain memory but not times
        assert "times" not in data["results"]["test_func"]
        assert "memory" in data["results"]["test_func"]

    @pytest.mark.skipif(
        "pandas" not in pytest.importorskip("pandas").__name__,
        reason="pandas is not installed",
    )
    def test_dataframe_formatter_with_time_disabled(self) -> None:
        """Test that DataFrameFormatter correctly handles time=False."""
        formatter = DataFrameFormatter()
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024, 2048, 3072],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=3, memory=True, time=False)

        output = formatter.format(results, stats, config)

        # Check that the result is a DataFrame
        assert isinstance(output, pd.DataFrame)

        # Verify columns - should not include time columns
        assert "Time (s)" not in output.columns
        assert "Avg Time (s)" not in output.columns
        assert "Min Time (s)" not in output.columns
        assert "Max Time (s)" not in output.columns

        # But should include memory columns
        assert "Avg Memory (KB)" in output.columns
        assert "Max Memory (KB)" in output.columns

        # Check data values
        assert len(output) == 1
        assert output["Function"].iloc[0] == "test_func"
        assert output["Avg Memory (KB)"].iloc[0] == TEST_AVG_MEMORY
        assert output["Max Memory (KB)"].iloc[0] == TEST_MAX_MEMORY

    def test_simple_formatter_with_time_disabled(self) -> None:
        """Test that SimpleFormatter correctly handles time=False."""
        formatter = SimpleFormatter(
            metric="avg_memory",
        )  # Use memory metric since time is disabled
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024, 2048, 3072],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": TEST_AVG_MEMORY * 1024,
                    "max_memory": TEST_MAX_MEMORY * 1024,
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=3, memory=True, time=False)

        output = formatter.format(results, stats, config)

        # Should output the avg_memory value
        assert output.strip() == str(TEST_AVG_MEMORY)

    def test_simple_formatter_with_custom_format_time_disabled(self) -> None:
        """Test SimpleFormatter with custom format function when time is disabled."""

        def custom_format(method_name: str, value: float) -> str:
            return f"{method_name}: {value/1024:.2f} MB"

        formatter = SimpleFormatter(metric="avg_memory", item_format=custom_format)
        results: dict[str, ResultType] = {
            "test_func": {
                "memory": [1024 * 1024, 2048 * 1024],
                # no times key due to time=False
            },
        }
        stats: dict[str, StatType] = {
            "test_func": complete_stat(
                {
                    "avg_memory": 1024 * 1024,  # 1 MB in bytes
                },
                memory=True,
            ),
        }
        config = BenchConfig(trials=2, memory=True, time=False)

        output = formatter.format(results, stats, config)
        assert output.strip() == "test_func: 1.00 MB"
