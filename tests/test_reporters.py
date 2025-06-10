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

import pytest

from easybench.core import BenchConfig, ResultType, StatType
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
    StreamReporter,
    TableFormatter,
)

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

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
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

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
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

        result = formatter._format_metric(
            value,
            min_value,
            max_value,
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
        assert "results" in data
        assert data["config"]["trials"] == 1
        assert "test_func" in data["results"]
        assert data["results"]["test_func"]["avg"] == TEST_TIME_VALUE

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
        assert data["results"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["results"]["test_func"]["max_memory"] == TEST_MAX_MEMORY

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

        assert "output" in data["results"]["test_func"]
        assert data["results"]["test_func"]["output"] == ["result"]

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

        assert data["config"]["memory_unit"] == "MB"
        assert data["results"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["results"]["test_func"]["max_memory"] == TEST_MAX_MEMORY


class TestSimpleFormatter:
    """Tests for the SimpleFormatter class."""

    def test_init_with_valid_metric(self) -> None:
        """Test initialization with valid metrics."""
        from easybench.reporters import SimpleFormatter

        for metric in get_args(MetricType):
            if metric == "def":
                continue
            formatter = SimpleFormatter(metric=metric)
            assert formatter.metric == metric

    def test_init_with_invalid_metric(self) -> None:
        """Test that 'def' is not a valid metric for SimpleFormatter."""
        from easybench.reporters import SimpleFormatter

        with pytest.raises(
            ValueError,
            match="'def' is not a valid metric for Simple formatter",
        ):
            SimpleFormatter(metric="def")  # type: ignore [arg-type]

    def test_format_single_trial(self) -> None:
        """Test formatting with single trial results."""
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

        with pytest.raises(ValueError, match="'non_existent' is not a valid metric"):
            SimpleFormatter(metric="non_existent")  # type: ignore [assignment, arg-type]

    def test_format_with_custom_formatter(self) -> None:
        """Test SimpleFormatter with custom format function."""
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        from easybench.reporters import SimpleFormatter

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
        import pandas as pd  # Import inside the test to avoid dependency issues

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

        # Mock pandas import failure
        import builtins
        from types import ModuleType

        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals_: Mapping[str, object] | None = None,
            locals_: Mapping[str, object] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == "pandas":
                error_msg = "No module named 'pandas'"
                raise ImportError(error_msg)
            return original_import(name, globals_, locals_, fromlist, level)

        try:
            builtins.__import__ = mock_import  # type: ignore [method-assign, assignment]
            with pytest.raises(
                ImportError,
                match="pandas is required for DataFrame output",
            ):
                formatter.format(results, stats, config)
        finally:
            builtins.__import__ = original_import

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
        from unittest.mock import MagicMock

        formatter = TableFormatter()

        # Create a concrete subclass that implements _send
        class ConcreteReporter(Reporter):
            def _send(self, formatted_output: Formatted) -> None:
                pass

        reporter = ConcreteReporter(formatter)

        assert reporter.formatter is formatter

        # Mock the _send method to track if it's called
        original_send = reporter._send
        mock_send = MagicMock()
        reporter._send = mock_send  # type: ignore [method-assign]

        # Test data
        results: dict[str, ResultType] = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats: dict[str, StatType] = {
            "test_func": complete_stat({"avg": TEST_TIME_VALUE}),
        }
        config = BenchConfig(trials=1)

        # Call report which should call _send
        reporter.report(results, stats, config)

        # Verify _send was called
        mock_send.assert_called_once()

        # Restore the original method
        reporter._send = original_send  # type: ignore [method-assign]

    def test_stream_reporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the StreamReporter class."""
        formatter = TableFormatter()
        reporter = StreamReporter(formatter)

        test_output = "Test benchmark output"
        reporter._send(test_output)

        captured = capsys.readouterr()
        assert test_output in captured.out

    def test_stream_reporter_with_custom_file(self) -> None:
        """Test StreamReporter with a custom file."""
        formatter = TableFormatter()
        output = io.StringIO()
        reporter = StreamReporter(formatter, file=output)

        test_output = "Test benchmark output"
        reporter._send(test_output)

        assert test_output in output.getvalue()

    def test_console_reporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the ConsoleReporter class."""
        formatter = TableFormatter()
        reporter = ConsoleReporter(formatter)

        test_output = "Test benchmark output"
        reporter._send(test_output)

        captured = capsys.readouterr()
        assert test_output in captured.out

    def test_file_reporter(self, tmp_path: Path) -> None:
        """Test the FileReporter class with a path object."""
        formatter = TableFormatter()
        output_file = tmp_path / "benchmark_results.txt"
        # Fixed parameter order: path first, then formatter
        reporter = FileReporter(str(output_file), formatter)

        test_output = "Test benchmark output"
        reporter._send(test_output)

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
        reporter._send(test_output)

        assert output_file.exists()
        content = output_file.read_text()
        assert test_output in content

    def test_file_reporter_append(self, tmp_path: Path) -> None:
        """Test the FileReporter class with append mode."""
        formatter = TableFormatter()
        output_file = tmp_path / "benchmark_results.txt"

        # First write - fixed parameter order
        reporter = FileReporter(str(output_file), formatter)
        reporter._send("First line")

        # Second write with append mode - fixed parameter order
        reporter = FileReporter(str(output_file), formatter, mode="a")
        reporter._send("Second line")

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
        reporter._send(test_output)

        assert len(callback_results) == 1
        assert callback_results[0] == test_output

    def test_reporter_report(self) -> None:
        """Test the report method of Reporter."""
        # Import MagicMock
        from unittest.mock import MagicMock

        # Create a mock formatter
        formatter = MagicMock(spec=Formatter)
        formatter.format.return_value = "Formatted output"

        # Create a concrete reporter subclass that implements _send
        class TestReporter(Reporter):
            def _send(self, formatted_output: str | object) -> None:
                pass

        # Create a reporter with the mock formatter and mock its _send method
        reporter = TestReporter(formatter)
        reporter._send = MagicMock()  # type: ignore [method-assign]

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

        # Verify _send was called with the formatter's output
        reporter._send.assert_called_once_with("Formatted output")


class TestSimpleConsoleReporter:
    """Tests for the SimpleConsoleReporter class."""

    def test_simple_console_reporter_basic(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test basic SimpleConsoleReporter functionality."""
        from easybench.reporters import SimpleConsoleReporter

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
        from easybench.reporters import SimpleConsoleReporter

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
        from easybench.reporters import SimpleConsoleReporter

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
        import io

        from easybench.reporters import SimpleStreamReporter

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
        from easybench.core import get_reporter
        from easybench.reporters import ConsoleReporter

        reporter = get_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_get_simple_reporter(self) -> None:
        """Test getting a simple reporter."""
        from easybench.core import get_reporter
        from easybench.reporters import SimpleConsoleReporter

        reporter = get_reporter("simple")
        assert isinstance(reporter, SimpleConsoleReporter)

    def test_get_plot_reporter(self) -> None:
        """Test getting a plot reporter."""
        from easybench.core import get_reporter
        from easybench.visualization import PlotReporter

        reporter = get_reporter("plot")
        assert isinstance(reporter, PlotReporter)

    def test_get_reporter_with_kwargs(self) -> None:
        """Test getting a reporter with custom kwargs."""
        from easybench.core import get_reporter
        from easybench.reporters import ConsoleReporter, TableFormatter

        # Create a custom formatter to pass through kwargs
        custom_formatter = TableFormatter()
        reporter = get_reporter("console", {"formatter": custom_formatter})

        assert isinstance(reporter, ConsoleReporter)
        assert reporter.formatter is custom_formatter

    def test_get_reporter_with_invalid_name(self) -> None:
        """Test getting a reporter with an invalid name."""
        from easybench.core import get_reporter

        with pytest.raises(ValueError, match="Unknown reporter type:"):
            get_reporter("invalid_name")

    def test_get_file_reporter(self, tmp_path: Path) -> None:
        """Test getting a file reporter with 'file' name."""
        from easybench.core import get_reporter
        from easybench.reporters import FileReporter

        output_path = tmp_path / "results.txt"
        reporter = get_reporter("file", {"path": str(output_path)})

        assert isinstance(reporter, FileReporter)
        assert reporter.path == output_path

    def test_get_reporter_from_file_extension(self, tmp_path: Path) -> None:
        """Test getting a file reporter from file extension."""
        from easybench.core import get_reporter
        from easybench.reporters import CSVFormatter, FileReporter, JSONFormatter

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
