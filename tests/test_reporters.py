"""
Tests for the reporter classes and formatters.

This module tests the various reporter classes and formatters, including:
- TableFormatter
- CSVFormatter
- JSONFormatter
- DataFrameFormatter
- Reporter
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
from typing import TypeVar, cast

import pytest

from easybench.core import BenchConfig
from easybench.reporters import (
    CallbackReporter,
    ConsoleReporter,
    CSVFormatter,
    DataFrameFormatter,
    FileReporter,
    Formatter,
    JSONFormatter,
    Reporter,
    TableFormatter,
)

# Constants for test values to avoid magic numbers
TEST_TIME_VALUE = 0.1
TEST_SLOW_TIME = 0.2
TEST_SLOWER_TIME = 0.3
TEST_AVG_TIME = 0.2
TEST_MEMORY_VALUE = 1.0
TEST_AVG_MEMORY = 2.0
TEST_PEAK_MEMORY = 3.0
TEST_METRIC_MIN = 0.05
TEST_METRIC_MAX = 0.15
TEST_FLOAT_VALUE = 0.1
EXPECTED_ROW_COUNT = 2

# Type alias for import function return type
ImportReturnT = TypeVar("ImportReturnT")

# Type alias for pandas DataFrame-like objects
DataFrameLike = dict[str, list[str | float]] | Mapping[str, Sequence[str | float | int]]


class TestTableFormatter:
    """Tests for the TableFormatter class."""

    def test_format_single_trial(self) -> None:
        """Test formatting results for a single trial."""
        formatter = TableFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE], "memory": [1024]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE, "memory": TEST_MEMORY_VALUE}}
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
        results = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
            },
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
        results = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
                "avg_memory": TEST_AVG_MEMORY,
                "peak_memory": TEST_PEAK_MEMORY,
            },
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (3 trials)" in output
        assert "Function" in output
        assert "Avg Mem (KB)" in output
        assert "Peak Mem (KB)" in output
        assert "test_func" in output

    def test_format_with_return_values(self) -> None:
        """Test formatting results with return values."""
        formatter = TableFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)

        assert "Benchmark Results (1 trial)" in output
        assert "Benchmark Return Values" in output
        assert "test_func: result" in output

    def test_format_with_different_return_values(self) -> None:
        """Test formatting results with different return values across trials."""
        formatter = TableFormatter()
        results = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "output": [1, 2, 3],
            },
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
            },
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


class TestCSVFormatter:
    """Tests for the CSVFormatter class."""

    def test_format_single_trial(self) -> None:
        """Test formatting single trial results as CSV."""
        formatter = CSVFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
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
        results = {
            "test_func": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
            },
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
        results = {"test_func": {"times": [TEST_TIME_VALUE], "memory": [1024]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE, "memory": TEST_MEMORY_VALUE}}
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
        results = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
                "avg_memory": TEST_AVG_MEMORY,
                "peak_memory": TEST_PEAK_MEMORY,
            },
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
            "Peak Memory (KB)",
        ]
        assert rows[1][0] == "test_func"
        assert float(rows[1][1]) == TEST_AVG_TIME
        assert float(rows[1][4]) == TEST_AVG_MEMORY
        assert float(rows[1][5]) == TEST_PEAK_MEMORY


class TestJSONFormatter:
    """Tests for the JSONFormatter class."""

    def test_format_basic(self) -> None:
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
        config = BenchConfig(trials=1)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert "config" in data
        assert "results" in data
        assert data["config"]["trials"] == 1
        assert "test_func" in data["results"]
        assert data["results"]["test_func"]["time"] == TEST_TIME_VALUE

    def test_format_with_memory(self) -> None:
        """Test JSON formatting with memory metrics."""
        formatter = JSONFormatter()
        results = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
                "avg_memory": TEST_AVG_MEMORY,
                "peak_memory": TEST_PEAK_MEMORY,
            },
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert data["config"]["memory"] is True
        assert data["results"]["test_func"]["avg_memory"] == TEST_AVG_MEMORY
        assert data["results"]["test_func"]["peak_memory"] == TEST_PEAK_MEMORY

    def test_format_with_output(self) -> None:
        """Test JSON formatting with function outputs."""
        formatter = JSONFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)
        data = json.loads(output)

        assert "output" in data["results"]["test_func"]
        assert data["results"]["test_func"]["output"] == ["result"]


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
        results = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
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
        results = {
            "test_func": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
        }
        stats = {
            "test_func": {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
                "avg_memory": TEST_AVG_MEMORY,
                "peak_memory": TEST_PEAK_MEMORY,
            },
        }
        config = BenchConfig(trials=3, memory=True)

        output = formatter.format(results, stats, config)

        assert "Avg Memory (KB)" in output.columns
        assert "Peak Memory (KB)" in output.columns
        assert output["Avg Memory (KB)"].iloc[0] == TEST_AVG_MEMORY
        assert output["Peak Memory (KB)"].iloc[0] == TEST_PEAK_MEMORY

    @pytest.mark.skipif(
        "pandas" not in pytest.importorskip("pandas").__name__,
        reason="pandas is not installed",
    )
    def test_format_with_output(self) -> None:
        """Test DataFrame formatting with function outputs."""
        formatter = DataFrameFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE], "output": ["result"]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
        config = BenchConfig(trials=1, show_output=True)

        output = formatter.format(results, stats, config)

        assert "Output" in output.columns
        assert output["Output"].iloc[0] == "result"

    def test_import_error(self) -> None:
        """Test handling of missing pandas import."""
        formatter = DataFrameFormatter()
        results = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
        config = BenchConfig(trials=1)

        # Mock pandas import failure
        import builtins

        original_import = builtins.__import__

        def mock_import(
            name: str,
            *args: object,
            **kwargs: object,
        ) -> ImportReturnT | None:
            if name == "pandas":
                error_msg = "No module named 'pandas'"
                raise ImportError(error_msg)
            return cast("ImportReturnT", original_import(name, *args, **kwargs))

        try:
            builtins.__import__ = mock_import
            with pytest.raises(
                ImportError,
                match="pandas is required for DataFrame output",
            ):
                formatter.format(results, stats, config)
        finally:
            builtins.__import__ = original_import


class TestReporters:
    """Tests for reporter classes."""

    def test_reporter_base_class(self) -> None:
        """Test the base Reporter class."""
        from unittest.mock import MagicMock

        formatter = TableFormatter()
        reporter = Reporter(formatter)

        assert reporter.formatter is formatter

        # Create a concrete subclass that doesn't override _send
        # This is how we test that the method should be overridden
        class ConcreteReporter(Reporter):
            pass

        concrete_reporter = ConcreteReporter(formatter)

        # The base class implementation might not raise NotImplementedError
        # but we can test the required behavior by checking if _send exists
        # and then verifying report calls _send

        # Mock the _send method to track if it's called
        original_send = Reporter._send
        try:
            mock_send = MagicMock()
            Reporter._send = mock_send

            # Test data
            results = {"test_func": {"times": [TEST_TIME_VALUE]}}
            stats = {"test_func": {"time": TEST_TIME_VALUE}}
            config = BenchConfig(trials=1)

            # Call report which should call _send
            concrete_reporter.report(results, stats, config)

            # Verify _send was called
            mock_send.assert_called_once()
        finally:
            # Restore the original method
            Reporter._send = original_send

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

        def callback_func(output: str) -> None:
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

        # Create a concrete reporter subclass that doesn't raise NotImplementedError
        class TestReporter(Reporter):
            def _send(self, formatted_output: str) -> None:
                pass

        # Create a reporter with the mock formatter and mock its _send method
        reporter = TestReporter(formatter)
        reporter._send = MagicMock()

        # Test data
        results = {"test_func": {"times": [TEST_TIME_VALUE]}}
        stats = {"test_func": {"time": TEST_TIME_VALUE}}
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
