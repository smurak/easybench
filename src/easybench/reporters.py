"""
Reporter classes for different output formats and destinations.

This module provides formatters for converting benchmark results to various formats
and reporters for sending formatted results to different destinations.
"""

from __future__ import annotations

import csv
import json
import sys
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TextIO,
    TypeAlias,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from .core import BenchConfig, ResultsType, SortType

T = TypeVar("T")

Stats: TypeAlias = dict[str, dict[str, float]]
Formatted: TypeAlias = "str | pd.DataFrame"


class Formatter(ABC):
    """Base class for all result formatters."""

    @abstractmethod
    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> Formatted:
        """
        Format benchmark results.

        Args:
            results: Dictionary mapping benchmark names to result data
                Example:
                {
                    "bench_append": {
                        "times": [0.01, 0.012],  # Execution times per trial
                        "memory": [10.5, 10.2],   # Memory usage (if enabled)
                        "output": ["result", "result2"]  # Function outputs (if enabled)
                    },
                    "bench_insert": {
                        "times": [0.02, 0.019, 0.021],
                        "memory": [12.1, 12.0, 12.3],
                        "output": ["result", "result2", "result3"]
                    }
                }
            stats: Dictionary of calculated statistics
                Example:
                {
                    "bench_append": {
                        "avg": 0.011, "min": 0.01, "max": 0.012,
                        "avg_memory": 10.35, "peak_memory": 10.5
                    },
                    "bench_insert": {
                        "avg": 0.02, "min": 0.019, "max": 0.021,
                        "avg_memory": 12.13, "peak_memory": 12.3
                    }
                }
            config: Benchmark configuration

        Returns:
            Formatted results in the appropriate format

        """

    def sort_keys(
        self,
        stats: Stats,
        config: BenchConfig,
    ) -> list[str]:
        """
        Sort stats keys based on the config.

        Args:
            stats: Dictionary of precomputed statistics for each benchmark
            config: Benchmark config

        Returns:
            Sorted list of stats keys

        """
        if config.sort_by in ("avg", "min", "max", "avg_memory", "peak_memory"):
            return sorted(
                stats.keys(),
                key=lambda method_name: stats[method_name][config.sort_by],
                reverse=config.reverse,
            )

        if config.reverse:
            return list(stats.keys())[::-1]

        return list(stats.keys())


class TableFormatter(Formatter):
    """Format results as text tables (current default format)."""

    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> str:
        """Format results as text tables."""
        output = []
        self.max_name_len = max(len(name) for name in results)
        self.sorted_methods = self.sort_keys(stats, config)

        # Add title line
        output.append(self._format_title(config))

        # Format header
        header = self._format_header(config)
        output.append(header)

        # Add dash line
        dash_length = self._calculate_dash_length(config)
        output.append("-" * dash_length)

        # Format result lines
        if config.trials == 1:
            self._format_single_trial_results(
                output,
                stats,
                config,
            )
        else:
            self._format_multiple_trial_results(
                output,
                stats,
                config,
            )

        # Add function outputs if requested
        if config.show_output:
            self._format_output_section(output, results)

        return "\n".join(output) + "\n"

    def _format_title(self, config: BenchConfig) -> str:
        """Format the benchmark title."""
        return (
            f"\nBenchmark Results ({config.trials} trial"
            f"{'s' if config.trials > 1 else ''}):\n"
        )

    def _format_header(self, config: BenchConfig) -> str:
        """Format the header row."""
        if config.trials == 1:
            header = f"{'Function':<{self.max_name_len + 2}} {'Time (s)':<11}"
            if config.memory:
                header += f" {'Memory (KB)'}"
        else:
            header = (
                f"{'Function':<{self.max_name_len + 2}} {'Avg Time (s)'} "
                f"{'Min Time (s)'} {'Max Time (s)'}"
            )
            if config.memory:
                header += f" {'Avg Mem (KB)'} {'Peak Mem (KB)'}"
        return header

    def _calculate_dash_length(self, config: BenchConfig) -> int:
        """Calculate dash line length."""
        if config.trials == 1:
            return self.max_name_len + (14 if not config.memory else 26)
        return self.max_name_len + (38 if not config.memory else 68)

    def _format_single_trial_results(
        self,
        output: list[str],
        stats: Stats,
        config: BenchConfig,
    ) -> None:
        """Format results for a single trial benchmark."""
        for method_name in self.sorted_methods:
            stat = stats[method_name]
            time_val = f"{stat['time']:.6f}".ljust(12)
            line = f"{method_name}".ljust(self.max_name_len + 2) + f" {time_val}"
            if config.memory:
                mem_val = f"{stat['memory']:.6f}".ljust(12)
                line += f" {mem_val}"
            output.append(line)

    def _format_multiple_trial_results(
        self,
        output: list[str],
        stats: Stats,
        config: BenchConfig,
    ) -> None:
        """Format results for multiple trial benchmarks."""
        # Find min and max values for each metric for coloring
        min_avg = min(stat["avg"] for stat in stats.values())
        max_avg = max(stat["avg"] for stat in stats.values())
        min_min = min(stat["min"] for stat in stats.values())
        max_min = max(stat["min"] for stat in stats.values())
        min_max = min(stat["max"] for stat in stats.values())
        max_max = max(stat["max"] for stat in stats.values())

        if config.memory:
            min_avg_memory = min(stat["avg_memory"] for stat in stats.values())
            max_avg_memory = max(stat["avg_memory"] for stat in stats.values())
            min_peak_memory = min(stat["peak_memory"] for stat in stats.values())
            max_peak_memory = max(stat["peak_memory"] for stat in stats.values())

        for method_name in self.sorted_methods:
            stat = stats[method_name]

            # Format values with appropriate coloring
            avg_val = self._format_metric(
                stat["avg"],
                min_avg,
                max_avg,
                color=config.color,
            )
            min_val = self._format_metric(
                stat["min"],
                min_min,
                max_min,
                color=config.color,
            )
            max_val = self._format_metric(
                stat["max"],
                min_max,
                max_max,
                color=config.color,
            )

            # Format the function name with proper alignment
            method_col = f"{method_name}".ljust(self.max_name_len + 2)
            line = f"{method_col} {avg_val} {min_val} {max_val}"

            if config.memory:
                # Format memory values with proper coloring
                avg_mem = self._format_metric(
                    stat["avg_memory"],
                    min_avg_memory,
                    max_avg_memory,
                    color=config.color,
                )
                peak_mem = self._format_metric(
                    stat["peak_memory"],
                    min_peak_memory,
                    max_peak_memory,
                    color=config.color,
                )
                line += f" {avg_mem} {peak_mem}"

            output.append(line)

    def _format_output_section(
        self,
        output: list[str],
        results: ResultsType,
    ) -> None:
        """Format the output section."""
        output.append("\nBenchmark Return Values:")
        output.append("-" * max(self.max_name_len + 10, 30))

        for method_name in self.sorted_methods:
            if "output" not in results[method_name]:
                continue

            return_values = results[method_name]["output"]
            if len({str(val) for val in return_values}) == 1:
                output.append(f"{method_name}: {return_values[0]}")
            else:
                output.append(f"{method_name}:")
                for i, val in enumerate(return_values):
                    output.append(f"  Trial {i+1}: {val}")

    def _format_metric(
        self,
        value: float,
        min_value: float,
        max_value: float,
        *,
        color: bool = True,
    ) -> str:
        """
        Format a metric value with appropriate coloring.

        Args:
            value: The value to format
            min_value: The minimum value across all benchmarks
            max_value: The maximum value across all benchmarks
            color: Whether to use colored output

        Returns:
            Formatted string with color codes if color is True

        """
        # Define ANSI color codes
        min_color = "\033[32m"  # GREEN
        max_color = "\033[31m"  # RED
        reset = "\033[0m"
        formatted = f"{value:.6f}".ljust(12)

        if not color:
            return formatted

        if value == min_value:
            return f"{min_color}{formatted}{reset}"
        if value == max_value:
            return f"{max_color}{formatted}{reset}"
        return formatted


class CSVFormatter(Formatter):
    """Format results as CSV."""

    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> str:
        """Format results as CSV."""
        _ = results  # unused but avoids ARG002
        output = StringIO()
        writer = csv.writer(output)

        # Write header row
        if config.trials == 1:
            header = ["Function", "Time (s)"]
            if config.memory:
                header.append("Memory (KB)")
        else:
            header = ["Function", "Avg Time (s)", "Min Time (s)", "Max Time (s)"]
            if config.memory:
                header.extend(["Avg Memory (KB)", "Peak Memory (KB)"])

        writer.writerow(header)

        # Write data rows
        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]
            if config.trials == 1:
                row = [method_name, stat["time"]]
                if config.memory:
                    row.append(stat["memory"])
            else:
                row = [method_name, stat["avg"], stat["min"], stat["max"]]
                if config.memory:
                    row.extend([stat["avg_memory"], stat["peak_memory"]])

            writer.writerow(row)

        return output.getvalue()


class JSONFormatter(Formatter):
    """Format results as JSON."""

    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> str:
        """Format results as JSON."""
        output_data: dict[str, dict[str, Any]] = {
            "config": {
                "trials": config.trials,
                "memory": config.memory,
                "sort_by": config.sort_by,
                "reverse": config.reverse,
            },
            "results": {},
        }

        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]
            output_data["results"][method_name] = stat

            # Add output values if requested
            if config.show_output and "output" in results[method_name]:
                output_data["results"][method_name]["output"] = results[method_name][
                    "output"
                ]

        return json.dumps(output_data, indent=2)


class DataFrameFormatter(Formatter):
    """Format results as pandas DataFrame."""

    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> pd.DataFrame:
        """Format results as DataFrame."""
        try:
            import pandas as pd
        except ImportError as err:
            error_msg = (
                "pandas is required for DataFrame output. "
                "Install with pip install pandas."
            )
            raise ImportError(error_msg) from err

        data = []
        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]
            row: dict[str, Any] = {"Function": method_name}

            # Add stats based on number of trials
            if config.trials == 1:
                row["Time (s)"] = stat["time"]
                if config.memory:
                    row["Memory (KB)"] = stat["memory"]
            else:
                row.update(
                    {
                        "Avg Time (s)": stat["avg"],
                        "Min Time (s)": stat["min"],
                        "Max Time (s)": stat["max"],
                    },
                )
                if config.memory:
                    row.update(
                        {
                            "Avg Memory (KB)": stat["avg_memory"],
                            "Peak Memory (KB)": stat["peak_memory"],
                        },
                    )

            # Add output if requested
            if config.show_output and "output" in results[method_name]:
                # Just take first output
                row["Output"] = results[method_name]["output"][0]

            data.append(row)

        return pd.DataFrame(data)


class SimpleFormatter(Formatter):
    """Format results as concise metric values."""

    def __init__(
        self,
        metric: SortType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
    ) -> None:
        """
        Initialize with a metric to output.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, peak_memory)
            item_format: Optional function to format individual values
                         Takes method_name and value as arguments
            list_format: Optional function to join multiple values

        """
        self.metric = metric
        if metric not in ("avg", "min", "max", "avg_memory", "peak_memory"):
            msg = f"'{metric}' is not a valid metric for Simple formatter"
            raise ValueError(msg)

        # Use provided functions or defaults
        self.item_format = item_format or (lambda _, value: str(value))
        self.list_format = list_format or (lambda values: "\n".join(values) + "\n")

    def format(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> str:
        """
        Format results as concise metric values.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Simple metric values as a string

        """
        values = []
        _ = results

        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]

            # Get the value for the specified metric
            value: float | None = None
            if self.metric in stat:
                value = stat[self.metric]
            elif config.trials == 1 and self.metric == "avg":
                value = stat.get("time", None)
            elif config.trials == 1 and self.metric == "avg_memory":
                value = stat.get("memory", None)

            if value is not None:
                values.append(self.item_format(method_name, value))
            else:
                values.append("")

        return self.list_format(values)


class Reporter:
    """Base reporter class for sending benchmark results to destinations."""

    def __init__(self, formatter: Formatter) -> None:
        """
        Initialize reporter with a formatter.

        Args:
            formatter: Formatter to use for formatting results

        """
        self.formatter = formatter

    def report(
        self,
        results: ResultsType,
        stats: Stats,
        config: BenchConfig,
    ) -> None:
        """
        Report benchmark results.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        """
        formatted = self.formatter.format(
            results=results,
            stats=stats,
            config=config,
        )
        self._send(formatted)

    @abstractmethod
    def _send(self, formatted_output: Formatted) -> None:
        """
        Send formatted output to the destination.

        Args:
            formatted_output: The formatted output to send

        """


class StreamReporter(Reporter):
    """Reporter that sends output to a stream."""

    def __init__(
        self,
        formatter: Formatter | None = None,
        file: TextIO | None = None,
    ) -> None:
        """
        Initialize stream reporter.

        Args:
            formatter: Formatter to use (defaults to TableFormatter)
            file: File object to write to (defaults to sys.stdout)

        """
        super().__init__(formatter or TableFormatter())
        self.file = file

    def _send(self, formatted_output: Formatted) -> None:
        """
        Print formatted output to stream.

        Args:
            formatted_output: The formatted output to print

        """
        _file = self.file or sys.stdout
        print(formatted_output, file=_file, end="")


class ConsoleReporter(StreamReporter):
    """Reporter that sends output to the console (stdout)."""

    def __init__(self, formatter: Formatter | None = None) -> None:
        """
        Initialize console reporter.

        Args:
            formatter: Formatter to use (defaults to TableFormatter)

        """
        super().__init__(formatter=formatter, file=None)


class FileReporter(Reporter):
    """Reporter that sends output to a file."""

    def __init__(
        self,
        path: str | Path,
        formatter: Formatter | None = None,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize file reporter.

        Args:
            path: Path to the output file
            formatter: Formatter to use (defaults based on file extension)
            mode: File open mode ('w' for write, 'a' for append)
            encoding: File encoding

        """
        if formatter is None:
            # Infer formatter from file extension
            path_obj = Path(path)
            ext = path_obj.suffix.lower()
            if ext == ".csv":
                formatter = CSVFormatter()
            elif ext == ".json":
                formatter = JSONFormatter()
            else:
                formatter = TableFormatter()

        super().__init__(formatter)
        self.path = Path(path)
        self.mode = mode
        self.encoding = encoding

    def _send(self, formatted_output: Formatted) -> None:
        """
        Write formatted output to file.

        Args:
            formatted_output: The formatted output to write

        """
        with self.path.open(self.mode, encoding=self.encoding) as f:
            f.write(formatted_output)


class CallbackReporter(Reporter):
    """Reporter that sends output to a callback function."""

    def __init__(
        self,
        callback: Callable[[Formatted], None],
        formatter: Formatter | None = None,
    ) -> None:
        """
        Initialize callback reporter.

        Args:
            callback: Function to call with formatted output
            formatter: Formatter to use (defaults to DataFrameFormatter)

        """
        super().__init__(formatter or DataFrameFormatter())
        self.callback = callback

    def _send(self, formatted_output: Formatted) -> None:
        """
        Send formatted output to callback.

        Args:
            formatted_output: The formatted output to send

        """
        self.callback(formatted_output)


class SimpleStreamReporter(StreamReporter):
    """Reporter that outputs concise metric values to a stream."""

    def __init__(
        self,
        metric: SortType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
        file: TextIO | None = None,
    ) -> None:
        """
        Initialize simple stream reporter.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, peak_memory)
            item_format: Optional function to format individual values
                         Takes method_name and value as arguments
            list_format: Optional function to join multiple values
            file: File object to write to (defaults to sys.stdout)

        """
        formatter = SimpleFormatter(
            metric=metric,
            item_format=item_format,
            list_format=list_format,
        )
        super().__init__(formatter=formatter, file=file)


class SimpleConsoleReporter(SimpleStreamReporter):
    """Reporter that outputs concise metric values to the console (stdout)."""

    def __init__(
        self,
        metric: SortType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
    ) -> None:
        """
        Initialize simple console reporter.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, peak_memory)
            item_format: Optional function to format individual values
                         Takes method_name and value as arguments
            list_format: Optional function to join multiple values

        """
        super().__init__(
            metric=metric,
            item_format=item_format,
            list_format=list_format,
            file=None,
        )
