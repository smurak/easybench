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
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TextIO,
    TypeAlias,
    TypeVar,
    get_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from matplotlib.figure import Figure

    from .core import BenchConfig, ResultsType, StatsType

T = TypeVar("T")

Formatted: TypeAlias = "str | pd.DataFrame | Figure"
MetricType = Literal["avg", "min", "max", "avg_memory", "max_memory"]


class MemoryUnit(str, Enum):
    """Memory unit options for display."""

    BYTES = "B"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    GIGABYTES = "GB"

    def __str__(self) -> str:
        """Convert unit to string."""
        return self.value

    def convert_bytes(self, byte_size: float) -> float:
        """
        Convert byte size to this memory unit.

        Args:
            byte_size (float): The size in bytes.

        Returns:
            float: The converted size in this memory unit.

        """
        unit_divisors = {
            MemoryUnit.BYTES: 1,
            MemoryUnit.KILOBYTES: 1024,
            MemoryUnit.MEGABYTES: 1024**2,
            MemoryUnit.GIGABYTES: 1024**3,
        }

        divisor = unit_divisors[self]
        return byte_size / divisor

    @classmethod
    def from_config(cls, config: BenchConfig) -> MemoryUnit:
        """Get memory unit from config."""
        unit = MemoryUnit.KILOBYTES

        if isinstance(config.memory, str):
            unit = MemoryUnit(config.memory.upper())
        elif isinstance(config.memory, MemoryUnit):
            unit = config.memory

        return unit


class Formatter(ABC):
    """Base class for all result formatters."""

    @abstractmethod
    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Formatted:
        """
        Format benchmark results.

        Args:
            results: Dictionary mapping benchmark names to result data.
            stats: Dictionary of calculated statistics.
            config: Benchmark configuration

        Returns:
            Formatted results in the appropriate format

        """

    def sort_keys(
        self,
        stats: StatsType,
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
        if config.sort_by in get_args(MetricType):
            metric: MetricType = config.sort_by  # type: ignore [assignment]
            return sorted(
                stats.keys(),
                key=lambda method_name: stats[method_name][metric],
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
        stats: StatsType,
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
        memory_unit = MemoryUnit.from_config(config)
        if config.trials == 1:
            header = f"{'Function':<{self.max_name_len + 2}} {'Time (s)':<11}"
            if config.memory:
                header += f" Memory ({memory_unit})"
        else:
            header = (
                f"{'Function':<{self.max_name_len + 2}} {'Avg Time (s)'} "
                f"{'Min Time (s)'} {'Max Time (s)'}"
            )
            if config.memory:
                header += f" Avg Mem ({memory_unit}) Max Mem ({memory_unit})"
        return header

    def _calculate_dash_length(self, config: BenchConfig) -> int:
        """Calculate dash line length."""
        if config.trials == 1:
            return self.max_name_len + (14 if not config.memory else 26)
        return self.max_name_len + (38 if not config.memory else 68)

    def _format_single_trial_results(
        self,
        output: list[str],
        stats: StatsType,
        config: BenchConfig,
    ) -> None:
        """Format results for a single trial benchmark."""
        memory_unit = MemoryUnit.from_config(config)
        for method_name in self.sorted_methods:
            stat = stats[method_name]
            time_val = f"{stat['avg']:.6f}".ljust(12)
            line = f"{method_name}".ljust(self.max_name_len + 2) + f" {time_val}"
            if config.memory:
                mem_val = f"{memory_unit.convert_bytes(stat['avg_memory']):.6f}".ljust(
                    12,
                )
                line += f" {mem_val}"
            output.append(line)

    def _format_multiple_trial_results(
        self,
        output: list[str],
        stats: StatsType,
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
        memory_unit = MemoryUnit.from_config(config)

        if config.memory:
            min_avg_memory = min(stat["avg_memory"] for stat in stats.values())
            max_avg_memory = max(stat["avg_memory"] for stat in stats.values())
            min_max_memory = min(stat["max_memory"] for stat in stats.values())
            max_max_memory = max(stat["max_memory"] for stat in stats.values())

        color = config.color if len(stats) > 1 else False
        for method_name in self.sorted_methods:
            stat = stats[method_name]

            # Format values with appropriate coloring
            avg_val = self._format_metric(
                stat["avg"],
                min_avg,
                max_avg,
                color=color,
            )
            min_val = self._format_metric(
                stat["min"],
                min_min,
                max_min,
                color=color,
            )
            max_val = self._format_metric(
                stat["max"],
                min_max,
                max_max,
                color=color,
            )

            # Format the function name with proper alignment
            method_col = f"{method_name}".ljust(self.max_name_len + 2)
            line = f"{method_col} {avg_val} {min_val} {max_val}"

            if config.memory:
                # Format memory values with proper coloring and conversion
                avg_mem = self._format_metric(
                    memory_unit.convert_bytes(stat["avg_memory"]),
                    memory_unit.convert_bytes(min_avg_memory),
                    memory_unit.convert_bytes(max_avg_memory),
                    color=color,
                )
                peak_mem = self._format_metric(
                    memory_unit.convert_bytes(stat["max_memory"]),
                    memory_unit.convert_bytes(min_max_memory),
                    memory_unit.convert_bytes(max_max_memory),
                    color=color,
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
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """Format results as CSV."""
        memory_unit = MemoryUnit.from_config(config)
        _ = results  # unused but avoids ARG002
        output = StringIO()
        writer = csv.writer(output)

        # Write header row
        if config.trials == 1:
            header = ["Function", "Time (s)"]
            if config.memory:
                header.append(f"Memory ({memory_unit})")
        else:
            header = ["Function", "Avg Time (s)", "Min Time (s)", "Max Time (s)"]
            if config.memory:
                header.extend(
                    [
                        f"Avg Memory ({memory_unit})",
                        f"Max Memory ({memory_unit})",
                    ],
                )

        writer.writerow(header)

        # Write data rows
        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]
            if config.trials == 1:
                row = [method_name, stat["avg"]]
                if config.memory:
                    row.append(memory_unit.convert_bytes(stat["avg_memory"]))
            else:
                row = [method_name, stat["avg"], stat["min"], stat["max"]]
                if config.memory:
                    row.extend(
                        [
                            memory_unit.convert_bytes(stat["avg_memory"]),
                            memory_unit.convert_bytes(stat["max_memory"]),
                        ],
                    )

            writer.writerow(row)

        return output.getvalue()


class JSONFormatter(Formatter):
    """Format results as JSON."""

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """Format results as JSON."""
        memory_unit = MemoryUnit.from_config(config)
        output_data: dict[str, dict[str, Any]] = {
            "config": {
                "trials": config.trials,
                "memory": config.memory,
                "sort_by": config.sort_by,
                "reverse": config.reverse,
                "memory_unit": memory_unit,
            },
            "results": {},
        }

        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name].copy()

            # Convert memory values to the specified unit
            if config.memory:
                if "avg_memory" in stat:
                    stat["avg_memory"] = memory_unit.convert_bytes(stat["avg_memory"])
                if "max_memory" in stat:
                    stat["max_memory"] = memory_unit.convert_bytes(stat["max_memory"])

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
        stats: StatsType,
        config: BenchConfig,
    ) -> pd.DataFrame:
        """
        Format results as DataFrame.

        Memory values are converted to the specified unit.
        """
        memory_unit = MemoryUnit.from_config(config)
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
                row["Time (s)"] = stat["avg"]
                if config.memory:
                    row[f"Memory ({memory_unit})"] = memory_unit.convert_bytes(
                        stat["avg_memory"],
                    )
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
                            f"Avg Memory ({memory_unit})": (
                                memory_unit.convert_bytes(stat["avg_memory"])
                            ),
                            f"Max Memory ({memory_unit})": (
                                memory_unit.convert_bytes(stat["max_memory"])
                            ),
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
        metric: MetricType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
    ) -> None:
        """
        Initialize with a metric to output.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, max_memory)
            item_format: Optional function to format individual values
                         Takes method_name and value as arguments
            list_format: Optional function to join multiple values

        """
        self.metric = metric
        if metric not in ("avg", "min", "max", "avg_memory", "max_memory"):
            msg = f"'{metric}' is not a valid metric for Simple formatter"
            raise ValueError(msg)

        # Use provided functions or defaults
        self.item_format = item_format or (lambda _, value: str(value))
        self.list_format = list_format or (lambda values: "\n".join(values) + "\n")

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """
        Format results as concise metric values.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics (memory values in bytes)
            config: Benchmark configuration

        Returns:
            Simple metrics as a string with memory values converted to specified unit

        """
        memory_unit = MemoryUnit.from_config(config)
        values = []
        _ = results

        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name]

            # Get the value for the specified metric
            value: float | None = None
            if self.metric in stat:
                value = stat[self.metric]
                # Convert if it's a memory metric
                if self.metric in ("avg_memory", "max_memory"):
                    value = memory_unit.convert_bytes(value)

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
        stats: StatsType,
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
        metric: MetricType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
        file: TextIO | None = None,
    ) -> None:
        """
        Initialize simple stream reporter.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, max_memory)
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
        metric: MetricType = "avg",
        item_format: Callable[[str, float], str] | None = None,
        list_format: Callable[[list[str]], str] | None = None,
    ) -> None:
        """
        Initialize simple console reporter.

        Args:
            metric: The metric to output (avg, min, max, avg_memory, max_memory)
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
