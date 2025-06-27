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

from .utils import visual_ljust, visual_width

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from matplotlib.figure import Figure

    from .core import BenchConfig, ResultsType, StatsType, StatType

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


class TimeUnit(str, Enum):
    """Time unit options for display."""

    NANOSECONDS = "ns"
    MICROSECONDS = "μs"
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "m"

    def __str__(self) -> str:
        """Convert unit to string."""
        return self.value

    def convert_seconds(self, seconds: float) -> float:
        """
        Convert seconds to this time unit.

        Args:
            seconds (float): The time in seconds.

        Returns:
            float: The converted time in this time unit.

        """
        unit_multipliers = {
            TimeUnit.NANOSECONDS: 1e9,
            TimeUnit.MICROSECONDS: 1e6,
            TimeUnit.MILLISECONDS: 1e3,
            TimeUnit.SECONDS: 1,
            TimeUnit.MINUTES: 1 / 60,
        }

        multiplier = unit_multipliers[self]
        return seconds * multiplier

    @classmethod
    def from_config(cls, config: BenchConfig) -> TimeUnit:
        """Get time unit from config."""
        unit = TimeUnit.SECONDS

        if isinstance(config.time, str):
            time_str = config.time.lower()
            # Handle 'us' as an alternative for microseconds
            unit = TimeUnit.MICROSECONDS if time_str == "us" else TimeUnit(time_str)

        elif isinstance(config.time, TimeUnit):
            unit = config.time

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

    def shrink_data(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> None:
        """
        Modify the given results and stats in place by removing any unnecessary entries.

        This operation mutates the input data structures directly.

        Args:
            results (ResultsType): The benchmark results to be pruned.
            stats (StatsType): The corresponding statistical data to be pruned.
            config (BenchConfig): Configuration specifying which data to keep or remove.

        Returns:
            None

        """
        self._shrink_time_data(results, stats, config)
        self._shrink_memory_data(results, stats, config)

    def _shrink_time_data(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> None:
        """
        Modify the given results and stats by removing any unnecessary time entries.

        This operation mutates the input data structures directly.

        Args:
            results (ResultsType): The benchmark results to be pruned.
            stats (StatsType): The corresponding statistical data to be pruned.
            config (BenchConfig): Configuration specifying which data to keep or remove.

        Returns:
            None

        """
        if not config.time:
            for result in results.values():
                if "times" in result:
                    del result["times"]
            for stat in stats.values():
                if "avg" in stat:
                    del stat["avg"]
                if "max" in stat:
                    del stat["max"]
                if "min" in stat:
                    del stat["min"]

    def _shrink_memory_data(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> None:
        """
        Modify the given results and stats by removing any unnecessary memory entries.

        This operation mutates the input data structures directly.

        Args:
            results (ResultsType): The benchmark results to be pruned.
            stats (StatsType): The corresponding statistical data to be pruned.
            config (BenchConfig): Configuration specifying which data to keep or remove.

        Returns:
            None

        """
        if not config.memory:
            for result in results.values():
                if "memory" in result:
                    del result["memory"]
            for stat in stats.values():
                if "avg_memory" in stat:
                    del stat["avg_memory"]
                if "max_memory" in stat:
                    del stat["max_memory"]


class TableFormatter(Formatter):
    """Format results as text tables (current default format)."""

    def __init__(self, precision: int = 6) -> None:
        """
        Initialize.

        Args:
            precision: Number of decimal places to display for numeric values

        """
        super().__init__()
        self.precision = precision

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """Format results as text tables."""
        output = []
        self.max_name_len = max(visual_width(name) for name in results)
        self.sorted_methods = self.sort_keys(stats, config)

        # Pre-calculate all formatting data in one go
        self.formatting_data = self._prepare_formatting_data(stats, config)

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
            self._format_single_trial_results(output, config)
        else:
            self._format_multiple_trial_results(output, config)

        # Add function outputs if requested
        if config.show_output:
            self._format_output_section(output, results)

        return "\n".join(output) + "\n"

    def _prepare_formatting_data(
        self,
        stats: StatsType,
        config: BenchConfig,
    ) -> dict[str, Any]:
        """
        Prepare all formatting data.

        Returns:
            Dictionary containing all pre-calculated formatting data

        """
        memory_unit = MemoryUnit.from_config(config)
        time_unit = TimeUnit.from_config(config)

        data: dict[str, Any] = {
            "memory_unit": memory_unit,
            "time_unit": time_unit,
            "converted_stats": {},
            "column_widths": {},
            "extremes": {},
        }

        # Convert all values once and store them
        for method_name in stats:
            stat = stats[method_name]
            if config.time:
                converted = {
                    "avg_time": time_unit.convert_seconds(stat["avg"]),
                    "min_time": time_unit.convert_seconds(stat["min"]),
                    "max_time": time_unit.convert_seconds(stat["max"]),
                }
            else:
                converted = {}

            if config.memory:
                avg_memory = stat.get("avg_memory", 0.0)
                max_memory = stat.get("max_memory", avg_memory)
                converted["avg_memory"] = memory_unit.convert_bytes(avg_memory)
                converted["max_memory"] = memory_unit.convert_bytes(max_memory)

            data["converted_stats"][method_name] = converted

        # Calculate column widths based on converted values
        self._calculate_column_widths(data, config)

        # Calculate extremes for coloring (only for multiple trials)
        if config.trials > 1:
            self._calculate_extremes(data, config)

        return data

    def _calculate_column_widths(
        self,
        data: dict[str, Any],
        config: BenchConfig,
    ) -> None:
        """Calculate optimal column widths and store in formatting data."""
        converted_stats = data["converted_stats"]
        memory_unit = data["memory_unit"]
        time_unit = data["time_unit"]

        # Calculate time column widths
        if config.time:
            # Collect all time values for width calculation
            time_values = []
            for converted in converted_stats.values():
                time_values.extend(
                    [
                        converted["avg_time"],
                        converted["min_time"],
                        converted["max_time"],
                    ],
                )

            max_time_str_len = max(
                len(f"{val:.{self.precision}f}") for val in time_values
            )

            if config.trials == 1:
                time_header_len = len(f"Time ({time_unit})")
                data["column_widths"]["time"] = (
                    max(max_time_str_len, time_header_len) + 2
                )
            else:
                headers = [
                    f"Avg Time ({time_unit})",
                    f"Min Time ({time_unit})",
                    f"Max Time ({time_unit})",
                ]
                for i, key in enumerate(["avg_time", "min_time", "max_time"]):
                    data["column_widths"][key] = (
                        max(max_time_str_len, len(headers[i])) + 2
                    )

        # Calculate memory column widths if needed
        if config.memory:
            memory_values = []
            for converted in converted_stats.values():
                memory_values.extend([converted["avg_memory"], converted["max_memory"]])

            max_memory_str_len = max(
                len(f"{val:.{self.precision}f}") for val in memory_values
            )

            if config.trials == 1:
                memory_header_len = len(f"Memory ({memory_unit})")
                data["column_widths"]["memory"] = (
                    max(max_memory_str_len, memory_header_len) + 2
                )
            else:
                headers = [f"Avg Mem ({memory_unit})", f"Max Mem ({memory_unit})"]
                for i, key in enumerate(["avg_memory", "max_memory"]):
                    data["column_widths"][key] = (
                        max(max_memory_str_len, len(headers[i])) + 2
                    )

    def _calculate_extremes(self, data: dict[str, Any], config: BenchConfig) -> None:
        """Calculate min/max values for coloring and store in formatting data."""
        converted_stats = data["converted_stats"]

        # Time extremes
        if config.time:
            time_metrics = ["avg_time", "min_time", "max_time"]
            for metric in time_metrics:
                values = [converted[metric] for converted in converted_stats.values()]
                data["extremes"][f"min_{metric}"] = min(values)
                data["extremes"][f"max_{metric}"] = max(values)

        # Memory extremes
        if config.memory:
            memory_metrics = ["avg_memory", "max_memory"]
            for metric in memory_metrics:
                values = [converted[metric] for converted in converted_stats.values()]
                data["extremes"][f"min_{metric}"] = min(values)
                data["extremes"][f"max_{metric}"] = max(values)

    def _format_title(self, config: BenchConfig) -> str:
        """Format the benchmark title."""
        return (
            f"\nBenchmark Results ({config.trials} trial"
            f"{'s' if config.trials > 1 else ''}):\n"
        )

    def _format_header(self, config: BenchConfig) -> str:
        """Format the header row."""
        memory_unit = self.formatting_data["memory_unit"]
        time_unit = self.formatting_data["time_unit"]
        column_widths = self.formatting_data["column_widths"]

        if config.trials == 1:
            header = "Function".ljust(self.max_name_len + 2)
            if config.time:
                header += f"Time ({time_unit})".rjust(column_widths["time"])
            if config.memory:
                header += f"Memory ({memory_unit})".rjust(column_widths["memory"])
        else:
            header = "Function".ljust(self.max_name_len + 2)
            if config.time:
                header += f"Avg Time ({time_unit})".rjust(column_widths["avg_time"])
                header += f"Min Time ({time_unit})".rjust(column_widths["min_time"])
                header += f"Max Time ({time_unit})".rjust(column_widths["max_time"])
            if config.memory:
                header += f"Avg Mem ({memory_unit})".rjust(column_widths["avg_memory"])
                header += f"Max Mem ({memory_unit})".rjust(column_widths["max_memory"])
        return header

    def _calculate_dash_length(self, config: BenchConfig) -> int:
        """Calculate dash line length."""
        column_widths = self.formatting_data["column_widths"]

        total_width = self.max_name_len + 2

        if config.trials == 1:
            if config.time:
                total_width += column_widths["time"]
            if config.memory:
                total_width += column_widths["memory"]
        else:
            if config.time:
                total_width += (
                    column_widths["avg_time"]
                    + column_widths["min_time"]
                    + column_widths["max_time"]
                )
            if config.memory:
                total_width += column_widths["avg_memory"] + column_widths["max_memory"]

        return total_width

    def _format_single_trial_results(
        self,
        output: list[str],
        config: BenchConfig,
    ) -> None:
        """Format results for a single trial benchmark."""
        converted_stats = self.formatting_data["converted_stats"]
        column_widths = self.formatting_data["column_widths"]

        for method_name in self.sorted_methods:
            converted = converted_stats[method_name]
            line = visual_ljust(method_name, self.max_name_len + 2)

            if config.time:
                time_val = f"{converted['avg_time']:.{self.precision}f}".rjust(
                    column_widths["time"],
                )
                line += time_val

            if config.memory:
                mem_val = f"{converted['avg_memory']:.{self.precision}f}".rjust(
                    column_widths["memory"],
                )
                line += mem_val

            output.append(line)

    def _format_multiple_trial_results(
        self,
        output: list[str],
        config: BenchConfig,
    ) -> None:
        """Format results for multiple trial benchmarks."""
        converted_stats = self.formatting_data["converted_stats"]
        column_widths = self.formatting_data["column_widths"]
        extremes = self.formatting_data["extremes"]

        color = config.color if len(converted_stats) > 1 else False

        for method_name in self.sorted_methods:
            converted = converted_stats[method_name]
            line = visual_ljust(method_name, self.max_name_len + 2)

            if config.time:
                # Format time values with coloring
                avg_val = self._format_metric(
                    converted["avg_time"],
                    extremes["min_avg_time"],
                    extremes["max_avg_time"],
                    width=column_widths["avg_time"],
                    color=color,
                )
                min_val = self._format_metric(
                    converted["min_time"],
                    extremes["min_min_time"],
                    extremes["max_min_time"],
                    width=column_widths["min_time"],
                    color=color,
                )
                max_val = self._format_metric(
                    converted["max_time"],
                    extremes["min_max_time"],
                    extremes["max_max_time"],
                    width=column_widths["max_time"],
                    color=color,
                )

                line += (
                    avg_val.rjust(column_widths["avg_time"])
                    + min_val.rjust(column_widths["min_time"])
                    + max_val.rjust(column_widths["max_time"])
                )

            if config.memory:
                # Format memory values with coloring
                avg_mem = self._format_metric(
                    converted["avg_memory"],
                    extremes["min_avg_memory"],
                    extremes["max_avg_memory"],
                    width=column_widths["avg_memory"],
                    color=color,
                )
                peak_mem = self._format_metric(
                    converted["max_memory"],
                    extremes["min_max_memory"],
                    extremes["max_max_memory"],
                    width=column_widths["max_memory"],
                    color=color,
                )
                line += avg_mem.rjust(column_widths["avg_memory"]) + peak_mem.rjust(
                    column_widths["max_memory"],
                )

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
        width: int,
        *,
        color: bool = True,
    ) -> str:
        """
        Format a metric value with appropriate coloring.

        Args:
            value: The value to format
            min_value: The minimum value across all benchmarks
            max_value: The maximum value across all benchmarks
            width: The width to format the value to
            color: Whether to use colored output

        Returns:
            Formatted string with color codes if color is True

        """
        # Define ANSI color codes
        min_color = "\033[32m"  # GREEN
        max_color = "\033[31m"  # RED
        reset = "\033[0m"
        formatted = f"{value:.{self.precision}f}".rjust(width)

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
        _ = results  # unused but avoids ARG002
        output = StringIO()
        writer = csv.writer(output)

        # Write header row
        header = self._create_header_row(config)
        writer.writerow(header)

        # Write data rows
        for method_name in self.sort_keys(stats, config):
            row = self._create_data_row(method_name, stats[method_name], config)
            writer.writerow(row)

        return output.getvalue()

    def _create_header_row(self, config: BenchConfig) -> list[str]:
        """Create the CSV header row based on configuration."""
        memory_unit = MemoryUnit.from_config(config)
        time_unit = TimeUnit.from_config(config)

        header = ["Function"]

        if config.trials == 1:
            if config.time:
                header.append(f"Time ({time_unit})")
            if config.memory:
                header.append(f"Memory ({memory_unit})")
        else:
            if config.time:
                header.extend(
                    [
                        f"Avg Time ({time_unit})",
                        f"Min Time ({time_unit})",
                        f"Max Time ({time_unit})",
                    ],
                )
            if config.memory:
                header.extend(
                    [
                        f"Avg Memory ({memory_unit})",
                        f"Max Memory ({memory_unit})",
                    ],
                )

        return header

    def _create_data_row(
        self,
        method_name: str,
        stat: StatType,
        config: BenchConfig,
    ) -> list[str | float]:
        """Create a data row for the given method."""
        memory_unit = MemoryUnit.from_config(config)
        time_unit = TimeUnit.from_config(config)

        if config.trials == 1:
            row: list[str | float] = [method_name]
            if config.time:
                row.append(time_unit.convert_seconds(stat["avg"]))
            if config.memory:
                row.append(memory_unit.convert_bytes(stat["avg_memory"]))
        else:
            row = [method_name]
            if config.time:
                row.extend(
                    [
                        time_unit.convert_seconds(stat["avg"]),
                        time_unit.convert_seconds(stat["min"]),
                        time_unit.convert_seconds(stat["max"]),
                    ],
                )
            if config.memory:
                row.extend(
                    [
                        memory_unit.convert_bytes(stat["avg_memory"]),
                        memory_unit.convert_bytes(stat["max_memory"]),
                    ],
                )

        return row


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
        time_unit = TimeUnit.from_config(config)
        self.shrink_data(results, stats, config)
        output_data: dict[str, Any] = {
            "config": config.model_dump(exclude={"reporters", "progress"}),
            "stats": {},
            "results": results,
        }

        for method_name in self.sort_keys(stats, config):
            stat = stats[method_name].copy()

            # Convert time values to the specified unit
            if config.time:
                for key in ["avg", "min", "max"]:
                    if key in stat:
                        stat[key] = time_unit.convert_seconds(stat[key])  # type: ignore [literal-required]

            # Convert memory values to the specified unit
            if config.memory:
                if "avg_memory" in stat:
                    stat["avg_memory"] = memory_unit.convert_bytes(stat["avg_memory"])
                if "max_memory" in stat:
                    stat["max_memory"] = memory_unit.convert_bytes(stat["max_memory"])

            output_data["stats"][method_name] = stat

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
        time_unit = TimeUnit.from_config(config)
        try:
            import pandas as pd  # noqa: PLC0415
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
                if config.time:
                    row[f"Time ({time_unit})"] = time_unit.convert_seconds(stat["avg"])
                if config.memory:
                    row[f"Memory ({memory_unit})"] = memory_unit.convert_bytes(
                        stat["avg_memory"],
                    )
            else:
                if config.time:
                    row.update(
                        {
                            f"Avg Time ({time_unit})": time_unit.convert_seconds(
                                stat["avg"],
                            ),
                            f"Min Time ({time_unit})": time_unit.convert_seconds(
                                stat["min"],
                            ),
                            f"Max Time ({time_unit})": time_unit.convert_seconds(
                                stat["max"],
                            ),
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
        time_unit = TimeUnit.from_config(config)
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
                # Convert if it's a time metric
                elif self.metric in ("avg", "min", "max"):
                    value = time_unit.convert_seconds(value)

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
        self.report_formatted(formatted)

    @abstractmethod
    def report_formatted(self, formatted_output: Formatted) -> None:
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

    def report_formatted(self, formatted_output: Formatted) -> None:
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

    def report_formatted(self, formatted_output: Formatted) -> None:
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

    def report_formatted(self, formatted_output: Formatted) -> None:
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
