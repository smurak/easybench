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
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from .core import BenchConfig

T = TypeVar("T")


class Formatter(ABC):
    """Base class for all result formatters."""

    @abstractmethod
    def format(
        self,
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> str | pd.DataFrame:
        """
        Format benchmark results.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            sorted_methods: List of method names in sorted order
            config: Benchmark configuration
            max_name_len: Length of the longest method name

        Returns:
            Formatted results in the appropriate format

        """


class TableFormatter(Formatter):
    """Format results as text tables (current default format)."""

    def format(
        self,
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> str:
        """Format results as text tables."""
        output = []

        # Add title line
        output.append(self._format_title(config))

        # Format header
        header = self._format_header(config, max_name_len)
        output.append(header)

        # Add dash line
        dash_length = self._calculate_dash_length(config, max_name_len)
        output.append("-" * dash_length)

        # Format result lines
        if config.trials == 1:
            self._format_single_trial_results(
                output,
                sorted_methods,
                stats,
                config,
                max_name_len,
            )
        else:
            self._format_multiple_trial_results(
                output,
                sorted_methods,
                stats,
                config,
                max_name_len,
            )

        # Add function outputs if requested
        if config.show_output:
            self._format_output_section(output, sorted_methods, results, max_name_len)

        return "\n".join(output) + "\n"

    def _format_title(self, config: BenchConfig) -> str:
        """Format the benchmark title."""
        return (
            f"\nBenchmark Results ({config.trials} trial"
            f"{'s' if config.trials > 1 else ''}):\n"
        )

    def _format_header(self, config: BenchConfig, max_name_len: int) -> str:
        """Format the header row."""
        if config.trials == 1:
            header = f"{'Function':<{max_name_len + 2}} {'Time (s)':<11}"
            if config.memory:
                header += f" {'Memory (KB)'}"
        else:
            header = (
                f"{'Function':<{max_name_len + 2}} {'Avg Time (s)'} "
                f"{'Min Time (s)'} {'Max Time (s)'}"
            )
            if config.memory:
                header += f" {'Avg Mem (KB)'} {'Peak Mem (KB)'}"
        return header

    def _calculate_dash_length(self, config: BenchConfig, max_name_len: int) -> int:
        """Calculate dash line length."""
        if config.trials == 1:
            return max_name_len + (14 if not config.memory else 26)
        return max_name_len + (38 if not config.memory else 68)

    def _format_single_trial_results(
        self,
        output: list[str],
        sorted_methods: list[str],
        stats: dict[str, dict[str, float]],
        config: BenchConfig,
        max_name_len: int,
    ) -> None:
        """Format results for a single trial benchmark."""
        for method_name in sorted_methods:
            stat = stats[method_name]
            time_val = f"{stat['time']:.6f}".ljust(12)
            line = f"{method_name}".ljust(max_name_len + 2) + f" {time_val}"
            if config.memory:
                mem_val = f"{stat['memory']:.6f}".ljust(12)
                line += f" {mem_val}"
            output.append(line)

    def _format_multiple_trial_results(
        self,
        output: list[str],
        sorted_methods: list[str],
        stats: dict[str, dict[str, float]],
        config: BenchConfig,
        max_name_len: int,
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

        for method_name in sorted_methods:
            stat = stats[method_name]

            # Format values with appropriate coloring
            avg_val = self._format_metric(
                stat["avg"],
                min_avg,
                max_avg,
                use_color=config.use_color,
            )
            min_val = self._format_metric(
                stat["min"],
                min_min,
                max_min,
                use_color=config.use_color,
            )
            max_val = self._format_metric(
                stat["max"],
                min_max,
                max_max,
                use_color=config.use_color,
            )

            # Format the function name with proper alignment
            method_col = f"{method_name}".ljust(max_name_len + 2)
            line = f"{method_col} {avg_val} {min_val} {max_val}"

            if config.memory:
                # Format memory values with proper coloring
                avg_mem = self._format_metric(
                    stat["avg_memory"],
                    min_avg_memory,
                    max_avg_memory,
                    use_color=config.use_color,
                )
                peak_mem = self._format_metric(
                    stat["peak_memory"],
                    min_peak_memory,
                    max_peak_memory,
                    use_color=config.use_color,
                )
                line += f" {avg_mem} {peak_mem}"

            output.append(line)

    def _format_output_section(
        self,
        output: list[str],
        sorted_methods: list[str],
        results: dict[str, dict[str, Any]],
        max_name_len: int,
    ) -> None:
        """Format the output section."""
        output.append("\nBenchmark Return Values:")
        output.append("-" * max(max_name_len + 10, 30))

        for method_name in sorted_methods:
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
        use_color: bool = True,
    ) -> str:
        """
        Format a metric value with appropriate coloring.

        Args:
            value: The value to format
            min_value: The minimum value across all benchmarks
            max_value: The maximum value across all benchmarks
            use_color: Whether to use colored output

        Returns:
            Formatted string with color codes if use_color is True

        """
        # Define ANSI color codes
        min_color = "\033[32m"  # GREEN
        max_color = "\033[31m"  # RED
        reset = "\033[0m"
        formatted = f"{value:.6f}".ljust(12)

        if not use_color:
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
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> str:
        """Format results as CSV."""
        _, _ = results, max_name_len  # unused but avoids ARG002
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
        for method_name in sorted_methods:
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
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> str:
        """Format results as JSON."""
        _ = max_name_len  # unused but avoids ARG002
        output_data = {
            "config": {
                "trials": config.trials,
                "memory": config.memory,
                "sort_by": config.sort_by,
                "reverse": config.reverse,
            },
            "results": {},
        }

        for method_name in sorted_methods:
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
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> pd.DataFrame:
        """Format results as DataFrame."""
        _ = max_name_len  # unused but avoids ARG002
        try:
            import pandas as pd
        except ImportError as err:
            error_msg = (
                "pandas is required for DataFrame output. "
                "Install with pip install pandas."
            )
            raise ImportError(error_msg) from err

        # Using max_name_len for consistent method signature, even if not used here
        data = []
        for method_name in sorted_methods:
            stat = stats[method_name]
            row = {"Function": method_name}

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
                row["Output"] = results[method_name]["output"][
                    0
                ]  # Just take first output

            data.append(row)

        return pd.DataFrame(data)


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
        results: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, float]],
        sorted_methods: list[str],
        config: BenchConfig,
        max_name_len: int,
    ) -> None:
        """
        Report benchmark results.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            sorted_methods: List of method names in sorted order
            config: Benchmark configuration
            max_name_len: Length of the longest method name

        """
        formatted = self.formatter.format(
            results=results,
            stats=stats,
            sorted_methods=sorted_methods,
            config=config,
            max_name_len=max_name_len,
        )
        self._send(formatted)

    def _send(self, formatted_output: str | pd.DataFrame) -> None:
        """
        Send formatted output to the destination.

        Args:
            formatted_output: The formatted output to send

        """
        # Base implementation does nothing


class ConsoleReporter(Reporter):
    """Reporter that sends output to the console."""

    def __init__(
        self,
        formatter: Formatter | None = None,
        file: TextIO | None = None,
    ) -> None:
        """
        Initialize console reporter.

        Args:
            formatter: Formatter to use (defaults to TableFormatter)
            file: File object to write to (defaults to sys.stdout)

        """
        super().__init__(formatter or TableFormatter())
        self.file = file

    def _send(self, formatted_output: str | pd.DataFrame) -> None:
        """
        Print formatted output to console.

        Args:
            formatted_output: The formatted output to print

        """
        _file = self.file or sys.stdout
        print(formatted_output, file=_file, end="")


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

    def _send(self, formatted_output: str | pd.DataFrame) -> None:
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
        callback: Callable[[str | pd.DataFrame], None],
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

    def _send(self, formatted_output: str | pd.DataFrame) -> None:
        """
        Send formatted output to callback.

        Args:
            formatted_output: The formatted output to send

        """
        self.callback(formatted_output)
