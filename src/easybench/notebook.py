"""IPython/Jupyter Notebook magic commands for EasyBench."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from IPython.core.magic import Magics, cell_magic, magics_class

from .cli import create_parser, parse_args_to_config
from .core import (
    BenchConfig,
    EasyBench,
    ensure_full_config,
    measure_execution,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from IPython.core.interactiveshell import ExecutionResult, InteractiveShell

    from .utils import ResultType


@magics_class
class EasyBenchMagics(Magics):
    """Magics for the EasyBench library."""

    @cell_magic
    def easybench(self, line: str, cell: str) -> None:
        """
        Run the cell code as a benchmark and display performance metrics.

        Usage:
            %%easybench [options]
            <your code>

        Options:
            --trials=N        Number of trials to run (default: 1)
            --memory          Enable memory measurement
            --memory-unit=UNIT Memory unit (B/KB/MB/GB)
            --warmups=N       Number of warmup runs (default: 0)
            --loops-per-trial=N  Loops per trial (default: 1)
            --clip-outliers=FLOAT Clip outliers percentage (0.0-1.0)
            --time-unit=UNIT  Time unit (s/ms/us/ns/m)
            --no-time         Disable time measurement

        Example:
            %%easybench --trials=5 --memory
            result = []
            for i in range(1000000):
                result.append(i)

        """
        # Return None if not in IPython environment
        if self.shell is None:
            return

        # Parse options
        config = self._parse_options(line)

        # Create a benchmark runner for cell execution
        bench_runner = CellBenchRunner(self.shell)

        # Execute benchmark
        bench_runner.run_cell_benchmark(cell, config)

    def _parse_options(self, line: str) -> BenchConfig:
        """Parse magic options into a BenchConfig object."""
        # Create a parser based on the CLI parser
        parser = create_parser()

        # Parse arguments
        try:
            args = parser.parse_args(line.split())
        except SystemExit:
            # Handle --help flag manually to avoid exiting
            return BenchConfig(trials=1)

        # Convert parsed args to config
        partial_config = parse_args_to_config(args)

        return ensure_full_config(partial_config, BenchConfig(trials=1))


class CellBenchRunner:
    """Helper class to run benchmarks on Jupyter cell code."""

    def __init__(self, shell: InteractiveShell) -> None:
        """Initialize with the IPython shell."""
        self.shell = shell

    def run_cell_benchmark(self, cell: str, config: BenchConfig) -> object:
        """Run benchmark on cell code and return results."""
        capture_output = config.show_output or config.return_output
        results: ResultType = {}

        if config.time:
            results["times"] = []
        if config.memory:
            results["memory"] = []
        if capture_output:
            results["output"] = []

        # Run the code for each trial (including warmups)
        for i in range(config.warmups + config.trials):
            is_warmup = i < config.warmups

            # Measure a single trial
            is_first = i == config.warmups
            execution_time, memory_usage, result = measure_execution(
                execution_func=cast(
                    "Callable[[], ExecutionResult]",
                    lambda is_first=is_first: self.shell.run_cell(
                        cell,
                        silent=not is_first,
                    ),
                ),
                measure_memory=bool(config.memory),
                loops=config.loops_per_trial,
            )
            output = result.result if result is not None else None

            # Record results (except during warmup)
            if not is_warmup:
                if config.time:
                    results["times"].append(execution_time)

                if config.memory and memory_usage is not None:
                    results["memory"].append(memory_usage)

                if config.return_output or config.show_output:
                    results["output"].append(output)

        # Create a benchmark instance to process and report results
        bench = EasyBench(config)

        # Add result to "Cell Code" key
        all_results = {"Cell Code": results}

        # Process results (apply outlier clipping if configured)
        processed_results = bench.process_results(all_results, config)

        # Report results
        bench.report_results(processed_results, config)

        return output


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register the EasyBenchMagics magics."""
    ipython.register_magics(EasyBenchMagics)
