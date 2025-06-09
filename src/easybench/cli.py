"""Command-line interface for running benchmarks with EasyBench."""

import argparse
import importlib
import inspect
import logging
import sys
import types
from pathlib import Path

from .core import BenchConfig, EasyBench, FunctionBench, PartialBenchConfig

# Configure logger
logger = logging.getLogger(__name__)


def discover_benchmark_files(path: str | Path = "benchmarks") -> list[Path]:
    """
    Discover benchmark files based on the input path.

    If path is a directory, find all Python files starting with 'bench_'.

    Args:
        path: Path to a benchmark file or directory containing benchmark files

    Returns:
        List of paths to benchmark files

    """
    path = Path(path)

    # If the path is a file
    if path.is_file():
        return [path]

    # If the path is a directory
    if path.is_dir():
        return sorted(path.glob("bench_*.py"))

    # If the path doesn't exist
    logger.error("Path not found: %s", path)
    return []


def load_benchmark_module(file_path: Path) -> types.ModuleType | None:
    """
    Load a Python module from a file path.

    Args:
        file_path: Path to the Python file to load

    Returns:
        The loaded module or None if loading fails

    """
    # Get the module name from the file path
    module_name = file_path.stem

    # Add the parent directory to sys.path if it's not there already
    parent_dir = str(file_path.parent.absolute())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        # Import the module
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, SyntaxError):
        logger.exception("Error loading %s", file_path)
        return None


def discover_benchmarks(
    module: types.ModuleType,
) -> dict[str, types.FunctionType | EasyBench]:
    """
    Discover all benchmarks in a module.

    Benchmarks can be either:
    - Functions that start with 'bench_'
    - Classes that inherit from EasyBench and start with 'Bench'

    Args:
        module: Module to inspect for benchmarks

    Returns:
        Dictionary mapping benchmark names to objects

    """
    # Find all benchmark functions (ones that start with 'bench_')
    benchmarks: dict[str, types.FunctionType | EasyBench] = {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("bench_")
    }

    # Find all benchmark classes
    # (ones that inherit from EasyBench and start with 'Bench')
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.startswith("Bench") and issubclass(obj, EasyBench) and obj != EasyBench:
            try:
                # Create an instance of the benchmark class
                benchmarks[name] = obj()
            except (TypeError, ValueError, RuntimeError):
                logger.exception("Error initializing %s", name)

    return benchmarks


def run_benchmarks(
    benchmarks: dict[str, types.FunctionType | EasyBench],
    config: PartialBenchConfig | None = None,
    source_id: str | None = None,
) -> None:
    """
    Run all discovered benchmarks.

    Args:
        benchmarks: Dictionary mapping benchmark names to objects
        config: Configuration for benchmarks, can be complete or partial
        source_id: Identifier for the benchmark source (e.g., file name)

    """
    # Print a separator line for better visual distinction
    if source_id:
        separator = "=" * 80
        print(  # noqa: T201
            f"\n{separator}\nRunning benchmarks from: {source_id}\n{separator}",
        )

    for name, benchmark in benchmarks.items():
        print(f"\n=== Running benchmark: {name} ===")  # noqa: T201

        try:
            if inspect.isfunction(benchmark):
                # Create a FunctionBench wrapper for the function
                function_bench = FunctionBench(benchmark, func_name=name)

                # Run the benchmark
                function_bench.bench(config=config)
            else:
                benchmark.bench(config=config)
        except (ValueError, TypeError, RuntimeError, AttributeError):
            logger.exception("Error running benchmark %s", name)


def cli_main() -> None:
    """
    Execute the main command line interface for easybench.

    Usage:
    easybench [`--trials N`] [`--memory`] [`--sort-by METRIC`] [`--reverse`]
    [`--show-output`] [`path`]
    """
    default_config = BenchConfig()
    parser = argparse.ArgumentParser(
        description="Run benchmarks in the specified file or directory",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="benchmarks",
        help=(
            "Benchmark file or directory containing benchmark files "
            "(default: benchmarks directory)"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Number of trials to run for each benchmark "
            f"(default: {default_config.trials})"
        ),
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Measure memory usage during benchmarks",
    )
    parser.add_argument(
        "--sort-by",
        choices=["def", "avg", "min", "max", "avg_memory", "max_memory"],
        help="Sort results by the specified metric",
    )
    parser.add_argument("--reverse", action="store_true", help="Reverse the sort order")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Display function return values",
    )

    args = parser.parse_args()

    try:
        # Create config from CLI arguments
        config = PartialBenchConfig(
            trials=args.trials,
            memory=args.memory or None,
            sort_by=args.sort_by,
            reverse=args.reverse or None,
            color=False if args.no_color else None,
            show_output=args.show_output or None,
        )

        # Discover benchmark files
        benchmark_files = discover_benchmark_files(args.path)
        if not benchmark_files:
            logger.error("No benchmark files found at %s", args.path)
            return

        logger.info("Found %d benchmark files:", len(benchmark_files))
        for file_path in benchmark_files:
            logger.info("  %s", file_path)

        # Load modules and run benchmarks
        for file_path in benchmark_files:
            logger.info("Processing %s", file_path)
            try:
                module = load_benchmark_module(file_path)
                if module:
                    benchmarks = discover_benchmarks(module)
                    if benchmarks:
                        try:
                            run_benchmarks(
                                benchmarks,
                                config=config,
                                source_id=str(file_path),
                            )
                        except (ValueError, TypeError):
                            logger.exception(
                                "Error running benchmarks in %s",
                                file_path,
                            )
                    else:
                        logger.info("No benchmarks found in %s", file_path)
                else:
                    logger.warning("Failed to load module from %s", file_path)
            except (ImportError, ModuleNotFoundError):
                logger.exception("Error processing %s", file_path)
    except Exception:
        logger.exception("Error in easybench CLI")


if __name__ == "__main__":
    # Set up basic configuration for logging when module is run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    cli_main()
