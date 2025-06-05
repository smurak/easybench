"""
Core functionality for the EasyBench benchmark library.

This module provides classes and functions for running and measuring performance
benchmarks with support for fixtures, memory tracking, and convenient result reporting.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
import tracemalloc
import types
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict


from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

from .reporters import (
    ConsoleReporter,
    Reporter,
)

# Configure logger
logger = logging.getLogger(__name__)

# Define scope type for better type hinting
ScopeType = Literal["trial", "function", "class"]

# Sort type
SortType = Literal["def", "avg", "min", "max", "avg_memory", "peak_memory"]

# Generic types
T = TypeVar("T")
V = TypeVar("V")


class ResultType(TypedDict):
    """Type of benchmark result."""

    times: list[float]
    memory: NotRequired[list[float]]
    output: NotRequired[list[object]]


ResultsType: TypeAlias = dict[str, ResultType]
FixtureRegistry: TypeAlias = dict[ScopeType, dict[str, object]]


class PartialBenchConfig(BaseModel):
    """Partial configuration for EasyBench with optional values."""

    model_config = {"arbitrary_types_allowed": True}

    trials: int | None = None
    sort_by: SortType | None = None
    reverse: bool | None = None
    memory: bool | None = None
    color: bool | None = None
    show_output: bool | None = None
    return_output: bool | None = None
    reporters: list[Reporter] | None = None

    def merge_with(self, config: BenchConfig) -> BenchConfig:
        """
        Merge partial configuration with a complete configuration.

        Args:
            config: Complete configuration to use as base

        Returns:
            A complete BenchConfig with non-None values from this partial config

        """
        result = config.model_copy(deep=True)

        # Update non-None fields from partial config
        for field_name, field_value in self.model_dump().items():
            if field_value is not None:
                setattr(result, field_name, field_value)

        return result


class BenchConfig(PartialBenchConfig):
    """Complete configuration for EasyBench with required values."""

    trials: int = 5
    sort_by: SortType = "def"
    reverse: bool = False
    memory: bool = False
    color: bool = True
    show_output: bool = False
    return_output: bool = False
    reporters: list[Reporter] = [ConsoleReporter()]


def ensure_full_config(
    config: PartialBenchConfig | None,
    base_config: BenchConfig,
) -> BenchConfig:
    """
    Ensure we have a full BenchConfig.

    This allows both BenchConfig and PartialBenchConfig to be passed to methods
    that require configuration.

    Args:
        config: The configuration to process
        base_config: The base configuration to use if config is None or partial

    Returns:
        A complete BenchConfig instance

    """
    if config is None:
        return base_config.model_copy(deep=True)

    if isinstance(config, BenchConfig):
        return config

    # Must be a PartialBenchConfig
    return config.merge_with(base_config)


# Store fixture objects by scope
_fixture_registry: FixtureRegistry = {
    "trial": {},  # Run once per trial
    "function": {},  # Run once per function
    "class": {},  # Run once per class
}


def fixture(
    scope: ScopeType = "trial",
    fixture_registry: FixtureRegistry | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Define a fixture function.

    Args:
        scope: Lifecycle scope of the fixture. Valid values are:
               "trial": Run once per trial (default)
               "function": Run once per function
               "class": Run once per class
        fixture_registry: Registry to store fixtures in

    Returns:
        Decorator function that registers the fixture

    """
    if fixture_registry is None:
        fixture_registry = _fixture_registry

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if scope not in fixture_registry:
            fixture_registry[scope] = {}
        fixture_registry[scope][func.__name__] = func
        return func

    return decorator


class EasyBench:
    """Base class for benchmark classes."""

    # Default Benchmark Config
    bench_config = BenchConfig()

    def __init__(self, bench_config: BenchConfig | None = None) -> None:
        """
        Initialize the benchmark class with optional configuration.

        Args:
            bench_config: Configuration for the benchmark

        """
        # [IMPORTANT!] This init must be idempotent!
        if bench_config is not None:
            self.bench_config = bench_config
        else:
            self.bench_config = self.__class__.bench_config.model_copy(deep=True)

    def __init_subclass__(cls, **kwargs: object) -> None:
        """
        Handle subclass initialization to ensure proper configuration.

        Args:
            **kwargs: Additional keyword arguments

        """
        super().__init_subclass__(**kwargs)

        if cls.__init__ is not EasyBench.__init__:
            original_init = cls.__init__

            def safe_init(self: EasyBench, *args: object, **kwargs: object) -> None:
                config = cast("BenchConfig", kwargs.get("bench_config"))
                EasyBench.__init__(
                    self,
                    bench_config=config,
                )
                original_init(self, *args, **kwargs)  # type: ignore [arg-type]

            cls.__init__ = safe_init  # type: ignore [method-assign]

    # Default empty implementations of special methods
    def setup_class(self) -> None:
        """Set up resources before all benchmarks in the class."""

    def teardown_class(self) -> None:
        """Teardown method called once after all benchmarks in the class."""

    def setup_function(self) -> None:
        """Set up resources before each benchmark function."""

    def teardown_function(self) -> None:
        """Teardown method called after each benchmark function."""

    def setup_trial(self) -> None:
        """Set up resources before each trial execution."""

    def teardown_trial(self) -> None:
        """Teardown method called after each trial execution."""

    def _initialize_bench_params(
        self,
        config: PartialBenchConfig | None = None,
        fixture_registry: FixtureRegistry | None = None,
    ) -> tuple[
        BenchConfig,
        FixtureRegistry,
    ]:
        """
        Initialize benchmark parameters with defaults from config if not provided.

        Args:
            config: Configuration with optional parameters, can be complete or partial
            fixture_registry: Registry containing fixtures to use for the benchmarks

        Returns:
            Tuple of complete BenchConfig and fixture registry

        """
        # Create a complete config
        complete_config = ensure_full_config(config, self.bench_config)

        if fixture_registry is None:
            fixture_registry = _fixture_registry

        # If sorting by memory metrics is requested,
        # ensure memory measurement is enabled
        if (
            complete_config.sort_by in ("avg_memory", "peak_memory")
            and not complete_config.memory
        ):
            complete_config.memory = True
            logger.info(
                "Note: Enabled memory measurement because sort_by='%s' was specified",
                complete_config.sort_by,
            )

        return complete_config, fixture_registry

    def _run_benchmarks(
        self,
        config: BenchConfig,
        fixture_registry: FixtureRegistry,
    ) -> ResultsType:
        """
        Execute all benchmark methods for the specified number of trials.

        Args:
            config: Benchmark configuration
            fixture_registry: Registry containing fixtures

        Returns:
            Dictionary mapping benchmark names to their results

        """
        benchmark_methods = self._discover_benchmark_methods()
        results: ResultsType = {}
        values: dict[str, object] = {}

        capture_output = config.show_output or config.return_output

        with self._manage_scope("class", values, fixture_registry):
            for method_name, method in benchmark_methods.items():
                result_dict: ResultType = {"times": []}
                if config.memory:
                    result_dict["memory"] = []
                if capture_output:
                    result_dict["output"] = []

                results[method_name] = result_dict

                with self._manage_scope("function", values, fixture_registry):
                    for _ in range(config.trials):
                        with self._manage_scope("trial", values, fixture_registry):
                            # Run the benchmark and record the time, memory, and result
                            execution_time, memory_usage, func_result = (
                                self._run_single_benchmark(
                                    method=method,
                                    values=values,
                                    memory=config.memory,
                                    capture_output=capture_output,
                                )
                            )

                            results[method_name]["times"].append(execution_time)

                            if config.memory and memory_usage is not None:
                                results[method_name]["memory"].append(memory_usage)

                            if capture_output:
                                results[method_name]["output"].append(func_result)

        return results

    def bench(
        self,
        config: PartialBenchConfig | None = None,
        fixture_registry: FixtureRegistry | None = None,
        **kwargs: object,
    ) -> ResultsType:
        """
        Run all benchmark methods for the specified number of trials.

        Args:
            config: Configuration for the benchmark, can be complete or partial
            fixture_registry: Registry containing fixtures to use for the benchmarks
            **kwargs: Legacy keyword arguments for backward compatibility

        Returns:
            Dictionary mapping benchmark names to their results

        """
        # Support legacy keyword arguments
        if kwargs and not config:
            try:
                config = PartialBenchConfig(**kwargs)  # type: ignore [arg-type]
            except TypeError as e:
                msg = f"Invalid keywords: {e}"
                raise ValueError(msg) from e
        elif kwargs:
            logger.warning(
                "Both config and keyword arguments provided. "
                "Using config and ignoring keyword arguments.",
            )

        # Initialize parameters
        complete_config, fixture_registry = self._initialize_bench_params(
            config,
            fixture_registry,
        )

        # Run all benchmarks
        results = self._run_benchmarks(
            config=complete_config,
            fixture_registry=fixture_registry,
        )

        # Display results using reporters
        self._display_results(
            results=results,
            config=complete_config,
        )

        return results

    class ScopeManager:
        """Context manager for handling benchmark scopes."""

        def __init__(
            self,
            bench_instance: EasyBench,
            scope: ScopeType,
            values: dict[str, object],
            fixture_registry: FixtureRegistry,
        ) -> None:
            """
            Initialize scope manager.

            Args:
                bench_instance: The benchmark instance
                scope: The scope type (trial, function, class)
                values: Dictionary to store fixture values
                fixture_registry: Registry containing fixtures

            """
            self.bench_instance = bench_instance
            self.scope = scope
            self.values = values
            self.fixture_registry = fixture_registry
            self.generators: list[types.GeneratorType] = []

        def __enter__(self) -> EasyBench.ScopeManager:
            """
            Set up resources for the scope.

            Returns:
                Self for context manager protocol

            """
            match self.scope:
                case "class":
                    self.bench_instance.setup_class()
                case "function":
                    self.bench_instance.setup_function()
                case "trial":
                    self.bench_instance.setup_trial()
                case _:
                    scope_err = f"Invalid scope: {self.scope}"
                    raise ValueError(scope_err)

            # Set up fixtures
            fixtures = self.fixture_registry[self.scope]
            self.generators, _values = self.setup_fixtures(fixtures)
            self.values.update(_values)
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: object,
        ) -> None:
            """
            Clean up resources when exiting the context.

            Args:
                exc_type: Exception type if an exception was raised
                exc_val: Exception value if an exception was raised
                exc_tb: Exception traceback if an exception was raised

            """
            # Clean up fixtures
            self.teardown_fixtures(self.generators)
            match self.scope:
                case "class":
                    self.bench_instance.teardown_class()
                case "function":
                    self.bench_instance.teardown_function()
                case "trial":
                    self.bench_instance.teardown_trial()
                case _:
                    scope_err = f"Invalid scope: {self.scope}"
                    raise ValueError(scope_err)

        def setup_fixtures(
            self,
            fixtures: dict[str, object],
        ) -> tuple[list[types.GeneratorType], dict[str, object]]:
            """
            Set up fixtures for a given scope.

            Args:
                fixtures: The fixtures to set up

            Returns:
                Tuple of teardown generators and fixture values

            """
            teardown_generators = []
            values = {}

            # Process each fixture item
            try:
                for name, obj in fixtures.items():
                    result = obj() if callable(obj) else obj
                    # Check if the fixture function returned a generator (used yield)
                    if isinstance(result, types.GeneratorType):
                        # Handle generator-based fixtures (with yield)
                        value = next(result)
                        teardown_generators.append(result)
                    else:
                        # Handle return-based fixtures
                        value = result
                    values[name] = value
            except (TypeError, ValueError, RuntimeError) as error:
                self.teardown_fixtures(teardown_generators)
                error_msg = f"Error setting up fixture '{name}'"
                raise RuntimeError(error_msg) from error

            return teardown_generators, values

        def teardown_fixtures(self, generators: list[types.GeneratorType]) -> None:
            """
            Tear down fixtures.

            Args:
                generators: List of generators to teardown

            """
            # Move try-except outside the loop to avoid performance overhead
            errors = []
            for gen in generators:
                try:
                    next(gen)  # This should raise StopIteration
                except StopIteration:  # noqa: PERF203
                    # Expected - generator is exhausted
                    pass
                except (RuntimeError, ValueError) as e:
                    # Log error but continue cleanup
                    errors.append((gen, e))

            # Report errors after loop completes
            for gen, error in errors:
                logger.warning(
                    "Error during teardown of fixture '%s': %s",
                    gen,
                    str(error),
                )

    def _manage_scope(
        self,
        scope: ScopeType,
        values: dict[str, object],
        fixture_registry: FixtureRegistry,
    ) -> AbstractContextManager:
        """
        Context manager for setting up and tearing down the benchmark class.

        Args:
            scope: Scope of the fixtures to manage
            values: Dictionary to store fixture values
            fixture_registry: Registry containing fixtures

        Returns:
            A context manager for the specified scope

        """
        return self.ScopeManager(self, scope, values, fixture_registry)

    def _run_single_benchmark(
        self,
        *,
        method: Callable[..., object],
        values: dict[str, object],
        memory: bool = False,
        capture_output: bool = False,
    ) -> tuple[float, float | None, object | None]:
        """
        Run a single benchmark method with the required fixtures.

        Args:
            method: The benchmark method to run
            values: Dictionary containing fixture values
            memory: Whether to measure memory usage
            capture_output: Whether to capture and return function result

        Returns:
            A tuple containing:
            - execution time in seconds
            - memory usage in bytes (None if not measured)
            - function result (None if not captured)

        """
        # Extract the fixtures needed by this method
        required_fixtures = self._get_required_fixtures(method)
        fixture_args = {
            name: values[name] for name in required_fixtures if name in values
        }

        # Run the benchmark
        memory_usage = None
        result = None

        if memory:
            # Reset tracemalloc state
            if tracemalloc.is_tracing():
                tracemalloc.stop()

            tracemalloc.start()
            before_current, _ = tracemalloc.get_traced_memory()

            start_time = time.perf_counter()
            if capture_output:
                result = method(**fixture_args)
            else:
                method(**fixture_args)
            end_time = time.perf_counter()

            # get memory usage
            _, after_peak = tracemalloc.get_traced_memory()
            memory_usage = after_peak - before_current

            tracemalloc.stop()
        else:
            start_time = time.perf_counter()
            if capture_output:
                result = method(**fixture_args)
            else:
                method(**fixture_args)
            end_time = time.perf_counter()

        return end_time - start_time, memory_usage, result

    def _discover_benchmark_methods(self) -> dict[str, Callable[..., object]]:
        """
        Find all callable attributes in the class that start with 'bench_'.

        Returns:
            Dictionary mapping benchmark names to method objects

        """
        benchmark_methods = {}

        # Find all attributes that are callable and start with 'bench_'
        for name in self.__class__.__dict__:
            if name.startswith("bench_"):
                attr = getattr(self, name)
                if callable(attr):
                    benchmark_methods[name] = attr

        return benchmark_methods

    def _get_required_fixtures(self, method: Callable[..., object]) -> list[str]:
        """
        Determine which fixtures are required by a method.

        Args:
            method: The method to inspect

        Returns:
            List of parameter names required by the method

        """
        sig = inspect.signature(method)
        return list(sig.parameters.keys())

    def _display_results(
        self,
        results: ResultsType,
        config: BenchConfig,
    ) -> None:
        """
        Display benchmark results using configured reporters.

        Args:
            results: Dictionary mapping benchmark names to result data
            config: Benchmark configuration

        """
        if not results:
            logger.info("\nNo benchmark results to display.")
            return

        # Calculate statistics
        stats = self._calculate_statistics(
            results,
            trials=config.trials,
            memory=config.memory,
        )

        # Use each reporter to report the results
        for reporter in config.reporters:
            reporter.report(
                results=results,
                stats=stats,
                config=config,
            )

    def _calculate_statistics(
        self,
        results: ResultsType,
        *,
        trials: int,
        memory: bool,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate statistics from benchmark results.

        Args:
            results: Dictionary of benchmark results
            trials: Number of trials executed
            memory: Whether memory usage was measured

        Returns:
            Dictionary of calculated statistics

        """
        stats = {}
        for method_name, data in results.items():
            times = data["times"]

            if trials == 1:
                # For single trial, just store the single value
                stats[method_name] = {"time": times[0]}
                if memory:
                    stats[method_name]["memory"] = (
                        data["memory"][0] / 1024
                    )  # Convert to KB
            else:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                stats[method_name] = {"avg": avg_time, "min": min_time, "max": max_time}

                if memory:
                    memory_values = data["memory"]
                    avg_memory = sum(memory_values) / len(memory_values) / 1024
                    peak_memory = max(memory_values) / 1024
                    stats[method_name].update(
                        {"avg_memory": avg_memory, "peak_memory": peak_memory},
                    )

        return stats


class FunctionBench(EasyBench):
    """Wrapper class to run function-based benchmarks."""

    def __init__(
        self,
        func: Callable[..., object],
        func_name: str | None = None,
        bench_config: BenchConfig | None = None,
    ) -> None:
        """
        Initialize a function benchmark.

        Args:
            func: The function to benchmark
            func_name: Name to use for the function (defaults to func.__name__)
            bench_config: Configuration for the benchmark

        """
        super().__init__(bench_config=bench_config)

        if not callable(func):
            error_msg = "func must be callable"
            raise TypeError(error_msg)

        if func_name is None:
            func_name = getattr(func, "__name__", None)
            if func_name == "<lambda>" or func_name is None:
                error_msg = (
                    "func_name must be specified for lambda or unnamed functions"
                )
                raise ValueError(error_msg)

        # Store the original function and its name
        self._original_func = func
        self._func_name = func_name

        # Add the function directly as a method with bench_ prefix
        setattr(self, f"bench_{func_name}", func)

        # Copy the original function's signature and docstring to __call__
        trials_param = inspect.Parameter(
            "bench_trials",
            inspect.Parameter.KEYWORD_ONLY,
            default=1,
        )
        self.__signature__ = inspect.Signature(
            [
                *inspect.signature(func).parameters.values(),
                trials_param,
            ],
        )
        self.__doc__ = func.__doc__

    def _discover_benchmark_methods(self) -> dict[str, Callable[..., object]]:
        """
        Find all callable attributes in the class that start with 'bench_'.

        Returns:
            Dictionary mapping benchmark names to method objects

        """
        benchmark_methods = {}

        # Find all attributes that are callable and start with 'bench_'
        for name in dir(self):
            if name.startswith("bench_"):
                attr = getattr(self, name)
                if callable(attr):
                    # remove 'bench_' prefix
                    benchmark_methods[name.removeprefix("bench_")] = attr

        return benchmark_methods

    def __call__(
        self,
        *args: object,
        bench_trials: int = 1,
        **kwargs: object,
    ) -> object:
        """
        Call the benchmarked function with the given arguments.

        Args:
            *args: Positional arguments to pass to the function
            bench_trials: Number of benchmark trials to run
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The return value of the benchmarked function

        """
        # Convert positional arguments to keyword arguments
        sig = inspect.signature(self._original_func)
        param_names = list(sig.parameters.keys())

        for i, arg in enumerate(args):
            if i < len(param_names):
                kwargs[param_names[i]] = arg

        # Register parameters as fixtures
        fixture_registry: FixtureRegistry = {"trial": {}, "function": {}, "class": {}}
        for name, value in kwargs.items():
            fixture_registry["trial"][name] = lambda v=value: v

        # Configure to show results and always return them
        original_return_output = self.bench_config.return_output

        self.bench_config.return_output = True

        # Run the benchmark and get results
        results = self.bench(trials=bench_trials, fixture_registry=fixture_registry)

        # Restore original config
        self.bench_config.return_output = original_return_output

        # Extract the return value from the results
        func_name = self._func_name
        if bench_trials > 0 and func_name in results and "output" in results[func_name]:
            return results[func_name]["output"][0]

        return None
