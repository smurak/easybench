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
from collections.abc import Callable, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
)

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict


from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from contextlib import AbstractContextManager

from .reporters import (
    ConsoleReporter,
    FileReporter,
    MemoryUnit,
    Reporter,
    SimpleConsoleReporter,
    TimeUnit,
)

# Generic types
T = TypeVar("T")
V = TypeVar("V")

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(  # type: ignore[no-redef]
        iterable: Iterable[T],
        **kwargs: object,
    ) -> Iterable[T]:
        """Create simple tqdm fallback."""
        _ = kwargs
        return iterable


# Configure logger
logger = logging.getLogger(__name__)

# Define scope type for better type hinting
ScopeType = Literal["trial", "function", "class"]

# Sort type
SortType = Literal["def", "avg", "min", "max", "avg_memory", "max_memory"]


def get_reporter(name: str, kwargs: dict | None = None) -> Reporter:
    """Convert string to reporter."""
    kwargs = kwargs or {}

    match name.lower():
        case "console":
            return ConsoleReporter(**kwargs)
        case "simple":
            return SimpleConsoleReporter(**kwargs)
        case "file":
            return FileReporter(**kwargs)
        case _ if name.endswith((".csv", ".json")):
            return FileReporter(name, **kwargs)
        case "plot" | "boxplot":
            from .visualization import BoxPlotFormatter, PlotReporter

            return PlotReporter(BoxPlotFormatter(**kwargs))
        case "lineplot":
            from .visualization import LinePlotFormatter, PlotReporter

            return PlotReporter(LinePlotFormatter(**kwargs))
        case _:
            err = f"Unknown reporter type: {name}"
            raise ValueError(err)


class ResultType(TypedDict):
    """Type of benchmark result."""

    times: list[float]
    memory: NotRequired[list[float]]
    output: NotRequired[list[object]]


class StatType(TypedDict):
    """Type of benchmark statistics."""

    avg: float
    min: float
    max: float
    avg_memory: NotRequired[float]
    max_memory: NotRequired[float]


ResultsType: TypeAlias = dict[str, ResultType]
StatsType: TypeAlias = dict[str, StatType]
FixtureRegistry: TypeAlias = dict[ScopeType, dict[str, object]]

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class ParametrizedFunction(Protocol[P, R_co]):
    """Function with _bench_params."""

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R_co:
        """Call method."""
        ...

    _bench_params: list[BenchParams]


class CustomizedFunction(Protocol[P, R_co]):
    """Function with _bench_customize."""

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R_co:
        """Call method."""
        ...

    _bench_customize: dict[str, Any]


class PartialBenchConfig(BaseModel):
    """Partial configuration for EasyBench with optional values."""

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    trials: int | None = None
    loops_per_trial: int | None = None
    warmups: int | None = None
    sort_by: SortType | None = None
    reverse: bool | None = None
    memory: bool | MemoryUnit | str | None = None
    time: TimeUnit | str | None = None
    color: bool | None = None
    show_output: bool | None = None
    return_output: bool | None = None
    reporters: list[Reporter] | None = None
    progress: bool | Callable | None = None

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

    @field_validator("reporters", mode="before")
    @classmethod
    def validate_and_convert_reporters(cls, v: list | None) -> list[Reporter] | None:
        """Validate and convert reporters."""
        if v is None:
            return None
        if not isinstance(v, list):
            msg = "reporters must be a list"
            raise TypeError(msg)

        converted_reporters: list[Reporter] = []
        for item in v:
            if isinstance(item, str):
                # Convert string to reporter
                reporter = get_reporter(item)
                converted_reporters.append(reporter)
            elif (
                isinstance(item, (tuple, list))
                and isinstance(item[0], str)
                and isinstance(item[1], dict)
            ):
                reporter = get_reporter(item[0], item[1])
                converted_reporters.append(reporter)
            elif isinstance(item, Reporter):
                # すでにReporterオブジェクトの場合はそのまま
                converted_reporters.append(item)
            else:
                msg = f"Invalid reporter type: {type(item)}"
                raise TypeError(msg)

        return converted_reporters

    @field_validator("loops_per_trial", mode="before")
    @classmethod
    def validate_loops_per_trial(cls, v: int | None) -> int | None:
        """Validate loops_per_trial."""
        if v is not None and v < 1:
            msg = "loops_per_trial must be at least 1"
            raise ValueError(msg)
        return v

    @field_validator("warmups", mode="before")
    @classmethod
    def validate_warmups(cls, v: int | None) -> int | None:
        """Validate warmups."""
        if v is not None and v < 0:
            msg = "warmups must be at least 0"
            raise ValueError(msg)
        return v


class BenchConfig(PartialBenchConfig):
    """Complete configuration for EasyBench with required values."""

    trials: int = 5
    loops_per_trial: int = 1
    warmups: int = 0
    sort_by: SortType = "def"
    reverse: bool = False
    memory: bool | MemoryUnit | str = False
    time: TimeUnit | str = TimeUnit.SECONDS
    color: bool = True
    show_output: bool = False
    return_output: bool = False
    reporters: list[Reporter] = [ConsoleReporter()]
    progress: bool | Callable = False


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


class BenchParams(BaseModel):
    """
    Class to store parameters for easybench decorators.

    This class allows grouping parameters for various easybench decorators together,
    making it easier to reuse parameter sets across multiple benchmarks.

    Attributes:
        name: Optional name for this parameter set (used for comparison display)
        params: Dictionary of parameters for @bench decorator
        fn_params: Dictionary of parameters for @bench.fn_params decorator

    Example:
        ```python
        params = BenchParams(
            name="Large dataset",
            params={"item": 123, "big_list": lambda: list(range(1_000_000))},
        )

        @bench(params)
        def add_item(item, big_list):
            big_list.append(item)
        ```

    """

    name: str | None = None
    params: dict[str, Any] = {}
    fn_params: dict[str, Any] = {}

    model_config = {
        "arbitrary_types_allowed": True,
    }


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


def parametrize(params_list: list[BenchParams]) -> Callable:
    """
    Create a decorator for parametrized benchmarks in EasyBench classes.

    Example:
        ```python
        params1 = BenchParams(name="small", params={"big_list": 1_000})
        params2 = BenchParams(name="big", params={"big_list": 100_000})

        class BenchList(EasyBench):
            @parametrize([params1, params2])
            def bench_append(self, big_list):
                big_list.append(0)
        ```

    Args:
        params_list: List of BenchParams instances with benchmark configurations

    Returns:
        A decorator function that marks the method for parametrized benchmarking

    """

    def decorator(func: Callable) -> Callable:
        func = cast("ParametrizedFunction", func)
        func._bench_params = params_list  # noqa: SLF001
        return func

    return decorator


def customize(*, loops_per_trial: int | None = None) -> Callable:
    """
    Create a decorator for customizing benchmark settings for specific methods.

    Example:
        ```python
        class BenchList(EasyBench):
            @customize(loops_per_trial=1000)
            def bench_append(self):
                self.big_list.append(0)
        ```

    Args:
        loops_per_trial: Number of loops per trial for this specific benchmark method

    Returns:
        A decorator function that applies custom benchmark settings to the method

    """

    def decorator(func: Callable) -> Callable:
        func = cast("CustomizedFunction", func)
        func._bench_customize = {"loops_per_trial": loops_per_trial}  # noqa: SLF001
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
            complete_config.sort_by in ("avg_memory", "max_memory")
            and not complete_config.memory
        ):
            complete_config.memory = True
            logger.info(
                "Note: Enabled memory measurement because sort_by='%s' was specified",
                complete_config.sort_by,
            )

        return complete_config, fixture_registry

    def _get_loops_per_trial(
        self,
        method: Callable[..., object],
        config: BenchConfig,
    ) -> int:
        """
        Get the number of loops per trial, considering method customization.

        Args:
            method: The benchmark method
            config: Benchmark configuration

        Returns:
            Number of loops per trial

        """
        loops_per_trial = config.loops_per_trial
        if hasattr(method, "_bench_customize"):
            method = cast("CustomizedFunction", method)
            custom_config = method._bench_customize  # noqa: SLF001
            if custom_config.get("loops_per_trial") is not None:
                loops_per_trial = custom_config["loops_per_trial"]
        return loops_per_trial

    def _create_trial_range(
        self,
        method: Callable[..., object],
        config: BenchConfig,
    ) -> range | Iterable:
        """
        Create a trial range with optional progress tracking.

        Args:
            method: The benchmark method
            config: Benchmark configuration

        Returns:
            Trial range

        """
        total_trials = config.warmups + config.trials

        if config.progress:
            progress_func = tqdm if config.progress is True else config.progress
            trial_range = progress_func(
                range(total_trials),
                desc=f"Method: {getattr(method, '__name__', 'unknown')}",
                total=total_trials,
            )
        else:
            trial_range = range(total_trials)

        return trial_range

    def _run_benchmark_trials(
        self,
        method: Callable[..., object],
        config: BenchConfig,
        fixture_registry: FixtureRegistry,
        values: dict[str, object],
    ) -> ResultType:
        """
        Run a single benchmark method for the specified number of trials.

        Args:
            method: The benchmark method to run
            config: Benchmark configuration
            fixture_registry: Registry containing fixtures
            values: Dictionary to store fixture values

        Returns:
            Dictionary containing benchmark results

        """
        capture_output = config.show_output or config.return_output
        result_dict: ResultType = {"times": []}
        if config.memory:
            result_dict["memory"] = []
        if capture_output:
            result_dict["output"] = []

        # Get customized loops per trial setting
        loops_per_trial = self._get_loops_per_trial(method, config)

        with self._manage_scope("function", values, fixture_registry):
            warmup = True
            # Create trial range with progress tracking if enabled
            trial_range = self._create_trial_range(method, config)

            for i in trial_range:
                if i == config.warmups:
                    warmup = False

                with self._manage_scope("trial", values, fixture_registry):
                    # Run the benchmark and record the time, memory, and result
                    execution_time, memory_usage, func_result = (
                        self._run_single_benchmark(
                            method=method,
                            values=values,
                            memory=bool(config.memory),
                            capture_output=capture_output,
                            loops_per_trial=loops_per_trial,
                        )
                    )

                    if not warmup:
                        result_dict["times"].append(execution_time)

                        if config.memory and memory_usage is not None:
                            result_dict["memory"].append(memory_usage)

                        if capture_output:
                            result_dict["output"].append(func_result)

        return result_dict

    def _process_parametrized_method(
        self,
        method_name: str,
        method: Callable[..., object],
        config: BenchConfig,
        fixture_registry: FixtureRegistry,
        values: dict[str, object],
    ) -> ResultsType:
        """
        Process a parametrized benchmark method with multiple parameter sets.

        Args:
            method_name: Name of the benchmark method
            method: The benchmark method to run
            config: Benchmark configuration
            fixture_registry: Registry containing fixtures
            values: Dictionary to store fixture values

        Returns:
            A dictionary of all results

        """
        all_results: ResultsType = {}

        # Get the parameter sets from the method
        params_list = getattr(method, "_bench_params", [])

        # Setup progress for parameter sets if enabled
        if config.progress:
            progress_func = tqdm if config.progress is True else config.progress
            param_iter = progress_func(
                enumerate(params_list),
                desc=f"Params for {method_name}",
                total=len(params_list),
            )
        else:
            param_iter = enumerate(params_list)

        # Run benchmarks for each parameter set
        for i, params in param_iter:
            # Create a name for this parameter set
            param_name = f"params_{i+1}"
            if params.name:
                param_name = params.name

            result_name = f"{method_name} ({param_name})"

            # Register parameter fixtures
            param_fixtures = {}

            # Apply function parameters
            if params.fn_params:
                for name, value in params.fn_params.items():
                    param_fixtures[name] = lambda v=value: v

            # Apply bench parameters
            if params.params:
                for name, value in params.params.items():
                    if callable(value) and not isinstance(value, type):
                        param_fixtures[name] = value
                    else:
                        param_fixtures[name] = lambda v=value: v

            # Save original fixtures to restore later
            original_fixtures = fixture_registry["trial"].copy()

            # Apply parameter fixtures
            fixture_registry["trial"].update(param_fixtures)

            # Run the benchmark for this parameter set
            all_results[result_name] = self._run_benchmark_trials(
                method=method,
                config=config,
                fixture_registry=fixture_registry,
                values=values,
            )

            # Restore original fixtures
            fixture_registry["trial"] = original_fixtures

        return all_results

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

        with self._manage_scope("class", values, fixture_registry):
            # Use progress bar if enabled
            if config.progress:
                # Get the progress function (tqdm or custom)
                progress_func = tqdm if config.progress is True else config.progress
                # Apply progress bar to methods
                method_items = progress_func(
                    benchmark_methods.items(),
                    desc="Benchmarking",
                    total=len(benchmark_methods),
                )
            else:
                method_items = benchmark_methods.items()

            for method_name, method in method_items:
                # Check if this method is parametrized
                if hasattr(method, "_bench_params"):
                    # Process parametrized method
                    param_results = self._process_parametrized_method(
                        method_name=method_name,
                        method=method,
                        config=config,
                        fixture_registry=fixture_registry,
                        values=values,
                    )
                    results.update(param_results)
                else:
                    # Regular (non-parametrized) method
                    results[method_name] = self._run_benchmark_trials(
                        method=method,
                        config=config,
                        fixture_registry=fixture_registry,
                        values=values,
                    )

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
        loops_per_trial: int = 1,
    ) -> tuple[float, float | None, object | None]:
        """
        Run a single benchmark method with the required fixtures.

        Args:
            method: The benchmark method to run
            values: Dictionary containing fixture values
            memory: Whether to measure memory usage
            capture_output: Whether to capture and return function result
            loops_per_trial: Number of loops per trial

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
            for i in range(loops_per_trial):
                if capture_output or i == 0:
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
            for i in range(loops_per_trial):
                if capture_output or i == 0:
                    result = method(**fixture_args)
                else:
                    method(**fixture_args)
            end_time = time.perf_counter()

        return (end_time - start_time) / loops_per_trial, memory_usage, result

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
        stats = self._calculate_statistics(results)

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
    ) -> StatsType:
        """
        Calculate statistics from benchmark results.

        Args:
            results: Dictionary of benchmark results

        Returns:
            Dictionary of calculated statistics

        """
        stats: StatsType = {}
        for method_name, data in results.items():
            times = data["times"]

            stats[method_name] = {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

            if "memory" in data:
                memory_values = data["memory"]
                avg_memory = sum(memory_values) / len(memory_values)
                max_memory = max(memory_values)
                stats[method_name].update(
                    {"avg_memory": avg_memory, "max_memory": max_memory},
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
