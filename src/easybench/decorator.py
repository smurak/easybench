"""
Benchmark decorator utilities for easier function benchmarking.

This module provides decorators for benchmarking functions with configurable
parameters and settings. It allows for easy setup and running of benchmarks.
"""

import inspect
from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload

from .core import (
    BenchConfig,
    BenchParams,
    EasyBench,
    FixtureRegistry,
    FunctionBench,
    PartialBenchConfig,
    ResultsType,
    ResultType,
    SortType,
)
from .reporters import ConsoleReporter, MemoryUnit, Reporter, TimeUnit

# Type for decorated functions
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class BenchmarkableFunction(Protocol[P, R_co]):
    """Function with bench."""

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R_co:
        """Call method."""
        ...

    bench: FunctionBench
    fixture_registry: FixtureRegistry
    last_results: ResultsType


class BenchDecorator:
    """Decorator for benchmarking functions."""

    def _initialize_bench_attributes(self, func: Callable) -> BenchmarkableFunction:
        """
        Initialize benchmark attributes on a function if they don't exist.

        Args:
            func: The function to initialize benchmark attributes for

        Returns:
            The function cast as BenchmarkableFunction with all necessary attributes

        """
        func = cast("BenchmarkableFunction", func)
        if not hasattr(func, "bench"):
            func.bench = FunctionBench(func)
            func.fixture_registry = {
                "trial": {},
                "function": {},
                "class": {},
            }
            # Initialize last_results together with bench
            func.last_results = {}

        return func

    def __call__(
        self,
        *args: object,
        **kwargs: object,
    ) -> Callable:
        """
        EasyBench benchmark decorator.

        Example:
            ```python
            @bench(
                item=123,
                big_list=lambda: list(range(1_000_000)),
            )
            def add_item(item, big_list):
                big_list.append(item)
            ```

            # Using a list of BenchParams for comparison
            ```python
            params1 = BenchParams(
                name="Small", params={"lst": lambda: list(range(1000))},
            )
            params2 = BenchParams(
                name="Large", params={"lst": lambda: list(range(10000))},
            )

            @bench([params1, params2])
            def pop_first(lst):
                return lst.pop(0)
            ```

        Returns:
            A decorated function with benchmark capabilities

        """
        # Handle list of BenchParams as first argument
        if (
            len(args) == 1
            and isinstance(args[0], list)
            and not kwargs
            and all(isinstance(param, BenchParams) for param in args[0])
        ):
            return self._decorate_with_bench_param_list(args[0])

        # Handle BenchParams instance as first argument
        if len(args) == 1 and isinstance(args[0], BenchParams) and not kwargs:
            return self._decorate_with_bench_param(args[0])

        # Handle direct function decoration: @bench
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return self._decorate(args[0])

        # Handle parameterized decoration: @bench(param=value)
        def decorator(func: Callable) -> Callable:
            return self._decorate(func, **kwargs)

        return decorator

    def _process_single_param_set(
        self,
        func: Callable,
        params: BenchParams,
        index: int,
        func_name: str,
    ) -> tuple[str, ResultType | None, BenchConfig | None]:
        """
        Process a single parameter set for benchmarking.

        Args:
            func: The original function to benchmark
            params: The parameter set to use
            index: The index of this parameter in the list
            func_name: Name of the function

        Returns:
            A tuple of (param_name, results_dict, config)

        """
        # Determine parameter name
        param_name = f"params_{index+1}"
        if params.name:
            param_name = params.name

        # Decorate with this parameter set (with empty reporters)
        decorated_func = self._decorate_with_bench_param(params, reporters=[])(func)

        # Extract results from the function
        bench_func = cast("BenchmarkableFunction", decorated_func)
        results_dict = None
        config = None

        if bench_func.last_results:
            results_dict = bench_func.last_results[func_name]
            config = bench_func.bench.bench_config.model_copy(deep=True)

        return param_name, results_dict, config

    def _display_comparison_results(
        self,
        all_results: ResultsType,
        first_config: BenchConfig,
        original_reporters: list[Reporter] | None,
    ) -> None:
        """
        Display comparison results.

        Args:
            all_results: Dictionary of all benchmark results
            first_config: The first benchmark configuration
            original_reporters: Original reporters from the function

        """
        comparison_config = first_config
        if original_reporters is not None:
            comparison_config.reporters = original_reporters
        elif comparison_config.reporters == []:
            comparison_config.reporters = [ConsoleReporter()]

        comparison = EasyBench(bench_config=comparison_config)
        comparison._display_results(  # noqa: SLF001
            results=all_results,
            config=comparison_config,
        )

    def _decorate_with_bench_param_list(
        self,
        params_list: list[BenchParams],
    ) -> Callable:
        """
        Set up the function for benchmarking using a list of BenchParams instances.

        This allows comparing different parameter configurations for the same function.

        Args:
            params_list: List of BenchParams instances with benchmark configurations

        Returns:
            A decorator function that configures the function with multiple
            BenchParams and compares results

        """

        def decorator(func: Callable) -> Callable:
            # Store original function
            orig_func = func
            func_name = func.__name__

            # Store original reporters from the function if available
            original_reporters = None
            if hasattr(func, "bench") and hasattr(func.bench, "bench_config"):
                bench_func = cast("BenchmarkableFunction", func)
                original_reporters = bench_func.bench.bench_config.reporters

            # Store results for each parameter set
            all_results: ResultsType = {}
            first_config = None

            # Run benchmarks for each parameter set
            for i, params in enumerate(params_list):
                param_name, results, config = self._process_single_param_set(
                    func,
                    params,
                    i,
                    func_name,
                )

                if results:
                    all_results[f"{func_name} ({param_name})"] = results
                    if first_config is None:
                        first_config = config

            # Display results if we have results
            if len(all_results) >= 1 and first_config:
                self._display_comparison_results(
                    all_results,
                    first_config,
                    original_reporters,
                )

            # Return the original function
            return orig_func

        return decorator

    def _decorate_with_bench_param(
        self,
        param: BenchParams,
        reporters: list[Reporter] | None = None,
    ) -> Callable:
        """
        Set up the function for benchmarking using a BenchParams instance.

        Args:
            param: BenchParams instance with benchmark configuration
            reporters: List of reporters for benchmark

        Returns:
            A decorator function that configures the function with the BenchParams

        """

        def decorator(func: Callable) -> Callable:
            # Initialize benchmark attributes
            func = self._initialize_bench_attributes(func)

            if reporters is not None:
                func.bench.bench_config.reporters = reporters

            # Apply function parameters
            if param.fn_params:
                for name, value in param.fn_params.items():
                    func.fixture_registry["trial"][name] = lambda v=value: v

            # Apply bench parameters
            if param.params:
                for name, value in param.params.items():
                    if callable(value) and not isinstance(value, type):
                        # For callables like lambdas, register the callable itself
                        func.fixture_registry["trial"][name] = value
                    else:
                        # For values, create a lambda that returns the value
                        func.fixture_registry["trial"][name] = lambda v=value: v

            # Run benchmark if all parameters are satisfied
            self._maybe_run_benchmark(func)

            return func

        return decorator

    def _decorate(self, func: Callable, **kwargs: object) -> Callable:
        """
        Set up the function for benchmarking.

        Args:
            func: The function to benchmark
            **kwargs: Benchmark parameters

        Returns:
            The decorated function

        """
        # Initialize benchmark attributes
        func = self._initialize_bench_attributes(func)

        # Register variables as fixtures
        for name, value in kwargs.items():
            if callable(value) and not isinstance(value, type):
                # For callables like lambdas, register the callable itself
                func.fixture_registry["trial"][name] = value
            else:
                # For values, create a lambda that returns the value
                func.fixture_registry["trial"][name] = lambda v=value: v

        # Check if all required parameters are available and run if they are
        self._maybe_run_benchmark(func)
        return func

    def _reset_bench(
        self,
        func: BenchmarkableFunction,
        *,
        reset_params: bool = True,
        reset_config: bool = True,
    ) -> None:
        """
        Reset benchmark settings.

        Args:
            func: The benchmarked function
            reset_params: Whether to reset parameters
            reset_config: Whether to reset configuration

        """
        if reset_params:
            func.fixture_registry = {
                "trial": {},
                "function": {},
                "class": {},
            }

        if reset_config:
            func.bench.bench_config = BenchConfig()

    def fn_params(self, **kwargs: object) -> Callable:
        """
        Configure benchmark function parameters.

        Example:
            ```python
            @bench.fn_params(op=lambda x: x + 1)
            def apply_operation(op):
                return op(12345)
            ```

        Returns:
            A decorator function that configures benchmark parameters

        """

        def decorator(func: Callable) -> Callable:
            # Initialize benchmark attributes
            func = self._initialize_bench_attributes(func)

            # Register function parameters as fixtures
            for name, value in kwargs.items():
                func.fixture_registry["trial"][name] = lambda v=value: v

            self._maybe_run_benchmark(func)
            return func

        return decorator

    def _maybe_run_benchmark(self, func: BenchmarkableFunction) -> None:
        """Run the benchmark if all required parameters are available."""
        # Get function signature to determine required parameters
        sig = inspect.signature(func)

        # Get parameters that have no defaults and must be provided
        required_params = {
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        }

        # Check if all required parameters are available in fixtures
        provided_params = set(func.fixture_registry["trial"].keys())

        if required_params.issubset(provided_params):
            # All parameters are satisfied, run the benchmark
            results = func.bench.bench(fixture_registry=func.fixture_registry)
            func.last_results = results

            # Reset bench
            self._reset_bench(func, reset_config=False)

    @overload
    def config(self) -> Callable: ...

    @overload
    def config(
        self,
        *,
        trials: int | None = None,
        loops_per_trial: int | None = None,
        sort_by: SortType | None = None,
        reverse: bool | None = None,
        memory: bool | MemoryUnit | str | None = None,
        time: bool | TimeUnit | str | None = None,
        color: bool | None = None,
        show_output: bool | None = None,
        reporters: list[Reporter] | None = None,
        progress: bool | Callable | None = None,
    ) -> Callable: ...

    def config(self, **kwargs: Any) -> Callable:
        """
        Configure benchmark settings.

        Example:
            ```python
            @bench(
                item=123,
                big_list=lambda: list(range(1_000_000)),
            )
            @bench.config(trials=5, loops_per_trial=10, memory=True)
            def add_item(item, big_list):
                big_list.append(item)
            ```

        Returns:
            A decorator function that configures benchmark settings

        """

        def decorator(func: Callable) -> Callable:
            # Initialize benchmark attributes
            func = self._initialize_bench_attributes(func)

            # Create partial config from kwargs
            partial_config = PartialBenchConfig(**kwargs)

            # Update benchmark config with merged values
            func.bench.bench_config = partial_config.merge_with(func.bench.bench_config)

            # Check if all required parameters are available
            self._maybe_run_benchmark(func)
            return func

        return decorator

    def grid(
        self,
        params_lists: Iterable[Iterable[BenchParams]],
    ) -> Callable:
        """
        Create a decorator that applies a Cartesian product of parameter lists.

        This creates all combinations of the parameter sets for benchmarking.

        Example:
            ```python
            sizes = [
                BenchParams(name="Small", params={"size": 100}),
                BenchParams(name="Large", params={"size": 10000}),
            ]
            ops = [
                BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)}),
                BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()}),
            ]

            @bench.grid([sizes, ops])
            def operation(size, op):
                # Will run with all combinations:
                # (Small, Append), (Small, Pop), (Large, Append), (Large, Pop)
                lst = list(range(size))
                op(lst)
            ```

        Args:
            params_lists: An iterable of iterables of BenchParams to combine

        Returns:
            A decorator function that applies all parameter combinations

        """

        def decorator(func: Callable) -> Callable:
            # Convert all parameter lists to actual lists
            converted_lists = []
            for params in params_lists:
                if isinstance(params, (list, tuple)):
                    converted_lists.append(list(params))
                else:
                    # Handle iterables by converting to list
                    converted_lists.append(list(params))

            # Handle the case with no parameter lists
            if not converted_lists:
                return func

            # Handle the case with just one parameter list
            if len(converted_lists) == 1:
                return self._decorate_with_bench_param_list(converted_lists[0])(func)

            # Start with the first list
            result_params = converted_lists[0]

            # Multiply with each subsequent list to get Cartesian product
            for params_list in converted_lists[1:]:
                new_result = []
                for param1 in result_params:
                    for param2 in params_list:
                        # Use the multiplication operator defined in BenchParams
                        combined_param = param1 * param2
                        new_result.append(combined_param)
                result_params = new_result

            # Apply the combined parameters using _decorate_with_bench_param_list
            return self._decorate_with_bench_param_list(result_params)(func)

        return decorator


# Create decorator instance
bench = BenchDecorator()
