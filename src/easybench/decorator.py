"""
Benchmark decorator utilities for easier function benchmarking.

This module provides decorators for benchmarking functions with configurable
parameters and settings. It allows for easy setup and running of benchmarks.
"""

import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload

from .core import (
    BenchConfig,
    FixtureRegistry,
    FunctionBench,
    PartialBenchConfig,
    SortType,
)
from .reporters import Reporter

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


class BenchDecorator:
    """Decorator for benchmarking functions."""

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

        Returns:
            A decorated function with benchmark capabilities

        """
        # Handle direct function decoration: @bench
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return self._decorate(args[0])

        # Handle parameterized decoration: @bench(param=value)
        def decorator(func: Callable) -> Callable:
            return self._decorate(func, **kwargs)

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
        func = cast("BenchmarkableFunction", func)
        # Create FunctionBench instance if not already present
        if not hasattr(func, "bench"):
            func.bench = FunctionBench(func)
            func.fixture_registry = {
                "trial": {},
                "function": {},
                "class": {},
            }

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
            # Create FunctionBench instance if needed
            func = cast("BenchmarkableFunction", func)
            if not hasattr(func, "bench"):
                func.bench = FunctionBench(func)
                func.fixture_registry = {
                    "trial": {},
                    "function": {},
                    "class": {},
                }

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
        required_params = set(sig.parameters.keys())

        # Check if all required parameters are available in fixtures
        provided_params = set(func.fixture_registry["trial"].keys())

        if required_params.issubset(provided_params):
            # All parameters are satisfied, run the benchmark
            func.bench.bench(fixture_registry=func.fixture_registry)

            # Reset bench
            self._reset_bench(func, reset_config=False)

    @overload
    def config(self) -> Callable: ...

    @overload
    def config(
        self,
        *,
        trials: int | None = None,
        sort_by: SortType | None = None,
        reverse: bool | None = None,
        memory: bool | None = None,
        color: bool | None = None,
        show_output: bool | None = None,
        reporters: list[Reporter] | None = None,
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
            @bench.config(trials=5, memory=True)
            def add_item(item, big_list):
                big_list.append(item)
            ```

        Returns:
            A decorator function that configures benchmark settings

        """

        def decorator(func: Callable) -> Callable:
            func = cast("BenchmarkableFunction", func)
            # Create FunctionBench instance if needed
            if not hasattr(func, "bench"):
                func.bench = FunctionBench(func)
                func.fixture_registry = {
                    "trial": {},
                    "function": {},
                    "class": {},
                }

            # Create partial config from kwargs
            partial_config = PartialBenchConfig(**kwargs)

            # Update benchmark config with merged values
            func.bench.bench_config = partial_config.merge_with(func.bench.bench_config)

            # Check if all required parameters are available
            self._maybe_run_benchmark(func)
            return func

        return decorator


# Create decorator instance
bench = BenchDecorator()
