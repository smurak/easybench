"""
Tests for the EasyBench class functionality.

This module contains various test cases for the EasyBench class, including:
- Output formatting tests
- Time measurement tests
- Memory measurement tests
- Fixture handling tests
- Lifecycle method tests
- Configuration tests
"""

import logging
import time
from collections.abc import Callable, Generator, Iterable
from typing import Any, ClassVar, cast
from unittest import mock
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from easybench import BenchConfig, EasyBench, fixture
from easybench.core import (
    BenchParams,
    ConsoleReporter,
    FixtureRegistry,
    FunctionBench,
    PartialBenchConfig,
    ScopeType,
    SimpleConsoleReporter,
    TableFormatter,
    customize,
    parametrize,
)

# Constants to replace magic numbers
MIN_SLEEP_TIME = 0.05
MIN_MEMORY_KB = 100
MED_MEMORY_KB = 500
LARGE_MEMORY_KB = 1000
EXPECTED_RETURN_VAL = 42
DEFAULT_TRIALS = 3
CONFIG_TRIALS_5 = 5
CONFIG_TRIALS_10 = 10
LIFECYCLE_CALLS_THRESHOLD = 3  # New constant for test_lifecycle_method_execution_order
EXPECTED_SUM_RESULT = 5  # New constant for test_function_direct_call
LAMBDA_DEFAULT_PARAM = 1
LAMBDA_MULTIPLY_FACTOR = 2
EXPECTED_RESULT_5 = 5
EXPECTED_RESULT_10 = 10
EXPECTED_RESULT_15 = 15
# Add tolerance for time comparisons to handle Windows timer resolution
TIME_COMPARISON_TOLERANCE = 0.01
# Constants for time unit conversion tests
EXPECTED_SLEEP_TIME_SEC = 0.01  # 10ms sleep time
MIN_MS_TIME = 5  # Minimum expected time in milliseconds
MAX_MS_TIME = 100  # Maximum expected time in milliseconds
MIN_US_TIME = 1000  # Minimum expected time in microseconds
MAX_US_TIME = 100000  # Maximum expected time in microseconds
MIN_S_TIME = 0.001  # Minimum expected time in seconds
MAX_S_TIME = 0.1  # Maximum expected time in seconds


class TestEasyBenchOutput:
    """Tests for EasyBench output formatting."""

    def test_default_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test benchmark output format with default settings."""

        class SimpleBench(EasyBench):
            bench_config = BenchConfig()

            def bench_simple(self) -> int:
                return 10

        bench = SimpleBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "bench_simple" in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_single_trial_format(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test output format when only one trial is configured."""

        class SingleTrialBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_single(self) -> int:
                return 10

        bench = SingleTrialBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == 1
        assert "bench_single" in parsed_out["functions"]
        assert parsed_out["is_single_trial"]
        assert "Time (s)" in captured.out
        assert "Avg Time" not in captured.out

    def test_multiple_benchmarks_display(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test when multiple benchmark methods are executed and displayed together."""

        class MultipleBench(EasyBench):
            bench_config = BenchConfig()

            def bench_first(self) -> int:
                return 10

            def bench_second(self) -> int:
                return 20

        bench = MultipleBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "bench_first" in parsed_out["functions"]
        assert "bench_second" in parsed_out["functions"]

    def test_memory_metrics_display(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test output format when memory measurement is enabled."""

        class MemoryBench(EasyBench):
            bench_config = BenchConfig(memory=True)

            def bench_memory(self) -> list[int]:
                return [0] * 1000

        bench = MemoryBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["has_memory_metrics"]
        assert "Avg Mem" in captured.out
        assert "Max Mem" in captured.out

    def test_return_values_display(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that function return values are displayed when show_output=True."""

        class ShowOutputBench(EasyBench):
            bench_config = BenchConfig(show_output=True)

            def bench_return_value(self) -> int:
                return EXPECTED_RETURN_VAL

        bench = ShowOutputBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "Benchmark Return Values" in captured.out
        assert str(EXPECTED_RETURN_VAL) in captured.out

    def test_results_sorting(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that results are sorted according to the specified criteria."""

        class SortedBench(EasyBench):
            bench_config = BenchConfig(sort_by="avg", trials=DEFAULT_TRIALS)

            def bench_slow(self) -> None:
                time.sleep(MIN_SLEEP_TIME)

            def bench_fast(self) -> None:
                pass

        bench = SortedBench()
        bench.bench()

        captured = capsys.readouterr()
        # When sorted by average time, 'fast' should appear before 'slow'
        fast_pos = captured.out.find("bench_fast")
        slow_pos = captured.out.find("bench_slow")
        assert fast_pos < slow_pos

    def test_no_benchmarks_output(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test output when there are no benchmark methods."""

        class EmptyBench(EasyBench):
            pass

        bench = EmptyBench()
        with caplog.at_level(logging.INFO):
            bench.bench()

        assert "No benchmark results to display" in caplog.text

    def test_definition_order_sorting(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that sort_by='def' correctly preserves method definition order."""

        class DefinitionOrderBench(EasyBench):
            # Define methods deliberately not in alphabetical order
            def bench_c(self) -> None:
                pass

            def bench_a(self) -> None:
                time.sleep(0.1)

            def bench_b(self) -> None:
                time.sleep(0.0001)

        # Test with sort_by="def" (should preserve c, a, b order)
        bench1 = DefinitionOrderBench()
        bench1.bench(config=PartialBenchConfig(sort_by="def"))

        captured_def = capsys.readouterr()
        c_pos = captured_def.out.find("bench_c")
        a_pos = captured_def.out.find("bench_a")
        b_pos = captured_def.out.find("bench_b")
        # Verify definition order is preserved (c, a, b)
        assert 0 < c_pos < a_pos < b_pos, "Definition order not preserved"

        # Compare with sort_by="avg" (should be c, b, a - from fast to slow)
        bench2 = DefinitionOrderBench()
        bench2.bench(config=PartialBenchConfig(sort_by="avg"))

        captured_avg = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured_avg.out)
        sorted_methods = list(parsed_out["functions"].keys())

        # With sort_by="avg", should be ordered by execution time (fast to slow)
        assert sorted_methods == [
            "bench_c",
            "bench_b",
            "bench_a",
        ], "Expected performance order not followed"

    def test_include_exclude_with_parametrize_names(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that include/exclude can filter by parametrize parameter names."""
        small_params = BenchParams(name="small", params={"value": 10})
        medium_params = BenchParams(name="medium", params={"value": 100})
        large_params = BenchParams(name="large", params={"value": 1000})

        class ParamBench(EasyBench):
            @parametrize([small_params, medium_params, large_params])
            def bench_func(self, value: int) -> int:
                return value * 2

            def bench_other(self) -> int:
                return 42

        # Test include with parameter name
        bench = ParamBench()
        config = BenchConfig(include="medium")
        bench.bench(config=config)
        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Should only include the medium parameter benchmark
        assert len(parsed_out["functions"]) == 1
        assert "bench_func (medium)" in parsed_out["functions"]
        assert "bench_func (small)" not in parsed_out["functions"]
        assert "bench_func (large)" not in parsed_out["functions"]
        assert "bench_other" not in parsed_out["functions"]

        # Test exclude with parameter name
        bench = ParamBench()
        config = BenchConfig(exclude="small|large")
        bench.bench(config=config)
        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Should exclude the small and large parameter benchmarks
        assert "bench_func (medium)" in parsed_out["functions"]
        assert "bench_other" in parsed_out["functions"]
        assert "bench_func (small)" not in parsed_out["functions"]
        assert "bench_func (large)" not in parsed_out["functions"]

        # Test include with function name and parameter name
        bench = ParamBench()
        bench.bench(include=r"bench_func \(small\)")
        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Should only include the small parameter benchmark
        assert len(parsed_out["functions"]) == 1
        assert "bench_func (small)" in parsed_out["functions"]


class TestEasyBenchTime:
    """Tests for time measurement functionality in EasyBench."""

    def test_accurate_time_measurement(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test accurate time measurement using sleep."""

        class TimeBench(EasyBench):
            bench_config = BenchConfig()

            def bench_sleep(self) -> None:
                time.sleep(MIN_SLEEP_TIME)

        bench = TimeBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "bench_sleep" in parsed_out["functions"]
        # Add tolerance to timing assertions for Windows timer resolution
        assert (
            parsed_out["functions"]["bench_sleep"]["avg"]
            >= MIN_SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )
        assert (
            parsed_out["functions"]["bench_sleep"]["min"]
            >= MIN_SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )
        assert (
            parsed_out["functions"]["bench_sleep"]["max"]
            >= MIN_SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )

    def test_relative_timing_accuracy(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test the accuracy of relative time measurements."""

        class RelativeTimeBench(EasyBench):
            bench_config = BenchConfig(trials=DEFAULT_TRIALS)

            def bench_slow(self) -> None:
                time.sleep(0.05)

            def bench_fast(self) -> None:
                pass

        bench = RelativeTimeBench()
        bench.bench(trials=10)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert (
            parsed_out["functions"]["bench_fast"]["avg"]
            < parsed_out["functions"]["bench_slow"]["avg"]
        )


class TestEasyBenchMemory:
    """Tests for memory measurement functionality in EasyBench."""

    @pytest.mark.parametrize("kb_size", [MIN_MEMORY_KB, MED_MEMORY_KB])
    def test_memory_allocation_measurement(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
        kb_size: int,
    ) -> None:
        """Test memory measurement using predictable memory allocation."""

        class MemoryAllocBench(EasyBench):
            bench_config = BenchConfig(memory=True, trials=DEFAULT_TRIALS)

            def __init__(self, kb_size: int) -> None:
                super().__init__()
                self.kb_size = kb_size

            def bench_allocate(self) -> list[int]:
                return allocate_memory(self.kb_size)

        bench = MemoryAllocBench(kb_size)
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["has_memory_metrics"]
        assert (
            kb_size
            <= parsed_out["functions"]["bench_allocate"]["avg_memory"]
            < kb_size * 2
        )
        assert (
            kb_size
            <= parsed_out["functions"]["bench_allocate"]["max_memory"]
            < kb_size * 2
        )

    def test_relative_memory_usage(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test accurate measurement of relative memory usage between methods."""

        class MemoryComparisonBench(EasyBench):
            bench_config = BenchConfig(memory=True, trials=DEFAULT_TRIALS)

            def bench_small_alloc(self) -> list[int]:
                return allocate_memory(MIN_MEMORY_KB)

            def bench_large_alloc(self) -> list[int]:
                return allocate_memory(MED_MEMORY_KB)

        bench = MemoryComparisonBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert (
            parsed_out["functions"]["bench_small_alloc"]["avg_memory"]
            < parsed_out["functions"]["bench_large_alloc"]["avg_memory"]
        )

    def test_memory_with_return_values(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that memory and return values are displayed with show_output=True."""

        class ShowMemoryBench(EasyBench):
            bench_config = BenchConfig(memory=True, show_output=True)

            def bench_test(self) -> str:
                return "This is a test."

        bench = ShowMemoryBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["has_memory_metrics"]
        assert "Benchmark Return Values" in captured.out
        assert "This is a test." in captured.out

    def test_memory_auto_enable_with_sort(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that memory is automatically enabled when sorting by memory metrics."""
        # Capture logs at INFO level
        with caplog.at_level(logging.INFO):

            class AutoMemoryBench(EasyBench):
                def bench_test(self) -> list[int]:
                    return [0] * 1000

            bench = AutoMemoryBench()
            bench.bench(sort_by="avg_memory")

        # Check that memory was enabled (in logs now, not stdout)
        assert "Note: Enabled memory measurement" in caplog.text

        # Still check for memory columns in stdout
        captured = capsys.readouterr()
        assert "Avg Mem" in captured.out


class TestEasyBenchFixtures:
    """Tests for the fixture system in EasyBench."""

    def test_basic_fixture_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that fixtures are correctly passed to benchmark methods."""

        @fixture(scope="trial")
        def test_value() -> int:
            return EXPECTED_RETURN_VAL

        class FixtureBench(EasyBench):
            bench_config = BenchConfig()

            def bench_use_fixture(self, test_value: int) -> None:
                assert test_value == EXPECTED_RETURN_VAL

        bench = FixtureBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_use_fixture" in captured.out

    def test_trial_scoped_fixture(self) -> None:
        """Test that trial-scoped fixtures are recreated for each trial."""
        counter = 0

        @fixture(scope="trial")
        def trial_counter() -> int:
            nonlocal counter
            counter += 1
            return counter

        class TrialFixtureBench(EasyBench):
            bench_config = BenchConfig(trials=DEFAULT_TRIALS)
            values: ClassVar[list[int]] = []

            def bench_collect(self, trial_counter: int) -> None:
                self.values.append(trial_counter)

        bench = TrialFixtureBench()
        bench.bench()

        assert bench.values == [1, 2, 3]

    def test_function_scoped_fixture(self) -> None:
        """Test that function-scoped fixtures are created once per function."""
        counter = 0

        @fixture(scope="function")
        def function_counter() -> int:
            nonlocal counter
            counter += 1
            return counter

        class FunctionFixtureBench(EasyBench):
            bench_config = BenchConfig(trials=DEFAULT_TRIALS)
            values1: ClassVar[list[int]] = []
            values2: ClassVar[list[int]] = []

            def bench_collect1(self, function_counter: int) -> None:
                self.values1.append(function_counter)

            def bench_collect2(self, function_counter: int) -> None:
                self.values2.append(function_counter)

        bench = FunctionFixtureBench()
        bench.bench()

        assert bench.values1 == [1, 1, 1]
        assert bench.values2 == [2, 2, 2]

    def test_class_scoped_fixture(self) -> None:
        """Test that class-scoped fixtures are created once per class."""
        counter = 0

        @fixture(scope="class")
        def class_counter() -> int:
            nonlocal counter
            counter += 1
            return counter

        class ClassFixtureBench(EasyBench):
            bench_config = BenchConfig(trials=2)
            values1: ClassVar[list[int]] = []
            values2: ClassVar[list[int]] = []

            def bench_collect1(self, class_counter: int) -> None:
                self.values1.append(class_counter)

            def bench_collect2(self, class_counter: int) -> None:
                self.values2.append(class_counter)

        bench = ClassFixtureBench()
        bench.bench()

        # Class fixtures are created only once for the class,
        # so all values should be the same
        assert len(set(bench.values1 + bench.values2)) == 1
        assert bench.values1[0] == 1

    def test_fixture_with_generator_syntax(self) -> None:
        """Test fixtures using generator syntax with yield."""
        setup_called = False
        teardown_called = False

        @fixture(scope="function")
        def generator_fixture() -> Generator[str, None, None]:
            nonlocal setup_called, teardown_called
            setup_called = True
            yield "generated value"
            teardown_called = True

        class GeneratorBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_use_generator(self, generator_fixture: str) -> None:
                assert generator_fixture == "generated value"

        bench = GeneratorBench()
        bench.bench()

        assert setup_called is True
        assert teardown_called is True

    def test_custom_fixture_registry(self) -> None:
        """Test using a custom fixture registry."""
        custom_registry: FixtureRegistry = {"trial": {}, "function": {}, "class": {}}

        @fixture(scope="trial", fixture_registry=custom_registry)
        def custom_fixture() -> str:
            return "custom value"

        class CustomRegistryBench(EasyBench):
            bench_config = BenchConfig(trials=1)
            result: str | None = None

            def bench_test(self, custom_fixture: str | None = None) -> None:
                # Should be None with default registry
                self.result = custom_fixture

        bench = CustomRegistryBench()
        bench.bench()  # Using default registry
        assert bench.result is None

        # Run with custom registry
        bench.bench(fixture_registry=custom_registry)
        assert bench.result == "custom value"

    def test_parametrize_grid(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parametrize.grid for creating Cartesian products of parameter sets."""
        # Define size parameter sets
        small = BenchParams(name="Small", params={"size": 10})
        large = BenchParams(name="Large", params={"size": 100})

        # Define operation parameter sets
        append = BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)})
        pop = BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()})

        class GridBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            # Use parametrize.grid to create a Cartesian product of parameters
            @parametrize.grid([[small, large], [append, pop]])
            def bench_operation(
                self,
                size: int,
                op: Callable[[list[int]], Any],
            ) -> None:
                lst = list(range(size))
                op(lst)

        bench = GridBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations were created
        assert "bench_operation (Small x Append)" in parsed_out["functions"]
        assert "bench_operation (Small x Pop)" in parsed_out["functions"]
        assert "bench_operation (Large x Append)" in parsed_out["functions"]
        assert "bench_operation (Large x Pop)" in parsed_out["functions"]

        # Ensure we have exactly 4 combinations (2 * 2)
        assert len(parsed_out["functions"]) == 2 * 2

        # Test comparison with stacked parametrize decorators
        class StackedBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            # Same result using stacked parametrize decorators
            @parametrize([append, pop])
            @parametrize([small, large])
            def bench_operation(
                self,
                size: int,
                op: Callable[[list[int]], Any],
            ) -> None:
                lst = list(range(size))
                op(lst)

        bench2 = StackedBench()
        bench2.bench()

        captured2 = capsys.readouterr()
        parsed_out2 = parse_benchmark_output(captured2.out)

        # Verify both approaches produce the same combinations
        assert set(parsed_out["functions"].keys()) == set(
            parsed_out2["functions"].keys(),
        )

    def test_parametrize_grid_with_simple_values(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that parametrize.grid works correctly with simple values."""
        # Define parameter sets with simple values instead of functions
        small = BenchParams(name="Small", params={"size": 100})
        medium = BenchParams(name="Medium", params={"size": 1000})
        large = BenchParams(name="Large", params={"size": 10000})

        add = BenchParams(name="Add", params={"op_name": "add", "value": 5})
        multiply = BenchParams(
            name="Multiply",
            params={"op_name": "multiply", "value": 2},
        )

        params_list1 = [small, medium, large]
        params_list2 = [add, multiply]

        class GridBench(EasyBench):
            @parametrize.grid([params_list1, params_list2])
            def bench_operation(self, size: int, op_name: str, value: int) -> int:
                result = 0
                if op_name == "add":
                    # Simple operation that adds 'value' to each item in a range
                    for i in range(size):
                        result += i + value
                else:
                    # Simple operation that multiplies each item by 'value' in a range
                    for i in range(size):
                        result += i * value
                return result

        bench = GridBench()
        bench.bench(trials=1)  # Just one trial for testing

        # Capture and parse output
        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Check that all 6 combinations were executed
        expected_combinations = [
            "bench_operation (Small x Add)",
            "bench_operation (Small x Multiply)",
            "bench_operation (Medium x Add)",
            "bench_operation (Medium x Multiply)",
            "bench_operation (Large x Add)",
            "bench_operation (Large x Multiply)",
        ]

        for combination in expected_combinations:
            assert (
                combination in parsed_out["functions"]
            ), f"Missing combination: {combination}"

        # Total should be 6 functions
        assert len(parsed_out["functions"]) == len(params_list1) * len(params_list2)

    def test_default_args_in_benchmark_method(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that benchmark methods with default arguments work correctly."""
        # Track what values were used in the benchmark
        used_values: list[tuple] = []

        class DefaultArgsBench(EasyBench):
            bench_config = BenchConfig(trials=DEFAULT_TRIALS, show_output=True)

            def bench_with_defaults(
                self,
                a: int = 10,
                b: str = "default",
                c: None = None,
            ) -> tuple:
                """Benchmark method with various default arguments."""
                result = (a, b, c)
                used_values.append(result)
                return result

            def bench_mixed_args(self, a: int = 5, fixture_arg: None = None) -> tuple:
                """Benchmark with mix of default args and fixture args."""
                result = (a, fixture_arg)
                used_values.append(result)
                return result

        # Create and register a fixture for the second benchmark
        fixture_value = "fixture_value"
        fixture_registry: FixtureRegistry = {"trial": {}, "function": {}, "class": {}}
        fixture_registry["trial"]["fixture_arg"] = lambda: fixture_value

        # Run the benchmark
        bench = DefaultArgsBench()
        bench.bench(fixture_registry=fixture_registry)

        # Capture console output
        captured = capsys.readouterr()

        # Verify output contains expected return values
        assert "(10, 'default', None)" in captured.out
        assert f"(5, '{fixture_value}')" in captured.out

        # Verify the first method used default values
        default_calls = [v for v in used_values if v[1] == "default"]
        assert len(default_calls) == DEFAULT_TRIALS
        for call in default_calls:
            assert call == (10, "default", None)

        # Verify the second method used default value for 'a'
        # but fixture value for 'fixture_arg'
        mixed_calls = [v for v in used_values if v[1] == fixture_value]
        assert len(mixed_calls) == DEFAULT_TRIALS
        for call in mixed_calls:
            assert call == (5, fixture_value)


class TestEasyBenchLifecycle:
    """Tests for lifecycle methods in EasyBench."""

    def test_lifecycle_method_execution_order(self) -> None:
        """Test that lifecycle methods are called in the correct order."""
        calls: list[str] = []

        class LifecycleBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            def setup_class(self) -> None:
                calls.append("setup_class")

            def teardown_class(self) -> None:
                calls.append("teardown_class")

            def setup_function(self) -> None:
                calls.append("setup_function")

            def teardown_function(self) -> None:
                calls.append("teardown_function")

            def setup_trial(self) -> None:
                calls.append("setup_trial")

            def teardown_trial(self) -> None:
                calls.append("teardown_trial")

            def bench_test1(self) -> None:
                calls.append("bench_test1")

            def bench_test2(self) -> None:
                calls.append("bench_test2")

        bench = LifecycleBench()
        bench.bench()

        # Check the sequence of method calls
        assert calls[0] == "setup_class"  # First call

        # For each benchmark function:
        # - setup_function
        # - For each trial:
        #   - setup_trial
        #   - benchmark method
        #   - teardown_trial
        # - teardown_function

        # Check first benchmark function
        function1_idx = calls.index("setup_function")
        assert calls[function1_idx : function1_idx + 7] == [
            "setup_function",
            "setup_trial",
            "bench_test1",
            "teardown_trial",
            "setup_trial",
            "bench_test1",
            "teardown_trial",
        ]

        # Check second benchmark function
        function2_idx = calls.index("setup_function", function1_idx + 1)
        assert calls[function2_idx : function2_idx + 7] == [
            "setup_function",
            "setup_trial",
            "bench_test2",
            "teardown_trial",
            "setup_trial",
            "bench_test2",
            "teardown_trial",
        ]

        # Last call should be teardown_class
        assert calls[-1] == "teardown_class"

    def test_lifecycle_with_fixtures_interaction(self) -> None:
        """Test interaction between lifecycle methods and fixtures."""
        calls: list[str] = []

        @fixture(scope="trial")
        def trial_fixture() -> Generator[str, None, None]:
            calls.append("trial_fixture_setup")
            yield "trial"
            calls.append("trial_fixture_teardown")

        @fixture(scope="function")
        def function_fixture() -> Generator[str, None, None]:
            calls.append("function_fixture_setup")
            yield "function"
            calls.append("function_fixture_teardown")

        class FixtureLifecycleBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def setup_function(self) -> None:
                calls.append("setup_function")

            def teardown_function(self) -> None:
                calls.append("teardown_function")

            def setup_trial(self) -> None:
                calls.append("setup_trial")

            def teardown_trial(self) -> None:
                calls.append("teardown_trial")

            def bench_test(self, trial_fixture: str, function_fixture: str) -> None:
                calls.append(f"bench_test({trial_fixture}, {function_fixture})")

        bench = FixtureLifecycleBench()
        bench.bench()

        # Check proper order: fixture setup, method execution, fixture teardown
        function_idx = calls.index("function_fixture_setup")
        assert calls[function_idx : function_idx + 7] == [
            "function_fixture_setup",
            "setup_trial",
            "trial_fixture_setup",
            "bench_test(trial, function)",
            "trial_fixture_teardown",
            "teardown_trial",
            "function_fixture_teardown",
        ]

    def test_exception_handling_in_lifecycle(self) -> None:
        """Test handling exceptions in lifecycle methods."""
        calls: list[str] = []

        class ExceptionBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            def setup_class(self) -> None:
                calls.append("setup_class")

            def teardown_class(self) -> None:
                calls.append("teardown_class")

            def setup_function(self) -> None:
                calls.append("setup_function")

            def teardown_function(self) -> None:
                calls.append("teardown_function")

            def setup_trial(self) -> None:
                calls.append("setup_trial")
                # Raise exception on 2nd trial
                if len(calls) > LIFECYCLE_CALLS_THRESHOLD:
                    msg = "Test exception"
                    raise ValueError(msg)

            def teardown_trial(self) -> None:
                calls.append("teardown_trial")

            def bench_test(self) -> None:
                calls.append("bench_test")

        bench = ExceptionBench()
        # Verify teardown_function and teardown_class are called
        # even when exception occurs
        with pytest.raises(ValueError, match="Test exception"):
            bench.bench()

        assert "setup_class" in calls
        assert "teardown_class" in calls
        assert calls[-1] == "teardown_class"  # Last call should be teardown_class


class TestEasyBenchConfig:
    """Tests for configuration options in EasyBench."""

    def test_config_inheritance(self) -> None:
        """Test that configuration is properly inherited in subclasses."""

        class BaseBench(EasyBench):
            bench_config = BenchConfig(trials=CONFIG_TRIALS_10, memory=True)

        class DerivedBench(BaseBench):
            pass

        bench = DerivedBench()
        assert bench.bench_config.trials == CONFIG_TRIALS_10
        assert bench.bench_config.memory is True

    def test_config_override(self) -> None:
        """Test that configuration can be overridden in subclasses."""

        class BaseBench(EasyBench):
            bench_config = BenchConfig(trials=CONFIG_TRIALS_10, memory=True)

        class DerivedBench(BaseBench):
            bench_config = BenchConfig(trials=CONFIG_TRIALS_5, memory=False)

        bench = DerivedBench()
        assert bench.bench_config.trials == CONFIG_TRIALS_5
        assert bench.bench_config.memory is False

    def test_runtime_config_override(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that configuration can be overridden at runtime."""

        class ConfigBench(EasyBench):
            bench_config = BenchConfig(trials=5)

            def bench_test(self) -> None:
                pass

        bench = ConfigBench()
        # Use config parameter instead of partial_config
        config = PartialBenchConfig(trials=3, memory=True)
        bench.bench(config=config)

        captured = capsys.readouterr()
        assert "3 trials" in captured.out
        assert "Avg Mem" in captured.out

    def test_sorting_configuration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that sort_by configuration works correctly."""

        class SortByBench(EasyBench):

            def bench_b(self) -> None:
                pass

            def bench_c(self) -> None:
                time.sleep(0.01)

        # Default sort ("def") preserves definition order
        bench = SortByBench()
        bench.bench(config=PartialBenchConfig(sort_by="def"))

        captured_def = capsys.readouterr()
        b_pos = captured_def.out.find("b ")
        c_pos = captured_def.out.find("c ")
        assert b_pos < c_pos

        # Sort by average time (ascending)
        bench = SortByBench()
        bench.bench(config=PartialBenchConfig(sort_by="avg"))

        captured_avg = capsys.readouterr()
        b_pos = captured_avg.out.find("b ")
        c_pos = captured_avg.out.find("c ")
        assert b_pos < c_pos

        # Sort by average time (descending)
        bench = SortByBench()
        bench.bench(config=PartialBenchConfig(sort_by="avg", reverse=True))

        captured_rev = capsys.readouterr()
        c_pos = captured_rev.out.find("c ")
        b_pos = captured_rev.out.find("b ")
        assert c_pos < b_pos

    def test_using_complete_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that a complete BenchConfig can be passed directly."""

        class CompleteConfigBench(EasyBench):
            def bench_test(self) -> None:
                pass

        bench = CompleteConfigBench()
        # Use a complete BenchConfig instead of PartialBenchConfig
        config = BenchConfig(trials=2, memory=True, show_output=True)
        bench.bench(config=config)

        captured = capsys.readouterr()
        assert "2 trials" in captured.out
        assert "Avg Mem" in captured.out

    def test_color_output_control(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that color setting controls colored output."""

        class ColorBench(EasyBench):
            def bench_test(self) -> None:
                pass

        # With color enabled
        bench = ColorBench()
        bench.bench(config=PartialBenchConfig(color=True))

        captured_color = capsys.readouterr()
        assert "\x1b[" not in captured_color.out
        assert "\033[" not in captured_color.out

        # With color disabled
        bench = ColorBench()
        bench.bench(config=PartialBenchConfig(color=False))

        captured_no_color = capsys.readouterr()
        assert "\x1b[" not in captured_no_color.out
        assert "\033[" not in captured_no_color.out

    def test_color_highlighting_min_max(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that colors are correctly applied to min and max values."""

        class ColorTargetBench(EasyBench):
            def bench_slow(self) -> None:
                time.sleep(0.01)  # Slowest - should be red (max)

            def bench_fast(self) -> None:
                pass  # Fastest - should be green (min)

        bench = ColorTargetBench()
        bench.bench(trials=DEFAULT_TRIALS, color=True)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Green and red color codes for min/max values
        green_code = "\033[32m"
        red_code = "\033[31m"

        # Both color codes should appear in the output
        assert green_code in captured.out
        assert red_code in captured.out

        assert parsed_out["color"]["avg"]["green"] == "bench_fast"
        assert parsed_out["color"]["avg"]["red"] == "bench_slow"

        assert parsed_out["color"]["min"]["green"] == "bench_fast"
        assert parsed_out["color"]["min"]["red"] == "bench_slow"

        assert parsed_out["color"]["max"]["green"] == "bench_fast"
        assert parsed_out["color"]["max"]["red"] == "bench_slow"

    def test_different_benchmark_order_with_color(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that colors are correctly applied with different benchmark order."""

        class ColorTargetBench(EasyBench):
            def bench_slow(self) -> None:
                time.sleep(MIN_SLEEP_TIME)  # Slowest - should be red (max)

            def bench_fast(self) -> None:
                pass  # Fastest - should be green (min)

        bench = ColorTargetBench()
        bench.bench(trials=5, color=True)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert parsed_out["color"]["avg"]["green"] == "bench_fast"
        assert parsed_out["color"]["avg"]["red"] == "bench_slow"

        assert parsed_out["color"]["min"]["green"] == "bench_fast"
        assert parsed_out["color"]["min"]["red"] == "bench_slow"

        assert parsed_out["color"]["max"]["green"] == "bench_fast"
        assert parsed_out["color"]["max"]["red"] == "bench_slow"

    def test_memory_metrics_color_highlighting(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that colors are correctly applied to memory metrics."""

        class ColorTargetBench(EasyBench):
            def bench_small_memory(self) -> None:
                allocate_memory(MIN_MEMORY_KB)

            def bench_medium_memory(self) -> None:
                allocate_memory(MED_MEMORY_KB)

            def bench_large_memory(self) -> None:
                allocate_memory(LARGE_MEMORY_KB)

        bench = ColorTargetBench()
        # Use config instead of partial_config
        config = PartialBenchConfig(
            trials=DEFAULT_TRIALS,
            color=True,
            memory=True,
        )
        bench.bench(config=config)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert parsed_out["color"]["avg_memory"]["green"] == "bench_small_memory"
        assert parsed_out["color"]["avg_memory"]["red"] == "bench_large_memory"

        assert parsed_out["color"]["max_memory"]["green"] == "bench_small_memory"
        assert parsed_out["color"]["max_memory"]["red"] == "bench_large_memory"

    def test_time_unit_configuration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the time unit configuration affects the output display."""

        class TimeUnitBench(EasyBench):
            def bench_test(self) -> None:
                time.sleep(0.001)  # Sleep for 1ms

        # Test with milliseconds
        bench1 = TimeUnitBench()
        bench1.bench(config=PartialBenchConfig(time="ms"))

        captured1 = capsys.readouterr()
        assert "Time (ms)" in captured1.out or "Avg Time (ms)" in captured1.out

        # Test with microseconds
        bench2 = TimeUnitBench()
        bench2.bench(config=PartialBenchConfig(time="μs"))

        captured2 = capsys.readouterr()
        assert "Time (μs)" in captured2.out or "Avg Time (μs)" in captured2.out

        # Test with seconds (default)
        bench3 = TimeUnitBench()
        bench3.bench()

        captured3 = capsys.readouterr()
        assert "Time (s)" in captured3.out or "Avg Time (s)" in captured3.out

    def test_disable_time_measurement(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that time=False disables time measurement in the output."""

        class NoTimeBench(EasyBench):
            def bench_test(self) -> None:
                time.sleep(0.001)  # Sleep for 1ms

        # Test with time=False
        bench = NoTimeBench()
        bench.bench(config=PartialBenchConfig(time=False, memory=True))

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time (s)" not in captured.out
        assert "Avg Time (s)" not in captured.out
        assert "Min Time (s)" not in captured.out
        assert "Max Time (s)" not in captured.out
        # But memory columns should be present since memory=True
        assert "Mem (KB)" in captured.out or "Avg Mem (KB)" in captured.out

    def test_time_auto_enable_with_sort(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that time measurement is auto-enabled when sorting by time metrics."""

        class SortTimeBench(EasyBench):
            def bench_test1(self) -> None:
                time.sleep(0.001)  # Sleep for 1ms

            def bench_test2(self) -> None:
                time.sleep(0.002)  # Sleep for 2ms

        # Sort by avg time but with time=False
        # System should auto-enable time measurement
        # since the sort criterion requires it
        bench = SortTimeBench()
        bench.bench(config=PartialBenchConfig(time=False, sort_by="avg"))

        captured = capsys.readouterr()
        # Time columns should be present despite time=False because of sort_by="avg"
        assert "Time (s)" in captured.out or "Avg Time (s)" in captured.out

    def test_combined_time_memory_settings(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test various combinations of time and memory measurement settings."""

        class ComboBench(EasyBench):
            def bench_test(self) -> None:
                time.sleep(0.001)  # Sleep for 1ms
                # Also allocate some memory
                _ = [0] * 10000

        # Test 1: time=False, memory=True
        bench1 = ComboBench()
        bench1.bench(config=PartialBenchConfig(time=False, memory=True))

        captured1 = capsys.readouterr()
        assert "Time" not in captured1.out
        assert "Mem" in captured1.out

        # Test 2: time=True, memory=False
        bench2 = ComboBench()
        bench2.bench(config=PartialBenchConfig(time=True, memory=False))

        captured2 = capsys.readouterr()
        assert "Time" in captured2.out
        assert "Mem" not in captured2.out

    def test_time_disabled_with_different_reporters(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that time=False works with different reporters."""

        class MultiReporterBench(EasyBench):
            def bench_test(self) -> None:
                pass

        # Test with both console and simple reporters
        bench = MultiReporterBench()
        bench.bench(
            config=PartialBenchConfig(
                time=False,
                memory=True,
                reporters=["console", "simple"],  # type: ignore [list-item]
            ),
        )

        captured = capsys.readouterr()
        # Time columns should not be present in either report
        assert "Time" not in captured.out
        assert "Avg Time" not in captured.out
        # But memory columns should be present
        assert "Mem" in captured.out or "Memory" in captured.out

    def test_time_disabled_with_warmups(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that warmups work correctly when time=False."""
        warmup_count = 0

        class WarmupBench(EasyBench):
            def bench_test(self) -> None:
                nonlocal warmup_count
                warmup_count += 1

        # Use a high number of warmups but time=False
        bench = WarmupBench()
        bench.bench(config=PartialBenchConfig(warmups=5, trials=3, time=False))

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        # But warmups should still have executed (plus actual trials)
        assert warmup_count == 5 + 3  # 5 warmups + 3 trials

    def test_time_disabled_with_parametrize(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that parametrized benchmarks work when time=False."""

        class ParamBench(EasyBench):
            small = BenchParams(name="Small", params={"size": 10})
            large = BenchParams(name="Large", params={"size": 100})

            @parametrize([small, large])
            def bench_create_list(self, size: int) -> list:
                return list(range(size))

        # Run with time=False
        bench = ParamBench()
        bench.bench(config=PartialBenchConfig(time=False, memory=True))

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        # But both parameter sets should be present
        assert "Small" in captured.out
        assert "Large" in captured.out
        # And memory columns should be present
        assert "Mem" in captured.out or "Memory" in captured.out

    def test_time_disabled_with_loops_per_trial(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that loops_per_trial works correctly when time=False."""
        loop_count = 0

        class LoopBench(EasyBench):
            def bench_test(self) -> None:
                nonlocal loop_count
                loop_count += 1

        # Use loops_per_trial but with time=False
        bench = LoopBench()
        bench.bench(
            config=PartialBenchConfig(
                loops_per_trial=5,
                trials=2,
                time=False,
                memory=True,
            ),
        )

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        # But loops should still have executed
        assert loop_count == 5 * 2  # 5 loops * 2 trials

    def test_us_alternative_for_microseconds(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that 'us' can be used as an alternative for microseconds."""

        class TimeUnitBench(EasyBench):
            def bench_test(self) -> None:
                time.sleep(0.001)  # Sleep for 1ms = 1000μs

        # Test with "us" which should be equivalent to "μs"
        bench = TimeUnitBench()
        bench.bench(config=PartialBenchConfig(time="us"))

        captured = capsys.readouterr()
        # Should display microseconds symbol in output
        assert "Time (μs)" in captured.out or "Avg Time (μs)" in captured.out

    @patch("time.perf_counter")
    def test_time_unit_conversion(
        self,
        mock_perf_counter: mock.MagicMock,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that time values are correctly converted to the specified unit."""
        # Configure the mock to return predetermined values for start/end times
        # This simulates a function that takes exactly 0.01 seconds (10ms)
        bench_time = 0.01
        mock_perf_counter.side_effect = [0.0, bench_time]

        class TimeConversionBench(EasyBench):
            def bench_sleep(self) -> None:
                # With mocked perf_counter, we don't need to actually sleep
                pass

        # Test with milliseconds
        bench1 = TimeConversionBench()
        bench1.bench(config=PartialBenchConfig(time="ms", trials=1))

        captured1 = capsys.readouterr()
        parsed_out1 = parse_benchmark_output(captured1.out)

        ms_time = parsed_out1["functions"]["bench_sleep"]["time"]

        # Should be around 10 in milliseconds (0.01s * 1000)
        assert ms_time == bench_time * 1000

        # Reset mock for next test
        mock_perf_counter.reset_mock()
        bench_time = 0.01
        mock_perf_counter.side_effect = [0.0, bench_time]

        # Test with microseconds
        bench2 = TimeConversionBench()
        bench2.bench(config=PartialBenchConfig(time="us", trials=1))

        captured2 = capsys.readouterr()
        parsed_out2 = parse_benchmark_output(captured2.out)
        us_time = parsed_out2["functions"]["bench_sleep"]["time"]

        # Should be around 10000 in microseconds (0.01s * 1000000)
        assert us_time == bench_time * 1000000

        # Reset mock for next test
        mock_perf_counter.reset_mock()
        bench_time = 0.01
        mock_perf_counter.side_effect = [0.0, bench_time]

        # Test with seconds
        bench3 = TimeConversionBench()
        bench3.bench(config=PartialBenchConfig(time="s", trials=1))

        captured3 = capsys.readouterr()
        parsed_out3 = parse_benchmark_output(captured3.out)
        s_time = parsed_out3["functions"]["bench_sleep"]["time"]

        # Should be exactly 0.01 in seconds
        assert s_time == bench_time

    def test_warmups_execution(self) -> None:
        """Test that warmup executions occur but don't affect benchmark results."""
        execution_count = 0
        execution_times: list[float] = []
        trials = 3
        warmups = 2

        class WarmupBench(EasyBench):
            bench_config = BenchConfig(trials=trials, warmups=warmups, show_output=True)

            def bench_count_executions(self) -> int:
                nonlocal execution_count, execution_times
                execution_count += 1
                # Record when this execution happened
                execution_times.append(time.perf_counter())
                return execution_count

        bench = WarmupBench()
        results = bench.bench()

        # Total executions should be trials + warmups
        assert execution_count == trials + warmups

        # Check that the benchmark results only include the non-warmup executions
        assert len(results["bench_count_executions"]["times"]) == trials

        # The values in the results should be the last 3 executions (3, 4, 5)
        # and not include the warmup executions (1, 2)
        assert "output" in results["bench_count_executions"]
        outputs = results["bench_count_executions"]["output"]
        assert set(outputs) == {3, 4, 5}

    def test_warmups_with_zero(self) -> None:
        """Test that zero warmups works correctly."""
        execution_count = 0
        trials = 2
        warmups = 0

        class NoWarmupBench(EasyBench):
            bench_config = BenchConfig(trials=trials, warmups=warmups)

            def bench_count_executions(self) -> int:
                nonlocal execution_count
                execution_count += 1
                return execution_count

        bench = NoWarmupBench()
        results = bench.bench(config=PartialBenchConfig(show_output=True))

        # Total executions should equal trials when warmups=0
        assert execution_count == trials

        # Check outputs are as expected (1, 2)
        assert "output" in results["bench_count_executions"]
        outputs = results["bench_count_executions"]["output"]
        assert outputs == [1, 2]

    def test_negative_warmups_validation(self) -> None:
        """Test that negative warmups values raise a validation error."""
        with pytest.raises(ValueError, match="warmups must be at least 0"):
            PartialBenchConfig(warmups=-1)

        # Also test with bench method
        class WarmupBench(EasyBench):
            def bench_test(self) -> None:
                pass

        bench = WarmupBench()
        with pytest.raises(ValueError, match="warmups must be at least 0"):
            bench.bench(warmups=-1)

    def test_warmups_runtime_override(self) -> None:
        """Test that warmups can be overridden at runtime."""
        execution_count = 0
        trials = 2
        warmups = 1

        class DefaultWarmupBench(EasyBench):
            bench_config = BenchConfig(trials=trials, warmups=warmups)

            def bench_count_executions(self) -> int:
                nonlocal execution_count
                execution_count += 1
                return execution_count

        bench = DefaultWarmupBench()
        # Override warmups at runtime
        new_warmups = 3
        bench.bench(config=PartialBenchConfig(warmups=new_warmups, show_output=True))

        # Total executions should be trials + warmups = 2 + 3 = 5
        assert execution_count == trials + new_warmups


class TestFunctionBench:
    """Tests for the FunctionBench class."""

    def test_function_wrapper_creation(self) -> None:
        """Test creating a FunctionBench wrapper for a function."""

        def my_function(a: int, b: int) -> int:
            return a + b

        fb = FunctionBench(my_function)
        assert fb._original_func == my_function
        assert fb._func_name == "my_function"
        assert hasattr(fb, "bench_my_function")

    def test_function_direct_call(self) -> None:
        """Test directly calling a FunctionBench object as a function."""

        def add(a: int, b: int) -> int:
            return a + b

        fb = FunctionBench(add)
        result = fb(2, EXPECTED_SUM_RESULT - 2)
        assert result == EXPECTED_SUM_RESULT

    def test_function_benchmark_execution(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test executing a benchmark using FunctionBench."""

        def slow_function() -> int:
            time.sleep(0.001)
            return EXPECTED_RETURN_VAL

        fb = FunctionBench(slow_function)
        fb.bench(trials=DEFAULT_TRIALS)

        captured = capsys.readouterr()
        assert "slow_function" in captured.out
        assert "Avg Time" in captured.out

    def test_lambda_function_with_name(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test benchmarking a named lambda function."""
        # Lambda functions need an explicit name
        fb = FunctionBench(
            lambda x=LAMBDA_DEFAULT_PARAM: x * LAMBDA_MULTIPLY_FACTOR,
            func_name="double",
        )
        fb.bench(trials=CONFIG_TRIALS_5)

        captured = capsys.readouterr()
        assert "double" in captured.out

    def test_lambda_function_without_name(self) -> None:
        """Test that lambda functions without name raise ValueError."""
        with pytest.raises(ValueError, match="func_name must be specified"):
            FunctionBench(lambda x: x * 2)

    def test_non_callable_function(self) -> None:
        """Test that non-callable objects raise TypeError."""
        with pytest.raises(TypeError, match="func must be callable"):
            FunctionBench(cast("Callable[..., object]", "not_callable"))

    def test_function_bench_void_return(self) -> None:
        """Test FunctionBench with a function that returns None."""

        def void_function() -> None:
            pass

        fb = FunctionBench(void_function)
        result = fb()
        assert result is None

    def test_function_bench_with_args_kwargs(self) -> None:
        """Test FunctionBench with positional and keyword arguments."""

        def add_multiply(a: int, b: int, c: int = 1) -> int:
            return (a + b) * c

        fb = FunctionBench(add_multiply)

        # Test with positional args
        result = fb(2, 3)
        assert result == EXPECTED_RESULT_5

        # Test with keyword args
        result = fb(a=2, b=3, c=2)
        assert result == EXPECTED_RESULT_10

        # Test with mixed args
        result = fb(2, b=3, c=3)
        assert result == EXPECTED_RESULT_15

    def test_function_bench_with_include_pattern(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test FunctionBench with include pattern."""

        def test_func() -> int:
            return 42

        fb = FunctionBench(test_func)

        # The benchmark method in FunctionBench will be named "test_func"
        # (without the "bench_" prefix that's added internally)
        results = fb.bench(include="test")

        # Should run the test
        assert "test_func" in results
        captured = capsys.readouterr()
        assert "test_func" in captured.out

        # Should not run with non-matching pattern
        results = fb.bench(include="nonexistent")
        assert not results

    def test_function_bench_with_exclude_pattern(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test FunctionBench with exclude pattern."""

        def test_func() -> int:
            return 42

        fb = FunctionBench(test_func)

        # Should not run with matching exclude pattern
        results = fb.bench(exclude="test")
        assert not results

        # Should run with non-matching exclude pattern
        results = fb.bench(exclude="nonexistent")
        assert "test_func" in results
        captured = capsys.readouterr()
        assert "test_func" in captured.out


class TestEasyBenchConfiguration:
    """Additional tests for EasyBench configuration."""

    def test_merge_partial_config_empty(self) -> None:
        """Test merging an empty partial config."""
        base_config = BenchConfig(trials=CONFIG_TRIALS_5, memory=True)
        partial_config = PartialBenchConfig()

        merged = partial_config.merge_with(base_config)

        assert merged.trials == CONFIG_TRIALS_5
        assert merged.memory is True

    def test_merge_partial_config_reporters(self) -> None:
        """Test merging partial config with reporters."""
        base_config = BenchConfig(trials=5)

        # Default reporter should exist
        assert len(base_config.reporters) == 1

        # Create a new reporter
        new_reporter = ConsoleReporter(TableFormatter())
        partial_config = PartialBenchConfig(reporters=[new_reporter])

        merged = partial_config.merge_with(base_config)

        # Should replace the reporters list with new one
        assert len(merged.reporters) == 1
        assert merged.reporters[0] is new_reporter

    def test_merge_partial_config_empty_reporters(self) -> None:
        """Test merging partial config with empty reporters list."""
        base_config = BenchConfig(trials=5)
        # Create partial config with empty reporters list
        partial_config = PartialBenchConfig(reporters=[])

        merged = partial_config.merge_with(base_config)

        assert len(merged.reporters) == 0

    def test_merge_partial_config_with_reporters_none(self) -> None:
        """Test merging partial config with None reporters."""
        # Create a base config with a custom reporter
        custom_reporter = ConsoleReporter(TableFormatter())
        base_config = BenchConfig(trials=5, reporters=[custom_reporter])

        # Create a partial config with None reporters (should not override)
        partial_config = PartialBenchConfig(reporters=None)

        merged = partial_config.merge_with(base_config)

        # Should keep the original reporters
        assert len(merged.reporters) == 1
        reporter = merged.reporters[0]
        assert isinstance(reporter, ConsoleReporter)
        assert isinstance(reporter.formatter, TableFormatter)

    def test_validate_and_convert_reporters_none(self) -> None:
        """Test that None input returns None."""
        result = PartialBenchConfig.validate_and_convert_reporters(None)
        assert result is None

    def test_validate_and_convert_reporters_strings(self) -> None:
        """Test converting a list of reporter strings."""
        reporters = ["console", "simple"]
        result = PartialBenchConfig.validate_and_convert_reporters(reporters)

        assert result is not None
        assert len(result) == len(reporters)
        assert isinstance(result[0], ConsoleReporter)
        assert isinstance(result[1], SimpleConsoleReporter)

    def test_validate_and_convert_reporters_with_kwargs(self) -> None:
        """Test converting a reporter specification with kwargs."""
        # Create a tuple of (name, kwargs)
        reporter_spec = [("simple", {"item_format": lambda n, v: f"{n}: {v}"})]
        result = PartialBenchConfig.validate_and_convert_reporters(reporter_spec)

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], SimpleConsoleReporter)

    def test_validate_and_convert_reporters_with_instances(self) -> None:
        """Test that Reporter instances are kept as-is."""
        # Create a reporter instance
        reporter = ConsoleReporter(TableFormatter())
        reporters = [reporter]

        result = PartialBenchConfig.validate_and_convert_reporters(reporters)

        assert result is not None
        assert len(result) == 1
        assert result[0] is reporter  # Should be the exact same instance

    def test_validate_and_convert_reporters_mixed(self) -> None:
        """Test converting a mixed list of reporter specifications."""
        # Create a reporter instance
        reporter = ConsoleReporter(TableFormatter())

        # Mix of string, tuple, and instance
        reporters = ["simple", ("console", {}), reporter]

        result = PartialBenchConfig.validate_and_convert_reporters(reporters)

        assert result is not None
        assert len(result) == len(reporters)
        assert isinstance(result[0], SimpleConsoleReporter)
        assert isinstance(result[1], ConsoleReporter)
        assert result[2] is reporter  # Should be the exact same instance

    def test_validate_and_convert_reporters_not_list(self) -> None:
        """Test that non-list input raises TypeError."""
        with pytest.raises(TypeError, match="reporters must be a list"):
            PartialBenchConfig.validate_and_convert_reporters("console")  # type: ignore [arg-type]

    def test_validate_and_convert_reporters_invalid_item(self) -> None:
        """Test that invalid item in list raises TypeError."""
        with pytest.raises(TypeError, match="Invalid reporter type:"):
            PartialBenchConfig.validate_and_convert_reporters([123])


class TestEasyBenchScopeManager:
    """Tests for the ScopeManager class in EasyBench."""

    def test_scope_manager_invalid_scope(self) -> None:
        """Test ScopeManager with an invalid scope."""
        bench = EasyBench()
        values: dict[str, object] = {}
        fixture_registry: FixtureRegistry = {"trial": {}, "function": {}, "class": {}}

        # Create a ScopeManager with an invalid scope
        manager = EasyBench.ScopeManager(
            bench,
            cast("ScopeType", "invalid_scope"),
            values,
            fixture_registry,
        )

        # This should raise ValueError when entering the context
        with pytest.raises(ValueError, match="Invalid scope: invalid_scope"), manager:
            pass

    def test_scope_manager_setup_fixture_error(self) -> None:
        """Test ScopeManager when a fixture setup fails."""
        bench = EasyBench()
        values: dict[str, object] = {}

        # Create fixture that raises an error
        def failing_fixture() -> None:
            error_msg = "Fixture setup error"
            raise ValueError(error_msg)

        fixture_registry: FixtureRegistry = {
            "trial": {"failing": failing_fixture},
            "function": {},
            "class": {},
        }

        # Create a ScopeManager
        manager = EasyBench.ScopeManager(bench, "trial", values, fixture_registry)

        # This should raise RuntimeError when entering the context
        with pytest.raises(RuntimeError, match="Error setting up fixture"), manager:
            pass

    def test_scope_manager_teardown_fixture_error(self) -> None:
        """Test ScopeManager when a fixture teardown fails."""
        bench = EasyBench()
        values: dict[str, object] = {}

        # Create generator fixture that raises during teardown
        def failing_generator() -> Generator[str, None, None]:
            yield "value"
            error_msg = "Fixture teardown error"
            raise ValueError(error_msg)

        fixture_registry: FixtureRegistry = {
            "trial": {"failing": failing_generator},
            "function": {},
            "class": {},
        }

        # Create a ScopeManager
        manager = EasyBench.ScopeManager(bench, "trial", values, fixture_registry)

        # This should not raise an exception (errors are logged)
        with manager:
            pass


class TestParametrizedDecorator:
    """Tests for the parametrize decorator in EasyBench classes."""

    def test_basic_parametrize_usage(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test the parametrize decorator with a single parameter set."""
        # Define a simple parameter set
        params = BenchParams(params={"value": 42})

        class SimpleParamBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            @parametrize([params])
            def bench_test(self, value: int) -> int:
                return value

        bench = SimpleParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_test (params_1)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_with_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test parametrize decorator with named parameter sets."""
        # Define parameter set with a name
        params = BenchParams(name="CustomName", params={"value": 42})

        class NamedParamBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            @parametrize([params])
            def bench_test(self, value: int) -> int:
                return value

        bench = NamedParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_test (CustomName)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_multiple_sets(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator with multiple parameter sets."""
        # Define multiple parameter sets
        small = BenchParams(name="Small", params={"size": 100})
        medium = BenchParams(name="Medium", params={"size": 500})
        large = BenchParams(name="Large", params={"size": 1000})

        class MultiParamBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            @parametrize([small, medium, large])
            def bench_process(self, size: int) -> list[int]:
                return list(range(size))

        bench = MultiParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_process (Small)" in captured.out
        assert "bench_process (Medium)" in captured.out
        assert "bench_process (Large)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_with_lambda(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator with lambda function parameters."""
        # Define parameter set with lambda
        params = BenchParams(params={"get_data": lambda: list(range(100))})

        class LambdaParamBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            @parametrize([params])
            def bench_process(self, get_data: list[int]) -> int:
                return sum(get_data)

        bench = LambdaParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_process (params_1)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_with_function_params(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator with function parameters."""

        def multiply_by_two(x: int) -> int:
            return x * 2

        # Define parameter set with function parameter
        params = BenchParams(
            params={"value": 10},
            fn_params={"transformer": multiply_by_two},
        )

        class FunctionParamBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            @parametrize([params])
            def bench_transform(
                self,
                value: int,
                transformer: Callable[[int], int],
            ) -> int:
                return transformer(value)

        bench = FunctionParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_transform (params_1)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_with_multiple_fixtures(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator with multiple fixture parameters."""

        # Create a fixture
        @fixture(scope="trial")
        def extra_value() -> int:
            return 5

        # Define parameter set with multiple parameters
        params = BenchParams(params={"base_value": 10})

        class MultiFixtureBench(EasyBench):
            bench_config = BenchConfig(trials=1, show_output=True)

            @parametrize([params])
            def bench_calculate(self, base_value: int, extra_value: int) -> int:
                return base_value + extra_value

        bench = MultiFixtureBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_calculate (params_1)" in captured.out
        assert "Return Values" in captured.out
        assert "15" in captured.out  # 10 + 5

    def test_parametrize_with_memory_tracking(
        self,
        capsys: pytest.CaptureFixture[str],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test parametrize decorator with memory tracking."""
        # Define parameter sets with different memory usage
        small = BenchParams(name="Small", params={"kb": 100})
        large = BenchParams(name="Large", params={"kb": 500})

        class MemoryParamBench(EasyBench):
            bench_config = BenchConfig(trials=2, memory=True)

            @parametrize([small, large])
            def bench_allocate(self, kb: int) -> list[int]:
                return allocate_memory(kb)

        bench = MemoryParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_allocate (Small)" in captured.out
        assert "bench_allocate (Large)" in captured.out
        assert "Avg Mem" in captured.out
        assert "Max Mem" in captured.out

    def test_parametrize_with_generator_fixture(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator with generator fixtures."""

        # Define parameter set with generator
        def value_generator() -> Generator[int, None, None]:
            yield 42
            # This cleanup should run after the benchmark

        params = BenchParams(params={"value": value_generator})

        class GeneratorParamBench(EasyBench):
            bench_config = BenchConfig(trials=1, show_output=True)

            @parametrize([params])
            def bench_use_value(self, value: int) -> int:
                return value * 2

        bench = GeneratorParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_use_value (params_1)" in captured.out
        assert "Return Values" in captured.out
        assert "84" in captured.out  # 42 * 2

    def test_parametrize_isolation(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that parametrize runs are properly isolated."""
        # Define parameter sets that modify state
        set1 = BenchParams(name="First", params={"data": lambda: [10]})
        set2 = BenchParams(name="Second", params={"data": lambda: [20]})

        class IsolationParamBench(EasyBench):
            bench_config = BenchConfig(trials=1, show_output=True)

            @parametrize([set1, set2])
            def bench_modify(self, data: list[int]) -> int:
                data.append(5)  # Modify the data
                return sum(data)

        bench = IsolationParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_modify (First)" in captured.out
        assert "bench_modify (Second)" in captured.out
        assert "Return Values" in captured.out
        assert "15" in captured.out  # 10 + 5
        assert "25" in captured.out  # 20 + 5

    def test_parametrize_with_class_config(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test parametrize decorator inherits class configuration."""
        # Define parameter sets
        param1 = BenchParams(name="A", params={"x": 1})
        param2 = BenchParams(name="B", params={"x": 2})

        class ConfiguredParamBench(EasyBench):
            # Set a specific configuration at class level
            bench_config = BenchConfig(trials=3, memory=True, show_output=True)

            @parametrize([param1, param2])
            def bench_test(self, x: int) -> int:
                return x * 10

        bench = ConfiguredParamBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "Benchmark Results (3 trials)" in captured.out
        assert "bench_test (A)" in captured.out
        assert "bench_test (B)" in captured.out
        assert "Avg Mem" in captured.out
        assert "Return Values" in captured.out
        assert "10" in captured.out  # 1 * 10
        assert "20" in captured.out  # 2 * 10

    def test_loops_per_trial_configuration(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test configuring and overriding loops per trial."""
        loop_count = 0
        trials = 2
        loops_per_trial = 3

        class LoopsBench(EasyBench):
            # Set loops_per_trial in the configuration
            bench_config = BenchConfig(trials=trials, loops_per_trial=loops_per_trial)

            def bench_count_loops(self) -> int:
                nonlocal loop_count
                loop_count += 1
                return loop_count

        # Create an instance and run with the default configuration
        bench1 = LoopsBench()
        bench1.bench()

        # With 2 trials and 3 loops per trial, we expect 6 calls
        assert loop_count == trials * loops_per_trial

        # Reset counter and override loops_per_trial at runtime
        loop_count = 0

        loops_per_trial2 = 2
        bench2 = LoopsBench()
        # Override to 2 loops per trial
        bench2.bench(trials=trials, loops_per_trial=loops_per_trial2)

        # With 2 trials and 2 loops per trial, we expect 4 calls
        assert loop_count == trials * loops_per_trial2

        captured = capsys.readouterr()
        assert "bench_count_loops" in captured.out

    def test_loops_per_trial_configuration2(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that loops_per_trial can be configured per method with @customize."""
        trials = 2
        loops = 10

        class CustomLoopsBench(EasyBench):
            bench_config = BenchConfig(trials=trials)

            def setup_class(self) -> None:
                self.counter = 0

            @customize(loops_per_trial=loops)
            def bench_with_custom_loops(self) -> None:
                self.counter += 1

            def bench_with_default_loops(self) -> None:
                self.counter += 1

        bench = CustomLoopsBench()
        bench.bench()

        # The counter should increase by:
        # - bench_with_custom_loops: 2 trials * 10 loops = 20 increments
        # - bench_with_default_loops: 2 trials * 1 loop = 2 increments
        # Total: 22 increments
        assert bench.counter == trials * loops + trials

        captured = capsys.readouterr()
        assert "bench_with_custom_loops" in captured.out
        assert "bench_with_default_loops" in captured.out
        assert "Benchmark Results" in captured.out

    def test_parametrize_with_customize(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that parametrize and customize can be used together."""
        # Define parameter sets
        small = BenchParams(name="Small", params={"size": 100})
        large = BenchParams(name="Large", params={"size": 500})

        trials = 2
        loops = 5

        class CombinedDecorators(EasyBench):
            bench_config = BenchConfig(trials=trials)

            def setup_class(self) -> None:
                self.counter = 0

            @parametrize([small, large])
            @customize(loops_per_trial=loops)
            def bench_test(self, size: int) -> None:
                # This should run with 5 loops per trial for each parameter set
                _ = size
                self.counter += 1

        bench = CombinedDecorators()
        bench.bench()

        # The counter should increase by:
        # - bench_test with Small param: 2 trials * 5 loops = 10 increments
        # - bench_test with Large param: 2 trials * 5 loops = 10 increments
        # Total: 20 increments
        assert bench.counter == trials * loops * 2

        captured = capsys.readouterr()
        assert "bench_test (Small)" in captured.out
        assert "bench_test (Large)" in captured.out
        assert "Benchmark Results" in captured.out

    def test_progress_option_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that progress=True enables progress tracking."""

        class ProgressBench(EasyBench):
            def bench_test1(self) -> None:
                pass

            def bench_test2(self) -> None:
                pass

        # Run with progress=True (should use tqdm)
        bench = ProgressBench()
        bench.bench(config=PartialBenchConfig(progress=True))

        # Just verify execution completes without errors
        # We're not testing tqdm's output, just that our code handles it correctly
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "bench_test1" in captured.out
        assert "bench_test2" in captured.out

    def test_custom_progress_function(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test using a custom progress function."""
        # Track calls to our custom progress function
        progress_calls: list[dict[str, Any]] = []

        # Create a custom progress function that records calls
        def custom_progress(
            iterable: Iterable,
            desc: str | None = None,
            total: int | None = None,
        ) -> Iterable:
            progress_calls.append({"desc": desc, "total": total})
            # Just yield the items, no actual progress visualization
            yield from iterable

        class CustomProgressBench(EasyBench):
            bench_config = BenchConfig(trials=3)

            def bench_test1(self) -> None:
                pass

            def bench_test2(self) -> None:
                pass

        bench = CustomProgressBench()
        # Use our custom progress function
        bench.bench(config=PartialBenchConfig(progress=custom_progress))

        # Verify our progress function was called
        assert len(progress_calls) > 0
        # Should have one call for the benchmark methods list
        assert any(call["desc"] == "Benchmarking" for call in progress_calls)
        # Should have at least one call for each benchmark method's trials
        assert any(
            "Function: bench_test1" in call.get("desc", "") for call in progress_calls
        )
        assert any(
            "Function: bench_test2" in call.get("desc", "") for call in progress_calls
        )

        # Verify benchmark still ran correctly
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "bench_test1" in captured.out
        assert "bench_test2" in captured.out

    def test_progress_with_parametrized_benchmark(self) -> None:
        """Test progress option with parametrized benchmarks."""
        # Track calls to our custom progress function
        progress_calls: list[dict[str, Any]] = []

        def custom_progress(
            iterable: Iterable,
            desc: str | None = None,
            total: int | None = None,
        ) -> Iterable:
            progress_calls.append({"desc": desc, "total": total})
            yield from iterable

        # Define parameter sets
        small = BenchParams(name="Small", params={"size": 10})
        large = BenchParams(name="Large", params={"size": 100})

        class ParamProgressBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            @parametrize([small, large])
            def bench_test(self, size: int) -> None:
                pass

        bench = ParamProgressBench()
        # Use our custom progress function
        bench.bench(config=PartialBenchConfig(progress=custom_progress))

        # Verify our progress function was called for parametrized benchmarks
        assert len(progress_calls) > 0
        # Should have calls for benchmark methods, parameter sets, and trials
        assert any(call["desc"] == "Benchmarking" for call in progress_calls)
        assert any(
            "Params for bench_test" in call.get("desc", "") for call in progress_calls
        )
        assert any(
            "Function: bench_test" in call.get("desc", "") for call in progress_calls
        )

    def test_customize_with_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test customize decorator with the name option."""
        custom_name = "CustomBenchmarkName"

        class NameCustomizeBench(EasyBench):
            bench_config = BenchConfig(trials=2)

            @customize(name=custom_name)
            def bench_test(self) -> int:
                return 42

        bench = NameCustomizeBench()
        bench.bench()

        captured = capsys.readouterr()
        assert custom_name in captured.out
        assert "bench_test" not in captured.out
        assert "Benchmark Results" in captured.out

    def test_customize_name_with_parametrize(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that customize name option works correctly with parametrize."""
        # Define parameter sets
        small = BenchParams(name="Small", params={"size": 100})
        large = BenchParams(name="Large", params={"size": 500})

        custom_name = "SizeTest"

        class CombinedNameDecorators(EasyBench):
            bench_config = BenchConfig(trials=2)

            @parametrize([small, large])
            @customize(name=custom_name)
            def bench_test(self, size: int) -> None:
                _ = size

        bench = CombinedNameDecorators()
        bench.bench()

        captured = capsys.readouterr()
        # The parametrize names should be appended to the custom name
        assert f"{custom_name} (Small)" in captured.out
        assert f"{custom_name} (Large)" in captured.out
        assert "bench_test" not in captured.out
        assert "Benchmark Results" in captured.out

    def test_customize_name_with_parametrize_grid(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test customize name with parametrize.grid for complex combinations."""
        # Define parameter sets
        sizes = [
            BenchParams(name="Small", params={"size": 10}),
            BenchParams(name="Large", params={"size": 100}),
        ]

        operations = [
            BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)}),
            BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()}),
        ]

        custom_name = "OperationTest"

        class GridNameBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            @parametrize.grid([sizes, operations])
            @customize(name=custom_name)
            def bench_operation(
                self,
                size: int,
                op: Callable[[list[int]], Any],
            ) -> None:
                lst = list(range(size))
                op(lst)

        bench = GridNameBench()
        bench.bench()

        captured = capsys.readouterr()
        # Verify all combinations have the custom name
        assert f"{custom_name} (Small x Append)" in captured.out
        assert f"{custom_name} (Small x Pop)" in captured.out
        assert f"{custom_name} (Large x Append)" in captured.out
        assert f"{custom_name} (Large x Pop)" in captured.out
        # Original function name should not appear
        assert "bench_operation" not in captured.out


class TestConfigValidation:
    """Tests for configuration validation in BenchConfig and PartialBenchConfig."""

    def test_benchconfig_validation(self) -> None:
        """Test ValidationError for BenchConfig."""
        # Test with a misspelled parameter
        with pytest.raises(ValidationError):
            BenchConfig(loops_per_trials=10)  # type: ignore [call-arg]

        # Test with an entirely made-up parameter
        with pytest.raises(ValidationError):
            BenchConfig(nonexistent_param=42)  # type: ignore [call-arg]

    def test_partialbenchconfig_validation(self) -> None:
        """Test ValidationError for PartialBenchConfig."""
        # Test with a misspelled parameter
        with pytest.raises(ValidationError):
            PartialBenchConfig(loops_per_trials=10)  # type: ignore [call-arg]

        # Test with an entirely made-up parameter
        with pytest.raises(ValidationError):
            PartialBenchConfig(nonexistent_param=42)  # type: ignore [call-arg]


class TestEasyBenchMethodFiltering:
    """Tests for filtering benchmark methods with include/exclude patterns."""

    def test_include_string_pattern(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test filtering benchmark methods with a string include pattern."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_one(self) -> int:
                return 1

            def bench_two(self) -> int:
                return 2

            def bench_three(self) -> int:
                return 3

        bench = FilterBench()
        bench.bench(include="two")

        captured = capsys.readouterr()
        assert "bench_one" not in captured.out
        assert "bench_two" in captured.out
        assert "bench_three" not in captured.out

    def test_include_or_pattern(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test filtering benchmark methods with a list of include patterns."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_one(self) -> int:
                return 1

            def bench_two(self) -> int:
                return 2

            def bench_three(self) -> int:
                return 3

        bench = FilterBench()
        bench.bench(include="one|three")

        captured = capsys.readouterr()
        assert "bench_one" in captured.out
        assert "bench_two" not in captured.out
        assert "bench_three" in captured.out

    def test_exclude_string_pattern(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test filtering benchmark methods with a string exclude pattern."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_one(self) -> int:
                return 1

            def bench_two(self) -> int:
                return 2

            def bench_three(self) -> int:
                return 3

        bench = FilterBench()
        bench.bench(exclude="two")

        captured = capsys.readouterr()
        assert "bench_one" in captured.out
        assert "bench_two" not in captured.out
        assert "bench_three" in captured.out

    def test_exclude_or_pattern(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test filtering benchmark methods with a list of exclude patterns."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_one(self) -> int:
                return 1

            def bench_two(self) -> int:
                return 2

            def bench_three(self) -> int:
                return 3

        bench = FilterBench()
        bench.bench(exclude="one|three")

        captured = capsys.readouterr()
        assert "bench_one" not in captured.out
        assert "bench_two" in captured.out
        assert "bench_three" not in captured.out

    def test_combined_include_exclude_patterns(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test combining include and exclude patterns."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_fast_one(self) -> int:
                return 1

            def bench_fast_two(self) -> int:
                return 2

            def bench_slow_three(self) -> int:
                return 3

        bench = FilterBench()
        # Include methods with "fast" but exclude those with "two"
        bench.bench(include="fast", exclude="two")

        captured = capsys.readouterr()
        assert "bench_fast_one" in captured.out
        assert "bench_fast_two" not in captured.out
        assert "bench_slow_three" not in captured.out

    def test_regex_patterns(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test using regex patterns for filtering."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_test_a1(self) -> int:
                return 1

            def bench_test_a2(self) -> int:
                return 2

            def bench_test_b1(self) -> int:
                return 3

            def bench_other(self) -> int:
                return 4

        bench = FilterBench()
        # Use regex to match methods ending with a number
        bench.bench(include=r"_[ab]\d$")

        captured = capsys.readouterr()
        assert "bench_test_a1" in captured.out
        assert "bench_test_a2" in captured.out
        assert "bench_test_b1" in captured.out
        assert "bench_other" not in captured.out

    def test_no_matches(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test behavior when no methods match the filtering criteria."""

        class FilterBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def bench_one(self) -> int:
                return 1

            def bench_two(self) -> int:
                return 2

        with caplog.at_level(logging.WARNING):
            bench = FilterBench()
            results = bench.bench(include="nonexistent")

            assert not results, "Results should be empty when no methods match"
            assert any(
                "No benchmark methods found to run" in record.message
                for record in caplog.records
            )

    def test_definition_order_preservation(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that method definition order is preserved."""

        class OrderBench(EasyBench):
            bench_config = BenchConfig(trials=1, sort_by="def")

            def bench_c(self) -> None:
                pass

            def bench_a(self) -> None:
                pass

            def bench_b(self) -> None:
                pass

        bench = OrderBench()
        bench.bench()

        captured = capsys.readouterr()
        # Extract the order from the output
        output_lines = captured.out.splitlines()
        c_idx = next(
            (i for i, line in enumerate(output_lines) if "bench_c" in line),
            -1,
        )
        a_idx = next(
            (i for i, line in enumerate(output_lines) if "bench_a" in line),
            -1,
        )
        b_idx = next(
            (i for i, line in enumerate(output_lines) if "bench_b" in line),
            -1,
        )

        # Verify that the methods appear in definition order
        assert c_idx < a_idx < b_idx, "Methods should be displayed in definition order"

    def test_instance_methods_discovery(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test discovery of benchmark methods defined at instance level."""

        class DynamicBench(EasyBench):
            bench_config = BenchConfig(trials=1)

            def __init__(self) -> None:
                super().__init__()
                # Add a dynamic instance method
                self.bench_dynamic = self._dynamic_method

            def _dynamic_method(self) -> int:
                return 42

            def bench_normal(self) -> int:
                return 10

        bench = DynamicBench()
        bench.bench()

        captured = capsys.readouterr()
        assert "bench_dynamic" in captured.out
        assert "bench_normal" in captured.out

    def test_bench_params_multiplication(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test BenchParams multiplication for combined parameter sets."""
        # Define base parameter sets
        small_size = 10
        large_size = 100
        sizes = [
            BenchParams(name="Small", params={"size": small_size}),
            BenchParams(name="Large", params={"size": large_size}),
        ]

        operations = [
            BenchParams(name="Add", fn_params={"operation": lambda x: x + 1}),
            BenchParams(name="Multiply", fn_params={"operation": lambda x: x * 2}),
        ]

        # Create combined parameter sets through multiplication
        combined_params = [size * op for size in sizes for op in operations]

        class MultiplicationBench(EasyBench):
            @parametrize(combined_params)
            def bench_operation(
                self,
                size: int,
                operation: Callable[[int], int],
            ) -> int:
                return operation(size)

        bench = MultiplicationBench()
        bench.bench()

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations are present
        assert "bench_operation (Small x Add)" in parsed_out["functions"]
        assert "bench_operation (Small x Multiply)" in parsed_out["functions"]
        assert "bench_operation (Large x Add)" in parsed_out["functions"]
        assert "bench_operation (Large x Multiply)" in parsed_out["functions"]

        # Verify correct results based on operations
        for func_name, func_data in parsed_out["functions"].items():
            if "Small x Add" in func_name and "output" in func_data:
                assert func_data["output"][0] == small_size + 1  # 10 + 1
            elif "Small x Multiply" in func_name and "output" in func_data:
                assert func_data["output"][0] == small_size * 2  # 10 * 2
            elif "Large x Add" in func_name and "output" in func_data:
                assert func_data["output"][0] == large_size + 1  # 100 + 1
            elif "Large x Multiply" in func_name and "output" in func_data:
                assert func_data["output"][0] == large_size * 2  # 100 * 2

    def test_bench_params_multiplication_params_only(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test BenchParams multiplication with params only."""
        # Define parameter sets with params only
        small_sizes = BenchParams(name="Small", params={"size": 10})
        large_sizes = BenchParams(name="Large", params={"size": 100})

        simple_data = BenchParams(name="Simple", params={"data_type": "list"})
        complex_data = BenchParams(name="Complex", params={"data_type": "dict"})

        # Create combined parameter sets through multiplication
        combined_params = [
            size * data
            for size in [small_sizes, large_sizes]
            for data in [simple_data, complex_data]
        ]

        class MultiplicationBench(EasyBench):
            @parametrize(combined_params)
            def bench_data_structure(
                self,
                size: int,
                data_type: str,
            ) -> int:
                if data_type == "list":
                    return len([0] * size)
                # dict
                return len({i: i for i in range(size)})

        bench = MultiplicationBench()
        bench.bench(config=PartialBenchConfig(show_output=True))

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations are present
        assert "bench_data_structure (Small x Simple)" in parsed_out["functions"]
        assert "bench_data_structure (Small x Complex)" in parsed_out["functions"]
        assert "bench_data_structure (Large x Simple)" in parsed_out["functions"]
        assert "bench_data_structure (Large x Complex)" in parsed_out["functions"]

        # Verify correct results for each combination
        assert "return_values" in parsed_out
        return_values = parsed_out["return_values"]

        assert return_values["bench_data_structure (Small x Simple)"] == "10"
        assert return_values["bench_data_structure (Small x Complex)"] == "10"
        assert return_values["bench_data_structure (Large x Simple)"] == "100"
        assert return_values["bench_data_structure (Large x Complex)"] == "100"

    def test_bench_params_multiplication_fn_params_only(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test BenchParams multiplication with fn_params only."""
        # Define parameter sets with fn_params only
        add_op = BenchParams(name="Add", fn_params={"operation": lambda x, y: x + y})
        mult_op = BenchParams(
            name="Multiply",
            fn_params={"operation": lambda x, y: x * y},
        )

        small_value = BenchParams(name="Small", fn_params={"value": lambda: 5})
        large_value = BenchParams(name="Large", fn_params={"value": lambda: 10})

        # Create combined parameter sets through multiplication
        combined_params = [
            op * val for op in [add_op, mult_op] for val in [small_value, large_value]
        ]

        class MultiplicationBench(EasyBench):
            @parametrize(combined_params)
            def bench_calculate(
                self,
                operation: Callable[[int, int], int],
                value: Callable[..., int],
            ) -> int:
                return operation(
                    value(),
                    2,
                )  # Apply operation with value and constant 2

        bench = MultiplicationBench()
        bench.bench(config=PartialBenchConfig(show_output=True))

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations are present
        assert "bench_calculate (Add x Small)" in parsed_out["functions"]
        assert "bench_calculate (Add x Large)" in parsed_out["functions"]
        assert "bench_calculate (Multiply x Small)" in parsed_out["functions"]
        assert "bench_calculate (Multiply x Large)" in parsed_out["functions"]

        # Verify correct results for each combination
        assert "return_values" in parsed_out
        return_values = parsed_out["return_values"]

        assert return_values["bench_calculate (Add x Small)"] == "7"  # 5 + 2
        assert return_values["bench_calculate (Add x Large)"] == "12"  # 10 + 2
        assert return_values["bench_calculate (Multiply x Small)"] == "10"  # 5 * 2
        assert return_values["bench_calculate (Multiply x Large)"] == "20"  # 10 * 2

    def test_bench_params_multiplication_mixed(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test BenchParams multiplication with both params and fn_params."""
        # Define parameter sets with both params and fn_params
        list_type = BenchParams(
            name="List",
            params={"collection_type": "list"},
            fn_params={"create": lambda size: list(range(size))},
        )
        dict_type = BenchParams(
            name="Dict",
            params={"collection_type": "dict"},
            fn_params={"create": lambda size: {i: i for i in range(size)}},
        )

        length = 100
        small_size = BenchParams(
            name="Small",
            params={"size": 10},
            fn_params={"check": lambda x: len(x) < length},
        )
        large_size = BenchParams(
            name="Large",
            params={"size": 100},
            fn_params={"check": lambda x: len(x) >= length},
        )

        # Create combined parameter sets through multiplication
        combined_params = [
            ctype * size
            for ctype in [list_type, dict_type]
            for size in [small_size, large_size]
        ]

        class MultiplicationBench(EasyBench):
            @parametrize(combined_params)
            def bench_collection(
                self,
                collection_type: str,
                size: int,
                create: Callable[[int], object],
                check: Callable[[object], bool],
            ) -> tuple[str, int, bool]:
                collection = create(size)
                is_valid = check(collection)
                return (collection_type, size, is_valid)

        bench = MultiplicationBench()
        bench.bench(config=PartialBenchConfig(show_output=True))

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations are present
        assert "bench_collection (List x Small)" in parsed_out["functions"]
        assert "bench_collection (List x Large)" in parsed_out["functions"]
        assert "bench_collection (Dict x Small)" in parsed_out["functions"]
        assert "bench_collection (Dict x Large)" in parsed_out["functions"]

        # Verify return values contain expected data
        # Small collections should pass the "small" check and fail the "large" check
        assert "('list', 10, True)" in captured.out
        assert "('dict', 10, True)" in captured.out

        # Large collections should fail the "small" check and pass the "large" check
        assert "('list', 100, True)" in captured.out
        assert "('dict', 100, True)" in captured.out

    def test_multiple_parametrize_decorators_variants(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test multiple parametrize decorators with different parameter types."""
        # Create separate parameter sets
        # params only
        sizes = [
            BenchParams(name="Small", params={"size": 10}),
            BenchParams(name="Large", params={"size": 100}),
        ]

        # fn_params only
        operations = [
            BenchParams(name="Add", fn_params={"operation": lambda x: x + 1}),
            BenchParams(name="Multiply", fn_params={"operation": lambda x: x * 2}),
        ]

        # mixed params and fn_params
        formats = [
            BenchParams(
                name="String",
                params={"format_type": "str"},
                fn_params={"formatter": lambda x: str(x)},
            ),
            BenchParams(
                name="Hex",
                params={"format_type": "hex"},
                fn_params={"formatter": lambda x: hex(x)},
            ),
        ]

        class CartesianBench(EasyBench):
            # Apply multiple parametrize decorators - should create a Cartesian product
            # This creates 8 combinations (2 sizes * 2 operations * 2 formats)
            @parametrize(formats)
            @parametrize(operations)
            @parametrize(sizes)
            def bench_operation(
                self,
                size: int,
                operation: Callable[[int], int],
                format_type: str,
                formatter: Callable[[int], str],
            ) -> str:
                result = operation(size)
                return f"{format_type}:{formatter(result)}"

        bench = CartesianBench()
        bench.bench(config=PartialBenchConfig(show_output=True))

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Should have 8 results
        assert len(parsed_out["functions"]) == (
            len(sizes) * len(operations) * len(formats)
        )

        # Verify some specific combinations
        combinations = [
            "bench_operation (Small x Add x String)",
            "bench_operation (Small x Multiply x Hex)",
            "bench_operation (Large x Add x String)",
            "bench_operation (Large x Multiply x Hex)",
        ]

        for combination in combinations:
            assert combination in parsed_out["functions"]

        # Verify return values match expected calculations
        assert "return_values" in parsed_out
        return_values = parsed_out["return_values"]

        assert return_values["bench_operation (Small x Add x String)"] == "str:11"
        assert "hex:0x14" in return_values["bench_operation (Small x Multiply x Hex)"]
        assert return_values["bench_operation (Large x Add x String)"] == "str:101"
        assert "hex:0xc8" in return_values["bench_operation (Large x Multiply x Hex)"]


class TestTimeOverrideScenarios:
    """Tests for scenarios where time measurement might be overridden."""

    def test_time_false_with_time_sort(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test behavior when time=False but sort_by requires time measurement."""

        class SortTimeBench(EasyBench):
            def bench_fast(self) -> None:
                pass

            def bench_slow(self) -> None:
                time.sleep(0.001)

        # Sort by avg time but with time=False
        # System should auto-enable time measurement to allow sorting
        bench = SortTimeBench()
        bench.bench(config=PartialBenchConfig(time=False, sort_by="avg"))

        captured = capsys.readouterr()
        # Time columns should be present despite time=False because of sort_by="avg"
        assert "Time" in captured.out or "Avg Time" in captured.out


class TestTimeDisabledAdvanced:
    """Advanced tests for time=False functionality in EasyBench."""

    def test_time_disabled_with_different_memory_units(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that time=False works with different memory units."""

        class MemoryUnitsBench(EasyBench):
            def bench_allocate(self) -> list:
                # Allocate some memory
                return [0] * 1_000_000

        memory_units = ["B", "KB", "MB", "GB"]
        for unit in memory_units:
            bench = MemoryUnitsBench()
            bench.bench(config=PartialBenchConfig(time=False, memory=unit))

            captured = capsys.readouterr()
            # Time columns should not be present
            assert "Time" not in captured.out
            assert "Avg Time" not in captured.out
            # Memory should be present with the correct unit
            assert (
                f"Mem ({unit})" in captured.out or f"Avg Mem ({unit})" in captured.out
            )

    def test_time_disabled_with_show_output(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that time=False works with show_output=True."""

        class OutputBench(EasyBench):
            def bench_return_value(self) -> str:
                return "Hello, World!"

        bench = OutputBench()
        bench.bench(config=PartialBenchConfig(time=False, show_output=True))

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        assert "Avg Time" not in captured.out
        # But output should be shown
        assert "Hello, World!" in captured.out

    def test_time_disabled_with_multiple_configurations(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test time=False with multiple other configuration options."""

        class ComplexBench(EasyBench):
            def bench_test1(self) -> list:
                time.sleep(0.001)  # Small sleep
                return [0] * 100_000

            def bench_test2(self) -> list:
                time.sleep(0.002)  # Longer sleep
                return [0] * 200_000

        bench = ComplexBench()
        bench.bench(
            config=PartialBenchConfig(
                time=False,  # Disable time
                memory="MB",  # Memory in MB
                loops_per_trial=10,  # Multiple loops
                warmups=2,  # With warmups
                show_output=True,  # Show return value
                sort_by="avg_memory",  # Sort by memory since time is disabled
                reverse=True,  # Reverse sort
            ),
        )

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        assert "Avg Time" not in captured.out
        # But all other features should work
        assert "MB" in captured.out  # Memory unit
        assert "[0" in captured.out  # Output showing

        # Check sorting by memory (test2 uses more memory so should be first)
        bench_test2_index = -1
        bench_test1_index = -1
        lines = captured.out.strip().split("\n")

        for i, line in enumerate(lines):
            if "bench_test2" in line:
                bench_test2_index = i
            elif "bench_test1" in line:
                bench_test1_index = i

        assert bench_test2_index >= 0, "Benchmark functions not found in output"
        assert bench_test1_index >= 0, "Benchmark functions not found in output"
        assert (
            bench_test2_index < bench_test1_index
        ), "Results not sorted correctly by memory usage"

    def test_both_time_and_memory_disabled(self) -> None:
        """Test that an error is raised when both time and memory are disabled."""
        # Test PartialBenchConfig
        with pytest.raises(
            ValueError,
            match="At least one of 'time' or 'memory' must be enabled",
        ):
            PartialBenchConfig(time=False, memory=False)

        # Test BenchConfig
        with pytest.raises(
            ValueError,
            match="At least one of 'time' or 'memory' must be enabled",
        ):
            BenchConfig(time=False, memory=False)

        # Test with EasyBench class
        class InvalidBench(EasyBench):
            def bench_test(self) -> None:
                pass

        bench = InvalidBench()
        with pytest.raises(
            ValueError,
            match="At least one of 'time' or 'memory' must be enabled",
        ):
            bench.bench(config=PartialBenchConfig(time=False, memory=False))


class TestEasyBenchClipOutliers:
    """Tests for the clip_outliers parameter in BenchConfig."""

    def test_clip_outliers_validation(self) -> None:
        """Test validation of the clip_outliers parameter."""
        # Valid values should work
        clip = 0.1
        config1 = PartialBenchConfig(clip_outliers=clip)
        assert config1.clip_outliers == clip

        clip2 = 0.4
        config2 = PartialBenchConfig(clip_outliers=clip2)
        assert config2.clip_outliers == clip2

        clip3 = 0.0
        config2 = PartialBenchConfig(clip_outliers=clip3)
        assert config2.clip_outliers == clip3

        # None should work (default)
        config3 = PartialBenchConfig()
        assert config3.clip_outliers is None

        with pytest.raises(
            ValueError,
            match="clip_outliers must be between 0.0 and 1.0",
        ):
            PartialBenchConfig(clip_outliers=-0.1)

        with pytest.raises(
            ValueError,
            match="clip_outliers must be between 0.0 and 1.0",
        ):
            PartialBenchConfig(clip_outliers=1.0)

    @patch("time.perf_counter")
    def test_clip_outliers_functionality(
        self,
        mock_perf_counter: mock.MagicMock,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that clip_outliers properly removes outliers from measurements."""
        # Create a sequence of mock times with an outlier
        # Normal values: 0.1s, outlier: 1.0s
        sec = 0.1
        mock_perf_counter.side_effect = [
            # First trial: 0.1s
            0.0,
            sec,
            # Second trial: 0.1s
            sec,
            sec * 2,
            # Third trial: 0.1s
            sec * 2,
            sec * 3,
            # Fourth trial: 0.1s
            sec * 3,
            sec * 4,
            # Fifth trial: 1.0s (outlier)
            sec * 4,
            1 + sec * 4,
        ]

        class ClipOutliersBench(EasyBench):
            def bench_test(self) -> None:
                # This function doesn't actually do anything since we're mocking time
                pass

        # Test without clipping outliers
        bench1 = ClipOutliersBench()
        bench1.bench(trials=5)

        captured1 = capsys.readouterr()
        parsed_out1 = parse_benchmark_output(captured1.out)

        # With no clipping, the average should include the outlier
        avg_with_outlier = parsed_out1["functions"]["bench_test"]["avg"]
        assert avg_with_outlier == pytest.approx((1 + sec * 4) / 5)

        # Reset the mock for next test
        mock_perf_counter.reset_mock()
        mock_perf_counter.side_effect = [
            # First trial: 0.1s
            0.0,
            sec,
            # Second trial: 0.1s
            sec,
            sec * 2,
            # Third trial: 0.1s
            sec * 2,
            sec * 3,
            # Fourth trial: 0.1s
            sec * 3,
            sec * 4,
            # Fifth trial: 1.0s (outlier)
            sec * 4,
            1 + sec * 4,
        ]

        # Test with clipping outliers
        bench2 = ClipOutliersBench()
        bench2.bench(trials=5, clip_outliers=0.1)  # Clip 10% from each end

        captured2 = capsys.readouterr()
        parsed_out2 = parse_benchmark_output(captured2.out)

        # With clipping, the average should exclude the outlier
        avg_with_clipping = parsed_out2["functions"]["bench_test"]["avg"]
        assert avg_with_clipping < (1 + sec * 4) / 5

    @patch("time.perf_counter")
    def test_clip_outliers_edge_cases(
        self,
        mock_perf_counter: mock.MagicMock,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test clip_outliers with edge case values."""
        # Create sequence of mock times with high variance
        # 0.01s, 0.05s, 0.1s, 0.5s, 1.0s
        mock_perf_counter.side_effect = [
            0.0,
            0.01,  # Trial 1
            0.01,
            0.06,  # Trial 2
            0.06,
            0.16,  # Trial 3
            0.16,
            0.66,  # Trial 4
            0.66,
            1.66,  # Trial 5
        ]

        class ClipEdgeCasesBench(EasyBench):
            def bench_test(self) -> None:
                pass

        # Test with small clip value (0.01)
        bench1 = ClipEdgeCasesBench()
        bench1.bench(trials=5, clip_outliers=0.01)

        captured1 = capsys.readouterr()
        parsed_out1 = parse_benchmark_output(captured1.out)

        # Should still include most values
        avg1 = parsed_out1["functions"]["bench_test"]["avg"]

        # Reset for next test
        mock_perf_counter.reset_mock()
        mock_perf_counter.side_effect = [
            0.0,
            0.01,  # Trial 1
            0.01,
            0.06,  # Trial 2
            0.06,
            0.16,  # Trial 3
            0.16,
            0.66,  # Trial 4
            0.66,
            1.66,  # Trial 5
        ]

        # Test with large clip value (0.49)
        bench2 = ClipEdgeCasesBench()
        bench2.bench(trials=5, clip_outliers=0.49)

        captured2 = capsys.readouterr()
        parsed_out2 = parse_benchmark_output(captured2.out)

        # Should exclude most extreme values
        avg2 = parsed_out2["functions"]["bench_test"]["avg"]

        # With more aggressive clipping, average should be closer to the median
        assert avg2 < avg1

    @patch("time.perf_counter")
    def test_clip_outliers_with_few_samples(
        self,
        mock_perf_counter: mock.MagicMock,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test clip_outliers when there are few samples."""
        # Create a sequence with just 3 trials
        middle = 0.1
        mock_perf_counter.side_effect = [
            0.0,
            middle,  # Trial 1 (0.1)
            middle,
            middle * 2,  # Trial 2 (0.1)
            middle * 2,
            1 + middle * 2,  # Trial 3 (1.0)
            1 + middle * 2,
            1 + middle * 3,  # Trial 4 (0.1)
            1 + middle * 3,
            1 + middle * 4,  # Trial 5 (0.1)
        ]

        class SmallSampleBench(EasyBench):
            def bench_test(self) -> None:
                pass

        # Test with clipping on a small sample
        bench = SmallSampleBench()
        bench.bench(trials=5, clip_outliers=0.25)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        avg = parsed_out["functions"]["bench_test"]["avg"]
        assert avg == middle

    def test_clip_outliers_runtime_override(self) -> None:
        """Test that clip_outliers can be overridden at runtime."""
        # Create a base config with clip_outliers
        clip = 0.1
        base_config = BenchConfig(trials=3, clip_outliers=clip)
        assert base_config.clip_outliers == clip

        # Override with a partial config
        clip2 = 0.2
        partial_config = PartialBenchConfig(clip_outliers=clip2)
        merged = partial_config.merge_with(base_config)
        assert merged.clip_outliers == clip2

    def test_clip_outliers_with_memory(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that clip_outliers works correctly with memory measurements."""
        outlier_count = 3

        class MemoryClipBench(EasyBench):
            def __init__(self) -> None:
                super().__init__()
                self.trial_counter = 0

            def bench_memory_outlier(self) -> list[int]:
                # Increment trial counter
                self.trial_counter += 1

                # For one specific trial, allocate significantly more memory (outlier)
                if self.trial_counter == outlier_count:  # Third trial is the outlier
                    return allocate_memory(1000)  # Large memory allocation
                return allocate_memory(100)  # Normal memory allocation

        # Test without clipping outliers
        bench1 = MemoryClipBench()
        bench1.bench(trials=5, memory=True)

        captured1 = capsys.readouterr()
        parsed_out1 = parse_benchmark_output(captured1.out)

        # With no clipping, the average should include the outlier
        avg_with_outlier = parsed_out1["functions"]["bench_memory_outlier"][
            "avg_memory"
        ]

        # Test with clipping outliers
        bench2 = MemoryClipBench()
        bench2.bench(trials=5, memory=True, clip_outliers=0.1)  # Clip 10% from each end

        captured2 = capsys.readouterr()
        parsed_out2 = parse_benchmark_output(captured2.out)

        # With clipping, the average should exclude the outlier
        avg_with_clipping = parsed_out2["functions"]["bench_memory_outlier"][
            "avg_memory"
        ]

        # The clipped average should be lower since the high outlier is removed
        assert avg_with_clipping < avg_with_outlier

        # The clipped average should be closer to the normal allocation size
        assert abs(avg_with_clipping - 100) < abs(avg_with_outlier - 100)
