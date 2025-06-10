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
from collections.abc import Callable, Generator
from typing import Any, ClassVar, cast

import pytest

from easybench import BenchConfig, EasyBench, fixture
from easybench.core import (
    BenchParams,
    FixtureRegistry,
    PartialBenchConfig,
    ScopeType,
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


class TestFunctionBench:
    """Tests for the FunctionBench class."""

    def test_function_wrapper_creation(self) -> None:
        """Test creating a FunctionBench wrapper for a function."""

        def my_function(a: int, b: int) -> int:
            return a + b

        from easybench.core import FunctionBench

        fb = FunctionBench(my_function)
        assert fb._original_func == my_function
        assert fb._func_name == "my_function"
        assert hasattr(fb, "bench_my_function")

    def test_function_direct_call(self) -> None:
        """Test directly calling a FunctionBench object as a function."""

        def add(a: int, b: int) -> int:
            return a + b

        from easybench.core import FunctionBench

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

        from easybench.core import FunctionBench

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
        from easybench.core import FunctionBench

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
        from easybench.core import FunctionBench

        with pytest.raises(ValueError, match="func_name must be specified"):
            FunctionBench(lambda x: x * 2)

    def test_non_callable_function(self) -> None:
        """Test that non-callable objects raise TypeError."""
        from easybench.core import FunctionBench

        with pytest.raises(TypeError, match="func must be callable"):
            FunctionBench(cast("Callable[..., object]", "not_callable"))

    def test_function_bench_void_return(self) -> None:
        """Test FunctionBench with a function that returns None."""
        from easybench.core import FunctionBench

        def void_function() -> None:
            pass

        fb = FunctionBench(void_function)
        result = fb()
        assert result is None

    def test_function_bench_with_args_kwargs(self) -> None:
        """Test FunctionBench with positional and keyword arguments."""
        from easybench.core import FunctionBench

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


class TestEasyBenchConfiguration:
    """Additional tests for EasyBench configuration."""

    def test_merge_partial_config_empty(self) -> None:
        """Test merging an empty partial config."""
        from easybench.core import BenchConfig, PartialBenchConfig

        base_config = BenchConfig(trials=CONFIG_TRIALS_5, memory=True)
        partial_config = PartialBenchConfig()

        merged = partial_config.merge_with(base_config)

        assert merged.trials == CONFIG_TRIALS_5
        assert merged.memory is True

    def test_merge_partial_config_reporters(self) -> None:
        """Test merging partial config with reporters."""
        from easybench.core import BenchConfig, PartialBenchConfig
        from easybench.reporters import ConsoleReporter, TableFormatter

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
        from easybench.core import BenchConfig, PartialBenchConfig

        base_config = BenchConfig(trials=5)
        # Create partial config with empty reporters list
        partial_config = PartialBenchConfig(reporters=[])

        merged = partial_config.merge_with(base_config)

        assert len(merged.reporters) == 0

    def test_merge_partial_config_with_reporters_none(self) -> None:
        """Test merging partial config with None reporters."""
        from easybench.core import BenchConfig, PartialBenchConfig
        from easybench.reporters import ConsoleReporter, TableFormatter

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
        from easybench.core import PartialBenchConfig

        result = PartialBenchConfig.validate_and_convert_reporters(None)
        assert result is None

    def test_validate_and_convert_reporters_strings(self) -> None:
        """Test converting a list of reporter strings."""
        from easybench.core import PartialBenchConfig
        from easybench.reporters import ConsoleReporter, SimpleConsoleReporter

        reporters = ["console", "simple"]
        result = PartialBenchConfig.validate_and_convert_reporters(reporters)

        assert result is not None
        assert len(result) == len(reporters)
        assert isinstance(result[0], ConsoleReporter)
        assert isinstance(result[1], SimpleConsoleReporter)

    def test_validate_and_convert_reporters_with_kwargs(self) -> None:
        """Test converting a reporter specification with kwargs."""
        from easybench.core import PartialBenchConfig
        from easybench.reporters import SimpleConsoleReporter

        # Create a tuple of (name, kwargs)
        reporter_spec = [("simple", {"item_format": lambda n, v: f"{n}: {v}"})]
        result = PartialBenchConfig.validate_and_convert_reporters(reporter_spec)

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], SimpleConsoleReporter)

    def test_validate_and_convert_reporters_with_instances(self) -> None:
        """Test that Reporter instances are kept as-is."""
        from easybench.core import PartialBenchConfig
        from easybench.reporters import ConsoleReporter, TableFormatter

        # Create a reporter instance
        reporter = ConsoleReporter(TableFormatter())
        reporters = [reporter]

        result = PartialBenchConfig.validate_and_convert_reporters(reporters)

        assert result is not None
        assert len(result) == 1
        assert result[0] is reporter  # Should be the exact same instance

    def test_validate_and_convert_reporters_mixed(self) -> None:
        """Test converting a mixed list of reporter specifications."""
        from easybench.core import PartialBenchConfig
        from easybench.reporters import (
            ConsoleReporter,
            SimpleConsoleReporter,
            TableFormatter,
        )

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
        from easybench.core import PartialBenchConfig

        with pytest.raises(TypeError, match="reporters must be a list"):
            PartialBenchConfig.validate_and_convert_reporters("console")  # type: ignore [arg-type]

    def test_validate_and_convert_reporters_invalid_item(self) -> None:
        """Test that invalid item in list raises TypeError."""
        from easybench.core import PartialBenchConfig

        with pytest.raises(TypeError, match="Invalid reporter type:"):
            PartialBenchConfig.validate_and_convert_reporters([123])


class TestEasyBenchScopeManager:
    """Tests for the ScopeManager class in EasyBench."""

    def test_scope_manager_invalid_scope(self) -> None:
        """Test ScopeManager with an invalid scope."""
        from easybench.core import EasyBench

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
        from easybench.core import EasyBench

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
        from easybench.core import EasyBench

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
        from collections.abc import Callable

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
        from collections.abc import Generator

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
