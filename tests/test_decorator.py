"""
Tests for the bench decorator functionality in easybench.

This module tests various aspects of the bench decorator, including:
- Output formatting
- Time measurement
- Memory usage measurement
- Function parameter handling
- Configuration options
- Return value display
- Single and multiple trial handling
"""

import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, cast

import pytest

from easybench import BenchConfig, BenchParams, bench

if TYPE_CHECKING:
    from easybench.decorator import BenchmarkableFunction

# Constants for magic values
DEFAULT_TRIALS = 5
SINGLE_TRIAL = 1
MULTIPLE_TRIALS = 3
SLEEP_TIME = 0.05
# Add tolerance for time comparisons to handle Windows timer resolution
TIME_COMPARISON_TOLERANCE = 0.01
SMALL_KB = 100
MEDIUM_KB = 500
LARGE_KB = 1000
EXPECTED_VALUE = 10


class TestBenchDecoratorOutput:
    """Tests for the output formatting of the bench decorator."""

    def test_bench_decorator_display(self, capsys: pytest.CaptureFixture) -> None:
        """Test that basic benchmark output is displayed correctly."""

        @bench.config(progress=False)
        def test() -> int:
            return 10

        captured = capsys.readouterr()
        out = captured.out
        assert captured.err == ""
        assert "Benchmark Results" in out
        assert "test" in out
        assert "bench_test" not in out
        assert "Avg Time" in out
        assert "Min Time" in out
        assert "Max Time" in out
        assert "0.000" in out

    def test_bench_decorator_display2_insufficient_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that no output is generated when required parameters are missing."""

        @bench
        def test2(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bench_decorator_display2_sufficient_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that output is generated when required parameters are provided."""

        @bench(value=10)
        def test2(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "test2" in captured.out
        assert "bench_test" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display3_insufficient_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that no output when some required parameters are missing."""

        @bench(value=10)
        def test3(value: int, value2: int) -> int:
            return value + value2

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bench_decorator_display3_sufficient_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that output is generated when all required parameters are provided."""

        @bench(value=10, value2=20)
        def test3(value: int, value2: int) -> int:
            return value + value2

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "test3" in captured.out
        assert "bench_test" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display_config(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that bench.config properly configures the number of trials."""

        @bench.config(trials=MULTIPLE_TRIALS)
        def test() -> int:
            return 10

        captured = capsys.readouterr()
        assert f"Benchmark Results ({MULTIPLE_TRIALS} trials)" in captured.out
        assert "test" in captured.out
        assert "bench_test" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display_config2(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that bench decorator can be combined with config decorator."""

        @bench(value=10)
        @bench.config(trials=MULTIPLE_TRIALS)
        def test(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert f"Benchmark Results ({MULTIPLE_TRIALS} trials)" in captured.out
        assert "test" in captured.out
        assert "bench_test" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display_config2_insufficient(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that config doesn't run benchmarks when parameters are insufficient."""

        @bench.config(trials=MULTIPLE_TRIALS)
        def test(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bench_decorator_display_config_one_trial(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test the output format when only one trial is configured."""

        @bench(value=10)
        @bench.config(trials=SINGLE_TRIAL)
        def function1(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert f"Benchmark Results ({SINGLE_TRIAL} trial)" in captured.out
        assert "function1" in captured.out
        assert "bench_function1" not in captured.out
        assert "Avg Time" not in captured.out
        assert "Min Time" not in captured.out
        assert "Max Time" not in captured.out
        assert "Time" in captured.out

    def test_bench_decorator_config_with_benchconfig(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that bench.config accepts a BenchConfig object."""
        custom_config = BenchConfig(trials=SINGLE_TRIAL, memory=True, show_output=True)

        @bench(value=10)
        @bench.config(custom_config)
        def function_with_config(value: int) -> int:
            return value

        captured = capsys.readouterr()
        assert f"Benchmark Results ({SINGLE_TRIAL} trial)" in captured.out
        assert "function_with_config" in captured.out
        assert "bench_function_with_config" not in captured.out
        assert "Time" in captured.out
        assert "Memory" in captured.out
        assert "Return Values" in captured.out
        assert "10" in captured.out  # Verify return value is shown

    def test_bench_decorator_config_with_benchconfig_and_kwargs(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that kwargs override values in BenchConfig when both are provided."""
        custom_config = BenchConfig(trials=MULTIPLE_TRIALS, memory=False)

        @bench(value=10)
        @bench.config(custom_config, trials=SINGLE_TRIAL, memory=True)
        def function_with_config_override(value: int) -> int:
            return value

        captured = capsys.readouterr()
        # Should use SINGLE_TRIAL from kwargs, not MULTIPLE_TRIALS from config
        assert f"Benchmark Results ({SINGLE_TRIAL} trial)" in captured.out
        # Should show memory because kwargs override config
        assert "Memory" in captured.out
        assert "function_with_config_override" in captured.out

    def test_bench_decorator_display_fn_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that fn_params properly sets function parameters for benchmarking."""

        @bench.fn_params(func=lambda x: x)
        def function_test(func: Callable[[Any], Any]) -> int:
            return func(10)

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "function_test" in captured.out
        assert "bench_function_test" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display_fn_params2_insufficient(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that fn_params doesn't run benchmarks when missing parameters."""

        @bench.fn_params(func=lambda x: x)
        def function_test2(func: Callable[[Any], Any], value: int) -> int:
            return func(value)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_bench_decorator_display_fn_params2_sufficient(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that fn_params combined with bench with all parameters."""

        @bench(value=100)
        @bench.fn_params(func=lambda x: x)
        def function_test2(func: Callable[[Any], Any], value: int) -> int:
            return func(value)

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "function_test2" in captured.out
        assert "bench_function_test2" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_function_params(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that lambda functions can be used as fixture values."""

        @bench(number=lambda: 10)
        def plus_one(number: int) -> int:
            return number + 1

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "plus_one" in captured.out
        assert "bench_plus_one" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_function_params2(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that named functions can be used as fixture values."""

        def get_list() -> list[int]:
            return list(range(100))

        @bench(big_list=get_list)
        def append_list(big_list: list[int]) -> None:
            big_list.append(-1)

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "append_list" in captured.out
        assert "bench_append_list" not in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_display_config_memory(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that memory measurement is enabled when configured."""

        @bench.config(memory=True)
        def print_ten() -> int:
            return 10

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "print_ten" in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out
        assert "Avg Mem" in captured.out
        assert "Max Mem" in captured.out

    def test_bench_decorator_display_config_no_memory(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that memory measurement is disabled when configured."""

        @bench.config(memory=False)
        def print_ten() -> int:
            return 10

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "print_ten" in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out
        assert "Avg Mem" not in captured.out
        assert "Max Mem" not in captured.out

    def test_bench_decorator_bench_method(self, capsys: pytest.CaptureFixture) -> None:
        """Test direct invocation of the bench method on a decorated function."""
        # Constant for expected return value

        @bench
        def return_value(value: int) -> int:
            return value

        decorated_func = cast("BenchmarkableFunction", return_value)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        decorated_func(EXPECTED_VALUE)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        x = decorated_func.bench(EXPECTED_VALUE)
        assert x == EXPECTED_VALUE

        captured = capsys.readouterr()
        assert "Benchmark Results (1 trial)" in captured.out
        assert "return_value" in captured.out
        assert "Avg Time" not in captured.out
        assert "Min Time" not in captured.out
        assert "Max Time" not in captured.out
        assert "Time" in captured.out

    def test_bench_decorator_bench_method_trials(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test specifying the number of trials when directly invoking bench."""

        @bench
        def return_value(value: int) -> int:
            return value

        decorated_func = cast("BenchmarkableFunction", return_value)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        decorated_func.bench(100, bench_trials=13)

        captured = capsys.readouterr()
        assert "Benchmark Results (13 trials)" in captured.out
        assert "return_value" in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_decorator_parse_output(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output for basic configuration."""

        @bench
        def test() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == DEFAULT_TRIALS
        assert "test" in parsed_out["functions"]
        assert not parsed_out["is_single_trial"]
        assert not parsed_out["has_memory_metrics"]
        assert not parsed_out["has_return_values"]

    def test_bench_decorator_parse_output2(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output for single trial configuration."""

        @bench.config(trials=SINGLE_TRIAL)
        def test2() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == SINGLE_TRIAL
        assert "test2" in parsed_out["functions"]
        assert parsed_out["is_single_trial"]
        assert not parsed_out["has_memory_metrics"]
        assert not parsed_out["has_return_values"]

    def test_bench_decorator_parse_output3(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output with memory metrics enabled."""

        @bench.config(trials=MULTIPLE_TRIALS, memory=True)
        def test3() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == MULTIPLE_TRIALS
        assert "test3" in parsed_out["functions"]
        assert not parsed_out["is_single_trial"]
        assert parsed_out["has_memory_metrics"]
        assert not parsed_out["has_return_values"]

    def test_bench_decorator_parse_output4(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output for single trial with memory metrics."""

        @bench.config(trials=SINGLE_TRIAL, memory=True)
        def test4() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == SINGLE_TRIAL
        assert "test4" in parsed_out["functions"]
        assert parsed_out["is_single_trial"]
        assert parsed_out["has_memory_metrics"]
        assert not parsed_out["has_return_values"]

    def test_bench_decorator_parse_output5(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output with return values displayed."""

        @bench.config(trials=SINGLE_TRIAL, memory=True, show_output=True)
        def test4() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == SINGLE_TRIAL
        assert "test4" in parsed_out["functions"]
        assert parsed_out["is_single_trial"]
        assert parsed_out["has_memory_metrics"]
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["test4"] == "10"

    def test_bench_decorator_parse_output6(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the output with different return values across trials."""
        count = 0

        @bench.config(trials=MULTIPLE_TRIALS, memory=True, show_output=True)
        def test5() -> int:
            nonlocal count
            count += 1
            return count

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == MULTIPLE_TRIALS
        assert "test5" in parsed_out["functions"]
        assert not parsed_out["is_single_trial"]
        assert parsed_out["has_memory_metrics"]
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["test5"] == ["1", "2", "3"]

    def test_bench_decorator_parse_output7(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the output for single trial with return values but no memory."""

        @bench.config(trials=SINGLE_TRIAL, memory=False, show_output=True)
        def test4() -> int:
            return 10

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == SINGLE_TRIAL
        assert "test4" in parsed_out["functions"]
        assert parsed_out["is_single_trial"]
        assert not parsed_out["has_memory_metrics"]
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["test4"] == "10"

    def test_bench_decorator_parse_output8(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the output for multi trials with return values but no memory."""
        count = 0

        @bench.config(trials=MULTIPLE_TRIALS, memory=False, show_output=True)
        def test5() -> int:
            nonlocal count
            count += 1
            return count

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == MULTIPLE_TRIALS
        assert "test5" in parsed_out["functions"]
        assert not parsed_out["is_single_trial"]
        assert not parsed_out["has_memory_metrics"]
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["test5"] == ["1", "2", "3"]

    def test_bench_decorator_parse_output9(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the output when using a mutable object across trials."""

        @bench(big_list=list(range(100)))
        @bench.config(trials=MULTIPLE_TRIALS, show_output=True)
        def append_list(big_list: list[int]) -> int:
            big_list.append(123)
            return len(big_list)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == MULTIPLE_TRIALS
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["append_list"] == ["101", "102", "103"]

    def test_bench_decorator_parse_output10(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test parsing the benchmark output when using a lambda-generated fixture."""

        @bench(big_list=lambda: list(range(100)))
        @bench.config(trials=MULTIPLE_TRIALS, show_output=True)
        def append_list(big_list: list[int]) -> int:
            big_list.append(123)
            return len(big_list)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert parsed_out["trials"] == MULTIPLE_TRIALS
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["append_list"] == "101"

    def test_reset_bench_functionality(self, capsys: pytest.CaptureFixture) -> None:
        """Test that bench configuration gets reset between runs."""

        @bench
        def first_func() -> int:
            return 5

        captured = capsys.readouterr()  # Capture first run output

        # Run again with different config
        @bench.config(trials=SINGLE_TRIAL, show_output=True)
        def second_func() -> int:
            return 10

        captured = capsys.readouterr()  # Capture second run output
        assert f"Benchmark Results ({SINGLE_TRIAL} trial)" in captured.out
        assert "second_func" in captured.out
        assert "Return Values" in captured.out
        assert "10" in captured.out

        @bench
        def third_func() -> int:
            return 10

        captured = capsys.readouterr()  # Capture second run output
        assert f"Benchmark Results ({DEFAULT_TRIALS} trials)" in captured.out


class TestBenchDecoratorTime:
    """Tests for the time measurement functionality of the bench decorator."""

    def test_bench_decorator_time(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that time measurement works properly with a single trial."""

        @bench(seconds=SLEEP_TIME)
        @bench.config(trials=SINGLE_TRIAL)
        def sleep_func(seconds: float) -> float:
            time.sleep(seconds)
            return seconds

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "sleep_func" in parsed_out["functions"]
        # Add tolerance to timing assertion for Windows timer resolution
        assert (
            parsed_out["functions"]["sleep_func"]["time"]
            >= SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )

    def test_bench_decorator_time2(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that time measurement works properly with multiple trials."""

        @bench(seconds=SLEEP_TIME)
        @bench.config(trials=MULTIPLE_TRIALS)
        def sleep_func(seconds: float) -> float:
            time.sleep(seconds)
            return seconds

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "sleep_func" in parsed_out["functions"]
        # Add tolerance to timing assertions for Windows timer resolution
        assert (
            parsed_out["functions"]["sleep_func"]["avg"]
            >= SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )
        assert (
            parsed_out["functions"]["sleep_func"]["min"]
            >= SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )
        assert (
            parsed_out["functions"]["sleep_func"]["max"]
            >= SLEEP_TIME - TIME_COMPARISON_TOLERANCE
        )

    def test_bench_decorator_time3(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that time measurement doesn't include the fixture setup time."""

        def get_value() -> int:
            time.sleep(SLEEP_TIME)
            return 10

        @bench(value=get_value)
        @bench.config(trials=MULTIPLE_TRIALS)
        def return_value(value: int) -> int:
            return value

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "return_value" in parsed_out["functions"]
        # Given Windows timer resolution,
        # we should just check it's significantly smaller
        # rather than using a strict comparison that may fail due to timer imprecision
        assert parsed_out["functions"]["return_value"]["avg"] < SLEEP_TIME / 2
        assert parsed_out["functions"]["return_value"]["min"] < SLEEP_TIME / 2

    def test_bench_decorator_time4(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that time measurement doesn't include the fixture teardown time."""

        def get_value() -> Generator[int, None, None]:
            yield 10
            time.sleep(SLEEP_TIME)

        @bench(value=get_value)
        @bench.config(trials=MULTIPLE_TRIALS)
        def return_value(value: int) -> int:
            return value

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)
        assert "return_value" in parsed_out["functions"]
        # Given Windows timer resolution,
        # we should just check it's significantly smaller
        # rather than using a strict comparison that may fail due to timer imprecision
        assert parsed_out["functions"]["return_value"]["avg"] < SLEEP_TIME / 2
        assert parsed_out["functions"]["return_value"]["min"] < SLEEP_TIME / 2


class TestBenchDecoratorMemory:
    """Tests for the memory measurement functionality of the bench decorator."""

    @pytest.mark.parametrize("kb_size", [SMALL_KB, MEDIUM_KB, LARGE_KB])
    def test_bench_decorator_memory_basic(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
        kb_size: int,
    ) -> None:
        """Test basic memory measurement with a predictable memory allocation."""

        @bench(kb_size=kb_size)
        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def allocate_memory_(kb_size: int) -> None:
            allocate_memory(kb_size)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert "allocate_memory_" in parsed_out["functions"]

        assert parsed_out["has_memory_metrics"]

        assert (
            kb_size
            <= parsed_out["functions"]["allocate_memory_"]["avg_memory"]
            < kb_size * 2
        )
        assert (
            kb_size
            <= parsed_out["functions"]["allocate_memory_"]["max_memory"]
            < kb_size * 2
        )

    def test_bench_decorator_memory_before(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that memory allocated before the benchmark isn't counted."""

        def allocate_memory_(kb_size: int) -> int:
            allocate_memory(kb_size)
            return kb_size

        kb_size = LARGE_KB

        @bench(kb_size=allocate_memory(kb_size))
        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def no_allocate_memory(kb_size: int) -> int:
            return kb_size

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert "no_allocate_memory" in parsed_out["functions"]

        assert parsed_out["has_memory_metrics"]

        assert parsed_out["functions"]["no_allocate_memory"]["avg_memory"] < kb_size
        assert parsed_out["functions"]["no_allocate_memory"]["max_memory"] < kb_size

    def test_bench_decorator_memory_before2(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that memory allocated within fixture isn't counted in the benchmark."""

        def allocate_memory_(kb_size: int) -> int:
            allocate_memory(kb_size)
            return kb_size

        kb_size = LARGE_KB

        @bench(kb_size=lambda kb_size=kb_size: allocate_memory(kb_size))
        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def no_allocate_memory(kb_size: int) -> int:
            return kb_size

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert "no_allocate_memory" in parsed_out["functions"]

        assert parsed_out["has_memory_metrics"]

        assert parsed_out["functions"]["no_allocate_memory"]["avg_memory"] < kb_size
        assert parsed_out["functions"]["no_allocate_memory"]["max_memory"] < kb_size

    def test_bench_decorator_memory_after(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that memory allocated during teardown isn't counted in the benchmark."""

        def allocate_memory_(kb_size: int) -> Generator[int, None, None]:
            yield kb_size
            allocate_memory(kb_size)

        kb_size = LARGE_KB

        @bench(kb_size=lambda kb_size=kb_size: allocate_memory(kb_size))
        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def no_allocate_memory(kb_size: int) -> int:
            return kb_size

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        assert "no_allocate_memory" in parsed_out["functions"]

        assert parsed_out["has_memory_metrics"]

        assert parsed_out["functions"]["no_allocate_memory"]["avg_memory"] < kb_size
        assert parsed_out["functions"]["no_allocate_memory"]["max_memory"] < kb_size

    def test_memory_comparison(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
        allocate_memory: Callable[[int], list[int]],
    ) -> None:
        """Test that relative memory usage between functions is measured accurately."""

        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def small_alloc() -> None:
            allocate_memory(SMALL_KB)

        captured_small = capsys.readouterr()
        parsed_small = parse_benchmark_output(captured_small.out)

        @bench.config(memory=True, trials=MULTIPLE_TRIALS)
        def large_alloc() -> None:
            allocate_memory(MEDIUM_KB)

        captured_large = capsys.readouterr()
        parsed_large = parse_benchmark_output(captured_large.out)

        assert (
            parsed_small["functions"]["small_alloc"]["avg_memory"]
            < parsed_large["functions"]["large_alloc"]["avg_memory"]
        )


class TestBenchParamsDecorator:
    """Tests for the BenchParams functionality with bench decorator."""

    def test_bench_param_basic(self, capsys: pytest.CaptureFixture) -> None:
        """Test basic usage of BenchParams with bench decorator."""
        params = BenchParams(
            params={"value": 10},
        )

        @bench(params)
        def test_func(value: int) -> int:
            return value * 2

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "test_func" in captured.out
        assert "Avg Time" in captured.out
        assert "Min Time" in captured.out
        assert "Max Time" in captured.out

    def test_bench_param_with_config(self, capsys: pytest.CaptureFixture) -> None:
        """Test BenchParams with configuration options."""
        params = BenchParams(
            params={"value": 10},
        )

        @bench(params)
        @bench.config(trials=SINGLE_TRIAL, memory=True)
        def test_func(value: int) -> int:
            return value * 2

        captured = capsys.readouterr()
        assert f"Benchmark Results ({SINGLE_TRIAL} trial)" in captured.out
        assert "test_func" in captured.out
        assert "Time" in captured.out
        assert "Memory" in captured.out
        # Single trial format doesn't have Avg/Min/Max
        assert "Avg Time" not in captured.out

    def test_bench_param_with_fn_params(self, capsys: pytest.CaptureFixture) -> None:
        """Test BenchParams with function parameters."""

        def multiply(x: int) -> int:
            return x * 2

        params = BenchParams(
            params={"value": 10},
            fn_params={"operation": multiply},
        )

        @bench(params)
        def test_func(value: int, operation: Callable[[int], int]) -> int:
            return operation(value)

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "test_func" in captured.out
        assert "Avg Time" in captured.out

    def test_bench_param_with_lambda_callable(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test BenchParams with lambda functions."""

        # Create a function that returns a transformer function
        def get_transformer() -> Callable[[int], int]:
            return lambda x: x * 3

        params = BenchParams(params={"value": 10, "transformer": get_transformer})

        @bench(params)
        def test_func(value: int, transformer: Callable[[int], int]) -> int:
            return transformer(value)

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "test_func" in captured.out
        assert "Avg Time" in captured.out

    def test_bench_param_reuse(self, capsys: pytest.CaptureFixture) -> None:
        """Test reusing the same BenchParams for multiple functions."""
        params = BenchParams(params={"value": 10})

        @bench(params)
        def test_func1(value: int) -> int:
            return value * 2

        captured1 = capsys.readouterr()
        assert "Benchmark Results (5 trials)" in captured1.out
        assert "test_func1" in captured1.out

        @bench(params)  # Reuse the same params
        def test_func2(value: int) -> int:
            return value + 5

        captured2 = capsys.readouterr()
        assert "Benchmark Results (5 trials)" in captured2.out
        assert "test_func2" in captured2.out

    def test_bench_param_with_list(self, capsys: pytest.CaptureFixture) -> None:
        """Test using a list of BenchParams for parameter comparison."""
        # Create multiple parameter sets for comparison
        params1 = BenchParams(
            name="Small",
            params={"size": 100},
        )

        params2 = BenchParams(
            name="Medium",
            params={"size": 500},
        )

        params3 = BenchParams(
            name="Large",
            params={"size": 1000},
        )

        # Use the list of params with the bench decorator
        @bench([params1, params2, params3])
        @bench.config(trials=2)
        def create_sorted_list(size: int) -> list[int]:
            return sorted(range(size))

        captured = capsys.readouterr()

        # Verify comparison output
        assert "Benchmark Results" in captured.out
        assert "create_sorted_list (Small)" in captured.out
        assert "create_sorted_list (Medium)" in captured.out
        assert "create_sorted_list (Large)" in captured.out
        assert "Avg Time" in captured.out
        # Each parameter set should have been executed
        assert (
            "100" not in captured.out
        )  # Sorted lists don't show return values by default

    def test_bench_param_with_single_item_list(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test using a list with a single BenchParams for parameter testing."""
        # Create a single parameter set
        params = BenchParams(
            name="Standard",
            params={"size": 500},
        )

        # Use a single-item list with the bench decorator
        @bench([params])  # Note: This is a list containing one item
        @bench.config(trials=2)
        def create_sorted_list(size: int) -> list[int]:
            return sorted(range(size))

        captured = capsys.readouterr()

        # Verify output
        assert "Benchmark Results" in captured.out
        assert "create_sorted_list (Standard)" in captured.out
        assert "Avg Time" in captured.out

    def test_bench_param_combined_output(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that multiple BenchParams instances produce a single combined output."""
        # Create multiple parameter sets
        params1 = BenchParams(
            name="Small",
            params={"size": 100},
        )

        params2 = BenchParams(
            name="Medium",
            params={"size": 500},
        )

        params3 = BenchParams(
            name="Large",
            params={"size": 1000},
        )

        # Use the list of params with the bench decorator
        @bench([params1, params2, params3])
        @bench.config(trials=2)
        def create_sorted_list(size: int) -> list[int]:
            return sorted(range(size))

        captured = capsys.readouterr()

        # Verify that only one benchmark results section is displayed
        benchmark_results_count = captured.out.count("Benchmark Results")
        assert (
            benchmark_results_count == 1
        ), "Expected only one benchmark results section"

        # Verify that all parameter sets appear in the combined output
        assert "create_sorted_list (Small)" in captured.out
        assert "create_sorted_list (Medium)" in captured.out
        assert "create_sorted_list (Large)" in captured.out

        # Verify that individual benchmark runs aren't printed separately
        separate_small_output = (
            "Benchmark Results" in captured.out
            and "create_sorted_list (Small)" in captured.out
            and "create_sorted_list (Medium)" not in captured.out
        )
        assert (
            not separate_small_output
        ), "Expected no separate output for Small parameter set"

    def test_bench_decorator_loops_per_trial(
        self,
        capsys: pytest.CaptureFixture,
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test configuring loops per trial with the bench decorator."""
        loops_count = 3  # Use a custom number of loops per trial

        @bench.config(loops_per_trial=loops_count)
        def test_loops() -> int:
            return 10

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out

        # Create another test with explicit parameter to verify
        # that the configuration is applied properly
        test_value = 0

        @bench.config(loops_per_trial=loops_count)
        def count_loops() -> int:
            nonlocal test_value
            test_value += 1
            return test_value

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # With DEFAULT_TRIALS (5) and loops_per_trial=3, we expect
        # test_value to be 5*3 = 15 after the benchmark runs
        # (Each trial runs the function 3 times)
        assert test_value == DEFAULT_TRIALS * loops_count
        assert "count_loops" in parsed_out["functions"]

    def test_bench_grid(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test bench.grid for creating Cartesian products of parameter sets."""
        # Define size parameter sets
        small = BenchParams(name="Small", params={"size": 10})
        large = BenchParams(name="Large", params={"size": 100})

        # Define operation parameter sets
        append = BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)})
        pop = BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()})

        # Use bench.grid to create a Cartesian product of parameters
        @bench.grid([[small, large], [append, pop]])
        @bench.config(trials=1)
        def operation(
            size: int,
            op: Callable[[list[int]], Any],
        ) -> None:
            lst = list(range(size))
            op(lst)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all combinations were created
        assert "operation (Small × Append)" in parsed_out["functions"]
        assert "operation (Small × Pop)" in parsed_out["functions"]
        assert "operation (Large × Append)" in parsed_out["functions"]
        assert "operation (Large × Pop)" in parsed_out["functions"]

        # Ensure we have exactly 4 combinations (2 * 2)
        assert len(parsed_out["functions"]) == 2 * 2

    def test_bench_grid_with_three_parameter_sets(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test bench.grid with three parameter sets for more complex combinations."""
        # Define three different parameter sets to combine
        sizes = [
            BenchParams(name="Small", params={"size": 5}),
            BenchParams(name="Large", params={"size": 20}),
        ]

        operations = [
            BenchParams(name="Add", fn_params={"operation": lambda x: x + 1}),
            BenchParams(name="Multiply", fn_params={"operation": lambda x: x * 2}),
        ]

        formats = [
            BenchParams(name="String", fn_params={"formatter": lambda x: str(x)}),
            BenchParams(name="Hex", fn_params={"formatter": lambda x: hex(x)}),
        ]

        @bench.grid([sizes, operations, formats])
        @bench.config(trials=1, show_output=True)
        def process_number(
            size: int,
            operation: Callable[[int], int],
            formatter: Callable[[int], str],
        ) -> str:
            result = operation(size)
            return formatter(result)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Should create 2*2*2=8 combinations
        assert len(parsed_out["functions"]) == 2 * 2 * 2

        # Check specific combinations
        assert "process_number (Small × Add × String)" in parsed_out["functions"]
        assert "process_number (Small × Add × Hex)" in parsed_out["functions"]
        assert "process_number (Small × Multiply × String)" in parsed_out["functions"]
        assert "process_number (Large × Add × String)" in parsed_out["functions"]

        # Check that return values are correct
        assert "return_values" in parsed_out
        return_values = parsed_out["return_values"]

        # Check specific values
        # 5 + 1 as string
        assert return_values["process_number (Small × Add × String)"] == "6"
        # 5 + 1 as hex
        assert "0x6" in return_values["process_number (Small × Add × Hex)"]
        # 5 * 2 as string
        assert return_values["process_number (Small × Multiply × String)"] == "10"
        # 20 + 1 as string
        assert return_values["process_number (Large × Add × String)"] == "21"


class TestBenchDecoratorRun:
    """Tests for the bench decorator's run method."""

    def test_bench_decorator_run(self, capsys: pytest.CaptureFixture) -> None:
        """Test the bench.run method for grouped functions."""
        # Clean up any existing test groups (in case other tests created them)
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:
            # Test 1: Running a non-existent group should raise an error
            with pytest.raises(ValueError, match="No functions in group"):
                bench.run("nonexistent_group")

            # Test 2: Create an empty group and verify it raises an error
            bench._deferred_functions["empty_group"] = []
            with pytest.raises(
                ValueError,
                match="Group 'empty_group' exists but contains no functions",
            ):
                bench.run("empty_group")

            # Test 3: Basic group functionality with config inheritance
            @bench.config(defer="test_group", trials=3)
            def func1(x: int = 10) -> int:
                return x * 2

            @bench.config(defer="test_group", trials=5, show_output=True)
            def func2(x: int = 20) -> int:
                return x * 3

            # Run the group (should use func2's config as it was added last)
            results = bench.run("test_group")

            # Verify results
            assert "func1" in results
            assert "func2" in results

            # Verify func2's config was used
            captured = capsys.readouterr()
            assert "5 trials" in captured.out
            assert "Return Values" in captured.out

            # Test 4: Custom config overrides function configs
            custom_config = BenchConfig(
                trials=1,
                memory=True,
                show_output=False,  # Override func2's show_output=True
            )

            # Run the group with custom config
            results = bench.run("test_group", config=custom_config)

            # Verify custom config was used
            captured = capsys.readouterr()
            assert "1 trial" in captured.out  # Custom trials=1
            assert (
                "Memory" in captured.out or "Mem" in captured.out
            )  # Custom memory=True
            assert "Return Values" not in captured.out  # Custom show_output=False

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_bench_decorator_run_loops_per_trial(
        self,
    ) -> None:
        """Test that loops_per_trial settings are respected for each function."""
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        execution_counts = {"func1": 0, "func2": 0}

        try:

            count_loop = 2
            count_loop2 = 3

            @bench.config(defer="loops_group", loops_per_trial=count_loop, trials=1)
            def func1() -> int:
                execution_counts["func1"] += 1
                return 10

            @bench.config(defer="loops_group", loops_per_trial=count_loop2, trials=1)
            def func2() -> int:
                execution_counts["func2"] += 1
                return 20

            # Run with a custom config that doesn't override loops_per_trial
            custom_config = BenchConfig(trials=1)
            results = bench.run("loops_group", config=custom_config)

            # Each function should be executed according to
            # its own loops_per_trial setting
            assert execution_counts["func1"] == count_loop
            assert execution_counts["func2"] == count_loop2

            # Verify results
            assert "func1" in results
            assert "func2" in results

        finally:
            bench._deferred_functions = original_deferred

    def test_bench_decorator_run_result_combination(self) -> None:
        """Test that results from multiple functions are correctly combined."""
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench.config(defer="result_group", trials=1, show_output=True)
            def func1() -> str:
                return "result1"

            value = 42

            @bench.config(defer="result_group", trials=1, show_output=True)
            def func2() -> int:
                return value

            results = bench.run("result_group")

            # Verify results are correctly combined
            assert "func1" in results
            assert "func2" in results
            assert results["func1"]["output"][0] == "result1"
            assert results["func2"]["output"][0] == value

            # Verify time and memory measurements are present
            assert "times" in results["func1"]
            assert "times" in results["func2"]

        finally:
            bench._deferred_functions = original_deferred

    def test_bench_decorator_run_with_param_list(self) -> None:
        """Test that bench.run with a custom config applies to parameter lists."""
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        execution_counts: dict[str, int] = {}

        try:
            # Create parameter lists
            params1 = BenchParams(name="Small", params={"size": 10})
            params2 = BenchParams(name="Large", params={"size": 100})
            params = [params1, params2]

            custom_trials = 3  # Set a specific number of trials

            @bench(params)
            @bench.config(defer="param_list_group")
            def test_func(size: int) -> int:
                # Track how many times this gets called
                execution_counts[f"size_{size}"] = (
                    execution_counts.get(f"size_{size}", 0) + 1
                )
                return size

            # Run with custom config that specifies trials
            custom_config = BenchConfig(trials=custom_trials)
            results = bench.run("param_list_group", config=custom_config)

            # Verify results contain both parameter versions
            assert "test_func (Small)" in results
            assert "test_func (Large)" in results

            # Verify each parameter set was executed the expected number of times
            # For each parameter set, we expect trials*loops_per_trial executions
            # Default loops_per_trial is 1, so we expect custom_trials executions
            assert execution_counts["size_10"] == custom_trials
            assert execution_counts["size_100"] == custom_trials
            assert all(len(v["times"]) == custom_trials for v in results.values())

        finally:
            bench._deferred_functions = original_deferred


class TestBenchDecoratorTimeDisabled:
    """Tests for the bench decorator with time measurement disabled."""

    def test_bench_decorator_time_disabled(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that time=False disables time measurement in the output."""

        @bench
        @bench.config(trials=SINGLE_TRIAL, time=False)
        def simple_function() -> int:
            return EXPECTED_VALUE

        simple_function()

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        assert "Avg Time" not in captured.out
        assert "Min Time" not in captured.out
        assert "Max Time" not in captured.out

    def test_bench_decorator_time_disabled_with_memory(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that with time=False and memory=True, only memory is measured."""

        @bench
        @bench.config(trials=SINGLE_TRIAL, time=False, memory=True)
        def allocate_memory() -> list:
            # Allocate some memory
            return [0] * 10000

        allocate_memory()

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        # Memory columns should be present
        assert "Mem" in captured.out or "Memory" in captured.out

    def test_bench_decorator_time_disabled_runtime(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that runtime benchmarking with time=False works properly."""

        @bench
        @bench.config(time=False)
        def runtime_function(size: int) -> list:
            return [0] * size

        # Use .bench() method with time=False
        value = 1000
        runtime_function = cast("BenchmarkableFunction", runtime_function)
        result = runtime_function.bench(value)

        captured = capsys.readouterr()
        # Time columns should not be present
        assert "Time" not in captured.out
        assert "Avg Time" not in captured.out
        # But function should still return correctly
        assert isinstance(result, list)
        assert len(result) == value


class TestBenchDecoratorDefaultArgs:
    """Tests for the bench decorator with default args."""

    def test_bench_with_default_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that benchmark decorator works with functions with default arguments."""
        results = []

        # Function with default arguments
        @bench
        @bench.config(trials=1, show_output=True)
        def func_with_defaults(a: int, b: int = 20, c: str = "default") -> tuple:
            """Benchmark function with default args."""
            result = (a, b, c)
            results.append(result)
            return result

        # Function with some arguments provided and some using defaults
        @bench(a=10)
        @bench.config(trials=1, show_output=True)
        def func_with_partial_args(a: int, b: int = 20, c: str = "default") -> tuple:
            result = (a, b, c)
            results.append(result)
            return result

        # Function with all arguments provided (overriding defaults)
        @bench(a=10, b=30, c="override")
        @bench.config(trials=1, show_output=True)
        def func_with_all_args(a: int, b: int = 20, c: str = "default") -> tuple:
            result = (a, b, c)
            results.append(result)
            return result

        # Using BenchParams to provide arguments
        params = BenchParams(name="Custom", params={"a": 10, "b": 40})

        @bench([params])
        @bench.config(trials=1, show_output=True)
        def func_with_params(a: int, b: int = 20, c: str = "default") -> tuple:
            result = (a, b, c)
            results.append(result)
            return result

        # Verify the output contains expected values
        captured = capsys.readouterr()
        output = captured.out

        # Check that we have the expected results
        assert (
            "(10, 20, 'default')" in output
        )  # func_with_partial_args (b and c use defaults)
        assert (
            "(10, 30, 'override')" in output
        )  # func_with_all_args (all args provided)
        assert (
            "(10, 40, 'default')" in output
        )  # func_with_params (a and b provided, c uses default)

        # First function won't run because 'a' is required and not provided
        assert "func_with_defaults" not in output

        # Check the actual values used in the function calls
        assert (10, 20, "default") in results  # func_with_partial_args
        assert (10, 30, "override") in results  # func_with_all_args
        assert (10, 40, "default") in results  # func_with_params
        assert len([r for r in results if r == (10, 20, "default")]) == 1
        assert len([r for r in results if r == (10, 30, "override")]) == 1
        assert len([r for r in results if r == (10, 40, "default")]) == 1

    def test_bench_with_fn_params_defaults(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that fn_params decorator works with default arguments."""
        results = []

        # Using fn_params to provide required arguments,
        # letting defaults handle the rest
        @bench.fn_params(a=lambda x, y: x + y)
        @bench.config(trials=1, show_output=True)
        def func_with_fn_params(a: Callable, b: int = 123, c: int = 456) -> tuple:
            result = a(b, c)
            results.append(result)
            return result

        # Using fn_params to override one default but not the other
        value = 987

        @bench.fn_params(a=min, b=max)
        @bench.config(trials=1, show_output=True)
        def func_with_fn_params_override(
            a: Callable,
            b: Callable = min,
            c: int = 1000,
        ) -> tuple:
            result = b(a(value, c), 0)
            results.append(result)
            return result

        captured = capsys.readouterr()
        output = captured.out

        # Check output contains expected values
        assert "579" in output
        assert str(value) in output

        # Check the actual values used in function calls
        assert [123 + 456, value] == results


class TestBenchDecoratorDefer:
    """Tests for the defer option in the bench decorator's config method."""

    def test_defer_boolean(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that a function with defer=True is not executed immediately."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench
            @bench.config(defer=True, trials=1)
            def deferred_function() -> int:
                return 42

            # Since defer=True, no output should be generated
            captured = capsys.readouterr()
            assert captured.out == ""

            # Directly execute the function's bench method
            deferred_function = cast("BenchmarkableFunction", deferred_function)
            deferred_function.bench.bench()

            # Now we should see output
            captured = capsys.readouterr()
            assert "Benchmark Results" in captured.out
            assert "deferred_function" in captured.out

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_defer_group(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that a function with string defer is added to the correct group."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench
            @bench.config(defer="test_group", trials=1)
            def grouped_function() -> int:
                return 42

            # Since defer="test_group", no output should be generated
            captured = capsys.readouterr()
            assert captured.out == ""

            # Check that the function was added to the group
            assert "test_group" in bench._deferred_functions
            assert len(bench._deferred_functions["test_group"]) == 1
            assert (
                bench._deferred_functions["test_group"][0].__name__
                == "grouped_function"
            )

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_defer_multiple_functions(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that multiple functions can be added to the same group."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench
            @bench.config(defer="multi_group", trials=1)
            def first_function() -> int:
                return 1

            value = 2

            @bench
            @bench.config(defer="multi_group", trials=1)
            def second_function() -> int:
                return value

            # No output should be generated for either function
            captured = capsys.readouterr()
            assert captured.out == ""

            # Check that both functions were added to the group
            assert "multi_group" in bench._deferred_functions
            assert len(bench._deferred_functions["multi_group"]) == value
            assert (
                bench._deferred_functions["multi_group"][0].__name__ == "first_function"
            )
            assert (
                bench._deferred_functions["multi_group"][1].__name__
                == "second_function"
            )

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_run_group(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the run method runs all functions in a group."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench
            @bench.config(defer="run_group", trials=1)
            def func1() -> int:
                return 10

            @bench
            @bench.config(defer="run_group", trials=1)
            def func2() -> int:
                return 20

            # Run the group
            results = bench.run("run_group")

            # Check output
            captured = capsys.readouterr()
            assert "Benchmark Results" in captured.out
            assert "func1" in captured.out
            assert "func2" in captured.out

            # Check results
            assert "func1" in results
            assert "func2" in results

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_run_with_custom_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the run method uses the provided config."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:

            @bench
            @bench.config(defer="config_group", trials=5)
            def config_func() -> int:
                return 42

            # Run with custom config
            custom_config = BenchConfig(trials=2, show_output=True)
            _ = bench.run("config_group", config=custom_config)

            # Check output
            captured = capsys.readouterr()
            assert (
                "Benchmark Results (2 trials)" in captured.out
            )  # Should use custom trials=2
            assert "config_func" in captured.out
            assert "Return Values" in captured.out  # Should show output
            assert "42" in captured.out  # Should show the return value

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_run_nonexistent_group(self) -> None:
        """Test that run raises an error when the group doesn't exist."""
        with pytest.raises(ValueError, match="No functions in group"):
            bench.run("nonexistent_group")

    def test_run_empty_group(self) -> None:
        """Test that run raises an error when the group is empty."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:
            # Create an empty group
            bench._deferred_functions["empty_group"] = []

            with pytest.raises(
                ValueError,
                match="Group 'empty_group' exists but contains no functions",
            ):
                bench.run("empty_group")

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_defer_with_parametrize(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that defer works with parametrize."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:
            # Create parameter sets
            small = BenchParams(name="Small", params={"size": 10})
            large = BenchParams(name="Large", params={"size": 100})

            @bench([small, large])
            @bench.config(defer="param_group", trials=1)
            def create_list(size: int) -> list:
                return list(range(size))

            # No output should be generated
            captured = capsys.readouterr()
            assert captured.out == ""

            # Run the group
            results = bench.run("param_group")

            # Check output
            captured = capsys.readouterr()
            assert "Benchmark Results" in captured.out
            assert "create_list (Small)" in captured.out
            assert "create_list (Large)" in captured.out

            # Check results
            assert "create_list (Small)" in results
            assert "create_list (Large)" in results

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred

    def test_defer_restore_config_after_run(self) -> None:
        """Test that original config is restored with parameter lists."""
        # Clean up any existing test groups
        original_deferred = bench._deferred_functions.copy()
        bench._deferred_functions.clear()

        try:
            # Create parameter sets
            small = BenchParams(name="Small", params={"size": 10})
            large = BenchParams(name="Large", params={"size": 100})

            original_trials = 7  # Distinct value to check for restoration

            # Set up a function with parameter list and specific config
            params = [small, large]

            @bench(params)
            @bench.config(
                defer="config_restore_group",
                trials=original_trials,
                memory=True,
            )
            def test_func(size: int) -> list:
                return list(range(size))

            # Store original config for later comparison
            original_config = test_func.bench.bench_config.model_copy(deep=True)

            # Run the group with a different config
            custom_config = BenchConfig(trials=2, memory=False)
            _ = bench.run("config_restore_group", config=custom_config)

            # Verify original config was restored
            assert test_func.bench.bench_config.trials == original_trials
            assert test_func.bench.bench_config.memory == original_config.memory

            # Verify the function still works with its original configuration
            assert test_func.params_list is not None
            assert len(test_func.params_list) == len(params)

        finally:
            # Restore the original deferred functions
            bench._deferred_functions = original_deferred


class TestBenchDecoratorRecursion:
    """Tests for the bench decorator with recursive functions."""

    def test_basic_recursive_function(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that a recursive function works with the bench decorator."""

        @bench(n=10)
        def fibonacci(n: int) -> int:
            """Calculate nth Fibonacci number recursively."""
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fibonacci(n - 1) + fibonacci(n - 2)

        # Fibonacci(10) should be 55
        result = fibonacci(10)
        fib_10 = 55
        assert result == fib_10

        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "fibonacci" in captured.out
        assert "Avg Time" in captured.out

    def test_recursive_function_with_params(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that a recursive function works with BenchParams for multiple values."""

        # Use smaller range for testing (n=15+ gets very slow with naive recursion)
        @bench([BenchParams(name=f"n={n}", params={"n": n}) for n in range(5, 10)])
        def fibonacci(n: int) -> int:
            """Calculate nth Fibonacci number recursively."""
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fibonacci(n - 1) + fibonacci(n - 2)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Verify all parameter versions were benchmarked
        assert "fibonacci (n=5)" in parsed_out["functions"]
        assert "fibonacci (n=6)" in parsed_out["functions"]
        assert "fibonacci (n=7)" in parsed_out["functions"]
        assert "fibonacci (n=8)" in parsed_out["functions"]
        assert "fibonacci (n=9)" in parsed_out["functions"]

    def test_recursive_function_with_show_output(
        self,
        capsys: pytest.CaptureFixture[str],
        parse_benchmark_output: Callable[[str], dict[str, Any]],
    ) -> None:
        """Test that recursive function results are correctly captured and displayed."""

        @bench(n=7)
        @bench.config(trials=1, show_output=True)
        def fibonacci(n: int) -> int:
            """Calculate nth Fibonacci number recursively."""
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fibonacci(n - 1) + fibonacci(n - 2)

        captured = capsys.readouterr()
        parsed_out = parse_benchmark_output(captured.out)

        # Check that the function output is correct and displayed
        assert parsed_out["has_return_values"]
        assert parsed_out["return_values"]["fibonacci"] == "13"  # Fibonacci(7) = 13
