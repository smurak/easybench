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

from easybench import bench

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

        @bench
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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
        from easybench.decorator import BenchParams

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
