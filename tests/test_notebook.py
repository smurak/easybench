"""
Tests for the IPython/Jupyter Notebook magic commands in easybench.notebook.

This module tests:
- Loading the IPython extension
- Parsing magic command options
- Running cell benchmarks
- Results formatting
"""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

from IPython.core.interactiveshell import ExecutionResult

from easybench.core import BenchConfig
from easybench.notebook import CellBenchRunner, EasyBenchMagics, load_ipython_extension

# Constants to avoid magic numbers
DEFAULT_TRIALS = 1
CUSTOM_TRIALS = 5
WARMUPS = 2
EXAMPLE_CODE = "x = [i for i in range(100)]"
MOCK_EXECUTION_TIME = 0.1
MOCK_MEMORY_USAGE = 100
MOCK_RESULT = "test result"
TRIALS_COUNT = 2  # Number of trials in tests


class TestLoadIPythonExtension:
    """Tests for the load_ipython_extension function."""

    def test_load_extension(self) -> None:
        """Test that the extension is registered with IPython."""
        mock_ipython = MagicMock()

        load_ipython_extension(mock_ipython)

        mock_ipython.register_magics.assert_called_once()
        # Verify that EasyBenchMagics is registered
        args, _ = mock_ipython.register_magics.call_args
        assert args[0] == EasyBenchMagics


class TestEasyBenchMagics:
    """Tests for the EasyBenchMagics class."""

    def test_init(self) -> None:
        """Test initialization of the EasyBenchMagics class."""
        # Skip actual initialization and just set the shell attribute directly
        magics = EasyBenchMagics()

        assert hasattr(magics, "shell")

    def test_parse_options_default(self) -> None:
        """Test parsing options with default values."""
        magics = EasyBenchMagics()
        magics.shell = MagicMock()  # type: ignore[assignment]

        config = magics._parse_options("")

        assert config.trials == DEFAULT_TRIALS
        assert config.memory is False
        assert config.time is not False

    def test_parse_options_custom(self) -> None:
        """Test parsing custom options."""
        magics = EasyBenchMagics()
        magics.shell = MagicMock()  # type: ignore[assignment]

        config = magics._parse_options(
            f"--trials={CUSTOM_TRIALS} --memory --warmups={WARMUPS}",
        )

        assert config.trials == CUSTOM_TRIALS
        assert config.memory is True
        assert config.warmups == WARMUPS

    @patch("easybench.notebook.CellBenchRunner")
    def test_easybench_magic(
        self,
        mock_runner_class: MagicMock,
    ) -> None:
        """Test the %%easybench cell magic."""
        mock_shell = MagicMock()
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        magics = EasyBenchMagics()
        magics.shell = mock_shell  # type: ignore[assignment]

        # Test with simple code
        result = magics.easybench(f"--trials={CUSTOM_TRIALS}", EXAMPLE_CODE)

        # Check that the runner was created with the shell
        mock_runner_class.assert_called_once_with(mock_shell)

        # Check that run_cell_benchmark was called with the right arguments
        mock_runner.run_cell_benchmark.assert_called_once()
        args, _ = mock_runner.run_cell_benchmark.call_args
        assert args[0] == EXAMPLE_CODE
        assert isinstance(args[1], BenchConfig)
        assert args[1].trials == CUSTOM_TRIALS

        assert result is None

    def test_easybench_no_shell(self) -> None:
        """Test the %%easybench magic when shell is None."""
        magics = EasyBenchMagics()
        magics.shell = None

        result = magics.easybench("", "")

        assert result is None


class TestCellBenchRunner:
    """Tests for the CellBenchRunner class."""

    def test_init(self) -> None:
        """Test initialization of the CellBenchRunner class."""
        mock_shell = MagicMock()
        runner = CellBenchRunner(mock_shell)

        assert runner.shell == mock_shell

    @patch("easybench.notebook.measure_execution")
    @patch("easybench.notebook.EasyBench")
    def test_run_cell_benchmark(
        self,
        mock_easy_bench_class: MagicMock,
        mock_measure_execution: MagicMock,
    ) -> None:
        """Test running a cell benchmark."""
        mock_shell = MagicMock()

        # Make measure_execution actually call the execution_func
        def side_effect_func(
            execution_func: Callable[[], None],
            **_: object,
        ) -> tuple[float, int, ExecutionResult]:
            execution_func()  # This will call shell.ex(EXAMPLE_CODE)
            result = ExecutionResult(None)
            result.result = MOCK_RESULT  # type: ignore[assignment]
            return (MOCK_EXECUTION_TIME, MOCK_MEMORY_USAGE, result)

        mock_measure_execution.side_effect = side_effect_func

        mock_easy_bench = MagicMock()
        mock_easy_bench_class.return_value = mock_easy_bench

        runner = CellBenchRunner(mock_shell)

        config = BenchConfig(trials=TRIALS_COUNT, memory=True, show_output=True)

        result = runner.run_cell_benchmark(EXAMPLE_CODE, config)

        # Check that ex was called on the shell with the cell code
        assert mock_shell.run_cell.call_count == TRIALS_COUNT
        mock_shell.run_cell.assert_called_with(
            EXAMPLE_CODE,
            silent=True,
        )

        # Check that measure_execution was called with the right arguments
        assert (
            mock_measure_execution.call_count == TRIALS_COUNT
        )  # Called once for each trial

        # Check that EasyBench was created with the config
        mock_easy_bench_class.assert_called_once_with(config)

        # Check that process_results and report_results were called
        mock_easy_bench.process_results.assert_called_once()
        mock_easy_bench.report_results.assert_called_once()

        # Result should be the last measured result
        assert result == MOCK_RESULT

    @patch("easybench.notebook.measure_execution")
    @patch("easybench.notebook.EasyBench")
    def test_run_cell_benchmark_with_warmups(
        self,
        mock_easy_bench_class: MagicMock,
        mock_measure_execution: MagicMock,
    ) -> None:
        """Test running a cell benchmark with warmups."""
        mock_shell = MagicMock()
        warmup_result = "warmup result"
        trial_result = "trial result"

        # Make measure_execution call the execution_func and return predefined results
        call_count = 0

        def side_effect_func(
            execution_func: Callable[[], None],
            **_: object,
        ) -> tuple[float, int, ExecutionResult]:
            nonlocal call_count
            execution_func()  # This will call shell.ex(EXAMPLE_CODE)
            call_count += 1
            if call_count == 1:  # First call (warmup)
                result = ExecutionResult(None)
                result.result = warmup_result  # type: ignore[assignment]
                return (0.2, 120, result)
            # Second call (trial)
            result = ExecutionResult(None)
            result.result = trial_result  # type: ignore[assignment]
            return (MOCK_EXECUTION_TIME, MOCK_MEMORY_USAGE, result)

        mock_measure_execution.side_effect = side_effect_func

        mock_easy_bench = MagicMock()
        mock_easy_bench_class.return_value = mock_easy_bench

        runner = CellBenchRunner(mock_shell)

        config = BenchConfig(trials=1, warmups=1, memory=True)

        result = runner.run_cell_benchmark(EXAMPLE_CODE, config)

        # Check that ex was called twice (once for warmup, once for trial)
        warmup_plus_trial = 2
        assert mock_shell.run_cell.call_count == warmup_plus_trial

        # Check that measure_execution was called twice
        assert mock_measure_execution.call_count == warmup_plus_trial

        # Check results (should only include trial results, not warmup)
        args, _ = mock_easy_bench.process_results.call_args
        results = args[0]
        assert "Cell Code" in results
        assert "times" in results["Cell Code"]
        assert len(results["Cell Code"]["times"]) == 1
        assert (
            results["Cell Code"]["times"][0] == MOCK_EXECUTION_TIME
        )  # Only trial time, not warmup

        # Result should be the last execution result (from the trial, not the warmup)
        assert result == trial_result

    @patch("easybench.notebook.measure_execution")
    @patch("easybench.notebook.EasyBench")
    def test_run_cell_benchmark_loops_per_trial(
        self,
        mock_easy_bench_class: MagicMock,
        mock_measure_execution: MagicMock,
    ) -> None:
        """Test running a cell benchmark with multiple loops per trial."""
        mock_shell = MagicMock()

        # Make measure_execution call the execution_func
        def side_effect_func(
            execution_func: Callable[[], None],
            **_: object,
        ) -> tuple[float, int, ExecutionResult]:
            execution_func()  # This will call shell.ex(EXAMPLE_CODE)
            result = ExecutionResult(None)
            result.result = MOCK_RESULT  # type: ignore[assignment]
            return (MOCK_EXECUTION_TIME, MOCK_MEMORY_USAGE, result)

        mock_measure_execution.side_effect = side_effect_func

        mock_easy_bench = MagicMock()
        mock_easy_bench_class.return_value = mock_easy_bench

        runner = CellBenchRunner(mock_shell)

        loops = 5
        config = BenchConfig(trials=1, loops_per_trial=loops)

        runner.run_cell_benchmark(EXAMPLE_CODE, config)

        # Verify that loops_per_trial was passed to measure_execution
        _, kwargs = mock_measure_execution.call_args
        assert kwargs["loops"] == loops
