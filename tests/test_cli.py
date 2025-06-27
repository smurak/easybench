"""
Tests for the command-line interface (CLI) functionality of easybench.

This module tests various aspects of the CLI, including:
- Finding benchmark files
- Loading benchmark modules
- Discovering benchmarks in modules
- Running benchmarks via CLI
- Command-line argument handling
- Error handling
"""

import logging
import tempfile
import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Protocol, cast
from unittest.mock import ANY, MagicMock, patch

import pytest

from easybench.cli import (
    cli_main,
    discover_benchmark_files,
    discover_benchmarks,
    load_benchmark_module,
    run_benchmarks,
)
from easybench.core import EasyBench, PartialBenchConfig

# Constants to avoid magic numbers
NUM_BENCHMARK_FILES = 2
DEFAULT_TEST_VALUE = 42
SMALL_SLEEP_TIME = 0.01
MIN_TEST_TRIALS = 2
DEFAULT_TEST_TRIALS = 10
NUM_FUNCTIONBENCH_CALLS = 2
NUM_EXAMPLE_LIST_SIZE = 10000


# Define a Protocol for mock modules to avoid using Any
class MockModuleProtocol(Protocol):
    """Protocol defining the structure of mock modules used in tests."""

    bench_func1: Callable[[], int]
    bench_func2: Callable[[], int]
    not_a_bench: Callable[[], int]
    BenchClass1: type
    BenchClass2: type
    NotABench: type


# Common fixtures for testing
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing.

    Returns:
        Generator[Path, None, None]: A generator yielding a Path object pointing to
        a temporary directory that will be automatically cleaned up after the test.

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_benchmark_files(temp_dir: Path) -> tuple[Path, list[Path], Path]:
    """
    Create mock benchmark files for testing.

    Args:
        temp_dir: Temporary directory fixture for creating test files

    Returns:
        tuple: A tuple containing (temp_dir, list_of_benchmark_files, normal_file)

    """
    bench_file1 = temp_dir / "bench_test1.py"
    bench_file2 = temp_dir / "bench_test2.py"
    normal_file = temp_dir / "normal_file.py"

    bench_file1.touch()
    bench_file2.touch()
    normal_file.touch()

    return temp_dir, [bench_file1, bench_file2], normal_file


@pytest.fixture
def create_module_file() -> Callable[[Path, str, str], Path]:
    """
    Create Python module files with given content.

    Returns:
        callable: A function that creates a file with the specified content in the
        specified directory and returns the file path.

    """

    def _create_file(directory: Path, filename: str, content: str) -> Path:
        """
        Create a file with the given content.

        Args:
            directory: Directory where the file will be created
            filename: Name of the file to create
            content: Content to write to the file

        Returns:
            Path: Path object for the created file

        """
        file_path = Path(directory) / filename
        with file_path.open("w") as f:
            f.write(content)
        return file_path

    return _create_file


@pytest.fixture
def cli_args_mock() -> MagicMock:
    """Create a mock for CLI arguments."""
    args = MagicMock()
    args.path = "test_dir"
    args.trials = DEFAULT_TEST_TRIALS
    args.loops_per_trial = 1
    args.memory = False
    args.memory_unit = None
    args.sort_by = "avg"
    args.reverse = False
    args.no_color = False
    args.show_output = False
    args.time_unit = "s"
    args.warmups = 0
    args.no_progress = True
    args.include = None
    args.exclude = None
    args.include_files = None
    args.exclude_files = None
    args.clip_outliers = None
    return args


class TestDiscoverBenchmarkFiles:
    """Tests for discovering benchmark files in directories."""

    def test_discover_benchmark_files(
        self,
        mock_benchmark_files: tuple[Path, list[Path], Path],
    ) -> None:
        """Test that benchmark files are correctly discovered."""
        temp_dir, expected_files, normal_file = mock_benchmark_files

        found_files = discover_benchmark_files(temp_dir)

        assert len(found_files) == NUM_BENCHMARK_FILES
        assert expected_files[0] in found_files
        assert expected_files[1] in found_files
        assert normal_file not in found_files

    def test_discover_benchmark_files_empty_dir(self, temp_dir: Path) -> None:
        """Test discovery with an empty directory."""
        found_files = discover_benchmark_files(temp_dir)
        assert len(found_files) == 0

    def test_discover_benchmark_files_nonexistent_dir(self) -> None:
        """Test discovery with a non-existent directory."""
        found_files = discover_benchmark_files("/path/that/does/not/exist")
        assert len(found_files) == 0

    def test_discover_benchmark_files_directory(self, temp_dir: Path) -> None:
        """Test discovering benchmark files from a directory."""
        # Create a subdirectory for the test
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir()

        # Create mock benchmark files in the subdirectory
        bench_file1 = sub_dir / "bench_test1.py"
        bench_file2 = sub_dir / "bench_test2.py"
        bench_file1.touch()
        bench_file2.touch()

        # Discover benchmark files in the subdirectory
        found_files = discover_benchmark_files(str(sub_dir))

        num_files = 2
        assert len(found_files) == num_files
        assert bench_file1 in found_files
        assert bench_file2 in found_files

    def test_discover_benchmark_files_single_file(self, temp_dir: Path) -> None:
        """Test discovering benchmark files with a single file path."""
        # Create a valid benchmark file
        valid_bench_file = temp_dir / "bench_test.py"
        valid_bench_file.touch()

        # Discover benchmark files with a single file path
        found_files = discover_benchmark_files(str(valid_bench_file))

        assert len(found_files) == 1
        assert found_files[0] == valid_bench_file

        # Create a valid benchmark file (not start with bench_)
        bench_file = temp_dir / "test.py"
        bench_file.touch()

        # Discover benchmark files (not start with bench_)
        found_files = discover_benchmark_files(str(bench_file))

        assert len(found_files) == 1

        # Create a benchmark file (non-py extension)
        non_py_ext_file = temp_dir / "bench_test.txt"
        non_py_ext_file.touch()

        # Discover benchmark files with an non-py extension
        found_files = discover_benchmark_files(str(non_py_ext_file))

        assert len(found_files) == 1

    def test_discover_benchmark_files_nonexistent(self) -> None:
        """Test discovering benchmark files with a non-existent path."""
        found_files = discover_benchmark_files("nonexistent_path")
        assert len(found_files) == 0

    def test_discover_benchmark_files_with_include_files(
        self,
        temp_dir: Path,
    ) -> None:
        """Test discovering benchmark files with include_files filter."""
        # Create test benchmark files
        bench_file1 = temp_dir / "bench_test1.py"
        bench_file2 = temp_dir / "bench_test2.py"
        bench_file3 = temp_dir / "bench_feature_test.py"
        normal_file = temp_dir / "normal_file.py"

        bench_file1.touch()
        bench_file2.touch()
        bench_file3.touch()
        normal_file.touch()

        # Test include_files with pattern matching only bench_test files
        found_files = discover_benchmark_files(
            temp_dir,
            include_files=r"bench_test\d+\.py",
        )

        num_test_files = 2
        assert len(found_files) == num_test_files
        assert bench_file1 in found_files
        assert bench_file2 in found_files
        assert bench_file3 not in found_files
        assert normal_file not in found_files

        # Test include_files with pattern matching feature files
        found_files = discover_benchmark_files(temp_dir, include_files=r"feature")

        assert len(found_files) == 1
        assert bench_file3 in found_files
        assert bench_file1 not in found_files
        assert bench_file2 not in found_files

    def test_discover_benchmark_files_with_exclude_files(
        self,
        temp_dir: Path,
    ) -> None:
        """Test discovering benchmark files with exclude_files filter."""
        # Create test benchmark files
        bench_file1 = temp_dir / "bench_test1.py"
        bench_file2 = temp_dir / "bench_test2.py"
        bench_file3 = temp_dir / "bench_feature_test.py"

        bench_file1.touch()
        bench_file2.touch()
        bench_file3.touch()

        # Test exclude_files with pattern excluding bench_test2 file
        found_files = discover_benchmark_files(temp_dir, exclude_files=r"test2\.py")

        num_not_test2 = 2
        assert len(found_files) == num_not_test2
        assert bench_file1 in found_files
        assert bench_file2 not in found_files
        assert bench_file3 in found_files

        # Test exclude_files with pattern excluding feature files
        found_files = discover_benchmark_files(temp_dir, exclude_files=r"feature")

        num_not_feature = 2
        assert len(found_files) == num_not_feature
        assert bench_file1 in found_files
        assert bench_file2 in found_files
        assert bench_file3 not in found_files

    def test_discover_benchmark_files_with_include_and_exclude_files(
        self,
        temp_dir: Path,
    ) -> None:
        """Test discovering benchmark files with include_files and exclude_files."""
        # Create test benchmark files
        bench_file1 = temp_dir / "bench_test1.py"
        bench_file2 = temp_dir / "bench_test2.py"
        bench_file3 = temp_dir / "bench_feature_test1.py"
        bench_file4 = temp_dir / "bench_feature_test2.py"

        bench_file1.touch()
        bench_file2.touch()
        bench_file3.touch()
        bench_file4.touch()

        # Test combining include_files and exclude_files
        # Include only files with test\d pattern but exclude test2
        found_files = discover_benchmark_files(
            temp_dir,
            include_files=r"test\d",
            exclude_files=r"test2",
        )

        num_test1_files = 2
        assert len(found_files) == num_test1_files
        assert bench_file1 in found_files
        assert bench_file2 not in found_files
        assert bench_file3 in found_files
        assert bench_file4 not in found_files

        # Another test with different patterns
        # Include only feature files but exclude feature_test2
        found_files = discover_benchmark_files(
            temp_dir,
            include_files=r"feature",
            exclude_files=r"test2",
        )

        assert len(found_files) == 1
        assert bench_file1 not in found_files
        assert bench_file2 not in found_files
        assert bench_file3 in found_files
        assert bench_file4 not in found_files


class TestLoadBenchmarkModule:
    """Tests for loading benchmark modules from files."""

    def test_load_benchmark_module(
        self,
        temp_dir: Path,
        create_module_file: Callable,
    ) -> None:
        """Test loading a benchmark module from a file."""
        module_content = """
def bench_test():
    return 42
"""
        file_path = create_module_file(temp_dir, "bench_simple.py", module_content)

        module = load_benchmark_module(file_path)

        assert module is not None
        assert hasattr(module, "bench_test")
        assert module.bench_test() == DEFAULT_TEST_VALUE

    def test_load_benchmark_module_error(
        self,
        temp_dir: Path,
        create_module_file: Callable,
    ) -> None:
        """Test handling errors when loading a module."""
        module_content = """
def bench_test():
    return 42
# Missing closing parenthesis
print("Hello"
"""
        file_path = create_module_file(temp_dir, "bench_error.py", module_content)

        module = load_benchmark_module(file_path)

        assert module is None

    def test_load_benchmark_module_import_error(
        self,
        temp_dir: Path,
        create_module_file: Callable,
    ) -> None:
        """Test handling import errors when loading a module."""
        module_content = """
import non_existent_module
def bench_test():
    return non_existent_module.func()
"""
        file_path = create_module_file(
            temp_dir,
            "bench_import_error.py",
            module_content,
        )

        module = load_benchmark_module(file_path)

        assert module is None


class TestDiscoverBenchmarks:
    """Tests for discovering benchmark functions and classes in modules."""

    @pytest.fixture
    def mock_module(self) -> types.ModuleType:
        """
        Create a mock module with benchmark functions and classes.

        Returns:
            types.ModuleType: A mock module containing benchmark functions and classes

        """
        # Use MagicMock instead of types.ModuleType for better attribute handling
        mock_mod = MagicMock(name="mock_module")

        # Add benchmark functions
        def bench_func1() -> int:
            return 1

        def bench_func2() -> int:
            return 2

        def not_a_bench() -> int:
            return 3

        mock_mod.bench_func1 = bench_func1
        mock_mod.bench_func2 = bench_func2
        mock_mod.not_a_bench = not_a_bench

        # Add benchmark classes
        class BenchClass1(EasyBench):
            def bench_test(self) -> int:
                return 10

        class BenchClass2(EasyBench):
            def bench_test(self) -> int:
                return 20

        class NotABench:
            pass

        mock_mod.BenchClass1 = BenchClass1
        mock_mod.BenchClass2 = BenchClass2
        mock_mod.NotABench = NotABench

        return cast("types.ModuleType", mock_mod)

    def test_discover_benchmarks(self, mock_module: types.ModuleType) -> None:
        """Test discovering benchmark functions and classes in a module."""
        benchmarks = discover_benchmarks(mock_module)

        # Verify functions were discovered
        assert "bench_func1" in benchmarks
        assert "bench_func2" in benchmarks
        assert "not_a_bench" not in benchmarks

        # Verify classes were discovered and instantiated
        assert "BenchClass1" in benchmarks
        assert isinstance(benchmarks["BenchClass1"], mock_module.BenchClass1)
        assert "BenchClass2" in benchmarks
        assert isinstance(benchmarks["BenchClass2"], mock_module.BenchClass2)
        assert "NotABench" not in benchmarks

    def test_discover_benchmarks_error_handling(self) -> None:
        """Test handling errors when initializing benchmark classes."""
        # Use MagicMock instead of types.ModuleType
        mock_mod = MagicMock(name="test_module")

        # Add a benchmark class that raises an error when instantiated
        class BrokenBench(EasyBench):
            def __init__(self) -> None:
                error_msg = "This class is broken"
                raise ValueError(error_msg)

        mock_mod.BrokenBench = BrokenBench

        # Discover benchmarks (should handle the error)
        benchmarks = discover_benchmarks(cast("types.ModuleType", mock_mod))

        # Verify the broken class wasn't added to the benchmarks
        assert "BrokenBench" not in benchmarks

    def test_discover_no_benchmarks(self) -> None:
        """Test discovering benchmarks in a module with none."""
        # Use MagicMock instead of types.ModuleType
        mock_mod = MagicMock(name="empty_module")

        def regular_function() -> bool:
            return True

        mock_mod.regular_function = regular_function

        benchmarks = discover_benchmarks(cast("types.ModuleType", mock_mod))

        assert len(benchmarks) == 0


class TestRunBenchmarks:
    """Tests for running benchmarks through the CLI."""

    @patch("easybench.cli.FunctionBench")
    def test_run_benchmarks_functions(self, mock_function_bench: MagicMock) -> None:
        """Test running function-based benchmarks."""

        # Create mock functions
        def bench_func1() -> int:
            return 1

        def bench_func2() -> int:
            return 2

        benchmarks: dict[str, Any] = {
            "bench_func1": bench_func1,
            "bench_func2": bench_func2,
        }

        # Configure the mock
        mock_instance = MagicMock()
        mock_function_bench.return_value = mock_instance

        # Run benchmarks with PartialBenchConfig
        partial_config = PartialBenchConfig(trials=10, memory=True)
        run_benchmarks(benchmarks, config=partial_config)

        # Verify FunctionBench was created for each function
        assert mock_function_bench.call_count == NUM_FUNCTIONBENCH_CALLS
        mock_function_bench.assert_any_call(bench_func1, func_name="bench_func1")
        mock_function_bench.assert_any_call(bench_func2, func_name="bench_func2")

        # Verify bench() was called on each instance with the right parameters
        assert mock_instance.bench.call_count == NUM_FUNCTIONBENCH_CALLS
        mock_instance.bench.assert_called_with(config=partial_config)

    def test_run_benchmarks_classes(self) -> None:
        """Test running class-based benchmarks."""

        # Create mock EasyBench instances
        class MockBench(EasyBench):
            def bench_test(self) -> int:
                return 10

        # Create instances with mocked bench method
        mock_instance1 = MagicMock(spec=MockBench)
        mock_instance2 = MagicMock(spec=MockBench)

        benchmarks: dict[str, Any] = {
            "MockBench1": mock_instance1,
            "MockBench2": mock_instance2,
        }

        # Run benchmarks with PartialBenchConfig
        partial_config = PartialBenchConfig(
            trials=10,
            memory=True,
            sort_by="avg",
            reverse=True,
        )
        run_benchmarks(benchmarks, config=partial_config)

        # Verify bench() was called on each instance with the right parameters
        assert mock_instance1.bench.call_count == 1
        mock_instance1.bench.assert_called_with(config=partial_config)

        assert mock_instance2.bench.call_count == 1
        mock_instance2.bench.assert_called_with(config=partial_config)

    @patch("easybench.cli.FunctionBench")
    def test_run_benchmarks_with_exception(
        self,
        mock_function_bench: MagicMock,
    ) -> None:
        """Test running benchmarks that raise exceptions."""

        def bench_func() -> None:
            error_msg = "Benchmark error"
            raise ValueError(error_msg)

        benchmarks: dict[str, Any] = {"bench_func": bench_func}

        # Make the FunctionBench.bench method raise an exception
        mock_instance = MagicMock()
        error_msg = "Benchmark execution error"
        mock_instance.bench.side_effect = ValueError(error_msg)
        mock_function_bench.return_value = mock_instance

        # Run benchmarks (should not crash)
        with patch("easybench.cli.logger.exception") as mock_log:
            run_benchmarks(benchmarks)

        # Verify the error was logged
        mock_log.assert_any_call("Error running benchmark %s", "bench_func")


class TestCliMain:
    """Tests for the main CLI entry point."""

    @pytest.fixture
    def cli_mocks(self) -> Generator[dict[str, MagicMock], None, None]:
        """
        Set up mock objects for CLI main tests.

        Returns:
            Dictionary of mock objects for CLI testing

        """
        with (
            patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
            patch("easybench.cli.discover_benchmark_files") as mock_discover_files,
            patch("easybench.cli.load_benchmark_module") as mock_load_module,
            patch("easybench.cli.discover_benchmarks") as mock_discover_benchmarks,
            patch("easybench.cli.run_benchmarks") as mock_run_benchmarks,
        ):
            yield {
                "parse_args": mock_parse_args,
                "discover_files": mock_discover_files,
                "load_module": mock_load_module,
                "discover_benchmarks": mock_discover_benchmarks,
                "run_benchmarks": mock_run_benchmarks,
            }

    def test_cli_main(
        self,
        cli_mocks: dict[str, MagicMock],
        cli_args_mock: MagicMock,
    ) -> None:
        """Test the main CLI entry point."""
        # Setup mocks
        cli_mocks["parse_args"].return_value = cli_args_mock

        # Mock finding benchmark files
        mock_file1 = Path("test_dir/bench_test1.py")
        mock_file2 = Path("test_dir/bench_test2.py")
        cli_mocks["discover_files"].return_value = [mock_file1, mock_file2]

        # Mock loading modules and discovering benchmarks
        mock_module1 = MagicMock()
        mock_module2 = MagicMock()
        cli_mocks["load_module"].side_effect = [mock_module1, mock_module2]
        mock_benchmarks1 = {"bench1": MagicMock()}
        mock_benchmarks2 = {"bench2": MagicMock()}
        cli_mocks["discover_benchmarks"].side_effect = [
            mock_benchmarks1,
            mock_benchmarks2,
        ]

        # Run the CLI
        cli_main()

        # Verify the correct functions were called
        cli_mocks["discover_files"].assert_called_once_with(
            "test_dir",
            include_files=None,
            exclude_files=None,
        )
        cli_mocks["load_module"].assert_any_call(mock_file1)
        cli_mocks["load_module"].assert_any_call(mock_file2)
        assert cli_mocks["load_module"].call_count == NUM_FUNCTIONBENCH_CALLS
        cli_mocks["discover_benchmarks"].assert_any_call(mock_module1, config=ANY)
        cli_mocks["discover_benchmarks"].assert_any_call(mock_module2, config=ANY)
        assert cli_mocks["discover_benchmarks"].call_count == NUM_FUNCTIONBENCH_CALLS

        # Verify run_benchmarks was called with the right parameters
        assert cli_mocks["run_benchmarks"].call_count == NUM_FUNCTIONBENCH_CALLS

        # Get the arguments from each call to run_benchmarks
        calls = cli_mocks["run_benchmarks"].call_args_list

        # First call verification
        args1, kwargs1 = calls[0]
        assert args1[0] == mock_benchmarks1
        assert kwargs1["source_id"] == str(mock_file1)
        assert kwargs1["config"] is not None

        # Second call verification
        args2, kwargs2 = calls[1]
        assert args2[0] == mock_benchmarks2
        assert kwargs2["source_id"] == str(mock_file2)
        assert kwargs2["config"] is not None

        # Verify the config fields individually
        for kwargs in [kwargs1, kwargs2]:
            config = kwargs["config"]
            assert isinstance(config, PartialBenchConfig)
            assert config.trials == DEFAULT_TEST_TRIALS
            assert config.sort_by == "avg"
            # These are converted to None in the CLI
            assert config.memory is None
            assert config.reverse is None
            assert config.show_output is None
            # This depends on cli_args_mock.no_color which is False
            assert config.color is None

    @patch("argparse.ArgumentParser.parse_args")
    @patch("easybench.cli.discover_benchmark_files")
    def test_cli_main_no_files(
        self,
        mock_discover_files: MagicMock,
        mock_parse_args: MagicMock,
        cli_args_mock: MagicMock,
    ) -> None:
        """Test CLI behavior when no benchmark files are found."""
        # Setup mocks
        cli_args_mock.path = "empty_dir"
        mock_parse_args.return_value = cli_args_mock

        # Mock finding no benchmark files
        mock_discover_files.return_value = []

        # Run the CLI
        cli_main()

        # Verify the correct functions were called
        mock_discover_files.assert_called_once_with(
            "empty_dir",
            include_files=None,
            exclude_files=None,
        )

    @patch("argparse.ArgumentParser.parse_args")
    @patch("easybench.cli.discover_benchmark_files")
    @patch("easybench.cli.load_benchmark_module")
    @patch("easybench.cli.discover_benchmarks")
    def test_cli_main_no_benchmarks(
        self,
        mock_discover_benchmarks: MagicMock,
        mock_load_module: MagicMock,
        mock_discover_files: MagicMock,
        mock_parse_args: MagicMock,
        cli_args_mock: MagicMock,
    ) -> None:
        """Test CLI behavior when no benchmarks are found in a module."""
        # Setup mocks
        mock_parse_args.return_value = cli_args_mock

        # Mock finding a benchmark file but no benchmarks in it
        mock_file = Path("test_dir/bench_test.py")
        mock_discover_files.return_value = [mock_file]

        mock_module = MagicMock()
        mock_load_module.return_value = mock_module

        mock_discover_benchmarks.return_value = {}

        # Run the CLI
        cli_main()

        # Verify the correct functions were called
        mock_discover_files.assert_called_once_with(
            "test_dir",
            include_files=None,
            exclude_files=None,
        )
        mock_load_module.assert_called_once_with(mock_file)
        mock_discover_benchmarks.assert_called_once_with(mock_module, config=ANY)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("easybench.cli.discover_benchmark_files")
    @patch("easybench.cli.load_benchmark_module")
    def test_cli_main_module_load_error(
        self,
        mock_load_module: MagicMock,
        mock_discover_files: MagicMock,
        mock_parse_args: MagicMock,
        cli_args_mock: MagicMock,
    ) -> None:
        """Test CLI behavior when a module fails to load."""
        # Setup mocks
        mock_parse_args.return_value = cli_args_mock

        # Mock finding a benchmark file but failing to load it
        mock_file = Path("test_dir/bench_test.py")
        mock_discover_files.return_value = [mock_file]

        mock_load_module.return_value = None

        # Run the CLI
        cli_main()

        # Verify the correct functions were called
        mock_discover_files.assert_called_once_with(
            "test_dir",
            include_files=None,
            exclude_files=None,
        )
        mock_load_module.assert_called_once_with(mock_file)

    @patch("argparse.ArgumentParser.parse_args")
    @patch("easybench.cli.discover_benchmark_files")
    def test_cli_main_unexpected_exception(
        self,
        mock_discover_files: MagicMock,
        mock_parse_args: MagicMock,
        cli_args_mock: MagicMock,
    ) -> None:
        """Test handling of unexpected exceptions in CLI."""
        # Setup mocks
        mock_parse_args.return_value = cli_args_mock

        # Make discover_benchmark_files raise an exception
        error_msg = "Unexpected error"
        mock_discover_files.side_effect = RuntimeError(error_msg)

        # Verify the CLI catches the exception and logs an error
        with patch("easybench.cli.logger.exception") as mock_log:
            cli_main()

        # Check that an error message was logged
        mock_log.assert_any_call("Error in easybench CLI")


class TestCliArguments:
    """Tests for CLI argument parsing and handling."""

    @pytest.fixture
    def cli_setup(self) -> Generator[dict[str, MagicMock], None, None]:
        """
        Set up for CLI argument tests.

        Returns:
            Generator[dict[str, MagicMock], None, None]:
                Dictionary of mock objects for CLI testing

        """
        with (
            patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
            patch("easybench.cli.discover_benchmark_files") as mock_discover_files,
            patch("easybench.cli.load_benchmark_module") as mock_load_module,
            patch("easybench.cli.discover_benchmarks") as mock_discover_benchmarks,
            patch("easybench.cli.run_benchmarks") as mock_run_benchmarks,
        ):

            # Create a fixture that provides all the mocks
            yield {
                "parse_args": mock_parse_args,
                "discover_files": mock_discover_files,
                "load_module": mock_load_module,
                "discover_benchmarks": mock_discover_benchmarks,
                "run_benchmarks": mock_run_benchmarks,
            }

    def test_default_arguments(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test using default arguments when none are provided."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.path = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Set up other mocks
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with expected PartialBenchConfig
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert args[0] == mock_benchmarks
        assert kwargs["config"].trials is None
        assert kwargs["config"].memory is None
        assert kwargs["config"].sort_by is None
        assert kwargs["config"].reverse is None
        assert kwargs["config"].color is None
        assert kwargs["config"].show_output is None

    def test_memory_flag(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test the --memory flag."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = True  # Enable memory flag
        mock_args.memory_unit = None
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with memory=True
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].memory

    def test_memory_unit_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test the --memory-unit parameter."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = True
        mock_args.memory_unit = "MB"  # Set memory unit to MB
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with memory=MB
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].memory == "MB"

    def test_memory_unit_parameter2(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test the --memory-unit parameter."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = False  # Set memory to False
        mock_args.memory_unit = "B"  # Set memory unit to B
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with memory=MB
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].memory == "B"

    def test_sort_by_options(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test different sort_by options."""
        sort_options = ["avg", "min", "max", "avg_memory", "max_memory", "def"]

        for sort_option in sort_options:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.directory = "benchmarks"
            mock_args.trials = None
            mock_args.loops_per_trial = None
            mock_args.memory = False
            mock_args.sort_by = sort_option
            mock_args.reverse = False
            mock_args.no_color = False
            mock_args.show_output = False
            mock_args.time_unit = "s"
            mock_args.warmups = None
            mock_args.no_progress = None
            mock_args.memory_unit = None
            mock_args.include = None
            mock_args.exclude = None
            mock_args.include_files = None
            mock_args.exclude_files = None
            mock_args.clip_outliers = None
            cli_setup["parse_args"].return_value = mock_args

            # Mock finding benchmark files
            mock_file = Path("benchmarks/bench_test.py")
            cli_setup["discover_files"].return_value = [mock_file]

            # Mock loading module and discovering benchmarks
            mock_module = MagicMock()
            cli_setup["load_module"].return_value = mock_module
            mock_benchmarks = {"bench1": MagicMock()}
            cli_setup["discover_benchmarks"].return_value = mock_benchmarks

            # Reset mocks
            cli_setup["run_benchmarks"].reset_mock()

            # Run the CLI
            cli_main()

            # Verify run_benchmarks was called with the correct sort_by
            cli_setup["run_benchmarks"].assert_called_once()
            args, kwargs = cli_setup["run_benchmarks"].call_args
            assert kwargs["config"].sort_by == sort_option

    def test_trials_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test specifying different trial counts."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = DEFAULT_TEST_VALUE  # Custom trial count
        mock_args.loops_per_trial = 1
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with the correct trials
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].trials == DEFAULT_TEST_VALUE

    def test_loops_per_trial_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test specifying loops per trial."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = 5  # Custom loops per trial count
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with the correct loops_per_trial
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].loops_per_trial == mock_args.loops_per_trial

    def test_warmups_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test specifying warmup trials count."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = DEFAULT_TEST_VALUE  # Custom warmups count
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with the correct warmups
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].warmups == DEFAULT_TEST_VALUE

    def test_no_progress_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test specifying no_progress flag."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = True  # Enable no_progress flag
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = None
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with progress=False
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].progress is False

    def test_include_exclude_options(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test the --include and --exclude options."""
        # Test cases to verify
        test_cases: list[dict[str, Any]] = [
            {"include": "fast_*", "exclude": None},
            {"include": None, "exclude": "slow_*"},
            {"include": "bench_*", "exclude": "*_slow"},
        ]

        for case in test_cases:
            # Setup mocks
            mock_args = MagicMock()
            mock_args.path = "benchmarks"
            mock_args.trials = None
            mock_args.loops_per_trial = None
            mock_args.memory = False
            mock_args.sort_by = None
            mock_args.reverse = False
            mock_args.no_color = False
            mock_args.show_output = False
            mock_args.time_unit = "s"
            mock_args.warmups = None
            mock_args.no_progress = None
            mock_args.memory_unit = None
            mock_args.include = case["include"]
            mock_args.exclude = case["exclude"]
            mock_args.include_files = None
            mock_args.exclude_files = None
            mock_args.clip_outliers = None
            cli_setup["parse_args"].return_value = mock_args

            # Mock finding benchmark files
            mock_file = Path("benchmarks/bench_test.py")
            cli_setup["discover_files"].return_value = [mock_file]

            # Mock loading module and discovering benchmarks
            mock_module = MagicMock()
            cli_setup["load_module"].return_value = mock_module
            mock_benchmarks = {"bench1": MagicMock()}
            cli_setup["discover_benchmarks"].return_value = mock_benchmarks

            # Reset mocks
            cli_setup["run_benchmarks"].reset_mock()

            # Run the CLI
            cli_main()

            # Verify run_benchmarks was called with the correct include/exclude patterns
            cli_setup["run_benchmarks"].assert_called_once()
            args, kwargs = cli_setup["run_benchmarks"].call_args
            assert kwargs["config"].include == case["include"]
            assert kwargs["config"].exclude == case["exclude"]

    def test_include_exclude_filtering(
        self,
        cli_setup: dict[str, MagicMock],
        cli_args_mock: MagicMock,
    ) -> None:
        """Test that include/exclude patterns actually filter the benchmarks."""
        # Setup mocks
        cli_args_mock.include = "bench_*"
        cli_args_mock.exclude = "*_slow"
        cli_setup["parse_args"].return_value = cli_args_mock

        # Mock finding benchmark file
        mock_file = Path("test_dir/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and creating benchmarks dictionary
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module

        # Create mock benchmarks that should be filtered
        mock_benchmarks = {
            "bench_fast": MagicMock(),  # Should be included
            "bench_slow": MagicMock(),  # Should be excluded (*_slow)
            "test_bench": MagicMock(),  # Should be excluded (not bench_*)
            "bench_test_slow": MagicMock(),  # Should be excluded (*_slow)
        }
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Define a mock implementation for run_benchmarks that verifies filtering
        def mock_run_with_filter(*args: dict, **kwargs: object) -> None:
            # Verify only the expected benchmark is passed
            _ = kwargs
            actual_benchmarks = args[0]
            assert list(actual_benchmarks.keys()) == ["bench_fast"]

        cli_setup["run_benchmarks"].side_effect = mock_run_with_filter

        # Run CLI
        cli_main()

        # Verify run_benchmarks was called
        cli_setup["run_benchmarks"].assert_called_once()

    def test_clip_outliers_parameter(self, cli_setup: dict[str, MagicMock]) -> None:
        """Test specifying clip_outliers parameter."""
        # Setup mocks
        clip_value = 0.1
        mock_args = MagicMock()
        mock_args.directory = "benchmarks"
        mock_args.trials = None
        mock_args.loops_per_trial = None
        mock_args.memory = False
        mock_args.sort_by = None
        mock_args.reverse = False
        mock_args.no_color = False
        mock_args.show_output = False
        mock_args.time_unit = "s"
        mock_args.warmups = None
        mock_args.no_progress = None
        mock_args.memory_unit = None
        mock_args.include = None
        mock_args.exclude = None
        mock_args.include_files = None
        mock_args.exclude_files = None
        mock_args.clip_outliers = clip_value
        cli_setup["parse_args"].return_value = mock_args

        # Mock finding benchmark files
        mock_file = Path("benchmarks/bench_test.py")
        cli_setup["discover_files"].return_value = [mock_file]

        # Mock loading module and discovering benchmarks
        mock_module = MagicMock()
        cli_setup["load_module"].return_value = mock_module
        mock_benchmarks = {"bench1": MagicMock()}
        cli_setup["discover_benchmarks"].return_value = mock_benchmarks

        # Run the CLI
        cli_main()

        # Verify run_benchmarks was called with the correct clip_outliers value
        cli_setup["run_benchmarks"].assert_called_once()
        args, kwargs = cli_setup["run_benchmarks"].call_args
        assert kwargs["config"].clip_outliers == clip_value


class TestCliEdgeCases:
    """Tests for edge cases and error handling in CLI."""

    @pytest.fixture
    def cli_mocks(self) -> Generator[dict[str, MagicMock], None, None]:
        """
        Set up mock objects for CLI edge case tests.

        Returns:
            Dictionary of mock objects for CLI testing

        """
        with (
            patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
            patch("easybench.cli.discover_benchmark_files") as mock_discover_files,
            patch("easybench.cli.load_benchmark_module") as mock_load_module,
            patch("easybench.cli.discover_benchmarks") as mock_discover_benchmarks,
            patch("easybench.cli.run_benchmarks") as mock_run_benchmarks,
        ):
            yield {
                "parse_args": mock_parse_args,
                "discover_files": mock_discover_files,
                "load_module": mock_load_module,
                "discover_benchmarks": mock_discover_benchmarks,
                "run_benchmarks": mock_run_benchmarks,
            }

    def test_benchmarks_with_exceptions(
        self,
        cli_mocks: dict[str, MagicMock],
        cli_args_mock: MagicMock,
    ) -> None:
        """Test handling benchmarks that raise exceptions during execution."""
        # Setup mocks
        cli_mocks["parse_args"].return_value = cli_args_mock

        # Mock finding benchmark file
        mock_file = Path("test_dir/bench_test.py")
        cli_mocks["discover_files"].return_value = [mock_file]

        # Mock loading module
        mock_module = MagicMock()
        cli_mocks["load_module"].return_value = mock_module

        # Create a benchmark function that raises an exception
        def failing_benchmark() -> None:
            error_msg = "Benchmark failure"
            raise ValueError(error_msg)

        mock_benchmarks = {"failing_bench": failing_benchmark}
        cli_mocks["discover_benchmarks"].return_value = mock_benchmarks

        # Make run_benchmarks throw an exception
        error_msg = "Benchmark execution error"
        cli_mocks["run_benchmarks"].side_effect = ValueError(error_msg)

        # Patch print to capture error messages
        with patch("builtins.print"):
            # Run the CLI (should handle the exception gracefully)
            cli_main()  # This should not raise an exception outside the function

        # Verify functions were called correctly
        cli_mocks["discover_files"].assert_called_once()
        cli_mocks["load_module"].assert_called_once()
        cli_mocks["discover_benchmarks"].assert_called_once()
        cli_mocks["run_benchmarks"].assert_called_once()


class TestCliIntegration:
    """Integration tests for CLI functionality."""

    def test_real_benchmark_execution(
        self,
        temp_dir: Path,
        create_module_file: Callable,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test executing a real benchmark through the CLI."""
        # Create a simple benchmark file
        bench_content = """
# Simple benchmark file for testing
import time

def bench_test():
    \"\"\"A benchmark test function\"\"\"
    return 42

def bench_fast():
    \"\"\"A fast benchmark function\"\"\"
    return "fast result"

def bench_slow():
    \"\"\"A slow benchmark function\"\"\"
    time.sleep(0.01)  # Small sleep to ensure it's slower
    return "slow result"
"""
        bench_file = create_module_file(temp_dir, "bench_test_simple.py", bench_content)

        # Setup logging to capture logs in output
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # Simple format to match tests
        )

        # Run the CLI with the benchmark file
        abs_path = str(bench_file.parent.absolute())
        with patch(
            "sys.argv",
            [
                "easybench",
                "--trials",
                str(MIN_TEST_TRIALS),
                "--show-output",
                "--no-color",
                abs_path,
            ],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err  # Combine stdout and stderr

        # Check for core elements that should be present
        assert "Benchmark Results" in output

        # Check if any benchmark runs were logged
        benchmark_functions = ["bench_test", "bench_fast", "bench_slow"]
        found_benchmark = False
        for func_name in benchmark_functions:
            if func_name in output:
                found_benchmark = True
                break

        assert found_benchmark, "No benchmark function was found in output"

    def test_benchmark_with_fixture(
        self,
        temp_dir: Path,
        create_module_file: Callable,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test executing a benchmark that uses fixtures."""
        # Create a benchmark file with fixtures
        fixture_content = """
# Benchmark file with fixtures
from easybench import fixture

@fixture(scope="trial")
def test_list():
    return [1, 2, 3, 4, 5]

def bench_append(test_list):
    \"\"\"Append to a list\"\"\"
    test_list.append(6)
    return len(test_list)

def bench_pop(test_list):
    \"\"\"Remove from a list\"\"\"
    return test_list.pop()
"""
        bench_file = create_module_file(
            temp_dir,
            "bench_test_fixture.py",
            fixture_content,
        )

        # Setup logging to capture logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

        # Run the CLI with the benchmark file
        abs_path = str(bench_file.parent.absolute())
        with patch(
            "sys.argv",
            [
                "easybench",
                "--trials",
                str(MIN_TEST_TRIALS),
                "--show-output",
                "--no-color",
                abs_path,
            ],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err  # Combine stdout and stderr

        # Check for key elements
        assert "Benchmark Results" in output

        # Check for any of our benchmark functions
        benchmark_functions = ["bench_append", "bench_pop"]
        found_benchmark = False
        for func_name in benchmark_functions:
            if func_name in output:
                found_benchmark = True
                break

        assert found_benchmark, "No benchmark function was found in output"

    def test_benchmark_with_memory_flag(
        self,
        temp_dir: Path,
        create_module_file: Callable,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test executing a benchmark with memory measurement."""
        # Create a benchmark file
        bench_content = """
def bench_memory_test():
    \"\"\"A memory-intensive benchmark\"\"\"
    # Create a large list to test memory usage
    big_list = [i for i in range(10000)]
    return len(big_list)
"""
        bench_file = create_module_file(temp_dir, "bench_memory.py", bench_content)

        # Run the CLI with memory flag
        abs_path = str(bench_file.parent.absolute())
        with patch(
            "sys.argv",
            ["easybench", "--memory", "--trials", "1", "--no-color", abs_path],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err  # Combine stdout and stderr

        # Check for memory measurement in output
        assert "Memory (KB)" in output, "Memory measurement not shown in output"

    def test_include_exclude_integration(
        self,
        temp_dir: Path,
        create_module_file: Callable,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test the include and exclude options in a real benchmark execution."""
        # Create a benchmark file with multiple benchmark functions
        bench_content = """
# Benchmark file with various functions for testing include/exclude
import time

def bench_test():
    \"\"\"A benchmark test function\"\"\"
    return 42

def bench_fast():
    \"\"\"A fast benchmark test function\"\"\"
    return "fast"

def bench_slow():
    \"\"\"A slow benchmark test function\"\"\"
    time.sleep(0.01)
    return "slow"

def not_a_bench():
    return "exclude me"
"""
        bench_file = create_module_file(
            temp_dir,
            "bench_test_include_exclude.py",
            bench_content,
        )

        # Run the CLI with include and exclude patterns
        abs_path = str(bench_file.parent.absolute())
        with patch(
            "sys.argv",
            [
                "easybench",
                "--trials",
                str(MIN_TEST_TRIALS),
                "--include",
                "bench_",
                "--exclude",
                "slow",
                "--no-color",
                "--no-progress",
                abs_path,
            ],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Check for core elements
        assert "Benchmark Results" in output

        # The following should be included (matches include pattern but not exclude)
        assert "bench_fast" in output
        assert "bench_test" in output

        # The following should be excluded (matches exclude pattern)
        assert "bench_slow" not in output
        assert "not_a_bench" not in output

    def test_include_exclude_filters_only_methods(
        self,
        temp_dir: Path,
        create_module_file: Callable,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that include/exclude patterns only filter benchmark method names."""
        # Create a benchmark file with a class name that contains 'slow' in it
        bench_content = """
from easybench import EasyBench

# This class has 'Slow' in its name, but --exclude=slow shouldn't filter it out
class Benchslow(EasyBench):
    def bench_fast(self):
        \"\"\"A fast benchmark method that should be included\"\"\"
        return "fast result"

    def bench_slow(self):
        \"\"\"A slow benchmark method that should be excluded\"\"\"
        return "slow result"

# Another class with no 'slow' in its name
class Benchfast(EasyBench):
    def bench_another_slow(self):
        \"\"\"A slow benchmark method that should be excluded\"\"\"
        return "another slow result"

    def bench_another_fast(self):
        \"\"\"A fast benchmark method that should be included\"\"\"
        return "another fast result"
"""
        # Deliberately name the file with 'slow' in it,
        # to test that file names aren't filtered
        bench_file = create_module_file(
            temp_dir,
            "slow_benchmark.py",
            bench_content,
        )

        # Run the CLI with exclude pattern that would match
        # both the class name and file name
        # But it should only filter method names
        abs_path = str(bench_file)
        with patch(
            "sys.argv",
            [
                "easybench",
                "--trials",
                "1",
                "--exclude",
                "slow",  # Should only exclude methods with 'slow' in name
                "--no-color",
                "--no-progress",
                abs_path,
            ],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Check that proper filtering occurred:
        # 1. Both classes should be found even though one has 'slow' in its name
        assert "Benchslow" in output
        assert "Benchfast" in output

        # 2. Methods with 'slow' in their names should be excluded
        assert "bench_fast" in output
        assert "bench_slow" not in output
        assert "bench_another_fast" in output
        assert "bench_another_slow" not in output

        # Now test with include pattern
        with patch(
            "sys.argv",
            [
                "easybench",
                "--trials",
                "1",
                "--include",
                "fast",  # Should only include methods with 'fast' in name
                "--no-color",
                "--no-progress",
                abs_path,
            ],
        ):
            cli_main()

        # Capture and check output
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Check proper filtering with include:
        # 1. Both classes should be included regardless of their names
        assert "Benchslow" in output
        assert "Benchfast" in output

        # 2. Only methods with 'fast' in their names should be included
        assert "bench_fast" in output
        assert "bench_slow" not in output
        assert "bench_another_fast" in output
        assert "bench_another_slow" not in output


def test_discover_benchmark_files_directory() -> None:
    """Test discovering benchmark files from a directory."""
    with (
        patch("pathlib.Path.is_dir", return_value=True),
        patch(
            "pathlib.Path.glob",
            return_value=[Path("bench_test1.py"), Path("bench_test2.py")],
        ),
    ):
        files = discover_benchmark_files("some_dir")
        num_files = 2
        assert len(files) == num_files
        assert files[0].name == "bench_test1.py"
        assert files[1].name == "bench_test2.py"


def test_discover_benchmark_files_single_file() -> None:
    """Test discovering benchmark files with a single file path."""
    with (
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.is_dir", return_value=False),
    ):
        # Valid benchmark filename
        test_file = Path("bench_test.py")
        files = discover_benchmark_files(test_file)
        assert len(files) == 1
        assert files[0] == test_file


def test_discover_benchmark_files_nonexistent() -> None:
    """Test discovering benchmark files with a non-existent path."""
    with (
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=False),
    ):
        files = discover_benchmark_files("nonexistent")
        assert len(files) == 0
