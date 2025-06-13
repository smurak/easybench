"""
Tests for utility functions in easybench.

This module contains test cases for the utility functions in the easybench package,
particularly focusing on environment information gathering.
"""

import sys
from unittest.mock import patch

from easybench import get_bench_env
from easybench.utils import (
    measure_timer_overhead,
    visual_ljust,
    visual_rjust,
    visual_width,
)


class TestGetBenchEnv:
    """Tests for the get_bench_env function."""

    def test_get_bench_env_structure(self) -> None:
        """Test that get_bench_env returns a dictionary with the expected structure."""
        env_info = get_bench_env()

        # Check the top-level keys
        assert isinstance(env_info, dict)
        assert "os" in env_info
        assert "cpu" in env_info
        assert "python" in env_info
        assert "perf_counter" in env_info

        # Check the OS section
        os_info = env_info["os"]
        assert isinstance(os_info, dict)
        assert "system" in os_info
        assert "release" in os_info
        assert "version" in os_info
        assert "architecture" in os_info
        assert "machine" in os_info

        # Check the CPU section
        cpu_info = env_info["cpu"]
        assert isinstance(cpu_info, dict)
        assert "count" in cpu_info
        assert "processor" in cpu_info

        # Check the Python section
        python_info = env_info["python"]
        assert isinstance(python_info, dict)
        assert "version" in python_info
        assert "implementation" in python_info
        assert "compiler" in python_info

        # Check the perf_counter section
        perf_counter_info = env_info["perf_counter"]
        assert isinstance(perf_counter_info, dict)
        assert "resolution" in perf_counter_info
        assert "implementation" in perf_counter_info
        assert "monotonic" in perf_counter_info
        assert "timer_overhead" in perf_counter_info

    def test_get_bench_env_values(self) -> None:
        """Test that get_bench_env returns valid values for each key."""
        env_info = get_bench_env()

        # Verify OS values
        assert isinstance(env_info["os"]["system"], str)
        assert isinstance(env_info["os"]["release"], str)
        assert isinstance(env_info["os"]["version"], str)
        assert isinstance(env_info["os"]["architecture"], str)
        assert isinstance(env_info["os"]["machine"], str)

        # Verify CPU values
        cpu_info = env_info["cpu"]
        assert isinstance(cpu_info["count"], (int, str))
        if isinstance(cpu_info["count"], int):
            assert cpu_info["count"] > 0
        assert isinstance(cpu_info["processor"], str)

        # Verify Python values
        python_info = env_info["python"]
        assert isinstance(python_info["version"], str)
        assert isinstance(python_info["implementation"], str)
        assert isinstance(python_info["compiler"], str)

        # Verify perf_counter values
        perf_counter_info = env_info["perf_counter"]
        assert isinstance(perf_counter_info["resolution"], float)
        assert isinstance(perf_counter_info["implementation"], str)
        assert isinstance(perf_counter_info["monotonic"], bool)
        assert isinstance(perf_counter_info["timer_overhead"], float)
        assert perf_counter_info["timer_overhead"] > 0

    def test_measure_timer_overhead(self) -> None:
        """Test that measure_timer_overhead returns a positive float value."""
        overhead = measure_timer_overhead(iterations=1_000_000)
        assert isinstance(overhead, float)
        assert overhead > 0

    def test_pyodide_detection(self) -> None:
        """Test pyodide detection logic in get_bench_env."""
        # First test without pyodide in sys.modules
        if "pyodide" in sys.modules:
            # Save the original module
            original_pyodide = sys.modules["pyodide"]
            del sys.modules["pyodide"]

        try:
            env_info = get_bench_env()
            assert "environment" not in env_info["python"]
            assert "pyodide_version" not in env_info["python"]
        finally:
            # Restore pyodide if it was originally present
            if "original_pyodide" in locals():
                sys.modules["pyodide"] = original_pyodide

        # Now test with a mock pyodide module
        mock_pyodide = type("pyodide", (), {"__version__": "1.0.0"})()
        with patch.dict("sys.modules", {"pyodide": mock_pyodide}):
            env_info = get_bench_env()
            assert env_info["python"]["environment"] == "pyodide"
            assert env_info["python"]["pyodide_version"] == "1.0.0"

        # Test with pyodide module without __version__
        mock_pyodide_no_version = type("pyodide", (), {})()
        with patch.dict("sys.modules", {"pyodide": mock_pyodide_no_version}):
            env_info = get_bench_env()
            assert env_info["python"]["environment"] == "pyodide"
            assert env_info["python"]["pyodide_version"] == "unknown"


class TestVisualWidthFunctions:
    """Tests for the visual width calculation and text alignment functions."""

    # ruff: noqa: PLR2004
    def test_visual_width(self) -> None:
        """Test that visual_width correctly calculates display width of strings."""
        # ASCII characters should have width 1
        assert visual_width("hello") == 5
        assert visual_width("hello world") == 11
        assert visual_width("123") == 3

        # Empty string should have width 0
        assert visual_width("") == 0

        # Wide characters (CJK) should have width 2
        assert visual_width("漢字") == 4  # 2 characters, each width 2
        assert visual_width("テスト") == 6  # 3 characters, each width 2
        assert visual_width("안녕하세요") == 10  # 5 Korean characters, each width 2

        # Mixed width strings
        assert visual_width("hello漢字") == 9  # 5 (ASCII) + 4 (CJK)
        assert visual_width("テスト123") == 9  # 6 (CJK) + 3 (ASCII)
        assert visual_width("漢字abc漢字") == 11  # 4 (CJK) + 3 (ASCII) + 4 (CJK)

        # Special characters - most non-wide characters have width 1
        assert visual_width("!@#$%^&*()") == 10
        assert visual_width("こんにちは! Hello!") == 18  # 10 (CJK) + 8 (ASCII)

    def test_visual_ljust(self) -> None:
        """Test that visual_ljust correctly left-justifies text."""
        # Basic ASCII strings
        assert (
            visual_ljust("hello", 10) == "hello     "
        )  # 5 chars + 5 spaces = 10 width
        assert visual_ljust("x", 5) == "x    "  # 1 char + 4 spaces = 5 width

        # Wide characters
        assert visual_ljust("漢字", 10) == "漢字      "  # 4 width + 6 spaces = 10 width
        assert (
            visual_ljust("テスト", 10) == "テスト    "
        )  # 6 width + 4 spaces = 10 width

        # Mixed width characters
        assert (
            visual_ljust("hello漢字", 15) == "hello漢字      "
        )  # 9 width + 6 spaces = 15 width

        # When string is already wider than target width
        assert visual_ljust("hello", 3) == "hello"  # No truncation, returns original

        # With custom fill character
        assert visual_ljust("hello", 10, "-") == "hello-----"
        assert visual_ljust("漢字", 8, "=") == "漢字===="

        # Empty string
        assert visual_ljust("", 5) == "     "  # 5 spaces

    def test_visual_rjust(self) -> None:
        """Test that visual_rjust correctly right-justifies text."""
        # Basic ASCII strings
        assert (
            visual_rjust("hello", 10) == "     hello"
        )  # 5 spaces + 5 chars = 10 width
        assert visual_rjust("x", 5) == "    x"  # 4 spaces + 1 char = 5 width

        # Wide characters
        assert visual_rjust("漢字", 10) == "      漢字"  # 6 spaces + 4 width = 10 width
        assert (
            visual_rjust("テスト", 10) == "    テスト"
        )  # 4 spaces + 6 width = 10 width

        # Mixed width characters
        assert (
            visual_rjust("hello漢字", 15) == "      hello漢字"
        )  # 6 spaces + 9 width = 15 width

        # When string is already wider than target width
        assert visual_rjust("hello", 3) == "hello"  # No truncation, returns original

        # With custom fill character
        assert visual_rjust("hello", 10, "-") == "-----hello"
        assert visual_rjust("漢字", 8, "=") == "====漢字"

        # Empty string
        assert visual_rjust("", 5) == "     "  # 5 spaces
