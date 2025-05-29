# ruff: noqa: T201
"""
Run CI checks locally similar to GitHub Actions workflows.

This script runs tests and linting checks that mirror the GitHub Actions
configuration in the .github/workflows directory.
"""

import argparse
import os
import platform
import subprocess
import sys

# ANSI color codes for colorized output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def colorize(text: str, color: str) -> str:
    """Add color to text if supported."""
    if os.name == "nt" and os.environ.get("TERM") != "xterm":
        # Windows without proper terminal doesn't support ANSI codes
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def run_command(cmd: list[str], env: dict | None = None) -> bool:
    """Run a command and return True if it succeeds."""
    cmd_str = " ".join(cmd)
    print(colorize(f"\n=== Running: {cmd_str} ===", "bold"))

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(colorize(result.stderr, "yellow"))

        if result.returncode != 0:
            print(colorize(f"Command failed with exit code {result.returncode}", "red"))
            return False
        print(colorize("Command succeeded!", "green"))
    except Exception as e:  # noqa: BLE001
        print(colorize(f"Error executing command: {e}", "red"))
        return False
    else:
        return True


def run_tests(python_version: str | None = None) -> bool:
    """Run pytest with coverage."""
    print(colorize("\n=== Running Tests ===", "bold"))

    # Use current Python version if not specified
    if not python_version:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    print(f"Using Python {python_version}")

    cmd = ["pytest", "--cov=src/easybench", "tests/"]
    return run_command(cmd)


def run_black_check() -> bool:
    """Run black code formatter in check mode."""
    print(colorize("\n=== Running Black Format Check ===", "bold"))
    cmd = [sys.executable, "-m", "black", "--check", "."]
    return run_command(cmd)


def run_ruff_check() -> bool:
    """Run ruff linter."""
    print(colorize("\n=== Running Ruff Linter ===", "bold"))
    cmd = [sys.executable, "-m", "ruff", "check", "."]
    return run_command(cmd)


def run_mypy() -> bool:
    """Run mypy type checker."""
    print(colorize("\n=== Running MyPy Type Check ===", "bold"))
    cmd = [sys.executable, "-m", "mypy", "."]
    return run_command(cmd)


def run_all_checks(python_version: str | None = None) -> bool:
    """Run all checks and return True if all pass."""
    results = []

    # Display environment information
    print(colorize("=== Environment Information ===", "bold"))
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    # Run checks
    results.append(run_tests(python_version))
    results.append(run_black_check())
    results.append(run_ruff_check())
    results.append(run_mypy())

    # Summary
    print(colorize("\n=== Checks Summary ===", "bold"))
    all_passed = all(results)

    if all_passed:
        print(colorize("✓ All checks passed successfully!", "green"))
    else:
        print(colorize("✗ Some checks failed!", "red"))

    return all_passed


def main() -> int:
    """Parse arguments and run checks."""
    parser = argparse.ArgumentParser(
        description="Run CI checks locally similar to GitHub Actions.",
    )

    parser.add_argument("--test", action="store_true", help="Run pytest with coverage")
    parser.add_argument(
        "--black",
        action="store_true",
        help="Run black code format check",
    )
    parser.add_argument("--ruff", action="store_true", help="Run ruff linter")
    parser.add_argument("--mypy", action="store_true", help="Run mypy type checking")
    parser.add_argument("--python-version", help="Specify Python version to test with")

    args = parser.parse_args()

    # If no specific checks are specified, run all checks
    run_specific = args.test or args.black or args.ruff or args.mypy

    if not run_specific:
        success = run_all_checks(args.python_version)
    else:
        results = []
        if args.test:
            results.append(run_tests(args.python_version))
        if args.black:
            results.append(run_black_check())
        if args.ruff:
            results.append(run_ruff_check())
        if args.mypy:
            results.append(run_mypy())

        success = all(results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
