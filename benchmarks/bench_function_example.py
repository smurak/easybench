"""Example benchmark functions for testing list operations performance."""

from easybench import fixture


@fixture(scope="trial")
def sample_list() -> list[int]:
    """
    Create a sample list for benchmarking.

    Returns:
        list[int]: A list containing integers from 0 to 99999.

    """
    return list(range(100000))


def bench_append(sample_list: list[int]) -> None:
    """Benchmark appending to a list."""
    sample_list.append(42)


def bench_insert_start(sample_list: list[int]) -> None:
    """Benchmark inserting at the beginning of a list."""
    sample_list.insert(0, 42)


def bench_insert_middle(sample_list: list[int]) -> None:
    """Benchmark inserting at the middle of a list."""
    sample_list.insert(len(sample_list) // 2, 42)


def bench_pop_end(sample_list: list[int]) -> None:
    """Benchmark popping from the end of a list."""
    sample_list.pop()


def bench_pop_start(sample_list: list[int]) -> None:
    """Benchmark popping from the beginning of a list."""
    sample_list.pop(0)
