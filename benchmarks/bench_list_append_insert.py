"""Benchmark comparing different list operations like append and insert."""

from easybench import BenchConfig, EasyBench, fixture


@fixture(scope="trial")
def big_list() -> list[int]:
    """
    Create a large list for benchmarking purposes.

    Returns:
        list[int]: A list with one million integers.

    """
    return list(range(1_000_000))


class BenchList(EasyBench):
    """
    Benchmark class for comparing various list operations.

    Benchmarks the performance of different list operations like
    append, insert, and pop at various positions.
    """

    bench_config = BenchConfig(
        memory=True,
        sort_by="avg",
        trials=10,
    )

    def bench_append(self, big_list: list[int]) -> None:
        """
        Benchmark appending an item to the end of a list.

        Args:
            big_list: A large list to perform operations on.

        """
        big_list.append(-1)

    def bench_insert_start(self, big_list: list[int]) -> None:
        """
        Benchmark inserting an item at the start of a list.

        Args:
            big_list: A large list to perform operations on.

        """
        big_list.insert(0, -1)

    def bench_insert_middle(self, big_list: list[int]) -> None:
        """
        Benchmark inserting an item at the middle of a list.

        Args:
            big_list: A large list to perform operations on.

        """
        big_list.insert(len(big_list) // 2, -1)

    def bench_pop(self, big_list: list[int]) -> None:
        """
        Benchmark popping an item from the end of a list.

        Args:
            big_list: A large list to perform operations on.

        """
        big_list.pop()

    def bench_pop_zero(self, big_list: list[int]) -> None:
        """
        Benchmark popping an item from the start of a list.

        Args:
            big_list: A large list to perform operations on.

        """
        big_list.pop(0)


if __name__ == "__main__":
    BenchList().bench()
