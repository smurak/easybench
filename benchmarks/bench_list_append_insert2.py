"""Benchmark for comparing different list operations in Python."""

from easybench import BenchConfig, EasyBench, customize


class BenchListOperation(EasyBench):
    """
    Benchmark class for measuring performance of various list operations.

    This class tests operations like append, insert, and pop at different positions
    in a list to compare their performance characteristics.
    """

    def __init__(self, size: int = 1_000_000) -> None:
        """
        Initialize benchmark with specified list size.

        Args:
            size: The size of the list to operate on

        """
        self.size = size

    def setup_trial(self) -> None:
        """Set up a fresh list before each trial."""
        self.big_list = list(range(self.size))

    @customize(loops_per_trial=1000)
    def bench_append(self) -> None:
        """Benchmark appending an element to the end of the list."""
        self.big_list.append(-1)

    def bench_insert_start(self) -> None:
        """Benchmark inserting an element at the beginning of the list."""
        self.big_list.insert(0, -1)

    def bench_insert_middle(self) -> None:
        """Benchmark inserting an element in the middle of the list."""
        self.big_list.insert(len(self.big_list) // 2, -1)

    @customize(loops_per_trial=1000)
    def bench_pop(self) -> None:
        """Benchmark removing an element from the end of the list."""
        self.big_list.pop()

    def bench_pop_zero(self) -> None:
        """Benchmark removing an element from the beginning of the list."""
        self.big_list.pop(0)


if __name__ == "__main__":
    BenchListOperation(size=1_000_000).bench(
        config=BenchConfig(
            trials=100,
            warmups=100,
            loops_per_trial=100,
            memory=True,
            sort_by="avg",
            color=True,
        ),
    )
