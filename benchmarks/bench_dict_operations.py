"""Benchmark for comparing different dictionary operations in Python."""

from easybench import BenchConfig, EasyBench
from easybench.reporters import ConsoleReporter
from easybench.visualization import BoxPlotFormatter, PlotReporter


class BenchDictOperation(EasyBench):
    """
    Benchmark class for measuring performance of various dictionary operations.

    This class tests operations like adding, accessing, updating, and removing
    key-value pairs to compare their performance characteristics.
    """

    def __init__(self, size: int = 1_000_000) -> None:
        """
        Initialize benchmark with specified dictionary size.

        Args:
            size: The size of the dictionary to operate on

        """
        super().__init__()
        self.size = size

    def setup_trial(self) -> None:
        """Set up a fresh dictionary before each trial."""
        self.big_dict = {f"key_{i}": f"value_{i}" for i in range(self.size)}
        # Keys that don't exist in the dictionary
        self.new_key = f"key_{self.size + 1}"
        # Keys that exist in different positions
        self.first_key = "key_0"
        self.middle_key = f"key_{self.size // 2}"
        self.last_key = f"key_{self.size - 1}"

    def bench_add_new_key(self) -> None:
        """Benchmark adding a new key-value pair to the dictionary."""
        self.big_dict[self.new_key] = "key_-1"

    def bench_update_existing(self) -> None:
        """Benchmark updating an existing key in the dictionary."""
        self.big_dict[self.middle_key] = "key_-1"

    def bench_access_first(self) -> None:
        """Benchmark accessing the first key in the dictionary."""
        _ = self.big_dict[self.first_key]

    def bench_access_middle(self) -> None:
        """Benchmark accessing a key in the middle of the dictionary."""
        _ = self.big_dict[self.middle_key]

    def bench_access_last(self) -> None:
        """Benchmark accessing the last key in the dictionary."""
        _ = self.big_dict[self.last_key]

    def bench_get_existing(self) -> None:
        """Benchmark using .get() for an existing key."""
        _ = self.big_dict.get(self.middle_key)

    def bench_get_missing(self) -> None:
        """Benchmark using .get() for a missing key."""
        _ = self.big_dict.get(self.new_key)

    def bench_delete_key(self) -> None:
        """Benchmark removing a key from the dictionary."""
        del self.big_dict[self.middle_key]

    def bench_pop_key(self) -> None:
        """Benchmark popping a key from the dictionary."""
        self.big_dict.pop(self.middle_key)


if __name__ == "__main__":
    BenchDictOperation(size=100_000).bench(
        config=BenchConfig(
            trials=100,
            memory=False,
            sort_by="avg",
            reporters=[
                ConsoleReporter(),
                PlotReporter(
                    BoxPlotFormatter(
                        showfliers=True,
                        log_scale=True,
                        engine="seaborn",
                        orientation="horizontal",
                        width=0.5,
                        linewidth=0.5,
                    ),
                ),
            ],
        ),
    )
