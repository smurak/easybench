# mypy: disable-error-code="list-item"
"""Benchmark for sorting algorithm."""

import random
from heapq import heapify, heappop

from easybench import BenchConfig, BenchParams, EasyBench, parametrize


def get_random_list(size: int) -> list[int]:
    """Get random list."""
    nums = list(range(size))
    random.shuffle(nums)
    return nums


def get_params(size: int) -> BenchParams:
    """Get BenchParams."""
    return BenchParams(
        name=f"size: {size}",
        params={"arr": lambda s=size: get_random_list(s)},
    )


class BenchSort(EasyBench):
    """Benchmark class for sorting."""

    bench_config = BenchConfig(
        trials=100,
        warmups=20,
        memory="MB",
        sort_by="avg",
        progress=False,
        reporters=[
            "console",
            ("boxplot-sns", {"log_scale": True}),
            ("lineplot-sns", {"log_scale": True}),
        ],
        include=r"\(size: 100\)",
    )

    params_list: tuple[BenchParams, ...] = (get_params(100), get_params(1000))

    @parametrize(params_list)
    def bench_bubble_sort(self, arr: list[int]) -> list[int]:
        """Bubble sort."""
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    @parametrize(params_list)
    def bench_selection_sort(self, arr: list[int]) -> list[int]:
        """Perform selection sort."""
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @parametrize(params_list)
    def bench_insertion_sort(self, arr: list[int]) -> list[int]:
        """Perform insertion sort."""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @parametrize(params_list)
    def bench_merge_sort(self, arr: list[int]) -> list[int]:
        """Perform merge sort."""

        def merge_sort_algorithm(arr: list[int]) -> list[int]:
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = merge_sort_algorithm(arr[:mid])
            right = merge_sort_algorithm(arr[mid:])

            return merge(left, right)

        def merge(left: list[int], right: list[int]) -> list[int]:
            result = []
            i = j = 0

            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

            result.extend(left[i:])
            result.extend(right[j:])
            return result

        return merge_sort_algorithm(arr)

    @parametrize(params_list)
    def bench_quick_sort(self, arr: list[int]) -> list[int]:
        """Perform quick sort."""

        def quick_sort_algorithm(arr: list[int], low: int, high: int) -> None:
            if low < high:
                pivot_index = partition(arr, low, high)
                quick_sort_algorithm(arr, low, pivot_index - 1)
                quick_sort_algorithm(arr, pivot_index + 1, high)

        def partition(arr: list[int], low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1

            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]

            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1

        quick_sort_algorithm(arr, 0, len(arr) - 1)
        return arr

    @parametrize(params_list)
    def bench_python_sort(self, arr: list[int]) -> list[int]:
        """Perform Python's built-in sort."""
        arr.sort()
        return arr

    @parametrize(params_list)
    def bench_heap_sort(self, arr: list[int]) -> list[int]:
        """Perform heap sort."""

        def heapify(arr: list[int], n: int, i: int) -> None:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[largest] < arr[left]:
                largest = left

            if right < n and arr[largest] < arr[right]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        n = len(arr)

        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        # Extract elements one by one
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify(arr, i, 0)

        return arr

    @parametrize(params_list)
    def bench_heapq_sort(self, arr: list[int]) -> list[int]:
        """Perform heapq-based sort."""
        result = []
        arr_copy = arr.copy()
        heapify(arr_copy)
        while arr_copy:
            result.append(heappop(arr_copy))
        return result


if __name__ == "__main__":
    # Run the benchmark
    results = BenchSort().bench(include=".", trials=10, progress=True)
