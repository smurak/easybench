# EasyBench

[![Tests](https://github.com/smurak/easybench/actions/workflows/test.yml/badge.svg)](https://github.com/smurak/easybench/actions)
[![Docs](https://readthedocs.org/projects/easybench/badge/?version=latest)](https://easybench.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/easybench.svg)](https://pypi.org/project/easybench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**Docs**](https://easybench.readthedocs.io/)

A simple and easy-to-use Python benchmarking library.

English | [日本語](README_ja.md)

## Features

- Three benchmarking styles: decorator, class-based, and command-line
- Measure both execution time and memory usage ([see limitations](#memory-measurement-limitations))
- Rich visualizations (Boxplot, Violinplot, Lineplot, Histogram, Barplot)
- Advanced options: warmup runs, multiple loops per trial, outlier trimming
- Parametrized benchmarks for comparing function performance with different inputs
- pytest-like fixtures and lifecycle hooks (setup/teardown)
- Flexible configuration: time/memory units, progress tracking, filtering
- Multiple output formats (text tables, CSV, JSON, pandas.DataFrame)
- Extensible reporting system for custom outputs

## Installation

```bash
pip install easybench
```

### Optional Dependencies

EasyBench supports optional dependencies for additional features:

```bash
# Install with visualization support
pip install easybench[all]
```

The `all` option includes:

- `matplotlib`: For visualization and plotting benchmark results
- `seaborn`: For enhanced statistical visualizations
- `pandas`: For outputting benchmark results as DataFrames
- `tqdm`: For progress tracking during benchmark execution

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smurak/easybench/blob/main/notebooks/easybench_quickstart.ipynb)

There are 3 ways to benchmark with `easybench`:

1. `@bench` decorator

    ```python
    from easybench import bench
    
    # Add @bench with function parameters
    @bench(item=123, big_list=lambda: list(range(1_000_000)))
    def insert_first(item, big_list):
        big_list.insert(0, item)
    ```

> [!TIP]  
> When you need fresh data for each trial, use a function or lambda to generate new data on demand.  
> (like `lambda: list(range(1_000_000))` in the above)

2. `EasyBench` class

    ```python
    from easybench import EasyBench, BenchConfig
    
    class BenchListOperation(EasyBench):
        # Benchmark configuration
        bench_config = BenchConfig(
            trials=10,
            memory=True,
            sort_by="avg"
        )
    
        # Setup for each trial
        def setup_trial(self):
            self.big_list = list(range(1_000_000))
    
        # Benchmark methods (must start with bench_)
        def bench_insert_first(self):
            self.big_list.insert(0, 123)
    
        # You can define multiple benchmark methods
        def bench_pop_first(self):
            self.big_list.pop(0)
    
    if __name__ == "__main__":
        # Run benchmark
        BenchListOperation().bench()
    ```

3. `easybench` command

    1. Create a `benchmarks` directory
    2. Put `bench_*.py` scripts in the directory:

        ```python
        from easybench import fixture
        
        # Fixture for each trial
        @fixture(scope="trial")
        def big_list():
            return list(range(1_000_000))
        
        # Benchmark functions (must start with bench_)
        def bench_insert_first(big_list):
            big_list.insert(0, 123)
        
        # You can define multiple benchmark functions
        def bench_pop_first(big_list):
            big_list.pop(0)
        ```

    3. Run `easybench` command

        ```bash
        easybench --trials 10 --memory --sort-by avg
        ```

**Example of benchmark results:**

* Single benchmark

    ```
    Benchmark Results (5 trials):
    
    Function        Avg Time (s)  Min Time (s)  Max Time (s)
    --------------------------------------------------------
    insert_first        0.001568      0.001071      0.003265
    ```

* Multiple benchmarks

  ![EasyBench Benchmark Result](https://raw.githubusercontent.com/smurak/easybench/main/images/easybench_screenshot.png)

* Boxplot Visualization

  ![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)

* Violinplot Visualization

  ![Violinplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_violinplot.png)

* Lineplot Visualization

  ![Lineplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_lineplot.png)

* Histogram Visualization

  ![Histplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_histplot.png)

* Barplot Visualization

  ![Barplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_barplot.png)

## Usage

For detailed usage instructions, please refer to the [**documentation**](https://easybench.readthedocs.io/).

## Memory Measurement Limitations

> [!NOTE]
> EasyBench uses Python's built-in `tracemalloc` module to measure memory usage.  
> This has some important limitations:
>
> - `tracemalloc` only tracks memory allocations made through Python's memory manager
> - Memory allocated by C extensions (like NumPy, Pandas, or other native libraries) often bypasses Python's memory manager and won't be accurately measured
> - The reported memory usage reflects Python objects only, not the total process memory consumption
>
> For applications heavily using C extensions, consider using external profilers like `memory_profiler` or system monitoring tools for more accurate measurements.

## License

[MIT](./LICENSE)
