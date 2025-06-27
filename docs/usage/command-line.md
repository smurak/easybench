## Command Line Interface (`easybench` command)

To run multiple benchmarks at once, use the `easybench` command:

```bash
easybench [options] [path]
```

* By default, it runs files named `bench_*.py` in the `benchmarks` directory
* You can specify either a directory containing benchmark files or a specific benchmark file
* Your benchmark scripts need to follow these conventions:
    * Class-based benchmarks: Any class that inherits from the `EasyBench` base class will be automatically detected and included
    * Function-based benchmarks: Functions whose names begin with `bench_` will be recognized as benchmark functions

### Command Options

```bash
easybench [--trials N] [--loops-per-trial N] [--warmups N] [--memory] [--memory-unit UNIT] [--sort-by METRIC] [--reverse] [--no-color] [--show-output] [--time-unit UNIT] [--no-progress] [--progress] [--include PATTERN] [--exclude PATTERN] [--include-files PATTERN] [--exclude-files PATTERN] [--no-time] [--clip-outliers VALUE] [path]
```

- `--trials N`: Number of trials (default: 5)
- `--loops-per-trial N`: Number of loops to run per trial for improved precision
- `--warmups N`: Number of warmup trials to run before benchmarking
- `--memory`: Enable memory measurement
- `--memory-unit UNIT`: Memory unit for displaying results (B/KB/MB/GB)
- `--sort-by METRIC`: Sort criterion (def/avg/min/max/avg_memory/max_memory)
- `--reverse`: Sort results in descending order
- `--no-color`: Disable colored output
- `--show-output`: Display function return values
- `--time-unit UNIT`: Time unit for displaying results (s/ms/us/ns/m)
- `--no-progress`: Disable progress bars during benchmarking
- `--progress`: Enable progress bars during benchmarking
- `--include PATTERN`: Regular expression pattern to include only matching benchmark functions
- `--exclude PATTERN`: Regular expression pattern to exclude matching benchmark functions
- `--include-files PATTERN`: Regular expression pattern to include only matching benchmark files
- `--exclude-files PATTERN`: Regular expression pattern to exclude matching benchmark files
- `--no-time`: Disable time measurement reporting
- `--clip-outliers VALUE`: Clip values on both sides (minimum and maximum) based on the specified proportion (between 0 and 0.5)
- `path`: Directory containing benchmark files or a specific benchmark file (default: "benchmarks")

### Function-based Benchmark Example

Example of a function-based benchmark to be run from the command line:

```python
# Filename: benchmarks/bench_list_operations.py
from easybench import fixture

@fixture(scope="trial")
def big_list():
    return list(range(1_000_000))

def bench_insert_first(big_list):
    """Insert an element at the beginning of the list"""
    big_list.insert(0, 123)

def bench_pop_first(big_list):
    """Remove the first element from the list"""
    big_list.pop(0)
```

Save this file in the `benchmarks` folder and run the `easybench` command to benchmark both functions and compare the results:

```bash
easybench --trials 10 --memory
```

You can also run a specific benchmark file directly:

```bash
easybench --trials 10 --memory benchmarks/bench_list_operations.py
```
