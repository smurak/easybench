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

## Features

- Three flexible benchmarking styles (decorator, class-based, and command-line)
- Measure both execution time and estimated memory usage ([see limitations](#memory-measurement-limitations))
- Visualization of benchmark results as boxplots for analyzing distribution and outliers
- Parametrized benchmarks to compare the same function with different input sizes
- A pytest-like fixture system for easy test data setup
- Complete lifecycle hooks (setup/teardown) for fine-grained benchmark control
- Switch between normal function execution and benchmarked execution on demand
- Customizable benchmark configuration with sorting and formatting options
- Command-line tool to run multiple benchmarks at once
- Multiple output formats (text tables, CSV, JSON, pandas.DataFrame)
- Extensible reporting system for custom output destinations

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

    * When you need fresh data for each trial, use a function or lambda to generate new data on demand.  
      (like `lambda: list(range(1_000_000))` in the above)

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

## Usage

### Decorator-based Benchmarks (`@bench` decorator)

#### **Basic usage**

The `@bench` decorator provides the simplest way to benchmark a function:

```python
from easybench import bench

# Add @bench with function parameters
@bench(item=123, big_list=list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
```

#### **Fresh data**

In the example above, `big_list` is created once and the same list is used for all trials.  
When you need fresh data for each trial, use a function or lambda to generate new data on demand:

```python
from easybench import bench

# Create a new list for each trial
@bench(item=-1, big_list=lambda: list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
```

#### **Function parameters** (`@bench.fn_params`)

Sometimes you may want to use functions as parameters.  
In such cases, use the `@bench.fn_params` decorator:

```python
def pop_first(some_list):
    """Remove the first element from the list"""
    some_list.pop(0)

@bench(big_list=list(range(1_000_000)))
@bench.fn_params(func=pop_first)
def apply_function(big_list, func):
    func(big_list)
```

#### **Configuration** (`@bench.config`)

To customize benchmark settings, use the `@bench.config` decorator:

```python
@bench(big_list=list(range(10_000_000)))
@bench.config(trials=10, memory=True)
def pop_first(big_list):
    big_list.pop(0)
```

* Place the `@bench.config` decorator before (below) other bench decorators.  


Main configuration options:

* `trials`: Number of trials (default: `5`)
* `memory`: Also measure memory usage
  * `False`: Disable memory measurement (default)
  * `True`: Show memory in kilobytes
  * `"B"`, `"KB"`, `"MB"`, `"GB"`: Show memory in bytes, kilobytes, megabytes, or gigabytes
* `time`: Specify time measurement unit
  * `"s"`: Display time in seconds (default)
  * `"ms"`: Display time in milliseconds
  * `"μs"` or `"us"`: Display time in microseconds
  * `"ns"`: Display time in nanoseconds
  * `"m"`: Display time in minutes
* For other options, see "Configuration Options" below

#### **Multiple parameter sets** (`BenchParams`)

When you want to benchmark a function with multiple parameter sets,
you can pass a list of parameter sets created with `BenchParams` to the `@bench` decorator:

```python
from easybench import bench, BenchParams

# Define parameter sets
small = BenchParams(
    name="Small",                                 # Parameter set name
    params={"lst": lambda: list(range(10_000))},  # Parameters for @bench
)
large = BenchParams(
    name="Large",
    params={"lst": lambda: list(range(1_000_000))}
)

# Benchmark with multiple parameter sets
@bench([small, large])
def pop_first(lst):
    return lst.pop(0)
```

#### **On-demand benchmarking**

If you want to execute a function while simultaneously measuring its performance, use the `.bench()` method:

```python
@bench
def insert_first(item, big_list):
    big_list.insert(0, item)
    return len(big_list)

# Run as a normal function (without benchmarking)
result = insert_first(3, list(range(1_000_000)))

# Run with benchmarking
result = insert_first.bench(3, list(range(1_000_000)))
print(result)  # 1000001
```
* By default, the benchmark runs for `1` trial.
* To run multiple trials, specify the `bench_trials` parameter:
  ```python
  result = insert_first.bench(3, list(range(1_000_000)), bench_trials=10)
  ```
* When running multiple trials, the `.bench()` method returns the value from the first trial.
  ```python
  result = insert_first.bench(3, list(range(100_000)), bench_trials=10)
  print(result)  # 100001
  ```

### Class-based Benchmarks (`EasyBench` class)

For comparing multiple benchmarks or more complex setups, the class-based approach is useful:

```python
from easybench import EasyBench, BenchConfig

class BenchListOperation(EasyBench):

    # Benchmark configuration
    bench_config = BenchConfig(
        trials=10,     # Number of trials
        memory=True,   # Measure memory usage
        sort_by="avg"  # Sort by average time
    )

    # Runs before each trial
    def setup_trial(self):
        self.big_list = list(range(10_000_000))

    # Benchmark methods (must start with bench_)
    def bench_insert_first(self):
        self.big_list.insert(0, 123)

    def bench_pop_first(self):
        self.big_list.pop(0)

if __name__ == "__main__":
    BenchListOperation().bench()
```

How to use the class-based approach:
1. Create a class that inherits from `EasyBench`
2. Configure benchmark settings with the `bench_config` class variable
3. Prepare for each trial in the `setup_trial` method
4. Methods starting with `bench_` will be benchmarked
5. Call the `bench()` method to execute the benchmarks
   * `bench()` displays the results on screen and returns a dictionary of measured value

#### Lifecycle Methods

In class-based benchmarks, you can use the following lifecycle methods:

```python
class BenchExample(EasyBench):
    def setup_class(self):
        # Run once before all benchmarks in the class
        pass
        
    def teardown_class(self):
        # Run once after all benchmarks in the class
        pass
        
    def setup_function(self):
        # Run before each benchmark function
        pass
        
    def teardown_function(self):
        # Run after each benchmark function
        pass
        
    def setup_trial(self):
        # Run before each trial
        pass
        
    def teardown_trial(self):
        # Run after each trial
        pass
```

#### Parametrized Benchmarks (`parametrize` decorator)

You can run the same benchmark method with different parameter sets using the `parametrize` decorator:

```python
from easybench import BenchParams, EasyBench, parametrize

class BenchListOperations(EasyBench):
    # Define parameter sets with BenchParams
    small_params = BenchParams(
        name="Small List",
        params={"size": 10_000}
    )
    
    large_params = BenchParams(
        name="Large List",
        params={"size": 1_000_000}
    )
    
    # Apply parametrize decorator with a list of parameter sets
    @parametrize([small_params, large_params])
    def bench_create_list(self, size):
        return list(range(size))

if __name__ == "__main__":
    BenchListOperations().bench()
```

This will run the benchmark with each parameter set and include the parameter set name in the results:

```
Benchmark Results (5 trials):

Function                         Avg Time (s) Min Time (s) Max Time (s)
--------------------------------------------------------------------
bench_create_list (Small List)   0.000442     0.000309     0.000855    
bench_create_list (Large List)   0.092680     0.062617     0.129535    
```

#### Fixtures (`fixture` decorator)

To provide common test data, you can use pytest-style fixtures:

```python
from easybench import EasyBench, fixture

# Define a fixture
@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

class BenchListOperation(EasyBench):
    # Receive the fixture as an argument
    def bench_insert_first(self, big_list):
        big_list.insert(0, -1)

    def bench_pop_first(self, big_list):
        big_list.pop(0)

if __name__ == "__main__":
    BenchListOperation().bench()
```

The `scope` parameter of the `fixture` decorator specifies the lifetime of the fixture:
- `"trial"`: Created for each trial (default)
- `"function"`: Created once per benchmark function
- `"class"`: Created once per benchmark class


#### Configuration Options

The following settings are available in the `BenchConfig` class:

```python
from easybench import BenchConfig, EasyBench, customize

class MyBenchmark(EasyBench):
    bench_config = BenchConfig(
        trials=5,               # Number of trials
        warmups=2,              # Number of warmup trials before actual measurement
        sort_by="avg",          # Sort criterion
        reverse=False,          # Sort order (False=ascending, True=descending)
        memory="MB",            # Enable memory measurement and show in megabytes
        color=True,             # Use color output in results
        show_output=False,      # Display function return values
        loops_per_trial=1,      # Number of function executions per trial (see explanation below)
        reporters=["console"],  # Custom reporters (see explanation below)
        progress=True,          # Enable progress tracking with tqdm
        include="list",         # Only run benchmarks with "list" in their name
        exclude="slow",         # Skip benchmarks with "slow" in their name
    )
    
    # You can also customize settings for individual methods
    @customize(loops_per_trial=1000)
    def bench_fast_operation(self):
        # This method uses 1000 loops per trial
        pass
```

Sorting options (`sort_by`):

- `"def"`: Definition order (default)
- `"avg"`: Average execution time
- `"min"`: Minimum execution time
- `"max"`: Maximum execution time
- `"avg_memory"`: Average memory usage (when `memory=True`)
- `"max_memory"`: Maximum memory usage (when `memory=True`)

Memory measurement options (`memory`):

- `False`: Disable memory measurement (default)
- `True`: Enable memory measurement and display in kilobytes
- `"B"`: Display memory usage in bytes
- `"KB"`: Display memory usage in kilobytes
- `"MB"`: Display memory usage in megabytes
- `"GB"`: Display memory usage in gigabytes

Time measurement options (`time`):

- `"s"`: Display time in seconds (default)
- `"ms"`: Display time in milliseconds
- `"μs"` or `"us"`: Display time in microseconds
- `"ns"`: Display time in nanoseconds
- `"m"`: Display time in minutes

Progress tracking options (`progress`):

- `False`: Disable progress tracking
- `True`: Enable progress tracking using tqdm (default)
- Custom function: Use a custom progress tracking function that follows the tqdm interface

Benchmark selection options:

- `include`: Regular expression pattern to include only benchmark functions with matching names
- `exclude`: Regular expression pattern to exclude benchmark functions with matching names

  - For parametrized benchmarks, these options match against the full name (e.g., "bench_func (param_name)")
  - When both options are specified, `exclude` takes precedence

### Improving Measurement Accuracy with `warmups`

When benchmarking, the initial runs might be affected by various factors like code compilation, 
cache warmup, or other system effects. To get more stable and accurate measurements, you can 
use the `warmups` parameter to specify how many trial runs should be performed before the actual 
measurement begins:

```python
@bench
@bench.config(trials=5, warmups=3, time="ms")
def my_function():
    # This function will be run 3 times as warmup (results discarded)
    # before the 5 actual trials that are measured
    # ...
```

How `warmups` works:

- Before actual measurements begin, the function is executed `warmups` times
- Each warmup is a complete trial execution including setup_trial/teardown_trial
- Results from warmup trials are discarded and not included in measurements
- After warmups are complete, regular trials begin with results being recorded

When to use:

- For functions that need JIT compilation to reach optimal performance
- When the system needs time to "warm up" caches or reach steady state
- When you notice that the first few runs consistently show different performance characteristics


### Improving Timer Precision with `loops_per_trial`

In environments with poor timer resolution (e.g., certain virtual machines or systems where `time.perf_counter()` has limited precision), you may need to run a function multiple times to get meaningful timing results.

Additionally, when benchmarking very fast operations (less than a few microseconds), the overhead of function calls themselves can significantly impact measurement results.  
In such cases, using the `loops_per_trial` parameter can help distribute the function call overhead and achieve more accurate measurements.

The `loops_per_trial` parameter allows you to specify how many times a function should be executed in a single timing measurement (trial):

```python
# Measure the average time it takes to append the number 1 to a list
# that initially contains 100 elements, repeated 10000 times
# (Note: the same list instance is used throughout the 10000 append operations)
# Repeat this process 500 times to perform the benchmark
@bench(small_list=lambda: list(range(100)))
@bench.config(trials=500, loops_per_trial=10000, time="us")
def append_item(small_list):
    small_list.append(1)
```

How `loops_per_trial` works:

- The function is executed `loops_per_trial` times in a loop within a single timing measurement (trial)
- The total execution time is divided by `loops_per_trial` to get the average time per execution
- This provides more accurate measurements for very fast operations where individual timing would be affected by timer resolution limits

When to use:

- For very fast operations (microsecond or nanoseconds)
- In environments with poor timer precision
- When you notice high variability in timing results for simple operations

#### Memory Measurement Limitations

EasyBench uses Python's built-in `tracemalloc` module to measure memory usage.  
This has some important limitations:
- `tracemalloc` only tracks memory allocations made through Python's memory manager
- Memory allocated by C extensions (like NumPy, Pandas, or other native libraries) often bypasses Python's memory manager and won't be accurately measured
- The reported memory usage reflects Python objects only, not the total process memory consumption
For applications heavily using C extensions, consider using external profilers like `memory_profiler` or system monitoring tools for more accurate measurements.

### Command Line Interface (`easybench` command)

To run multiple benchmarks at once, use the `easybench` command:

```bash
easybench [options] [path]
```

* By default, it runs files named `bench_*.py` in the `benchmarks` directory
* You can specify either a directory containing benchmark files or a specific benchmark file
* Benchmark scripts must follow these rules:
  * For class-based benchmarks, class names should start with `Bench`
  * For function-based benchmarks, function names should start with `bench_`

#### Command Options

```bash
easybench [--trials N] [--loops-per-trial N] [--warmups N] [--memory] [--memory-unit UNIT] [--sort-by METRIC] [--reverse] [--no-color] [--show-output] [--time-unit UNIT] [--no-progress] [--include PATTERN] [--exclude PATTERN] [--include-files PATTERN] [--exclude-files PATTERN] [path]
```

- `--trials N`: Number of trials (default: 5)
- `--loops-per-trial N`: Number of loops to run per trial for improved precision
- `--warmups N`: Number of warmup trials to run before timing
- `--memory`: Enable memory measurement
- `--memory-unit UNIT`: Memory unit for displaying results (B/KB/MB/GB)
- `--sort-by METRIC`: Sort criterion (def/avg/min/max/avg_memory/max_memory)
- `--reverse`: Sort results in descending order
- `--no-color`: Disable colored output
- `--show-output`: Display function return values
- `--time-unit UNIT`: Time unit for displaying results (s/ms/us/ns/m)
- `--no-progress`: Disable progress bars during benchmarking
- `--include PATTERN`: Regular expression pattern to include only matching benchmark functions
- `--exclude PATTERN`: Regular expression pattern to exclude matching benchmark functions
- `--include-files PATTERN`: Regular expression pattern to include only matching benchmark files
- `--exclude-files PATTERN`: Regular expression pattern to exclude matching benchmark files
- `path`: Directory containing benchmark files or a specific benchmark file (default: "benchmarks")

#### Function-based Benchmark Example

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

## Advanced Usage

### Custom Output (`Formatter` and `Reporter`)

EasyBench uses a mechanism called **Reporter** to output benchmark results. By default, `ConsoleReporter` is used.

`ConsoleReporter` is a reporter that outputs data to the console screen, and by default formats the data in tabular form (`TableFormatter`). In EasyBench, you can change both the `Formatter` (output format) and the `Reporter` (output method) to enable various forms of output.

#### Using Reporters

To use reporters, set them as a list in the `reporters` parameter of your benchmark configuration (`BenchConfig` or `@bench.config`). Since `reporters` is a list, you can specify multiple output methods simultaneously.

There are three ways to specify reporters:

1. **As a string**: Specify the reporter name as a string
   - `"console"`: Standard tabular console output
   - `"simple"`: Simple console output
   - `"boxplot"`: Visualization with boxplot
   - `"violinplot"`: Visualization with violinplot
   - `"boxplot-sns"`: Visualization with seaborn-styled boxplot
   - `"violinplot-sns"`: Visualization with seaborn-styled violinplot
   - `"lineplot"`: Visualization with lineplot
   - `"lineplot-sns"`: Visualization with seaborn-styled lineplot
   - `"*.csv"` or `"*.json"`: File output

2. **With arguments**: Specify in the format `(reporter_name, parameter_dict)`

3. **As Reporter objects**: Directly specify an instance of a Reporter class

* Usage example

    ```python
    from easybench import BenchConfig
    from easybench.reporters import FileReporter
    
    # Multiple output formats with different specification methods
    config = BenchConfig(
        reporters=[
            "console",                          # Specified as string
            ("simple", {"metric": "min"}),      # Specified with arguments
            ("boxplot", {"log_scale": False}),  # Plot with arguments
            "results.csv",                      # Specified as file path
            FileReporter("results.json"),       # Specified as object
        ]
    )
    ```

#### Creating Custom Reporters

For advanced use cases, you can create custom reporters:

```python
from easybench.reporters import (
    Reporter, TableFormatter, JSONFormatter, CSVFormatter
)

# Custom reporter example - sends results to a web API
class WebAPIReporter(Reporter):
    def __init__(self, api_url, auth_token):
        super().__init__(JSONFormatter())  # Use JSON format
        self.api_url = api_url
        self.auth_token = auth_token
    
    def report_formatted(self, formatted_output):
        # Send formatted results to an API endpoint
        import requests
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        requests.post(self.api_url, headers=headers, json=formatted_output)

# Use with BenchConfig
bench_config = BenchConfig(
    reporters=[
        ConsoleReporter(),  # Still show in console
        WebAPIReporter("https://api.example.com/benchmarks", "my_token")
    ]
)
```

### Boxplot Visualization (`BoxPlotFormatter`)

You can visualize benchmark results as boxplots, which is useful for analyzing distribution and outliers across multiple trials:

```python
from easybench import BenchConfig, EasyBench, customize
from easybench.visualization import BoxPlotFormatter, PlotReporter


class BenchList(EasyBench):
    bench_config = BenchConfig(
        trials=100,
        warmups=100,
        loops_per_trial=100,
        reporters=[
            PlotReporter(
                BoxPlotFormatter(
                    showfliers=True,           # Show outliers
                    log_scale=True,            # Use logarithmic scale
                    engine="seaborn",          # Use seaborn as plotting engine
                    orientation="horizontal",  # Horizontal or vertical orientation
                    width=0.5,                 # Box width (passed directly to seaborn's boxplot)
                    linewidth=0.5,             # Line width (passed directly to seaborn's boxplot)
                ),
            ),
        ],
    )

    def setup_trial(self):
        self.big_list = list(range(1_000_000))

    @customize(loops_per_trial=1000)
    def bench_append(self):
        self.big_list.append(-1)

    def bench_insert_start(self):
        self.big_list.insert(0, -1)

    def bench_insert_middle(self):
        self.big_list.insert(len(self.big_list) // 2, -1)

    @customize(loops_per_trial=1000)
    def bench_pop(self):
        self.big_list.pop()

    def bench_pop_zero(self):
        self.big_list.pop(0)


if __name__ == "__main__":
    BenchList().bench()
```

![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)

### Main `BoxPlotFormatter` options

- `showfliers`: Whether to show outliers (default: `True`)
- `log_scale`: Whether to use logarithmic scale (default: `False`)
- `data_limit`: Specify axis data range (e.g., `(0, 0.01)`)
- `trim_outliers`: Percentile for trimming outliers (0.0 to 0.5)
- `winsorize_outliers`: Percentile for winsorizing outliers (0.0 to 0.5)
- `figsize`: Figure size (default: `(10, 6)`)
- `engine`: Plotting engine (`"matplotlib"` or `"seaborn"`)
- `orientation`: Boxplot orientation (`"vertical"` or `"horizontal"`)
- `sns_theme`: Dictionary of seaborn theme parameters (e.g., `{"style": "darkgrid", "palette": "Set2"}`)

### `PlotReporter` options

- `formatter`: Plot formatter to use (e.g., `BoxPlotFormatter`)
- `show`: Whether to display the plot on screen (default: `True`)
- `save_path`: File path to save the plot
- `dpi`: Image resolution (default: `100`)


To use boxplots, you need to install `matplotlib`:

```bash
pip install matplotlib
```

If you want to use the seaborn engine, also install `seaborn`:

```bash
pip install seaborn
```

## License

[MIT](https://github.com/smurak/easybench/blob/main/LICENSE)
