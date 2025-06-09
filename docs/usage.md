# Usage

## Decorator-based Benchmarks (`@bench` decorator)

### **Basic usage**

The `@bench` decorator provides the simplest way to benchmark a function:

```python
from easybench import bench

# Add @bench with function parameters
@bench(item=123, big_list=list(range(1_000_000)))
def add_item(item, big_list):
    big_list.append(item)
```

### **Fresh data**

In the example above, `big_list` is created once and the same list is used for all trials.  
When you need fresh data for each trial, use a function or lambda to generate new data on demand:

```python
from easybench import bench

# Create a new list for each trial
@bench(item=-1, big_list=lambda: list(range(1_000_000)))
def append(item, big_list):
    big_list.append(item)
```

### **Function parameters** (`@bench.fn_params`)

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

### **Configuration** (`@bench.config`)

To customize benchmark settings, use the `@bench.config` decorator:

```python
@bench(big_list=list(range(10_000_000)))
@bench.config(trials=10, memory=True)
def pop_last(big_list):
    big_list.pop()
```

!!! tip
    Place the `@bench.config` decorator before (below) other bench decorators.  


Main configuration options:

* `trials`: Number of trials (default: `5`)
* `memory`: Also measure memory usage (default: `False`)
  * `False`: Disable memory measurement
  * `True`: Show memory in kilobytes
  * `"B"`, `"KB"`, `"MB"`, `"GB"`: Show memory in bytes, kilobytes, megabytes, or gigabytes
* For other options, see "Configuration Options" below

### **Multiple parameter sets** (`BenchParams`)

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

### **On-demand benchmarking**

If you want to run the benchmark only when needed, use the `.bench()` method:

```python
@bench
def append_item(item, big_list):
    big_list.append(item)
    return len(big_list)

# Run as a normal function (without benchmarking)
result = append_item(3, list(range(1_000_000)))

# Run with benchmarking
result = append_item.bench(3, list(range(1_000_000)))
print(result)  # 10000001
```

* By default, the benchmark runs for `1` trial.
* To run multiple trials, specify the `bench_trials` parameter:
  ```python
  result = append_item.bench(3, list(range(1_000_000)), bench_trials=10)
  ```
* When running multiple trials, the `.bench()` method returns the value from the first trial.
  ```python
  result = append_item.bench(3, [1,2,3], bench_trials=10)
  print(result)  # 4
  ```

---

## Class-based Benchmarks (`EasyBench` class)

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
    def bench_append(self):
        self.big_list.append(-1)

    def bench_insert_start(self):
        self.big_list.insert(0, -1)

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

### Lifecycle Methods

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

### Fixtures (`fixture` decorator)

To provide common test data, you can use pytest-style fixtures:

```python
from easybench import EasyBench, fixture

# Define a fixture
@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

class BenchListOperation(EasyBench):
    # Receive the fixture as an argument
    def bench_append(self, big_list):
        big_list.append(-1)

    def bench_insert_start(self, big_list):
        big_list.insert(0, -1)

if __name__ == "__main__":
    BenchListOperation().bench()
```

The `scope` parameter of the `fixture` decorator specifies the lifetime of the fixture:

- `"trial"`: Created for each trial (default)
- `"function"`: Created once per benchmark function
- `"class"`: Created once per benchmark class


### Configuration Options

The following settings are available in the `BenchConfig` class:

```python
from easybench import BenchConfig, EasyBench

class MyBenchmark(EasyBench):
    bench_config = BenchConfig(
        trials=5,            # Number of trials
        sort_by="avg",       # Sort criterion
        reverse=False,       # Sort order (False=ascending, True=descending)
        memory=True,         # Enable memory measurement (True or "B"/"KB"/"MB"/"GB")
        color=True,          # Use color output in results
        show_output=False,   # Display function return values
        reporters=[]         # Custom reporters (see explanation below)
    )
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

### Memory Measurement Limitations

!!! note
    EasyBench uses Python's built-in `tracemalloc` module to measure memory usage.  
    This has some important limitations:
   
    - `tracemalloc` only tracks memory allocations made through Python's memory manager
    - Memory allocated by C extensions (like NumPy, Pandas, or other native libraries) often bypasses Python's memory manager and won't be accurately measured
    - The reported memory usage reflects Python objects only, not the total process memory consumption
   
    For applications heavily using C extensions, consider using external profilers like `memory_profiler` or system monitoring tools for more accurate measurements.

---

## Command Line Interface (`easybench` command)

To run multiple benchmarks at once, use the `easybench` command:

```bash
easybench [options] [path]
```

* By default, it runs files named `bench_*.py` in the `benchmarks` directory
* You can specify either a directory containing benchmark files or a specific benchmark file
* Benchmark scripts must follow these rules:
  * For class-based benchmarks, class names should start with `Bench`
  * For function-based benchmarks, function names should start with `bench_`

### Command Options

```bash
easybench [--trials N] [--memory] [--sort-by METRIC] [--reverse] [--no-color] [--show-output] [path]
```

- `--trials N`: Number of trials (default: 5)
- `--memory`: Enable memory measurement
- `--sort-by METRIC`: Sort criterion (def/avg/min/max/avg_memory/max_memory)
- `--reverse`: Sort results in descending order
- `--no-color`: Disable colored output
- `--show-output`: Display function return values
- `path`: Directory containing benchmark files or a specific benchmark file (default: "benchmarks")

### Function-based Benchmark Example

Example of a function-based benchmark to be run from the command line:

```python
# Filename: benchmarks/bench_list_operations.py
from easybench import fixture

@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

def bench_append(big_list):
    """Append an element to the end of the list"""
    big_list.append(-1)

def bench_insert_start(big_list):
    """Insert an element at the beginning of the list"""
    big_list.insert(0, -1)
```

Save this file in the `benchmarks` folder and run the `easybench` command to benchmark both functions and compare the results:

```bash
easybench --trials 10 --memory
```

You can also run a specific benchmark file directly:

```bash
easybench --trials 10 --memory benchmarks/bench_list_operations.py
```
