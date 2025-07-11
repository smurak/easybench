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

### Lifecycle Methods

In class-based benchmarks, you can use the following lifecycle methods:

```python
class BenchExample(EasyBench):
    def setup_class(self):
        # Run once before all benchmarks in the class
        ...
        
    def teardown_class(self):
        # Run once after all benchmarks in the class
        ...
        
    def setup_function(self):
        # Run before each benchmark function
        ...
        
    def teardown_function(self):
        # Run after each benchmark function
        ...
        
    def setup_trial(self):
        # Run before each trial
        ...
        
    def teardown_trial(self):
        # Run after each trial
        ...
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

#### Combination of parameter sets (`parametrize.grid`)

For more complex parametrized testing, `parametrize.grid` creates all combinations of multiple parameter lists:

```python
from easybench import BenchParams, EasyBench, parametrize

class BenchContainerOperations(EasyBench):
    # Define two sets of parameters
    sizes = [
        BenchParams(name="Small", params={"size": 100}),
        BenchParams(name="Large", params={"size": 10000}),
    ]
    
    operations = [
        BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)}),
        BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()}),
    ]
    
    # Create all combinations of all parameter combinations
    @parametrize.grid([sizes, operations])
    def bench_operation(self, size, op):
        lst = list(range(size))
        op(lst)

if __name__ == "__main__":
    BenchContainerOperations().bench()
```

This creates four parameter combinations:

```
Benchmark Results (5 trials):

Function                            Avg Time (s)  Min Time (s)  Max Time (s)
----------------------------------------------------------------------------
bench_operation (Small x Append)        0.000260      0.000004      0.001239
bench_operation (Small x Pop)           0.000005      0.000004      0.000008
bench_operation (Large x Append)        0.001642      0.000277      0.006804
bench_operation (Large x Pop)           0.000204      0.000183      0.000247
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


### Configuration Options

The following settings are available in the `BenchConfig` class:

```python
from easybench import BenchConfig, EasyBench, customize

class MyBenchmark(EasyBench):
    bench_config = BenchConfig(
        trials=5,               # Number of trials
        warmups=2,              # Number of warmup trials before actual measurement
        sort_by="avg",          # Sort criterion
        reverse=False,          # Sort order (False=ascending, True=descending)
        memory=True,            # Enable memory measurement (True or "B"/"KB"/"MB"/"GB")
        color=True,             # Use color output in results
        show_output=False,      # Display function return values
        loops_per_trial=1,      # Number of function executions per trial (see explanation below)
        reporters=["console"],  # Custom reporters (see "Advanced Usage")
        progress=True,          # Enable progress tracking with tqdm
        include=None,           # Regex pattern to include only matching benchmarks
        exclude=None,           # Regex pattern to exclude matching benchmarks
        clip_outliers=None,     # Clip maximum values based on the specified proportion (between 0 and 1)
    )

    # You can also customize settings for individual methods
    @customize(loops_per_trial=1000, name="Pass")
    def bench_fast_operation(self):
        # This method uses 1000 loops per trial
        # and will be displayed as "Pass" in results
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

- `False`: Disable time measurement reporting
- `True`: Enable time measurement reporting in seconds
- `"s"`: Display time in seconds (default)
- `"ms"`: Display time in milliseconds
- `"μs"` or `"us"`: Display time in microseconds
- `"ns"`: Display time in nanoseconds
- `"m"`: Display time in minutes

Progress tracking options (`progress`):

- `False`: Disable progress tracking (default)
- `True`: Enable progress tracking using tqdm
- Custom function: Use a custom progress tracking function that follows the tqdm interface

Benchmark selection options:

- `include`: Regular expression pattern to include only benchmark functions with matching names
- `exclude`: Regular expression pattern to exclude benchmark functions with matching names

  - For parametrized benchmarks, these options match against the full name (e.g., "bench_func (param_name)")
  - When both options are specified, `exclude` takes precedence

#### Improving Measurement Accuracy with `warmups`

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

#### Improving Timer Precision with `loops_per_trial`

In environments with poor timer resolution (e.g., certain virtual machines or systems where `time.perf_counter()` has limited precision), you may need to run a function multiple times to get meaningful timing results.

Additionally, when benchmarking very fast operations (less than a few microseconds), the overhead of timer calls can significantly impact measurement results.  
In such cases, using the `loops_per_trial` parameter can help distribute the timer call overhead and achieve more accurate measurements.

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

### Memory Measurement Limitations

!!! note
    EasyBench uses Python's built-in `tracemalloc` module to measure memory usage.  
    This has some important limitations:
   
    - `tracemalloc` only tracks memory allocations made through Python's memory manager
    - Memory allocated by C extensions (like NumPy, Pandas, or other native libraries) often bypasses Python's memory manager and won't be accurately measured
    - The reported memory usage reflects Python objects only, not the total process memory consumption
   
    For applications heavily using C extensions, consider using external profilers like `memory_profiler` or system monitoring tools for more accurate measurements.
