## Decorator-based Benchmarks (`@bench` decorator)

### **Basic usage**

The `@bench` decorator provides the simplest way to benchmark a function:

```python
from easybench import bench

# Add @bench with function parameters
@bench(item=123, big_list=list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
```

### **Fresh data**

In the example above, `big_list` is created once and the same list is used for all trials.  
When you need fresh data for each trial, use a function or lambda to generate new data on demand:

```python
from easybench import bench

# Create a new list for each trial
@bench(item=-1, big_list=lambda: list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
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
def pop_first(big_list):
    big_list.pop(0)
```

You can also pass a `BenchConfig` object as a positional argument:

```python
from easybench import bench, BenchConfig

my_config = BenchConfig(trials=10, memory=True)

@bench(big_list=list(range(10_000_000)))
@bench.config(my_config)
def pop_first(big_list):
    big_list.pop(0)
```

!!! tip
    Place the `@bench.config` decorator before (below) other bench decorators.  


Main configuration options:

* `trials`: Number of trials (default: `5`)
* `memory`: Also measure memory usage (default: `False`)
    * `False`: Disable memory measurement
    * `True`: Show memory in kilobytes
    * `"B"`, `"KB"`, `"MB"`, `"GB"`: Show memory in bytes, kilobytes, megabytes, or gigabytes
* `time`: Specify time measurement unit (default: `"s"`)
    * `"s"`: Display time in seconds
    * `"ms"`: Display time in milliseconds 
    * `"μs"` or `"us"`: Display time in microseconds
    * `"ns"`: Display time in nanoseconds
    * `"m"`: Display time in minutes
    * `False`: Disable time measurement reporting
* For other options, see [Configuration Options](./class-based.md#configuration-options)

### **Deferred benchmarking** (`@bench.config(defer=...)`)

By default, benchmarks are executed immediately when the function is defined. However, you can defer the execution using the `defer` option:

```python
from easybench import bench

# Define benchmark but don't run yet
@bench(item=123, big_list=lambda: list(range(1_000_000)))
@bench.config(defer=True)
def insert_first(item, big_list):
    big_list.insert(0, item)
```

When `defer=True` is set, the benchmark will only be executed when you explicitly call the `.bench()` method as described in the on-demand benchmarking section.

#### **Grouped benchmarks**

You can also group multiple benchmarks together and run them as a batch:

```python
@bench.config(defer="examples")
def example1():
    return list(range(100_000))

@bench.config(defer="examples")
def example2():
    return list(range(1_000))

# Run all benchmarks in the "examples" group
_ = bench.run("examples")
```

This is useful when you want to organize related benchmarks and run them together with consistent configuration. You can also specify a custom configuration when running a group:

```python
# Run with custom configuration
bench.run("examples", config=BenchConfig(trials=10, memory=True))
```

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

#### **Combination of parameter sets** (`bench.grid`)

To benchmark a function with all combinations of different parameter sets, you can use `bench.grid`:

```python
from easybench import bench, BenchParams

# Define size parameter sets
small = BenchParams(name="Small", params={"size": 10})
large = BenchParams(name="Large", params={"size": 100})

# Define operation parameter sets
append = BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)})
pop = BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()})

# Create a Cartesian product of all parameter combinations
@bench.grid([[small, large], [append, pop]])
def operation(size, op):
    lst = list(range(size))
    op(lst)
```

This creates and benchmarks all combinations of parameters:

- Small × Append
- Small × Pop
- Large × Append
- Large × Pop

### **On-demand benchmarking**

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
