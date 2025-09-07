# Customizing Output

## `Formatter` and `Reporter`

In EasyBench, benchmark results are output using a mechanism called **reporter**. By default, `ConsoleReporter` is used.

`ConsoleReporter` is a reporter that outputs data to the console screen, and by default, it formats data in a tabular format (`TableFormatter`). In EasyBench, you can output in various formats by changing this `Formatter` (output format) and `Reporter` (output method).

### Using Reporters

To use a reporter, set it as a list in the `reporters` parameter of the benchmark configuration (`BenchConfig` or `@bench.config`). The `reporters` is a list format, allowing you to specify multiple output methods simultaneously.

There are three ways to specify a Reporter:

1. **Specify as a string**: Specify the reporter name as a string

    - `"console"`: Standard tabular console output
    - `"simple"`: Simple console output
    - `"boxplot"`: Visualization with boxplot
    - `"violinplot"`: Visualization with violinplot
    - `"lineplot"`: Visualization with lineplot
    - `"histplot"`: Visualization with histogram
    - `"barplot"`: Visualization with barplot
    - `"boxplot-sns"`: Visualization with seaborn-style boxplot
    - `"violinplot-sns"`: Visualization with seaborn-style violinplot
    - `"lineplot-sns"`: Visualization with seaborn-style lineplot
    - `"histplot-sns"`: Visualization with seaborn-style histplot
    - `"barplot-sns"`: Visualization with seaborn-style barplot
    - `"*.csv"` or `"*.json"`: File output

2. **Specify with arguments**: Specify in the format `(reporter_name, parameter_dict)`  
   (e.g., `("boxplot", {"log_scale": False})`)

3. **Specify with Reporter object**: Directly specify an instance of the Reporter class  
   (e.g., `FileReporter("results.json")`)


Example usage:

```python
from easybench import BenchConfig
from easybench.reporters import FileReporter

# Multiple output formats with various specification methods
config = BenchConfig(
    ...
    reporters=[
        "console",                          # Specified as a string
        ("simple", {"metric": "min"}),      # Specified with arguments
        ("boxplot", {"log_scale": False}),  # Plot with arguments
        "results.csv",                      # Specified as a file path
        FileReporter("results.json"),       # Specified as an object
    ]
)
```

### Creating Custom Reporters

For advanced use cases, you can create your own reporters.  
For details, refer to [Creating Custom Reporters and Formatters](custom-reporters.md).

### Registering Custom Reporters

When you register a custom reporter with a name, you can use it by specifying it as a string:

```python
from easybench import BenchConfig, set_reporter
from easybench.reporters import ConsoleReporter, SimpleFormatter
from easybench.visualization import PlotReporter, LinePlotFormatter

# [Method 1] Using function calls

# 1. Create a function that returns a reporter object
def create_log_plot(**kwargs):
    return PlotReporter(LinePlotFormatter(log_scale=True, **kwargs))

# 2. Register that function with a name
set_reporter("log-lineplot", create_log_plot)  # Register as "log-lineplot"


# [Method 2] Using decorator syntax

# 1. Apply decorator to a function that returns a reporter object
@set_reporter("custom-simple")  # Register as "custom-simple"
def create_simple_reporter(**kwargs):
    return ConsoleReporter(SimpleFormatter(**kwargs))


# Use the registered reporters
bench_config = BenchConfig(
    reporters=[
        "console",
        "log-lineplot",
        ("custom-simple", {"metric": "min"}),
    ]
)
```
