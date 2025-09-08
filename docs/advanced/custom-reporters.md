# Creating Custom Reporters and Formatters

In EasyBench, benchmark result output is handled by two components:

1. **Formatter**: Converts benchmark results into a specific format (text table, CSV, JSON, etc.)
2. **Reporter**: Sends formatted data to a specific destination (console, file, API, etc.)

These two types of components allow you to customize "what to output" and "where to output it" separately.

## Formatter

A Formatter is a component that converts benchmark results into a specific format.
For example, it can output in table format, CSV, JSON, DataFrame, etc.

### List of Built-in Formatters

EasyBench includes the following standard formatters:

- `TableFormatter`: Text table format (default)
- `SimpleFormatter`: Concise text output
- `CSVFormatter`: CSV format
- `JSONFormatter`: JSON format
- `DataFrameFormatter`: pandas DataFrame format
- Various plot formatters: `BoxPlotFormatter`, `LinePlotFormatter`, etc.

### How to Implement a Formatter

To create a custom formatter, inherit from the `Formatter` class and implement the `format` method.

```python
from easybench.reporters import Formatted, Formatter
from easybench.core import BenchConfig
from easybench.utils import ResultsType, StatsType

class CustomFormatter(Formatter):
    """Custom formatter."""
    
    def format(
        self,
        results: ResultsType,  # Benchmark result data
        stats: StatsType,      # Calculated statistics
        config: BenchConfig,   # Benchmark configuration
    ) -> Formatted:            # Formatted output
        """Format results in a custom format."""
        # Implement your custom formatting logic here
        formatted_output = "Your custom formatting here"
        return formatted_output
```

#### Arguments of the `format` Method

##### 1. `results` (ResultsType)

The value of the `results` argument is a dictionary containing raw data for each benchmark function that was executed:

```python
{
    "bench_function_a": {
        "times": [0.001, 0.0012, 0.0011],       # Execution time for each run (seconds)
        "memory": [1024, 1028, 1022],           # Memory usage for each run (bytes)
        "output": ["result1", "result1", "result1"]  # Return values for each run
    },
    "bench_function_b": {
        "times": [0.002, 0.0019, 0.0021],
        "memory": [2048, 2050, 2045],
        "output": [42, 42, 42]
    }
}
```

!!! warning "Note"
    Each key (`times`, `memory`, `output`) may not exist depending on the benchmark configuration. For example, benchmark results with memory tracking disabled will not have the `memory` key.

##### 2. `stats` (StatsType)

The value of the `stats` argument is a dictionary of automatically calculated statistics from benchmark results:

```python
{
    "bench_function_a": {
        "avg": 0.0011,           # Average execution time (seconds)
        "min": 0.001,            # Minimum execution time
        "max": 0.0012,           # Maximum execution time
        "avg_memory": 1024.67,   # Average memory usage (bytes)
        "max_memory": 1028       # Maximum memory usage
    },
    "bench_function_b": {
        "avg": 0.002,
        "min": 0.0019,
        "max": 0.0021,
        "avg_memory": 2047.67,
        "max_memory": 2050
    }
}
```

!!! info "Hint"
    Statistics are pre-calculated and passed to avoid redundant calculation across multiple formatters.  
    If you want to perform your own statistical calculations, use the raw data from the `results` argument.

##### 3. `config` (BenchConfig)

The value of the `config` argument is a `BenchConfig` object containing benchmark settings:

```python
BenchConfig(
    trials=10,              # Number of executions
    warmups=2,              # Number of warmup runs
    time="ms",              # Time measurement option (True/False/"m"/"s"/"ms"/"us"/"ns")
    memory="KB",            # Memory measurement option (True/False/"B"/"KB"/"MB"/"GB")
    sort_by="avg",          # Result sorting criteria
    reverse=False,          # Whether to reverse sort order
    show_output=False,      # Whether to display output values
    color=True              # Whether to use colored output
)
```

#### Return Value of the `format` Method (`Formatted`)

The `format` method returns a value of the following type:

- `str`: Text format output (table, CSV, JSON, etc.)
- `pd.DataFrame`: pandas DataFrame format
- `matplotlib.figure.Figure`: Graph format

### Implementation Example: XMLFormatter

Here is an example implementation of a Formatter that outputs in XML format:

```python
from easybench.reporters import Formatter, TimeUnit, MemoryUnit
from easybench.core import BenchConfig
from easybench.utils import ResultsType, StatsType

class XMLFormatter(Formatter):
    """Format benchmark results in XML format"""
    
    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """Convert to XML format"""
        time_unit = TimeUnit.from_config(config)
        memory_unit = MemoryUnit.from_config(config)
        
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<benchmark trials="{config.trials}">')
        
        # Get sorted list of function names
        methods = self.sort_keys(stats, config)
        
        for method_name in methods:
            stat = stats[method_name]
            lines.append(f'  <function name="{method_name}">')
            
            # Time measurement results
            if config.time:
                avg_time = time_unit.convert_seconds(stat["avg"])
                min_time = time_unit.convert_seconds(stat["min"])
                max_time = time_unit.convert_seconds(stat["max"])
                
                lines.append(f'    <time unit="{time_unit}">')
                lines.append(f'      <average>{avg_time:.6f}</average>')
                lines.append(f'      <minimum>{min_time:.6f}</minimum>')
                lines.append(f'      <maximum>{max_time:.6f}</maximum>')
                lines.append('    </time>')
            
            # Memory measurement results
            if config.memory:
                avg_mem = memory_unit.convert_bytes(stat["avg_memory"])
                max_mem = memory_unit.convert_bytes(stat["max_memory"])
                
                lines.append(f'    <memory unit="{memory_unit}">')
                lines.append(f'      <average>{avg_mem:.2f}</average>')
                lines.append(f'      <maximum>{max_mem:.2f}</maximum>')
                lines.append('    </memory>')
            
            lines.append('  </function>')
        
        lines.append('</benchmark>')
        return '\n'.join(lines)
```

## Reporter

A Reporter is a component that sends data transformed by a specified Formatter to a specific destination (console, file, API, etc.).

### List of Built-in Reporters

EasyBench includes the following standard reporters:

- `ConsoleReporter`: Console output (default)
- `FileReporter`: File output
- `CallbackReporter`: Output to a callback function
- `SimpleConsoleReporter`: Concise console output
- `PlotReporter`: Graph output

### How to Implement a Reporter

To create a custom reporter, inherit from the `Reporter` class and implement the `report_formatted` method.

```python
from easybench.reporters import Reporter, Formatted

class MyCustomReporter(Reporter):
    """Custom reporter."""
    
    def report_formatted(self, formatted_output: Formatted) -> None:
        """Report the formatted output."""
        ...
```

Also, set a `Formatter` object to the `formatter` attribute.  
By default, it is designed to input as the first argument during initialization, as follows:

```python
class Reporter:
    def __init__(self, formatter: Formatter) -> None:
        self.formatter = formatter
```

#### `formatted_output` Argument (Formatted)

`formatted_output` is the same format as the return value of the `format` method of `Formatter`:

1. **String** (`str`): Text format output
2. **DataFrame** (`pd.DataFrame`): Tabular data
3. **Figure** (`matplotlib.figure.Figure`): Graph image

### Implementation Example: SlackReporter

Here is an example implementation of a Reporter that sends benchmark results to Slack:

```python
from easybench.reporters import Reporter, TableFormatter

class SlackReporter(Reporter):
    """Reporter that sends benchmark results to Slack"""
    
    def __init__(self, webhook_url, channel="#benchmarks", formatter=None):
        # Use TableFormatter by default
        super().__init__(formatter or TableFormatter())
        self.webhook_url = webhook_url
        self.channel = channel
    
    def report_formatted(self, formatted_output: str) -> None:
        """
        Send formatted output to Slack.
        
        Args:
            formatted_output: Data output from Formatter (supports string only)
        """
        import requests
        
        payload = {
            "channel": self.channel,
            "text": "Benchmark Results",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "```\n" + formatted_output + "\n```"}
                }
            ]
        }
        requests.post(self.webhook_url, json=payload)
```

## Integration of Formatter and Reporter

Custom-created `Formatter` and `Reporter` can be used together as follows:

```python
from easybench import BenchConfig

# Benchmark configuration using custom Formatter and Reporter
config = BenchConfig(
    trials=50,
    reporters=[
        # Use custom reporter
        SlackReporter(
            webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
            formatter=XMLFormatter()  # Specify formatter instance
        ),
        # Also use standard console output
        "console"
    ]
)
```

For information on how to register custom reporters, refer to [Customizing Output](customize-output.md).
