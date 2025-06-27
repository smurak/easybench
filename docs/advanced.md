# Advanced Usage

## Custom Output (`Formatter` and `Reporter`)

EasyBench uses a mechanism called **Reporter** to output benchmark results. By default, `ConsoleReporter` is used.

`ConsoleReporter` is a reporter that outputs data to the console screen, and by default formats the data in tabular form (`TableFormatter`). In EasyBench, you can change both the `Formatter` (output format) and the `Reporter` (output method) to enable various forms of output.

### Using Reporters

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
    - `"histplot"`: Visualization with histplot (histogram)
    - `"histplot-sns"`: Visualization with seaborn-styled histplot
    - `"barplot"`: Visualization with barplot
    - `"barplot-sns"`: Visualization with seaborn-styled barplot
    - `"*.csv"` or `"*.json"`: File output

2. **With arguments**: Specify in the format `(reporter_name, parameter_dict)`

3. **As Reporter objects**: Directly specify an instance of a Reporter class


Usage example:

```python
from easybench import BenchConfig
from easybench.reporters import FileReporter

# Multiple output formats with different specification methods
config = BenchConfig(
    ...
    reporters=[
        "console",                          # Specified as string
        ("simple", {"metric": "min"}),      # Specified with arguments
        ("boxplot", {"log_scale": False}),  # Plot with arguments
        "results.csv",                      # Specified as file path
        FileReporter("results.json"),       # Specified as object
    ]
)
```

### Creating Custom Reporters

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

## Boxplot Visualization (`BoxPlotFormatter`)

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
- `figsize`: Figure size (default: `(10, 6)`)
- `engine`: Plotting engine (`"matplotlib"` or `"seaborn"`)
- `orientation`: Boxplot orientation (`"vertical"` or `"horizontal"`)
- `sns_theme`: Dictionary of seaborn theme parameters passed to `sns.set_theme()` (e.g., `{"style": "darkgrid", "palette": "Set2", "context": "notebook"}`)

### `PlotReporter` options

- `formatter`: Plot formatter to use (e.g., `BoxPlotFormatter`)
- `show`: Whether to display the plot on screen (default: `True`)
- `save_path`: File path to save the plot
- `dpi`: Image resolution (default: `100`)

!!! note
    To use boxplots, you need to install `matplotlib`:
    ```bash
    pip install matplotlib
    ```
    
    If you want to use the seaborn engine, also install `seaborn`:
    ```bash
    pip install seaborn
    ```
