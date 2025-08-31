## Jupyter Notebook Magic Command (`%%easybench`)

### Basic Usage

The `%%easybench` magic command lets you benchmark code within a Jupyter Notebook cell:

```python
%%easybench --trials=3 --memory
# Write the code you want to benchmark below this line
result = []
for i in range(1_000_000):
    result.append(i)
```

### Setup

To use the magic command, you first need to load the extension:

```python
%load_ext easybench
```

### Options

The `%%easybench` magic command supports the following options:

- `--trials=N`: Number of trials to run (default: 1)
- `--memory`: Enable memory measurement
- `--memory-unit=UNIT`: Memory unit (B/KB/MB/GB)
- `--warmups=N`: Number of warmup runs (default: 0)
- `--loops-per-trial=N`: Number of loops per trial (default: 1)
- `--clip-outliers=FLOAT`: Outlier clipping rate (0.0 to 1.0)
- `--time-unit=UNIT`: Time unit (s/ms/us/ns/m)
- `--no-time`: Disable time measurement
- `--reporters REPORTER [REPORTER ...]`: Reporters to use (can specify multiple). Examples: console, simple, boxplot, violinplot, lineplot, histplot, barplot, results.csv, results.json

### Detailed Example

You can combine multiple options:

```python
%%easybench --trials=10 --memory --memory-unit=MB --warmups=2 --time-unit=ms --reporters lineplot console
# Create a list
data = [i for i in range(100_000)]

# Perform operations on the data
sorted_data = sorted(data, reverse=True)
```
