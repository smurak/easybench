# EasyBench

A simple and easy-to-use Python benchmarking library.

## Features

- Three benchmarking styles: decorator, class-based, and command-line
- Measure both execution time and memory usage ([see limitations](usage/class-based.md#memory-measurement-limitations))
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
