# EasyBench

A simple and easy-to-use Python benchmarking library.

## Features

- Three flexible benchmarking styles (decorator, class-based, and command-line)
- Measure both execution time and estimated memory usage ([see limitations](usage.md#memory-measurement-limitations))
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
