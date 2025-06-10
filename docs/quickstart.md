# Quick Start

There are 3 ways to benchmark with `easybench`:

1. `@bench` decorator

    ```python
    from easybench import bench
    
    # Add @bench with function parameters
    @bench(item=123, big_list=lambda: list(range(1_000_000)))
    def add_item(item, big_list):
        big_list.append(item)
    ```

    !!! tip
    
        When you need fresh data for each trial, use a function or lambda to generate new data on demand.  
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
        def bench_append(self):
            self.big_list.append(123)
    
        # You can define multiple benchmark methods
        def bench_pop(self):
            self.big_list.pop()
    
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
        def bench_append(big_list):
            big_list.append(123)
        
        # You can define multiple benchmark functions
        def bench_pop(big_list):
            big_list.pop()
        ```

    3. Run `easybench` command

        ```bash
        easybench --trials 10 --memory --sort-by avg
        ```

**Example of benchmark results:**

* Single benchmark

    ```
    Benchmark Results (5 trials):
    
    Function   Avg Time (s) Min Time (s) Max Time (s)
    ----------------------------------------------
    add_item   0.002393     0.000939     0.007362   
    ```

* Multiple benchmarks

    ![EasyBench Benchmark Result](https://raw.githubusercontent.com/smurak/easybench/main/images/easybench_screenshot.png)

* Boxplot Visualization

    ![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)