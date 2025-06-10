# ruff: noqa: FBT001, FBT002, PLR0913
"""
Visualization tools for benchmark results.

This module provides formatters and reporters for creating visual representations
of benchmark results using matplotlib.
"""

from abc import abstractmethod
from typing import Literal

try:
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
except ImportError as err:
    error_msg = (
        "matplotlib is required for visualization. "
        "Install with pip install matplotlib."
    )
    raise ImportError(error_msg) from err

from .core import BenchConfig, ResultsType, StatsType
from .reporters import Formatted, Formatter, MemoryUnit, Reporter

# Constants for magic values
MAX_PERCENTILE_THRESHOLD = 0.5
MIN_PERCENTILE_THRESHOLD = 0.0


class PlotFormatter(Formatter):
    """Base class for formatters that produce matplotlib figures."""

    @abstractmethod
    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a matplotlib visualization.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the visualization

        """


class BoxplotFormatter(PlotFormatter):
    """Format benchmark results as a matplotlib boxplot."""

    def __init__(
        self,
        showfliers: bool = True,
        log_scale: bool = False,
        data_limit: tuple[float, float] | None = None,
        trim_outliers: float | None = None,
        winsorize_outliers: float | None = None,
        figsize: tuple[int, int] = (10, 6),
        label_rotation_threshold: int = 4,
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        orientation: Literal["vertical", "horizontal"] = "horizontal",
        **boxplot_kwargs: object,
    ) -> None:
        """
        Initialize BoxplotFormatter.

        Args:
            showfliers: Whether to show outliers in the boxplot (default: True)
            log_scale: Whether to use logarithmic scale for y-axis (default: False)
            data_limit: Optional tuple of (min, max) for data axis limits
            trim_outliers: Optional percentile for trimming outliers (0.0-0.5)
            winsorize_outliers: Optional percentile for winsorizing outliers (0.0-0.5)
            figsize: Figure size (default: (10, 6))
            label_rotation_threshold: Rotate x-axis labels if method count exceeds this.
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of boxplot ('vertical' or 'horizontal')
            **boxplot_kwargs: Additional keyword arguments for boxplot function

        """
        self.log_scale = log_scale
        self.data_limit = data_limit
        self.trim_outliers = trim_outliers
        self.winsorize_outliers = winsorize_outliers
        self.figsize = figsize
        self.boxplot_kwargs = boxplot_kwargs
        self.label_rotation_threshold = label_rotation_threshold
        self.showfliers = showfliers
        self.engine = engine
        self.orientation = orientation

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a matplotlib boxplot.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the boxplot

        """
        # Extract and preprocess data
        time_data, labels = self._preprocess_data(results, stats, config)

        # Create figure and axes based on memory tracking
        if config.memory:
            # Create a figure with two subplots when memory tracking is enabled
            fig, (ax_time, ax_mem) = plt.subplots(
                2,
                1,
                figsize=(self.figsize[0], self.figsize[1] * 1.8),
            )

            # Process time data
            box_data_time = [time_data[method] for method in labels]
            if self.engine == "seaborn":
                self._create_seaborn_boxplot(ax_time, box_data_time, labels)
            else:
                self._create_matplotlib_boxplot(ax_time, box_data_time, labels)

            # Apply styling to time plot
            self._apply_styling(ax_time, config, labels, title_suffix="")

            # Process memory data using dedicated method
            self._process_memory_subplot(ax_mem, results, config, labels)
        else:
            # Original behavior for timing-only plots
            fig, ax = plt.subplots(figsize=self.figsize)
            box_data = [time_data[method] for method in labels]
            if self.engine == "seaborn":
                self._create_seaborn_boxplot(ax, box_data, labels)
            else:
                self._create_matplotlib_boxplot(ax, box_data, labels)
            self._apply_styling(ax, config, labels)

        plt.tight_layout()
        return fig

    def _process_memory_subplot(
        self,
        ax: plt.Axes,
        results: ResultsType,
        config: BenchConfig,
        labels: list[str],
    ) -> None:
        """
        Process and create memory usage subplot.

        Args:
            ax: The matplotlib axes for the memory subplot
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration
            labels: List of method names to include

        """
        # Extract memory data
        memory_data = self._preprocess_memory_data(results, config, labels)
        box_data_mem = [memory_data[method] for method in labels]

        # Create memory plot
        if self.engine == "seaborn":
            self._create_seaborn_boxplot(ax, box_data_mem, labels)
        else:
            self._create_matplotlib_boxplot(ax, box_data_mem, labels)

        # Apply styling to memory plot with proper unit
        memory_unit = MemoryUnit.from_config(config)
        self._apply_styling(
            ax,
            config,
            labels,
            title_suffix="Memory Usage",
            unit=str(memory_unit),
        )

    def _preprocess_data(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> tuple[dict[str, list[float]], list[str]]:
        """Extract and preprocess benchmark data."""
        data: dict[str, list[float]] = {}
        labels = []

        # Sort method names according to configuration
        sorted_methods = self.sort_keys(stats, config)

        # Extract time data without processing if no outlier handling is needed
        if self.trim_outliers is None and self.winsorize_outliers is None:
            # No outlier handling needed, extract raw data and return early
            for method_name in sorted_methods:
                if "times" in results[method_name]:
                    data[method_name] = results[method_name]["times"]
                    labels.append(method_name)
            return data, labels

        # If we reach here, we need numpy for outlier handling
        try:
            import numpy as np
        except ImportError as err:
            error_msg = (
                "numpy is required for outlier handling in BoxplotFormatter. "
                "Install with pip install numpy."
            )
            raise ImportError(error_msg) from err

        # Extract time data for each method with outlier handling
        for method_name in sorted_methods:
            if "times" in results[method_name]:
                times = results[method_name]["times"]

                # Apply outlier handling if requested
                if (
                    self.trim_outliers is not None
                    and MIN_PERCENTILE_THRESHOLD
                    < self.trim_outliers
                    < MAX_PERCENTILE_THRESHOLD
                ):
                    # Trim outliers by percentile
                    lower = float(np.percentile(times, self.trim_outliers * 100))
                    upper = float(np.percentile(times, 100 - self.trim_outliers * 100))
                    times = [t for t in times if lower <= t <= upper]

                if (
                    self.winsorize_outliers is not None
                    and MIN_PERCENTILE_THRESHOLD
                    < self.winsorize_outliers
                    < MAX_PERCENTILE_THRESHOLD
                ):
                    # Winsorize outliers (clip to percentile values)
                    lower = float(np.percentile(times, self.winsorize_outliers * 100))
                    upper = float(
                        np.percentile(times, 100 - self.winsorize_outliers * 100),
                    )
                    times = [min(max(float(t), lower), upper) for t in times]

                data[method_name] = times
                labels.append(method_name)

        return data, labels

    def _preprocess_memory_data(
        self,
        results: ResultsType,
        config: BenchConfig,
        labels: list[str],
    ) -> dict[str, list[float]]:
        """Extract and preprocess memory usage data."""
        memory_data: dict[str, list[float]] = {}
        memory_unit = MemoryUnit.from_config(config)

        # Process each method's memory data
        for method_name in labels:
            if "memory" in results[method_name]:
                # Convert memory values to the specified unit
                memory_values = [
                    memory_unit.convert_bytes(value)
                    for value in results[method_name]["memory"]
                ]

                # Apply outlier handling if needed
                if (
                    self.trim_outliers is not None
                    and MIN_PERCENTILE_THRESHOLD
                    < self.trim_outliers
                    < MAX_PERCENTILE_THRESHOLD
                ):
                    try:
                        import numpy as np

                        lower = float(
                            np.percentile(memory_values, self.trim_outliers * 100),
                        )
                        upper = float(
                            np.percentile(
                                memory_values,
                                100 - self.trim_outliers * 100,
                            ),
                        )
                        memory_values = [
                            m for m in memory_values if lower <= m <= upper
                        ]
                    except ImportError:
                        pass  # Fallback to using raw values if numpy is not available

                if (
                    self.winsorize_outliers is not None
                    and MIN_PERCENTILE_THRESHOLD
                    < self.winsorize_outliers
                    < MAX_PERCENTILE_THRESHOLD
                ):
                    try:
                        import numpy as np

                        lower = float(
                            np.percentile(memory_values, self.winsorize_outliers * 100),
                        )
                        upper = float(
                            np.percentile(
                                memory_values,
                                100 - self.winsorize_outliers * 100,
                            ),
                        )
                        memory_values = [
                            min(max(float(m), lower), upper) for m in memory_values
                        ]
                    except ImportError:
                        pass  # Fallback to using raw values if numpy is not available

                memory_data[method_name] = memory_values
            else:
                # Use placeholder if no memory data available
                memory_data[method_name] = [0.0]

        return memory_data

    def _create_matplotlib_boxplot(
        self,
        ax: plt.Axes,
        box_data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a boxplot using matplotlib."""
        if self.orientation == "horizontal":
            try:
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    orientation=self.orientation,
                    **self.boxplot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back to vert parameter if orientation is not supported
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    vert=(self.orientation == "vertical"),
                    **self.boxplot_kwargs,  # type: ignore[arg-type]
                )
            # Set y-tick positions explicitly before setting labels
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels)
        else:
            try:
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    orientation=self.orientation,
                    **self.boxplot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back to vert parameter if orientation is not supported
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    vert=(self.orientation == "vertical"),
                    **self.boxplot_kwargs,  # type: ignore[arg-type]
                )
            # Set x-tick positions explicitly before setting labels
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)

    def _create_seaborn_boxplot(
        self,
        ax: plt.Axes,
        box_data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a boxplot using seaborn."""
        try:
            import seaborn as sns
        except ImportError as err:
            error_msg = (
                "seaborn is required for seaborn engine. "
                "Install with pip install seaborn."
            )
            raise ImportError(error_msg) from err

        # Try to use seaborn directly with the data
        sns.boxplot(
            data=box_data,
            ax=ax,
            showfliers=self.showfliers,
            orient=("h" if self.orientation == "horizontal" else "v"),
            **self.boxplot_kwargs,  # type: ignore[arg-type]
        )

        if self.orientation == "horizontal":
            # Set y-tick positions explicitly before setting labels
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
        else:
            # Set x-tick positions explicitly before setting labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)

    def _apply_styling(
        self,
        ax: plt.Axes,
        config: BenchConfig,
        labels: list[str],
        title_suffix: str = "",
        unit: str = "seconds",
    ) -> None:
        """Apply common styling to the plot."""
        # Set the scale (log or linear)
        self._set_axis_scale(ax)

        # Set axis limits if provided
        self._set_axis_limits(ax)

        # Set the plot title
        title = self._get_plot_title(config.trials, title_suffix)
        ax.set_title(title)

        # Set appropriate axis labels
        self._set_axis_labels(ax, unit)

        # Apply label rotation if needed
        self._apply_label_rotation(ax, labels)

    def _set_axis_scale(self, ax: plt.Axes) -> None:
        """Set the axis scale (log or linear) based on orientation."""
        if self.log_scale:
            if self.orientation == "horizontal":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")

    def _set_axis_limits(self, ax: plt.Axes) -> None:
        """Set axis limits if provided."""
        if self.data_limit is not None:
            if self.orientation == "horizontal":
                ax.set_xlim(self.data_limit)
            else:
                ax.set_ylim(self.data_limit)

    def _get_plot_title(self, trials: int, title_suffix: str = "") -> str:
        """Generate the plot title."""
        base_title = f"Benchmark Results ({trials} trials)"
        if title_suffix:
            return f"{title_suffix} {base_title}"
        return base_title

    def _set_axis_labels(self, ax: plt.Axes, unit: str) -> None:
        """Set appropriate axis labels based on orientation and measurement type."""
        label_text = "Time (seconds)" if unit == "seconds" else f"Memory ({unit})"

        if self.orientation == "horizontal":
            ax.set_xlabel(label_text)
        else:
            ax.set_ylabel(label_text)

    def _apply_label_rotation(self, ax: plt.Axes, labels: list[str]) -> None:
        """Apply rotation to axis labels if needed."""
        if len(labels) > self.label_rotation_threshold:
            if self.orientation == "horizontal":
                pass  # Currently no rotation for horizontal orientation
            else:
                # For vertical plots, rotate x-tick labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


class PlotReporter(Reporter):
    """Reporter that displays benchmark results as a matplotlib visualization."""

    def __init__(
        self,
        formatter: PlotFormatter,
        show: bool = True,
        save_path: str | None = None,
        dpi: int = 100,
    ) -> None:
        """
        Initialize PlotReporter for matplotlib visualizations.

        Args:
            formatter: A PlotFormatter that returns a matplotlib Figure object
            show: Whether to display the plot (default: True)
            save_path: Optional file path to save the plot
            dpi: DPI for saving the image (default: 100)

        """
        super().__init__(formatter)
        self.show = show
        self.save_path = save_path
        self.dpi = dpi

    def _send(self, formatted_output: Formatted) -> None:
        """
        Display or save the formatted output (matplotlib Figure).

        Args:
            formatted_output: Matplotlib Figure object

        """
        if not isinstance(formatted_output, Figure):
            msg = "PlotReporter requires a matplotlib Figure object"
            raise TypeError(msg)

        # Save the figure if a path is provided
        if self.save_path:
            formatted_output.savefig(self.save_path, dpi=self.dpi)

        # Show the figure if requested
        if self.show:
            plt.show()
        else:
            plt.close(formatted_output)
