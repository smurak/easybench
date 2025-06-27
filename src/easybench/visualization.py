# ruff: noqa: FBT001, FBT002, PLR0913
"""
Visualization tools for benchmark results.

This module provides formatters and reporters for creating visual representations
of benchmark results using matplotlib.
"""

import contextlib
from abc import abstractmethod
from collections.abc import Generator
from typing import Any, Literal

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
from .reporters import Formatted, Formatter, MemoryUnit, MetricType, Reporter, TimeUnit

# Constants for magic values
MAX_PERCENTILE_THRESHOLD = 0.5
MIN_PERCENTILE_THRESHOLD = 0.0

# Default seaborn theme settings
DEFAULT_SNS_THEME = {"style": "darkgrid", "palette": "Set2"}


@contextlib.contextmanager
def set_seaborn_theme(theme_params: dict[str, Any] | None = None) -> Generator:
    """
    Context manager for temporarily setting a seaborn theme.

    Args:
        theme_params: Dictionary of parameters to pass to sns.set_theme()
                    If None, uses default theme

    """
    try:
        import seaborn as sns  # noqa: PLC0415

        # Use provided theme or default
        theme_to_use = theme_params if theme_params is not None else DEFAULT_SNS_THEME

        # Store current theme settings (if possible)
        with sns.plotting_context():
            # Apply the theme
            sns.set_theme(**theme_to_use)
            yield
    except ImportError:
        # If seaborn is not installed, just yield without doing anything
        yield


def _handle_seaborn_engine_and_theme(
    engine: str,
    sns_theme: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """
    Handle seaborn engine and theme settings.

    Args:
        engine: The plotting engine to use
        sns_theme: Optional dictionary of parameters for sns.set_theme()

    Returns:
        tuple: (engine, theme_params) where engine is the possibly updated engine
               and theme_params are the theme parameters to use

    """
    # If sns_theme is provided, ensure engine is set to "seaborn"
    if sns_theme is not None:
        return "seaborn", sns_theme

    # If engine is "seaborn" but no theme provided, use default theme
    if engine == "seaborn":
        return engine, DEFAULT_SNS_THEME

    # Otherwise, return original engine and None for theme
    return engine, None


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


class DistributionPlotFormatter(PlotFormatter):
    """Format benchmark results as a matplotlib distribution plot (box or violin)."""

    def __init__(
        self,
        showfliers: bool = True,
        log_scale: bool = False,
        figsize: tuple[int, int] = (10, 6),
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        orientation: Literal["vertical", "horizontal"] = "horizontal",
        plot_type: Literal["box", "violin"] = "box",
        sns_theme: dict[str, Any] | None = None,
        **plot_kwargs: object,
    ) -> None:
        """
        Initialize DistributionPlotFormatter.

        Args:
            showfliers: Whether to show outliers (only for boxplot) (default: True)
            log_scale: Whether to use logarithmic scale for y-axis (default: False)
            figsize: Figure size (default: (10, 6))
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of plot ('vertical' or 'horizontal')
            plot_type: Type of plot ('box' for boxplot or 'violin' for violinplot)
            sns_theme: Optional dictionary of seaborn theme parameters
            **plot_kwargs: Additional keyword arguments for plot function

        """
        self.log_scale = log_scale
        self.figsize = figsize
        self.plot_kwargs = plot_kwargs
        self.showfliers = showfliers
        self.orientation = orientation
        self.plot_type = plot_type

        # Handle seaborn engine and theme relationship
        self.engine, self.sns_theme = _handle_seaborn_engine_and_theme(
            engine,
            sns_theme,
        )

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a distribution plot.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the distribution plot

        """
        # Use the temporary seaborn theme context if needed
        with (
            set_seaborn_theme(self.sns_theme)
            if self.engine == "seaborn"
            else contextlib.nullcontext()
        ):
            # Extract and preprocess data
            time_data, labels = self._preprocess_data(results, stats, config)
            time_unit = TimeUnit.from_config(config)

            # Create figure and axes based on memory tracking
            if config.time and config.memory:
                # Create a figure with two subplots when memory tracking is enabled
                fig, (ax_time, ax_mem) = plt.subplots(
                    2,
                    1,
                    figsize=(self.figsize[0], self.figsize[1] * 1.8),
                )

                # Process time data
                box_data_time = [time_data[method] for method in labels]
                if self.engine == "seaborn":
                    self._create_seaborn_plot(ax_time, box_data_time, labels)
                else:
                    self._create_matplotlib_plot(ax_time, box_data_time, labels)

                # Apply styling to time plot with time unit
                self._apply_styling(
                    ax_time,
                    config,
                    title_suffix="",
                    unit=str(time_unit),
                )

                # Process memory data using dedicated method
                self._process_memory_subplot(ax_mem, results, config, labels)
            else:
                # Original behavior for timing-only plots
                fig, ax = plt.subplots(figsize=self.figsize)

                if config.time:
                    box_data = [time_data[method] for method in labels]
                    if self.engine == "seaborn":
                        self._create_seaborn_plot(ax, box_data, labels)
                    else:
                        self._create_matplotlib_plot(ax, box_data, labels)

                    # Apply styling with time unit
                    self._apply_styling(
                        ax,
                        config,
                        unit=str(time_unit),
                    )

                if config.memory:
                    self._process_memory_subplot(ax, results, config, labels)

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
            config: Benchmark configuration
            labels: List of method names to include

        """
        # Extract memory data
        memory_data = self._preprocess_memory_data(results, config, labels)
        box_data_mem = [memory_data[method] for method in labels]

        # Create memory plot
        if self.engine == "seaborn":
            self._create_seaborn_plot(ax, box_data_mem, labels)
        else:
            self._create_matplotlib_plot(ax, box_data_mem, labels)

        # Apply styling to memory plot with proper unit
        memory_unit = MemoryUnit.from_config(config)
        self._apply_styling(
            ax,
            config,
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
        time_unit = TimeUnit.from_config(config)

        # Sort method names according to configuration
        sorted_methods = self.sort_keys(stats, config)

        if not config.time:
            labels = [name for name in sorted_methods if "memory" in results[name]]
            return data, labels

        # Extract time data
        for method_name in sorted_methods:
            if "times" in results[method_name]:
                # Convert time values to the specified unit
                data[method_name] = [
                    time_unit.convert_seconds(t) for t in results[method_name]["times"]
                ]
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
                memory_data[method_name] = memory_values
            else:
                # Use placeholder if no memory data available
                memory_data[method_name] = [0.0]

        return memory_data

    def _create_matplotlib_plot(
        self,
        ax: plt.Axes,
        data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a plot using matplotlib."""
        if self.plot_type == "box":
            self._create_matplotlib_boxplot(ax, data, labels)
        elif self.plot_type == "violin":
            self._create_matplotlib_violinplot(ax, data, labels)
        else:
            msg = f"Unsupported plot type: {self.plot_type}"
            raise ValueError(msg)

    def _create_seaborn_plot(
        self,
        ax: plt.Axes,
        data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a plot using seaborn."""
        if self.plot_type == "box":
            self._create_seaborn_boxplot(ax, data, labels)
        elif self.plot_type == "violin":
            self._create_seaborn_violinplot(ax, data, labels)
        else:
            msg = f"Unsupported plot type: {self.plot_type}"
            raise ValueError(msg)

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
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back to vert parameter if orientation is not supported
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    vert=(self.orientation == "vertical"),
                    **self.plot_kwargs,  # type: ignore[arg-type]
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
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back to vert parameter if orientation is not supported
                ax.boxplot(
                    box_data,
                    showfliers=self.showfliers,
                    vert=(self.orientation == "vertical"),
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )
            # Set x-tick positions explicitly before setting labels
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)

    def _create_matplotlib_violinplot(
        self,
        ax: plt.Axes,
        violin_data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a violin plot using matplotlib."""
        # For violin plot in matplotlib, we need to adjust the positions
        positions = range(1, len(violin_data) + 1)

        if self.orientation == "horizontal":
            try:
                ax.violinplot(
                    violin_data,
                    positions=positions,
                    orientation=self.orientation,
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back if some parameters aren't supported
                ax.violinplot(
                    violin_data,
                    positions=positions,
                    vert=False,
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )

            # Set y-tick positions explicitly before setting labels
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels)
        else:
            try:
                ax.violinplot(
                    violin_data,
                    positions=positions,
                    orientation=self.orientation,
                    **self.plot_kwargs,  # type: ignore[arg-type]
                )
            except TypeError:
                # Fall back if some parameters aren't supported
                ax.violinplot(
                    violin_data,
                    positions=positions,
                    vert=True,
                    **self.plot_kwargs,  # type: ignore[arg-type]
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
            import seaborn as sns  # noqa: PLC0415
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
            **self.plot_kwargs,  # type: ignore[arg-type]
        )

        if self.orientation == "horizontal":
            # Set y-tick positions explicitly before setting labels
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
        else:
            # Set x-tick positions explicitly before setting labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)

    def _create_seaborn_violinplot(
        self,
        ax: plt.Axes,
        violin_data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a violin plot using seaborn."""
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError as err:
            error_msg = (
                "seaborn is required for seaborn engine. "
                "Install with pip install seaborn."
            )
            raise ImportError(error_msg) from err

        # Try to use seaborn directly with the data
        sns.violinplot(
            data=violin_data,
            ax=ax,
            orient=("h" if self.orientation == "horizontal" else "v"),
            **self.plot_kwargs,  # type: ignore[arg-type]
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
        title_suffix: str = "",
        unit: str = "s",
    ) -> None:
        """Apply common styling to the plot."""
        # Set the scale (log or linear)
        self._set_axis_scale(ax)

        # Set the plot title
        title = self._get_plot_title(config.trials, title_suffix)
        ax.set_title(title)

        # Set appropriate axis labels
        self._set_axis_labels(ax, unit)

        # Apply label rotation if needed
        if self.orientation != "horizontal":
            # Use autofmt_xdate for automatic rotation of x labels
            ax.figure.autofmt_xdate()

    def _set_axis_scale(self, ax: plt.Axes) -> None:
        """Set the axis scale (log or linear) based on orientation."""
        if self.log_scale:
            if self.orientation == "horizontal":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")

    def _get_plot_title(self, trials: int, title_suffix: str = "") -> str:
        """Generate the plot title."""
        base_title = f"Benchmark Results ({trials} trials)"
        if title_suffix:
            return f"{title_suffix} {base_title}"
        return base_title

    def _set_axis_labels(self, ax: plt.Axes, unit: str) -> None:
        """Set appropriate axis labels based on orientation and measurement type."""
        if any(unit == item.value for item in TimeUnit):
            label_text = f"Time ({unit})"
        else:
            label_text = f"Memory ({unit})"

        if self.orientation == "horizontal":
            ax.set_xlabel(label_text)
        else:
            ax.set_ylabel(label_text)


class BoxPlotFormatter(DistributionPlotFormatter):
    """Format benchmark results as a matplotlib boxplot."""

    def __init__(
        self,
        showfliers: bool = True,
        log_scale: bool = False,
        figsize: tuple[int, int] = (10, 6),
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        orientation: Literal["vertical", "horizontal"] = "horizontal",
        sns_theme: dict[str, Any] | None = None,
        **boxplot_kwargs: object,
    ) -> None:
        """
        Initialize BoxPlotFormatter.

        Args:
            showfliers: Whether to show outliers in the boxplot (default: True)
            log_scale: Whether to use logarithmic scale for y-axis (default: False)
            figsize: Figure size (default: (10, 6))
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of boxplot ('vertical' or 'horizontal')
            sns_theme: Optional dictionary of seaborn theme parameters
            **boxplot_kwargs: Additional keyword arguments for boxplot function

        """
        super().__init__(
            showfliers=showfliers,
            log_scale=log_scale,
            figsize=figsize,
            engine=engine,
            orientation=orientation,
            plot_type="box",
            sns_theme=sns_theme,
            **boxplot_kwargs,
        )


class ViolinPlotFormatter(DistributionPlotFormatter):
    """Format benchmark results as a matplotlib violin plot."""

    def __init__(
        self,
        log_scale: bool = False,
        figsize: tuple[int, int] = (10, 6),
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        orientation: Literal["vertical", "horizontal"] = "horizontal",
        sns_theme: dict[str, Any] | None = None,
        **violinplot_kwargs: object,
    ) -> None:
        """
        Initialize ViolinPlotFormatter.

        Args:
            log_scale: Whether to use logarithmic scale for y-axis (default: False)
            figsize: Figure size (default: (10, 6))
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of violin plot ('vertical' or 'horizontal')
            sns_theme: Optional dictionary of seaborn theme parameters
            **violinplot_kwargs: Additional keyword arguments for violin plot function

        """
        super().__init__(
            showfliers=True,
            log_scale=log_scale,
            figsize=figsize,
            engine=engine,
            orientation=orientation,
            plot_type="violin",
            sns_theme=sns_theme,
            **violinplot_kwargs,
        )


class HistPlotFormatter(PlotFormatter):
    """
    Format benchmark results as histograms showing time/memory distribution.

    This formatter creates histograms visualizing the distribution of execution times
    (and optionally memory usage), which is useful for analyzing performance variations.
    """

    def __init__(
        self,
        bins: int | str = "auto",
        log_scale: bool = False,
        figsize: tuple[int, int] = (10, 6),
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        sns_theme: dict[str, Any] | None = None,
        **hist_kwargs: object,
    ) -> None:
        """
        Initialize HistPlotFormatter.

        Args:
            bins: Number of bins or binning strategy (default: "auto")
            log_scale: Whether to use logarithmic scale for x-axis (default: False)
            figsize: Figure size (default: (10, 6))
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            sns_theme: Optional dictionary of seaborn theme parameters
            **hist_kwargs: Additional keyword arguments for histogram function

        """
        self.bins = bins
        self.log_scale = log_scale
        self.figsize = figsize
        self.hist_kwargs = hist_kwargs

        # Handle seaborn engine and theme relationship
        self.engine, self.sns_theme = _handle_seaborn_engine_and_theme(
            engine,
            sns_theme,
        )

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a histogram.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the visualization

        """
        # Apply seaborn theme if needed
        with (
            set_seaborn_theme(self.sns_theme)
            if self.engine == "seaborn"
            else contextlib.nullcontext()
        ):
            # Sort method names according to configuration
            sorted_methods = self.sort_keys(stats, config)

            # Extract and preprocess time data
            time_data = self._preprocess_data(results, sorted_methods, config)
            time_unit = TimeUnit.from_config(config)

            # Create figure and axes based on memory tracking
            if config.time and config.memory:
                fig, (ax_time, ax_mem) = plt.subplots(
                    2,
                    1,
                    figsize=(self.figsize[0], self.figsize[1] * 1.8),
                )

                # Create time histogram
                self._create_time_histogram(ax_time, time_data, sorted_methods)

                # Apply styling to time plot
                self._apply_styling(
                    ax_time,
                    config,
                    title_suffix="",
                    unit=str(time_unit),
                    show_legend=True,
                )

                # Create memory histogram
                self._process_memory_subplot(ax_mem, results, config, sorted_methods)
            else:
                # Single plot for time data or memory data only
                fig, ax = plt.subplots(figsize=self.figsize)

                if config.time:
                    # Create time histogram
                    self._create_time_histogram(ax, time_data, sorted_methods)
                    # Apply styling to time plot
                    self._apply_styling(
                        ax,
                        config,
                        unit=str(time_unit),
                        show_legend=True,
                    )

                if config.memory and not config.time:
                    # Memory-only plot
                    self._process_memory_subplot(ax, results, config, sorted_methods)

            plt.tight_layout()
            return fig

    def _preprocess_data(
        self,
        results: ResultsType,
        sorted_methods: list[str],
        config: BenchConfig,
    ) -> dict[str, list[float]]:
        """Extract and preprocess benchmark time data."""
        time_data: dict[str, list[float]] = {}
        time_unit = TimeUnit.from_config(config)

        for method_name in sorted_methods:
            if "times" in results[method_name]:
                times = [
                    time_unit.convert_seconds(t) for t in results[method_name]["times"]
                ]
                time_data[method_name] = times

        return time_data

    def _preprocess_memory_data(
        self,
        results: ResultsType,
        sorted_methods: list[str],
        config: BenchConfig,
    ) -> dict[str, list[float]]:
        """Extract and preprocess benchmark memory data."""
        memory_data: dict[str, list[float]] = {}
        memory_unit = MemoryUnit.from_config(config)

        for method_name in sorted_methods:
            if "memory" in results[method_name]:
                memory_values = [
                    memory_unit.convert_bytes(m) for m in results[method_name]["memory"]
                ]
                memory_data[method_name] = memory_values

        return memory_data

    def _create_time_histogram(
        self,
        ax: plt.Axes,
        time_data: dict[str, list[float]],
        sorted_methods: list[str],
    ) -> None:
        """Create histogram of time data using the selected engine."""
        for method_name in sorted_methods:
            if method_name in time_data:
                times = time_data[method_name]
                if self.engine == "seaborn":
                    self._create_seaborn_histogram(ax, times, method_name)
                else:
                    self._create_matplotlib_histogram(ax, times, method_name)

    def _create_matplotlib_histogram(
        self,
        ax: plt.Axes,
        values: list[float],
        label: str,
    ) -> None:
        """Create a histogram using matplotlib."""
        ax.hist(
            values,
            bins=self.bins,
            label=label,
            **{
                "alpha": 0.7,
                **self.hist_kwargs,
            },  # type: ignore[arg-type]
        )

    def _create_seaborn_histogram(
        self,
        ax: plt.Axes,
        values: list[float],
        label: str,
    ) -> None:
        """Create a histogram using seaborn."""
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError as err:
            error_msg = (
                "seaborn is required for seaborn engine. "
                "Install with pip install seaborn."
            )
            raise ImportError(error_msg) from err

        # Use seaborn's histplot for better styling
        sns.histplot(
            values,
            bins=self.bins,
            label=label,
            ax=ax,
            alpha=0.7,  # Some transparency for overlapping histograms
            **self.hist_kwargs,  # type: ignore[arg-type]
        )

    def _process_memory_subplot(
        self,
        ax: plt.Axes,
        results: ResultsType,
        config: BenchConfig,
        sorted_methods: list[str],
    ) -> None:
        """
        Process and create memory usage histogram.

        Args:
            ax: The matplotlib axes for the memory subplot
            results: Dictionary mapping benchmark names to result data
            config: Benchmark configuration
            sorted_methods: List of method names to include

        """
        # Extract memory data
        memory_data = self._preprocess_memory_data(results, sorted_methods, config)
        memory_unit = MemoryUnit.from_config(config)

        # Plot memory data for each method
        for method_name in sorted_methods:
            if method_name in memory_data:
                memory_values = memory_data[method_name]
                if self.engine == "seaborn":
                    self._create_seaborn_histogram(
                        ax,
                        memory_values,
                        method_name,
                    )
                else:
                    self._create_matplotlib_histogram(
                        ax,
                        memory_values,
                        method_name,
                    )

        # Apply styling to memory plot
        self._apply_styling(
            ax,
            config,
            title_suffix="Memory Usage",
            unit=str(memory_unit),
            show_legend=True,
        )

    def _apply_styling(
        self,
        ax: plt.Axes,
        config: BenchConfig,
        title_suffix: str = "",
        unit: str = "s",
        show_legend: bool = False,
    ) -> None:
        """Apply common styling to the plot."""
        # Apply log scale if configured
        if self.log_scale:
            ax.set_xscale("log")

        # Set the plot title
        title = self._get_plot_title(config.trials, title_suffix)
        ax.set_title(title)

        # Set axis labels based on unit type
        self._set_axis_labels(ax, unit)

        # Show legend if requested
        if show_legend:
            ax.legend()

    def _get_plot_title(self, trials: int, title_suffix: str = "") -> str:
        """Generate the plot title."""
        base_title = f"Benchmark Results ({trials} trials)"
        if title_suffix:
            return f"{title_suffix} {base_title}"
        return base_title

    def _set_axis_labels(self, ax: plt.Axes, unit: str) -> None:
        """Set appropriate axis labels based on measurement type."""
        if any(unit == item.value for item in TimeUnit):
            ax.set_xlabel(f"Time ({unit})")
        else:
            ax.set_xlabel(f"Memory Usage ({unit})")

        ax.set_ylabel("Frequency")


class LinePlotFormatter(PlotFormatter):
    """
    Format benchmark results as a line plot showing time/memory across trials.

    This formatter creates a visualization of how execution time (and optionally memory)
    changes across trial runs, which is useful for determining how many warmup
    runs are needed before measurements stabilize.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 6),
        log_scale: bool = False,
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        sns_theme: dict[str, Any] | None = None,
        **plot_kwargs: object,
    ) -> None:
        """
        Initialize LinePlotFormatter.

        Args:
            figsize: Figure size (default: (10, 6))
            log_scale: Whether to use logarithmic scale for y-axis (default: False)
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            sns_theme: Optional dictionary of seaborn theme parameters
            **plot_kwargs: Additional keyword arguments for plot function

        """
        self.figsize = figsize
        self.log_scale = log_scale
        self.plot_kwargs = plot_kwargs

        # Handle seaborn engine and theme relationship
        self.engine, self.sns_theme = _handle_seaborn_engine_and_theme(
            engine,
            sns_theme,
        )

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a trial progression plot.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the visualization

        """
        # Apply seaborn theme if needed
        with (
            set_seaborn_theme(self.sns_theme)
            if self.engine == "seaborn"
            else contextlib.nullcontext()
        ):
            # Sort method names according to configuration
            sorted_methods = self.sort_keys(stats, config)

            # Extract and preprocess time data
            time_data = self._preprocess_data(results, sorted_methods, config)
            time_unit = TimeUnit.from_config(config)

            # Create figure and axes based on memory tracking
            if config.time and config.memory:
                fig, (ax_time, ax_mem) = plt.subplots(
                    2,
                    1,
                    figsize=(self.figsize[0], self.figsize[1] * 1.8),
                    sharex=True,
                )

                # Create time plot using appropriate engine
                self._create_time_plot(ax_time, time_data, sorted_methods)

                # Apply styling to time plot
                self._apply_styling(
                    ax_time,
                    config,
                    title_suffix="",
                    unit=str(time_unit),
                    show_legend=True,
                )

                # Create memory subplot
                self._process_memory_subplot(ax_mem, results, config, sorted_methods)
            else:
                # Single plot for time data only
                fig, ax = plt.subplots(figsize=self.figsize)

                if config.time:
                    # Create time plot using appropriate engine
                    self._create_time_plot(ax, time_data, sorted_methods)
                    # Apply styling to time plot
                    self._apply_styling(
                        ax,
                        config,
                        unit=str(time_unit),
                        show_legend=True,
                    )

                if config.memory:
                    self._process_memory_subplot(ax, results, config, sorted_methods)

            plt.tight_layout()
            return fig

    def _preprocess_data(
        self,
        results: ResultsType,
        sorted_methods: list[str],
        config: BenchConfig,
    ) -> dict[str, tuple[list[int], list[float]]]:
        """Extract and preprocess benchmark time data."""
        time_data: dict[str, tuple[list[int], list[float]]] = {}
        time_unit = TimeUnit.from_config(config)

        for method_name in sorted_methods:
            if "times" in results[method_name]:
                times = [
                    time_unit.convert_seconds(t) for t in results[method_name]["times"]
                ]
                x_values = list(range(1, len(times) + 1))
                time_data[method_name] = (x_values, times)

        return time_data

    def _preprocess_memory_data(
        self,
        results: ResultsType,
        sorted_methods: list[str],
        config: BenchConfig,
    ) -> dict[str, tuple[list[int], list[float]]]:
        """Extract and preprocess benchmark memory data."""
        memory_data: dict[str, tuple[list[int], list[float]]] = {}
        memory_unit = MemoryUnit.from_config(config)

        for method_name in sorted_methods:
            if "memory" in results[method_name]:
                memory_values = [
                    memory_unit.convert_bytes(m) for m in results[method_name]["memory"]
                ]
                x_values = list(range(1, len(memory_values) + 1))
                memory_data[method_name] = (x_values, memory_values)

        return memory_data

    def _create_time_plot(
        self,
        ax: plt.Axes,
        time_data: dict[str, tuple[list[int], list[float]]],
        sorted_methods: list[str],
    ) -> None:
        """Create time plot using the selected engine."""
        for method_name in sorted_methods:
            if method_name in time_data:
                x_values, times = time_data[method_name]
                if self.engine == "seaborn":
                    self._create_seaborn_lineplot(ax, x_values, times, method_name)
                else:
                    self._create_matplotlib_lineplot(ax, x_values, times, method_name)

    def _create_matplotlib_lineplot(
        self,
        ax: plt.Axes,
        x_values: list[int],
        y_values: list[float],
        label: str,
    ) -> None:
        """Create a line plot using matplotlib."""
        ax.plot(
            x_values,
            y_values,
            label=label,
            **self.plot_kwargs,  # type: ignore[arg-type]
        )

    def _create_seaborn_lineplot(
        self,
        ax: plt.Axes,
        x_values: list[int],
        y_values: list[float],
        label: str,
    ) -> None:
        """Create a line plot using seaborn."""
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError as err:
            error_msg = (
                "seaborn is required for seaborn engine. "
                "Install with pip install seaborn."
            )
            raise ImportError(error_msg) from err

        sns.lineplot(
            x=x_values,
            y=y_values,
            label=label,
            ax=ax,
            **self.plot_kwargs,  # type: ignore[arg-type]
        )

    def _process_memory_subplot(
        self,
        ax: plt.Axes,
        results: ResultsType,
        config: BenchConfig,
        sorted_methods: list[str],
    ) -> None:
        """
        Process and create memory usage subplot.

        Args:
            ax: The matplotlib axes for the memory subplot
            results: Dictionary mapping benchmark names to result data
            config: Benchmark configuration
            sorted_methods: List of method names to include

        """
        # Extract memory data
        memory_data = self._preprocess_memory_data(results, sorted_methods, config)
        memory_unit = MemoryUnit.from_config(config)

        # Plot memory data for each method
        for method_name in sorted_methods:
            if method_name in memory_data:
                x_values, memory_values = memory_data[method_name]
                if self.engine == "seaborn":
                    self._create_seaborn_lineplot(
                        ax,
                        x_values,
                        memory_values,
                        method_name,
                    )
                else:
                    self._create_matplotlib_lineplot(
                        ax,
                        x_values,
                        memory_values,
                        method_name,
                    )

        # Apply styling to memory plot
        self._apply_styling(
            ax,
            config,
            title_suffix="Memory Usage",
            unit=str(memory_unit),
            show_legend=True,
            xlabel="Trials",
        )

    def _apply_styling(
        self,
        ax: plt.Axes,
        config: BenchConfig,
        title_suffix: str = "",
        unit: str = "s",
        show_legend: bool = False,
        xlabel: str = "Trials",
    ) -> None:
        """Apply common styling to the plot."""
        # Apply log scale if configured
        if self.log_scale:
            ax.set_yscale("log")

        # Set the plot title
        title = self._get_plot_title(config.trials, title_suffix)
        ax.set_title(title)

        # Set axis labels
        ax.set_xlabel(xlabel)
        self._set_ylabel(ax, unit)

        # Show legend if requested
        if show_legend:
            ax.legend()

    def _get_plot_title(self, trials: int, title_suffix: str = "") -> str:
        """Generate the plot title."""
        base_title = f"Benchmark Results ({trials} trials)"
        if title_suffix:
            return f"{title_suffix} {base_title}"
        return base_title

    def _set_ylabel(self, ax: plt.Axes, unit: str) -> None:
        """Set the y-axis label based on the unit type."""
        if any(unit == item.value for item in TimeUnit):
            label_text = f"Time ({unit})"
        else:
            label_text = f"Memory Usage ({unit})"

        ax.set_ylabel(label_text)


class BarPlotFormatter(PlotFormatter):
    """Format benchmark results as a bar plot showing selected metrics."""

    def __init__(
        self,
        metric: MetricType | list[MetricType] | None = None,
        log_scale: bool = False,
        figsize: tuple[int, int] = (10, 6),
        engine: Literal["matplotlib", "seaborn"] = "matplotlib",
        orientation: Literal["vertical", "horizontal"] = "horizontal",
        sns_theme: dict[str, Any] | None = None,
        **plot_kwargs: object,
    ) -> None:
        """
        Initialize BarPlotFormatter.

        Args:
            metric: The metric(s) to display. Can be a single metric
                or a list of metrics: "avg", "min", "max", "avg_memory", "max_memory"
            log_scale: Whether to use logarithmic scale for data axis (default: False)
            figsize: Figure size (default: (10, 6))
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of plot ('vertical' or 'horizontal')
            sns_theme: Optional dictionary of seaborn theme parameters
            **plot_kwargs: Additional keyword arguments for bar plot function

        """
        self.metric = metric
        self.log_scale = log_scale
        self.figsize = figsize
        self.orientation = orientation
        self.plot_kwargs = plot_kwargs

        # Handle seaborn engine and theme relationship
        self.engine, self.sns_theme = _handle_seaborn_engine_and_theme(
            engine,
            sns_theme,
        )

    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> Figure:
        """
        Format benchmark results as a bar plot.

        Args:
            results: Dictionary mapping benchmark names to result data
            stats: Dictionary of calculated statistics
            config: Benchmark configuration

        Returns:
            Matplotlib Figure object containing the bar plot

        """
        _ = results

        metrics: list = []
        if self.metric is None:
            metrics = ["avg", "avg_memory"]
        else:
            metrics = [self.metric] if isinstance(self.metric, str) else self.metric

        # Apply seaborn theme if needed
        with (
            set_seaborn_theme(self.sns_theme)
            if self.engine == "seaborn"
            else contextlib.nullcontext()
        ):
            # Sort method names according to configuration
            sorted_methods = self.sort_keys(stats, config)

            # Check if metrics are valid and available
            valid_metrics = []
            for metric in metrics:
                if metric in ("avg", "min", "max") and not config.time:
                    continue
                if metric in ("avg_memory", "max_memory") and not config.memory:
                    continue
                valid_metrics.append(metric)

            if not valid_metrics:
                error_msg = "No valid metrics to plot"
                raise ValueError(error_msg)

            # Create figure with appropriate number of subplots
            n_plots = len(valid_metrics)
            if n_plots > 1:
                fig, _axes = plt.subplots(
                    n_plots,
                    1,
                    figsize=(self.figsize[0], self.figsize[1] * n_plots * 0.7),
                    squeeze=False,
                )
                axes = list(_axes.flatten())
            else:
                fig, ax = plt.subplots(figsize=self.figsize)
                axes = [ax]

            # Process each metric
            for i, metric in enumerate(valid_metrics):
                ax = axes[i]
                self._create_bar_plot_for_metric(
                    ax=ax,
                    metric=metric,
                    sorted_methods=sorted_methods,
                    stats=stats,
                    config=config,
                )

            plt.tight_layout()
            return fig

    def _create_bar_plot_for_metric(
        self,
        ax: plt.Axes,
        metric: MetricType,
        sorted_methods: list[str],
        stats: StatsType,
        config: BenchConfig,
    ) -> None:
        """Create a bar plot for a specific metric."""
        # Extract data for the given metric
        values = []
        labels = []

        for method_name in sorted_methods:
            if metric not in stats[method_name]:
                continue

            labels.append(method_name)

            # Convert values based on metric type
            if metric in ("avg", "min", "max"):
                time_unit = TimeUnit.from_config(config)
                values.append(time_unit.convert_seconds(stats[method_name][metric]))
            else:  # avg_memory or max_memory
                memory_unit = MemoryUnit.from_config(config)
                values.append(memory_unit.convert_bytes(stats[method_name][metric]))

        # Create the plot using the appropriate engine
        if self.engine == "seaborn":
            self._create_seaborn_bar_plot(ax, values)
        else:
            self._create_matplotlib_bar_plot(ax, values)

        # Apply styling
        self._apply_styling(ax, labels, metric, config)

    def _create_matplotlib_bar_plot(
        self,
        ax: plt.Axes,
        values: list[float],
    ) -> None:
        """Create a bar plot using matplotlib."""
        if self.orientation == "horizontal":
            for i, value in enumerate(values):
                ax.barh(
                    i,
                    value,
                    **self.plot_kwargs,  # type: ignore [arg-type]
                )
        else:
            for i, value in enumerate(values):
                ax.bar(
                    i,
                    value,
                    **self.plot_kwargs,  # type: ignore [arg-type]
                )

    def _create_seaborn_bar_plot(
        self,
        ax: plt.Axes,
        values: list[float],
    ) -> None:
        """Create a bar plot using seaborn."""
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError as err:
            error_msg = (
                "seaborn is required for seaborn engine. "
                "Install with pip install seaborn."
            )
            raise ImportError(error_msg) from err

        if self.orientation == "horizontal":
            for i, value in enumerate(values):
                sns.barplot(
                    x=[value],
                    y=[i],
                    ax=ax,
                    orient="h",
                    **self.plot_kwargs,  # type: ignore [arg-type]
                )
        else:
            for i, value in enumerate(values):
                sns.barplot(
                    x=[i],
                    y=[value],
                    ax=ax,
                    orient="v",
                    **self.plot_kwargs,  # type: ignore [arg-type]
                )

    def _apply_styling(
        self,
        ax: plt.Axes,
        labels: list[str],
        metric: MetricType,
        config: BenchConfig,
    ) -> None:
        """Apply styling to the plot."""
        # Set scale (log or linear)
        if self.log_scale:
            if self.orientation == "horizontal":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")

        # Set title based on metric
        self._set_title(ax, metric, config)

        # Set labels and ticks
        if self.orientation == "horizontal":
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

            # Set x-axis label based on metric type
            self._set_axis_label(ax, metric, config, "x")
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)

            # Use autofmt_xdate for automatic rotation of x labels
            if self.orientation == "vertical":
                ax.figure.autofmt_xdate()

            # Set y-axis label based on metric type
            self._set_axis_label(ax, metric, config, "y")

    def _set_title(self, ax: plt.Axes, metric: MetricType, config: BenchConfig) -> None:
        """Set plot title based on metric type."""
        metric_titles = {
            "avg": "Average Time",
            "min": "Minimum Time",
            "max": "Maximum Time",
            "avg_memory": "Average Memory Usage",
            "max_memory": "Maximum Memory Usage",
        }

        title = f"Benchmark Results [{metric_titles[metric]}] ({config.trials} trials)"
        ax.set_title(title)

    def _set_axis_label(
        self,
        ax: plt.Axes,
        metric: MetricType,
        config: BenchConfig,
        axis: Literal["x", "y"],
    ) -> None:
        """Set appropriate axis label based on metric type and orientation."""
        if metric in ("avg", "min", "max"):
            time_unit = TimeUnit.from_config(config)
            label_text = f"Time ({time_unit})"
        else:
            memory_unit = MemoryUnit.from_config(config)
            label_text = f"Memory Usage ({memory_unit})"

        if axis == "x" and self.orientation == "horizontal":
            ax.set_xlabel(label_text)
        elif axis == "y" and self.orientation == "vertical":
            ax.set_ylabel(label_text)


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

    def report_formatted(self, formatted_output: Formatted) -> None:
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
