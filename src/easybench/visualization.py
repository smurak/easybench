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
from .reporters import Formatted, Formatter, Reporter

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
        y_limit: tuple[float, float] | None = None,
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
            y_limit: Optional tuple of (min, max) for y-axis limits
            trim_outliers: Optional percentile for trimming outliers (0.0-0.5)
            winsorize_outliers: Optional percentile for winsorizing outliers (0.0-0.5)
            figsize: Figure size (default: (10, 6))
            label_rotation_threshold: Rotate x-axis labels if method count exceeds this.
            engine: Plotting backend to use ('matplotlib' or 'seaborn')
            orientation: Direction of boxplot ('vertical' or 'horizontal')
            **boxplot_kwargs: Additional keyword arguments for boxplot function

        """
        self.log_scale = log_scale
        self.y_limit = y_limit
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
        data, labels = self._preprocess_data(results, stats, config)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create boxplot with appropriate engine
        box_data = [data[method] for method in labels]
        if self.engine == "seaborn":
            self._create_seaborn_boxplot(ax, box_data, labels)
        else:  # matplotlib is the default
            self._create_matplotlib_boxplot(ax, box_data, labels)

        # Apply styling to plot
        self._apply_styling(ax, config, labels)

        plt.tight_layout()
        return fig

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

    def _create_matplotlib_boxplot(
        self,
        ax: plt.Axes,
        box_data: list[list[float]],
        labels: list[str],
    ) -> None:
        """Create a boxplot using matplotlib."""
        if self.orientation == "horizontal":
            ax.boxplot(
                box_data,
                showfliers=self.showfliers,
                orientation=self.orientation,
                **self.boxplot_kwargs,  # type: ignore[arg-type]
            )
            # Set y-tick positions explicitly before setting labels
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels)
        else:
            ax.boxplot(
                box_data,
                showfliers=self.showfliers,
                orientation=self.orientation,
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
    ) -> None:
        """Apply common styling to the plot."""
        # Apply log scale if requested
        if self.orientation == "horizontal":
            # For horizontal boxplot, apply log scale to x-axis
            if self.log_scale:
                ax.set_xscale("log")

            # Apply x-axis limits if provided
            if self.y_limit is not None:
                ax.set_xlim(self.y_limit)

            # Set labels and title
            ax.set_title(f"Benchmark Results ({config.trials} trials)")
            ax.set_xlabel("Time (seconds)")
        else:
            # For vertical boxplot (original behavior)
            if self.log_scale:
                ax.set_yscale("log")

            if self.y_limit is not None:
                ax.set_ylim(self.y_limit)

            ax.set_title(f"Benchmark Results ({config.trials} trials)")
            ax.set_ylabel("Time (seconds)")

            if len(labels) > self.label_rotation_threshold:
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
