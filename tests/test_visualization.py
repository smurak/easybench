"""
Tests for the visualization module.

This module tests the visualization components, including:
- BoxPlotFormatter
- LinePlotFormatter
- PlotReporter
"""

import tempfile
import warnings
from importlib.util import find_spec
from unittest import mock
from unittest.mock import MagicMock, patch

import matplotlib as mpl
import matplotlib.figure

mpl.use("Agg")  # Use non-interactive backend for testing
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from easybench import BenchConfig
from easybench.core import ResultsType, ResultType, StatsType, StatType
from easybench.visualization import (
    DEFAULT_SNS_THEME,
    BarPlotFormatter,
    BoxPlotFormatter,
    HistPlotFormatter,
    LinePlotFormatter,
    PlotReporter,
    ViolinPlotFormatter,
)

if TYPE_CHECKING:
    from easybench.reporters import MetricType

# Constants for test values
TEST_TIME_VALUE = 0.1
TEST_SLOW_TIME = 0.2
TEST_SLOWER_TIME = 0.3
TEST_AVG_TIME = 0.2
TEST_TIME_VALUES = [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]

# Constants for test configurations
NUM_TEST_FUNCTIONS = 2
DEFAULT_DPI = 100
CUSTOM_DPI = 200
EXPECTED_CALL_COUNT = 2  # Number of times boxplot is called during fallback tests
EXPECTED_LINEWIDTH = 2
EXPECTED_DATA_LENGTH = 2
CUSTOM_HIGH_DPI = 300

TEST_TRIALS = 5
MIN_TIME = 0.001
MAX_TIME = 0.01
AVG_TIME = 0.005
MIN_MEMORY = 1000
MAX_MEMORY = 5000
AVG_MEMORY = 3000


def complete_stat(
    dic: dict[str, float],
    memory: bool = False,  # noqa: FBT002, FBT001
    time: bool = True,  # noqa: FBT002, FBT001
) -> StatType:
    """Complete dictionaries for StatType."""
    stat: StatType = {}

    if time:
        stat.update(
            {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
        )

    if memory:
        stat.update(
            {
                "avg_memory": 0.0,
                "max_memory": 0.0,
            },
        )

    stat.update(dic)  # type: ignore [literal-required, typeddict-item]
    return stat


@pytest.fixture
def sample_results() -> dict[str, ResultType]:
    """Sample benchmark results for testing."""
    return {
        "test_func1": {"times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]},
        "test_func2": {"times": [TEST_SLOWER_TIME, TEST_TIME_VALUE, TEST_SLOW_TIME]},
    }


@pytest.fixture
def sample_stats() -> dict[str, StatType]:
    """Sample benchmark statistics for testing."""
    return {
        "test_func1": complete_stat(
            {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
            },
        ),
        "test_func2": complete_stat(
            {
                "avg": TEST_AVG_TIME,
                "min": TEST_TIME_VALUE,
                "max": TEST_SLOWER_TIME,
            },
        ),
    }


@pytest.fixture
def sample_config() -> BenchConfig:
    """Sample benchmark configuration for testing."""
    return BenchConfig(trials=3)


class TestBoxPlotFormatter:
    """Tests for the BoxPlotFormatter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = BoxPlotFormatter()
        assert formatter.showfliers is True
        assert formatter.log_scale is False
        assert formatter.figsize == (10, 6)
        assert formatter.engine == "matplotlib"
        assert formatter.orientation == "horizontal"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = BoxPlotFormatter(
            showfliers=False,
            log_scale=True,
            figsize=(8, 4),
            engine="seaborn",
            orientation="vertical",
        )
        assert formatter.showfliers is False
        assert formatter.log_scale is True
        assert formatter.figsize == (8, 4)
        assert formatter.engine == "seaborn"
        assert formatter.orientation == "vertical"

    def test_preprocess_data_no_outlier_handling(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test data preprocessing with no outlier handling."""
        formatter = BoxPlotFormatter()
        data, labels = formatter._preprocess_data(
            sample_results,
            sample_stats,
            sample_config,
        )

        assert len(data) == NUM_TEST_FUNCTIONS
        assert "test_func1" in data
        assert "test_func2" in data
        assert data["test_func1"] == TEST_TIME_VALUES
        assert len(labels) == NUM_TEST_FUNCTIONS
        assert "test_func1" in labels
        assert "test_func2" in labels

    @pytest.mark.parametrize("engine", ["matplotlib", "seaborn"])
    def test_format_with_different_engines(
        self,
        engine: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different plotting engines."""
        # Skip test if seaborn not installed when testing seaborn engine
        if engine == "seaborn":
            if not find_spec("seaborn"):
                pytest.skip("seaborn not installed")

            warnings.catch_warnings()
            warnings.simplefilter("ignore", PendingDeprecationWarning)

        formatter = BoxPlotFormatter(engine=engine)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0
        plt.close()

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_format_with_different_orientations(
        self,
        orientation: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different orientations."""
        formatter = BoxPlotFormatter(orientation=orientation)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0
        plt.close()

    def test_create_matplotlib_boxplot(self) -> None:
        """Test creation of boxplot with matplotlib engine."""
        formatter = BoxPlotFormatter()

        # Mock a matplotlib Axes object
        with mock.patch("matplotlib.pyplot.Axes") as mock_axes:
            mock_axes.boxplot = mock.MagicMock()
            formatter._create_matplotlib_boxplot(
                mock_axes,
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                ["test1", "test2"],
            )

            # Verify boxplot was called
            mock_axes.boxplot.assert_called_once()

            # Check that set_yticks was called for horizontal orientation
            mock_axes.set_yticks.assert_called_once()
            mock_axes.set_yticklabels.assert_called_once()

    def test_create_seaborn_boxplot(self) -> None:
        """Test creation of boxplot with seaborn engine."""
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = BoxPlotFormatter(engine="seaborn")

        with mock.patch("seaborn.boxplot") as mock_boxplot:
            # Mock a matplotlib Axes object
            mock_axes = mock.MagicMock()

            formatter._create_seaborn_boxplot(
                mock_axes,
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                ["test1", "test2"],
            )

            # Verify seaborn.boxplot was called
            mock_boxplot.assert_called_once()

    def test_create_matplotlib_boxplot_horizontal_fallback(self) -> None:
        """
        Test fallback to vert parameter when orientation isn't supported.

        Tests the horizontal orientation case.
        """
        formatter = BoxPlotFormatter(orientation="horizontal")

        # Create mock Axes
        mock_axes = mock.MagicMock()

        # Make the first boxplot call raise TypeError to simulate orientation
        # not being supported
        mock_axes.boxplot.side_effect = [TypeError("orientation not supported"), None]

        formatter._create_matplotlib_boxplot(
            mock_axes,
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            ["test1", "test2"],
        )

        # Check that boxplot was called twice (first with orientation, then with vert)
        assert mock_axes.boxplot.call_count == EXPECTED_CALL_COUNT

        # First call should have used orientation
        args, kwargs = mock_axes.boxplot.call_args_list[0]
        assert "orientation" in kwargs
        assert kwargs["orientation"] == "horizontal"

        # Second call should have used vert
        args, kwargs = mock_axes.boxplot.call_args_list[1]
        assert "orientation" not in kwargs
        assert "vert" in kwargs
        assert kwargs["vert"] is False  # horizontal means vert=False

        # Ensure tick settings were called
        mock_axes.set_yticks.assert_called_once()
        mock_axes.set_yticklabels.assert_called_once()

    def test_create_matplotlib_boxplot_vertical_fallback(self) -> None:
        """
        Test fallback to vert parameter when orientation isn't supported.

        Tests the vertical orientation case.
        """
        formatter = BoxPlotFormatter(orientation="vertical")

        # Create mock Axes
        mock_axes = mock.MagicMock()

        # Make the first boxplot call raise TypeError to simulate orientation
        # not being supported
        mock_axes.boxplot.side_effect = [TypeError("orientation not supported"), None]

        formatter._create_matplotlib_boxplot(
            mock_axes,
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            ["test1", "test2"],
        )

        # Check that boxplot was called twice (first with orientation, then with vert)
        assert mock_axes.boxplot.call_count == EXPECTED_CALL_COUNT

        # First call should have used orientation
        args, kwargs = mock_axes.boxplot.call_args_list[0]
        assert "orientation" in kwargs
        assert kwargs["orientation"] == "vertical"

        # Second call should have used vert
        args, kwargs = mock_axes.boxplot.call_args_list[1]
        assert "orientation" not in kwargs
        assert "vert" in kwargs
        assert kwargs["vert"] is True  # vertical means vert=True

        # Ensure tick settings were called
        mock_axes.set_xticks.assert_called_once()
        mock_axes.set_xticklabels.assert_called_once()

    def test_format_with_memory_enabled(
        self,
    ) -> None:
        """Test formatting with memory metrics enabled."""
        # Modify sample results and stats to include memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
            "test_func2": {
                "times": [TEST_SLOWER_TIME, TEST_TIME_VALUE, TEST_SLOW_TIME],
                "memory": [2048, 1024, 3072],
            },
        }

        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
            "test_func2": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
        }

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=3, memory=True)

        formatter = BoxPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch.object(plt, "tight_layout"),
        ):
            # Create mock axes for time and memory subplots
            mock_ax_time = mock.MagicMock()
            mock_ax_mem = mock.MagicMock()
            mock_fig = mock.MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax_time, mock_ax_mem))

            # Call format with memory enabled
            fig = formatter.format(memory_results, memory_stats, memory_config)

            # Verify subplots were created (2 plots for time and memory)
            mock_subplots.assert_called_once()
            args, kwargs = mock_subplots.call_args
            assert args == (2, 1)  # 2 rows, 1 column

            # Verify boxplot was called for both time and memory plots
            assert mock_ax_time.boxplot.call_count == 1
            assert mock_ax_mem.boxplot.call_count == 1

            # Verify memory subplot had styling applied
            mock_ax_mem.set_title.assert_called_once()

            assert fig is not None
        plt.close()

    def test_preprocess_memory_data(
        self,
    ) -> None:
        """Test memory data preprocessing functionality."""
        # Create sample results with memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],  # Memory in bytes
            },
            "test_func2": {
                "times": [TEST_SLOWER_TIME, TEST_TIME_VALUE, TEST_SLOW_TIME],
                "memory": [2048, 1024, 3072],  # Memory in bytes
            },
        }

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=3, memory=True)

        formatter = BoxPlotFormatter()
        labels = ["test_func1", "test_func2"]

        # Call _preprocess_memory_data directly
        memory_data = formatter._preprocess_memory_data(
            memory_results,
            memory_config,
            labels,
        )

        # Verify memory data was correctly preprocessed
        assert "test_func1" in memory_data
        assert "test_func2" in memory_data
        assert memory_data["test_func1"] == [1.0, 2.0, 3.0]  # Converted to KB
        assert memory_data["test_func2"] == [2.0, 1.0, 3.0]  # Converted to KB

    def test_format_with_memory_units(
        self,
    ) -> None:
        """Test formatting with different memory units."""
        # Modify sample results and stats to include memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024 * 1024],  # 1 MB in bytes
            },
        }

        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 1024 * 1024,  # 1 MB in bytes
                    "max_memory": 1024 * 1024,  # 1 MB in bytes
                },
                memory=True,
            ),
        }

        # Create a config with memory in MB
        memory_config = BenchConfig(trials=1, memory="MB")  # type: ignore [arg-type]

        formatter = BoxPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch.object(plt, "tight_layout"),
        ):
            # Create mock axes for time and memory subplots
            mock_ax_time = mock.MagicMock()
            mock_ax_mem = mock.MagicMock()
            mock_fig = mock.MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax_time, mock_ax_mem))

            # Call format with memory enabled
            formatter.format(memory_results, memory_stats, memory_config)

            # Verify memory subplot styling with MB unit
            memory_styling_call = [
                call
                for call in mock_ax_mem.set_title.call_args_list
                if "Memory Usage" in str(call)
            ]
            assert len(memory_styling_call) > 0

            # Verify proper axis label was set (with MB)
            if formatter.orientation == "horizontal":
                # For horizontal orientation, xlabel contains the unit
                assert mock_ax_mem.set_xlabel.call_count > 0
                label_call = mock_ax_mem.set_xlabel.call_args[0][0]
                assert "MB" in label_call
            else:
                # For vertical orientation, ylabel contains the unit
                assert mock_ax_mem.set_ylabel.call_count > 0
                label_call = mock_ax_mem.set_ylabel.call_args[0][0]
                assert "MB" in label_call

    def test_process_memory_subplot(
        self,
    ) -> None:
        """Test the _process_memory_subplot method."""
        # Create sample results with memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE],
                "memory": [1024],  # 1 KB in bytes
            },
            "test_func2": {
                "times": [TEST_TIME_VALUE],
                "memory": [2048],  # 2 KB in bytes
            },
        }

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=1, memory=True)

        formatter = BoxPlotFormatter()
        labels = ["test_func1", "test_func2"]

        # Mock matplotlib axes
        mock_ax = mock.MagicMock()

        # Call _process_memory_subplot directly
        formatter._process_memory_subplot(
            mock_ax,
            memory_results,
            memory_config,
            labels,
        )

        # Verify boxplot was created
        mock_ax.boxplot.assert_called_once()

        # Verify styling was applied
        mock_ax.set_title.assert_called_once()

        # Verify ticks and tick labels were set based on orientation
        if formatter.orientation == "horizontal":
            mock_ax.set_yticks.assert_called_once()
            mock_ax.set_yticklabels.assert_called_once()
        else:
            mock_ax.set_xticks.assert_called_once()
            mock_ax.set_xticklabels.assert_called_once()

    def test_sns_theme_application(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that sns_theme parameters are correctly applied."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        custom_theme = {"style": "darkgrid", "palette": "Set2"}
        formatter = BoxPlotFormatter(sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify that seaborn.set_theme was called with the custom theme
            mock_set_theme.assert_called_once_with(**custom_theme)
        plt.close()

    def test_sns_theme_forces_seaborn_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that providing sns_theme automatically sets engine to seaborn."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        # Create formatter with matplotlib engine but sns_theme provided
        custom_theme = {"style": "whitegrid"}
        formatter = BoxPlotFormatter(engine="matplotlib", sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.boxplot") as mock_boxplot,
        ):
            # Mock seaborn.boxplot to return a mock axes
            mock_boxplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn functions were called (indicating seaborn engine was used)
            mock_set_theme.assert_called_once_with(**custom_theme)
            mock_boxplot.assert_called()
        plt.close()

    def test_sns_theme_with_default_theme(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that seaborn engine uses default theme when no sns_theme provided."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = BoxPlotFormatter(engine="seaborn")

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.boxplot") as mock_boxplot,
        ):
            # Mock seaborn.boxplot to return a mock axes
            mock_boxplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None

            mock_set_theme.assert_called_once_with(**DEFAULT_SNS_THEME)
        plt.close()

    def test_sns_theme_none_with_matplotlib_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that matplotlib engine doesn't call seaborn.set_theme."""
        formatter = BoxPlotFormatter(engine="matplotlib", sns_theme=None)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn.set_theme was not called
            mock_set_theme.assert_not_called()
        plt.close()


class TestLinePlotFormatter:
    """Tests for the LinePlotFormatter class."""

    @pytest.fixture
    def mock_results(self) -> ResultsType:
        """Create sample results data for LinePlotFormatter tests."""
        return {
            "test_method1": {
                "times": [
                    MIN_TIME + (MAX_TIME - MIN_TIME) * i / TEST_TRIALS
                    for i in range(TEST_TRIALS)
                ],
                "memory": [
                    MIN_MEMORY + (MAX_MEMORY - MIN_MEMORY) * i / TEST_TRIALS
                    for i in range(TEST_TRIALS)
                ],
            },
            "test_method2": {
                "times": [
                    MAX_TIME - (MAX_TIME - MIN_TIME) * i / TEST_TRIALS
                    for i in range(TEST_TRIALS)
                ],
                "memory": [
                    MAX_MEMORY - (MAX_MEMORY - MIN_MEMORY) * i / TEST_TRIALS
                    for i in range(TEST_TRIALS)
                ],
            },
        }

    @pytest.fixture
    def mock_stats(self) -> StatsType:
        """Create sample stats data for LinePlotFormatter tests."""
        return {
            "test_method1": complete_stat(
                {
                    "avg": AVG_TIME,
                    "min": MIN_TIME,
                    "max": MAX_TIME,
                    "avg_memory": AVG_MEMORY,
                    "max_memory": MAX_MEMORY,
                },
                memory=True,
            ),
            "test_method2": complete_stat(
                {
                    "avg": AVG_TIME,
                    "min": MIN_TIME,
                    "max": MAX_TIME,
                    "avg_memory": AVG_MEMORY,
                    "max_memory": MAX_MEMORY,
                },
                memory=True,
            ),
        }

    @pytest.fixture
    def mock_config(self) -> BenchConfig:
        """Create sample benchmark configuration for LinePlotFormatter tests."""
        return BenchConfig(trials=TEST_TRIALS, memory=True)

    def test_init(self) -> None:
        """Test initialization and parameter setting."""
        formatter = LinePlotFormatter(
            figsize=(12, 8),
            log_scale=True,
            engine="matplotlib",
            linewidth=EXPECTED_LINEWIDTH,
            marker="o",
        )

        assert formatter.figsize == (12, 8)
        assert formatter.log_scale is True
        assert formatter.engine == "matplotlib"
        assert formatter.plot_kwargs["linewidth"] == EXPECTED_LINEWIDTH
        assert formatter.plot_kwargs["marker"] == "o"

    @pytest.mark.parametrize(
        ("memory_enabled", "expected_subplots"),
        [(True, 2), (False, 1)],
    )
    def test_format_subplots(
        self,
        mock_results: ResultsType,
        mock_stats: StatsType,
        mock_config: BenchConfig,
        memory_enabled: bool,  # noqa: FBT001
        expected_subplots: int,
    ) -> None:
        """Test that the correct number of subplots are created based on memory flag."""
        mock_config.memory = memory_enabled
        formatter = LinePlotFormatter()

        figure = formatter.format(mock_results, mock_stats, mock_config)

        assert isinstance(figure, matplotlib.figure.Figure)
        assert len(figure.axes) == expected_subplots

    def test_format_matplotlib_engine(
        self,
        mock_results: ResultsType,
        mock_stats: StatsType,
        mock_config: BenchConfig,
    ) -> None:
        """Test using matplotlib as the plotting engine."""
        formatter = LinePlotFormatter(engine="matplotlib")

        with (
            patch("matplotlib.axes.Axes.plot") as mock_plot,
            patch("matplotlib.axes.Axes.legend") as mock_legend,
        ):  # Add mock for legend
            # Make mock_plot return a list of artist objects
            mock_plot.return_value = [MagicMock()]

            figure = formatter.format(mock_results, mock_stats, mock_config)

            assert isinstance(figure, matplotlib.figure.Figure)
            assert mock_plot.called
            assert mock_legend.called

    def test_format_seaborn_engine(
        self,
        mock_results: ResultsType,
        mock_stats: StatsType,
        mock_config: BenchConfig,
    ) -> None:
        """Test using seaborn as the plotting engine."""
        try:
            # Check if seaborn is installed without importing it directly
            if not find_spec("seaborn"):
                pytest.skip("Seaborn not installed, skipping seaborn engine test")

            formatter = LinePlotFormatter(engine="seaborn")

            with (
                patch("seaborn.lineplot") as mock_lineplot,
                patch("matplotlib.axes.Axes.legend") as mock_legend,
            ):  # Add mock for legend
                # Make mock_lineplot return a list of artist objects
                mock_lineplot.return_value = MagicMock()

                figure = formatter.format(mock_results, mock_stats, mock_config)

                assert isinstance(figure, matplotlib.figure.Figure)
                assert mock_lineplot.called
                assert mock_legend.called
        except ImportError:
            pytest.skip("Seaborn not installed, skipping seaborn engine test")

    def test_preprocess_data(
        self,
        mock_results: ResultsType,
        mock_stats: StatsType,
        mock_config: BenchConfig,
    ) -> None:
        """Test data preprocessing."""
        formatter = LinePlotFormatter()
        sorted_methods = formatter.sort_keys(mock_stats, mock_config)

        time_data = formatter._preprocess_data(
            mock_results,
            sorted_methods,
            mock_config,
        )

        assert len(time_data) == EXPECTED_DATA_LENGTH
        assert "test_method1" in time_data
        assert "test_method2" in time_data

        # Check data structure: (x_values, times)
        x_values, times = time_data["test_method1"]
        assert len(x_values) == TEST_TRIALS
        assert len(times) == TEST_TRIALS
        assert isinstance(x_values, list)
        assert isinstance(times, list)
        assert x_values[0] == 1  # x starts at 1

    def test_preprocess_memory_data(
        self,
        mock_results: ResultsType,
        mock_stats: StatsType,
        mock_config: BenchConfig,
    ) -> None:
        """Test memory data preprocessing."""
        formatter = LinePlotFormatter()
        sorted_methods = formatter.sort_keys(mock_stats, mock_config)

        memory_data = formatter._preprocess_memory_data(
            mock_results,
            sorted_methods,
            mock_config,
        )

        assert len(memory_data) == EXPECTED_DATA_LENGTH
        assert "test_method1" in memory_data
        assert "test_method2" in memory_data

        # Check data structure: (x_values, memory_values)
        x_values, memory_values = memory_data["test_method1"]
        assert len(x_values) == TEST_TRIALS
        assert len(memory_values) == TEST_TRIALS
        assert isinstance(x_values, list)
        assert isinstance(memory_values, list)

    def test_apply_styling(self, mock_config: BenchConfig) -> None:
        """Test styling application."""
        formatter = LinePlotFormatter(log_scale=True)

        # Create a mock axes object
        mock_ax = MagicMock()

        # Apply styling to the mock axes
        formatter._apply_styling(
            mock_ax,
            mock_config,
            title_suffix="Test Plot",
            unit="ms",
            show_legend=True,
            xlabel="Custom X Label",
        )

        # Verify styling was applied
        mock_ax.set_yscale.assert_called_once_with("log")
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Custom X Label")
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.legend.assert_called_once()

    def test_sns_theme_application(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that sns_theme parameters are correctly applied."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        custom_theme = {"style": "darkgrid", "palette": "Set2"}
        formatter = LinePlotFormatter(sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify that seaborn.set_theme was called with the custom theme
            mock_set_theme.assert_called_once_with(**custom_theme)
        plt.close()

    def test_sns_theme_forces_seaborn_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that providing sns_theme automatically sets engine to seaborn."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        # Create formatter with matplotlib engine but sns_theme provided
        custom_theme = {"style": "whitegrid"}
        formatter = LinePlotFormatter(engine="matplotlib", sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.axes.Axes.legend"),
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.lineplot") as mock_lineplot,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            # Mock seaborn.lineplot to return a mock axes
            mock_lineplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn functions were called (indicating seaborn engine was used)
            mock_set_theme.assert_called_once_with(**custom_theme)
            mock_lineplot.assert_called()
        plt.close()

    def test_sns_theme_with_default_theme(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that seaborn engine uses default theme when no sns_theme provided."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = LinePlotFormatter(engine="seaborn")

        with (
            mock.patch("matplotlib.axes.Axes.legend"),
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.lineplot") as mock_lineplot,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            # Mock seaborn.lineplot to return a mock axes
            mock_lineplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None

            mock_set_theme.assert_called_once_with(**DEFAULT_SNS_THEME)
        plt.close()

    def test_sns_theme_none_with_matplotlib_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that matplotlib engine doesn't call seaborn.set_theme."""
        formatter = LinePlotFormatter(engine="matplotlib", sns_theme=None)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn.set_theme was not called
            mock_set_theme.assert_not_called()
        plt.close()


class TestPlotReporter:
    """Tests for the PlotReporter class."""

    def test_init(self) -> None:
        """Test initialization with different parameters."""
        formatter = MagicMock()
        reporter = PlotReporter(
            formatter,
            show=True,
            save_path="test.png",
            dpi=CUSTOM_HIGH_DPI,
        )

        assert reporter.formatter == formatter
        assert reporter.show is True
        assert reporter.save_path == "test.png"
        assert reporter.dpi == CUSTOM_HIGH_DPI

    def test_send_with_savefig(self) -> None:
        """Test saving a figure to file."""
        mock_formatter = MagicMock()
        mock_figure = MagicMock(spec=matplotlib.figure.Figure)
        mock_formatter.format.return_value = mock_figure

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            reporter = PlotReporter(
                mock_formatter,
                show=False,
                save_path=temp_file.name,
                dpi=100,
            )

            with patch.object(mock_figure, "savefig") as mock_savefig:
                reporter.report({}, {}, BenchConfig())

                # Check that savefig was called with the correct parameters
                mock_savefig.assert_called_once_with(temp_file.name, dpi=100)

    def test_send_with_show(self) -> None:
        """Test showing a figure."""
        mock_formatter = MagicMock()
        mock_figure = MagicMock(spec=matplotlib.figure.Figure)
        mock_formatter.format.return_value = mock_figure

        reporter = PlotReporter(mock_formatter, show=True)

        with patch("matplotlib.pyplot.show") as mock_show:
            reporter.report({}, {}, BenchConfig())

            # Check that plt.show was called
            mock_show.assert_called_once()

    def test_send_type_error(self) -> None:
        """Test handling the case where formatter doesn't return a Figure."""
        mock_formatter = MagicMock()
        # Return something that's not a Figure
        mock_formatter.format.return_value = "Not a Figure"

        reporter = PlotReporter(mock_formatter)

        with pytest.raises(TypeError):
            reporter.report({}, {}, BenchConfig())


class TestBoxPlotFormatterTimeUnits:
    """Test BoxPlotFormatter handling of different time units from BenchConfig."""

    def test_boxplot_formatter_with_time_units(self) -> None:
        """Test BoxPlotFormatter with different time units."""
        # Skip if matplotlib is not available
        pytest.importorskip("matplotlib")

        time_units = ["s", "ms", "μs", "us", "ns", "m"]
        # Create sample results and stats
        results: dict[str, ResultType] = {"test_func": {"times": [1.0, 2.0, 3.0]}}
        stats = {"test_func": complete_stat({"avg": 2.0, "min": 1.0, "max": 3.0})}

        for time_unit in time_units:
            config = BenchConfig(time=time_unit)

            # Test BoxPlotFormatter
            boxplot_formatter = BoxPlotFormatter()
            figure = boxplot_formatter.format(results, stats, config)

            # Check the label on the axis
            display_unit = "μs" if time_unit == "us" else time_unit
            assert hasattr(figure, "axes"), "Figure has no axes attribute"
            assert figure.axes, "Figure has no axes"

            axis = figure.axes[0]
            time_label = f"Time ({display_unit})"

            # The label might be on x-axis or y-axis depending on orientation
            assert (
                time_label in axis.get_xlabel() or time_label in axis.get_ylabel()
            ), f"Time unit {display_unit} not found in plot labels"


class TestViolinPlotFormatter:
    """Tests for the ViolinPlotFormatter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = ViolinPlotFormatter()
        assert formatter.log_scale is False
        assert formatter.figsize == (10, 6)
        assert formatter.engine == "matplotlib"
        assert formatter.orientation == "horizontal"
        assert formatter.plot_type == "violin"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = ViolinPlotFormatter(
            log_scale=True,
            figsize=(8, 4),
            engine="seaborn",
            orientation="vertical",
        )
        assert formatter.log_scale is True
        assert formatter.figsize == (8, 4)
        assert formatter.engine == "seaborn"
        assert formatter.orientation == "vertical"
        assert formatter.plot_type == "violin"

    def test_preprocess_data_no_outlier_handling(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test data preprocessing with no outlier handling."""
        formatter = ViolinPlotFormatter()
        data, labels = formatter._preprocess_data(
            sample_results,
            sample_stats,
            sample_config,
        )

        assert len(data) == NUM_TEST_FUNCTIONS
        assert "test_func1" in data
        assert "test_func2" in data
        assert data["test_func1"] == TEST_TIME_VALUES
        assert len(labels) == NUM_TEST_FUNCTIONS
        assert "test_func1" in labels
        assert "test_func2" in labels

    @pytest.mark.parametrize("engine", ["matplotlib", "seaborn"])
    def test_format_with_different_engines(
        self,
        engine: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different plotting engines."""
        # Skip test if seaborn not installed when testing seaborn engine
        if engine == "seaborn":
            if not find_spec("seaborn"):
                pytest.skip("seaborn not installed")

            warnings.catch_warnings()
            warnings.simplefilter("ignore", PendingDeprecationWarning)

        formatter = ViolinPlotFormatter(engine=engine)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0
        plt.close()

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_format_with_different_orientations(
        self,
        orientation: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different orientations."""
        formatter = ViolinPlotFormatter(orientation=orientation)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0
        plt.close()

    def test_create_matplotlib_violinplot(self) -> None:
        """Test creation of violinplot with matplotlib engine."""
        formatter = ViolinPlotFormatter()

        # Mock a matplotlib Axes object
        with mock.patch("matplotlib.pyplot.Axes") as mock_axes:
            mock_axes.violinplot = mock.MagicMock()
            formatter._create_matplotlib_violinplot(
                mock_axes,
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                ["test1", "test2"],
            )

            # Verify violinplot was called
            mock_axes.violinplot.assert_called_once()

            # Check that set_yticks was called for horizontal orientation
            mock_axes.set_yticks.assert_called_once()
            mock_axes.set_yticklabels.assert_called_once()

    def test_create_seaborn_violinplot(self) -> None:
        """Test creation of violinplot with seaborn engine."""
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = ViolinPlotFormatter(engine="seaborn")

        with mock.patch("seaborn.violinplot") as mock_violinplot:
            # Mock a matplotlib Axes object
            mock_axes = mock.MagicMock()

            formatter._create_seaborn_violinplot(
                mock_axes,
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                ["test1", "test2"],
            )

            # Verify seaborn.violinplot was called
            mock_violinplot.assert_called_once()

    def test_create_matplotlib_violinplot_horizontal_fallback(self) -> None:
        """
        Test fallback to vert parameter when orientation isn't supported.

        Tests the horizontal orientation case.
        """
        formatter = ViolinPlotFormatter(orientation="horizontal")

        # Create mock Axes
        mock_axes = mock.MagicMock()

        # Make the first violinplot call raise TypeError to simulate orientation
        # not being supported
        mock_axes.violinplot.side_effect = [
            TypeError("orientation not supported"),
            None,
        ]

        formatter._create_matplotlib_violinplot(
            mock_axes,
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            ["test1", "test2"],
        )

        # Check that violinplot was called twice
        # (first with orientation, then with vert)
        assert mock_axes.violinplot.call_count == EXPECTED_CALL_COUNT

        # Second call should have used vert
        args, kwargs = mock_axes.violinplot.call_args_list[1]
        assert "vert" in kwargs
        assert kwargs["vert"] is False  # horizontal means vert=False

        # Ensure tick settings were called
        mock_axes.set_yticks.assert_called_once()
        mock_axes.set_yticklabels.assert_called_once()

    def test_create_matplotlib_violinplot_vertical_fallback(self) -> None:
        """
        Test fallback to vert parameter when orientation isn't supported.

        Tests the vertical orientation case.
        """
        formatter = ViolinPlotFormatter(orientation="vertical")

        # Create mock Axes
        mock_axes = mock.MagicMock()

        # Make the first violinplot call raise TypeError to simulate orientation
        # not being supported
        mock_axes.violinplot.side_effect = [
            TypeError("orientation not supported"),
            None,
        ]

        formatter._create_matplotlib_violinplot(
            mock_axes,
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            ["test1", "test2"],
        )

        # Check that violinplot was called twice
        # (first with orientation, then with vert)
        assert mock_axes.violinplot.call_count == EXPECTED_CALL_COUNT

        # Second call should have used vert
        args, kwargs = mock_axes.violinplot.call_args_list[1]
        assert "vert" in kwargs
        assert kwargs["vert"] is True  # vertical means vert=True

        # Ensure tick settings were called
        mock_axes.set_xticks.assert_called_once()
        mock_axes.set_xticklabels.assert_called_once()

    def test_format_with_memory_enabled(
        self,
    ) -> None:
        """Test formatting with memory metrics enabled."""
        # Modify sample results and stats to include memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
            "test_func2": {
                "times": [TEST_SLOWER_TIME, TEST_TIME_VALUE, TEST_SLOW_TIME],
                "memory": [2048, 1024, 3072],
            },
        }

        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
            "test_func2": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
        }

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=3, memory=True)

        formatter = ViolinPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch.object(plt, "tight_layout"),
        ):
            # Create mock axes for time and memory subplots
            mock_ax_time = mock.MagicMock()
            mock_ax_mem = mock.MagicMock()
            mock_fig = mock.MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax_time, mock_ax_mem))

            # Call format with memory enabled
            fig = formatter.format(memory_results, memory_stats, memory_config)

            # Verify subplots were created (2 plots for time and memory)
            mock_subplots.assert_called_once()
            args, kwargs = mock_subplots.call_args
            assert args == (2, 1)  # 2 rows, 1 column

            # Verify violinplot was called for both time and memory plots
            assert mock_ax_time.violinplot.call_count == 1
            assert mock_ax_mem.violinplot.call_count == 1

            # Verify memory subplot had styling applied
            mock_ax_mem.set_title.assert_called_once()

            assert fig is not None
        plt.close()

    def test_sns_theme_application(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that sns_theme parameters are correctly applied."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        custom_theme = {"style": "darkgrid", "palette": "Set2"}
        formatter = ViolinPlotFormatter(sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify that seaborn.set_theme was called with the custom theme
            mock_set_theme.assert_called_once_with(**custom_theme)
        plt.close()

    def test_sns_theme_forces_seaborn_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that providing sns_theme automatically sets engine to seaborn."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        # Create formatter with matplotlib engine but sns_theme provided
        custom_theme = {"style": "whitegrid"}
        formatter = ViolinPlotFormatter(engine="matplotlib", sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.violinplot") as mock_violinplot,
        ):
            # Mock seaborn.violinplot to return a mock axes
            mock_violinplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn functions were called (indicating seaborn engine was used)
            mock_set_theme.assert_called_once_with(**custom_theme)
            mock_violinplot.assert_called()
        plt.close()

    def test_sns_theme_with_default_theme(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that seaborn engine uses default theme when no sns_theme provided."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = ViolinPlotFormatter(engine="seaborn")

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            mock.patch("seaborn.violinplot") as mock_violinplot,
        ):
            # Mock seaborn.violinplot to return a mock axes
            mock_violinplot.return_value = mock.MagicMock()

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None

            mock_set_theme.assert_called_once_with(**DEFAULT_SNS_THEME)
        plt.close()

    def test_sns_theme_none_with_matplotlib_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that matplotlib engine doesn't call seaborn.set_theme."""
        formatter = ViolinPlotFormatter(engine="matplotlib", sns_theme=None)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn.set_theme was not called
            mock_set_theme.assert_not_called()
        plt.close()


class TestBoxPlotFormatterTimeDisabled:
    """Tests for BoxPlotFormatter with time measurement disabled."""

    def test_format_time_disabled_memory_enabled(
        self,
        sample_stats: dict[str, StatType],
    ) -> None:
        """Test that formatter works correctly with time=False and memory=True."""
        # Modify sample results and config for this test
        results: ResultsType = {
            "test_func1": {"memory": [1024, 2048, 3072]},
            "test_func2": {"memory": [2048, 3072, 4096]},
        }
        config = BenchConfig(trials=3, time=False, memory=True)

        formatter = BoxPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, sample_stats, config)

            # Figure should be created
            assert isinstance(fig, plt.Figure)

            # Check that the figure has memory plot but not time plot
            for ax in fig.axes:
                if ax.get_ylabel():
                    assert "Memory" in ax.get_ylabel()
                    assert "Time" not in ax.get_ylabel()
                if ax.get_xlabel():
                    assert "Time" not in ax.get_xlabel()

        plt.close()


class TestLinePlotFormatterTimeDisabled:
    """Tests for LinePlotFormatter with time measurement disabled."""

    def test_format_time_disabled_memory_enabled(
        self,
        sample_stats: dict[str, StatType],
    ) -> None:
        """Test that formatter works correctly with time=False and memory=True."""
        # Modify sample results for this test
        results: ResultsType = {
            "test_func1": {"memory": [1024, 2048, 3072]},
            "test_func2": {"memory": [2048, 3072, 4096]},
        }
        config = BenchConfig(trials=3, time=False, memory=True)

        formatter = LinePlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, sample_stats, config)

            # Figure should be created
            assert isinstance(fig, plt.Figure)

            # Check that the figure has memory plot but not time plot
            for ax in fig.axes:
                if ax.get_ylabel():
                    assert "Memory" in ax.get_ylabel()
                    assert "Time" not in ax.get_ylabel()
                if ax.get_xlabel():
                    assert "Time" not in ax.get_xlabel()

        plt.close()


class TestViolinPlotFormatterTimeDisabled:
    """Tests for ViolinPlotFormatter with time measurement disabled."""

    def test_format_time_disabled_memory_enabled(
        self,
        sample_stats: dict[str, StatType],
    ) -> None:
        """Test that formatter works correctly with time=False and memory=True."""
        # Modify sample results for this test
        results: ResultsType = {
            "test_func1": {"memory": [1024, 2048, 3072]},
            "test_func2": {"memory": [2048, 3072, 4096]},
        }
        config = BenchConfig(trials=3, time=False, memory=True)

        formatter = ViolinPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, sample_stats, config)

            # Figure should be created
            assert isinstance(fig, plt.Figure)

            # Check that the figure has memory plot but not time plot
            for ax in fig.axes:
                if ax.get_ylabel():
                    assert "Memory" in ax.get_ylabel()
                    assert "Time" not in ax.get_ylabel()
                if ax.get_xlabel():
                    assert "Time" not in ax.get_xlabel()

        plt.close()


class TestViolinPlotFormatterTimeUnits:
    """Test ViolinPlotFormatter handling of different time units from BenchConfig."""

    def test_violinplot_formatter_with_time_units(self) -> None:
        """Test ViolinPlotFormatter with different time units."""
        # Skip if matplotlib is not available
        pytest.importorskip("matplotlib")

        time_units = ["s", "ms", "μs", "us", "ns", "m"]
        # Create sample results and stats
        results: dict[str, ResultType] = {"test_func": {"times": [1.0, 2.0, 3.0]}}
        stats = {"test_func": complete_stat({"avg": 2.0, "min": 1.0, "max": 3.0})}

        for time_unit in time_units:
            config = BenchConfig(time=time_unit)

            # Test ViolinPlotFormatter
            violinplot_formatter = ViolinPlotFormatter()
            figure = violinplot_formatter.format(results, stats, config)

            # Check the label on the axis
            display_unit = "μs" if time_unit == "us" else time_unit
            assert hasattr(figure, "axes"), "Figure has no axes attribute"
            assert figure.axes, "Figure has no axes"

            axis = figure.axes[0]
            time_label = f"Time ({display_unit})"

            # The label might be on x-axis or y-axis depending on orientation
            assert (
                time_label in axis.get_xlabel() or time_label in axis.get_ylabel()
            ), f"Time unit {display_unit} not found in plot labels"


class TestHistPlotFormatter:
    """Tests for the HistPlotFormatter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = HistPlotFormatter()
        assert formatter.log_scale is False
        assert formatter.figsize == (10, 6)
        assert formatter.bins == "auto"
        assert formatter.engine == "matplotlib"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        alpha = 0.7
        bins = 20
        formatter = HistPlotFormatter(
            bins=bins,
            log_scale=True,
            figsize=(8, 4),
            engine="seaborn",
            alpha=alpha,
            color="blue",
        )
        assert formatter.bins == bins
        assert formatter.log_scale is True
        assert formatter.figsize == (8, 4)
        assert formatter.engine == "seaborn"
        assert formatter.hist_kwargs["alpha"] == alpha
        assert formatter.hist_kwargs["color"] == "blue"

    def test_preprocess_data(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test data preprocessing."""
        formatter = HistPlotFormatter()
        sorted_methods = formatter.sort_keys(sample_stats, sample_config)

        time_data = formatter._preprocess_data(
            sample_results,
            sorted_methods,
            sample_config,
        )

        assert len(time_data) == NUM_TEST_FUNCTIONS
        assert "test_func1" in time_data
        assert "test_func2" in time_data
        assert time_data["test_func1"] == TEST_TIME_VALUES
        assert time_data["test_func2"] == [
            TEST_SLOWER_TIME,
            TEST_TIME_VALUE,
            TEST_SLOW_TIME,
        ]

    @pytest.mark.parametrize("engine", ["matplotlib", "seaborn"])
    def test_format_with_different_engines(
        self,
        engine: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different plotting engines."""
        # Skip test if seaborn not installed when testing seaborn engine
        if engine == "seaborn":
            if not find_spec("seaborn"):
                pytest.skip("seaborn not installed")

            warnings.catch_warnings()
            warnings.simplefilter("ignore", PendingDeprecationWarning)

        formatter = HistPlotFormatter(engine=engine)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0
        plt.close()

    def test_format_with_memory_enabled(
        self,
    ) -> None:
        """Test formatting with memory metrics enabled."""
        # Modify sample results and stats to include memory data
        memory_results: dict[str, ResultType] = {
            "test_func1": {
                "times": [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME],
                "memory": [1024, 2048, 3072],
            },
            "test_func2": {
                "times": [TEST_SLOWER_TIME, TEST_TIME_VALUE, TEST_SLOW_TIME],
                "memory": [2048, 1024, 3072],
            },
        }

        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
            "test_func2": complete_stat(
                {
                    "avg": TEST_AVG_TIME,
                    "min": TEST_TIME_VALUE,
                    "max": TEST_SLOWER_TIME,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
        }

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=3, memory=True)

        formatter = HistPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch.object(plt, "tight_layout"),
        ):
            # Create mock axes for time and memory subplots
            mock_ax_time = mock.MagicMock()
            mock_ax_mem = mock.MagicMock()
            mock_fig = mock.MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax_time, mock_ax_mem))

            # Call format with memory enabled
            fig = formatter.format(memory_results, memory_stats, memory_config)

            # Verify subplots were created (2 plots for time and memory)
            mock_subplots.assert_called_once()
            args, kwargs = mock_subplots.call_args
            assert args == (2, 1)  # 2 rows, 1 column

            assert fig is not None
        plt.close()

    def test_create_time_histogram(self) -> None:
        """Test histogram creation."""
        bins = 10
        formatter = HistPlotFormatter(bins=bins)

        # Mock a matplotlib Axes object
        mock_ax = mock.MagicMock()

        # Prepare test data
        time_data = {
            "test_func1": [0.1, 0.2, 0.3],
            "test_func2": [0.2, 0.3, 0.4],
        }

        methods = ["test_func1", "test_func2"]
        formatter._create_time_histogram(
            mock_ax,
            time_data,
            methods,
        )

        # Verify hist was called for each function
        assert mock_ax.hist.call_count == len(methods)

        # Check bins parameter was passed
        assert len(mock_ax.hist.call_args_list) == len(methods)
        for call_args in mock_ax.hist.call_args_list:
            _, kwargs = call_args
            assert kwargs["bins"] == bins

    def test_sns_theme_application(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that sns_theme parameters are correctly applied."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        custom_theme = {"style": "darkgrid", "palette": "Set2"}
        formatter = HistPlotFormatter(sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify that seaborn.set_theme was called with the custom theme
            mock_set_theme.assert_called_once_with(**custom_theme)
        plt.close()

    def test_sns_theme_forces_seaborn_engine(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that providing sns_theme automatically sets engine to seaborn."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        # Create formatter with matplotlib engine but sns_theme provided
        custom_theme = {"style": "whitegrid"}
        formatter = HistPlotFormatter(engine="matplotlib", sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify seaborn.set_theme was called
            mock_set_theme.assert_called_once_with(**custom_theme)
        plt.close()

    def test_log_scale_application(self) -> None:
        """Test that log_scale parameter is correctly applied to axes."""
        formatter = HistPlotFormatter(log_scale=True)

        # Create a mock axis
        mock_ax = mock.MagicMock()

        # Apply styling to the mock axis
        formatter._apply_styling(mock_ax, BenchConfig(trials=10))

        # Verify the x-axis scale was set to "log"
        mock_ax.set_xscale.assert_called_once_with("log")


class TestHistPlotFormatterTimeDisabled:
    """Tests for HistPlotFormatter with time measurement disabled."""

    def test_format_time_disabled_memory_enabled(
        self,
        sample_stats: dict[str, StatType],
    ) -> None:
        """Test that formatter works correctly with time=False and memory=True."""
        # Modify sample results for this test
        results: ResultsType = {
            "test_func1": {"memory": [1024, 2048, 3072]},
            "test_func2": {"memory": [2048, 3072, 4096]},
        }
        config = BenchConfig(trials=3, time=False, memory=True)

        formatter = HistPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, sample_stats, config)

            # Figure should be created
            assert isinstance(fig, plt.Figure)

            # Check that the figure has memory plot but not time plot
            for ax in fig.axes:
                if ax.get_xlabel():
                    assert "Memory" in ax.get_xlabel()
                    assert "Time" not in ax.get_xlabel()
                if ax.get_ylabel():
                    assert "Time" not in ax.get_ylabel()

        plt.close()


class TestHistPlotFormatterTimeUnits:
    """Test HistPlotFormatter handling of different time units from BenchConfig."""

    def test_histplot_formatter_with_time_units(self) -> None:
        """Test HistPlotFormatter with different time units."""
        # Skip if matplotlib is not available
        pytest.importorskip("matplotlib")

        time_units = ["s", "ms", "μs", "us", "ns", "m"]
        # Create sample results and stats
        results: dict[str, ResultType] = {"test_func": {"times": [1.0, 2.0, 3.0]}}
        stats = {"test_func": complete_stat({"avg": 2.0, "min": 1.0, "max": 3.0})}

        for time_unit in time_units:
            config = BenchConfig(time=time_unit)

            # Test HistPlotFormatter
            hist_formatter = HistPlotFormatter()
            figure = hist_formatter.format(results, stats, config)

            # Check the label on the axis
            display_unit = "μs" if time_unit == "us" else time_unit
            assert hasattr(figure, "axes"), "Figure has no axes attribute"
            assert figure.axes, "Figure has no axes"

            axis = figure.axes[0]
            time_label = f"Time ({display_unit})"

            # Time label should be on x-axis or y-axis
            assert (
                time_label in axis.get_xlabel() or time_label in axis.get_ylabel()
            ), f"Time unit {display_unit} not found in plot labels"


class TestBarPlotFormatter:
    """Tests for the BarPlotFormatter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = BarPlotFormatter()
        assert formatter.metric is None
        assert formatter.log_scale is False
        assert formatter.figsize == (10, 6)
        assert formatter.engine == "matplotlib"
        assert formatter.orientation == "horizontal"
        plt.close()

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = BarPlotFormatter(
            metric="min",
            log_scale=True,
            figsize=(8, 4),
            engine="seaborn",
            orientation="vertical",
        )
        assert formatter.metric == "min"
        assert formatter.log_scale is True
        assert formatter.figsize == (8, 4)
        assert formatter.engine == "seaborn"
        assert formatter.orientation == "vertical"
        plt.close()

    def test_format_with_single_metric(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with a single metric."""
        formatter = BarPlotFormatter(metric="avg")

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) == 1  # Only one subplot for one metric

        plt.close()

    def test_format_with_multiple_metrics(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with multiple metrics."""
        metrics: list[MetricType] = ["min", "avg", "max"]
        formatter = BarPlotFormatter(metric=metrics)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) == len(metrics)  # One subplot per metric

        plt.close()

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_format_with_different_orientations(
        self,
        orientation: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different orientations."""
        formatter = BarPlotFormatter(orientation=orientation)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0

        plt.close()

    @pytest.mark.parametrize("engine", ["matplotlib", "seaborn"])
    def test_format_with_different_engines(
        self,
        engine: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different plotting engines."""
        # Skip test if seaborn not installed when testing seaborn engine
        if engine == "seaborn":
            if not find_spec("seaborn"):
                pytest.skip("seaborn not installed")

            warnings.catch_warnings()
            warnings.simplefilter("ignore", PendingDeprecationWarning)

        formatter = BarPlotFormatter(engine=engine)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0

        plt.close()

    def test_log_scale_application(self) -> None:
        """Test that log_scale parameter is correctly applied to axes."""
        formatter = BarPlotFormatter(log_scale=True)

        # Create a mock axis
        mock_ax = mock.MagicMock()

        # Mock labels and prepare for the styling function
        labels = ["test1", "test2"]
        config = BenchConfig(trials=3)
        metric: MetricType = "avg"

        # Apply styling to the mock axis
        formatter._apply_styling(mock_ax, labels, metric, config)

        # Check that appropriate scale setting was called based on orientation
        if formatter.orientation == "horizontal":
            mock_ax.set_xscale.assert_called_once_with("log")
        else:
            mock_ax.set_yscale.assert_called_once_with("log")

        plt.close()

    def test_format_with_memory_enabled(
        self,
    ) -> None:
        """Test formatting with memory metrics enabled."""
        # Modify sample stats to include memory metrics
        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg": 0.001,
                    "min": 0.0008,
                    "max": 0.0012,
                    "avg_memory": 2048,
                    "max_memory": 3072,
                },
                memory=True,
            ),
            "test_func2": complete_stat(
                {
                    "avg": 0.002,
                    "min": 0.0018,
                    "max": 0.0022,
                    "avg_memory": 1024,
                    "max_memory": 2048,
                },
                memory=True,
            ),
        }

        # Create empty results (not actually used by BarPlotFormatter)
        results: ResultsType = {}

        # Create a config with memory enabled
        memory_config = BenchConfig(trials=3, memory=True)

        # Test with memory metric
        formatter = BarPlotFormatter(metric="avg_memory")

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, memory_stats, memory_config)
            assert fig is not None

            # Check that we have memory metrics in the plot
            for ax in fig.axes:
                if ax.get_xlabel():
                    if formatter.orientation == "horizontal":
                        assert "Memory" in ax.get_xlabel()
                    else:
                        assert "Memory" in ax.get_ylabel()

        plt.close()

    def test_format_time_disabled_memory_enabled(
        self,
    ) -> None:
        """Test that formatter works correctly with time=False and memory=True."""
        # Modify sample stats for this test
        memory_stats = {
            "test_func1": complete_stat(
                {
                    "avg_memory": 1024,
                    "max_memory": 2048,
                },
                memory=True,
                time=False,
            ),
            "test_func2": complete_stat(
                {
                    "avg_memory": 2048,
                    "max_memory": 4096,
                },
                memory=True,
                time=False,
            ),
        }

        # Create empty results (not used by BarPlotFormatter)
        results: ResultsType = {}

        # Create config with time disabled and memory enabled
        config = BenchConfig(trials=3, time=False, memory=True)

        formatter = BarPlotFormatter()

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(results, memory_stats, config)

            # Figure should be created
            assert isinstance(fig, plt.Figure)

            # Check that the figure has memory metrics but not time metrics
            for ax in fig.axes:
                if formatter.orientation == "horizontal":
                    if ax.get_xlabel():
                        assert "Memory" in ax.get_xlabel()
                        assert "Time" not in ax.get_xlabel()
                elif ax.get_ylabel():
                    assert "Memory" in ax.get_ylabel()
                    assert "Time" not in ax.get_ylabel()

        plt.close()

    def test_sns_theme_application(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test that sns_theme parameters are correctly applied."""
        # Skip if seaborn is not available
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        custom_theme = {"style": "darkgrid", "palette": "Set2"}
        formatter = BarPlotFormatter(sns_theme=custom_theme)

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("seaborn.set_theme") as mock_set_theme,
            warnings.catch_warnings(),
        ):
            # Ignore seaborn's PendingDeprecationWarning
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="seaborn",
            )

            fig = formatter.format(sample_results, sample_stats, sample_config)

            assert fig is not None
            # Verify that seaborn.set_theme was called with the custom theme
            mock_set_theme.assert_called_once_with(**custom_theme)

        plt.close()

    def test_create_matplotlib_bar_plot(self) -> None:
        """Test creation of bar plot with matplotlib engine."""
        formatter = BarPlotFormatter()

        # Mock a matplotlib Axes object
        mock_ax = mock.MagicMock()

        # Call the method with test values
        values = [10.0, 20.0]
        formatter._create_matplotlib_bar_plot(mock_ax, values)

        # Verify appropriate method was called based on orientation
        if formatter.orientation == "horizontal":
            assert mock_ax.barh.call_count == len(values)
        else:
            assert mock_ax.bar.call_count == len(values)

        plt.close()

    def test_create_seaborn_bar_plot(self) -> None:
        """Test creation of bar plot with seaborn engine."""
        if not find_spec("seaborn"):
            pytest.skip("seaborn not installed")

        formatter = BarPlotFormatter(engine="seaborn")

        with mock.patch("seaborn.barplot") as mock_barplot:
            # Mock a matplotlib Axes object
            mock_ax = mock.MagicMock()

            values = [10.0, 20.0]
            formatter._create_seaborn_bar_plot(mock_ax, values)

            # Verify seaborn.barplot was called the right number of times
            assert mock_barplot.call_count == len(values)

        plt.close()

    def test_create_bar_plot_for_metric(
        self,
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test creation of bar plot for a specific metric."""
        formatter = BarPlotFormatter()

        # Mock matplotlib Axes
        mock_ax = mock.MagicMock()

        # Set up methods to be tested
        with (
            mock.patch.object(formatter, "_create_matplotlib_bar_plot") as mock_plot,
            mock.patch.object(formatter, "_apply_styling") as mock_styling,
        ):
            formatter._create_bar_plot_for_metric(
                mock_ax,
                "avg",
                ["test_func1", "test_func2"],
                sample_stats,
                sample_config,
            )

            # Verify plotting and styling methods were called
            mock_plot.assert_called_once()
            mock_styling.assert_called_once()

        plt.close()

    def test_set_title(self) -> None:
        """Test setting plot title based on metric type."""
        formatter = BarPlotFormatter()

        # Mock a matplotlib Axes object
        mock_ax = mock.MagicMock()

        # Test with different metrics
        metrics: list[MetricType] = ["avg", "min", "max", "avg_memory", "max_memory"]

        for metric in metrics:
            formatter._set_title(mock_ax, metric, BenchConfig(trials=5))

            # Verify set_title was called with a string containing
            # the appropriate metric name
            mock_ax.set_title.assert_called()
            title = mock_ax.set_title.call_args[0][0]

            if metric == "avg":
                assert "Average Time" in title
            elif metric == "min":
                assert "Minimum Time" in title
            elif metric == "max":
                assert "Maximum Time" in title
            elif metric == "avg_memory":
                assert "Average Memory Usage" in title
            elif metric == "max_memory":
                assert "Maximum Memory Usage" in title

            assert "5 trials" in title  # Check trials count is included

            # Reset mock for next iteration
            mock_ax.reset_mock()

        plt.close()

    def test_set_axis_label_time(self) -> None:
        """Test setting axis label for time metrics."""
        formatter = BarPlotFormatter()

        # Mock a matplotlib Axes object
        mock_ax = mock.MagicMock()

        # Test with time metric
        formatter._set_axis_label(mock_ax, "avg", BenchConfig(time="s"), "x")

        # Verify axis label contains time unit
        if formatter.orientation == "horizontal":
            mock_ax.set_xlabel.assert_called_once()
            label = mock_ax.set_xlabel.call_args[0][0]
            assert "Time" in label
            assert "s" in label

        plt.close()

    def test_set_axis_label_memory(self) -> None:
        """Test setting axis label for memory metrics."""
        formatter = BarPlotFormatter()

        # Mock a matplotlib Axes object
        mock_ax = mock.MagicMock()

        # Test with memory metric
        formatter._set_axis_label(mock_ax, "avg_memory", BenchConfig(memory="KB"), "y")

        # Verify axis label contains memory unit
        if formatter.orientation == "vertical":
            mock_ax.set_ylabel.assert_called_once()
            label = mock_ax.set_ylabel.call_args[0][0]
            assert "Memory" in label
            assert "KB" in label

        plt.close()
