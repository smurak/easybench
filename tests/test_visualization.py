"""
Tests for the visualization module.

This module tests visualization-related classes including:
- PlotFormatter
- BoxplotFormatter
- PlotReporter

Tests require matplotlib and optionally seaborn.
"""

import warnings
from importlib.util import find_spec
from unittest import mock

import matplotlib.pyplot as plt
import pytest

from easybench.core import BenchConfig, ResultType, StatType
from easybench.visualization import BoxplotFormatter, PlotFormatter, PlotReporter

# Constants for test values
TEST_TIME_VALUE = 0.1
TEST_SLOW_TIME = 0.2
TEST_SLOWER_TIME = 0.3
TEST_AVG_TIME = 0.2
TEST_TIME_VALUES = [TEST_TIME_VALUE, TEST_SLOW_TIME, TEST_SLOWER_TIME]

# Constants for test configurations
DEFAULT_TRIM_OUTLIERS = 0.1
DEFAULT_LABEL_ROTATION_THRESHOLD = 5
NUM_TEST_FUNCTIONS = 2
DEFAULT_DPI = 100
CUSTOM_DPI = 200
EXPECTED_CALL_COUNT = 2  # Number of times boxplot is called during fallback tests


def complete_stat(
    dic: dict[str, float],
    memory: bool = False,  # noqa: FBT002, FBT001
) -> StatType:
    """Complete dictionaries for StatType."""
    stat: StatType = {
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
    }
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


class TestBoxplotFormatter:
    """Tests for the BoxplotFormatter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = BoxplotFormatter()
        assert formatter.showfliers is True
        assert formatter.log_scale is False
        assert formatter.data_limit is None
        assert formatter.figsize == (10, 6)
        assert formatter.engine == "matplotlib"
        assert formatter.orientation == "horizontal"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = BoxplotFormatter(
            showfliers=False,
            log_scale=True,
            data_limit=(0.0, 1.0),
            trim_outliers=DEFAULT_TRIM_OUTLIERS,
            winsorize_outliers=None,
            figsize=(8, 4),
            label_rotation_threshold=DEFAULT_LABEL_ROTATION_THRESHOLD,
            engine="seaborn",
            orientation="vertical",
        )
        assert formatter.showfliers is False
        assert formatter.log_scale is True
        assert formatter.data_limit == (0.0, 1.0)
        assert formatter.trim_outliers == DEFAULT_TRIM_OUTLIERS
        assert formatter.winsorize_outliers is None
        assert formatter.figsize == (8, 4)
        assert formatter.label_rotation_threshold == DEFAULT_LABEL_ROTATION_THRESHOLD
        assert formatter.engine == "seaborn"
        assert formatter.orientation == "vertical"

    def test_preprocess_data_no_outlier_handling(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test data preprocessing with no outlier handling."""
        formatter = BoxplotFormatter()
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

        formatter = BoxplotFormatter(engine=engine)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_format_with_different_orientations(
        self,
        orientation: str,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test formatting with different orientations."""
        formatter = BoxplotFormatter(orientation=orientation)  # type: ignore [arg-type]

        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None
            assert len(fig.axes) > 0

    def test_trim_outliers(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test outlier trimming functionality."""
        if not find_spec("numpy"):
            pytest.skip("numpy not installed")

        formatter = BoxplotFormatter(trim_outliers=DEFAULT_TRIM_OUTLIERS)
        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None

    def test_winsorize_outliers(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test outlier winsorization functionality."""
        if not find_spec("numpy"):
            pytest.skip("numpy not installed")

        formatter = BoxplotFormatter(winsorize_outliers=DEFAULT_TRIM_OUTLIERS)
        with (
            mock.patch("matplotlib.pyplot.show"),
            mock.patch("matplotlib.pyplot.savefig"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            fig = formatter.format(sample_results, sample_stats, sample_config)
            assert fig is not None

    def test_create_matplotlib_boxplot(self) -> None:
        """Test creation of boxplot with matplotlib engine."""
        formatter = BoxplotFormatter()

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

        formatter = BoxplotFormatter(engine="seaborn")

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
        formatter = BoxplotFormatter(orientation="horizontal")

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
        formatter = BoxplotFormatter(orientation="vertical")

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

        formatter = BoxplotFormatter()

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

        formatter = BoxplotFormatter()
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

        formatter = BoxplotFormatter()

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

        formatter = BoxplotFormatter()
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


class TestPlotReporter:
    """Tests for the PlotReporter class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        formatter = mock.MagicMock(spec=PlotFormatter)
        reporter = PlotReporter(formatter)

        assert reporter.show is True
        assert reporter.save_path is None
        assert reporter.dpi == DEFAULT_DPI

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = mock.MagicMock(spec=PlotFormatter)
        reporter = PlotReporter(
            formatter,
            show=False,
            save_path="plot.png",
            dpi=CUSTOM_DPI,
        )

        assert reporter.show is False
        assert reporter.save_path == "plot.png"
        assert reporter.dpi == CUSTOM_DPI

    def test_send_with_figure(self) -> None:
        """Test _send method with a matplotlib Figure."""
        from matplotlib.figure import Figure

        formatter = mock.MagicMock(spec=PlotFormatter)
        reporter = PlotReporter(formatter, show=True)

        # Create a mock Figure
        mock_fig = mock.MagicMock(spec=Figure)

        # Mock plt.show and plt.close to avoid actually showing a plot
        with mock.patch("matplotlib.pyplot.show") as mock_show:
            reporter._send(mock_fig)
            mock_show.assert_called_once()

    def test_send_with_non_figure(self) -> None:
        """Test _send method with a non-matplotlib Figure object."""
        formatter = mock.MagicMock(spec=PlotFormatter)
        reporter = PlotReporter(formatter)

        with pytest.raises(TypeError, match="requires a matplotlib Figure object"):
            reporter._send("not a figure")

    def test_send_with_save_path(self) -> None:
        """Test _send method with save_path."""
        from matplotlib.figure import Figure

        formatter = mock.MagicMock(spec=PlotFormatter)
        reporter = PlotReporter(
            formatter,
            show=False,
            save_path="test_plot.png",
            dpi=100,
        )

        # Create a mock Figure
        mock_fig = mock.MagicMock(spec=Figure)

        # Mock savefig to avoid actually saving a file
        with mock.patch.object(mock_fig, "savefig") as mock_savefig:
            reporter._send(mock_fig)
            mock_savefig.assert_called_once_with("test_plot.png", dpi=100)

    def test_report(
        self,
        sample_results: dict[str, ResultType],
        sample_stats: dict[str, StatType],
        sample_config: BenchConfig,
    ) -> None:
        """Test report method."""
        from matplotlib.figure import Figure

        # Create a mock formatter that returns a mock Figure
        mock_fig = mock.MagicMock(spec=Figure)
        formatter = mock.MagicMock(spec=PlotFormatter)
        formatter.format.return_value = mock_fig

        reporter = PlotReporter(formatter, show=False)

        # Mock _send method to avoid actually doing anything
        with mock.patch.object(reporter, "_send") as mock_send:
            reporter.report(sample_results, sample_stats, sample_config)
            mock_send.assert_called_once_with(mock_fig)

            # Verify formatter.format was called with right arguments
            formatter.format.assert_called_once_with(
                results=sample_results,
                stats=sample_stats,
                config=sample_config,
            )
