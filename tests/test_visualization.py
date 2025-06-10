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
        assert formatter.y_limit is None
        assert formatter.figsize == (10, 6)
        assert formatter.engine == "matplotlib"
        assert formatter.orientation == "horizontal"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        formatter = BoxplotFormatter(
            showfliers=False,
            log_scale=True,
            y_limit=(0.0, 1.0),
            trim_outliers=DEFAULT_TRIM_OUTLIERS,
            winsorize_outliers=None,
            figsize=(8, 4),
            label_rotation_threshold=DEFAULT_LABEL_ROTATION_THRESHOLD,
            engine="seaborn",
            orientation="vertical",
        )
        assert formatter.showfliers is False
        assert formatter.log_scale is True
        assert formatter.y_limit == (0.0, 1.0)
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
