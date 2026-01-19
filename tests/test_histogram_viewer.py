import pytest
import hist
import numpy as np
from analyzer.core.results import Histogram
from analyzer.cli.browser import HistogramViewer, SimpleSlider
from textual.app import App, ComposeResult
from textual.widgets import Select, Switch


# Helper app to host the widget
class HistogramApp(App):
    def __init__(self, histogram):
        super().__init__()
        self.histogram = histogram

    def compose(self) -> ComposeResult:
        yield HistogramViewer(self.histogram)


@pytest.mark.asyncio
async def test_histogram_viewer_logic():
    # 1. Create a 3D histogram
    # Axes: "pt", "eta", "phi"
    h = (
        hist.Hist.new.Regular(10, 0, 100, name="pt")
        .Regular(10, -3, 3, name="eta")
        .Regular(10, 0, 6.28, name="phi")
        .Double()
    )
    # Fill with some dummy data
    h.fill(
        pt=np.random.uniform(0, 100, 1000),
        eta=np.random.uniform(-3, 3, 1000),
        phi=np.random.uniform(0, 6.28, 1000),
    )

    hist_result = Histogram(name="test_hist", histogram=h, axes=h.axes)

    app = HistogramApp(hist_result)

    async with app.run_test() as pilot:
        viewer = app.query_one(HistogramViewer)

        # Initial check
        # With new default logic:
        # n=3. x_axis_idx = 3 - 2 = 1 ("eta")
        # y_axis_idx = 3 - 1 = 2 ("phi")

        assert viewer.x_axis_idx == 1
        assert viewer.y_axis_idx == 2

        # Check controls exist
        x_select = app.query_one("#x_axis_select", Select)
        assert x_select.value == 1

        y_select = app.query_one("#y_axis_select", Select)
        assert y_select.value == 2

        # "other" axes should now be 0 ("pt")
        # Expect "mode_0" Is now a switch. default=sum -> switch.value = False
        mode_0 = app.query_one("#mode_0", Switch)
        assert mode_0.value is False

        # Change X axis to "pt" (0)
        x_select.value = 0
        await pilot.pause(0.5)

        assert viewer.x_axis_idx == 0

        # Change Y axis to None (1D plot)
        y_select.value = None
        await pilot.pause(0.5)

        assert viewer.y_axis_idx is None

        # Now "pt" is X. "eta" (1), "phi" (2) are other.
        # Set "phi" (2) to slice.
        # Need to flip the switch for 2
        mode_2 = app.query_one("#mode_2", Switch)
        mode_2.value = True
        await pilot.pause(0.5)

        assert viewer.axis_configs[2]["mode"] == "slice"

        # Phi is Regular axis -> Continuous -> SimpleSlider
        # id "slider_2"
        slider_2 = app.query_one("#slider_2", SimpleSlider)
        assert slider_2 is not None

        # Interact with slider
        slider_2.action_slide_right()
        await pilot.pause(0.5)

        assert viewer.axis_configs[2]["bin"] == 1

        slider_2.action_slide_left()
        await pilot.pause(0.5)

        assert viewer.axis_configs[2]["bin"] == 0
