from __future__ import annotations
from textual.app import App, ComposeResult
from textual import on
from textual.reactive import reactive
from textual.widgets import (
    Header,
    Tree,
    Footer,
    Static,
    Pretty,
    Input,
    Button,
    Select,
    Label,
    Switch,
)
from textual.containers import Horizontal, Vertical
from analyzer.core.results import ResultGroup, Histogram
from math import ceil
from typing import Optional

from rich.console import RenderableType
from textual import events
from textual.binding import Binding
from textual.geometry import Offset, clamp
from textual.message import Message
from textual.reactive import var
from textual.scrollbar import ScrollBarRender
from textual.widget import Widget
from textual_plotext import PlotextPlot
from pylatexenc.latex2text import LatexNodes2Text


_latex_converter = LatexNodes2Text()


def latex_to_unicode(text: str) -> str:
    """Convert LaTeX strings to Unicode."""
    try:
        return _latex_converter.latex_to_text(text)
    except Exception:
        return text


def processHistogram(histogram, axes_config):
    """Process histogram by applying operations to each axis.

    Args:
        histogram: hist.Hist object
        axes_config: Dict mapping axis index to operation config
            - {"action": "sum"}: Sum over axis
            - {"action": "slice", "bin": int}: Select specific bin
            - {"action": "project"} or {"action": "keep"}: Keep axis

    Returns:
        Processed hist.Hist object
    """
    slices = []
    for i, ax in enumerate(histogram.axes):
        config = axes_config.get(i)

        if config is None:
            slices.append(slice(None))
            continue

        action = config.get("action")

        if action == "sum":
            slices.append(sum)
        elif action == "slice":
            slices.append(config.get("bin", 0))
        elif action in ("project", "keep"):
            slices.append(slice(None))
        else:
            slices.append(slice(None))

    return histogram[tuple(slices)]


class ResultBrowser(App):
    CSS = """
    ResultTree{
    width: 1fr;
    }
    ResultViewer{
    width: 2fr;
    }
    
    .controls_container {
        height: auto;
        dock: top;
        margin-bottom: 1;
    }
    .axis_selectors {
        width: 30%;
        height: auto;
    }
    .other_axes_controls {
        width: 70%;
        height: auto;
    }
    .axis_row {
        height: auto;
        align: left middle;
        margin: 1;
    }
    .axis_row > Label {
        width: 15;
        text-align: right;
        margin-right: 1;
    }
    .val_label {
        width: auto;
        min-width: 10;
        margin-left: 1;
    }
    PlotextPlot {
        height: 1fr;
    }
    """
    TITLE = "Result Browser"

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, results):
        super().__init__()
        self.results = results
        self.viewer = ResultViewer()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            with Horizontal():
                yield ResultTree(self.results)
                yield self.viewer
            yield Footer()

    @on(Tree.NodeSelected)
    def handleSelected(self, message):
        self.viewer.showResult(message.node.data)

    def action_quit(self):
        self.exit()


class ResultTree(Vertical):
    def __init__(self, results, **kwargs):
        super().__init__(**kwargs)
        self.results = results
        self.result_tree = Tree("/")
        self.result_tree.root.expand()
        self.filter_text = ""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search...")
        self.rebuildTree()
        yield self.result_tree

    @on(Input.Changed)
    def handleInputChanged(self, message: Input.Changed):
        self.filter_text = message.value
        self.rebuildTree()

    def rebuildTree(self):
        self.result_tree.clear()
        self.result_tree.root.expand()

        def handleResult(parent, result):
            def matches(r):
                if isinstance(r, ResultGroup):
                    return self.filter_text.lower() in r.name.lower() or any(
                        matches(r[child]) for child in r
                    )
                return self.filter_text.lower() in r.name.lower()

            should_add = matches(result) if self.filter_text else True

            if should_add:
                if isinstance(result, ResultGroup):
                    new_node = parent.add(result.name, data=result)
                    new_node.expand()
                    for child_name in result:
                        handleResult(new_node, result[child_name])
                else:
                    parent.add_leaf(result.name, data=result)

        handleResult(self.result_tree.root, self.results)


class SimpleSlider(Widget, can_focus=True):
    """A simple slider widget.
    Adapted from https://github.com/TomJGooding/textual-slider
    """

    BINDINGS = [
        Binding("right", "slide_right", "Slide Right", show=False),
        Binding("left", "slide_left", "Slide Left", show=False),
    ]

    COMPONENT_CLASSES = {"slider--slider"}

    DEFAULT_CSS = """
    SimpleSlider {
        width: 24;
        height: 3;
        min-height: 3;
        border: tall $border-blurred;
        background: $surface;
        padding: 0 2;

        & > .slider--slider {
            background: $panel-darken-2;
            color: $primary;
        }

        &:focus {
            border: tall $border;
            background-tint: $foreground 5%;
        }
    }
    """

    value: reactive[int] = reactive(0, init=False)
    """The value of the slider."""

    _slider_position: reactive[float] = reactive(0.0)
    _grabbed: var[Offset | None] = var[Optional[Offset]](None)
    _grabbed_position: var[float] = var(0.0)

    class Changed(Message):
        """Posted when the value of the slider changes.

        This message can be handled using an `on_slider_changed` method.
        """

        def __init__(self, slider: SimpleSlider, value: int) -> None:
            super().__init__()
            self.value: int = value
            self.slider: SimpleSlider = slider

        @property
        def control(self) -> SimpleSlider:
            return self.slider

    def __init__(
        self,
        min: int,
        max: int,
        *,
        step: int = 1,
        value: int | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Create a slider widget.

        Args:
            min: The minimum value of the slider.
            max: The maximum value of the slider.
            step: The step size of the slider.
            value: The initial value of the slider.
            name: The name of the slider.
            id: The ID of the slider in the DOM.
            classes: The CSS classes of the slider.
            disabled: Whether the slider is disabled or not.
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.min = min
        self.max = max
        self.step = step
        self.value = value if value is not None else min
        self._slider_position = (
            (self.value - self.min) / (self.number_of_steps / 100)
        ) / self.step

    @property
    def number_of_steps(self) -> int:
        return int((self.max - self.min) / self.step) + 1

    def validate_value(self, value: int) -> int:
        return clamp(value, self.min, self.max)

    def validate__slider_position(self, slider_position: float) -> float:
        max_position = (
            (self.max - self.min) / (self.number_of_steps / 100)
        ) / self.step
        return clamp(slider_position, 0, max_position)

    def watch_value(self) -> None:
        if not self._grabbed:
            self._slider_position = (
                (self.value - self.min) / (self.number_of_steps / 100)
            ) / self.step
        self.post_message(self.Changed(self, self.value))

    def render(self) -> RenderableType:
        style = self.get_component_rich_style("slider--slider")
        step_ratio = ceil(100 / self.number_of_steps)
        return ScrollBarRender(
            virtual_size=100,
            window_size=step_ratio,
            position=self._slider_position,
            style=style,
            vertical=False,
        )

    def action_slide_right(self) -> None:
        self.value = self.value + self.step

    def action_slide_left(self) -> None:
        self.value = self.value - self.step

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()

        mouse_x = event.x - self.styles.gutter.left
        mouse_y = event.y - self.styles.gutter.top

        if not (0 <= mouse_x < self.content_size.width) or not (
            0 <= mouse_y < self.content_size.height
        ):
            return

        step_ratio = ceil(100 / self.number_of_steps)
        thumb_size = max(1, step_ratio / (100 / self.content_size.width))

        self._slider_position = (
            (mouse_x - (thumb_size // 2)) / self.content_size.width
        ) * 100

        self._grabbed = event.screen_offset
        self.action_grab()

        self.value = (
            self.step * round(self._slider_position * (self.number_of_steps / 100))
            + self.min
        )

    def action_grab(self) -> None:
        self.capture_mouse()

        # Workaround for unexpected mouse grab and drag behaviour
        # depending on the currently focused widget.
        # Stolen from https://github.com/1j01/textual-paint
        self.can_focus = False

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        if self._grabbed:
            self.release_mouse()
            self._grabbed = None

            # Workaround for unexpected mouse behaviour mentioned above
            self.can_focus = True

        event.stop()

    def _on_mouse_capture(self, event: events.MouseCapture) -> None:
        self._grabbed = event.mouse_position
        self._grabbed_position = self._slider_position

    def _on_mouse_release(self, event: events.MouseRelease) -> None:
        self._grabbed = None
        event.stop()

    async def _on_mouse_move(self, event: events.MouseMove) -> None:
        if self._grabbed:
            mouse_move = event.screen_x - self._grabbed.x
            self._slider_position = self._grabbed_position + (
                mouse_move * (100 / self.content_size.width)
            )
            self.value = (
                self.step * round(self._slider_position * (self.number_of_steps / 100))
                + self.min
            )

        event.stop()

    async def _on_click(self, event: events.Click) -> None:
        event.stop()


class HistogramViewer(Widget):
    def __init__(self, histogram: Histogram, **kwargs):
        super().__init__(**kwargs)
        self.histogram_result = histogram
        self.hist_obj = histogram.histogram
        self.axes = self.hist_obj.axes

        self.x_axis_idx = 0
        self.y_axis_idx = None

        # Default defaults: last axes
        n = len(self.axes)
        if n >= 2:
            self.x_axis_idx = n - 2
            self.y_axis_idx = n - 1
        elif n == 1:
            self.x_axis_idx = n - 1
            self.y_axis_idx = None

        self.axis_configs = {}
        for i in range(len(self.axes)):
            if i == self.x_axis_idx or i == self.y_axis_idx:
                continue
            self.axis_configs[i] = {"mode": "sum", "bin": 0}

        # Debouncing for plot updates
        self._update_timer = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="controls_container"):
            with Vertical(classes="axis_selectors"):
                yield Label("X Axis")
                yield Select(
                    [(latex_to_unicode(ax.name), i) for i, ax in enumerate(self.axes)],
                    value=self.x_axis_idx,
                    id="x_axis_select",
                )
                yield Label("Y Axis (Optional)")
                yield Select(
                    [("None", None)]
                    + [
                        (latex_to_unicode(ax.name), i) for i, ax in enumerate(self.axes)
                    ],
                    value=self.y_axis_idx,
                    id="y_axis_select",
                )

            with Vertical(classes="other_axes_controls", id="other_axes_container"):
                pass

        yield PlotextPlot(id="plot")

    async def on_mount(self):
        await self.rebuildControls()
        self.updatePlot()

    async def rebuildControls(self):
        if not self.is_mounted:
            return
        container = self.query_one("#other_axes_container", Vertical)
        await container.remove_children()

        widgets_to_mount = []
        for i, ax in enumerate(self.axes):
            if i == self.x_axis_idx or i == self.y_axis_idx:
                continue

            if i not in self.axis_configs:
                self.axis_configs[i] = {"mode": "sum", "bin": 0}

            config = self.axis_configs[i]

            # Mode switch: Off = Sum, On = Slice
            is_slice = config["mode"] == "slice"
            mode_switch = Switch(value=is_slice, id=f"mode_{i}")

            row_children = [
                Label(f"{latex_to_unicode(ax.name)}:"),
                mode_switch,
                Label("Slice" if is_slice else "Sum"),
            ]

            if is_slice:
                import hist.axis

                if isinstance(ax, (hist.axis.Regular, hist.axis.Variable)):
                    slider = SimpleSlider(
                        min=0,
                        max=ax.size - 1,
                        value=config["bin"],
                        id=f"slider_{i}",
                    )
                    val_label = Label(
                        f"{self.getAxisValue(i, config['bin']):.2f}",
                        id=f"val_label_{i}",
                        classes="val_label",
                    )
                    row_children.append(slider)
                    row_children.append(val_label)
                else:
                    n_bins = ax.size
                    bin_select = Select(
                        [(str(b), b) for b in range(n_bins)],
                        value=config["bin"],
                        id=f"bin_{i}",
                    )
                    row_children.append(bin_select)

            row = Horizontal(*row_children, classes="axis_row")
            widgets_to_mount.append(row)

        await container.mount(*widgets_to_mount)

    def getAxisValue(self, axis_idx, bin_idx):
        ax = self.axes[axis_idx]
        if hasattr(ax, "centers"):
            try:
                return ax.centers[bin_idx]
            except IndexError:
                return 0.0
        return 0.0

    @on(Select.Changed)
    async def handleSelectChange(self, event: Select.Changed):
        trigger_id = event.control.id
        if not trigger_id:
            return

        if trigger_id == "x_axis_select":
            val = event.value
            if val == self.y_axis_idx:
                self.y_axis_idx = None
                self.query_one("#y_axis_select", Select).value = None
            self.x_axis_idx = val
            await self.rebuildControls()
            self.updatePlot()

        elif trigger_id == "y_axis_select":
            val = event.value
            if val == self.x_axis_idx:
                event.control.value = None
                self.y_axis_idx = None
            else:
                self.y_axis_idx = val
            await self.rebuildControls()
            self.updatePlot()

    @on(Switch.Changed)
    def handleSwitchChange(self, event: Switch.Changed):
        trigger_id = event.control.id
        if not trigger_id:
            return

        if trigger_id.startswith("mode_"):
            axis_idx = int(trigger_id.split("_")[1])
            new_mode = "slice" if event.value else "sum"
            self.axis_configs[axis_idx]["mode"] = new_mode

            async def rebuild_and_update():
                try:
                    await self.rebuildControls()
                    self.updatePlot()
                except Exception:
                    pass

            self.run_worker(rebuild_and_update())

        elif trigger_id.startswith("bin_"):
            axis_idx = int(trigger_id.split("_")[1])
            self.axis_configs[axis_idx]["bin"] = event.value
            self.updatePlot()

    @on(SimpleSlider.Changed)
    def handleSliderChange(self, event: SimpleSlider.Changed):
        trigger_id = event.control.id
        if trigger_id and trigger_id.startswith("slider_"):
            axis_idx = int(trigger_id.split("_")[1])
            self.axis_configs[axis_idx]["bin"] = event.value

            # Update label immediately for responsiveness
            try:
                lbl = self.query_one(f"#val_label_{axis_idx}", Label)
                lbl.update(f"{self.getAxisValue(axis_idx, event.value):.2f}")
            except Exception:
                pass

            # Debounce plot update to improve performance
            self.debouncedUpdatePlot()

    def debouncedUpdatePlot(self, delay: float = 0.1):
        """Debounced plot update to reduce rendering frequency."""
        if self._update_timer is not None:
            self._update_timer.stop()

        self._update_timer = self.set_timer(delay, self.updatePlot)

    async def _rebuild_and_plot(self):
        """Helper to rebuild controls."""
        try:
            await self.rebuildControls()
        except Exception:
            pass

    def updatePlot(self):
        if not self.is_mounted:
            return
        plt_widget = self.query_one("#plot", PlotextPlot)
        plt = plt_widget.plt
        plt.clear_data()
        plt.clear_figure()

        h = self.hist_obj

        # Prepare config for processHistogram
        # HistogramViewer stores config as {idx: {"mode": "sum"|"slice", "bin": int}}
        # processHistogram expects {idx: {"action": "sum"|"slice"|"keep", "bin": int}}

        process_src = {}
        for i, ax in enumerate(self.axes):
            if i == self.x_axis_idx or i == self.y_axis_idx:
                process_src[i] = {"action": "keep"}
            else:
                c = self.axis_configs.get(i, {"mode": "sum", "bin": 0})
                action = "sum" if c["mode"] == "sum" else "slice"
                process_src[i] = {"action": action, "bin": c["bin"]}

        try:
            h_projected = processHistogram(h, process_src)
        except Exception as e:
            plt.title(f"Error slicing: {e}")
            plt_widget.refresh()
            return

        if self.y_axis_idx is None:
            if len(h_projected.axes) != 1:
                plt.title("Error: Result is not 1D")
            else:
                ax = h_projected.axes[0]
                if len(ax.centers) > 0:
                    x_data = ax.centers
                    y_data = h_projected.values()
                    plt.bar(x_data, y_data, label=self.analysisLabel())
                    plt.xlabel(ax.name)
                else:
                    plt.title("Empty Histogram")
        else:
            if len(h_projected.axes) != 2:
                plt.title("Error: Result is not 2D")
            else:
                # 2D heatmap
                ax_x = h_projected.axes[0]
                ax_y = h_projected.axes[1]

                # Get the 2D histogram values
                values = h_projected.values()

                # plotext matrix_plot for 2D visualization
                import numpy as np

                # Create matrix plot (heatmap)
                # matrix_plot expects a list of lists
                data = values.T.tolist()
                plt.matrix_plot(data)

                # Set axis labels
                plt.xlabel(latex_to_unicode(ax_x.name))
                plt.ylabel(latex_to_unicode(ax_y.name))

                plt.title(
                    f"{latex_to_unicode(ax_x.name)} vs {latex_to_unicode(ax_y.name)}"
                )

        plt_widget.refresh()

    def analysisLabel(self):
        return "Data"


class ResultViewer(Widget):
    result = reactive(None, recompose=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def showResult(self, result):
        self.result = result

    def compose(self):
        if self.result is None:
            return
        w = self.result.widget()
        if w is None:
            if isinstance(self.result, Histogram):
                yield HistogramViewer(self.result)
            else:
                with Vertical():
                    yield Pretty(self.result)
        else:
            yield w
