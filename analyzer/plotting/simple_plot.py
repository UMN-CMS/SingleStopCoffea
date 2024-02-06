import pickle as pkl
import sys
from analyzer.utils import accumulate
import hist
import matplotlib.pyplot as plt
from analyzer.plotting.styles import *
from analyzer.plotting.core_plots import *
from analyzer.datasets import SampleManager
from pathlib import Path
import logging
import logging.handlers
from enum import Enum, auto
import atexit


def plotPulls(plotobj_pred, plotobj_obs, coupling, lumi):
    hopo = plotobj_obs
    hppo = plotobj_pred
    fig, ax = drawAs1DHist(hopo, yerr=True, fill=False)
    drawAs1DHist(ax, hppo, yerr=True, fill=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    ab = ax.bottom_axes[0]
    drawPull(ab, hppo, hopo)
    ab.set_ylabel(r"$\frac{pred - obs}{\sigma_{pred}}$")
    addEra(ax, lumi or 59.8)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    addTitles1D(ax, hopo.hist, top_pad=0.2)
    fig.tight_layout()
    return fig


def plotRatio(plotobj_pred, plotobj_obs, coupling, lumi):
    hopo = plotobj_obs
    hppo = plotobj_pred
    fig, ax = drawAs1DHist(hopo, yerr=True, fill=False)
    drawAs1DHist(ax, hppo, yerr=True, fill=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    ab = ax.bottom_axes[0]
    drawRatio(ab, hppo, hopo)
    ab.set_ylabel("Ratio")
    addEra(ax, lumi)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    addTitles1D(ax, hopo.hist, top_pad=0.2)
    fig.tight_layout()
    return fig


def plot1D(
    signal_plobjs,
    background_plobjs,
    lumi,
    coupling,
    sig_style="hist",
    scale="log",
    xlabel_override=None,
    add_label=None,
    top_pad=0.4,
):
    fig, ax = plt.subplots()
    for o in background_plobjs:
        drawAs1DHist(ax, o, yerr=False)
    for o in signal_plobjs:
        #drawAs1DHist(ax, o, yerr=False)
        if sig_style == "scatter":
            drawAsScatter(ax, o, yerr=True)
        elif sig_style == "hist":
            drawAs1DHist(ax, o, yerr=True, fill=False)

    ax.set_yscale(scale)
    addEra(ax, lumi)
    addPrelim(
        ax,
        additional_text=f"\n$\\lambda_{{{coupling}}}''$ Selection\n"
        + (add_label or ""),
    )
    hc = next(it.chain(signal_plobjs, background_plobjs)).hist
    addTitles1D(ax, hc, top_pad=top_pad)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*reversed(sorted(zip(labels, handles), key=lambda t: t[0])))
    extra_legend_args = {}
    if len(labels) > 5:
        extra_legend_args["prop"] = {"size": 10}
    ax.legend(handles, labels, **extra_legend_args)
    if xlabel_override:
        ax.set_xlabel(xlabel_override)
    fig.tight_layout()
    return fig


def plot2D(
    plot_obj,
    lumi,
    coupling,
    sig_style="hist",
    scale="log",
    add_label=None,
):
    fig, ax = plt.subplots()
    drawAs2DHist(ax, plot_obj)
    addEra(ax, lumi)
    pos = "in"
    addPrelim(
        ax,
        additional_text=f"\n$\\lambda_{{{coupling}}}''$ Selection\n"
        + (f"{add_label}," if add_label else "")
        + f"{plot_obj.title}",
        pos=pos,
        color="white",
    )
    addTitles2D(ax, plot_obj.hist)
    return fig


class _Split(object):
    def __new__(cls):
        return NoParam

    def __reduce__(self):
        return (_NoParamType, ())


class Plotter:
    Split = object.__new__(_Split)

    def _createLogger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter(f"[Plotter]: %(message)s"))
        self.logger.addHandler(stream_handler)
        self.logger.info("Creating plotter")

    def __init__(
        self,
        filenames,
        outdir,
        target_lumi=None,
        default_backgrounds=None,
        dataset_dir="datasets",
        coupling="312",
        default_axis_opts=None,
    ):
        loadStyles()
        self._createLogger()

        self.default_backgrounds = default_backgrounds or []
        self.default_axis_opts = default_axis_opts

        self.sample_manager = SampleManager()
        self.sample_manager.loadSamplesFromDirectory(dataset_dir)

        filenames = [filenames] if isinstance(filenames, str) else list(filesnames)
        results = [pkl.load(open(f, "rb")) for f in filenames]

        self.target_lumi = (
            target_lumi
            or self.sample_manager.getSet(list(results[0].results.keys())[0]).getLumi()
        )

        self.coupling = coupling

        self.histos = accumulate(
            [
                f.getMergedHistograms(self.sample_manager, self.target_lumi)
                for f in results
            ]
        )
        print(self.histos)
        used_samples = set(it.chain.from_iterable(x.results.keys() for x in results))
        lumis = [
            round(self.sample_manager.getSet(x).getLumi(), 4) for x in used_samples
        ]
        if (
            not target_lumi
            and len(
                set(
                    round(self.sample_manager.getSet(x).getLumi(), 4)
                    for x in used_samples
                )
            )
            > 1
        ):
            raise ValueError(
                "The underlying sampels have different luminosities, and you are not performing scaling"
            )

        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

    def __call__(self, *args, **kwargs):
        self.doPlot(*args, **kwargs)

    def plotPulls(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject(
            ho, self.sample_manager[hist_obs].getTitle(), self.sample_manager[hist_obs]
        )
        hppo = PlotObject(
            hp,
            self.sample_manager[hist_pred].getTitle(),
            self.sample_manager[hist_pred],
        )
        fig = plotPulls(hppo, hopo, self.coupling, self.target_lumi)
        fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
        plt.close(fig)

    def plotRatio(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject(
            ho, self.sample_manager[hist_obs].getTitle(), self.sample_manager[hist_obs]
        )
        hppo = PlotObject(
            hp,
            self.sample_manager[hist_pred].getTitle(),
            self.sample_manager[hist_pred],
        )
        fig = plotPulls(hppo, hopo, self.coupling, self.target_lumi)
        fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
        plt.close(fig)

    def doPlot(
        self,
        hist_name,
        *args,
        add_name=None,
        axis_opts=None,
        **kwargs,
    ):
        all_axis_opts = {**(self.default_axis_opts or {})}
        all_axis_opts.update((axis_opts or {}))
        h = self.histos[hist_name]
        axes_names = {x.name for x in h.axes}
        for n in all_axis_opts.keys():
            if n not in axes_names:
                raise KeyError(f"Name {n} is not an axis in {h}")
        to_split = [x for x, y in all_axis_opts.items() if y is Plotter.Split]
        all_axis_opts = {
            x: y for x, y in all_axis_opts.items() if y is not Plotter.Split
        }
        h = h[all_axis_opts]
        if to_split:
            split_axes = [list(x) for x in h.axes if x.name in to_split]
            split_names = [x.name for x in h.axes if x.name in to_split]
            for combo in it.product(*split_axes):
                if add_name:
                    add_name = add_name + "_"
                else:
                    add_name = ""
                new_name = add_name + "cuts__" + "_".join(str(x) for x in combo)
                f = dict(zip(split_names, (hist.loc(x) for x in combo)))
                to_pass = h[f]
                self.__doPlot(hist_name, to_pass, *args, **kwargs, add_name=new_name)
        else:
            self.__doPlot(hist_name, h, *args, **kwargs, add_name=add_name)

    def __doPlot(
        self,
        hist_name,
        hist,
        sig_set,
        bkg_set=None,
        scale="log",
        title="",
        add_name="",
        normalize=False,
        sig_style="hist",
        add_label=None,
        top_pad=0.4,
        xlabel_override=None,
    ):
        bkg_set = bkg_set if bkg_set is not None else self.default_backgrounds
        if add_label is None and add_name:
            add_label = add_name.title()
        self.logger.info(f"Now plotting {hist_name}")
        add_name = add_name + "_" if add_name else ""
        hc = hist[{"dataset": bkg_set + sig_set}]
        if normalize:
            hc = getNormalized(hc, "dataset")
        background_plobjs = createPlotObjects(
            hc, "dataset", self.sample_manager, cat_filter="^(?!signal)"
        )
        signal_plobjs = createPlotObjects(
            hc, "dataset", self.sample_manager, cat_filter="signal"
        )
        print(signal_plobjs, background_plobjs)

        if len(hist.axes) == 2:
            fig = plot1D(
                signal_plobjs,
                background_plobjs,
                self.target_lumi,
                self.coupling,
                sig_style=sig_style,
                xlabel_override=xlabel_override,
                add_label=add_label,
                top_pad=top_pad,
            )
            fig.tight_layout()
            fig.savefig(self.outdir / f"{add_name}{hist_name}.pdf")
            plt.close(fig)

        elif len(hist.axes) == 3:
            for x in hc.axes[0]:
                realh = hc[{"dataset": x}]
                po = PlotObject(realh, x, self.sample_manager[x].style)
                fig = plot2D(
                    po,
                    self.coupling,
                    self.target_lumi,
                    sig_style=sig_style,
                    add_label=add_label,
                )
                name = hist.name
                fig.tight_layout()
                fig.savefig(self.outdir / f"{add_name}{hist_name}_{x}.pdf")
                plt.close(fig)
