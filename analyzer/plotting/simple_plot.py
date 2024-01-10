import pickle as pkl
import sys
from coffea.processor import accumulate

sys.path.append(".")
import hist
import matplotlib.pyplot as plt

from analyzer.plotting.styles import *
from analyzer.plotting.core_plots import *
from analyzer.datasets import loadSamplesFromDirectory

from pathlib import Path
import logging
import logging.handlers
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor, wait
import multiprocess as mp
import atexit



loadStyles()


class _Split(object):
    def __new__(cls):
        return NoParam

    def __reduce__(self):
        return (_NoParamType, ())


class Plotter:
    Split = object.__new__(_Split)
    queue = mp.Queue()

    def __init__(
        self,
        filenames,
        outdir,
        default_backgrounds=None,
        dataset_dir="datasets",
        coupling="312",
        parallel=None,
        default_axis_opts=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.parallel = parallel
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter(f"[Plotter]: %(message)s"))
        self.logger.addHandler(stream_handler)
        if self.parallel:
            self.pool = ProcessPoolExecutor(self.parallel)
            self.futures = []

            atexit.register(self.finishRemaining)

            handler = logging.handlers.QueueHandler(self.queue)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(f"[Plotter]: %(message)s"))
            self.ql = logging.handlers.QueueListener(self.queue, stream_handler)
            self.ql.start()
            atexit.register(lambda: self.ql.stop())


        self.logger.info("Creating plotter")

        filenames = [filenames] if isinstance(filenames, str) else list(filesnames)
        self.data = [pkl.load(open(f, "rb")) for f in filenames]
        self.lumi = None
        sl = set(x.get("target_lumi") for x in self.data)
        if len(sl) == 1:
            self.lumi = sl.pop()
        else:
            self.logger.warn(
                "The loaded files have different target luminosities. This may result in issues."
            )

        self.histos = accumulate([f["histograms"] for f in self.data])
        self.default_backgrounds = default_backgrounds or []
        self.outdir = Path(outdir)
        self.manager = loadSamplesFromDirectory(dataset_dir)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.coupling = coupling
        self.default_axis_opts = default_axis_opts

        self.description = ""

    def __call__(self, *args, **kwargs):
        if self.parallel:
            self.logger.info("Adding job to pool")
            x = self.pool.submit(self.doPlot, *args, **kwargs)
            self.futures.append(x)
        else:
            self.doPlot(*args, **kwargs)
        # self.logger.info("".join('1' if x.done() else '0' for x in self.futures))

    def finishRemaining(self):
        self.logger.info("".join("1" if x.done() else "0" for x in self.futures))
        for x in self.futures:
            print(x.exception())
        if any(x.running() for x in self.futures):
            self.logger(f"Finalizing plots.")
            wait(self.futures)
        self.pool.shutdown()

    def plotPulls(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject(ho, self.manager[hist_obs].getTitle(), self.manager[hist_obs])
        hppo = PlotObject(
            hp, self.manager[hist_pred].getTitle(), self.manager[hist_pred]
        )
        fig, ax = drawAs1DHist(hopo, yerr=True, fill=False)
        drawAs1DHist(ax, hppo, yerr=True, fill=False)
        addAxesToHist(ax, num_bottom=1, bottom_pad=0)
        ab = ax.bottom_axes[0]
        drawPull(ab, hppo, hopo)
        ab.set_ylabel(r"$\frac{pred - obs}{\sigma_{pred}}$")
        addEra(ax, self.lumi or 59.8)
        addPrelim(ax, additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ ")
        addTitles1D(ax, ho, top_pad=0.2)
        fig.tight_layout()
        fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
        plt.close(fig)

    def plotRatio(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject(ho, self.manager[hist_obs].getTitle(), self.manager[hist_obs])
        hppo = PlotObject(
            hp, self.manager[hist_pred].getTitle(), self.manager[hist_pred]
        )
        fig, ax = drawAs1DHist(hopo, yerr=True, fill=False)
        drawAs1DHist(ax, hppo, yerr=True, fill=False)
        addAxesToHist(ax, num_bottom=1, bottom_pad=0)
        ab = ax.bottom_axes[0]
        drawRatio(ab, hppo, hopo)
        ab.set_ylabel("Ratio")
        addEra(ax, self.lumi or 59.8)
        addPrelim(ax, additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ ")
        addTitles1D(ax, ho, top_pad=0.2)
        fig.tight_layout()
        fig.savefig(self.outdir / f"ratio_{hist_obs}_{hist_pred}.pdf")
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
        self.description += f"{hist_name}: {hist.description}\n"
        hc = hist[{"dataset": bkg_set + sig_set}]

        def apply_after(ax):
            if xlabel_override:
                ax.set_xlabel(xlabel_override)

        if normalize:
            hc = getNormalized(hc, "dataset")
        if len(hist.axes) == 2:
            fig, ax = drawAs1DHist(
                hc,
                cat_axis="dataset",
                manager=self.manager,
                cat_filter="^(?!signal)",
                yerr=False,
            )
            if sig_style == "scatter":
                drawAsScatter(
                    ax,
                    hc,
                    cat_axis="dataset",
                    cat_filter="signal",
                    manager=self.manager,
                    yerr=True,
                )
            elif sig_style == "hist":
                drawAs1DHist(
                    ax,
                    hc,
                    cat_axis="dataset",
                    cat_filter="signal",
                    manager=self.manager,
                    yerr=True,
                    fill=False,
                )

            ax.set_yscale(scale)
            addEra(ax, self.lumi or 59.8)
            addPrelim(
                ax,
                additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ Selection\n"
                + (add_label or ""),
            )

            addTitles1D(ax, hc, top_pad=top_pad)
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(
                *reversed(sorted(zip(labels, handles), key=lambda t: t[0]))
            )
            extra_legend_args = {}
            if len(labels) > 5:
                extra_legend_args["prop"] = {"size": 10}
            ax.legend(handles, labels, **extra_legend_args)
            apply_after(ax)

            fig.tight_layout()
            fig.savefig(self.outdir / f"{add_name}{hist_name}.pdf")
            plt.close(fig)
        elif len(hist.axes) == 3:
            for x in hc.axes[0]:
                realh = hc[{"dataset": x}]
                if sig_style == "hist":
                    fig, ax = drawAs2DHist(PlotObject(realh, x, self.manager[x]))
                    addEra(ax, self.lumi or 59.8)
                    pos = "in"
                elif sig_style == "profile":
                    fig, ax = drawAs2DExtended(
                        PlotObject(realh, x, self.manager[x]),
                        top_stack=[PlotObject(realh[sum, :], x, self.manager[x])],
                        right_stack=[PlotObject(realh[:, sum], x, self.manager[x])],
                    )
                    addEra(ax.top_axes[-1], self.lumi or 59.8)
                    pos = "out"

                addPrelim(
                    ax,
                    additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ Selection\n"
                    + (f"{add_label}," if add_label else "")
                    + f"{self.manager[x].getTitle()}",
                    pos=pos,
                    color="white",
                )
                addTitles2D(ax, realh)
                apply_after(ax)
                name = hist.name
                fig.tight_layout()
                fig.savefig(self.outdir / f"{add_name}{hist_name}_{x}.pdf")
                plt.close(fig)
