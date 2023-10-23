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
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor, wait
import atexit


loadStyles()


class Plotter:
    def __init__(
        self,
        filenames,
        outdir,
        default_backgrounds=None,
        dataset_dir="datasets",
        coupling="312",
        parallel=None,
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(f"[Plotter]: %(message)s"))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

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

        self.description = ""
        self.parallel = parallel
        if self.parallel:
            self.pool = ProcessPoolExecutor(self.parallel)
            self.futures = []
            atexit.register(self.finishRemaining)

    def __call__(self, *args, **kwargs):
        if self.parallel:
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
        addPrelim(
               ax,
               additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ "
        )
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
        addPrelim(
               ax,
               additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ "
        )
        addTitles1D(ax, ho, top_pad=0.2)
        fig.tight_layout()
        fig.savefig(self.outdir / f"ratio_{hist_obs}_{hist_pred}.pdf")
        plt.close(fig)

    def doPlot(
        self,
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
    ):
        bkg_set = bkg_set if bkg_set is not None else self.default_backgrounds
        if add_label is None and add_name:
            add_label = add_name.title()
        self.logger.info(f"Now plotting {hist}")
        add_name = add_name + "_" if add_name else ""
        h = self.histos[hist]
        self.description += f"{hist}: {h.description}\n"
        hc = h[{"dataset": bkg_set + sig_set}]
        if normalize:
            hc = getNormalized(hc, "dataset")
        if len(h.axes) == 2:
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
            ax.legend(handles, labels)

            fig.tight_layout()
            fig.savefig(self.outdir / f"{add_name}{hist}.pdf")
            plt.close(fig)
        elif len(h.axes) == 3:
            for x in hc.axes[0]:
                realh = hc[{"dataset": x}]
                if sig_style == "hist":
                    fig, ax = drawAs2DHist(PlotObject(realh, x, self.manager[x]))
                    addTitles2D(ax, realh)
                    addPrelim(ax, "out")
                    addEra(ax, self.lumi or 59.8)
                elif sig_style == "profile":
                    fig, ax = drawAs2DExtended(
                        PlotObject(realh, x, self.manager[x]),
                        top_stack=[PlotObject(realh[sum, :], x, self.manager[x])],
                        right_stack=[PlotObject(realh[:, sum], x, self.manager[x])],
                    )
                    addTitles2D(ax, realh)
                    addPrelim(
                        ax,
                        additional_text=f"\n$\\lambda_{{{self.coupling}}}''$ Selection\n"
                        + f"{add_label},"
                        + f"{self.manager[x].getTitle()}",
                        pos="in",
                        color="white",
                    )
                    addEra(ax.top_axes[-1], self.lumi or 59.8)
                name = h.name
                fig.tight_layout()
                fig.savefig(self.outdir / f"{add_name}{hist}_{x}.pdf")
                plt.close(fig)
