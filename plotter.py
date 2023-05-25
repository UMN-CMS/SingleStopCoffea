import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial
from rich.progress import track
import concurrent.futures
from typing import List, Iterable, Callable
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable


font_dirs = ["./fonts"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font in font_files:
    font_manager.fontManager.addfont(font)

plt.style.use("style.mplstyle")


def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r


all_hists = getHistograms("output.pkl")


def addPrelim(ax):
    ax.text(
        0.02,
        0.98,
        "CMS Preliminary",
        style="italic",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def countHistAxes(hist):
    return len(hist.axes)




@dataclass
class SampleHistogram:
    name: str
    hist: hist.Hist

def draw1DHistogram(ax, vals, edges, label=None, orientation="vertical", **kwargs):
    widths = np.diff(edges)
    ret = ax.hist(edges[:-1], bins=edges, weights=vals, label=label, orientation=orientation)
    return ret

def drawScatter(ax, vals, edges, yerr=None, label=None, **kwargs):
    locs =  edges[:-1] + np.diff(edges) / 2
    ret = ax.errorbar(locs, vals, yerr=yerr, label=label, fmt="o")
    return ret

def make1DPlot(
    ax,
    hists: Iterable[SampleHistogram],
    draw_function: Callable,
    stack=True,
    sort=True,
    errors=False,
):
    hists = list(hists)
    hists = list(sorted(hists, key=lambda x: x.hist.sum().value) if sort else hists)
    bottom = None
    for sample_hist in hists:
        print(sample_hist)
        x = sample_hist.hist.to_numpy()
        vals, edges = sample_hist.hist.to_numpy()
        variance = sample_hist.hist.view().variance if errors else None
        draw_function(ax, vals, edges, label=sample_hist.name, bottom=bottom, yerr=variance)
        if stack:
            bottom = bottom + vals if bottom is not None else vals
    return ax

def make2DHistogram(ax, hists):
    pass


def makeStackPlot(ax, histogram, signal=None, dset_axis=0):
    signal = [] if signal is None else signal
    samples = list(s for s in histogram.axes[0] if s not in signal)
    samples = ["TT2018"]
    ax = make1DPlot(
        ax,
        [SampleHistogram(s, histogram[s, ...]) for s in samples],
        draw1DHistogram
    )
    ax = make1DPlot(
        ax,
        [SampleHistogram(s,histogram[s, ...]) for s in signal],
        drawScatter,
        errors=True
    )
    ax.set_yscale("log")
    xax = next(x for x in histogram.axes if x.name != "dataset")
    ax.set_xlabel(xax.label)
    ax.set_title(histogram.name)
    ax.set_ylabel("Weighted Events")
    ax.legend()
    addPrelim(ax)
    return ax




signals = [
                "signal_2000_600_Skim",
                "signal_1500_400_Skim",
                "signal_2000_1900_Skim",
                "signal_2000_400_Skim",
                "signal_2000_900_Skim",
                "signal_1500_600_Skim",
            ]

def makefig(o, vals):
    n, h = vals
    if countHistAxes(h) == 2:
        f, ax = plt.subplots()
        print(signals)
        ax = makeStackPlot(ax, h, signals)
        ax.set_yscale("log")
        xax = next(x for x in h.axes if x.name != "dataset")
        ax.set_xlabel(xax.label)
        ax.set_title(h.name)
        ax.set_ylabel("Weighted Events")
        # ax.legend()
        addPrelim(ax)
        f.savefig(o / n)

def make2DHist(ax, histogram):
    vals, e1,e2= histogram.to_numpy()
    ex = (e1[1:] + e1[:-1]) / 2
    ey = (e2[1:] + e2[:-1]) / 2
    vx,vy = np.meshgrid(ex,ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    ax.hist2d(x,y, bins=np.array([e1,e2]), weights=w)
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel(histogram.axes[1].label)
    return ax



def makefig2D(o, h):
    print(h)
    f, ax = plt.subplots()
    ax = make2DHist(ax, h)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    binwidth = 0.25

    draw1DHistogram(ax_histx, *h[:,sum].to_numpy())
    draw1DHistogram(ax_histy, *h[sum,:].to_numpy(), orientation="horizontal") 
    f.tight_layout()
    f.savefig(o / "test.pdf")



if __name__ == "__main__":
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    # makefig(outdir, ("m04_m", all_hists["m04_m"]))
    makefig2D(outdir, all_hists["m14_vs_m04"]["QCD2018", ...])
    # with concurrent.futures.ProcessPoolExecutor() as p:
    #    futs = p.map(partial(makefig, outdir), list(all_hists.items())[0:2])
    #    for f in track(futs):
    #        pass
