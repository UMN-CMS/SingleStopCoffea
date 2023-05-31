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
from typing import List, Iterable, Callable, Union, Dict, Any,  Optional
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools as it
import os
from scipy.optimize import curve_fit




try:
    from typing import  TypeAlias
    PathLike: TypeAlias = Union[str, bytes, os.PathLike]
    Number: TypeAlias = Union[int, float]
except ImportError:
    PathLike = Union[str, bytes, os.PathLike]
    Number = Union[int, float]


font_dirs = ["./fonts"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font in font_files:
    font_manager.fontManager.addfont(font)

plt.style.use("style.mplstyle")


def getHistograms(path: PathLike) -> Dict[str, hist.Hist]:
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r


all_hists = getHistograms("output.pkl")


def addPrelim(ax: mpl.axis.Axis) -> mpl.axis.Axis:
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
    return ax


def countHistAxes(hist: hist.Hist) -> int:
    return len(hist.axes)


@dataclass
class SampleHistogram:
    name: str
    hist: hist.Hist


def draw1DHistogram(
    ax: mpl.axis.Axis,
    vals: np.ndarray,
    edges: np.ndarray,
    label: Optional[str] = None,
    orientation: str = "vertical",
    plot_opts: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> mpl.axis.Axis:
    plot_opts = {} if plot_opts is None else plot_opts
    widths = np.diff(edges)
    ret = ax.hist(
        edges[:-1],
        bins=edges,
        weights=vals,
        label=label,
        orientation=orientation,
        **plot_opts,
    )
    return ret


def drawScatter(
    ax: mpl.axis.Axis,
    vals: np.ndarray,
    edges: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    plot_opts: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> mpl.axis.Axis:
    plot_opts = {} if plot_opts is None else plot_opts
    locs = edges[:-1] + np.diff(edges) / 2
    ret = ax.errorbar(locs, vals, yerr=yerr, label=label, fmt="o", **plot_opts)
    return ret


def make1DPlot(
    ax: mpl.axis.Axis,
    hists: Iterable[SampleHistogram],
    draw_function: Callable,
    stack: bool = True,
    sort: bool = True,
    errors: bool = False,
) -> mpl.axis.Axis:
    hists = list(hists)
    hists = list(sorted(hists, key=lambda x: x.hist.sum().value) if sort else hists)
    bottom = None
    for sample_hist in hists:
        print(sample_hist)
        x = sample_hist.hist.to_numpy()
        vals, edges = sample_hist.hist.to_numpy()
        variance = sample_hist.hist.view().variance if errors else None
        draw_function(
            ax, vals, edges, label=sample_hist.name, bottom=bottom, yerr=variance
        )
        if stack:
            bottom = bottom + vals if bottom is not None else vals
    return ax


def makeStackPlot(
    ax: mpl.axis.Axis, histogram: hist.Hist, signal: Optional[Iterable[str]] = None
) -> mpl.axis.Axis:
    signal = [] if signal is None else signal
    samples = list(s for s in histogram.axes[0] if s not in signal)
    samples = ["TT2018"]
    ax = make1DPlot(
        ax, [SampleHistogram(s, histogram[s, ...]) for s in samples], draw1DHistogram
    )
    ax = make1DPlot(
        ax,
        [SampleHistogram(s, histogram[s, ...]) for s in signal],
        drawScatter,
        errors=True,
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


def makefig(o: PathLike, vals) -> None:
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


def make2DHist(ax: mpl.axis.Axis, histogram: hist.Hist, plotopts) -> mpl.axis.Axis:
    vals, e1, e2 = histogram.to_numpy()
    ex = (e1[1:] + e1[:-1]) / 2
    ey = (e2[1:] + e2[:-1]) / 2
    vx, vy = np.meshgrid(ex, ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    ax.hist2d(x, y, bins=np.array([e1, e2]), weights=w, **plotopts)
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel(histogram.axes[1].label)
    return ax


def make2DProjection(
    ax: mpl.axis.Axis,
    h: hist.Hist,
    vlines: Optional[Iterable[Number]] = None,
    hlines: Optional[Iterable[Number]] = None,
    main_opts: Optional[Dict[str, Any]] = None,
    x_opts: Optional[Dict[str, Any]] = None,
    y_opts: Optional[Dict[str, Any]] = None,
) -> mpl.axis.Axis:
    main_opts = {} if main_opts is None else main_opts
    x_opts = {} if x_opts is None else x_opts
    y_opts = {} if y_opts is None else y_opts

    vlines = list(vlines) if vlines is not None else []
    hlines = list(hlines) if hlines is not None else []
    ax = make2DHist(ax, h, main_opts)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 2, pad=0.4, sharex=ax)
    ax_histy = divider.append_axes("right", 2, pad=0.4, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    binwidth = 0.25
    x, y = h[:, sum].to_numpy()
    draw1DHistogram(ax_histx, x, y, plot_opts=x_opts)
    x, y = h[sum, :].to_numpy()
    draw1DHistogram(ax_histy, x, y, orientation="horizontal", plot_opts=y_opts)
    for a, l in it.product([ax, ax_histx], vlines):
        a.axvline(x=l)
    for a, l in it.product([ax, ax_histy], hlines):
        a.axhline(y=l)
    return ax, ax_histx, ax_histy, divider


def make2DSlicedProjection(
    ax: mpl.axis.Axis,
    h: hist.Hist,
    h_cut: hist.Hist,
    add_fit=None,
    vlines: Optional[Iterable[Number]] = None,
    hlines: Optional[Iterable[Number]] = None,
) -> mpl.axis.Axis:
    ax, ax_x, _, div = make2DProjection(ax, h, vlines, hlines)

    ax2 = div.append_axes("right", 2, pad=0.4, sharey=ax)
    x, y = h_cut.to_numpy()

    if add_fit is not None:
        lx,ux = ax_x.get_xlim()
        space = np.linspace(lx,ux,200)
        ax_x.plot(space, add_fit(space))

    draw1DHistogram(ax2, x, y, orientation="horizontal")
    ax2.yaxis.set_tick_params(labelleft=False)
    ax2.text(
        0.95,
        0.97,
        "Post Cut",
        fontsize=14,
        transform=ax2.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )
    return ax


def autoPlot(
    outpath: PathLike,
    function,
    *args,
    fig_params=None,
    **kwargs,
):
    fig_params = {} if not fig_params else fig_params
    p = Path(outpath)
    p.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(**fig_params)
    ax = function(ax, *args, **kwargs)
    fig.savefig(p)
    plt.close()


if __name__ == "__main__":
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    # makefig(outdir, ("m04_m", all_hists["m04_m"]))
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # ax = make2DProjection(ax, all_hists["m14_vs_m04"]["QCD2018", ...], [1000])
    h = all_hists["m14_vs_m04"]["QCD2018", sum, ...]
    h = all_hists["m14_vs_m04"]["signal_2000_900_Skim", sum, ...]
    hx = h[:,sum]
    vals,edges = hx.to_numpy()
    edges = (edges[:-1] + edges[1:])/2
    p0 = [100, 1000, 200]
    coeff, var_matrix = curve_fit(gauss, edges, vals, p0=p0)
    _,mu,sig = coeff
    cutupper = mu+2.5 * sig
    cutlower = mu-2.5 * sig
    h2 = h[hist.loc(cutlower) : hist.loc(cutupper) : sum, :]
    autoPlot(
        outdir / "test.pdf",
        make2DSlicedProjection,
        h,
        h2,
        add_fit=lambda x: gauss(x, *coeff),
        vlines=[cutlower, cutupper],
        fig_params=dict(figsize=(12, 10)),
    )
