import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.pyplot import cm
import re
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial
from rich.progress import track
import concurrent.futures
from typing import List, Iterable, Callable, Union, Dict, Any, Optional
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools as it
import os
from scipy.optimize import curve_fit
import argparse
import sys


try:
    from typing import TypeAlias

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
    yerr: np.ndarray = None,
    plot_opts: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> mpl.axis.Axis:

    plot_opts = {} if plot_opts is None else plot_opts
    widths = np.diff(edges)
    kwargs = dict(
        label=label,
        orientation=orientation,
        align="edge",
    )
    if yerr is not None:
        args["yerr"] = yerr
    if orientation == "horizontal":
        if yerr is not None:
            args["xerr"] = yerr
        ret = ax.barh(edges[:-1], vals, height=widths, **kwargs, **plot_opts)
    elif orientation == "vertical":
        if yerr is not None:
            args["yerr"] = yerr
        ret = ax.bar(edges[:-1], vals, width=widths, **kwargs, **plot_opts)

    # print(h)
    # if yerr is not None:
    #    centers = (edges[:-1] + edges[1:])/2
    #    ret = ax.errorbar(centers, vals, yerr=yerr, color=h.color,  fmt="none")
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
    ret = ax.errorbar(locs, vals, yerr=yerr, label=label, fmt="+", **plot_opts)
    return ret


def make1DPlot(
    ax: mpl.axis.Axis,
    hists: Iterable[SampleHistogram],
    draw_function: Callable,
    stack: bool = True,
    sort: bool = True,
    errors: bool = False,
    plot_opts=None,
) -> mpl.axis.Axis:
    hists = list(hists)
    hists = list(sorted(hists, key=lambda x: x.hist.sum().value) if sort else hists)
    bottom = None

    color = iter(cm.rainbow(np.linspace(0, 1, len(hists))))
    plot_opts = plot_opts if plot_opts is not None else {}
    for sample_hist in hists:
        x = sample_hist.hist.to_numpy()
        vals, edges = sample_hist.hist.to_numpy()
        if bottom is None:
            bottom = np.zeros_like(vals)
        variance = np.sqrt(sample_hist.hist.view().variance) if errors else None
        to_pass = dict(label=sample_hist.name, yerr=variance, plot_opts=plot_opts)
        c = next(color)
        to_pass["plot_opts"]["color"] = c
        to_pass["plot_opts"]["ecolor"] = c
        to_pass["plot_opts"]["edgecolor"] = c
        if stack:
            to_pass["plot_opts"]["bottom"] = bottom
        draw_function(ax, vals, edges, **to_pass)
        if stack:
            bottom = bottom + vals
    return ax


def makeStackPlot(
    ax: mpl.axis.Axis,
    histogram: hist.Hist,
    signal: Optional[Iterable[str]] = None,
    backgrounds: Optional[Iterable[str]] = None,
    normalize: bool = False,
) -> mpl.axis.Axis:
    signal = [] if signal is None else signal
    samples = list(s for s in histogram.axes[0] if s not in signal)
    h_bkg = [
        SampleHistogram(s, histogram[s, ...])
        for s in (
            backgrounds
            if backgrounds is not None
            else (x for x in samples if x not in signals)
        )
    ]
    h_sig = [SampleHistogram(s, histogram[s, ...]) for s in signal]
    if normalize:
        for x in h_bkg:
            x.hist /= x.hist.sum().value
        for x in h_sig:
            x.hist /= x.hist.sum().value

    ax = make1DPlot(ax, h_bkg, draw1DHistogram)
    ax = make1DPlot(
        ax, h_sig, draw1DHistogram, errors=True, stack=False, plot_opts=dict(fill=False)
    )
    # ax.set_yscale("log")
    xax = next(x for x in histogram.axes if x.name != "dataset")
    ax.set_xlabel(xax.label)
    ax.set_title(histogram.name)
    ax.set_ylabel("Weighted Events")

    ax.legend(ncol=1, loc="best")
    return ax


def make2DHist(ax: mpl.axis.Axis, histogram: hist.Hist, plotopts) -> mpl.axis.Axis:
    vals, e1, e2 = histogram.to_numpy()
    ex = (e1[1:] + e1[:-1]) / 2
    ey = (e2[1:] + e2[:-1]) / 2
    vx, vy = np.meshgrid(ex, ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    ax.hist2d(x, y, bins=[e1, e2], weights=w, **plotopts)
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel(histogram.axes[1].label)
    return ax


def make2DProjection(
    ax: mpl.axis.Axis,
    h: hist.Hist,
    vlines: Optional[Iterable[Number]] = None,
    hlines: Optional[Iterable[Number]] = None,
    plotopts: Optional[Dict[str, Any]] = None,
    x_opts: Optional[Dict[str, Any]] = None,
    y_opts: Optional[Dict[str, Any]] = None,
) -> mpl.axis.Axis:
    plotopts = {} if plotopts is None else plotopts
    x_opts = {} if x_opts is None else x_opts
    y_opts = {} if y_opts is None else y_opts

    vlines = list(vlines) if vlines is not None else []
    hlines = list(hlines) if hlines is not None else []
    ax = make2DHist(ax, h, plotopts)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1, pad=0.2, sharex=ax)
    ax_histy = divider.append_axes("right", 1, pad=0.2, sharey=ax)
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
    location="vertical"
) -> mpl.axis.Axis:
    ax, ax_x, _, div = make2DProjection(ax, h, vlines, hlines)

    if location=="horizontal":
        ax2 = div.append_axes("right", 1, pad=0.4, sharey=ax)
        ax2.yaxis.set_tick_params(labelleft=False)
    elif location=="vertical":
        ax2 = div.append_axes("top", 1, pad=0.4, sharex=ax)
        ax2.xaxis.set_tick_params(labelbottom=False)
    x, y = h_cut.to_numpy()

    if add_fit is not None:
        lx, ux = ax_x.get_xlim()
        space = np.linspace(lx, ux, 200)
        ax_x.plot(space, add_fit(space))

    draw1DHistogram(ax2, x, y, orientation=location)
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
    fig_manip=None,
    fig_title=None,
    **kwargs,
):
    fig_params = {} if not fig_params else fig_params
    p = Path(outpath)
    p.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(**fig_params)
    ax = function(ax, *args, **kwargs)
    if fig_title: fig.suptitle(fig_title)
    if fig_manip: fig_manip(fig)
    fig.tight_layout()
    fig.savefig(p)
    plt.close()


signals = [
    "signal_1500_400_Skim",
    "signal_2000_1900_Skim",
    "signal_1000_400_Skim",
]


plot_func_map = {
    "stack": makeStackPlot,
    "2D": make2DHist,
    "2DProfile": make2DProjection,
    "2DProfileSlice": make2DSlicedProjection,
}

plot_all_func_map = {1: ["stack"], 2: ["2D", "2DProfileSlice", "2DProfile"]}
default_size = (5, 5)
default_signal_regex = r"signal.*"


def makePlots(
    title,
    hist,
    plot_funcs,
    actions,
    signal_regex,
    dset_regex,
    dset_axis,
    fig_size,
    outdir,
):
    all_slices = {}
    categories = []
    axes = list(x.name for x in hist.axes)
    for ax, act in actions.items():
        if act == "sum":
            all_slices[axes.index(ax)] = sum
        elif act == "split":
            a = next(x for x in hist.axes if x.name == ax)
            categories.append(zip(it.repeat(axes.index(ax)), a))

    datasets = list(next(x for x in hist.axes if x.name == dset_axis))
    all_datasets = list(x for x in datasets if re.search(dset_regex, x))
    signals = sorted([x for x in all_datasets if re.search(signal_regex, x)])
    bkgs = sorted([x for x in all_datasets if x not in signals])

    def doPlot(t, h):
        naxes = countHistAxes(h) - 1

        if naxes > 2:
            raise Exception("Need to further narrow")
        if naxes == 1:
            outfile = outdir / (t + ".pdf")
            autoPlot(
                outfile,
                plot_funcs[naxes],
                h,
                signal=signals,
                backgrounds=bkgs,
                fig_params=dict(figsize=fig_size),
                normalize=not bkgs,
            )
        if naxes == 2:
            for x in all_datasets:
                hd = h[{dset_axis: x}]
                outfile = outdir / f"{x}_{t}.pdf"
                autoPlot(
                    outfile,
                    plot_funcs[naxes],
                    hd,
                    plotopts={},
                    fig_params=dict(figsize=fig_size),
                )

    if categories:
        for cats in it.product(*categories):
            cat_slices = dict(cats)
            to_plot = hist[{**cat_slices, **all_slices}]
            t = title + "_" + "_".join(f"{axes[x]}eq{y}" for x, y in cats)
            doPlot(t, to_plot)
    else:
        to_plot = hist[all_slices]
        doPlot(title, to_plot)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter", description="Tools to make plots")
    parser.add_argument(
        "input_data", nargs="?", type=argparse.FileType("rb"), help="Input data"
    )
    parser.add_argument(
        "-l", "--list-axes", help="List axes and exit", action="store_true"
    )
    parser.add_argument(
        "-s",
        "--figure-size",
        default=default_size,
        help="Dimensions of output image",
        nargs=2,
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, help="Directory to output images", required=True
    )
    parser.add_argument(
        "-1",
        "--oned-plotter",
        default="stack",
        help="Name of function to use for plotting 1-D histograms",
        choices=plot_all_func_map[1],
    )
    parser.add_argument(
        "-2",
        "--twod-plotter",
        default="2D",
        help="Name of function to use for plotting 2-D histograms",
        choices=plot_all_func_map[2],
    )
    parser.add_argument(
        "-r",
        "--name-regex",
        default=r".*",
        help="Regex for selection which histograms to plot",
    )
    parser.add_argument(
        "-n",
        "--sig-regex",
        default=default_signal_regex,
        help="Regex for which datasets should be considered signal",
    )
    parser.add_argument(
        "-d",
        "--dataset-regex",
        default=".*",
        help="Regex for which datasets should be considered",
    )
    parser.add_argument(
        "-k",
        "--dataset-axis-name",
        default="dataset",
        help="Name of the axis holding the dataset",
    )
    parser.add_argument(
        "-a",
        "--axis-operations",
        nargs="*",
        help="axis_name=action pairs, where axis_name is the name of an axis in the histogram",
        action=ParseKwargs,
        default={},
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        default=8,
        type=int,
        help="axis_name=action pairs, where axis_name is the name of an axis in the histogram",
        action=ParseKwargs,
    )
    args = parser.parse_args()

    all_hists = pickle.load(args.input_data)
    hists = {x: y for x, y in all_hists.items() if re.search(args.name_regex, x)}
    if args.list_axes:
        for name, h in hists.items():
            print(f"{name:20}: {[x.name for x in h.axes]}")
        sys.exit(0)

    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)

    nthreads = args.num_threads

    plotters = {}
    plotters[1] = plot_func_map[args.oned_plotter]
    plotters[2] = plot_func_map[args.twod_plotter]
    actions = args.axis_operations
    dset_axis = args.dataset_axis_name
    dset_regex = args.dataset_regex
    signal_regex = args.sig_regex
    fig_size = args.figure_size

    def plotter(inval):
        title, hist = inval
        return makePlots(
            title,
            hist,
            plotters,
            actions,
            signal_regex,
            dset_regex,
            dset_axis,
            fig_size,
            outdir,
        )

    with concurrent.futures.ProcessPoolExecutor(nthreads) as pool:
        for x in pool.map(plotter, hists.items()):
            pass
