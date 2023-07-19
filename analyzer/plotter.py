import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

import re
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Iterable

from functools import partial, wraps
import itertools as it

import concurrent.futures

from collections import OrderedDict
import sys
import click
import logging

plot_logger = logging.getLogger("PlotLogger")
def loadStyles():
    font_dirs = ["./fonts"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use("style.mplstyle")


def loadHistograms(path): 
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r["histograms"]


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






def histRank(hist):
    return len(hist.axes)


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
        kwargs["yerr"] = yerr
    if orientation == "horizontal":
        if yerr is not None:
            kwargs["xerr"] = yerr
        ret = ax.barh(edges[:-1], vals, height=widths, **kwargs, **plot_opts)
    elif orientation == "vertical":
        if yerr is not None:
            kwargs["yerr"] = yerr
        ret = ax.bar(edges[:-1], vals, width=widths, **kwargs, **plot_opts)

    # print(h)
    # if yerr is not None:
    #    centers = (edges[:-1] + edges[1:])/2
    #    ret = ax.errorbar(centers, vals, yerr=yerr, color=h.color,  fmt="none")
    return ret


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
    vlines= None,
    hlines= None,
    plotopts= None,
    x_opts= None,
    y_opts= None,
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
    vlines= None,
    hlines= None,
    location="vertical",
) -> mpl.axis.Axis:
    ax, ax_x, _, div = make2DProjection(ax, h, vlines, hlines)

    if location == "horizontal":
        ax2 = div.append_axes("right", 1, pad=0.4, sharey=ax)
        ax2.yaxis.set_tick_params(labelleft=False)
    elif location == "vertical":
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
    outpath,
    function,
    *args,
    add_legend=True,
    **kwargs,
):
    p = Path(outpath)
    p.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots()
    ax = function(ax, *args, **kwargs)
    if add_legend:
        ax.legend()
    fig.tight_layout()
    print(f"Saving to {p}")
    fig.savefig(p)
    plt.close()


def collapseHistogram(hist, actions, base_title=""):
    ret = []
    all_slices = {}
    axes = list(x.name for x in hist.axes)
    categories = []
    for ax, (act, data) in actions.items():
        if act == "sum":
            all_slices[axes.index(ax)] = sum
        elif act == "split":
            vals = next(x for x in hist.axes if x.name == ax)
            if data:
                vals = [x for x in vals if re.search(data, x)]
            categories.append(zip(it.repeat(axes.index(ax)), vals))
        else:
            raise ValueError(f"Unknown Action {act}")
        # elif act == "filter":
        #    a = next(x for x in hist.axes if x.name == ax)
        #    all_slices[axes.index(ax)] = [x for x in a if re.search(data,x) ]
    if categories:
        for cats in it.product(*categories):
            cat_slices = dict(cats)
            t = "_".join(str(y) for x, y in cats)
            new_hist = hist[{**cat_slices, **all_slices}]
            new_hist.title = t
            ret.append(new_hist)
    else:
        new_hist = hist[all_slices]
        new_hist.title=base_title
        ret.append(new_hist)

    return ret



def createSets(title, hist, actions):
    axes = list(x.name for x in hist.axes)
    cats = []
    ret = {}
    for ax, (action, data) in actions.items():
        if action == "splitfile":
            vals = next(x for x in hist.axes if x.name == ax)
            if data:
                vals = [x for x in vals if re.search(data, x)]
            cats.append(zip(it.repeat(axes.index(ax)), vals))
    remaining_actions = {x: y for x, y in actions.items() if y[0] != "splitfile"}
    if cats:
        for cats in it.product(*cats):
            cat_slices = dict(cats)
            t = "__".join([title, *(f"{axes[x]}_eq_{y}" for x, y in cats)])
            ret[t] = collapseHistogram(hist[cat_slices], remaining_actions,title)
    else:
        ret[t] = collapseHistogram(hist, remaining_actions,title)
    return ret


def requiresDim(*dimensions):
    def decorator(func):
        def ret(ax, *args, **kwargs):
            for i, (x, y) in enumerate(zip(dimensions, args)):
                if x != y.setRank():
                    raise IndexError(
                        f"Plotting function requires that argument {i} have dimension {x}. Found dimension {y.setRank()}"
                    )
            return func(ax, *args, **kwargs)

        return ret

    return decorator



@requiresDim(1)
def drawScatter(ax, hist_set, yerr=False, title=""):
    plot_logger.info(f"Drawing scatter with hist set {hist_set}")
    axes = hist_set.setAxes()
    for title, hist in hist_set.hists():
        x = hist.axes.centers[0]
        y = hist.values()
        var = np.sqrt(hist.variances())
        if yerr:
            ax.errorbar(x, y, yerr=var, label=title, fmt="+")
        else:
            ax.scatter(x, y, label=title, marker="+")
    ax.set_ylabel("Events")
    ax.set_xlabel(axes[0].label)
    ax.set_title(hist_set.set_title)
    ax.set_title(hist_set.set_title)
    return ax


@requiresDim(1)
def drawHist(ax, hist_set, yerr=False):
    plot_logger.info(f"Drawing histogram with hist set {hist_set}")
    for title, hist in hist_set.hists():
        edges = hist.axes[0].edges[:-1]
        widths = hist.axes[0].widths
        vals = hist.values()
        var = np.sqrt(hist.variances())
        ax.bar(edges, vals, width=widths)
    ax.set_title(hist_set.set_title)
    return ax


@requiresDim(1, 1)
def drawScatterHist(ax, sig, bkg, yerr=False):
    drawHist(ax, sig, yerr)
    drawScatter(ax, sig, yerr)
    ax.set_title(sig.set_title)
    return ax

@requiresDim(2)
def draw2DHist(ax, hist_set):
    if len(hist_set) != 1:
        raise ValueError(f"While plotting {hist_set.set_title}, must have only one histogram in set, found {len(hist_set)}.")
    title,hist = list(hist_set.hists())[0]
    vals, e1, e2 = hist.to_numpy()
    ex = (e1[1:] + e1[:-1]) / 2
    ey = (e2[1:] + e2[:-1]) / 2
    vx, vy = np.meshgrid(ex, ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    ax.hist2d(x, y, bins=[e1, e2], weights=w)
    ax.set_xlabel(hist.axes[0].label)
    ax.set_ylabel(hist.axes[1].label)
    ax.set_title(f"{hist_set.set_title}")
    return ax


@requiresDim(1, 1)
def drawRatio(ax, sig, bkg, yerr=False):
    if len(sig) != len(bkg):
        raise ValueError()
    for (tsig, hsig), (tbkg, hbkg) in zip(sig.hists(), bkg.hists()):
        x = hsig.axes.centers[0]
        hv = hbkg.values()
        hs = hsig.values()
        mask = hv > 0
        values = hs[mask] / hv[mask]
        x = x[mask]
        ax.scatter(x, values, label=f"{tsig}", marker="+")
    print(sig.set_title)
    ax.set_ylabel("Ratio")
    ax.set_xlabel(sig.setAxes()[0].label)
    ax.set_title(f"{sig.set_title} / {bkg.set_title}")
    return ax


@click.group(chain=True, invoke_without_command=True)
@click.option("-v", "--verbose", count=True)
@click.pass_context
def cli(ctx, verbose):
    if verbose == 0:
        ll = logging.NOTSET
    if verbose == 1:
        ll = logging.WARNING
    if verbose == 2:
        ll = logging.INFO

    plot_logger = logging.getLogger("PlotLogger")
    ch = logging.StreamHandler().setLevel(ll)
    plot_logger.setLevel(ll)
    plot_logger.addHandler(ll)


@cli.command("hists")
@click.option(
    "-i",
    "--input",
    "histogram_file",
    required=True,
    type=click.Path(),
    help="The image file to open.",
)
@click.option(
    "-a",
    "--action",
    "actions",
    multiple=True,
    type=(str, str, str),
    help="The image file to open.",
)
@click.option(
    "-f",
    "--filter",
    "filter_re",
    type=str,
    help="Filter histograms",
)
@click.option("-n", "--normalize", default=False, is_flag=True)
@click.pass_context
def createSet(ctx, histogram_file, actions, filter_re, normalize):
    all_hists = pickle.load(open(histogram_file, "rb"))["histograms"]
    hists = {
        x: y
        for x, y in all_hists.items()
        if (re.search(filter_re, x) if filter_re else True)
    }
    actions = {x: (y, z) for x, y, z in actions}
    z = list(
        it.chain.from_iterable(
            HistogramSet.createSets(x, y, actions) for x, y in hists.items()
        )
    )
    if normalize:
        z = [x.normalize() for x in z]
    p = ctx.parent
    ctx.ensure_object(dict)
    p.ensure_object(dict)

    if p.obj.get("histogram_sets"):
        p.obj["histogram_sets"] = list(
            [*x, y] for x, y in zip(p.obj["histogram_sets"], z)
        )
    else:
        p.obj["histogram_sets"] = [[x] for x in z]


@cli.command("scatter")
@click.option("-o", "--outdir", type=click.Path(), required=True)
@click.option("-e", "--error", default=False, is_flag=True)
@click.pass_context
def commandScatter(ctx, outdir, error):
    hist_sets = ctx.parent.obj["histogram_sets"]
    print(hist_sets)
    outdir = Path(outdir)
    for hist_set in hist_sets:
        outpath = outdir / f"{hist_set[0].set_title}.pdf"
        print(outpath)
        autoPlot(outpath, drawScatter, *hist_set, yerr=error)

@cli.command("twod")
@click.option("-o", "--outdir", type=click.Path(), required=True)
@click.pass_context
def commandScatter(ctx, outdir):
    hist_sets = ctx.parent.obj["histogram_sets"]
    outdir = Path(outdir)
    for hist_set in hist_sets:
        outpath = outdir / f"{hist_set[0].set_title}.pdf"
        autoPlot(outpath, draw2DHist, *hist_set, add_legend=False)

@cli.command("scatterhist")
@click.option("-o", "--outdir", type=click.Path(), required=True)
@click.option("-e", "--error", default=False, is_flag=True)
@click.pass_context
def commandScatter(ctx, outdir, error):
    hist_sets = ctx.parent.obj["histogram_sets"]
    outdir = Path(outdir)
    for hist_set in hist_sets:
        outpath = outdir / f"{hist_set[0].set_title}.pdf"
        print(outpath)
        autoPlot(outpath, drawScatterHist, *hist_set, err=error)


@cli.command("ratio")
@click.option("-o", "--outdir", type=click.Path(), required=True)
@click.pass_context
def commandRatio(ctx, outdir):
    hist_sets = ctx.parent.obj["histogram_sets"]
    outdir = Path(outdir)
    for hist_set in hist_sets:
        outpath = outdir / f"{hist_set[0].set_title}.pdf"
        print(outpath)
        autoPlot(outpath, drawRatio, *hist_set)


@cli.command("list")
@click.option(
    "-i",
    "--input",
    "histogram_file",
    required=True,
    type=click.Path(),
    help="The image file to open.",
)
@click.pass_context
def listHistograms(ctx, histogram_file):
    all_hists = pickle.load(open(histogram_file, "rb"))
    for name, hist in all_hists.items():
        click.echo(f"{name}:{hist}")


if __name__ == "__main__":
    cli()
