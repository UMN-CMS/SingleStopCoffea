import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import importlib.resources as imp_res
from . import static
from dataclasses import dataclass
from functools import partial, wraps
import itertools as it
import hist
import re


def loadStyles():
    style = imp_res.files(static) / "style.mplstyle"


    font_dirs = [imp_res.files(static) / "fonts"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use(style)


def addPrelim(ax: mpl.axis.Axis) -> mpl.axis.Axis:
    ax.text(
        0.02,
        0.98,
        r"$\bf{CMS}\ \it{Preliminary}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    return ax


def histRank(hist):
    return len(hist.axes)


@dataclass
class PlotObject:
    name: str
    hist: hist.Hist


def autoSplit(func):
    @wraps(func)
    def inner(ax, hist, cat_axis=None, cat_filter=None, **kwargs):
        if cat_axis is None:
            if isinstance(hist, PlotObject):
                return func(ax, hist, **kwargs)
            else:
                return func(ax, PlotObject(hist.name, hist), **kwargs)
        else:
            other_axes = [a for a in hist.axes if a.name != cat_axis]
            titles = [a.label for a in other_axes]
            ax_title = hist.name
            to_plot = [
                PlotObject(n, hist[{cat_axis: n}])
                for n in hist.axes[cat_axis]
                if cat_filter is None or (re.search(cat_filter,n))
            ]
            to_plot = reversed(sorted(to_plot, key=lambda x: x.hist.sum().value))
            for p in to_plot:
                ax = func(ax, p, **kwargs)
            ax.set_xlabel(titles[0])
            if len(other_axes) > 1:
                ax.set_ylabel(titles[1])
            else:
                ax.set_ylabel("Events")
            ax.set_title(ax_title)
            ax.legend()
            return ax

    return inner


def magicPlot(func):
    @wraps(func)
    def inner(first, *args, **kwargs):
        if isinstance(first, plt.Axes):
            return func(first, *args, **kwargs)
        else:
            fig, ax = plt.subplots()
            return fig, func(ax, first, *args, **kwargs)

    return inner


@magicPlot
@autoSplit
def drawAsScatter(ax, p, yerr=True, **kwargs):
    hist = p.hist
    x = hist.axes.centers[0]
    y = hist.values()
    var = np.sqrt(hist.variances())
    if yerr:
        ax.errorbar(x, y, yerr=var, label=p.name, fmt="_", **kwargs)
    else:
        ax.scatter(x, y, label=p.name, marker="+", **kwargs)
    return ax


@magicPlot
@autoSplit
def drawAs1DHist(ax, plot_object, yerr=True):
    h = plot_object.hist
    a = h.axes[0]
    edges = a.edges
    widths = a.widths
    vals = h.values()
    errs = np.sqrt(h.variances())
    ret = ax.bar(
        edges[:-1],
        vals,
        width=widths,
        yerr=errs if yerr else None,
        label=plot_object.name,
        align="edge",
    )
    return ax


def addTitles(ax, hist):
    axes = hist.axes
    ax.set_xlabel(axes[0].label)
    has_var = hist.variances()
    if has_var is None:
        ax.set_ylabel("Events")
    else:
        ax.set_ylabel("Weighted Events")
    ax.set_title(hist.name)
    return ax

    


