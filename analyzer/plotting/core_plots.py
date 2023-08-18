import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import importlib.resources as imp_res
from . import static
from dataclasses import dataclass
from functools import partial, wraps
from analyzer.datasets import Style, Dataset
from typing import Optional
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
        fontsize=20
    )
    return ax

def addEra(ax, lumi) -> mpl.axis.Axis:
    ax.text(
        1,
        1,
        f"${lumi}\\, \\mathrm{{fb}}^{{-1}}$ (13 TeV) ",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=20
    )
    return ax


def histRank(hist):
    return len(hist.axes)


@dataclass
class PlotObject:
    hist: hist.Hist
    title: Optional[str]
    dataset: Optional[Dataset]

    def getStyle(self):
        if self.dataset:
            return self.dataset.style


def autoSplit(func):
    @wraps(func)
    def inner(ax, hist, cat_axis=None, cat_filter=None, manager=None, **kwargs):

        if cat_axis is None:
            if isinstance(hist, PlotObject):
                return func(ax, hist, **kwargs)
            else:
                return func(ax, PlotObject(hist), **kwargs)
        else:
            other_axes = [a for a in hist.axes if a.name != cat_axis]
            titles = [a.label for a in other_axes]
            ax_title = hist.name
            to_plot = [
                PlotObject(hist[{cat_axis: n}], manager[n].getTitle(), manager[n])
                for n in hist.axes[cat_axis]
                if cat_filter is None or (re.search(cat_filter, n))
            ]
            to_plot = reversed(sorted(to_plot, key=lambda x: x.hist.sum().value))
            for p in to_plot:
                ax = func(ax, p, **kwargs)
            ax.set_xlabel(titles[0])
            if len(other_axes) > 1:
                ax.set_ylabel(titles[1])
            else:
                ax.set_ylabel("Events")
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
    style = p.getStyle()
    hist = p.hist
    x = hist.axes.centers[0]
    y = hist.values()
    var = np.sqrt(hist.variances())
    if yerr:
        ax.errorbar(x, y, yerr=var, label=p.title, fmt="_", color=style.color, **kwargs)
    else:
        ax.scatter(x, y, label=p.title, marker="+", color=style.color, **kwargs)
    return ax


@magicPlot
@autoSplit
def drawAs1DHist(ax, plot_object, yerr=True, **kwargs):
    style = plot_object.getStyle()
    h = plot_object.hist
    a = h.axes[0]
    edges = a.edges
    widths = a.widths+0.01
    vals = h.values()
    errs = np.sqrt(h.variances())
    #ret = ax.bar(
    #    edges[:-1],
    #    vals,
    #    width=widths,
    #    yerr=errs if yerr else None,
    #    label=plot_object.title,
    #    align="edge",
    #    color=style.color,
    #    **kwargs
    #)
    ret = ax.fill_between(
        edges[:-1],
        vals,
        step="pre",
        label=plot_object.title,
        color=style.color,
        **kwargs
    )
    return ax


@magicPlot
def drawAs2DHist(ax, plot_object, **kwargs):
    h = plot_object.hist
    a1 = h.axes[0]
    a2 = h.axes[1]
    vals, e1, e2 = h.to_numpy()
    ex = a1.centers
    ey = a2.centers
    vx, vy = np.meshgrid(ex, ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    ax.hist2d(x, y, bins=[e1, e2], weights=w, **kwargs)
    ax.set_xlabel(a1.label)
    ax.set_ylabel(a2.label)
    return ax


def addTitles1D(ax, hist, exclude=None):
    exclude = exclude or {
        "dataset",
    }
    axes = [x for x in hist.axes if x.name not in exclude]
    ax.set_xlabel(axes[0].label)
    has_var = hist.variances()
    if has_var is None:
        ax.set_ylabel("Events")
    else:
        ax.set_ylabel("Weighted Events")
    #ax.set_title(hist.name)
    return ax


def addTitles2D(ax, hist):
    axes = hist.axes
    ax.set_xlabel(axes[0].label)
    ax.set_ylabel(axes[1].label)
    return ax
